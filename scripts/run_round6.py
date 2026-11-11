"""Round 6: planar microphone-array screening and a lightweight peak reranker.

Run in the isolated WSL Python 3.11 environment with Pyroomacoustics 0.7.7.
All comparisons are paired by source clips, room parameters, source positions,
noise seed, and scene recipe. The R6 acoustic evaluator is SRP-PHAT only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_localization.audio import add_noise_at_snr, normalize_active, read_audio  # noqa: E402
from voice_localization.geometry import microphone_array, pair_indices  # noqa: E402
from voice_localization.spatial import gcc_phat_pairs, make_grid, score_srp_points  # noqa: E402

OUT = ROOT / "artifacts" / "round6"
DATA = ROOT / "data" / "full"
VAL_SCENES = DATA / "manifests" / "scenes_validation.jsonl"
VAL_AUDIO = DATA / "manifests" / "audio_validation.jsonl"
VAL_MAPS = ROOT / "artifacts" / "round3" / "full_baseline_maps.npz"
ROOM = (8.0, 6.0, 3.0)
FS, C = 16_000, 343.0
MIC_Z = SOURCE_Z = 1.5
WALL_MARGIN, MIN_MIC_SPACING = 0.5, 0.75
MIN_SOURCE_SPACING, MIN_SOURCE_MIC = 0.75, 0.5
WINDOW, HOP, FFT = 512, 256, 1024
DURATION, CONTEXT = 2.0, 0.25
SAMPLES = round(FS * DURATION)
SOURCE_SAMPLES = round(FS * (DURATION + CONTEXT))
CROP_START = round(FS * CONTEXT)
SEED = 20261109
LAYOUT_NAMES = ("perimeter", "circle", "random", "fim_best", "entropy_best")

_AUDIO_INDEX: dict[str, dict] = {}
_LAYOUTS: dict[str, list[float]] = {}
_RERANKER: dict | None = None


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def runtime_path(value: str | Path) -> Path:
    """Resolve Windows manifest paths when the experiment runs inside WSL."""
    text = str(value)
    win = PureWindowsPath(text)
    if win.drive and os.name == "posix":
        path = Path("/mnt") / win.drive[0].lower() / Path(*win.parts[1:])
    else:
        path = Path(text)
    if not path.is_absolute():
        path = DATA / path
    return path


def load_audio_index(path: Path) -> dict[str, dict]:
    return {str(row["audio_id"]): row for row in read_jsonl(path)}


def geometry_layouts(random_count: int = 512) -> tuple[dict, list[dict]]:
    """Build fixed planar candidates and geometry-only prefilter scores."""
    grid_x = np.arange(WALL_MARGIN, ROOM[0] - WALL_MARGIN + 1e-6, 0.5)
    grid_y = np.arange(WALL_MARGIN, ROOM[1] - WALL_MARGIN + 1e-6, 0.5)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    library: dict[str, list[dict]] = {}
    prefilter_rows: list[dict] = []

    for m in (4, 6, 8):
        perimeter = microphone_array("perimeter", m, ROOM, SEED, WALL_MARGIN, MIC_Z)
        circle = microphone_array("circle", m, ROOM, SEED, WALL_MARGIN, MIC_Z)
        rng = np.random.default_rng(SEED + 10_000 + m)
        pool = [_random_layout(m, rng) for _ in range(random_count)]
        fim_scores, ent_scores, details = [], [], []
        for layout in pool:
            fim, entropy, crlb = _geometry_scores(layout[:, :2], points)
            fim_scores.append(fim)
            ent_scores.append(entropy)
            details.append({"fim_mean_logdet": fim, "sensitivity_entropy_norm": entropy,
                            "mean_crlb_trace_proxy": crlb})
        fim_idx = int(np.argmax(fim_scores))
        ent_idx = int(np.argmax(ent_scores))
        excluded = {fim_idx, ent_idx}
        random_idx = next(i for i in range(random_count) if i not in excluded)
        selected = [("perimeter", perimeter, None), ("circle", circle, None),
                    ("random", pool[random_idx], random_idx),
                    ("fim_best", pool[fim_idx], fim_idx),
                    ("entropy_best", pool[ent_idx], ent_idx)]
        library[str(m)] = []
        for name, xyz, pool_idx in selected:
            score = details[pool_idx] if pool_idx is not None else _geometry_score_record(xyz[:, :2], points)
            item = {"name": name, "positions_xyz": xyz.astype(float).tolist(),
                    "prefilter": score, "pool_index": pool_idx}
            library[str(m)].append(item)
            prefilter_rows.append({"M": m, **item["prefilter"], "layout": name,
                                   "pool_index": pool_idx,
                                   "positions_xy": xyz[:, :2].astype(float).tolist()})
    return library, prefilter_rows


def _random_layout(count: int, rng: np.random.Generator) -> np.ndarray:
    xy: list[np.ndarray] = []
    attempts = 0
    while len(xy) < count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError(f"não consegui gerar arranjo factível M={count}")
        point = np.array([rng.uniform(WALL_MARGIN, ROOM[0] - WALL_MARGIN),
                          rng.uniform(WALL_MARGIN, ROOM[1] - WALL_MARGIN)])
        if all(np.linalg.norm(point - other) >= MIN_MIC_SPACING for other in xy):
            xy.append(point)
    return np.column_stack((np.asarray(xy), np.full(count, MIC_Z))).astype(np.float32)


def _geometry_scores(mics_xy: np.ndarray, points_xy: np.ndarray) -> tuple[float, float, float]:
    m = len(mics_xy)
    pairs = pair_indices(m)
    incidence = np.zeros((len(pairs), m), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        incidence[k, i], incidence[k, j] = 1.0, -1.0
    covariance_inv = np.linalg.pinv(incidence @ incidence.T, rcond=1e-10)
    logdets, traces, sensitivities = [], [], []
    for point in points_xy:
        delta = point[None, :] - mics_xy
        unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        jac = np.asarray([unit[i] - unit[j] for i, j in pairs])
        fim = jac.T @ covariance_inv @ jac
        fim += np.eye(2) * 1e-9
        sign, logdet = np.linalg.slogdet(fim)
        logdets.append(float(logdet if sign > 0 else -30.0))
        traces.append(float(np.trace(np.linalg.pinv(fim))))
        sensitivities.append(float(math.sqrt(max(np.linalg.det(fim), 0.0))))
    sens = np.maximum(np.asarray(sensitivities), 1e-15)
    prob = sens / sens.sum()
    entropy = float(-np.sum(prob * np.log(prob)) / math.log(len(prob)))
    return float(np.mean(logdets)), entropy, float(np.mean(traces))


def _geometry_score_record(mics_xy: np.ndarray, points_xy: np.ndarray) -> dict:
    fim, entropy, crlb = _geometry_scores(mics_xy, points_xy)
    return {"fim_mean_logdet": fim, "sensitivity_entropy_norm": entropy,
            "mean_crlb_trace_proxy": crlb}


def draw_source_positions(n: int, layouts: list[dict], rng: np.random.Generator) -> list[list[float]]:
    """Sample truth positions from a geometry-independent common feasible region."""
    arrays = [np.asarray(row["positions_xyz"], dtype=np.float64)[:, :2] for row in layouts]
    accepted: list[np.ndarray] = []
    for _ in range(200_000):
        point = np.array([rng.uniform(WALL_MARGIN, ROOM[0] - WALL_MARGIN),
                          rng.uniform(WALL_MARGIN, ROOM[1] - WALL_MARGIN)])
        if any(np.linalg.norm(point - old) < MIN_SOURCE_SPACING for old in accepted):
            continue
        if any(np.min(np.linalg.norm(mics - point, axis=1)) < MIN_SOURCE_MIC for mics in arrays):
            continue
        accepted.append(point)
        if len(accepted) == n:
            return np.asarray(accepted).astype(float).tolist()
    raise RuntimeError(f"common feasible source region insufficient for N={n}")


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layout_path = OUT / "array_candidates.json"
    if layout_path.exists():
        library = json.loads(layout_path.read_text(encoding="utf-8"))
        print(f"reusing frozen geometry pool: {layout_path}", flush=True)
    else:
        library, score_rows = geometry_layouts(512)
        write_json(layout_path, library)
        write_json(OUT / "geometry_prefilter.json", score_rows)
        print(f"created {layout_path}", flush=True)
    val_audio = load_audio_index(VAL_AUDIO)
    scene_rows = read_jsonl(VAL_SCENES)
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in scene_rows:
        groups.setdefault((int(row["n_sources"]), int(row["n_microphones"])), []).append(row)
    triage: list[dict] = []
    for n in (1, 2, 3):
        for m in (4, 6, 8):
            group = groups[(n, m)]
            if len(group) < 100:
                raise RuntimeError(f"só há {len(group)} cenas para N={n}, M={m}")
            rng = np.random.default_rng(SEED + 100 * n + m)
            chosen = rng.choice(len(group), size=100, replace=False)
            layouts = library[str(m)]
            for j, ix in enumerate(chosen):
                source = group[int(ix)]
                audio_ids = source["audio_ids"]
                starts = source.get("metadata", {}).get("source_window_starts", [])
                if len(audio_ids) != n or len(starts) != n or any(a not in val_audio for a in audio_ids):
                    raise RuntimeError(f"áudio/recorte ausente em {source['scene_id']}")
                pos_rng = np.random.default_rng(SEED + n * 100_000 + m * 1000 + j)
                positions = draw_source_positions(n, layouts, pos_rng)
                triage.append({
                    "scene_id": f"r6-val-n{n}-m{m}-{j:03d}",
                    "template_scene_id": source["scene_id"], "split": "validation_screen",
                    "n_sources": n, "n_microphones": m, "audio_ids": audio_ids,
                    "source_window_starts": starts, "speaker_ids": source["speaker_ids"],
                    "source_positions_xy": positions,
                    "rt60_sabine_input": float(source.get("metadata", {}).get("rt60_sabine_input") or source["rt60_requested"]),
                    "snr_db": float(source["snr_requested_db"]),
                    "seed": int(SEED + 1_000_000 + n * 100_000 + m * 1_000 + j),
                    "noise_seed": int(SEED + 5_000_000 + n * 100_000 + m * 1_000 + j),
                })
    write_jsonl(OUT / "triage_scenes.jsonl", triage)
    write_json(OUT / "round6_protocol.json", {
        "status": "prepared; no vertical array variation",
        "seed": SEED, "room_m": ROOM, "sample_rate_hz": FS,
        "source_and_mic_z_m": 1.5, "grid_resolution_m": 0.1,
        "m_values": [4, 6, 8], "n_values": [1, 2, 3],
        "layouts_per_m": list(LAYOUT_NAMES), "random_layout_candidates_per_m": 512,
        "triage_scenes_per_n_m": 100, "triage_scene_count": len(triage),
        "triage_render_count": len(triage) * len(LAYOUT_NAMES),
        "layout_wall_margin_m": WALL_MARGIN, "minimum_mic_spacing_m": MIN_MIC_SPACING,
        "minimum_source_spacing_m": MIN_SOURCE_SPACING,
        "minimum_source_mic_distance_m": MIN_SOURCE_MIC,
        "triage_source_positions": "fresh random XY, fixed across all five layouts for each scene; common feasible region",
        "screen_audio_and_acoustics": "validation speech clips/starts, RT60 and SNR, rerendered per layout",
        "primary_method": "SRP-PHAT, 2D; frozen 0.60 m NMS decoder",
        "FIM_prefilter": "full-pair TDOA Jacobian with shared-reference covariance; maximize mean spatial logdet(FIM)",
        "entropy_prefilter": "normalized entropy of sqrt(det(FIM)) over room grid; simplified information-uniformity proxy, not an exact paper reproduction",
        "holdout": "fresh LibriSpeech dev-other speakers, after speaker/audio audit; do not reuse scenes_test or test-other",
        "no_cnn_retraining": True,
    })
    print(json.dumps({"triage_scenes": len(triage), "render_conditions": len(triage) * 5,
                      "layout_counts": {m: len(v) for m, v in library.items()}}, ensure_ascii=False), flush=True)


def scan_external(root: Path) -> None:
    rows = []
    for audio_path in sorted(root.rglob("*.flac")):
        audio_id = audio_path.stem
        speaker_id = audio_id.split("-")[0]
        info = sf.info(audio_path)
        if info.samplerate != FS:
            continue
        rows.append({"audio_id": audio_id, "speaker_id": speaker_id,
                     "path": audio_path.resolve().relative_to(DATA.resolve()).as_posix(),
                     "sample_rate": int(info.samplerate),
                     "duration_seconds": float(info.duration), "split": "dev-other-r6"})
    if not rows:
        raise RuntimeError(f"nenhum FLAC 16 kHz em {root}")
    old_files: set[str] = set()
    old_speakers: set[str] = set()
    for path in (DATA / "manifests" / "audio_train.jsonl",
                 DATA / "manifests" / "audio_validation.jsonl",
                 DATA / "manifests" / "audio_test.jsonl",
                 ROOT / "artifacts" / "round3" / "manifests" / "test_other_audio_exclusive.jsonl"):
        if path.exists():
            for row in read_jsonl(path):
                old_files.add(str(row["audio_id"]))
                old_speakers.add(str(row["speaker_id"]))
    ids = {str(r["audio_id"]) for r in rows}
    speakers = {str(r["speaker_id"]) for r in rows}
    audit = {"source": "LibriSpeech dev-other, OpenSLR 12", "audio_count": len(rows),
             "speaker_count": len(speakers), "old_audio_overlap_count": len(ids & old_files),
             "old_speaker_overlap_count": len(speakers & old_speakers),
             "overlapping_audio_ids": sorted(ids & old_files),
             "overlapping_speaker_ids": sorted(speakers & old_speakers),
             "speaker_disjoint": not bool(speakers & old_speakers),
             "audio_disjoint": not bool(ids & old_files)}
    write_jsonl(OUT / "dev_other_audio_r6.jsonl", rows)
    write_json(OUT / "dev_other_audio_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if not audit["speaker_disjoint"] or not audit["audio_disjoint"]:
        raise RuntimeError("LibriSpeech dev-other não é disjunto; holdout não será congelado")


def prepare_external(count_per_cell: int = 300) -> None:
    audio_path = OUT / "dev_other_audio_r6.jsonl"
    audit_path = OUT / "dev_other_audio_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("speaker_disjoint") or not audit.get("audio_disjoint"):
        raise RuntimeError("auditoria externa não aprova este holdout")
    audios = [r for r in read_jsonl(audio_path) if r["duration_seconds"] >= 1.0]
    by_speaker: dict[str, list[dict]] = {}
    for row in audios:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speakers = sorted(by_speaker)
    if len(speakers) < 3:
        raise RuntimeError("menos de três falantes disponíveis no dev-other")
    library = json.loads((OUT / "array_candidates.json").read_text(encoding="utf-8"))
    validation = read_jsonl(VAL_SCENES)
    rts = np.asarray([float(r["rt60_requested"]) for r in validation])
    snrs = np.asarray([float(r["snr_requested_db"]) for r in validation])
    records: list[dict] = []
    for n in (1, 2, 3):
        for m in (4, 6, 8):
            rng = np.random.default_rng(SEED + 20_000_000 + n * 100_000 + m * 1_000)
            arrays = library[str(m)]
            for j in range(count_per_cell):
                chosen_speakers = rng.choice(speakers, size=n, replace=False).tolist()
                audio_ids, starts = [], []
                for speaker in chosen_speakers:
                    row = by_speaker[str(speaker)][int(rng.integers(len(by_speaker[str(speaker)])))]
                    length = int(round(float(row["duration_seconds"]) * FS))
                    max_start = max(0, length - SOURCE_SAMPLES)
                    audio_ids.append(str(row["audio_id"]))
                    starts.append(int(rng.integers(max_start + 1)) if max_start else 0)
                positions = draw_source_positions(n, arrays, rng)
                index_seed = SEED + 30_000_000 + n * 1_000_000 + m * 10_000 + j
                records.append({
                    "scene_id": f"r6-devother-n{n}-m{m}-{j:03d}", "split": "dev-other-r6",
                    "n_sources": n, "n_microphones": m, "audio_ids": audio_ids,
                    "source_window_starts": starts, "speaker_ids": chosen_speakers,
                    "source_positions_xy": positions,
                    "rt60_sabine_input": float(rts[int(rng.integers(len(rts)))]),
                    "snr_db": float(snrs[int(rng.integers(len(snrs)))]),
                    "seed": int(index_seed), "noise_seed": int(index_seed + 5_000_000),
                })
    path = OUT / "holdout_scenes.jsonl"
    if path.exists():
        raise FileExistsError(f"holdout já congelado: {path}; remova apenas após auditoria explícita")
    write_jsonl(path, records)
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    write_json(OUT / "holdout_protocol.json", {
        "status": "frozen_before_round6_confirmation", "scene_manifest": path.name,
        "sha256": digest, "scene_count": len(records), "scenes_per_n_m": count_per_cell,
        "speaker_source": "LibriSpeech dev-other; disjoint from all existing project audio/speakers",
        "room_parameters": "RT60 and SNR sampled from the existing validation distributions",
        "positions": "fresh geometry-common-feasible XY, independently sampled, fixed across compared layouts",
        "layouts": "perimeter baseline and one R6 winner per M chosen on validation screen",
        "seed": SEED + 20_000_000, "opened_for_selection": False,
    })
    print(json.dumps({"holdout_scenes": len(records), "sha256": digest}, ensure_ascii=False), flush=True)


def _init_worker(audio_index: dict[str, dict], layouts: dict, reranker: dict | None) -> None:
    global _AUDIO_INDEX, _LAYOUTS, _RERANKER
    _AUDIO_INDEX, _LAYOUTS, _RERANKER = audio_index, layouts, reranker


@lru_cache(maxsize=256)
def _load_segment(audio_id: str, start: int) -> np.ndarray:
    row = _AUDIO_INDEX[audio_id]
    raw = read_audio(runtime_path(row["path"]), FS)
    normalized = normalize_active(raw)
    end = start + SOURCE_SAMPLES
    segment = normalized[start:end]
    if segment.size < SOURCE_SAMPLES:
        segment = np.pad(segment, (0, SOURCE_SAMPLES - segment.size))
    segment = np.asarray(segment, dtype=np.float32)
    if not np.isfinite(segment).all() or float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))) < 1e-4:
        raise ValueError(f"recorte sem fala utilizável: {audio_id}@{start}")
    return segment


def _render(clean_sources: list[np.ndarray], source_xy: np.ndarray, mics: np.ndarray,
            rt60: float, snr_db: float, room_seed: int, noise_seed: int) -> np.ndarray:
    sabine_rt60 = max(float(rt60), 0.05)
    np.random.seed(room_seed)
    if hasattr(pra, "random") and hasattr(pra.random, "seed"):
        pra.random.seed(numpy=room_seed, libroom=room_seed)
    absorption, max_order = pra.inverse_sabine(sabine_rt60, list(ROOM))
    room = pra.ShoeBox(list(ROOM), fs=FS, materials=pra.Material(absorption),
                       max_order=max_order, use_rand_ism=True)
    source_xyz = np.column_stack((source_xy, np.full(len(source_xy), SOURCE_Z)))
    for position, signal in zip(source_xyz, clean_sources):
        room.add_source(position.tolist(), signal=signal)
    room.add_microphone_array(mics.T)
    room.simulate()
    full = np.asarray(room.mic_array.signals, dtype=np.float32)
    clean = full[:, CROP_START:CROP_START + SAMPLES]
    if clean.shape[1] < SAMPLES:
        clean = np.pad(clean, ((0, 0), (0, SAMPLES - clean.shape[1])))
    mixed, _ = add_noise_at_snr(clean, snr_db, np.random.default_rng(noise_seed))
    return mixed


def _candidate_points(scores: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      k: int = 50, separation: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(scores, np.float32)
    local = arr >= maximum_filter(arr, size=3, mode="nearest")
    inds = np.flatnonzero(local.ravel())
    inds = inds[np.argsort(arr.ravel()[inds])[::-1]]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    selected: list[int] = []
    for idx in inds:
        if all(np.linalg.norm(grid[idx] - grid[j]) >= separation for j in selected):
            selected.append(int(idx))
            if len(selected) == k:
                break
    if len(selected) < k:
        for idx in np.argsort(arr.ravel())[::-1]:
            if all(np.linalg.norm(grid[idx] - grid[j]) >= separation for j in selected):
                selected.append(int(idx))
            if len(selected) == k:
                break
    return grid[selected].astype(np.float32), np.asarray(selected, dtype=np.int64)


def _candidate_features(scores: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(scores, np.float32)
    h, w = arr.shape
    peak, prominence, sharpness = [], [], []
    for idx in indices:
        y, x = divmod(int(idx), w)
        value = float(arr[y, x])
        y0, y1 = max(0, y - 5), min(h, y + 6)
        x0, x1 = max(0, x - 5), min(w, x + 6)
        patch = arr[y0:y1, x0:x1]
        py, px = np.ogrid[y0:y1, x0:x1]
        radius = np.sqrt((py - y) ** 2 + (px - x) ** 2)
        ring = patch[(radius >= 2.0) & (radius <= 5.0)]
        prominence.append(value - float(np.median(ring)) if ring.size else 0.0)
        neighbors = [arr[yy, xx] for yy, xx in
                     ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                      (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1))
                     if 0 <= yy < h and 0 <= xx < w]
        sharpness.append(value - float(np.mean(neighbors)) if neighbors else 0.0)
        peak.append(value)
    return {"peak": np.asarray(peak), "prom": np.asarray(prominence),
            "sharp": np.asarray(sharpness)}


def _rank_feature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values, method="average") / max(len(values), 1)


def _select_candidates(points: np.ndarray, feature_values: dict[str, np.ndarray], n: int,
                       weights: dict[str, float] | None = None, radius: float = 0.60) -> np.ndarray:
    weights = weights or {"peak": 1.0, "prom": 0.0, "sharp": 0.0}
    score = sum(float(weights.get(key, 0.0)) * _rank_feature(value)
                for key, value in feature_values.items())
    order = np.argsort(score)[::-1]
    selected = []
    for ix in order:
        point = points[int(ix)]
        if all(np.linalg.norm(point - other) >= radius for other in selected):
            selected.append(point)
            if len(selected) == n:
                return np.asarray(selected, dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def _baseline_points(scores: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     n: int, radius: float = 0.60) -> np.ndarray:
    arr = np.asarray(scores, np.float32)
    local = arr >= maximum_filter(arr, size=3, mode="nearest")
    inds = np.flatnonzero(local.ravel())
    inds = inds[np.argsort(arr.ravel()[inds])[::-1]]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    selected = []
    for idx in inds:
        p = grid[int(idx)]
        if all(np.linalg.norm(p - q) >= radius for q in selected):
            selected.append(p)
            if len(selected) == n:
                return np.asarray(selected, dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def _matched_errors(truth: np.ndarray, estimate: np.ndarray) -> list[float]:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if len(truth) != len(estimate):
        raise ValueError(f"localizador retornou {len(estimate)} posições para N={len(truth)}")
    d = np.linalg.norm(truth[:, None, :] - estimate[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(d)
    return d[rows, cols].astype(float).tolist()


def _process_scene(task: dict) -> dict:
    sources = [_load_segment(a, int(start)) for a, start in
               zip(task["audio_ids"], task["source_window_starts"])]
    truth = np.asarray(task["source_positions_xy"], dtype=np.float32)
    n, m = int(task["n_sources"]), int(task["n_microphones"])
    grid = make_grid(ROOM, 0.10)
    result = {"scene_id": task["scene_id"], "split": task["split"], "N": n, "M": m,
              "speaker_ids": task.get("speaker_ids", []), "audio_ids": task.get("audio_ids", []),
              "template_scene_id": task.get("template_scene_id"),
              "source_positions_xy": truth.astype(float).tolist(), "errors": {}, "speaker_errors": {}, "estimates": {},
              "render_seconds": {}, "score_seconds": {}}
    for layout_name in task["layout_names"]:
        mics = np.asarray(_LAYOUTS[str(m)][layout_name], dtype=np.float32)
        t0 = time.perf_counter()
        mixed = _render(sources, truth, mics, float(task["rt60_sabine_input"]),
                        float(task["snr_db"]), int(task["seed"]), int(task["noise_seed"]))
        render_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        corr, pairs = gcc_phat_pairs(mixed, WINDOW, HOP, FFT)
        flat = score_srp_points(corr, pairs, mics, grid.points_xy, FS, FFT, C)
        score_map = flat.reshape(grid.shape)
        estimates = {"srp": _baseline_points(score_map, grid.xs, grid.ys, n)}
        if _RERANKER:
            candidates, indices = _candidate_points(score_map, grid.xs, grid.ys, k=50)
            features = _candidate_features(score_map, indices)
            estimates["reranker"] = _select_candidates(candidates, features, n, _RERANKER)
        for method, pred in estimates.items():
            errors = _matched_errors(truth, pred)
            result["errors"].setdefault(method, {})[layout_name] = errors
            result["speaker_errors"].setdefault(method, {})[layout_name] = {
                str(speaker): float(error) for speaker, error in zip(task.get("speaker_ids", []), errors)
            }
            result["estimates"].setdefault(method, {})[layout_name] = np.asarray(pred).astype(float).tolist()
        result["render_seconds"][layout_name] = render_s
        result["score_seconds"][layout_name] = time.perf_counter() - t1
    return result


def smoke() -> None:
    prepare()
    library = json.loads((OUT / "array_candidates.json").read_text(encoding="utf-8"))
    triage = read_jsonl(OUT / "triage_scenes.jsonl")
    audio_index = load_audio_index(VAL_AUDIO)
    layouts = {m: {r["name"]: r["positions_xyz"] for r in rows} for m, rows in library.items()}
    _init_worker(audio_index, layouts, None)
    task = dict(triage[0], layout_names=["perimeter"])
    result = _process_scene(task)
    write_json(OUT / "smoke_result.json", result)
    print(json.dumps({"scene": result["scene_id"], "N": result["N"], "M": result["M"],
                      "render_seconds": result["render_seconds"], "errors": result["errors"]}), flush=True)


def probe_external_audio() -> None:
    """Validate a corpus file path/segment without evaluating holdout labels."""
    rows = [r for r in read_jsonl(OUT / "dev_other_audio_r6.jsonl")
            if r["duration_seconds"] >= SOURCE_SAMPLES / FS]
    if not rows:
        raise RuntimeError("nenhum áudio dev-other longo o suficiente para o probe")
    row = rows[0]
    _init_worker({str(row["audio_id"]): row}, {}, None)
    segment = _load_segment(str(row["audio_id"]), 0)
    summary = {"audio_id": row["audio_id"], "speaker_id": row["speaker_id"],
               "segment_samples": int(segment.size), "segment_rms": float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))),
               "path_exists": runtime_path(row["path"]).is_file(),
               "holdout_labels_opened": False}
    write_json(OUT / "external_audio_probe.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def _run_parallel(stage: str, tasks: list[dict], audio_index: dict, layouts: dict,
                  workers: int, reranker: dict | None = None) -> Path:
    output = OUT / f"{stage}_predictions.jsonl"
    done = {}
    if output.exists():
        for row in read_jsonl(output):
            done[row["scene_id"]] = row
    pending = [task for task in tasks if task["scene_id"] not in done]
    start = time.perf_counter()
    if pending:
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_init_worker,
                                 initargs=(audio_index, layouts, reranker)) as executor:
            futures = [executor.submit(_process_scene, task) for task in pending]
            with output.open("a", encoding="utf-8") as f:
                for i, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    done[row["scene_id"]] = row
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    f.flush()
                    if i % 25 == 0 or i == len(futures):
                        print(json.dumps({"stage": stage, "done": i, "total": len(futures),
                                          "elapsed_s": round(time.perf_counter() - start, 1)}), flush=True)
    timing = {"stage": stage, "scene_count": len(tasks), "layout_render_count": sum(len(t["layout_names"]) for t in tasks),
              "completed_scene_count": len(done), "seconds_wall": time.perf_counter() - start,
              "workers": workers, "reranker": reranker}
    write_json(OUT / f"timing_{stage}.json", timing)
    if len(done) != len(tasks):
        raise RuntimeError(f"{stage}: resultados {len(done)}/{len(tasks)}")
    return output


def run_triage(workers: int) -> None:
    prepare()
    library = json.loads((OUT / "array_candidates.json").read_text(encoding="utf-8"))
    tasks = read_jsonl(OUT / "triage_scenes.jsonl")
    for task in tasks:
        task["layout_names"] = list(LAYOUT_NAMES)
    audio_index = load_audio_index(VAL_AUDIO)
    layouts = {m: {r["name"]: r["positions_xyz"] for r in rows} for m, rows in library.items()}
    path = _run_parallel("triage", tasks, audio_index, layouts, workers)
    summarize_geometry(read_jsonl(path), "triage")


def _metric(errors: list[list[float]]) -> dict:
    values = np.concatenate([np.asarray(e, dtype=np.float64) for e in errors if len(e)])
    if not len(values):
        return {"scene_count": 0}
    return {"scene_count": len(errors), "source_count": int(len(values)),
            "mean_m": float(values.mean()), "median_m": float(np.median(values)),
            "p90_m": float(np.percentile(values, 90)), "over_1m": float(np.mean(values > 1.0)),
            "within_0_5m": float(np.mean(values <= 0.5))}


def _paired_bootstrap(base: list[list[float]], candidate: list[list[float]], seed: int,
                      repeats: int = 5000) -> dict:
    if len(base) != len(candidate) or not base:
        return {}
    rng = np.random.default_rng(seed)
    group_ids: dict[int, list[int]] = {}
    for i, (b, c) in enumerate(zip(base, candidate)):
        if len(b) != len(c):
            raise ValueError("baseline e candidato precisam ter o mesmo N por cena")
        group_ids.setdefault(len(b), []).append(i)
    grouped = []
    for n, ids in sorted(group_ids.items()):
        grouped.append((np.stack([np.asarray(base[i], dtype=np.float64) for i in ids]),
                        np.stack([np.asarray(candidate[i], dtype=np.float64) for i in ids])))
    mean_deltas, p90_improvements, over_deltas = [], [], []
    batch = 100
    for start in range(0, repeats, batch):
        b_parts, c_parts = [], []
        for bmat, cmat in grouped:
            ix = rng.integers(0, len(bmat), size=(min(batch, repeats - start), len(bmat)))
            b_parts.append(bmat[ix].reshape(len(ix), -1))
            c_parts.append(cmat[ix].reshape(len(ix), -1))
        bdraw = np.concatenate(b_parts, axis=1)
        cdraw = np.concatenate(c_parts, axis=1)
        mean_deltas.extend(np.mean(cdraw - bdraw, axis=1).tolist())
        p90_improvements.extend((np.percentile(bdraw, 90, axis=1) - np.percentile(cdraw, 90, axis=1)).tolist())
        over_deltas.extend((np.mean(cdraw > 1.0, axis=1) - np.mean(bdraw > 1.0, axis=1)).tolist())
    return {"mean_delta_m_candidate_minus_baseline": float(np.mean(np.concatenate(candidate)) - np.mean(np.concatenate(base))),
            "mean_delta_ci95_m": np.quantile(mean_deltas, [0.025, 0.975]).astype(float).tolist(),
            "p90_improvement_m_baseline_minus_candidate": float(np.percentile(np.concatenate(base), 90) - np.percentile(np.concatenate(candidate), 90)),
            "p90_improvement_ci95_m": np.quantile(p90_improvements, [0.025, 0.975]).astype(float).tolist(),
            "over_1m_delta_candidate_minus_baseline": float(np.mean(np.concatenate(candidate) > 1.0) - np.mean(np.concatenate(base) > 1.0)),
            "over_1m_delta_ci95": np.quantile(over_deltas, [0.025, 0.975]).astype(float).tolist(),
            "bootstrap_repeats": repeats, "resampling_unit": "scene cluster"}


def summarize_geometry(rows: list[dict], stage: str) -> dict:
    by_cell: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        for layout, errs in row["errors"]["srp"].items():
            by_cell.setdefault((row["N"], row["M"], layout), []).append({"scene_id": row["scene_id"], "errors": errs})
    report = {"stage": stage, "metrics_by_n_m_layout": [], "aggregate_by_m": [], "paired_vs_perimeter": [], "winners": {}}
    for (n, m, layout), items in sorted(by_cell.items()):
        errs = [x["errors"] for x in items]
        report["metrics_by_n_m_layout"].append({"N": n, "M": m, "layout": layout, **_metric(errs)})
    for m in (4, 6, 8):
        candidates = {}
        for layout in LAYOUT_NAMES:
            per_n = [by_cell.get((n, m, layout), []) for n in (1, 2, 3)]
            if not all(per_n):
                continue
            errs = [item["errors"] for group in per_n for item in group]
            candidates[layout] = (errs, _metric(errs))
            report["aggregate_by_m"].append({"M": m, "layout": layout, **candidates[layout][1]})
        base, base_metric = candidates["perimeter"]
        best_name = min(candidates, key=lambda name: 0.5 * (candidates[name][1]["mean_m"] / base_metric["mean_m"] + candidates[name][1]["p90_m"] / base_metric["p90_m"]))
        report["winners"][str(m)] = {"layout": best_name, "selection_objective": "minimize mean of normalized mean and P90 relative to perimeter on validation screen", "screen_metrics": candidates[best_name][1]}
        for layout, (err, metric) in candidates.items():
            report["paired_vs_perimeter"].append({"M": m, "layout": layout, **_paired_bootstrap(base, err, SEED + m * 100 + len(layout), repeats=5000)})
    write_json(OUT / f"{stage}_summary.json", report)
    if stage == "triage":
        write_json(OUT / "geometry_winners.json", {"status": "selected_on_validation_screen_only", "winners": report["winners"]})
    print(json.dumps({"stage": stage, "winners": report["winners"]}, ensure_ascii=False), flush=True)
    return report


def run_reranker() -> None:
    base = np.load(VAL_MAPS, allow_pickle=True)
    # NPZ members are lazy and compressed. Materialize large arrays once rather
    # than triggering a full decompression on every scene lookup.
    srp_maps = base["srp"]
    scene_ids = base["ids"]
    xs, ys = base["xs"], base["ys"]
    records = read_jsonl(VAL_SCENES)
    by_id = {row["scene_id"]: row for row in records}
    configs = {
        "peak_prom_025": {"peak": 1.0, "prom": 0.25, "sharp": 0.0},
        "peak_prom_050": {"peak": 1.0, "prom": 0.50, "sharp": 0.0},
        "peak_prom_100": {"peak": 1.0, "prom": 1.00, "sharp": 0.0},
        "peak_sharp_025": {"peak": 1.0, "prom": 0.0, "sharp": 0.25},
        "peak_sharp_050": {"peak": 1.0, "prom": 0.0, "sharp": 0.50},
        "peak_prom_sharp": {"peak": 1.0, "prom": 0.25, "sharp": 0.25},
        "prominence_only": {"peak": 0.0, "prom": 1.0, "sharp": 0.0},
    }
    names = list(configs)
    per_method: dict[str, list[list[float]]] = {"srp": []} | {name: [] for name in names}
    by_group: dict[tuple[int, int, str], list[list[float]]] = {}
    coverage = []
    t0 = time.perf_counter()
    for i, scene_id_raw in enumerate(scene_ids):
        scene_id = str(scene_id_raw)
        rec = by_id.get(scene_id)
        if rec is None:
            continue
        score = np.asarray(srp_maps[i], dtype=np.float32)
        truth = np.asarray(rec["source_positions_xy"], dtype=np.float32)
        candidates, indices = _candidate_points(score, xs, ys, 50, 0.10)
        features = _candidate_features(score, indices)
        d = np.linalg.norm(truth[:, None, :] - candidates[None, :, :], axis=2)
        coverage.extend(np.min(d, axis=1).tolist())
        baseline = _baseline_points(score, xs, ys, int(rec["n_sources"]))
        per_method["srp"].append(_matched_errors(truth, baseline))
        key = (int(rec["n_sources"]), int(rec["n_microphones"]), str(rec["geometry"]))
        by_group.setdefault((*key, "srp"), []).append(_matched_errors(truth, baseline))
        for name, weights in configs.items():
            pred = _select_candidates(candidates, features, int(rec["n_sources"]), weights)
            errs = _matched_errors(truth, pred)
            per_method[name].append(errs)
            by_group.setdefault((*key, name), []).append(errs)
    base_metric = _metric(per_method["srp"])
    metrics = []
    for method, errors in per_method.items():
        metrics.append({"method": method, **_metric(errors),
                        "paired_vs_srp": {} if method == "srp" else _paired_bootstrap(per_method["srp"], errors, SEED + len(method), 5000)})
    by_stratum = []
    for (n, m, geom, method), errors in sorted(by_group.items()):
        by_stratum.append({"N": n, "M": m, "geometry": geom, "method": method, **_metric(errors)})
    # The screen is exploratory: advance only a deterministic rule with a paired CI
    # supporting a primary-metric gain and no statistically clear harm to the >1 m rate.
    eligible = []
    for item in metrics:
        if item["method"] == "srp":
            continue
        paired = item["paired_vs_srp"]
        improves_mean = paired["mean_delta_ci95_m"][1] < 0
        improves_p90 = paired["p90_improvement_ci95_m"][0] > 0
        guard = paired["over_1m_delta_ci95"][0] <= 0
        if (improves_mean or improves_p90) and guard:
            eligible.append(item)
    selected = None
    if eligible:
        selected = min(eligible, key=lambda x: 0.5 * (x["mean_m"] / base_metric["mean_m"] + x["p90_m"] / base_metric["p90_m"]))
    summary = {"status": "validation_screen_only", "source_map": str(VAL_MAPS.relative_to(ROOT)),
               "candidate_count": 50, "baseline": base_metric, "metrics": metrics,
               "stratified_metrics": by_stratum,
               "oracle_top50_coverage_le_0_5m": float(np.mean(np.asarray(coverage) <= 0.5)),
               "selected_rule": None if selected is None else {"method": selected["method"], "weights": configs[selected["method"]]},
               "selection_rule": "either lower mean or lower P90, with <=1 pp higher >1 m rate; confirmation must be dev-other",
               "seconds_wall": time.perf_counter() - t0}
    write_json(OUT / "reranker_screen.json", summary)
    write_json(OUT / "reranker_choice.json", {"status": "selected_on_validation_screen_only",
                                               "method": None if selected is None else selected["method"],
                                               "weights": None if selected is None else configs[selected["method"]]})
    print(json.dumps({"baseline": base_metric, "selected": summary["selected_rule"],
                      "top50_oracle_le_0_5m": summary["oracle_top50_coverage_le_0_5m"]}, ensure_ascii=False), flush=True)


def run_confirm(workers: int) -> None:
    scenes_path = OUT / "holdout_scenes.jsonl"
    if not scenes_path.exists():
        raise FileNotFoundError("holdout congelado ausente; execute scan-external e prepare-external")
    protocol = json.loads((OUT / "holdout_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("opened_for_selection") is not False:
        raise RuntimeError("protocolo externo não está congelado como holdout")
    audit = json.loads((OUT / "dev_other_audio_audit.json").read_text(encoding="utf-8"))
    if not audit.get("speaker_disjoint") or not audit.get("audio_disjoint"):
        raise RuntimeError("auditoria de independência falhou")
    library = json.loads((OUT / "array_candidates.json").read_text(encoding="utf-8"))
    winner_data = json.loads((OUT / "geometry_winners.json").read_text(encoding="utf-8"))
    winner_names = {m: row["layout"] for m, row in winner_data["winners"].items()}
    tasks = read_jsonl(scenes_path)
    for task in tasks:
        win = winner_names[str(task["n_microphones"])]
        task["layout_names"] = list(dict.fromkeys(["perimeter", win]))
    audio_index = load_audio_index(OUT / "dev_other_audio_r6.jsonl")
    layouts = {m: {r["name"]: r["positions_xyz"] for r in rows} for m, rows in library.items()}
    choice = json.loads((OUT / "reranker_choice.json").read_text(encoding="utf-8"))
    output = _run_parallel("confirm", tasks, audio_index, layouts, workers, choice.get("weights"))
    summarize_confirmation(read_jsonl(output), winner_names)


def summarize_confirmation(rows: list[dict], winner_names: dict[str, str]) -> dict:
    output = {"status": "independent_dev-other_confirmation", "winners": winner_names,
              "primary_metrics": [], "paired_vs_perimeter": [], "speaker_cluster_sensitivity": []}
    grouped: dict[tuple[int, int, str, str], list[dict]] = {}
    for row in rows:
        win = winner_names[str(row["M"])]
        for method, layout_predictions in row["errors"].items():
            for layout, errs in layout_predictions.items():
                grouped.setdefault((row["N"], row["M"], layout, method), []).append(
                    {"scene_id": row["scene_id"], "speaker_ids": row.get("speaker_ids", []), "errors": errs,
                     "speaker_errors": row.get("speaker_errors", {}).get(method, {}).get(layout, {})})
    for (n, m, layout, method), items in sorted(grouped.items()):
        output["primary_metrics"].append({"N": n, "M": m, "layout": layout, "decoder": method,
                                          **_metric([x["errors"] for x in items])})
    for m in (4, 6, 8):
        for method in sorted({key[3] for key in grouped if key[1] == m}):
            base_items = []
            cand_items = []
            for n in (1, 2, 3):
                base_items.extend(grouped.get((n, m, "perimeter", method), []))
                candidate_name = winner_names[str(m)]
                cand_items.extend(grouped.get((n, m, candidate_name, method), []))
            if not base_items or not cand_items:
                continue
            b = [x["errors"] for x in base_items]
            c = [x["errors"] for x in cand_items]
            paired = _paired_bootstrap(b, c, SEED + m * 1000 + len(method), 10000)
            output["paired_vs_perimeter"].append({"M": m, "decoder": method,
                                                  "candidate_layout": winner_names[str(m)],
                                                  "baseline": _metric(b), "candidate": _metric(c), **paired})
            speaker_pairs: dict[str, list[tuple[float, float]]] = {}
            cand_by_scene = {x["scene_id"]: x for x in cand_items}
            for base_item in base_items:
                candidate_item = cand_by_scene[base_item["scene_id"]]
                for speaker, base_err in base_item["speaker_errors"].items():
                    if speaker in candidate_item["speaker_errors"]:
                        speaker_pairs.setdefault(speaker, []).append(
                            (float(base_err), float(candidate_item["speaker_errors"][speaker])))
            if len(speaker_pairs) >= 5:
                output["speaker_cluster_sensitivity"].append({
                    "M": m, "decoder": method, "candidate_layout": winner_names[str(m)],
                    **_speaker_bootstrap(speaker_pairs, SEED + m * 10_000 + len(method), 2000)})
    write_json(OUT / "confirmation_summary.json", output)
    print(json.dumps({"confirmation": output["paired_vs_perimeter"]}, ensure_ascii=False), flush=True)
    return output


def _speaker_bootstrap(pairs_by_speaker: dict[str, list[tuple[float, float]]], seed: int,
                       repeats: int = 2000) -> dict:
    """Sensitivity bootstrap that resamples complete speaker clusters."""
    speakers = sorted(pairs_by_speaker)
    arrays = [np.asarray(pairs_by_speaker[s], dtype=np.float64) for s in speakers]
    base_all = np.concatenate([a[:, 0] for a in arrays])
    candidate_all = np.concatenate([a[:, 1] for a in arrays])
    rng = np.random.default_rng(seed)
    mean_delta, p90_gain, over_delta = [], [], []
    for _ in range(repeats):
        draws = rng.integers(0, len(arrays), size=len(arrays))
        b = np.concatenate([arrays[i][:, 0] for i in draws])
        c = np.concatenate([arrays[i][:, 1] for i in draws])
        mean_delta.append(float(c.mean() - b.mean()))
        p90_gain.append(float(np.percentile(b, 90) - np.percentile(c, 90)))
        over_delta.append(float(np.mean(c > 1.0) - np.mean(b > 1.0)))
    return {"speaker_cluster_count": len(speakers), "speaker_cluster_scene_source_count": int(len(base_all)),
            "mean_delta_m_candidate_minus_baseline": float(candidate_all.mean() - base_all.mean()),
            "mean_delta_ci95_m": np.quantile(mean_delta, [0.025, 0.975]).astype(float).tolist(),
            "p90_improvement_m_baseline_minus_candidate": float(np.percentile(base_all, 90) - np.percentile(candidate_all, 90)),
            "p90_improvement_ci95_m": np.quantile(p90_gain, [0.025, 0.975]).astype(float).tolist(),
            "over_1m_delta_candidate_minus_baseline": float(np.mean(candidate_all > 1.0) - np.mean(base_all > 1.0)),
            "over_1m_delta_ci95": np.quantile(over_delta, [0.025, 0.975]).astype(float).tolist(),
            "bootstrap_repeats": repeats,
            "note": "sensibilidade por falante; o bootstrap principal continua reamostrando cenas"}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("smoke")
    q = sub.add_parser("triage"); q.add_argument("--workers", type=int, default=6)
    sub.add_parser("reranker")
    sub.add_parser("probe-external-audio")
    s = sub.add_parser("scan-external"); s.add_argument("root", type=Path)
    e = sub.add_parser("prepare-external"); e.add_argument("--count-per-cell", type=int, default=300)
    c = sub.add_parser("confirm"); c.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare": prepare()
    elif args.command == "smoke": smoke()
    elif args.command == "triage": run_triage(args.workers)
    elif args.command == "reranker": run_reranker()
    elif args.command == "probe-external-audio": probe_external_audio()
    elif args.command == "scan-external": scan_external(args.root)
    elif args.command == "prepare-external": prepare_external(args.count_per_cell)
    elif args.command == "confirm": run_confirm(args.workers)


if __name__ == "__main__":
    main()
