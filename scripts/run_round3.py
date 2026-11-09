"""Round 3: fixed, budgeted experiments for multi-source localization."""
from __future__ import annotations

# Import torch before NumPy on Windows to avoid conflicting OpenMP runtimes.
import torch
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from voice_localization.audio import anechoic_clean_mix, normalize_active, read_audio, read_multichannel_audio
from voice_localization.config import load_config
from voice_localization.dataset import load_map_sample
from voice_localization.manifest import AudioRecord, SceneRecord, read_jsonl, write_jsonl
from voice_localization.metrics import pairwise_assignment_errors
from voice_localization.model import build_model
from voice_localization.spatial import (
    gcc_phat_pairs,
    make_grid,
    score_srp_points,
    srp_phat_map,
)
from run_round2 import peaks

OUT = ROOT / "artifacts" / "round3"
VAL_PATH = ROOT / "data" / "full" / "manifests" / "scenes_validation.jsonl"
TRAIN_PATH = ROOT / "data" / "full" / "manifests" / "scenes_train.jsonl"
SEEDS = (42, 43)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_subset(records: list[SceneRecord], count: int, seed: int) -> list[SceneRecord]:
    groups: dict[tuple, list[SceneRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.n_sources, rec.n_microphones, rec.geometry)].append(rec)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    chosen: list[SceneRecord] = []
    keys = sorted(groups)
    while len(chosen) < min(count, len(records)):
        progressed = False
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return chosen


def triage_subset(records: list[SceneRecord]) -> list[SceneRecord]:
    """20 per N/M/geometry cell, spread over requested SNR and RT60 bins."""
    groups: dict[tuple, list[SceneRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.n_sources, rec.n_microphones, rec.geometry)].append(rec)
    rng = np.random.default_rng(20261108)
    chosen = []
    for key in sorted(groups):
        bins: dict[tuple[int, int], list[SceneRecord]] = defaultdict(list)
        for rec in groups[key]:
            si = min(3, max(0, int((rec.snr_requested_db - 10.0) // 5.0)))
            ti = min(3, max(0, int((rec.rt60_requested - 0.2) // 0.15)))
            bins[(si, ti)].append(rec)
        for rows in bins.values():
            rng.shuffle(rows)
        # One per joint bin per pass; empty bins are filled from remaining rows.
        selected = []
        for _ in range(20):
            available = [b for b in sorted(bins) if bins[b]]
            if not available:
                break
            b = available[len(selected) % len(available)]
            selected.append(bins[b].pop())
        if len(selected) < 20:
            rest = [row for rows in bins.values() for row in rows]
            rng.shuffle(rest)
            selected.extend(rest[: 20 - len(selected)])
        if len(selected) != 20:
            raise RuntimeError(f"triagem insuficiente para estrato {key}: {len(selected)}")
        chosen.extend(selected)
    if len(chosen) != 360 or len({r.scene_id for r in chosen}) != 360:
        raise RuntimeError("triagem congelada não contém 360 cenas únicas")
    return chosen


def load_model(seed: int):
    path = OUT.parent / f"cnn_mixture_seed{seed}" / "best.pt"
    if not path.exists():
        path = OUT.parent / f"cnn_mixture_seed{seed}" / "model.pt"
    if seed in SEEDS:
        path = OUT.parent / "round2" / "confirm" / f"seed{seed}" / "best.pt"
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model((1, 2, 4, 8)).to(DEVICE)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model, path


def rank_map(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float32)
    return (rankdata(a.ravel(), method="average").reshape(a.shape) / a.size).astype(np.float32)


def baseline_scene_data(records: list[SceneRecord], out_file: Path | None = None):
    models = {}
    checkpoint_paths = {}
    for seed in SEEDS:
        models[seed], checkpoint_paths[seed] = load_model(seed)
    srps, probs, truths, ids, ns, xs0, ys0 = [], {s: [] for s in SEEDS}, [], [], [], None, None
    started = time.perf_counter()
    with torch.no_grad():
        for i, rec in enumerate(records):
            features, _, n, truth, xs, ys, _ = load_map_sample(rec)
            inp = torch.from_numpy(features[None]).to(DEVICE)
            pseed = {}
            for seed in SEEDS:
                pseed[seed] = models[seed](inp).sigmoid()[0, 0].cpu().numpy().astype(np.float32)
                probs[seed].append(pseed[seed])
            srps.append(features[0].astype(np.float32))
            truths.append(np.asarray(truth, dtype=np.float32))
            ids.append(rec.scene_id)
            ns.append(n)
            xs0, ys0 = xs, ys
            if (i + 1) % 60 == 0 or i + 1 == len(records):
                print("BASELINE", i + 1, "/", len(records), "elapsed_s", round(time.perf_counter() - started, 1), flush=True)
    data = {
        "srp": np.stack(srps),
        "cnn42": np.stack(probs[42]),
        "cnn43": np.stack(probs[43]),
        "truths": np.array(truths, dtype=object),
        "ids": np.asarray(ids),
        "n_sources": np.asarray(ns, dtype=np.int16),
        "xs": xs0,
        "ys": ys0,
    }
    if out_file:
        np.savez_compressed(out_file, srp=data["srp"], cnn42=data["cnn42"], cnn43=data["cnn43"],
                            truths=data["truths"], ids=data["ids"], n_sources=data["n_sources"],
                            xs=data["xs"], ys=data["ys"])
    return data


def assign(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    dist = np.linalg.norm(ref[:, None, :] - est[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(dist)
    return rows, cols, dist[rows, cols]


def errors_for_estimates(records, estimates) -> list[np.ndarray]:
    return [pairwise_assignment_errors(np.asarray(r.source_positions_xy), e) for r, e in zip(records, estimates)]


def metric_summary(errors: list[np.ndarray], records: list[SceneRecord] | None = None) -> dict:
    e = np.concatenate(errors) if errors else np.array([], dtype=float)
    return {
        "scene_count": len(errors), "source_count": int(e.size),
        "mean_error_m": float(e.mean()) if e.size else None,
        "median_error_m": float(np.median(e)) if e.size else None,
        "p90_error_m": float(np.percentile(e, 90)) if e.size else None,
        "within_lt_0_1m_count": int(np.sum(e < 0.10)),
        "within_lt_0_1m_rate": float(np.mean(e < 0.10)) if e.size else None,
        "within_0_25m": float(np.mean(e <= 0.25)) if e.size else None,
        "within_0_5m": float(np.mean(e <= 0.50)) if e.size else None,
        "over_1m_count": int(np.sum(e > 1.0)), "over_1m_rate": float(np.mean(e > 1.0)) if e.size else None,
        "over_2m_count": int(np.sum(e > 2.0)), "over_2m_rate": float(np.mean(e > 2.0)) if e.size else None,
        "all_sources_lt_0_1m_scenes": int(sum(bool(np.all(x < .10)) for x in errors)),
        "all_sources_le_0_5m_scenes": int(sum(bool(np.all(x <= .50)) for x in errors)),
    }


def save_source_errors(path: Path, records: list[SceneRecord], estimates_by_method: dict[str, list[np.ndarray]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "scene_id", "source_index", "reference_x_m", "reference_y_m",
                                               "estimate_x_m", "estimate_y_m", "error_m"])
        writer.writeheader()
        for method, estimates in estimates_by_method.items():
            for rec, est in zip(records, estimates):
                ref = np.asarray(rec.source_positions_xy, dtype=np.float64)
                rows, cols, errs = assign(ref, est)
                for ri, ci, err in zip(rows, cols, errs):
                    writer.writerow({"method": method, "scene_id": rec.scene_id, "source_index": int(ri),
                                     "reference_x_m": f"{ref[ri,0]:.6f}", "reference_y_m": f"{ref[ri,1]:.6f}",
                                     "estimate_x_m": f"{est[ci,0]:.6f}", "estimate_y_m": f"{est[ci,1]:.6f}",
                                     "error_m": f"{err:.8f}"})


def baseline_estimates(data: dict, records: list[SceneRecord]) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    xs, ys = data["xs"], data["ys"]
    methods: dict[str, list[np.ndarray]] = {"srp512": [], "round2_fusion_seed42": [], "round2_fusion_seed43": [], "round2_fusion_meanmap": []}
    score_maps: dict[str, list[np.ndarray]] = {key: [] for key in methods}
    for i, rec in enumerate(records):
        n = rec.n_sources
        srp = data["srp"][i]
        methods["srp512"].append(peaks(srp, xs, ys, n, .60, True))
        score_maps["srp512"].append(srp)
        seed_fusions = []
        for seed in SEEDS:
            fusion = .75 * rank_map(srp) + .25 * data[f"cnn{seed}"][i]
            name = f"round2_fusion_seed{seed}"
            est = peaks(fusion, xs, ys, n, .60, True)
            methods[name].append(est); score_maps[name].append(fusion); seed_fusions.append(fusion)
        avg = np.mean(seed_fusions, axis=0).astype(np.float32)
        methods["round2_fusion_meanmap"].append(peaks(avg, xs, ys, n, .60, True))
        score_maps["round2_fusion_meanmap"].append(avg)
    return methods, score_maps


def estimate_latency_and_oracle(data, records):
    xs, ys = data["xs"], data["ys"]
    oracle_rows = []
    for i, rec in enumerate(records):
        srp = data["srp"][i]
        cand = peaks(srp, xs, ys, min(32, srp.size), .10, True)
        ref = np.asarray(rec.source_positions_xy)
        dist = np.linalg.norm(ref[:, None, :] - cand[None, :, :], axis=-1)
        _, _, best = assign(ref, cand)
        base = peaks(srp, xs, ys, rec.n_sources, .60, True)
        errors = pairwise_assignment_errors(ref, base)
        oracle_rows.append({"scene_id": rec.scene_id, "n_sources": rec.n_sources,
                            "oracle_lt_0_25": int(np.sum(best <= .25)), "oracle_lt_0_50": int(np.sum(best <= .5)),
                            "sources": int(rec.n_sources), "selected_over_1m": int(np.sum(errors > 1.0))})
    return oracle_rows


def save_metrics(path: Path, records, estimates_by_method):
    rows = []
    for method, estimates in estimates_by_method.items():
        errors = errors_for_estimates(records, estimates)
        rows.append({"method": method, **metric_summary(errors, records)})
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def paired_scene_bootstrap(records, errors_by_method, baseline, seed=20261108, repeats=10000):
    rng = np.random.default_rng(seed)
    n = len(records)
    ids = rng.integers(0, n, size=(repeats, n))
    output = {}
    base = errors_by_method[baseline]
    for name, errors in errors_by_method.items():
        if name == baseline:
            continue
        deltas = np.full((n, 3), np.nan, dtype=np.float64)
        for i, (a, b) in enumerate(zip(errors, base)):
            d = np.asarray(a) - np.asarray(b)
            deltas[i, : min(3, len(d))] = d[:3]
        sampled = deltas[ids]
        # Preserve N=1/2/3 within scenes: mean per source, not mean per scene.
        values = sampled.sum(axis=(1, 2)) / (np.isfinite(sampled).sum(axis=(1, 2)).clip(min=1))
        point = float(np.mean([np.mean(a - b) for a, b in zip(errors, base)]))
        output[name] = {"baseline": baseline, "mean_delta_m": point,
                        "ci95_m": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
                        "repeats": repeats, "unit": "scene", "negative_favors_candidate": True}
    return output


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    train = list(read_jsonl(TRAIN_PATH, SceneRecord))
    val = list(read_jsonl(VAL_PATH, SceneRecord))
    train_s = balanced_subset(train, 6000, 20261108)
    val_s = triage_subset(val)
    write_jsonl(train_s, OUT / "manifests" / "train.jsonl")
    write_jsonl(val_s, OUT / "manifests" / "triage.jsonl")
    checkpoint_info = {}
    for seed in SEEDS:
        p = OUT.parent / "round2" / "confirm" / f"seed{seed}" / "best.pt"
        checkpoint_info[str(seed)] = {"path": str(p), "sha256": sha256(p)}
    protocol = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "frozen_before_round3_evaluation",
        "validation_manifest_sha256": sha256(VAL_PATH),
        "train_subset_seed": 20261108, "triage_seed": 20261108,
        "triage_scenes": len(val_s), "train_subset_scenes": len(train_s),
        "triage_cell_counts": {str(k): sum((r.n_sources,r.n_microphones,r.geometry)==k for r in val_s)
                                for k in sorted({(r.n_sources,r.n_microphones,r.geometry) for r in val_s})},
        "baseline_checkpoints": checkpoint_info,
        "fixed_decoding": {"method": "3x3 local maxima + Euclidean NMS", "radius_m": .60,
                           "fusion": "0.75*rankdata(SRP)/map_size + 0.25*CNN_sigmoid"},
        "strict_precision_target": {"individual_source_errors_lt_m": .10, "minimum_count_on_independent_holdout": 100,
                                    "also_report_all_sources_within_threshold_per_scene": True},
        "promotion_gate": {"mean_improvement_m": .15, "p90_improvement_m": .25,
                           "relative_reduction_errors_over_1m": .20, "within_0_5m_must_not_drop": True},
        "hard_machine_budget_seconds": 43200,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("FROZEN", json.dumps({"triage": len(val_s), "train": len(train_s), "protocol": str(OUT / 'protocol.json')}, ensure_ascii=False), flush=True)


def baseline_stage():
    records = list(read_jsonl(OUT / "manifests" / "triage.jsonl", SceneRecord))
    started = time.perf_counter()
    data = baseline_scene_data(records, OUT / "triage_baseline_maps.npz")
    estimates, scores = baseline_estimates(data, records)
    rows = save_metrics(OUT / "triage_baseline_metrics.json", records, estimates)
    save_source_errors(OUT / "triage_baseline_source_errors.csv", records, estimates)
    oracle = estimate_latency_and_oracle(data, records)
    (OUT / "oracle_diagnostic.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    result = {r["method"]: {"mean": r["mean_error_m"], "P90": r["p90_error_m"],
                            "<0.10_count": r["within_lt_0_1m_count"], ">1_count": r["over_1m_count"]} for r in rows}
    (OUT / "timing_baseline.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("TRIAGE_BASELINE", json.dumps(result), flush=True)


def load_triage_arrays():
    records = list(read_jsonl(OUT / "manifests" / "triage.jsonl", SceneRecord))
    z = np.load(OUT / "triage_baseline_maps.npz", allow_pickle=True)
    data = {k: z[k] for k in z.files}
    return records, data


def acoustic_map(rec: SceneRecord, cfg, window: int, window_type: str = "hann"):
    audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
    mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
    mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
    result, grid = srp_phat_map(
        audio, mics, cfg.simulation.audio.sample_rate, tuple(cfg.simulation.room.dimensions),
        resolution_m=cfg.grid.resolution_m, speed_of_sound=cfg.simulation.room.speed_of_sound,
        window=window, hop=window // 2, fft_size=window * 2, phat_exponent=1.0, window_type=window_type,
    )
    return result, grid


def candidate_metrics(records, data, candidate_maps, name_prefix):
    xs, ys = data["xs"], data["ys"]
    estimates = {f"{name_prefix}_physical": [], f"{name_prefix}_fused_rank_both": []}
    for i, (rec, amap) in enumerate(zip(records, candidate_maps)):
        physical = peaks(amap, xs, ys, rec.n_sources, .60, True)
        fused = .75 * rank_map(amap) + .25 * rank_map(.5 * (data["cnn42"][i] + data["cnn43"][i]))
        estimates[f"{name_prefix}_physical"].append(physical)
        estimates[f"{name_prefix}_fused_rank_both"].append(peaks(fused, xs, ys, rec.n_sources, .60, True))
    return estimates


def run_h1():
    records, data = load_triage_arrays()
    cfg = load_config(ROOT / "configs" / "full.yaml")
    variants = [(2048, "hann"), (4096, "hann"), (2048, "rectangular")]
    all_estimates = {}
    started = time.perf_counter()
    for window, wtype in variants:
        name = f"w{window}_{wtype}"
        maps = []
        for i, rec in enumerate(records):
            m, _ = acoustic_map(rec, cfg, window, wtype)
            maps.append(m)
            if (i + 1) % 60 == 0 or i + 1 == len(records):
                print("H1", name, i + 1, "/", len(records), "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
        np.savez_compressed(OUT / f"h1_{name}.npz", maps=np.stack(maps))
        all_estimates.update(candidate_metrics(records, data, maps, name))
        save_metrics(OUT / f"h1_{name}_metrics.json", records,
                     {k: v for k,v in all_estimates.items() if k.startswith(name)})
    summary = save_metrics(OUT / "h1_metrics.json", records, all_estimates)
    save_source_errors(OUT / "h1_source_errors.csv", records, all_estimates)
    (OUT / "timing_h1.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("H1_RESULT", json.dumps([{"method": r["method"], "mean": r["mean_error_m"], "P90": r["p90_error_m"],
                                  "<0.1": r["within_lt_0_1m_count"], ">1": r["over_1m_count"]} for r in summary]), flush=True)


def block_band_gcc_maps(audio, microphones, rec, cfg, window, window_type="hann"):
    """Four temporal blocks × four frequency bands, with energy weights."""
    from voice_localization.spatial import frame_signal
    sr = cfg.simulation.audio.sample_rate
    fft_size = window * 2
    hop = window // 2
    bands = ((300, 800), (800, 1600), (1600, 3000), (3000, 6000))
    pairs = [(i, j) for i in range(len(audio)) for j in range(i + 1, len(audio))]
    win = np.hanning(window).astype(np.float32) if window_type == "hann" else np.ones(window, dtype=np.float32)
    block_size = int(round(.5 * sr))
    blocks = []
    for bi in range(4):
        segment = audio[:, bi * block_size:min((bi + 1) * block_size, audio.shape[1])]
        if segment.shape[1] < window:
            continue
        block_power = float(np.mean(segment.astype(np.float64) ** 2))
        blocks.append((bi, segment, block_power))
    if not blocks:
        return [], []
    max_power = max(x[2] for x in blocks)
    blocks = [x for x in blocks if x[2] >= .10 * max_power and x[2] > 1e-12]
    total_power = sum(x[2] for x in blocks) or 1.0
    freq = np.fft.rfftfreq(fft_size, 1 / sr)
    band_masks = [(freq >= low) & (freq < high) for low, high in bands]
    maps, weights = [], []
    grid = make_grid(tuple(cfg.simulation.room.dimensions), cfg.grid.resolution_m)
    for _, segment, block_power in blocks:
        channel_frames = [frame_signal(ch, window, hop) * win[None, :] for ch in segment]
        spectra = np.asarray([np.fft.rfft(frames, n=fft_size, axis=1) for frames in channel_frames])
        spectral_energy = np.mean(np.abs(spectra) ** 2, axis=(0, 1))
        all_energy = float(np.sum(spectral_energy)) + 1e-12
        for band_mask in band_masks:
            if not np.any(band_mask):
                maps.append(np.zeros(grid.shape, dtype=np.float32)); weights.append(0.0); continue
            band_energy = float(np.sum(spectral_energy[band_mask]))
            weight = (block_power / total_power) * (band_energy / all_energy)
            corrs = []
            for i, j in pairs:
                cross = spectra[i] * np.conj(spectra[j])
                cross[:, ~band_mask] = 0.0
                cross /= np.maximum(np.abs(cross), 1e-8)
                corr = np.fft.fftshift(np.fft.irfft(cross, n=fft_size, axis=1), axes=1)
                corrs.append(np.mean(corr, axis=0).astype(np.float32))
            score = score_srp_points(
                np.asarray(corrs), pairs, microphones, grid.points_xy, sr, fft_size,
                cfg.simulation.room.speed_of_sound,
            ).reshape(grid.shape)
            maps.append(score); weights.append(weight)
    weights = np.asarray(weights, dtype=np.float64)
    if not np.any(weights > 0):
        weights[:] = 1.0
    weights /= weights.sum()
    return maps, weights.tolist()


def map_consensus(maps, weights, xs, ys, n_sources, eps):
    votes = []
    for map_index, (score_map, weight) in enumerate(zip(maps, weights)):
        if weight <= 0:
            continue
        candidates = peaks(score_map, xs, ys, min(3, score_map.size), .20, True)
        ranked = rank_map(score_map)
        for point in candidates:
            ix = int(np.argmin(np.abs(xs - point[0])))
            iy = int(np.argmin(np.abs(ys - point[1])))
            votes.append((point.astype(np.float64), map_index, float(weight * ranked[iy, ix])))
    if not votes:
        return peaks(np.mean(maps, axis=0), xs, ys, n_sources, .60, True)
    parent = list(range(len(votes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    coords = np.stack([v[0] for v in votes])
    for i in range(len(votes)):
        for j in range(i + 1, len(votes)):
            if np.linalg.norm(coords[i] - coords[j]) <= eps:
                union(i, j)
    groups = defaultdict(list)
    for i, vote in enumerate(votes):
        groups[find(i)].append(vote)
    clusters = []
    for members in groups.values():
        distinct_maps = {x[1] for x in members}
        if len(distinct_maps) < 2:
            continue
        points = np.stack([x[0] for x in members])
        w = np.asarray([x[2] for x in members])
        if w.sum() <= 0:
            continue
        center = []
        for dim in range(2):
            order = np.argsort(points[:, dim])
            csum = np.cumsum(w[order])
            center.append(points[order[np.searchsorted(csum, csum[-1] / 2)], dim])
        clusters.append((float(w.sum()), np.asarray(center, dtype=np.float32)))
    clusters.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for _, point in clusters:
        if all(np.linalg.norm(point - old) >= .60 for old in selected):
            selected.append(point)
        if len(selected) == n_sources:
            break
    global_map = np.average(np.stack(maps), axis=0, weights=np.asarray(weights))
    for point in peaks(global_map, xs, ys, min(global_map.size, n_sources * 4), .60, True):
        if all(np.linalg.norm(point - old) >= .60 for old in selected):
            selected.append(point)
        if len(selected) == n_sources:
            break
    return np.asarray(selected[:n_sources], dtype=np.float32)


def run_h2():
    records, data = load_triage_arrays()
    cfg = load_config(ROOT / "configs" / "full.yaml")
    h1rows = json.loads((OUT / "h1_metrics.json").read_text(encoding="utf-8"))
    baseline = next(x for x in json.loads((OUT / "triage_baseline_metrics.json").read_text(encoding="utf-8"))
                    if x["method"] == "round2_fusion_meanmap")
    eligible = [r for r in h1rows if r["method"].endswith("fused_rank_both")
                and r["mean_error_m"] <= baseline["mean_error_m"] - .05
                and r["over_1m_rate"] <= baseline["over_1m_rate"]]
    if eligible:
        chosen = min(eligible, key=lambda x: x["mean_error_m"])
        prefix = chosen["method"].removesuffix("_fused_rank_both")
        window = int(prefix.split("_")[0][1:])
        window_type = "rectangular" if prefix.endswith("rectangular") else "hann"
    else:
        window, window_type = 512, "hann"
    all_estimates = {"h2_weighted_mean": [], "h2_consensus_eps025": [], "h2_consensus_eps050": []}
    started = time.perf_counter()
    for i, rec in enumerate(records):
        audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
        mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
        mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
        maps, weights = block_band_gcc_maps(audio, mics, rec, cfg, window, window_type)
        if not maps:
            maps, weights = [data["srp"][i]], [1.0]
        ranks = np.stack([rank_map(m) for m in maps])
        mean_map = np.average(ranks, axis=0, weights=np.asarray(weights))
        all_estimates["h2_weighted_mean"].append(peaks(mean_map, data["xs"], data["ys"], rec.n_sources, .60, True))
        all_estimates["h2_consensus_eps025"].append(map_consensus(maps, weights, data["xs"], data["ys"], rec.n_sources, .25))
        all_estimates["h2_consensus_eps050"].append(map_consensus(maps, weights, data["xs"], data["ys"], rec.n_sources, .50))
        if (i + 1) % 30 == 0 or i + 1 == len(records):
            print("H2", i + 1, "/", len(records), "window", window, window_type,
                  "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
    rows = save_metrics(OUT / "h2_metrics.json", records, all_estimates)
    save_source_errors(OUT / "h2_source_errors.csv", records, all_estimates)
    (OUT / "h2_config.json").write_text(json.dumps({"window": window, "window_type": window_type,
        "blocks_seconds": .5, "bands_hz": [[300,800],[800,1600],[1600,3000],[3000,6000]],
        "per_scene_energy_rejection": .10, "weights": "block_energy_fraction × band_energy_fraction"}, indent=2), encoding="utf-8")
    (OUT / "timing_h2.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("H2_RESULT", json.dumps([{"method": r["method"], "mean": r["mean_error_m"], "P90": r["p90_error_m"],
                                  "<0.1": r["within_lt_0_1m_count"], ">1": r["over_1m_count"]} for r in rows]), flush=True)


def top_peak_not_near(score_map, xs, ys, selected, radius=.60):
    candidates = peaks(score_map, xs, ys, min(score_map.size, 128), .10, True)
    for candidate in candidates:
        if all(np.linalg.norm(candidate - old) >= radius for old in selected):
            return candidate
    return peaks(score_map, xs, ys, 1, .01, True)[0]


def run_deemphasis(corrs, pairs, microphones, xs, ys, rec, cfg, window, width, depth):
    fft_size = window * 2
    sr = cfg.simulation.audio.sample_rate
    grid = make_grid(tuple(cfg.simulation.room.dimensions), cfg.grid.resolution_m)
    lags = np.arange(-(fft_size // 2), fft_size // 2, dtype=np.float64)
    residual = np.asarray(corrs, dtype=np.float32).copy()
    selected = []
    for _ in range(rec.n_sources):
        score = score_srp_points(residual, pairs, microphones, grid.points_xy, sr, fft_size,
                                 cfg.simulation.room.speed_of_sound).reshape(grid.shape)
        if selected:
            point = top_peak_not_near(score, xs, ys, selected)
        else:
            point = peaks(score, xs, ys, 1, .01, True)[0]
        selected.append(point)
        if len(selected) == rec.n_sources:
            break
        xyz = np.column_stack((np.asarray(rec.microphone_positions_xy),
                               np.full(rec.n_microphones, cfg.simulation.room.microphone_height)))
        source = np.array([point[0], point[1], cfg.simulation.room.source_height])
        for pi, (i, j) in enumerate(pairs):
            tau = (np.linalg.norm(source - xyz[i]) - np.linalg.norm(source - xyz[j])) * sr / cfg.simulation.room.speed_of_sound
            notch = 1.0 - depth * np.exp(-.5 * ((lags - tau) / width) ** 2)
            residual[pi] *= notch.astype(np.float32)
    return np.asarray(selected, dtype=np.float32)


def run_h3():
    records, data = load_triage_arrays()
    cfg = load_config(ROOT / "configs" / "full.yaml")
    h2cfg = json.loads((OUT / "h2_config.json").read_text(encoding="utf-8")) if (OUT / "h2_config.json").exists() else {}
    window = int(h2cfg.get("window", 512))
    wtype = h2cfg.get("window_type", "hann")
    variants = [(1, .5), (1, 1.), (3, .5), (3, 1.)]
    out = {f"h3_width{w}_depth{d:g}": [] for w, d in variants}
    started = time.perf_counter()
    for i, rec in enumerate(records):
        audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
        mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
        mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
        corrs, pairs = gcc_phat_pairs(audio, window, window // 2, window * 2, phat_exponent=1.0, window_type=wtype)
        for width, depth in variants:
            out[f"h3_width{width}_depth{depth:g}"].append(
                run_deemphasis(corrs, pairs, mics, data["xs"], data["ys"], rec, cfg, window, width, depth))
        if (i + 1) % 60 == 0 or i + 1 == len(records):
            print("H3", i + 1, "/", len(records), "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
    rows = save_metrics(OUT / "h3_metrics.json", records, out)
    save_source_errors(OUT / "h3_source_errors.csv", records, out)
    (OUT / "timing_h3.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("H3_RESULT", json.dumps([{"method": r["method"], "mean": r["mean_error_m"], "P90": r["p90_error_m"],
                                  "<0.1": r["within_lt_0_1m_count"], ">1": r["over_1m_count"]} for r in rows]), flush=True)


def quadratic_refine(score_map, estimate, xs, ys):
    refined = []
    for point in estimate:
        ix = int(np.argmin(np.abs(xs - point[0]))); iy = int(np.argmin(np.abs(ys - point[1])))
        dx = dy = 0.0
        if 0 < ix < len(xs) - 1:
            left, center, right = score_map[iy, ix-1], score_map[iy, ix], score_map[iy, ix+1]
            denom = left - 2 * center + right
            if denom < -1e-12:
                dx = float(np.clip(.5 * (left - right) / denom, -.5, .5))
        if 0 < iy < len(ys) - 1:
            left, center, right = score_map[iy-1, ix], score_map[iy, ix], score_map[iy+1, ix]
            denom = left - 2 * center + right
            if denom < -1e-12:
                dy = float(np.clip(.5 * (left - right) / denom, -.5, .5))
        refined.append([point[0] + dx * (xs[1]-xs[0]), point[1] + dy * (ys[1]-ys[0])])
    return preserve_separation(np.asarray(refined, dtype=np.float32), estimate, .60)


def preserve_separation(refined, original, radius):
    output = []
    for point, fallback in zip(refined, original):
        use = point if all(np.linalg.norm(point - old) >= radius for old in output) else fallback
        if any(np.linalg.norm(use - old) < radius for old in output):
            use = fallback
        output.append(np.asarray(use, dtype=np.float32))
    return np.asarray(output, dtype=np.float32)


def physical_refine(rec, initial, cfg, window=512, window_type="hann"):
    audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
    mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
    mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
    corr, pairs = gcc_phat_pairs(audio, window, window // 2, window * 2,
                                 phat_exponent=1.0, window_type=window_type)
    return physical_refine_from_corr(rec, initial, cfg, corr, pairs, window)


def physical_refine_from_corr(rec, initial, cfg, corr, pairs, window):
    mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
    mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
    all_refined, all_scores = [], []
    for point in initial:
        xx = np.clip(np.arange(point[0] - .20, point[0] + .2001, .025), 0, cfg.simulation.room.dimensions[0])
        yy = np.clip(np.arange(point[1] - .20, point[1] + .2001, .025), 0, cfg.simulation.room.dimensions[1])
        gx, gy = np.meshgrid(xx, yy, indexing="xy")
        candidates = np.column_stack((gx.ravel(), gy.ravel()))
        values = score_srp_points(corr, pairs, mics, candidates, cfg.simulation.audio.sample_rate,
                                  window*2, cfg.simulation.room.speed_of_sound)
        j = int(np.argmax(values))
        all_refined.append(candidates[j]); all_scores.append(float(values[j]))
    refined = np.asarray(all_refined, dtype=np.float32)
    for i in range(len(refined)):
        for j in range(i):
            if np.linalg.norm(refined[i] - refined[j]) < .60:
                # Keep the decoder's exact-N, spatially distinct output contract.
                return np.asarray(initial, dtype=np.float32)
    return refined


def run_refine():
    records, data = load_triage_arrays()
    cfg = load_config(ROOT / "configs" / "full.yaml")
    base, score_maps = baseline_estimates(data, records)
    h1_map_file = np.load(OUT / "h1_w2048_rectangular.npz")["maps"]
    methods = {
        "round2_fusion_meanmap_quadratic": [],
        "round2_fusion_meanmap_physical_local": [],
        "h1_w2048_rect_fused_quadratic": [],
        "h1_w2048_rect_fused_physical_local": [],
        "h1_window_ensemble_fused_physical_local": [],
    }
    started = time.perf_counter()
    for i, rec in enumerate(records):
        original = base["round2_fusion_meanmap"][i]
        score = score_maps["round2_fusion_meanmap"][i]
        methods["round2_fusion_meanmap_quadratic"].append(quadratic_refine(score, original, data["xs"], data["ys"]))
        methods["round2_fusion_meanmap_physical_local"].append(physical_refine(rec, original, cfg))
        h1_score = .75 * rank_map(h1_map_file[i]) + .25 * rank_map(.5 * (data["cnn42"][i] + data["cnn43"][i]))
        h1_estimate = peaks(h1_score, data["xs"], data["ys"], rec.n_sources, .60, True)
        methods["h1_w2048_rect_fused_quadratic"].append(
            quadratic_refine(h1_score, h1_estimate, data["xs"], data["ys"]))
        methods["h1_w2048_rect_fused_physical_local"].append(
            physical_refine(rec, h1_estimate, cfg, 2048, "rectangular"))
        ensemble_srp = .5 * rank_map(data["srp"][i]) + .5 * rank_map(h1_map_file[i])
        ensemble_score = .75 * ensemble_srp + .25 * rank_map(.5 * (data["cnn42"][i] + data["cnn43"][i]))
        ensemble_estimate = peaks(ensemble_score, data["xs"], data["ys"], rec.n_sources, .60, True)
        methods["h1_window_ensemble_fused_physical_local"].append(
            physical_refine(rec, ensemble_estimate, cfg, 512, "hann"))
        if (i + 1) % 60 == 0 or i + 1 == len(records):
            print("REFINE", i + 1, "/", len(records), "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
    rows = save_metrics(OUT / "refine_metrics.json", records, methods)
    save_source_errors(OUT / "refine_source_errors.csv", records, methods)
    (OUT / "timing_refine.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("REFINE_RESULT", json.dumps([{"method": r["method"], "mean": r["mean_error_m"], "P90": r["p90_error_m"],
                                      "<0.1": r["within_lt_0_1m_count"], ">1": r["over_1m_count"]} for r in rows]), flush=True)


def run_physics_pilot():
    """18 train scenes: direct/no-noise, reverberant/no-noise, saved mixture."""
    cfg = load_config(ROOT / "configs" / "full.yaml")
    records = list(read_jsonl(TRAIN_PATH, SceneRecord))
    selected = []
    seen = set()
    for rec in records:
        key = (rec.n_sources, rec.n_microphones, rec.geometry)
        if key not in seen:
            selected.append(rec); seen.add(key)
    if len(selected) != 18:
        raise RuntimeError(f"esperadas 18 combinações no piloto, recebidas {len(selected)}")
    audio_index = {a.audio_id: a for a in read_jsonl(ROOT / "data" / "full" / "manifests" / "audio_train.jsonl", AudioRecord)}
    per_method = {name: [] for name in ("direct_no_noise", "reverb_no_noise", "original_mixture")}
    started = time.perf_counter()
    for rec in selected:
        signals = []
        for audio_id, start in zip(rec.audio_ids, rec.metadata["source_window_starts"]):
            record = audio_index[audio_id]
            raw = normalize_active(read_audio(record.path, cfg.simulation.audio.sample_rate))
            size = round((cfg.simulation.audio.duration_seconds + cfg.simulation.audio.context_seconds) * cfg.simulation.audio.sample_rate)
            signal = np.zeros(size, dtype=np.float32)
            end = min(raw.size, int(start) + size)
            if end > int(start):
                signal[:end-int(start)] = raw[int(start):end]
            signals.append(signal)
        src_xy = np.asarray(rec.source_positions_xy, dtype=np.float32)
        src = np.column_stack((src_xy, np.full(len(src_xy), cfg.simulation.room.source_height, dtype=np.float32)))
        mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
        mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
        direct_full = anechoic_clean_mix(signals, src, mics, cfg.simulation.audio.sample_rate, cfg.simulation.room.speed_of_sound)
        crop = round(cfg.simulation.audio.context_seconds * cfg.simulation.audio.sample_rate)
        target_len = round(cfg.simulation.audio.duration_seconds * cfg.simulation.audio.sample_rate)
        direct = direct_full[:, crop:crop+target_len]
        if direct.shape[1] < target_len:
            direct = np.pad(direct, ((0,0),(0,target_len-direct.shape[1])))
        reverb = read_multichannel_audio(rec.clean_audio_path, cfg.simulation.audio.sample_rate)
        mixed = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
        for method, audio in (("direct_no_noise", direct), ("reverb_no_noise", reverb), ("original_mixture", mixed)):
            amap, grid = srp_phat_map(audio, mics, cfg.simulation.audio.sample_rate,
                tuple(cfg.simulation.room.dimensions), resolution_m=cfg.grid.resolution_m,
                speed_of_sound=cfg.simulation.room.speed_of_sound, window=512, hop=256, fft_size=1024)
            estimate = peaks(amap, grid.xs, grid.ys, rec.n_sources, .60, True)
            per_method[method].append(pairwise_assignment_errors(src_xy, estimate))
    result = {key: {**metric_summary(value), "strata": 18} for key,value in per_method.items()}
    (OUT / "physics_pilot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "timing_physics_pilot.json").write_text(json.dumps({"seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print("PHYSICS_PILOT", json.dumps(result), flush=True)


def stratified_metrics(records, errors_by_method, output_path):
    dimensions = [
        ("n_sources", lambda r: str(r.n_sources)),
        ("n_microphones", lambda r: str(r.n_microphones)),
        ("geometry", lambda r: r.geometry),
        ("snr_requested_db", lambda r: f"{int(r.snr_requested_db // 5) * 5}-{int(r.snr_requested_db // 5) * 5 + 5}"),
        ("rt60_realized", lambda r: f"{int((r.rt60_realized or r.rt60_requested) // .2) * .2:.1f}-{(int((r.rt60_realized or r.rt60_requested) // .2) + 1) * .2:.1f}"),
    ]
    rows = []
    for label, fn in dimensions:
        levels = sorted({fn(r) for r in records})
        for level in levels:
            indices = [i for i,r in enumerate(records) if fn(r) == level]
            for method, scene_errors in errors_by_method.items():
                values = np.concatenate([scene_errors[i] for i in indices])
                rows.append({"stratum": label, "level": level, "method": method,
                             "scene_count": len(indices), "source_count": len(values),
                             "mean_error_m": float(values.mean()), "p90_error_m": float(np.percentile(values,90)),
                             "within_lt_0_1m": float(np.mean(values < .10)),
                             "over_1m_rate": float(np.mean(values > 1.0))})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_full():
    records = list(read_jsonl(VAL_PATH, SceneRecord))
    cfg = load_config(ROOT / "configs" / "full.yaml")
    started = time.perf_counter()
    baseline_cache = OUT / "full_baseline_maps.npz"
    if baseline_cache.exists():
        z = np.load(baseline_cache, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        print("FULL_BASELINE_CACHE", len(data["ids"]), flush=True)
    else:
        data = baseline_scene_data(records, baseline_cache)
    base_estimates = {"round2_seed42": [], "round2_seed43": [], "round2_seed42_physical_local": [],
                      "round2_seed43_physical_local": []}
    candidate_estimates = {"h1_w2048_rect_seed42": [], "h1_w2048_rect_seed43": [],
                           "h1_w2048_rect_seed42_physical_local": [], "h1_w2048_rect_seed43_physical_local": []}
    h1_maps = []
    for i, rec in enumerate(records):
        amap, _ = acoustic_map(rec, cfg, 2048, "rectangular")
        h1_maps.append(amap)
        srp = data["srp"][i]
        audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
        mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
        mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
        corr512, pairs512 = gcc_phat_pairs(audio, 512, 256, 1024, phat_exponent=1.0, window_type="hann")
        corr2048, pairs2048 = gcc_phat_pairs(audio, 2048, 1024, 4096, phat_exponent=1.0, window_type="rectangular")
        for seed in SEEDS:
            cnn = data[f"cnn{seed}"][i]
            bscore = .75 * rank_map(srp) + .25 * cnn
            best = peaks(bscore, data["xs"], data["ys"], rec.n_sources, .60, True)
            base_estimates[f"round2_seed{seed}"].append(best)
            base_estimates[f"round2_seed{seed}_physical_local"].append(
                physical_refine_from_corr(rec, best, cfg, corr512, pairs512, 512))
            hscore = .75 * rank_map(amap) + .25 * rank_map(cnn)
            hest = peaks(hscore, data["xs"], data["ys"], rec.n_sources, .60, True)
            candidate_estimates[f"h1_w2048_rect_seed{seed}"].append(hest)
            candidate_estimates[f"h1_w2048_rect_seed{seed}_physical_local"].append(
                physical_refine_from_corr(rec, hest, cfg, corr2048, pairs2048, 2048))
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            np.savez_compressed(OUT / "full_h1_maps_progress.npz", maps=np.stack(h1_maps))
            print("FULL", i + 1, "/", len(records), "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
    np.savez_compressed(OUT / "full_h1_w2048_rect_maps.npz", maps=np.stack(h1_maps))
    all_estimates = {**base_estimates, **candidate_estimates}
    all_errors = {name: errors_for_estimates(records, values) for name, values in all_estimates.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in all_errors.items()]
    (OUT / "full_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "full_validation_source_errors.csv", records, all_estimates)
    stratified_metrics(records, all_errors, OUT / "full_validation_stratified.csv")
    paired = {}
    for seed in SEEDS:
        comparison = {
            "candidate_coarse": all_errors[f"h1_w2048_rect_seed{seed}"],
            "candidate_refined": all_errors[f"h1_w2048_rect_seed{seed}_physical_local"],
            "baseline_refined": all_errors[f"round2_seed{seed}_physical_local"],
        }
        paired[str(seed)] = paired_scene_bootstrap(records, comparison, "baseline_refined", seed=20261108+seed)
    (OUT / "full_validation_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    timing = {"seconds": time.perf_counter()-started, "scene_count": len(records),
              "finalist": "H1: 2048 rectangular SRP rank + 25% seed-specific CNN rank; physical 2D local refinement"}
    (OUT / "timing_full_validation.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("FULL_RESULT", json.dumps([{"method": r["method"], "mean": r["mean_error_m"], "median": r["median_error_m"],
                                     "P90": r["p90_error_m"], "<0.1": r["within_lt_0_1m_count"],
                                     ">1": r["over_1m_count"], "all_lt0.1_scenes": r["all_sources_lt_0_1m_scenes"]}
                                    for r in metrics]), flush=True)


def freeze_finalist():
    checkpoint_hashes = {}
    for seed in SEEDS:
        path = OUT.parent / "round2" / "confirm" / f"seed{seed}" / "best.pt"
        checkpoint_hashes[str(seed)] = {"path": str(path), "sha256": sha256(path)}
    frozen = {
        "frozen_before_independent_holdout_inference": True,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_basis": "triage then complete 3000-scene validation; candidate remains exploratory and does not pass the pre-registered promotion gate",
        "primary_seed": 42,
        "replication_seed": 43,
        "checkpoint_sha256": checkpoint_hashes,
        "comparators": {
            "current_round2": "512-sample Hann SRP rank ×0.75 + same-seed dilated-CNN sigmoid ×0.25; local peaks; NMS 0.60 m",
            "round2_plus_local_refinement": "current Round-2 score and decoder, then physical SRP local search with 512-sample Hann GCC",
            "round3_candidate": "2048-sample rectangular-window SRP rank ×0.75 + same-seed dilated-CNN rank ×0.25; local peaks; NMS 0.60 m; physical local search using the same 2048-sample rectangular GCC",
        },
        "local_refinement": {"half_width_m": 0.20, "step_m": 0.025,
                             "duplicate_rule": "if refined points violate 0.60 m separation, revert the whole scene to the original distinct estimates"},
        "independent_holdout": {"corpus": "LibriSpeech test-other", "scene_count": 600,
                                "seed": 20261109, "must_be_speaker_and_audio_disjoint": True,
                                "manifest_frozen_before_any_inference": True},
        "strict_precision_output": {"source_error_threshold_m": 0.10, "operator": "<",
                                    "minimum_unique_primary_seed_source_cases": 100},
        "no_post_holdout_tuning": True,
    }
    (OUT / "round3_finalist_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("FINALIST_FROZEN", json.dumps({"seed42": checkpoint_hashes["42"]["sha256"],
                                       "seed43": checkpoint_hashes["43"]["sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "baseline", "physics", "h1", "h2", "h3", "refine", "full", "freeze"])
    args = parser.parse_args()
    if args.stage == "prepare": setup()
    elif args.stage == "baseline": baseline_stage()
    elif args.stage == "physics": run_physics_pilot()
    elif args.stage == "h1": run_h1()
    elif args.stage == "h2": run_h2()
    elif args.stage == "h3": run_h3()
    elif args.stage == "refine": run_refine()
    elif args.stage == "full": run_full()
    elif args.stage == "freeze": freeze_finalist()


if __name__ == "__main__":
    main()

