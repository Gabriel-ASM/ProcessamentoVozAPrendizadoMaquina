"""Diagnostics and short experiments for mean/P90 sound-localization error.

Uses the frozen Round-4 triage subset and current validation caches. It never
opens test-other. Experimental methods are screening adaptations, not paper
reproductions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from voice_localization.manifest import SceneRecord, read_jsonl
from voice_localization.metrics import pairwise_assignment_errors
from voice_localization.spatial import frame_signal, make_grid, pair_indices, score_srp_points

OUT = ROOT / "artifacts" / "round5"
VAL_PATH = ROOT / "data" / "full" / "manifests" / "scenes_validation.jsonl"
BASE_CACHE = ROOT / "artifacts" / "round3" / "full_baseline_maps.npz"
PAIR_CACHE = ROOT / "artifacts" / "round4" / "validation_pairwise.npz"
FS, C, WINDOW, HOP, FFT = 16_000, 343.0, 512, 256, 1024


def triage_subset(records):
    """Recreate the fixed Round-4 triage selection, without importing Torch."""
    groups = defaultdict(list)
    for rec in records:
        groups[(rec.n_sources, rec.n_microphones, rec.geometry)].append(rec)
    rng = np.random.default_rng(20261108)
    chosen = []
    for key in sorted(groups):
        bins = defaultdict(list)
        for rec in groups[key]:
            si = min(3, max(0, int((rec.snr_requested_db - 10.0) // 5.0)))
            ti = min(3, max(0, int((rec.rt60_requested - 0.2) // 0.15)))
            bins[(si, ti)].append(rec)
        for rows in bins.values():
            rng.shuffle(rows)
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
            selected.extend(rest[:20 - len(selected)])
        chosen.extend(selected)
    if len(chosen) != 360 or len({r.scene_id for r in chosen}) != 360:
        raise RuntimeError("triagem Round-4 não foi reproduzida")
    return chosen


def rank_map(values):
    a = np.asarray(values, dtype=np.float32)
    return (rankdata(a.ravel(), method="average").reshape(a.shape) / a.size).astype(np.float32)


def peaks(scores, xs, ys, n, radius, local=True):
    arr = np.asarray(scores, np.float32)
    if local:
        mask = arr >= maximum_filter(arr, size=3, mode="nearest")
        inds = np.flatnonzero(mask.ravel())
        order = inds[np.argsort(arr.ravel()[inds])[::-1]]
    else:
        order = np.argsort(arr.ravel())[::-1]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    selected = []
    for i in order:
        p = pts[i]
        if all(np.linalg.norm(p - q) >= radius for q in selected):
            selected.append(p)
            if len(selected) == n:
                return np.asarray(selected)
    for i in np.argsort(arr.ravel())[::-1]:
        p = pts[i]
        if all(not np.array_equal(p, q) for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    return np.asarray(selected)


def local_candidates(score, xs, ys, k=32, separation_m=.10):
    local = score >= maximum_filter(score, size=3, mode="nearest")
    inds = np.flatnonzero(local.ravel())
    inds = inds[np.argsort(score.ravel()[inds])[::-1]]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    selected = []
    for idx in inds:
        point = grid[idx]
        if all(np.linalg.norm(point - q) >= separation_m for q in selected):
            selected.append(point)
            if len(selected) == k:
                break
    if len(selected) < k:
        for idx in np.argsort(score.ravel())[::-1]:
            point = grid[idx]
            if all(np.linalg.norm(point - q) >= separation_m for q in selected):
                selected.append(point)
                if len(selected) == k:
                    break
    return np.asarray(selected, dtype=np.float32)


def metric_summary(errors, records=None):
    e = np.concatenate(errors) if errors else np.array([], dtype=float)
    return {"scene_count": len(errors), "source_count": int(e.size),
            "mean_error_m": float(e.mean()), "median_error_m": float(np.median(e)),
            "p90_error_m": float(np.percentile(e, 90)),
            "within_lt_0_1m_count": int(np.sum(e < .1)), "within_lt_0_1m_rate": float(np.mean(e < .1)),
            "within_0_25m": float(np.mean(e <= .25)), "within_0_5m": float(np.mean(e <= .5)),
            "over_1m_count": int(np.sum(e > 1)), "over_1m_rate": float(np.mean(e > 1)),
            "over_2m_count": int(np.sum(e > 2)), "over_2m_rate": float(np.mean(e > 2)),
            "all_sources_lt_0_1m_scenes": int(sum(bool(np.all(x < .1)) for x in errors)),
            "all_sources_le_0_5m_scenes": int(sum(bool(np.all(x <= .5)) for x in errors))}


def save_source_errors(path, records, estimates_by_method):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scene_id", "source_index", "reference_x_m", "reference_y_m",
                                          "estimate_x_m", "estimate_y_m", "error_m"])
        w.writeheader()
        for method, estimates in estimates_by_method.items():
            for rec, est in zip(records, estimates):
                ref = np.asarray(rec.source_positions_xy, dtype=np.float64)
                dist = np.linalg.norm(ref[:, None, :] - np.asarray(est)[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(dist)
                for ri, ci in zip(rows, cols):
                    w.writerow({"method": method, "scene_id": rec.scene_id, "source_index": int(ri),
                                "reference_x_m": f"{ref[ri,0]:.6f}", "reference_y_m": f"{ref[ri,1]:.6f}",
                                "estimate_x_m": f"{est[ci,0]:.6f}", "estimate_y_m": f"{est[ci,1]:.6f}",
                                "error_m": f"{dist[ri,ci]:.8f}"})


def paired_scene_bootstrap_variable_n(candidate_errors, baseline_errors, seed=20261108, repeats=10000):
    rng = np.random.default_rng(seed)
    sums = np.asarray([np.sum(np.asarray(a) - np.asarray(b))
                       for a, b in zip(candidate_errors, baseline_errors)], dtype=np.float64)
    counts = np.asarray([len(a) for a in candidate_errors], dtype=np.int32)
    draws = np.empty(repeats)
    for start in range(0, repeats, 500):
        end = min(repeats, start + 500)
        ix = rng.integers(0, len(sums), size=(end - start, len(sums)))
        draws[start:end] = sums[ix].sum(axis=1) / counts[ix].sum(axis=1)
    return {"mean_delta_m": float(sums.sum() / counts.sum()),
            "ci95_m": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "repeats": repeats, "unit": "scene cluster; source-weighted aggregate",
            "negative_favors_candidate": True}


def read_audio(rec):
    audio, sr = sf.read(rec.audio_path, dtype="float32", always_2d=True)
    if sr != FS:
        raise ValueError(f"sample rate inesperado em {rec.scene_id}: {sr}")
    return audio.T


def log(event: str, **kw):
    print(json.dumps({"event": event, "t": time.strftime("%H:%M:%S"), **kw}), flush=True)


def load_triage():
    records = list(read_jsonl(VAL_PATH, SceneRecord))
    return triage_subset(records)


def geometry_metrics(rec: SceneRecord):
    mics = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
    sources = np.asarray(rec.source_positions_xy, dtype=np.float64)
    centered = mics - mics.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    array_condition = float(singular[0] / max(singular[-1], 1e-9))
    pair_lengths = np.asarray([np.linalg.norm(mics[i] - mics[j])
                               for i, j in pair_indices(rec.n_microphones)])
    source_gdop, source_cond = [], []
    for src in sources:
        vectors = src[None, :] - mics
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        # Independent reference-pair Jacobian for range-difference measurements.
        jac = unit[1:] - unit[0]
        fim = jac.T @ jac
        eig = np.linalg.eigvalsh(fim)
        source_gdop.append(float(np.sqrt(np.trace(np.linalg.pinv(fim)))) if eig.min() > 1e-10 else float("inf"))
        source_cond.append(float(eig.max() / max(eig.min(), 1e-10)))
    return {
        "array_condition": array_condition,
        "pair_min_m": float(pair_lengths.min()), "pair_median_m": float(np.median(pair_lengths)),
        "pair_max_m": float(pair_lengths.max()),
        "gdop_proxy_mean": float(np.mean(source_gdop)), "gdop_proxy_max": float(np.max(source_gdop)),
        "tdoa_condition_mean": float(np.mean(source_cond)),
    }


def stage_audit():
    records = load_triage()
    rows = []
    start = time.perf_counter()
    expected_sr = 16_000
    for idx, rec in enumerate(records):
        mixed, sr = sf.read(rec.audio_path, dtype="float32", always_2d=True)
        clean, clean_sr = sf.read(rec.clean_audio_path, dtype="float32", always_2d=True)
        mixed_finite = bool(np.isfinite(mixed).all())
        clean_finite = bool(np.isfinite(clean).all())
        same_shape = mixed.shape == clean.shape
        if same_shape:
            noise = mixed.astype(np.float64) - clean.astype(np.float64)
            p_clean = float(np.mean(clean.astype(np.float64) ** 2))
            p_noise = float(np.mean(noise ** 2))
            measured_snr = float(10 * np.log10(max(p_clean, 1e-20) / max(p_noise, 1e-20)))
            snr_delta = measured_snr - float(rec.snr_realized_db)
        else:
            measured_snr, snr_delta = float("nan"), float("nan")
        channel_rms = np.sqrt(np.mean(mixed.astype(np.float64) ** 2, axis=0))
        peak = float(np.max(np.abs(mixed)))
        corr = np.corrcoef(mixed.T) if mixed.shape[1] > 1 else np.ones((1, 1))
        g = geometry_metrics(rec)
        rows.append({
            "scene_id": rec.scene_id, "n_sources": rec.n_sources, "n_microphones": rec.n_microphones,
            "geometry": rec.geometry, "sample_rate": int(sr), "clean_sample_rate": int(clean_sr),
            "wav_subtype": sf.info(rec.audio_path).subtype,
            "samples": int(mixed.shape[0]), "audio_channels": int(mixed.shape[1]),
            "mixed_clean_same_shape": bool(same_shape), "finite": mixed_finite and clean_finite,
            "peak_abs": peak, "samples_abs_ge_0_999_frac": float(np.mean(np.abs(mixed) >= .999)),
            "samples_abs_gt_1_frac": float(np.mean(np.abs(mixed) > 1.0)),
            "rms_min_channel": float(channel_rms.min()), "rms_max_channel": float(channel_rms.max()),
            "channel_rms_ratio": float(channel_rms.max() / max(channel_rms.min(), 1e-12)),
            "zero_lag_channel_corr_median": float(np.median(corr[np.triu_indices(len(corr), 1)])),
            "snr_manifest_db": rec.snr_realized_db, "snr_measured_db": measured_snr,
            "snr_abs_delta_db": abs(snr_delta), **g,
        })
        if (idx + 1) % 60 == 0 or idx + 1 == len(records):
            log("audit", scenes=idx + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "channel_geometry_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    errors_by_method = defaultdict(list)
    error_path = ROOT / "artifacts" / "round4" / "triage_source_errors.csv"
    if error_path.exists():
        with error_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                errors_by_method[(r["method"], r["scene_id"])].append(float(r["error_m"]))
    baseline_errors = {sid: float(np.mean(v)) for (method, sid), v in errors_by_method.items()
                       if method == "round2_seed42"}
    valid_rows = [r for r in rows if r["scene_id"] in baseline_errors]
    by_m = {}
    for m in (4, 6, 8):
        group = [r for r in valid_rows if r["n_microphones"] == m]
        if group:
            by_m[str(m)] = {"scenes": len(group), "sources": sum(r["n_sources"] for r in group),
                            "snr_abs_delta_median_db": float(np.median([r["snr_abs_delta_db"] for r in group])),
                            "samples_abs_ge_0_999_frac_max": float(max(r["samples_abs_ge_0_999_frac"] for r in group)),
                            "gdop_proxy_median": float(np.median([r["gdop_proxy_mean"] for r in group])),
                            "baseline_scene_mean_error_m": float(np.mean([baseline_errors[r["scene_id"]] for r in group]))}
            errs = np.asarray([baseline_errors[r["scene_id"]] for r in group])
            gdops = np.asarray([r["gdop_proxy_mean"] for r in group])
            by_m[str(m)]["within_group_gdop_error_spearman"] = float(spearmanr(gdops, errs).statistic)
    summary = {
        "triage_scenes": len(rows), "valid_all_finite": bool(all(r["finite"] for r in rows)),
        "channel_count_matches_manifest": bool(all(r["audio_channels"] == r["n_microphones"] for r in rows)),
        "sample_rate_matches_16k": bool(all(r["sample_rate"] == expected_sr and r["clean_sample_rate"] == expected_sr for r in rows)),
        "sample_counts": sorted({r["samples"] for r in rows}),
        "wav_subtypes": sorted({r["wav_subtype"] for r in rows}),
        "samples_abs_ge_0_999_frac_max": float(max(r["samples_abs_ge_0_999_frac"] for r in rows)),
        "samples_abs_gt_1_frac_max": float(max(r["samples_abs_gt_1_frac"] for r in rows)),
        "snr_abs_delta_median_db": float(np.median([r["snr_abs_delta_db"] for r in rows])),
        "snr_abs_delta_p99_db": float(np.percentile([r["snr_abs_delta_db"] for r in rows], 99)),
        "channel_rms_ratio_median": float(np.median([r["channel_rms_ratio"] for r in rows])),
        "zero_lag_channel_corr_median": float(np.median([r["zero_lag_channel_corr_median"] for r in rows])),
        "synchronization_note": "As cenas são simuladas e salvas como canais de um único WAV multicanal; isso verifica alinhamento no gerador, não calibra relógios de hardware reais.",
        "geometry_note": "GDOP é apenas proxy local 2D para range-difference, não limite CRLB completo; multipath e fontes simultâneas não estão incluídos.",
        "by_microphone_count": by_m,
        "seconds": time.perf_counter() - start,
    }
    (OUT / "channel_geometry_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("audit_result", **summary)


def matched_candidate_errors(ref: np.ndarray, candidates: np.ndarray):
    ref = np.asarray(ref, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    dist = np.linalg.norm(ref[:, None, :] - candidates[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    out = np.full(len(ref), np.inf, dtype=np.float64)
    out[rows] = dist[rows, cols]
    return out


def stage_oracle():
    records = load_triage()
    base = np.load(BASE_CACHE, allow_pickle=True)
    ids = base["ids"]
    ix = {str(s): i for i, s in enumerate(ids)}
    xs, ys = base["xs"], base["ys"]
    methods = ("srp", "cnn42", "round2_fusion_seed42")
    ks = (1, 3, 5, 10, 20, 50)
    radii = (.10, .25, .50, 1.0)
    source_acc = {(m, k, r): [] for m in methods for k in ks for r in radii}
    scene_acc = {(m, k, r): [] for m in methods for k in ks for r in radii}
    start = time.perf_counter()
    for si, rec in enumerate(records):
        ci = ix[rec.scene_id]
        srp, cnn = base["srp"][ci], base["cnn42"][ci]
        maps = {"srp": srp, "cnn42": cnn,
                "round2_fusion_seed42": .75 * rank_map(srp) + .25 * cnn}
        for method, score in maps.items():
            candidates = local_candidates(score, xs, ys, k=max(ks), separation_m=.10)
            ref = np.asarray(rec.source_positions_xy, dtype=np.float64)
            for k in ks:
                e = matched_candidate_errors(ref, candidates[:k])
                for radius in radii:
                    hit = e <= radius
                    source_acc[(method, k, radius)].extend(hit.tolist())
                    scene_acc[(method, k, radius)].append(bool(np.all(hit)))
        if (si + 1) % 60 == 0 or si + 1 == len(records):
            log("oracle", scenes=si + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))
    rows = []
    for method in methods:
        for k in ks:
            for radius in radii:
                src = source_acc[(method, k, radius)]
                sc = scene_acc[(method, k, radius)]
                rows.append({"map": method, "top_k_candidates": k, "radius_m": radius,
                             "source_coverage": float(np.mean(src)), "sources_covered": int(np.sum(src)),
                             "source_count": len(src), "all_sources_scene_coverage": float(np.mean(sc)),
                             "scenes_all_sources_covered": int(np.sum(sc)), "scene_count": len(sc)})
    with (OUT / "oracle_candidate_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "oracle_candidate_coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Compare the actual NMS selection to the oracle bound from the same map.
    base_errors = []
    for rec in records:
        ci = ix[rec.scene_id]
        score = .75 * rank_map(base["srp"][ci]) + .25 * base["cnn42"][ci]
        est = peaks(score, xs, ys, rec.n_sources, .60, True)
        base_errors.extend(pairwise_assignment_errors(np.asarray(rec.source_positions_xy), est).tolist())
    summary = {"triage_scenes": len(records), "source_count": len(base_errors),
               "selected_baseline_mean_error_m": float(np.mean(base_errors)),
               "selected_baseline_p90_error_m": float(np.percentile(base_errors, 90)),
               "seconds": time.perf_counter() - start,
               "warning": "Oracle candidate coverage is a diagnostic upper bound on the fixed pool, not deployable performance; truth is used only for analysis."}
    (OUT / "oracle_candidate_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("oracle_result", **summary)


def pair_contributions(corr: np.ndarray, pairs, mics, points):
    lag_axis = np.arange(-(FFT // 2), FFT // 2, dtype=np.float64)
    contributions = np.empty((len(pairs), len(points)), dtype=np.float32)
    p3 = np.column_stack((points, np.full(len(points), 1.5)))
    for pi, (i, j) in enumerate(pairs):
        expected = (np.linalg.norm(p3 - mics[i], axis=1) - np.linalg.norm(p3 - mics[j], axis=1)) * FS / C
        contributions[pi] = np.interp(expected, lag_axis, corr[pi], left=0.0, right=0.0)
    return contributions


def robust_maps(contrib):
    med = np.median(contrib, axis=0)
    mad = np.median(np.abs(contrib - med[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-5)
    huber_raw = np.clip(contrib, med[None, :] - 1.5 * scale[None, :],
                        med[None, :] + 1.5 * scale[None, :]).mean(axis=0)
    pair_ranks = np.stack([rankdata(row, method="average") / len(row) for row in contrib]).astype(np.float32)
    median_rank = np.median(pair_ranks, axis=0)
    medrank = np.median(pair_ranks, axis=0)
    scale_rank = np.maximum(1.4826 * np.median(np.abs(pair_ranks - medrank[None, :]), axis=0), 1e-5)
    huber_rank = np.clip(pair_ranks, medrank[None, :] - 1.5 * scale_rank[None, :],
                         medrank[None, :] + 1.5 * scale_rank[None, :]).mean(axis=0)
    geometric_rank = np.exp(np.mean(np.log(np.maximum(pair_ranks, 1e-3)), axis=0))
    return {"robust_huber_raw": huber_raw, "robust_median_pair_rank": median_rank,
            "robust_huber_pair_rank": huber_rank,
            "robust_geometric_pair_rank": geometric_rank}


def stage_robust():
    records = load_triage()
    base = np.load(BASE_CACHE, allow_pickle=True)
    base_ix = {str(s): i for i, s in enumerate(base["ids"])}
    pair_cache = np.load(PAIR_CACHE, allow_pickle=False)
    pair_ix = {str(s): i for i, s in enumerate(pair_cache["ids"])}
    offsets, corr_all = pair_cache["offsets"], pair_cache["corr"]
    xs, ys = base["xs"], base["ys"]
    grid = make_grid((8.0, 6.0, 3.0), .10)
    methods = defaultdict(list)
    start = time.perf_counter()
    reconstruction_abs_max = 0.0
    reconstruction_abs_mean = 0.0
    for si, rec in enumerate(records):
        bi = base_ix[rec.scene_id]
        pi = pair_ix[rec.scene_id]
        a, b = int(offsets[pi]), int(offsets[pi + 1])
        pairs = pair_indices(rec.n_microphones)
        mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
        mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5)))
        contrib = pair_contributions(corr_all[a:b], pairs, mics, grid.points_xy)
        maps = robust_maps(contrib)
        cnn, srp = base["cnn42"][bi], base["srp"][bi]
        reconstruction_diff = np.abs(contrib.mean(axis=0).reshape(len(ys), len(xs)) - srp)
        if si == 0:
            reconstruction_abs_max = float(reconstruction_diff.max())
            reconstruction_abs_mean = float(reconstruction_diff.mean())
        baseline_score = .75 * rank_map(srp) + .25 * cnn
        methods["round2_seed42"].append(peaks(baseline_score, xs, ys, rec.n_sources, .60, True))
        for name, flat in maps.items():
            score = .75 * rank_map(flat.reshape(len(ys), len(xs))) + .25 * cnn
            methods[name].append(peaks(score, xs, ys, rec.n_sources, .60, True))
        if (si + 1) % 60 == 0 or si + 1 == len(records):
            log("robust", scenes=si + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), pred)
                     for r, pred in zip(records, estimates)] for name, estimates in methods.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in errors.items()]
    (OUT / "robust_gcc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "robust_gcc_source_errors.csv", records, dict(methods))
    baseline = errors["round2_seed42"]
    paired = {name: paired_scene_bootstrap_variable_n(errs, baseline, seed=20261108, repeats=10000)
              for name, errs in errors.items() if name != "round2_seed42"}
    (OUT / "robust_gcc_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    timing = {"seconds": time.perf_counter() - start, "scene_count": len(records),
              "methods": list(methods), "screen_only": True,
              "first_scene_srp_reconstruction_max_abs": reconstruction_abs_max,
              "first_scene_srp_reconstruction_mean_abs": reconstruction_abs_mean}
    (OUT / "timing_robust_gcc.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log("robust_result", metrics=metrics, paired=paired, **timing)


def blockwise_gcc_maps(audio, rec, grid, block_count=8):
    audio = np.asarray(audio, dtype=np.float32)
    frames = [frame_signal(ch, WINDOW, HOP) for ch in audio]
    win = np.hanning(WINDOW).astype(np.float32)
    spectra = np.stack([np.fft.rfft(f * win[None, :], n=FFT, axis=1) for f in frames])
    pairs = pair_indices(rec.n_microphones)
    pair_i = np.asarray([i for i, _ in pairs], dtype=np.int64)
    pair_j = np.asarray([j for _, j in pairs], dtype=np.int64)
    n_samples = audio.shape[1]
    centers = np.arange(spectra.shape[1]) * HOP + WINDOW / 2
    labels = np.minimum((centers / n_samples * block_count).astype(int), block_count - 1)
    mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
    mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5)))
    maps = []
    lag_axis = np.arange(-(FFT // 2), FFT // 2, dtype=np.float64)
    p3 = np.column_stack((grid.points_xy, np.full(len(grid.points_xy), 1.5)))
    for block in range(block_count):
        selected = spectra[:, labels == block]
        cross = selected[pair_i] * np.conj(selected[pair_j])
        cross /= np.maximum(np.abs(cross), 1e-8)
        corr = np.fft.fftshift(np.fft.irfft(np.mean(cross, axis=1), n=FFT, axis=1), axes=1).astype(np.float32)
        scores = np.zeros(len(grid.points_xy), dtype=np.float32)
        for pi, (i, j) in enumerate(pairs):
            lag = (np.linalg.norm(p3 - mics[i], axis=1) - np.linalg.norm(p3 - mics[j], axis=1)) * FS / C
            scores += np.interp(lag, lag_axis, corr[pi], left=0.0, right=0.0).astype(np.float32)
        scores /= max(len(pairs), 1)
        maps.append(scores.reshape(len(grid.ys), len(grid.xs)))
    return np.stack(maps)


def total_pairwise_block_gcc(audio, rec, block_count=4):
    """Four-block PHAT spectra reduced to one GCC per microphone pair."""
    audio = np.asarray(audio, dtype=np.float32)
    frames = [frame_signal(ch, WINDOW, HOP) for ch in audio]
    win = np.hanning(WINDOW).astype(np.float32)
    spectra = np.stack([np.fft.rfft(f * win[None, :], n=FFT, axis=1) for f in frames])
    pairs = pair_indices(rec.n_microphones)
    pair_i = np.asarray([i for i, _ in pairs], dtype=np.int64)
    pair_j = np.asarray([j for _, j in pairs], dtype=np.int64)
    centers = np.arange(spectra.shape[1]) * HOP + WINDOW / 2
    labels = np.minimum((centers / audio.shape[1] * block_count).astype(int), block_count - 1)
    block_corr, block_counts = [], []
    for b in range(block_count):
        selected = spectra[:, labels == b]
        block_counts.append(selected.shape[1])
        cross = selected[pair_i] * np.conj(selected[pair_j])
        cross /= np.maximum(np.abs(cross), 1e-8)
        corr = np.fft.irfft(np.mean(cross, axis=1), n=FFT, axis=1)
        block_corr.append(np.fft.fftshift(corr, axes=1).astype(np.float32))
    total = np.average(np.stack(block_corr), axis=0, weights=np.asarray(block_counts))
    return total.astype(np.float32), pairs


def pair_peak_lists(corr, pairs, mics, k=5):
    lag_axis = np.arange(-(FFT // 2), FFT // 2)
    all_lags, all_values = [], []
    for pi, (i, j) in enumerate(pairs):
        maxlag = int(np.ceil(np.linalg.norm(mics[i] - mics[j]) * FS / C)) + 2
        c = np.maximum(corr[pi], 0.0)
        mask = np.abs(lag_axis) <= maxlag
        local = (c >= maximum_filter(c, size=5, mode="nearest")) & mask
        inds = np.flatnonzero(local)
        inds = inds[np.argsort(c[inds])[::-1]]
        selected = []
        for idx in inds:
            if all(abs(lag_axis[idx] - lag_axis[jj]) >= 4 for jj in selected):
                selected.append(int(idx))
            if len(selected) >= k:
                break
        if len(selected) < k:
            fallback = np.flatnonzero(mask)
            fallback = fallback[np.argsort(c[fallback])[::-1]]
            for idx in fallback:
                if all(abs(lag_axis[idx] - lag_axis[jj]) >= 4 for jj in selected):
                    selected.append(int(idx))
                if len(selected) >= k:
                    break
        lags = lag_axis[selected].astype(np.float32)
        vals = c[selected]
        vals = vals / (float(vals.max()) + 1e-8)
        all_lags.append(lags); all_values.append(vals.astype(np.float32))
    return np.asarray(all_lags), np.asarray(all_values)


def joint_tdoa_decode(rec, candidate_map, corr, xs, ys, k_candidates=32, k_peaks=5, sigma_samples=2.5):
    candidates = local_candidates(candidate_map, xs, ys, k_candidates, separation_m=.10)
    n = rec.n_sources
    combinations = np.asarray(list(__import__("itertools").combinations(range(len(candidates)), n)), dtype=np.int32)
    assignments = np.asarray(list(__import__("itertools").permutations(range(k_peaks), n)), dtype=np.int32)
    xy = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
    mics = np.column_stack((xy, np.full(len(xy), 1.5)))
    points = np.column_stack((candidates, np.full(len(candidates), 1.5)))
    pairs = pair_indices(rec.n_microphones)
    peak_lags, peak_values = pair_peak_lists(corr, pairs, mics, k_peaks)
    set_scores = np.zeros(len(combinations), dtype=np.float64)
    for pi, (i, j) in enumerate(pairs):
        expected = (np.linalg.norm(points - mics[i], axis=1) - np.linalg.norm(points - mics[j], axis=1)) * FS / C
        quality = peak_values[pi][None, :] * np.exp(-.5 * ((expected[:, None] - peak_lags[pi][None, :]) /
                                                           sigma_samples) ** 2)
        best = np.zeros(len(combinations), dtype=np.float64)
        for assignment in assignments:
            score = np.zeros(len(combinations), dtype=np.float64)
            for source_idx, peak_idx in enumerate(assignment):
                score += quality[combinations[:, source_idx], peak_idx]
            best = np.maximum(best, score)
        set_scores += best
    return candidates[combinations[int(np.argmax(set_scores))]]


def stage_multisource():
    records = load_triage()
    base = np.load(BASE_CACHE, allow_pickle=True)
    base_ix = {str(s): i for i, s in enumerate(base["ids"])}
    xs, ys = base["xs"], base["ys"]
    grid = make_grid((8.0, 6.0, 3.0), .10)
    methods = defaultdict(list)
    start = time.perf_counter()
    for si, rec in enumerate(records):
        bi = base_ix[rec.scene_id]
        cnn, srp = base["cnn42"][bi], base["srp"][bi]
        base_score = .75 * rank_map(srp) + .25 * cnn
        methods["round2_seed42"].append(peaks(base_score, xs, ys, rec.n_sources, .60, True))
        audio = read_audio(rec)
        block_maps = blockwise_gcc_maps(audio, rec, grid, block_count=8)
        ranked = np.stack([rank_map(m) for m in block_maps])
        mean_all = ranked.mean(axis=0)
        top4 = np.partition(ranked, 4, axis=0)[4:].mean(axis=0)
        top2 = np.partition(ranked, 6, axis=0)[6:].mean(axis=0)
        # A multi-source persistence score: the best-supported half/quarter of
        # time blocks vote for each static source, rather than averaging away
        # source-specific activity across the whole clip.
        for name, agg in (("temporal_mean8", mean_all), ("temporal_top4of8", top4),
                          ("temporal_top2of8", top2)):
            score = .75 * rank_map(agg) + .25 * cnn
            methods[name].append(peaks(score, xs, ys, rec.n_sources, .60, True))
        if (si + 1) % 30 == 0 or si + 1 == len(records):
            log("multisource", scenes=si + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), pred)
                     for r, pred in zip(records, estimates)] for name, estimates in methods.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in errors.items()]
    (OUT / "multisource_temporal_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "multisource_temporal_source_errors.csv", records, dict(methods))
    stratified = []
    for n in (1, 2, 3):
        ix = [i for i, r in enumerate(records) if r.n_sources == n]
        for name, err in errors.items():
            vals = np.concatenate([err[i] for i in ix])
            stratified.append({"n_sources": n, "method": name, "source_count": int(len(vals)),
                               "mean_error_m": float(vals.mean()), "p90_error_m": float(np.percentile(vals, 90)),
                               "over_1m_rate": float(np.mean(vals > 1))})
    with (OUT / "multisource_temporal_by_n.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stratified[0])); w.writeheader(); w.writerows(stratified)
    baseline = errors["round2_seed42"]
    paired = {name: paired_scene_bootstrap_variable_n(err, baseline, seed=20261108, repeats=10000)
              for name, err in errors.items() if name != "round2_seed42"}
    (OUT / "multisource_temporal_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    timing = {"seconds": time.perf_counter() - start, "scene_count": len(records), "blocks": 8,
              "methods": list(methods), "screen_only": True}
    (OUT / "timing_multisource_temporal.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log("multisource_result", metrics=metrics, paired=paired, **timing)


def stage_multisource_full():
    """Exploratory full-validation evaluation of the single temporal finalist."""
    records = list(read_jsonl(VAL_PATH, SceneRecord))
    base = np.load(BASE_CACHE, allow_pickle=True)
    base_ix = {str(s): i for i, s in enumerate(base["ids"])}
    xs, ys = base["xs"], base["ys"]
    grid = make_grid((8.0, 6.0, 3.0), .10)
    protocol = {
        "status": "frozen_before_full_validation_inference",
        "method": "8 temporal blocks of 250ms; per-point mean of top 2 blockwise pair-rank SRP maps; 0.75 rank(SRP)+0.25 CNN42; NMS 0.60m",
        "selection_source": "360-scene fixed validation triage; full validation is exploratory and contains the triage scenes",
        "independent_test_used": False,
        "validation_scenes": len(records), "sample_rate_hz": FS, "window": WINDOW,
        "hop": HOP, "fft_size": FFT, "blocks": 8, "blocks_aggregated": 2,
        "grid_resolution_m": .10, "speed_of_sound_m_s": C,
    }
    (OUT / "multisource_temporal_full_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    baseline_est, candidate_est = [], []
    start = time.perf_counter()
    progress = OUT / "multisource_temporal_full_progress.json"
    for si, rec in enumerate(records):
        bi = base_ix[rec.scene_id]
        cnn, srp = base["cnn42"][bi], base["srp"][bi]
        base_score = .75 * rank_map(srp) + .25 * cnn
        baseline_est.append(peaks(base_score, xs, ys, rec.n_sources, .60, True))
        block_maps = blockwise_gcc_maps(read_audio(rec), rec, grid, block_count=8)
        ranked = np.stack([rank_map(m) for m in block_maps])
        top2 = np.partition(ranked, 6, axis=0)[6:].mean(axis=0)
        candidate_score = .75 * rank_map(top2) + .25 * cnn
        candidate_est.append(peaks(candidate_score, xs, ys, rec.n_sources, .60, True))
        if (si + 1) % 100 == 0 or si + 1 == len(records):
            progress.write_text(json.dumps({
                "done_scenes": si + 1,
                "baseline": [x.tolist() for x in baseline_est],
                "temporal_top2of8": [x.tolist() for x in candidate_est],
            }), encoding="utf-8")
            log("multisource_full", scenes=si + 1, total=len(records),
                elapsed_s=round(time.perf_counter() - start, 1))
    estimates = {"round2_seed42": baseline_est, "temporal_top2of8": candidate_est}
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), p)
                     for r, p in zip(records, rows)] for name, rows in estimates.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in errors.items()]
    (OUT / "multisource_temporal_full_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "multisource_temporal_full_source_errors.csv", records, estimates)
    paired = {"temporal_top2of8": paired_scene_bootstrap_variable_n(
        errors["temporal_top2of8"], errors["round2_seed42"], seed=20261108, repeats=10000)}
    (OUT / "multisource_temporal_full_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    by_stratum = []
    for key_name, group_func in (
        ("n_sources", lambda r: r.n_sources),
        ("n_microphones", lambda r: r.n_microphones),
        ("geometry", lambda r: r.geometry),
    ):
        groups = defaultdict(list)
        for i, rec in enumerate(records):
            groups[group_func(rec)].append(i)
        for group_value, indices in groups.items():
            for method, per_scene in errors.items():
                values = np.concatenate([per_scene[i] for i in indices])
                by_stratum.append({"stratum": key_name, "value": group_value, "method": method,
                                   "scene_count": len(indices), "source_count": int(len(values)),
                                   "mean_error_m": float(values.mean()),
                                   "p90_error_m": float(np.percentile(values, 90)),
                                   "over_1m_rate": float(np.mean(values > 1))})
    with (OUT / "multisource_temporal_full_stratified.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_stratum[0])); w.writeheader(); w.writerows(by_stratum)
    gate = {"mean_delta_m": metrics[1]["mean_error_m"] - metrics[0]["mean_error_m"],
            "p90_improvement_m": metrics[0]["p90_error_m"] - metrics[1]["p90_error_m"],
            "over_1m_rate_delta": metrics[1]["over_1m_rate"] - metrics[0]["over_1m_rate"],
            "within_0_5m_delta": metrics[1]["within_0_5m"] - metrics[0]["within_0_5m"]}
    (OUT / "multisource_temporal_full_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    timing = {"seconds": time.perf_counter() - start, "scene_count": len(records), "blocks": 8}
    (OUT / "timing_multisource_temporal_full.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log("multisource_full_result", metrics=metrics, paired=paired, gate=gate, **timing)


def stage_joint_full():
    """Full exploratory validation of the existing joint-TDOA set decoder."""
    records = list(read_jsonl(VAL_PATH, SceneRecord))
    base = np.load(BASE_CACHE, allow_pickle=True)
    base_ix = {str(s): i for i, s in enumerate(base["ids"])}
    xs, ys = base["xs"], base["ys"]
    protocol = {
        "status": "frozen_before_full_validation_inference",
        "method": "joint per-pair top-5 GCC-PHAT lag assignments over top-32 SRP+CNN candidate pool; sigma=2.5 samples; N known",
        "selection_history": "This exact decoder showed mean/P90 improvement on the fixed 360-scene triage but increased >1m errors; full validation is exploratory.",
        "independent_test_used": False, "validation_scenes": len(records),
        "grid_resolution_m": .10, "nms_radius_m": .60, "pair_peak_separation_samples": 4,
    }
    (OUT / "joint_tdoa_full_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    baseline_est, candidate_est = [], []
    start = time.perf_counter()
    progress = OUT / "joint_tdoa_full_progress.json"
    for si, rec in enumerate(records):
        bi = base_ix[rec.scene_id]
        base_score = .75 * rank_map(base["srp"][bi]) + .25 * base["cnn42"][bi]
        baseline_est.append(peaks(base_score, xs, ys, rec.n_sources, .60, True))
        corr, pairs = total_pairwise_block_gcc(read_audio(rec), rec, block_count=4)
        candidate_est.append(joint_tdoa_decode(rec, base_score, corr, xs, ys,
                                               k_candidates=32, k_peaks=5, sigma_samples=2.5))
        if (si + 1) % 100 == 0 or si + 1 == len(records):
            progress.write_text(json.dumps({"done_scenes": si + 1,
                                            "baseline": [x.tolist() for x in baseline_est],
                                            "joint_tdoa": [x.tolist() for x in candidate_est]}), encoding="utf-8")
            log("joint_full", scenes=si + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))
    estimates = {"round2_seed42": baseline_est, "joint_tdoa_from_baseline_pool": candidate_est}
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), p)
                     for r, p in zip(records, rows)] for name, rows in estimates.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in errors.items()]
    (OUT / "joint_tdoa_full_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "joint_tdoa_full_source_errors.csv", records, estimates)
    paired = {"joint_tdoa_from_baseline_pool": paired_scene_bootstrap_variable_n(
        errors["joint_tdoa_from_baseline_pool"], errors["round2_seed42"], seed=20261108, repeats=10000)}
    (OUT / "joint_tdoa_full_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    gate = {"mean_delta_m": metrics[1]["mean_error_m"] - metrics[0]["mean_error_m"],
            "p90_improvement_m": metrics[0]["p90_error_m"] - metrics[1]["p90_error_m"],
            "over_1m_rate_delta": metrics[1]["over_1m_rate"] - metrics[0]["over_1m_rate"],
            "within_0_5m_delta": metrics[1]["within_0_5m"] - metrics[0]["within_0_5m"]}
    (OUT / "joint_tdoa_full_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    timing = {"seconds": time.perf_counter() - start, "scene_count": len(records)}
    (OUT / "timing_joint_tdoa_full.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log("joint_full_result", metrics=metrics, paired=paired, gate=gate, **timing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("audit", "oracle", "robust", "multisource", "multisource-full", "joint-full"))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"audit": stage_audit, "oracle": stage_oracle, "robust": stage_robust,
     "multisource": stage_multisource, "multisource-full": stage_multisource_full,
     "joint-full": stage_joint_full}[args.stage]()


if __name__ == "__main__":
    main()
