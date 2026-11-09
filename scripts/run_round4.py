"""Budgeted Round 4 experiments: pairwise direct-path evidence and reflections.

All selection is confined to the existing validation split. The Round-3
independent test is intentionally not read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from voice_localization.manifest import SceneRecord, read_jsonl
from voice_localization.metrics import pairwise_assignment_errors
from voice_localization.spatial import frame_signal, make_grid, pair_indices, score_srp_points
from run_round3 import (balanced_subset, metric_summary, rank_map, save_source_errors,
                       stratified_metrics, triage_subset)
from run_round2 import peaks


OUT = ROOT / "artifacts" / "round4"
VAL_PATH = ROOT / "data" / "full" / "manifests" / "scenes_validation.jsonl"
TRAIN_PATH = ROOT / "data" / "full" / "manifests" / "scenes_train.jsonl"
CACHE_PATH = ROOT / "artifacts" / "round3" / "full_baseline_maps.npz"
FS = 16_000
C = 343.0
WINDOW = 512
HOP = 256
FFT = 1024
BLOCKS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)


def log(event: str, **values) -> None:
    print(json.dumps({"event": event, "t": time.strftime("%H:%M:%S"), **values}), flush=True)


def read_audio(rec: SceneRecord) -> np.ndarray:
    audio, sample_rate = sf.read(rec.audio_path, dtype="float32", always_2d=True)
    if sample_rate != FS:
        raise ValueError(f"sample rate inesperado em {rec.scene_id}: {sample_rate}")
    return audio.T


def pairwise_block_gcc(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Four time-block GCC-PHAT maps, pair axis first; returns raw GCC mean too."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("audio deve ter forma (M,T)")
    n_mics, n_samples = audio.shape
    frames = [frame_signal(ch, WINDOW, HOP) for ch in audio]
    win = np.hanning(WINDOW).astype(np.float32)
    spectra = np.stack([np.fft.rfft(f * win[None, :], n=FFT, axis=1) for f in frames])
    pair_list = pair_indices(n_mics)
    pair_i = np.asarray([p[0] for p in pair_list], dtype=np.int64)
    pair_j = np.asarray([p[1] for p in pair_list], dtype=np.int64)
    frame_centers = np.arange(spectra.shape[1]) * HOP + WINDOW / 2
    frame_blocks = np.minimum((frame_centers / n_samples * BLOCKS).astype(int), BLOCKS - 1)
    block_corrs = []
    block_counts = []
    for block in range(BLOCKS):
        selected = spectra[:, frame_blocks == block, :]
        block_counts.append(selected.shape[1])
        cross = selected[pair_i] * np.conj(selected[pair_j])
        cross /= np.maximum(np.abs(cross), 1e-8)
        corr = np.fft.irfft(np.mean(cross, axis=1), n=FFT, axis=1)
        block_corrs.append(np.fft.fftshift(corr, axes=1).astype(np.float32))
    block_corrs = np.stack(block_corrs)
    total = np.average(block_corrs, axis=0, weights=np.asarray(block_counts))
    return block_corrs, total.astype(np.float32), pair_list


def make_pair_features(block_corrs: np.ndarray, rec: SceneRecord, pairs: list[tuple[int, int]]) -> np.ndarray:
    n_pairs = len(pairs)
    x = np.zeros((n_pairs, 6, FFT), dtype=np.float32)
    for b in range(BLOCKS):
        corr = block_corrs[b]
        scale = np.max(np.abs(corr), axis=1, keepdims=True) + 1e-6
        x[:, b, :] = corr / scale
    mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
    mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5, dtype=np.float32)))
    pair_lengths = np.asarray([np.linalg.norm(mics[i] - mics[j]) for i, j in pairs], dtype=np.float32)
    x[:, 4, :] = (pair_lengths / 8.1)[:, None]
    x[:, 5, :] = rec.n_sources / 3.0
    return x


def make_pair_targets(rec: SceneRecord, pairs: list[tuple[int, int]], sigma_samples: float = 2.5) -> np.ndarray:
    """Permutation-invariant multi-source direct-path TDOA heatmaps."""
    mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
    mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5)))
    src_xy = np.asarray(rec.source_positions_xy, dtype=np.float64)
    sources = np.column_stack((src_xy, np.full(len(src_xy), 1.5)))
    lags = np.arange(-(FFT // 2), FFT // 2, dtype=np.float64)
    target = np.zeros((len(pairs), FFT), dtype=np.float32)
    for pi, (i, j) in enumerate(pairs):
        delays = (np.linalg.norm(sources - mics[i], axis=1) - np.linalg.norm(sources - mics[j], axis=1)) * FS / C
        for delay in delays:
            target[pi] = np.maximum(target[pi], np.exp(-0.5 * ((lags - delay) / sigma_samples) ** 2))
    return target


def load_manifests(train_count: int = 1200, train_mics: int | None = None):
    val_all = list(read_jsonl(VAL_PATH, SceneRecord))
    train_all = list(read_jsonl(TRAIN_PATH, SceneRecord))
    if train_mics is not None:
        train_all = [r for r in train_all if r.n_microphones == train_mics]
    val = triage_subset(val_all)
    train = balanced_subset(train_all, train_count, 20261108)
    if len(val) != 360 or len(train) != train_count:
        raise RuntimeError("amostras Round 4 incompletas")
    return train, val


def prepare_sample_cache(limit_train: int | None = None, limit_val: int | None = None,
                         train_count: int = 1200, train_mics: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_manifests(train_count, train_mics)
    if limit_train:
        train = train[:limit_train]
    if limit_val:
        val = val[:limit_val]
    start = time.perf_counter()
    train_x, train_y, train_scene = [], [], []
    for si, rec in enumerate(train):
        blocks, _, pairs = pairwise_block_gcc(read_audio(rec))
        train_x.append(make_pair_features(blocks, rec, pairs))
        train_y.append(make_pair_targets(rec, pairs))
        train_scene.append(np.full(len(pairs), si, dtype=np.int32))
        if (si + 1) % 100 == 0 or si + 1 == len(train):
            log("prepare_train", scenes=si + 1, total=len(train), elapsed_s=round(time.perf_counter() - start, 1))
    np.savez_compressed(OUT / "train_pairwise.npz", x=np.concatenate(train_x), y=np.concatenate(train_y),
                        scene_index=np.concatenate(train_scene), scene_ids=np.asarray([r.scene_id for r in train]))
    val_x, val_corr, offsets = [], [], [0]
    for si, rec in enumerate(val):
        blocks, corr, pairs = pairwise_block_gcc(read_audio(rec))
        val_x.append(make_pair_features(blocks, rec, pairs))
        val_corr.append(corr)
        offsets.append(offsets[-1] + len(pairs))
        if (si + 1) % 60 == 0 or si + 1 == len(val):
            log("prepare_val", scenes=si + 1, total=len(val), elapsed_s=round(time.perf_counter() - start, 1))
    np.savez_compressed(OUT / "validation_pairwise.npz", x=np.concatenate(val_x), corr=np.concatenate(val_corr),
                        offsets=np.asarray(offsets), ids=np.asarray([r.scene_id for r in val]))
    meta = {"train_scenes": len(train), "validation_scenes": len(val), "train_ids": [r.scene_id for r in train],
            "validation_ids": [r.scene_id for r in val], "features": ["4x block GCC-PHAT", "pair baseline", "known N"],
            "labels": "unordered union of exact geometric direct-path TDOAs; Gaussian sigma 2.5 samples",
            "device": str(DEVICE), "seconds": time.perf_counter() - start}
    (OUT / "protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("prepare_done", seconds=round(meta["seconds"], 1), device=str(DEVICE))


class DirectPathTDOA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(6, 24, kernel_size=9, padding=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(24, 32, kernel_size=7, padding=6, dilation=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(32, 24, kernel_size=5, padding=8, dilation=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(24, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_direct_path(epochs: int = 15, batch_size: int = 128):
    z = np.load(OUT / "train_pairwise.npz", allow_pickle=False)
    x = torch.from_numpy(z["x"])
    y = torch.from_numpy(z["y"])
    scene_index = z["scene_index"]
    n_scenes = len(z["scene_ids"])
    rng = np.random.default_rng(20261108)
    perm = rng.permutation(n_scenes)
    n_val_scenes = max(1, n_scenes // 10)
    val_scenes = set(perm[:n_val_scenes].tolist())
    is_val = np.asarray([int(s) in val_scenes for s in scene_index])
    train_ix = torch.from_numpy(np.flatnonzero(~is_val))
    val_ix = torch.from_numpy(np.flatnonzero(is_val))
    model = DirectPathTDOA().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(12.0, device=DEVICE))
    best_loss = float("inf")
    best_epoch, stale = 0, 0
    rng_torch = torch.Generator().manual_seed(20261108)
    start = time.perf_counter()
    log("train_start", train_pairs=len(train_ix), val_pairs=len(val_ix), device=str(DEVICE), epochs=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_ix[torch.randperm(len(train_ix), generator=rng_torch)]
        loss_sum, count = 0.0, 0
        for ix in order.split(batch_size):
            xb = x[ix].to(DEVICE, non_blocking=True)
            yb = y[ix].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(ix)
            count += len(ix)
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for ix in val_ix.split(batch_size):
                xb = x[ix].to(DEVICE)
                yb = y[ix].to(DEVICE)
                loss = criterion(model(xb), yb)
                val_sum += float(loss) * len(ix)
                val_count += len(ix)
        val_loss = val_sum / max(val_count, 1)
        log("train_epoch", epoch=epoch, train_loss=loss_sum / max(count, 1), val_loss=val_loss,
            elapsed_s=round(time.perf_counter() - start, 1))
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "sample_ids": z["scene_ids"].tolist()}, OUT / "direct_path_tdoa.pt")
        else:
            stale += 1
            if epoch >= 5 and stale >= 3:
                break
    summary = {"epochs_run": epoch, "best_epoch": best_epoch, "best_internal_val_bce": best_loss,
               "seconds": time.perf_counter() - start, "train_scenes": n_scenes,
               "pair_samples": int(len(x)), "device": str(DEVICE)}
    (OUT / "direct_path_training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("train_done", **summary)


def infer_direct_path() -> np.ndarray:
    z = np.load(OUT / "validation_pairwise.npz", allow_pickle=False)
    x = z["x"]
    state = torch.load(OUT / "direct_path_tdoa.pt", map_location=DEVICE, weights_only=False)
    model = DirectPathTDOA().to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.from_numpy(x[start:start + 512]).to(DEVICE)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32))
    return np.concatenate(outs)


def evaluate_direct_full(limit: int | None = None):
    """One frozen direct-path TDOA model versus Round-2 seed 42 on all validation scenes."""
    records = list(read_jsonl(VAL_PATH, SceneRecord))
    if limit:
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(limit, len(records)), dtype=int)]
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cache_ids = cache["ids"]
    srp_cache = cache["srp"]
    cnn42_cache = cache["cnn42"]
    cache_index = {str(scene_id): i for i, scene_id in enumerate(cache_ids)}
    xs, ys = cache["xs"], cache["ys"]
    grid = make_grid((8.0, 6.0, 3.0), 0.10)
    state_path = OUT / "direct_path_tdoa.pt"
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    protocol = {
        "status": "frozen_before_full_validation_inference",
        "frozen_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(state_path), "checkpoint_sha256": digest,
        "train_scene_count": 2400, "triage_scene_count": 360,
        "validation_scene_count": len(records), "independent_test_used": False,
        "method": "4-block GCC-PHAT direct-path TDOA CNN -> exact near-field SRP scoring -> local peaks NMS 0.60m",
        "comparator": "Round-2 seed42: 0.75*rank(SRP)+0.25*CNN sigmoid; NMS 0.60m",
        "selection_note": "The full validation includes the 360-scene exploratory triage; treat results as exploratory, not independent confirmation.",
    }
    (OUT / "full_validation_protocol_2400.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    model = DirectPathTDOA().to(DEVICE)
    state = torch.load(state_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    pending = []
    pending_samples = 0
    baseline_estimates, candidate_estimates = [], []
    progress_path = OUT / "full_validation_predictions_progress_2400.json"
    start = time.perf_counter()

    def flush():
        nonlocal pending, pending_samples
        if not pending:
            return
        x = np.concatenate([item[1] for item in pending], axis=0)
        pred_parts = []
        with torch.no_grad():
            for st in range(0, len(x), 512):
                xb = torch.from_numpy(x[st:st + 512]).to(DEVICE)
                pred_parts.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32))
        prediction = np.concatenate(pred_parts)
        offset = 0
        for rec, x_scene, pairs, total_corr in pending:
            n_pairs = len(pairs)
            nn_corr = prediction[offset:offset + n_pairs]
            offset += n_pairs
            ci = cache_index[rec.scene_id]
            score = 0.75 * rank_map(srp_cache[ci]) + 0.25 * cnn42_cache[ci]
            baseline_estimates.append(peaks(score, xs, ys, rec.n_sources, 0.60, True))
            mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
            mics = np.column_stack((mic_xy, np.full(len(mic_xy), 1.5, dtype=np.float32)))
            nn_map = score_srp_points(nn_corr, pairs, mics, grid.points_xy, FS, FFT, C).reshape(len(ys), len(xs))
            candidate_estimates.append(peaks(nn_map, xs, ys, rec.n_sources, 0.60, True))
        pending = []
        pending_samples = 0
        if len(baseline_estimates) % 100 < 20:
            payload = {"done_scenes": len(baseline_estimates),
                       "baseline": [x.tolist() for x in baseline_estimates],
                       "direct_path_nn": [x.tolist() for x in candidate_estimates]}
            progress_path.write_text(json.dumps(payload), encoding="utf-8")

    for i, rec in enumerate(records):
        blocks, corr, pairs = pairwise_block_gcc(read_audio(rec))
        x_scene = make_pair_features(blocks, rec, pairs)
        pending.append((rec, x_scene, pairs, corr))
        pending_samples += len(pairs)
        if pending_samples >= 512 or i + 1 == len(records):
            flush()
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            log("full_direct", scenes=i + 1, total=len(records), elapsed_s=round(time.perf_counter() - start, 1))

    estimates = {"round2_seed42": baseline_estimates, "direct_path_nn_2400": candidate_estimates}
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), e)
                     for r, e in zip(records, rows)] for name, rows in estimates.items()}
    metrics = [{"method": name, **metric_summary(errs, records)} for name, errs in errors.items()]
    (OUT / "full_validation_metrics_2400.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "full_validation_source_errors_2400.csv", records, estimates)
    stratified_metrics(records, errors, OUT / "full_validation_stratified_2400.csv")
    paired = {"direct_path_nn_2400": paired_scene_bootstrap_variable_n(
        errors["direct_path_nn_2400"], errors["round2_seed42"], seed=20261108, repeats=10000)}
    (OUT / "full_validation_paired_bootstrap_2400.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    delta = {"mean_delta_m": metrics[1]["mean_error_m"] - metrics[0]["mean_error_m"],
             "p90_improvement_m": metrics[0]["p90_error_m"] - metrics[1]["p90_error_m"],
             "relative_over_1m_reduction": (metrics[0]["over_1m_rate"] - metrics[1]["over_1m_rate"]) /
                                            max(metrics[0]["over_1m_rate"], 1e-8),
             "within_0_5m_delta": metrics[1]["within_0_5m"] - metrics[0]["within_0_5m"]}
    (OUT / "full_validation_gate_2400.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")
    (OUT / "timing_full_validation_2400.json").write_text(json.dumps(
        {"seconds": time.perf_counter() - start, "scene_count": len(records)}, indent=2), encoding="utf-8")
    log("full_direct_result", seconds=round(time.perf_counter() - start, 1), metrics=metrics, gate=delta, paired=paired)


def score_image_hypotheses(corr: np.ndarray, pairs: list[tuple[int, int]], mics: np.ndarray,
                           points_xy: np.ndarray, room: tuple[float, float, float],
                           sample_rate: int = FS, reflection_weight: float = 0.5) -> np.ndarray:
    """Direct SRP plus weighted mean of six first-order image-source SRP maps."""
    width, length, height = room
    z0 = float(np.mean(mics[:, 2]))
    p3 = np.column_stack((points_xy, np.full(len(points_xy), z0)))
    images = [p3,
              np.column_stack((-points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), z0))),
              np.column_stack((2 * width - points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), z0))),
              np.column_stack((points_xy[:, 0], -points_xy[:, 1], np.full(len(points_xy), z0))),
              np.column_stack((points_xy[:, 0], 2 * length - points_xy[:, 1], np.full(len(points_xy), z0))),
              np.column_stack((points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), -z0))),
              np.column_stack((points_xy[:, 0], points_xy[:, 1], np.full(len(points_xy), 2 * height - z0)))]
    lag_samples = np.arange(-(FFT // 2), FFT // 2, dtype=np.float64)
    scores = np.zeros((len(images), len(points_xy)), dtype=np.float32)
    for path_idx, path_points in enumerate(images):
        for pi, (i, j) in enumerate(pairs):
            lag = (np.linalg.norm(path_points - mics[i], axis=1) -
                   np.linalg.norm(path_points - mics[j], axis=1)) * sample_rate / C
            scores[path_idx] += np.interp(lag, lag_samples, corr[pi], left=0.0, right=0.0).astype(np.float32)
        scores[path_idx] /= max(len(pairs), 1)
    return scores[0] + reflection_weight * np.mean(scores[1:], axis=0)


def local_candidates(score: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: int = 32,
                     separation_m: float = 0.10) -> np.ndarray:
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


def pair_peak_lists(corr: np.ndarray, pairs: list[tuple[int, int]], mics: np.ndarray, k: int = 5):
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
        values = c[selected]
        values = values / (float(values.max()) + 1e-8)
        all_lags.append(lags)
        all_values.append(values.astype(np.float32))
    return np.asarray(all_lags), np.asarray(all_values)


def paired_scene_bootstrap_variable_n(candidate_errors, baseline_errors, seed=20261108, repeats=10000):
    """Paired cluster bootstrap; sample scenes and pool their source-level deltas."""
    rng = np.random.default_rng(seed)
    scene_delta_sum = np.asarray([np.sum(np.asarray(a) - np.asarray(b))
                                  for a, b in zip(candidate_errors, baseline_errors)], dtype=np.float64)
    scene_source_count = np.asarray([len(a) for a in candidate_errors], dtype=np.int32)
    point = float(scene_delta_sum.sum() / scene_source_count.sum())
    draws = np.empty(repeats, dtype=np.float64)
    n = len(scene_delta_sum)
    batch = 500
    for start in range(0, repeats, batch):
        end = min(repeats, start + batch)
        idx = rng.integers(0, n, size=(end - start, n))
        sums = scene_delta_sum[idx].sum(axis=1)
        counts = scene_source_count[idx].sum(axis=1)
        draws[start:end] = sums / counts
    return {"mean_delta_m": point,
            "ci95_m": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "repeats": repeats, "unit": "scene cluster; source-weighted aggregate",
            "negative_favors_candidate": True}


def joint_tdoa_decode(rec: SceneRecord, candidate_map: np.ndarray, corr: np.ndarray,
                      xs: np.ndarray, ys: np.ndarray, k_candidates: int = 32, k_peaks: int = 5,
                      sigma_samples: float = 2.5) -> np.ndarray:
    """Choose an unordered source set with distinct TDOA-peak assignments per pair."""
    candidates = local_candidates(candidate_map, xs, ys, k_candidates)
    n = rec.n_sources
    if len(candidates) < n:
        raise RuntimeError("pool de candidatos menor que N")
    combinations = np.asarray(list(itertools.combinations(range(len(candidates)), n)), dtype=np.int32)
    assignments = np.asarray(list(itertools.permutations(range(k_peaks), n)), dtype=np.int32)
    mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float64)
    mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5)))
    points = np.column_stack((candidates, np.full(len(candidates), 1.5)))
    pairs = pair_indices(rec.n_microphones)
    peak_lags, peak_values = pair_peak_lists(corr, pairs, mics, k_peaks)
    set_scores = np.zeros(len(combinations), dtype=np.float64)
    for pi, (i, j) in enumerate(pairs):
        expected = (np.linalg.norm(points - mics[i], axis=1) - np.linalg.norm(points - mics[j], axis=1)) * FS / C
        quality = peak_values[pi][None, :] * np.exp(-0.5 * ((expected[:, None] - peak_lags[pi][None, :]) / sigma_samples) ** 2)
        best = np.zeros(len(combinations), dtype=np.float64)
        for assignment in assignments:
            score = np.zeros(len(combinations), dtype=np.float64)
            for source_idx, peak_idx in enumerate(assignment):
                score += quality[combinations[:, source_idx], peak_idx]
            best = np.maximum(best, score)
        set_scores += best
    best_set = combinations[int(np.argmax(set_scores))]
    return candidates[best_set]


def evaluate_maps(stage: str = "screen", limit: int | None = None):
    train, val = load_manifests()
    if limit:
        selected = np.linspace(0, len(val) - 1, min(limit, len(val)), dtype=int)
        val = [val[i] for i in selected]
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cache_ids = cache["ids"]
    srp_cache = cache["srp"]
    cnn42_cache = cache["cnn42"]
    cache_index = {str(scene_id): i for i, scene_id in enumerate(cache_ids)}
    xs, ys = cache["xs"], cache["ys"]
    grid = make_grid((8.0, 6.0, 3.0), 0.10)
    nn_pred = infer_direct_path()
    vcache = np.load(OUT / "validation_pairwise.npz", allow_pickle=False)
    offsets, corr_all = vcache["offsets"], vcache["corr"]
    val_cache_index = {str(scene_id): i for i, scene_id in enumerate(vcache["ids"])}
    nn_maps, reflection_maps, tdoa_estimates, reflection_estimates = [], [], [], []
    estimates = {"round2_seed42": [], "direct_path_nn": [], "direct_path_plus_srp": [],
                 "first_order_reflection": [], "reflection_plus_cnn": [], "joint_tdoa_from_baseline_pool": [],
                 "joint_tdoa_from_neural_pool": []}
    hdiag = []
    start = time.perf_counter()
    for si, rec in enumerate(val):
        ci = cache_index[rec.scene_id]
        srp = srp_cache[ci]
        cnn = cnn42_cache[ci]
        base_map = 0.75 * rank_map(srp) + 0.25 * cnn
        estimates["round2_seed42"].append(peaks(base_map, xs, ys, rec.n_sources, 0.60, True))
        vi = val_cache_index[rec.scene_id]
        a, b = int(offsets[vi]), int(offsets[vi + 1])
        corr = corr_all[a:b]
        pairs = pair_indices(rec.n_microphones)
        mics_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
        mics = np.column_stack((mics_xy, np.full(len(mics_xy), 1.5, dtype=np.float32)))
        nn_corr = nn_pred[a:b]
        nn_map = score_srp_points(nn_corr, pairs, mics, grid.points_xy, FS, FFT, C).reshape(len(ys), len(xs))
        refl_flat = score_image_hypotheses(corr, pairs, mics, grid.points_xy, (8.0, 6.0, 3.0), reflection_weight=0.5)
        refl_map = refl_flat.reshape(len(ys), len(xs))
        estimates["direct_path_nn"].append(peaks(nn_map, xs, ys, rec.n_sources, 0.60, True))
        nn_fuse = 0.50 * rank_map(nn_map) + 0.50 * rank_map(srp)
        estimates["direct_path_plus_srp"].append(peaks(nn_fuse, xs, ys, rec.n_sources, 0.60, True))
        estimates["first_order_reflection"].append(peaks(refl_map, xs, ys, rec.n_sources, 0.60, True))
        refl_fuse = 0.75 * rank_map(refl_map) + 0.25 * cnn
        estimates["reflection_plus_cnn"].append(peaks(refl_fuse, xs, ys, rec.n_sources, 0.60, True))
        # Use pairwise uniqueness to rank source sets from a controlled 32-candidate pool.
        estimates["joint_tdoa_from_baseline_pool"].append(
            joint_tdoa_decode(rec, base_map, corr, xs, ys, k_candidates=32, k_peaks=5, sigma_samples=2.5))
        combined = 0.5 * rank_map(base_map) + 0.25 * rank_map(nn_map) + 0.25 * rank_map(refl_map)
        estimates["joint_tdoa_from_neural_pool"].append(
            joint_tdoa_decode(rec, combined, corr, xs, ys, k_candidates=32, k_peaks=5, sigma_samples=2.5))
        if (si + 1) % 30 == 0 or si + 1 == len(val):
            log("evaluate", scenes=si + 1, total=len(val), elapsed_s=round(time.perf_counter() - start, 1))
    metrics = []
    all_errors = {}
    for name, rows in estimates.items():
        errs = [pairwise_assignment_errors(np.asarray(rec.source_positions_xy), pred)
                for rec, pred in zip(val, rows)]
        all_errors[name] = errs
        metrics.append({"method": name, **metric_summary(errs, val)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "triage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "triage_source_errors.csv", val, estimates)
    np.savez_compressed(OUT / "triage_estimates.npz", **{k: np.array(v, dtype=object) for k, v in estimates.items()})
    summary = [{"method": m["method"], "mean": m["mean_error_m"], "median": m["median_error_m"],
                "p90": m["p90_error_m"], "over1": m["over_1m_rate"], "within05": m["within_0_5m"]} for m in metrics]
    timing = {"scenes": len(val), "seconds": time.perf_counter() - start, "device": str(DEVICE), "stage": stage}
    (OUT / "timing_screen.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    log("triage_result", **timing, metrics=summary)


def evaluate_m8_triage():
    """Evaluate the M=8 specialist only on the M=8 cells of frozen 360-scene triage."""
    _, val_all = load_manifests()
    val = [r for r in val_all if r.n_microphones == 8]
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cache_ids = cache["ids"]
    srp_cache = cache["srp"]
    cnn42_cache = cache["cnn42"]
    cache_index = {str(scene_id): i for i, scene_id in enumerate(cache_ids)}
    xs, ys = cache["xs"], cache["ys"]
    grid = make_grid((8.0, 6.0, 3.0), 0.10)
    nn_pred = infer_direct_path()
    vcache = np.load(OUT / "validation_pairwise.npz", allow_pickle=False)
    offsets, corr_all = vcache["offsets"], vcache["corr"]
    val_cache_index = {str(scene_id): i for i, scene_id in enumerate(vcache["ids"])}
    estimates = {"round2_seed42": [], "direct_path_nn_m8": []}
    start = time.perf_counter()
    for si, rec in enumerate(val):
        ci = cache_index[rec.scene_id]
        base_score = 0.75 * rank_map(srp_cache[ci]) + 0.25 * cnn42_cache[ci]
        estimates["round2_seed42"].append(peaks(base_score, xs, ys, rec.n_sources, 0.60, True))
        vi = val_cache_index[rec.scene_id]
        a, b = int(offsets[vi]), int(offsets[vi + 1])
        pairs = pair_indices(rec.n_microphones)
        mics = np.column_stack((np.asarray(rec.microphone_positions_xy, dtype=np.float32),
                                np.full(rec.n_microphones, 1.5, dtype=np.float32)))
        nn_map = score_srp_points(nn_pred[a:b], pairs, mics, grid.points_xy, FS, FFT, C).reshape(len(ys), len(xs))
        estimates["direct_path_nn_m8"].append(peaks(nn_map, xs, ys, rec.n_sources, 0.60, True))
        if (si + 1) % 20 == 0 or si + 1 == len(val):
            log("m8_triage", scenes=si + 1, total=len(val), elapsed_s=round(time.perf_counter() - start, 1))
    errors = {name: [pairwise_assignment_errors(np.asarray(r.source_positions_xy), p)
                     for r, p in zip(val, rows)] for name, rows in estimates.items()}
    metrics = [{"method": name, **metric_summary(err, val)} for name, err in errors.items()]
    (OUT / "m8_triage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_source_errors(OUT / "m8_triage_source_errors.csv", val, estimates)
    (OUT / "timing_m8_triage.json").write_text(json.dumps(
        {"scenes": len(val), "seconds": time.perf_counter() - start}, indent=2), encoding="utf-8")
    log("m8_triage_result", metrics=metrics, seconds=round(time.perf_counter() - start, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "full-direct", "m8-triage"))
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--train-scenes", type=int, default=1200)
    parser.add_argument("--train-mics", type=int)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_sample_cache(args.limit_train, args.limit_val, args.train_scenes, args.train_mics)
    elif args.stage == "train":
        train_direct_path(args.epochs, args.batch_size)
    elif args.stage == "evaluate":
        evaluate_maps(limit=args.limit_val)
    elif args.stage == "m8-triage":
        evaluate_m8_triage()
    else:
        evaluate_direct_full(args.limit_val)


if __name__ == "__main__":
    main()
