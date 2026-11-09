"""Avaliação do baseline SRP e de checkpoints em protocolo comum."""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

from .config import ProjectConfig
from .manifest import SceneRecord, read_jsonl
from .metrics import decode_peaks, localization_errors, localization_metrics
from .dataset import load_map_sample


def evaluate_srp(config: ProjectConfig, split: str = "test", output_path: str | Path | None = None) -> dict:
    manifest = Path(config.dataset.manifest_root) / _manifest_name(config, split)
    records = list(read_jsonl(manifest, SceneRecord))
    rows: list[dict] = []
    for record in records:
        started = time.perf_counter()
        features, _, n_sources, reference, xs, ys, _ = load_map_sample(record)
        estimate = decode_peaks(features[0], xs, ys, n_sources, config.grid.peak_suppression_radius_m)
        row = localization_metrics(reference, estimate).to_dict()
        row["source_errors_m"] = localization_errors(reference, estimate).tolist()
        row["reference_xy"] = np.asarray(reference, dtype=np.float32).tolist()
        row["estimate_xy"] = np.asarray(estimate, dtype=np.float32).tolist()
        row["inference_seconds"] = time.perf_counter() - started
        row.update(
            {
                "scene_id": record.scene_id,
                "n_sources": record.n_sources,
                "n_microphones": record.n_microphones,
                "geometry": record.geometry,
                "snr_requested_db": record.snr_requested_db,
                "rt60_requested": record.rt60_requested,
                "rt60_realized": record.rt60_realized,
                "snr_band": _snr_band(record.snr_requested_db),
                "rt60_band": _rt60_band(
                    record.rt60_realized if record.rt60_realized is not None else record.rt60_requested
                ),
            }
        )
        rows.append(row)
    summary = _summarize(rows)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("nenhuma cena para avaliar")
    errors = np.asarray(
        [error for row in rows for error in row.get("source_errors_m", [row["mean_error_m"]])],
        dtype=np.float64,
    )
    summary = {
        "scene_count": len(rows),
        "source_count": int(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_tolerance": float(np.mean(errors <= 0.50)),
    }
    measured_times = [float(row["inference_seconds"]) for row in rows if "inference_seconds" in row]
    if measured_times:
        summary["mean_inference_seconds"] = float(np.mean(measured_times))
    for field in ("n_sources", "n_microphones", "geometry", "snr_band", "rt60_band"):
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            if field not in row:
                continue
            grouped.setdefault(str(row[field]), []).append(row)
        if grouped:
            summary[f"by_{field}"] = {
                name: _summarize_metrics(group_rows) for name, group_rows in sorted(grouped.items())
            }
    return summary


def _summarize_metrics(rows: list[dict]) -> dict:
    errors = np.asarray(
        [error for row in rows for error in row.get("source_errors_m", [row["mean_error_m"]])],
        dtype=np.float64,
    )
    return {
        "scene_count": len(rows),
        "source_count": int(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_tolerance": float(np.mean(errors <= 0.50)),
        "mean_inference_seconds": float(
            np.mean([row["inference_seconds"] for row in rows if "inference_seconds" in row])
        ) if any("inference_seconds" in row for row in rows) else None,
    }


def _snr_band(value: float) -> str:
    lower = int(np.floor(value / 5.0) * 5)
    return f"{lower}-{lower + 5}dB"


def _rt60_band(value: float) -> str:
    lower = round(np.floor(value / 0.2) * 0.2, 1)
    return f"{lower:.1f}-{lower + 0.2:.1f}s"


def evaluate_checkpoint(
    config: ProjectConfig,
    checkpoint_path: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
) -> dict:
    """Avalia um checkpoint congelado no mesmo protocolo do baseline."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para avaliar um checkpoint") from exc
    from .dataset import make_torch_dataset
    from .model import build_model
    from .training import _collate, _device, _localization_batch_metrics

    manifest = Path(config.dataset.manifest_root) / _manifest_name(config, split)
    dataset = make_torch_dataset(manifest)
    records = list(read_jsonl(manifest, SceneRecord))
    record_by_id = {record.scene_id: record for record in records}
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=_collate)
    device = _device(config)
    state = torch.load(checkpoint_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model(batch["input"].to(device))
            batch_rows = _localization_batch_metrics(
                logits, batch, tolerance_m=0.50, suppression_radius_m=config.grid.peak_suppression_radius_m
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds_per_scene = (time.perf_counter() - started) / len(batch["scene_id"])
            for row, scene_id in zip(batch_rows, batch["scene_id"]):
                record = record_by_id[scene_id]
                row["reference_xy"] = np.asarray(record.source_positions_xy, dtype=np.float32).tolist()
                row["inference_seconds"] = seconds_per_scene
                row.update(
                    {
                        "scene_id": scene_id,
                        "n_sources": record.n_sources,
                        "n_microphones": record.n_microphones,
                        "geometry": record.geometry,
                        "rt60_requested": record.rt60_requested,
                        "rt60_realized": record.rt60_realized,
                        "snr_band": _snr_band(record.snr_requested_db),
                        "rt60_band": _rt60_band(
                            record.rt60_realized if record.rt60_realized is not None else record.rt60_requested
                        ),
                    }
                )
            rows.extend(batch_rows)
    summary = _summarize(rows)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary


def _manifest_name(config: ProjectConfig, split: str) -> str:
    suffix = "" if config.dataset.map_variant == "mixture" else f"_{config.dataset.map_variant}"
    return f"scenes_{split}{suffix}.jsonl"
