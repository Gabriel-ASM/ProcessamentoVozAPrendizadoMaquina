"""Fit the Pyroomacoustics RT60 correction from a training-only scene manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def calibrate(manifest_path: Path) -> dict:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty scene manifest: {manifest_path}")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("RT60 calibration must use the train split only")
    valid = [
        row
        for row in rows
        if row.get("rt60_realized") is not None
        and row.get("metadata", {}).get("rt60_sabine_input") is not None
    ]
    if len(valid) < 3:
        raise ValueError("at least three train scenes with measured RT60 are required")
    sabine = np.asarray([row["metadata"]["rt60_sabine_input"] for row in valid], dtype=np.float64)
    realized = np.asarray([row["rt60_realized"] for row in valid], dtype=np.float64)
    if not np.all(np.isfinite(sabine)) or not np.all(np.isfinite(realized)) or np.ptp(sabine) <= 0:
        raise ValueError("RT60 calibration values must be finite and span multiple Sabine inputs")
    slope, intercept = np.polyfit(sabine, realized, 1)
    residual = realized - (slope * sabine + intercept)
    return {
        "manifest": str(manifest_path),
        "split": "train",
        "scene_count": len(rows),
        "measured_scene_count": len(valid),
        "pyroomacoustics_version": _pyroom_version(),
        "fit": "rt60_realized = slope * rt60_sabine_input + intercept",
        "rt60_calibration_slope": float(slope),
        "rt60_calibration_intercept": float(intercept),
        "residual_rmse_seconds": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs_seconds": float(np.max(np.abs(residual))),
        "sabine_input_range_seconds": [float(np.min(sabine)), float(np.max(sabine))],
        "realized_rt60_range_seconds": [float(np.min(realized)), float(np.max(realized))],
    }


def _pyroom_version() -> str | None:
    try:
        import pyroomacoustics as pra

        return str(pra.__version__)
    except ImportError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
