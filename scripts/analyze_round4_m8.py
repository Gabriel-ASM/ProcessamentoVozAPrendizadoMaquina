"""Paired scene-level analysis of the M=8 specialist triage."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round4"


def main():
    by_method_scene = defaultdict(list)
    with (OUT / "m8_triage_source_errors.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_method_scene[(row["method"], row["scene_id"])].append(float(row["error_m"]))
    ids = sorted({sid for _, sid in by_method_scene})
    baseline = np.asarray([by_method_scene[("round2_seed42", sid)] for sid in ids], dtype=object)
    candidate = np.asarray([by_method_scene[("direct_path_nn_m8", sid)] for sid in ids], dtype=object)
    scene_delta = np.asarray([np.mean(c) - np.mean(b) for b, c in zip(baseline, candidate)])
    scene_over1_delta = np.asarray([np.mean(np.asarray(c) > 1) - np.mean(np.asarray(b) > 1)
                                    for b, c in zip(baseline, candidate)])
    rng = np.random.default_rng(20261108)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    out = {
        "scene_count": len(ids),
        "baseline_mean_m": float(np.mean(np.concatenate(baseline.tolist()))),
        "candidate_mean_m": float(np.mean(np.concatenate(candidate.tolist()))),
        "source_weighted_mean_delta_m": float(np.mean(np.concatenate(candidate.tolist())) -
                                               np.mean(np.concatenate(baseline.tolist()))),
        "scene_weighted_mean_delta_m": float(scene_delta.mean()),
        "scene_weighted_mean_delta_bootstrap_95pct_ci_m": [
            float(x) for x in np.quantile(scene_delta[idx].mean(axis=1), [.025, .975])],
        "scenes_improved": int(np.sum(scene_delta < 0)),
        "source_weighted_over1_rate_delta": float(np.mean(np.concatenate(candidate.tolist()) > 1) -
                                                   np.mean(np.concatenate(baseline.tolist()) > 1)),
        "scene_weighted_over1_rate_delta": float(scene_over1_delta.mean()),
        "scene_weighted_over1_rate_delta_bootstrap_95pct_ci": [
            float(x) for x in np.quantile(scene_over1_delta[idx].mean(axis=1), [.025, .975])],
        "bootstrap_replicates": 10000,
        "seed": 20261108,
        "note": "Triage-only exploratory interval; not independent confirmation.",
    }
    (OUT / "m8_triage_paired_bootstrap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
