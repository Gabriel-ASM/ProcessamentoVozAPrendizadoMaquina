"""Full-validation analysis and a small, deployment-observable routing diagnostic."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voice_localization.manifest import SceneRecord, read_jsonl

OUT = ROOT / "artifacts" / "round4"
VAL = ROOT / "data" / "full" / "manifests" / "scenes_validation.jsonl"


def summarize(errors):
    x = np.concatenate(errors)
    return {"source_count": int(len(x)), "mean_m": float(x.mean()), "median_m": float(np.median(x)),
            "p90_m": float(np.percentile(x, 90)), "within_0_1m": int(np.sum(x < .10)),
            "within_0_5m_rate": float(np.mean(x <= .5)), "over_1m": int(np.sum(x > 1.0)),
            "over_1m_rate": float(np.mean(x > 1.0)), "over_2m_rate": float(np.mean(x > 2.0))}


def cluster_bootstrap(candidate, baseline, seed=20261108, repeats=10000):
    delta_sum = np.asarray([np.sum(np.asarray(a) - np.asarray(b)) for a, b in zip(candidate, baseline)])
    counts = np.asarray([len(a) for a in candidate])
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    batch = 500
    for start in range(0, repeats, batch):
        end = min(repeats, start + batch)
        idx = rng.integers(0, len(counts), size=(end - start, len(counts)))
        estimates[start:end] = delta_sum[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"mean_delta_m": float(delta_sum.sum() / counts.sum()),
            "ci95_m": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))],
            "repeats": repeats}


def main():
    records = {r.scene_id: r for r in read_jsonl(VAL, SceneRecord)}
    grouped = defaultdict(dict)
    for row in csv.DictReader((OUT / "full_validation_source_errors_2400.csv").open(encoding="utf-8")):
        grouped[row["method"]].setdefault(row["scene_id"], {})[int(row["source_index"])] = float(row["error_m"])
    base = grouped["round2_seed42"]
    direct = grouped["direct_path_nn_2400"]
    scenarios = {
        "baseline": lambda r: "round2_seed42",
        "direct_path_all": lambda r: "direct_path_nn_2400",
        "direct_path_only_M4_M6": lambda r: "direct_path_nn_2400" if r.n_microphones in (4, 6) else "round2_seed42",
        "direct_path_only_perimeter": lambda r: "direct_path_nn_2400" if r.geometry == "perimeter" else "round2_seed42",
        "direct_path_only_M4_M6_perimeter": lambda r: "direct_path_nn_2400" if r.n_microphones in (4, 6) and r.geometry == "perimeter" else "round2_seed42",
    }
    results = {}
    route_errors = {}
    for name, choose in scenarios.items():
        per_scene = []
        for sid, rec in records.items():
            method = choose(rec)
            source_errors = grouped[method][sid]
            per_scene.append(np.asarray([source_errors[i] for i in sorted(source_errors)]))
        route_errors[name] = per_scene
        result = summarize(per_scene)
        if name != "baseline":
            result["paired_bootstrap_mean_delta"] = cluster_bootstrap(per_scene, route_errors["baseline"])
        results[name] = result
    (OUT / "full_validation_routing_diagnostic_2400.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
