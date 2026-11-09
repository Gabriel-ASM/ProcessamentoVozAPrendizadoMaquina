"""Summarize Round-4 exploratory validation results by acoustic stratum."""
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


def summary(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {"source_count": int(x.size), "mean_m": float(x.mean()), "median_m": float(np.median(x)),
            "p90_m": float(np.percentile(x, 90)), "over_1m_rate": float(np.mean(x > 1.0)),
            "within_0_5m_rate": float(np.mean(x <= 0.5))}


def main():
    records = {r.scene_id: r for r in read_jsonl(VAL, SceneRecord)}
    rows = list(csv.DictReader((OUT / "triage_source_errors.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    scene_method = defaultdict(list)
    groups_n = defaultdict(list)
    for row in rows:
        rec = records[row["scene_id"]]
        error = float(row["error_m"])
        groups[(row["method"], rec.n_sources, rec.n_microphones, rec.geometry)].append(error)
        groups_n[(row["method"], rec.n_sources)].append(error)
        scene_method[(row["method"], rec.scene_id)].append(error)
    output = []
    for key, vals in sorted(groups.items()):
        method, n, m, geom = key
        output.append({"method": method, "n_sources": n, "n_microphones": m, "geometry": geom,
                       **summary(vals)})
    fields = list(output[0])
    with (OUT / "triage_stratified.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)

    by_n = []
    for (method, n), vals in sorted(groups_n.items()):
        by_n.append({"method": method, "n_sources": n, **summary(vals)})
    with (OUT / "triage_by_n.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_n[0]))
        writer.writeheader(); writer.writerows(by_n)

    methods = sorted({key[0] for key in scene_method})
    scene_ids = sorted({key[1] for key in scene_method})
    baseline = {sid: float(np.mean(scene_method[("round2_seed42", sid)])) for sid in scene_ids}
    rng = np.random.default_rng(20261108)
    paired = {}
    for method in methods:
        if method == "round2_seed42":
            continue
        deltas = np.asarray([float(np.mean(scene_method[(method, sid)])) - baseline[sid] for sid in scene_ids])
        sampled = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(axis=1)
        paired[method] = {"scene_mean_delta_m": float(deltas.mean()),
                          "bootstrap_95pct_ci_m": [float(np.quantile(sampled, .025)),
                                                   float(np.quantile(sampled, .975))],
                          "scenes_better": int(np.sum(deltas < 0)), "scene_count": int(len(deltas))}
    (OUT / "triage_paired_bootstrap.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    scene_n = {sid: records[sid].n_sources for sid in scene_ids}
    paired_by_n = {}
    for n in (1, 2, 3):
        ids_n = [sid for sid in scene_ids if scene_n[sid] == n]
        b_n = {sid: float(np.mean(scene_method[("round2_seed42", sid)])) for sid in ids_n}
        paired_by_n[str(n)] = {}
        for method in methods:
            if method == "round2_seed42":
                continue
            d = np.asarray([float(np.mean(scene_method[(method, sid)])) - b_n[sid] for sid in ids_n])
            sample = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
            paired_by_n[str(n)][method] = {"scene_mean_delta_m": float(d.mean()),
                                           "bootstrap_95pct_ci_m": [float(np.quantile(sample, .025)),
                                                                    float(np.quantile(sample, .975))],
                                           "scenes_better": int(np.sum(d < 0)), "scene_count": len(ids_n)}
    (OUT / "triage_paired_by_n.json").write_text(json.dumps(paired_by_n, indent=2), encoding="utf-8")
    print(json.dumps({"strata_rows": len(output), "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
