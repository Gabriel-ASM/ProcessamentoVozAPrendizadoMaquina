"""Paired scene-cluster bootstrap and stratification for joint-TDOA full run."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round5"
VAL_PATH = ROOT / "data" / "full" / "manifests" / "scenes_validation.jsonl"
sys.path.insert(0, str(ROOT / "src"))
from voice_localization.manifest import SceneRecord, read_jsonl


def main():
    scene_errors = defaultdict(lambda: defaultdict(list))
    with (OUT / "joint_tdoa_full_source_errors.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene_errors[row["method"]][row["scene_id"]].append(float(row["error_m"]))

    records = list(read_jsonl(VAL_PATH, SceneRecord))
    rec_by_id = {r.scene_id: r for r in records}
    ids = sorted(scene_errors["round2_seed42"])
    base = [np.asarray(scene_errors["round2_seed42"][sid], dtype=np.float64) for sid in ids]
    cand = [np.asarray(scene_errors["joint_tdoa_from_baseline_pool"][sid], dtype=np.float64) for sid in ids]
    if len(ids) != len(records) or any(len(scene_errors[m][sid]) != rec_by_id[sid].n_sources
                                       for m in scene_errors for sid in ids):
        raise ValueError("Source-error CSV does not align with validation manifest")

    rng = np.random.default_rng(20261110)
    repeats = 5000
    stats = np.empty((repeats, 4), dtype=np.float64)
    for b in range(repeats):
        ix = rng.integers(0, len(ids), size=len(ids))
        eb = np.concatenate([base[i] for i in ix])
        ec = np.concatenate([cand[i] for i in ix])
        stats[b] = (np.mean(ec) - np.mean(eb),
                    np.percentile(eb, 90) - np.percentile(ec, 90),
                    np.mean(ec > 1.0) - np.mean(eb > 1.0),
                    np.mean(ec <= .5) - np.mean(eb <= .5))

    all_b, all_c = np.concatenate(base), np.concatenate(cand)
    names = ["mean_delta_candidate_minus_baseline_m", "p90_improvement_m",
             "over1_rate_delta_candidate_minus_baseline", "within05_rate_delta_candidate_minus_baseline"]
    summary = {"scene_count": len(ids), "source_count": len(all_b), "bootstrap_replicates": repeats,
               "seed": 20261110, "point_estimates": dict(zip(names, map(float, (
                   np.mean(all_c) - np.mean(all_b),
                   np.percentile(all_b, 90) - np.percentile(all_c, 90),
                   np.mean(all_c > 1.0) - np.mean(all_b > 1.0),
                   np.mean(all_c <= .5) - np.mean(all_b <= .5))))),
               "bootstrap_95pct_ci": {name: [float(v) for v in np.quantile(stats[:, j], [.025, .975])]
                                      for j, name in enumerate(names)},
               "note": "Paired bootstrap clusters by scene; validation was used for triage, so results are exploratory."}
    (OUT / "joint_tdoa_full_p90_bootstrap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    strata = []
    for key, field in (("n_sources", "n_sources"), ("n_microphones", "n_microphones"), ("geometry", "geometry")):
        groups = defaultdict(list)
        for i, sid in enumerate(ids):
            groups[getattr(rec_by_id[sid], field)].append(i)
        for value, indices in sorted(groups.items(), key=lambda item: str(item[0])):
            for method, arrays in (("round2_seed42", base), ("joint_tdoa_from_baseline_pool", cand)):
                errors = np.concatenate([arrays[i] for i in indices])
                strata.append({"stratum": key, "value": value, "method": method,
                               "scene_count": len(indices), "source_count": len(errors),
                               "mean_error_m": float(np.mean(errors)),
                               "p90_error_m": float(np.percentile(errors, 90)),
                               "over_1m_rate": float(np.mean(errors > 1.0))})
    with (OUT / "joint_tdoa_full_stratified.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(strata[0]))
        writer.writeheader()
        writer.writerows(strata)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
