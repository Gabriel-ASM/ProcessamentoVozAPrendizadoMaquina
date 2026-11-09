"""Cluster bootstrap for the full-validation temporal finalist P90/rates."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round5"


def main():
    grouped = defaultdict(lambda: defaultdict(list))
    with (OUT / "multisource_temporal_full_source_errors.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["method"]][row["scene_id"]].append(float(row["error_m"]))
    base, cand = grouped["round2_seed42"], grouped["temporal_top2of8"]
    ids = sorted(base)
    bsc = [np.asarray(base[s], dtype=np.float64) for s in ids]
    csc = [np.asarray(cand[s], dtype=np.float64) for s in ids]
    rng = np.random.default_rng(20261109)
    repeats = 5000
    p90_improve, over1_delta, within05_delta = np.empty(repeats), np.empty(repeats), np.empty(repeats)
    for r in range(repeats):
        idx = rng.integers(0, len(ids), size=len(ids))
        b = np.concatenate([bsc[i] for i in idx])
        c = np.concatenate([csc[i] for i in idx])
        p90_improve[r] = np.percentile(b, 90) - np.percentile(c, 90)
        over1_delta[r] = np.mean(c > 1.0) - np.mean(b > 1.0)
        within05_delta[r] = np.mean(c <= .5) - np.mean(b <= .5)
    summary = {
        "scene_count": len(ids), "source_count": sum(map(len, bsc)),
        "bootstrap_replicates": repeats, "seed": 20261109,
        "p90_improvement_m": float(np.percentile(np.concatenate(bsc), 90) -
                                    np.percentile(np.concatenate(csc), 90)),
        "p90_improvement_bootstrap_95pct_ci_m": [float(x) for x in np.quantile(p90_improve, [.025, .975])],
        "over1_rate_delta_candidate_minus_baseline": float(np.mean(np.concatenate(csc) > 1) -
                                                           np.mean(np.concatenate(bsc) > 1)),
        "over1_rate_delta_bootstrap_95pct_ci": [float(x) for x in np.quantile(over1_delta, [.025, .975])],
        "within05_rate_delta_candidate_minus_baseline": float(np.mean(np.concatenate(csc) <= .5) -
                                                              np.mean(np.concatenate(bsc) <= .5)),
        "within05_rate_delta_bootstrap_95pct_ci": [float(x) for x in np.quantile(within05_delta, [.025, .975])],
        "note": "Paired scene-cluster bootstrap; exploratory validation, not independent test.",
    }
    (OUT / "multisource_temporal_full_p90_bootstrap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
