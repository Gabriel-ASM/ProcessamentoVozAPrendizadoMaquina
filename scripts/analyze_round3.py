"""Small, deterministic post-processing for saved Round-3 CSV outputs."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round3"


def read_errors(path):
    scenes = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], row["scene_id"])
            scenes[key][int(row["source_index"])] = float(row["error_m"])
    return scenes


def bootstrap_pair(rows, baseline_name, candidate_name, repeats=10000, seed=20261108):
    baseline = defaultdict(list); candidate = defaultdict(list)
    for (method, scene), sources in rows.items():
        target = baseline if method == baseline_name else candidate if method == candidate_name else None
        if target is not None:
            target[scene] = [sources[i] for i in sorted(sources)]
    scenes = sorted(set(baseline) & set(candidate))
    delta_sum = np.asarray([sum(np.asarray(candidate[s]) - np.asarray(baseline[s])) for s in scenes], dtype=np.float64)
    source_count = np.asarray([len(baseline[s]) for s in scenes], dtype=np.float64)
    point = float(delta_sum.sum() / source_count.sum())
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, repeats, 1000):
        size = min(1000, repeats - start)
        idx = rng.integers(0, len(scenes), size=(size, len(scenes)))
        num = delta_sum[idx].sum(axis=1)
        den = source_count[idx].sum(axis=1)
        boot.extend((num / den).tolist())
    return {"baseline": baseline_name, "candidate": candidate_name, "mean_delta_m": point,
            "ci95_m": [float(np.quantile(boot,.025)), float(np.quantile(boot,.975))],
            "repeats": repeats, "unit": "scene", "negative_favors_candidate": True}


def main():
    rows = read_errors(OUT / "full_validation_source_errors.csv")
    result = {}
    for seed in (42,43):
        result[str(seed)] = {"against_coarse_baseline": {}, "against_refined_baseline": {}}
        candidates = {"candidate_coarse": f"h1_w2048_rect_seed{seed}",
                      "candidate_refined": f"h1_w2048_rect_seed{seed}_physical_local"}
        for label, baseline in (("against_coarse_baseline", f"round2_seed{seed}"),
                                ("against_refined_baseline", f"round2_seed{seed}_physical_local")):
            for candidate_label, candidate in candidates.items():
                result[str(seed)][label][candidate_label] = bootstrap_pair(
                    rows, baseline, candidate, seed=20261108+seed+len(label)+len(candidate_label))
    (OUT / "full_validation_paired_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
