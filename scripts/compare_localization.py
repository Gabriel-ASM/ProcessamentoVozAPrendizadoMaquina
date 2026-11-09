"""Bootstrap pareado por cena para relatórios de localização."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, list[float]] = {}
    for row in payload["rows"]:
        errors = row.get("source_errors_m")
        if errors is None:
            raise ValueError(f"resultado sem erros por fonte: {path} / {row.get('scene_id')}")
        output[str(row["scene_id"])] = [float(error) for error in errors]
    return output


def _summary(scene_errors: list[list[float]]) -> dict[str, float]:
    errors = np.asarray([error for scene in scene_errors for error in scene], dtype=np.float64)
    return {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_0_5m": float(np.mean(errors <= 0.5)),
    }


def _matrix(scene_errors: list[list[float]]) -> np.ndarray:
    width = max(len(errors) for errors in scene_errors)
    output = np.full((len(scene_errors), width), np.nan, dtype=np.float64)
    for index, errors in enumerate(scene_errors):
        output[index, : len(errors)] = errors
    return output


def _summary_array(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    return np.asarray(
        [
            np.mean(values),
            np.median(values),
            np.percentile(values, 90),
            np.mean(values <= 0.5),
        ],
        dtype=np.float64,
    )


def compare(reference_path: Path, candidate_path: Path, samples: int = 10_000, seed: int = 42) -> dict:
    reference, candidate = _rows(reference_path), _rows(candidate_path)
    if reference.keys() != candidate.keys():
        missing_reference = sorted(candidate.keys() - reference.keys())
        missing_candidate = sorted(reference.keys() - candidate.keys())
        raise ValueError(
            f"os métodos precisam avaliar as mesmas cenas; ausentes na referência={missing_reference[:5]}, "
            f"ausentes no candidato={missing_candidate[:5]}"
        )
    scene_ids = sorted(reference)
    reference_scenes = [reference[scene_id] for scene_id in scene_ids]
    candidate_scenes = [candidate[scene_id] for scene_id in scene_ids]
    metric_names = ("mean_error_m", "median_error_m", "p90_error_m", "within_0_5m")
    point_reference = _summary(reference_scenes)
    point_candidate = _summary(candidate_scenes)
    reference_matrix = _matrix(reference_scenes)
    candidate_matrix = _matrix(candidate_scenes)
    deltas = np.empty((samples, len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for sample in range(samples):
        indices = rng.integers(0, len(scene_ids), size=len(scene_ids))
        ref_metrics = _summary_array(reference_matrix[indices])
        cand_metrics = _summary_array(candidate_matrix[indices])
        deltas[sample] = cand_metrics - ref_metrics
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "scene_count": len(scene_ids),
        "source_count": sum(map(len, reference_scenes)),
        "bootstrap_samples": samples,
        "bootstrap_unit": "scene",
        "delta_direction": "candidate - reference",
        "reference_metrics": point_reference,
        "candidate_metrics": point_candidate,
        "delta_metrics": {
            name: {
                "point_estimate": point_candidate[name] - point_reference[name],
                "ci_95": np.percentile(deltas[:, index], [2.5, 97.5]).tolist(),
            }
            for index, name in enumerate(metric_names)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.samples < 100:
        parser.error("--samples deve ser >= 100")
    report = compare(args.reference, args.candidate, args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
