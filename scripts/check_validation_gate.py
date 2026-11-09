"""Aplica uma regra explícita de inclusão da linha Wave baseada na validação."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["scene_id"]): row for row in payload["rows"]}
    if not rows:
        raise ValueError(f"resultado vazio: {path}")
    return rows


def _errors(rows: dict[str, dict]) -> list[float]:
    return [float(error) for row in rows.values() for error in row["source_errors_m"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path("artifacts/final/validation"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=Path("artifacts/final/wave_validation_gate.json"))
    args = parser.parse_args()
    per_seed = []
    for seed in args.seeds:
        root = args.validation_root / f"seed{seed}"
        srp = _read(root / "srp_validation.json")
        wave = _read(root / "wave_srp_validation.json")
        cnn_mix = _read(root / "cnn_mixture_validation.json")
        cnn_wave = _read(root / "cnn_wave_validation.json")
        scene_ids = set(srp)
        if any(set(rows) != scene_ids for rows in (wave, cnn_mix, cnn_wave)):
            raise ValueError(f"seeds {seed}: os métodos não compartilham as mesmas cenas de validação")
        for scene_id in scene_ids:
            expected_n = len(srp[scene_id]["source_errors_m"])
            if any(len(rows[scene_id]["source_errors_m"]) != expected_n for rows in (wave, cnn_mix, cnn_wave)):
                raise ValueError(f"N divergente na cena {scene_id}, seed {seed}")
        per_seed.append({
            "seed": seed,
            "scene_count": len(scene_ids),
            "srp_mean_error_m": float(np.mean(_errors(srp))),
            "wave_srp_mean_error_m": float(np.mean(_errors(wave))),
            "wave_srp_delta_m": float(np.mean(_errors(wave)) - np.mean(_errors(srp))),
            "cnn_mixture_mean_error_m": float(np.mean(_errors(cnn_mix))),
            "cnn_wave_mean_error_m": float(np.mean(_errors(cnn_wave))),
            "cnn_wave_delta_m": float(np.mean(_errors(cnn_wave)) - np.mean(_errors(cnn_mix))),
        })
    mean_delta = float(np.mean([row["wave_srp_delta_m"] for row in per_seed]))
    output = {
        "decision_rule": "incluir a Wave-U-Net como ramo auxiliar se a média do erro pareado por fonte de Wave-U-Net + SRP-PHAT for estritamente menor que a de SRP-PHAT na validação, agregada igualmente entre seeds",
        "wave_branch_passes_validation_gate": mean_delta < 0.0,
        "mean_wave_srp_delta_m_across_seeds": mean_delta,
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "interpretation": "A validação seleciona o ramo; não é evidência de desempenho no teste. A decisão e os checkpoints devem permanecer congelados antes de abrir o teste reservado.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
