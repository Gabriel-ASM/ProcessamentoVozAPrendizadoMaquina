"""Confere se um run parcial/concluído corresponde à configuração atual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_localization.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", default="mixture")
    args = parser.parse_args()
    if not args.saved.exists():
        parser.error(f"config.json do run não existe: {args.saved}")
    expected = load_config(args.config)
    expected.train.seed = args.seed
    expected.dataset.map_variant = args.variant
    actual = json.loads(args.saved.read_text(encoding="utf-8"))
    if actual != json.loads(expected.resolved_json()):
        print(f"Configuração do run não corresponde a {args.config}, seed={args.seed}, variant={args.variant}", file=sys.stderr)
        return 2
    print("configuração compatível")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
