"""Render maps for the Streamlit demo without loading the PyTorch runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = json.loads(args.metadata.read_text(encoding="utf-8"))

    with np.load(args.input) as data:
        xs = data["xs"]
        ys = data["ys"]
        microphones = data["microphones"]
        truth = data["truth"]
        width, length = data["room_dimensions"][:2]
        for item in methods:
            index = int(item["index"])
            fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
            image = ax.imshow(
                data[f"map_{index}"],
                origin="lower",
                extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
                aspect="equal",
                cmap="viridis",
                alpha=0.78,
            )
            if item["result_key"].startswith("CNN") or item["result_key"].endswith("CNN"):
                colorbar_label = "Probabilidade CNN"
            elif "CNN" in item["result_key"]:
                colorbar_label = "Escore combinado SRP/CNN"
            else:
                colorbar_label = "Escore SRP-PHAT"
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
            estimate = data[f"estimate_{index}"]
            ax.scatter(microphones[:, 0], microphones[:, 1], marker="^", c="white", edgecolor="black", label="Microfones")
            ax.scatter(truth[:, 0], truth[:, 1], marker="x", c="red", s=95, linewidths=2, label="Posição real")
            ax.scatter(estimate[:, 0], estimate[:, 1], marker="o", facecolors="none", edgecolors="cyan", s=100, linewidths=2, label="Estimativa")
            ax.set_xlim(0, float(width))
            ax.set_ylim(0, float(length))
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(item["label"])
            ax.legend(loc="upper right", fontsize=8)
            fig.savefig(args.output_dir / f"method_{index}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
