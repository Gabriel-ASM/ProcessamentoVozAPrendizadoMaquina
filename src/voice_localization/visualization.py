"""Visualização intermediária de mapas, posições e métricas."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_scene(
    output_path: str | Path,
    room_dimensions: tuple[float, float, float],
    microphone_positions_xy: np.ndarray,
    reference_xy: np.ndarray,
    srp_map: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    predicted_xy: np.ndarray | None = None,
    title: str = "Localização de fontes",
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(
        srp_map,
        origin="lower",
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        aspect="equal",
        cmap="viridis",
    )
    axes[0].scatter(microphone_positions_xy[:, 0], microphone_positions_xy[:, 1], marker="^", c="white", edgecolor="black", label="microfones")
    axes[0].scatter(reference_xy[:, 0], reference_xy[:, 1], marker="x", c="red", s=80, label="referência")
    if predicted_xy is not None:
        axes[0].scatter(predicted_xy[:, 0], predicted_xy[:, 1], marker="o", facecolors="none", edgecolors="cyan", s=90, label="estimativa")
    axes[0].set(title="Mapa SRP-PHAT", xlabel="x (m)", ylabel="y (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(0, room_dimensions[0])
    axes[1].set_ylim(0, room_dimensions[1])
    axes[1].set_aspect("equal")
    axes[1].scatter(microphone_positions_xy[:, 0], microphone_positions_xy[:, 1], marker="^", c="tab:blue", label="microfones")
    axes[1].scatter(reference_xy[:, 0], reference_xy[:, 1], marker="x", c="tab:red", s=90, label="referência")
    if predicted_xy is not None:
        axes[1].scatter(predicted_xy[:, 0], predicted_xy[:, 1], marker="o", facecolors="none", edgecolors="tab:green", s=90, label="estimativa")
    axes[1].set(title=title, xlabel="x (m)", ylabel="y (m)")
    axes[1].title.set_fontsize(10)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

