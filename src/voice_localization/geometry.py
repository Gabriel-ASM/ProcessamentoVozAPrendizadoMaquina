"""Geometrias de microfones e amostragem de posições."""

from __future__ import annotations

import numpy as np


def microphone_array(
    geometry: str,
    count: int,
    room_dimensions: tuple[float, float, float],
    seed: int,
    margin: float = 0.5,
    height: float = 1.5,
) -> np.ndarray:
    """Retorna posições no formato ``(M, 3)``.

    ``perimeter`` usa pontos uniformemente distribuídos no perímetro do
    retângulo interno. ``circle`` usa uma circunferência centrada na sala.
    """
    if count < 2:
        raise ValueError("uma matriz precisa de pelo menos dois microfones")
    width, length, room_height = room_dimensions
    if not margin < min(width, length) / 2:
        raise ValueError("margin deixa a área interna vazia")
    if not 0 < height < room_height:
        raise ValueError("altura dos microfones fora da sala")
    rng = np.random.default_rng(seed)
    if geometry == "circle":
        radius = min(width, length) / 3.0
        angles = 2 * np.pi * np.arange(count) / count + rng.uniform(0, 2 * np.pi)
        xy = np.column_stack((width / 2 + radius * np.cos(angles), length / 2 + radius * np.sin(angles)))
    elif geometry == "perimeter":
        x0, x1 = margin, width - margin
        y0, y1 = margin, length - margin
        perimeter = 2 * ((x1 - x0) + (y1 - y0))
        distances = (np.arange(count) + 0.5) * perimeter / count
        xy = np.array([_point_on_rectangle(float(d), x0, x1, y0, y1) for d in distances])
        rng.shuffle(xy)
    else:
        raise ValueError(f"geometria desconhecida: {geometry}")
    return np.column_stack((xy, np.full(count, height))).astype(np.float32)


def _point_on_rectangle(distance: float, x0: float, x1: float, y0: float, y1: float) -> tuple[float, float]:
    a, b, c, d = x1 - x0, y1 - y0, x1 - x0, y1 - y0
    if distance < a:
        return x0 + distance, y0
    distance -= a
    if distance < b:
        return x1, y0 + distance
    distance -= b
    if distance < c:
        return x1 - distance, y1
    distance -= c
    return x0, y1 - min(distance, d)


def sample_source_positions(
    count: int,
    room_dimensions: tuple[float, float, float],
    microphone_positions: np.ndarray,
    seed: int,
    wall_margin: float = 0.5,
    min_source_distance: float = 0.75,
    min_source_microphone_distance: float = 0.5,
    height: float = 1.5,
    max_attempts: int = 10_000,
) -> np.ndarray:
    if count < 1:
        raise ValueError("é necessário pelo menos uma fonte")
    width, length, room_height = room_dimensions
    if not 0 < height < room_height:
        raise ValueError("altura das fontes fora da sala")
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    for _ in range(max_attempts):
        candidate = np.array(
            [
                rng.uniform(wall_margin, width - wall_margin),
                rng.uniform(wall_margin, length - wall_margin),
                height,
            ],
            dtype=np.float32,
        )
        if any(np.linalg.norm(candidate[:2] - other[:2]) < min_source_distance for other in accepted):
            continue
        if microphone_positions.size and np.min(np.linalg.norm(microphone_positions - candidate, axis=1)) < min_source_microphone_distance:
            continue
        accepted.append(candidate)
        if len(accepted) == count:
            return np.stack(accepted)
    raise RuntimeError("não foi possível posicionar todas as fontes dentro do limite de tentativas")


def grid_xy(room_dimensions: tuple[float, float, float], resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    if resolution_m <= 0:
        raise ValueError("resolution_m deve ser positivo")
    width, length, _ = room_dimensions
    xs = np.arange(0.0, width + resolution_m / 2, resolution_m, dtype=np.float32)
    ys = np.arange(0.0, length + resolution_m / 2, resolution_m, dtype=np.float32)
    return xs, ys


def pair_indices(count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(count) for j in range(i + 1, count)]
