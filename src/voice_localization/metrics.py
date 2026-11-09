"""Decodificação de picos e métricas invariantes à ordem das fontes."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class LocalizationMetrics:
    mean_error_m: float
    median_error_m: float
    p90_error_m: float
    within_tolerance: float
    source_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def decode_peaks(
    score_map: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    n_sources: int,
    suppression_radius_m: float = 0.30,
) -> np.ndarray:
    """Seleciona exatamente N máximos, com supressão de vizinhança em metros."""
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.shape != (len(ys), len(xs)):
        raise ValueError("score_map incompatível com xs/ys")
    if n_sources < 1:
        raise ValueError("n_sources deve ser positivo")
    flat_order = np.argsort(np.nan_to_num(scores, nan=-np.inf).ravel())[::-1]
    # Use explicitamente a mesma convenção de flatten row-major do mapa.
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    selected: list[np.ndarray] = []
    for index in flat_order:
        candidate = points[index]
        if all(np.linalg.norm(candidate - previous) >= suppression_radius_m for previous in selected):
            selected.append(candidate)
        if len(selected) == n_sources:
            break
    if len(selected) < n_sources:
        # Grade muito pequena ou raio muito grande: completamos sem duplicar a
        # interpretação do pico, mas mantemos o contrato de saída com N pontos.
        for index in flat_order:
            candidate = points[index]
            if not any(np.array_equal(candidate, previous) for previous in selected):
                selected.append(candidate)
            if len(selected) == n_sources:
                break
    if len(selected) != n_sources:
        raise ValueError("não há pontos suficientes na grade para decodificar N fontes")
    return np.asarray(selected, dtype=np.float32)


def decode_local_peaks(
    score_map: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    n_sources: int,
    suppression_radius_m: float = 0.30,
) -> np.ndarray:
    """Decode 3x3 local maxima, then apply spatial non-maximum suppression."""
    from scipy.ndimage import maximum_filter

    scores = np.asarray(score_map, dtype=np.float32)
    if scores.shape != (len(ys), len(xs)):
        raise ValueError("score_map incompatível com xs/ys")
    if n_sources < 1 or suppression_radius_m <= 0:
        raise ValueError("n_sources e suppression_radius_m devem ser positivos")
    local_mask = scores >= maximum_filter(scores, size=3, mode="nearest")
    candidate_indices = np.flatnonzero(local_mask.ravel())
    candidate_indices = candidate_indices[np.argsort(scores.ravel()[candidate_indices])[::-1]]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    selected = []
    for index in candidate_indices:
        point = points[index]
        if all(np.linalg.norm(point - previous) >= suppression_radius_m for previous in selected):
            selected.append(point)
            if len(selected) == n_sources:
                return np.asarray(selected, dtype=np.float32)
    for index in np.argsort(scores.ravel())[::-1]:
        point = points[index]
        if all(not np.array_equal(point, previous) for previous in selected):
            selected.append(point)
        if len(selected) == n_sources:
            break
    if len(selected) != n_sources:
        raise ValueError("não há pontos suficientes na grade para decodificar N fontes")
    return np.asarray(selected, dtype=np.float32)


def pairwise_assignment_errors(reference_xy: np.ndarray, estimate_xy: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_xy, dtype=np.float64)
    estimate = np.asarray(estimate_xy, dtype=np.float64)
    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("referência e estimativa devem ter a mesma forma (N, 2)")
    distances = np.linalg.norm(reference[:, None, :] - estimate[None, :, :], axis=-1)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(distances)
    except ImportError:
        rows, cols = _greedy_assignment(distances)
    errors = distances[rows, cols]
    return np.asarray(errors, dtype=np.float64)


def _greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    rows: list[int] = []
    cols: list[int] = []
    while remaining_rows:
        best = min(
            ((cost[r, c], r, c) for r in remaining_rows for c in remaining_cols),
            key=lambda item: item[0],
        )
        _, row, col = best
        rows.append(row)
        cols.append(col)
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    return np.asarray(rows), np.asarray(cols)


def localization_metrics(reference_xy: np.ndarray, estimate_xy: np.ndarray, tolerance_m: float = 0.50) -> LocalizationMetrics:
    errors = pairwise_assignment_errors(reference_xy, estimate_xy)
    if tolerance_m <= 0:
        raise ValueError("tolerance_m deve ser positivo")
    return LocalizationMetrics(
        mean_error_m=float(np.mean(errors)),
        median_error_m=float(np.median(errors)),
        p90_error_m=float(np.percentile(errors, 90)),
        within_tolerance=float(np.mean(errors <= tolerance_m)),
        source_count=int(len(errors)),
    )


def localization_errors(reference_xy: np.ndarray, estimate_xy: np.ndarray) -> np.ndarray:
    """Erros por fonte, após associação ótima e sem resumir por cena."""
    return pairwise_assignment_errors(reference_xy, estimate_xy)
