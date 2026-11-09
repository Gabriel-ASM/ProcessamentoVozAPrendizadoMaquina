"""STFT, GCC-PHAT e SRP-PHAT posicional em grade 2-D."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import grid_xy, pair_indices


@dataclass
class SpatialGrid:
    xs: np.ndarray
    ys: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)

    @property
    def points_xy(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.xs, self.ys, indexing="xy")
        return np.column_stack((xx.ravel(), yy.ravel()))


def make_grid(room_dimensions: tuple[float, float, float], resolution_m: float) -> SpatialGrid:
    xs, ys = grid_xy(room_dimensions, resolution_m)
    return SpatialGrid(xs, ys)


def frame_signal(signal: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Retorna frames ``(n_frames, window)`` com padding no fim."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("frame_signal espera sinal mono")
    if window <= 0 or hop <= 0:
        raise ValueError("window e hop devem ser positivos")
    if signal.size == 0:
        signal = np.zeros(window, dtype=np.float32)
    n_frames = max(1, int(np.ceil(max(signal.size - window, 0) / hop)) + 1)
    padded_size = (n_frames - 1) * hop + window
    padded = np.pad(signal, (0, max(0, padded_size - signal.size)))
    indices = np.arange(window)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[indices]


def gcc_phat_pairs(
    multichannel: np.ndarray,
    window: int = 512,
    hop: int = 256,
    fft_size: int = 1024,
    eps: float = 1e-8,
    phat_exponent: float = 1.0,
    window_type: str = "hann",
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Agrega GCC-PHAT por frames.

    O eixo de atraso é ``[-fft_size/2, fft_size/2)`` em amostras. Para o
    produto ``X_i * conj(X_j)``, atraso positivo significa que o canal i chega
    depois de j, consistente com ``distance_i - distance_j``.
    """
    audio = np.asarray(multichannel, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("multichannel deve ter forma (M, T)")
    mics, _ = audio.shape
    if fft_size < window:
        raise ValueError("fft_size precisa ser >= window")
    if window_type == "hann":
        analysis_window = np.hanning(window).astype(np.float32)
    elif window_type == "rectangular":
        analysis_window = np.ones(window, dtype=np.float32)
    else:
        raise ValueError("window_type deve ser 'hann' ou 'rectangular'")
    spectra = []
    for channel in audio:
        frames = frame_signal(channel, window, hop) * analysis_window[None, :]
        spectra.append(np.fft.rfft(frames, n=fft_size, axis=1))
    spectra = np.asarray(spectra)
    correlations: list[np.ndarray] = []
    pairs = pair_indices(mics)
    for i, j in pairs:
        cross = spectra[i] * np.conj(spectra[j])
        if phat_exponent < 0 or phat_exponent > 1:
            raise ValueError("phat_exponent deve estar entre 0 e 1")
        cross /= np.maximum(np.abs(cross), eps) ** phat_exponent
        corr = np.fft.irfft(cross, n=fft_size, axis=1)
        corr = np.fft.fftshift(corr, axes=1)
        correlations.append(np.mean(corr, axis=0).astype(np.float32))
    return np.asarray(correlations), pairs


def srp_phat_map(
    multichannel: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    room_dimensions: tuple[float, float, float],
    resolution_m: float = 0.10,
    speed_of_sound: float = 343.0,
    window: int = 512,
    hop: int = 256,
    fft_size: int = 1024,
    normalize: bool = False,
    phat_exponent: float = 1.0,
    window_type: str = "hann",
) -> tuple[np.ndarray, SpatialGrid]:
    """Calcula um mapa ``(grid_y, grid_x)`` de potência SRP-PHAT.

    A amostra de cada GCC é interpolada no atraso geométrico para cada ponto
    da grade. A posição z da grade é a altura média da matriz; a acústica pode
    ter sido simulada em 3-D, mas a saída é deliberadamente 2-D.
    """
    microphones = np.asarray(microphone_positions, dtype=np.float32)
    if microphones.ndim != 2 or microphones.shape[1] != 3:
        raise ValueError("microphone_positions deve ter forma (M, 3)")
    correlations, pairs = gcc_phat_pairs(
        multichannel, window, hop, fft_size,
        phat_exponent=phat_exponent, window_type=window_type,
    )
    grid = make_grid(room_dimensions, resolution_m)
    flat_map = score_srp_points(
        correlations, pairs, microphones, grid.points_xy, sample_rate,
        fft_size, speed_of_sound,
    ).reshape(grid.shape)
    if normalize:
        flat_map -= float(np.mean(flat_map))
        flat_map /= float(np.std(flat_map) + 1e-6)
    return flat_map, grid


def score_srp_points(
    correlations: np.ndarray,
    pairs: list[tuple[int, int]],
    microphone_positions: np.ndarray,
    points_xy: np.ndarray,
    sample_rate: int,
    fft_size: int,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Evaluate already-computed GCCs at arbitrary near-field (x,y) points."""
    microphones = np.asarray(microphone_positions, dtype=np.float64)
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy deve ter forma (P, 2)")
    points = np.column_stack((points_xy, np.full(len(points_xy), float(np.mean(microphones[:, 2])))))
    lag_samples = np.arange(-(fft_size // 2), fft_size // 2, dtype=np.float64)
    scores = np.zeros(len(points), dtype=np.float64)
    for correlation, (i, j) in zip(np.asarray(correlations), pairs):
        expected = (
            np.linalg.norm(points - microphones[i], axis=1)
            - np.linalg.norm(points - microphones[j], axis=1)
        ) * sample_rate / speed_of_sound
        scores += np.interp(expected, lag_samples, correlation, left=0.0, right=0.0)
    scores /= max(len(pairs), 1)
    return scores.astype(np.float32)


def target_occupancy(
    positions_xy: np.ndarray,
    grid: SpatialGrid,
    sigma_m: float = 0.15,
) -> np.ndarray:
    """Cria mapa de ocupação com gaussianas truncadas pelo máximo."""
    positions = np.asarray(positions_xy, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy deve ter forma (N, 2)")
    if sigma_m <= 0:
        raise ValueError("sigma_m deve ser positivo")
    points = grid.points_xy
    target = np.zeros(len(points), dtype=np.float32)
    denominator = 2.0 * sigma_m**2
    for position in positions:
        target = np.maximum(target, np.exp(-np.sum((points - position) ** 2, axis=1) / denominator))
    return target.reshape(grid.shape)
