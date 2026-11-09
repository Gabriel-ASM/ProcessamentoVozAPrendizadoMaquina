import numpy as np

from voice_localization.metrics import decode_peaks, localization_metrics
from voice_localization.audio import measured_rt60
from voice_localization.spatial import gcc_phat_pairs, make_grid, target_occupancy


def test_target_and_peak_decoder_use_xy_convention():
    grid = make_grid((2.0, 1.0, 3.0), 0.5)
    target = target_occupancy(np.array([[1.0, 0.5]], dtype=np.float32), grid, sigma_m=0.1)
    prediction = decode_peaks(target, grid.xs, grid.ys, n_sources=1, suppression_radius_m=0.2)
    assert prediction.shape == (1, 2)
    assert np.allclose(prediction[0], [1.0, 0.5])


def test_gcc_phat_returns_one_curve_per_pair():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=2048).astype(np.float32)
    multichannel = np.stack([signal, np.roll(signal, 4), np.roll(signal, -3)])
    correlations, pairs = gcc_phat_pairs(multichannel, window=128, hop=64, fft_size=256)
    assert correlations.shape == (3, 256)
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert np.all(np.isfinite(correlations))


def test_metrics_are_invariant_to_source_order():
    reference = np.array([[1.0, 1.0], [5.0, 4.0]], dtype=np.float32)
    estimate = np.array([[5.1, 4.0], [0.9, 1.0]], dtype=np.float32)
    metrics = localization_metrics(reference, estimate, tolerance_m=0.2)
    np.testing.assert_allclose(metrics.mean_error_m, 0.1, atol=1e-6)
    assert metrics.within_tolerance == 1.0


def test_rt60_measurement_uses_decay_crossings():
    t = np.arange(16_000, dtype=np.float32) / 16_000
    impulse = np.exp(-t / 0.0724).astype(np.float32)
    measured = measured_rt60(impulse, 16_000)
    assert measured is not None
    assert 0.4 <= measured <= 0.7
