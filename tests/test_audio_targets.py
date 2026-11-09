"""Contratos de reproducibilidad y pares supervisados del simulador."""

from pathlib import Path

import numpy as np
import pytest

from voice_localization.audio import read_multichannel_audio
from voice_localization.config import load_config
from voice_localization.manifest import SceneRecord, read_jsonl
from voice_localization.simulation import SceneGenerator


def _single_scene(tmp_path: Path, name: str, backend: str = "anechoic") -> SceneRecord:
    config = load_config(Path("configs/wave_pilot.yaml"))
    config.simulation.backend = backend
    config.dataset.output_root = str(tmp_path / name / "data")
    config.dataset.manifest_root = str(tmp_path / name / "manifests")
    config.dataset.save_clean_targets = True
    SceneGenerator(config, synthetic_audio=True).generate_split("train", 1)
    return next(read_jsonl(Path(config.dataset.manifest_root) / "scenes_train.jsonl", SceneRecord))


def test_anechoic_scene_has_aligned_clean_target_and_is_repeatable(tmp_path):
    first = _single_scene(tmp_path, "first")
    second = _single_scene(tmp_path, "second")
    assert first.clean_audio_path and second.clean_audio_path
    np.testing.assert_array_equal(
        read_multichannel_audio(first.audio_path), read_multichannel_audio(second.audio_path)
    )
    np.testing.assert_array_equal(
        read_multichannel_audio(first.clean_audio_path), read_multichannel_audio(second.clean_audio_path)
    )
    mixture = read_multichannel_audio(first.audio_path)
    clean = read_multichannel_audio(first.clean_audio_path)
    assert mixture.shape == clean.shape == (first.n_microphones, 32_000)
    assert np.isfinite(mixture).all() and np.isfinite(clean).all()
    noise = mixture.astype(np.float64) - clean.astype(np.float64)
    measured = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    assert measured == pytest.approx(first.snr_realized_db, abs=0.02)


def test_pyroom_randomized_image_source_is_repeatable_and_restores_numpy_rng(tmp_path):
    pytest.importorskip("pyroomacoustics")
    np.random.seed(918)
    expected_state = np.random.get_state()
    first = _single_scene(tmp_path, "pyroom-first", backend="pyroomacoustics")
    state_after = np.random.get_state()
    for expected, actual in zip(expected_state, state_after):
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        else:
            assert expected == actual
    second = _single_scene(tmp_path, "pyroom-second", backend="pyroomacoustics")
    np.testing.assert_array_equal(
        read_multichannel_audio(first.audio_path), read_multichannel_audio(second.audio_path)
    )
    np.testing.assert_array_equal(
        read_multichannel_audio(first.clean_audio_path), read_multichannel_audio(second.clean_audio_path)
    )
