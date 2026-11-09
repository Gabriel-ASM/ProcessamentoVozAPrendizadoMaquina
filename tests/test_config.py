from pathlib import Path

import pytest

from voice_localization.config import from_dict, load_config


def test_smoke_config_is_valid():
    config = load_config(Path("configs/smoke.yaml"))
    assert config.simulation.n_sources == (1, 2, 3)
    assert config.simulation.n_microphones == (4, 6, 8)
    assert config.simulation.audio.stft_window == 512


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        from_dict({"simulation": {"backend": "unknown"}})

