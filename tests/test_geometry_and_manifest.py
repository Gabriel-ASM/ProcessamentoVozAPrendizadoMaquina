import numpy as np
import pytest

from voice_localization.geometry import microphone_array, sample_source_positions
from voice_localization.manifest import SceneRecord, validate_scene_records


def test_microphones_and_sources_respect_constraints():
    microphones = microphone_array("circle", 6, (8.0, 6.0, 3.0), seed=1)
    sources = sample_source_positions(3, (8.0, 6.0, 3.0), microphones, seed=2)
    assert microphones.shape == (6, 3)
    assert sources.shape == (3, 3)
    assert np.all((sources[:, 0] >= 0.5) & (sources[:, 0] <= 7.5))
    assert np.all((sources[:, 1] >= 0.5) & (sources[:, 1] <= 5.5))
    assert np.allclose(sources[:, 2], 1.5)
    distances = np.linalg.norm(sources[:, None, :2] - sources[None, :, :2], axis=-1)
    assert np.min(distances[np.triu_indices(3, 1)]) >= 0.75


def _record(scene_id: str, split: str, speaker: str) -> SceneRecord:
    return SceneRecord(
        scene_id=scene_id,
        split=split,
        seed=1,
        audio_ids=[scene_id],
        speaker_ids=[speaker],
        n_sources=1,
        n_microphones=4,
        geometry="circle",
        room_dimensions=[8.0, 6.0, 3.0],
        source_positions_xy=[[2.0, 2.0]],
        microphone_positions_xy=[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        rt60_requested=0.3,
        snr_requested_db=20.0,
    )


def test_manifest_rejects_speaker_leakage():
    with pytest.raises(ValueError, match="falantes"):
        validate_scene_records([_record("a", "train", "speaker-1"), _record("b", "test", "speaker-1")])

