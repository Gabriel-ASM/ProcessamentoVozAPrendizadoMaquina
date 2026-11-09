from pathlib import Path

from voice_localization.config import load_config
from voice_localization.evaluation import evaluate_srp
from voice_localization.preprocess import extract_maps_for_split, validate_map_files
from voice_localization.simulation import SceneGenerator


def test_small_pipeline_generates_auditable_maps(tmp_path):
    config = load_config(Path("configs/smoke.yaml"))
    config.dataset.output_root = str(tmp_path / "data")
    config.dataset.manifest_root = str(tmp_path / "manifests")
    config.dataset.train_count = 0
    config.dataset.validation_count = 0
    config.dataset.test_count = 2
    generator = SceneGenerator(config, synthetic_audio=True)
    generator.generate_split("test", 2)
    extract_maps_for_split(config, "test")
    report = validate_map_files(config, "test")
    assert report["scene_count"] == 2
    summary = evaluate_srp(config, "test")
    assert summary["scene_count"] == 2
    assert summary["mean_error_m"] >= 0


def test_generation_recovers_completed_scene_sidecars_after_manifest_loss(tmp_path):
    config = load_config(Path("configs/smoke.yaml"))
    config.dataset.output_root = str(tmp_path / "data")
    config.dataset.manifest_root = str(tmp_path / "manifests")
    config.dataset.save_clean_targets = True
    generator = SceneGenerator(config, synthetic_audio=True)
    generated = generator.generate_split("test", 2)
    generated.manifest_path.unlink()

    def must_not_regenerate(*args, **kwargs):
        raise AssertionError("uma cena válida recuperada do sidecar não deve ser simulada novamente")

    generator._generate_scene = must_not_regenerate
    resumed = generator.generate_split("test", 2)
    assert resumed.count == 2
    assert resumed.manifest_path.exists()
    assert len(resumed.manifest_path.read_text(encoding="utf-8").splitlines()) == 2
