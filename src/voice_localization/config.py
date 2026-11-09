"""Configuração tipada e validação dos contratos do experimento."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class RoomConfig:
    dimensions: tuple[float, float, float] = (8.0, 6.0, 3.0)
    source_height: float = 1.5
    microphone_height: float = 1.5
    wall_margin: float = 0.5
    min_source_distance: float = 0.75
    min_source_microphone_distance: float = 0.5
    speed_of_sound: float = 343.0


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 2.0
    context_seconds: float = 0.25
    stft_window: int = 512
    stft_hop: int = 256
    fft_size: int = 1024
    active_rms_floor: float = 1e-4


@dataclass
class SimulationConfig:
    backend: str = "pyroomacoustics"
    rt60_range: tuple[float, float] = (0.2, 0.8)
    rt60_calibration_slope: float = 1.0
    rt60_calibration_intercept: float = 0.0
    snr_range_db: tuple[float, float] = (10.0, 30.0)
    n_sources: tuple[int, ...] = (1, 2, 3)
    n_microphones: tuple[int, ...] = (4, 6, 8)
    geometries: tuple[str, ...] = ("perimeter", "circle")
    room: RoomConfig = field(default_factory=RoomConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


@dataclass
class GridConfig:
    resolution_m: float = 0.10
    sigma_m: float = 0.15
    peak_suppression_radius_m: float = 0.30


@dataclass
class DatasetConfig:
    train_count: int = 24_000
    validation_count: int = 3_000
    test_count: int = 3_000
    seed: int = 42
    output_root: str = "data"
    corpus_root: str = "data/LibriSpeech"
    manifest_root: str = "data/manifests"
    audio_format: str = "wav"
    map_format: str = "npz"
    map_variant: str = "mixture"
    map_audio_variant: str = "mixture"
    save_clean_targets: bool = False
    generation_workers: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    run_root: str = "artifacts"
    positive_weight: float | None = None


@dataclass
class EnhancementConfig:
    max_microphones: int = 8
    base_channels: int = 8
    depth: int = 4
    batch_size: int = 1
    epochs: int = 5
    learning_rate: float = 5e-4


@dataclass
class ProjectConfig:
    profile: str = "smoke"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)

    def validate(self) -> None:
        sim = self.simulation
        room = sim.room
        audio = sim.audio
        if sim.backend not in {"pyroomacoustics", "anechoic"}:
            raise ValueError("simulation.backend deve ser 'pyroomacoustics' ou 'anechoic'")
        if sim.rt60_calibration_slope <= 0:
            raise ValueError("rt60_calibration_slope deve ser positivo")
        if len(room.dimensions) != 3 or any(v <= 0 for v in room.dimensions):
            raise ValueError("room.dimensions deve conter três valores positivos")
        if not (0 < room.source_height < room.dimensions[2]):
            raise ValueError("source_height deve estar dentro da sala")
        if not (0 < room.microphone_height < room.dimensions[2]):
            raise ValueError("microphone_height deve estar dentro da sala")
        if audio.sample_rate <= 0 or audio.duration_seconds <= 0:
            raise ValueError("sample_rate e duration_seconds devem ser positivos")
        if audio.stft_window <= 0 or audio.stft_hop <= 0 or audio.fft_size < audio.stft_window:
            raise ValueError("STFT inválida: window, hop e fft_size são incompatíveis")
        if any(n < 1 for n in sim.n_sources) or any(m < 2 for m in sim.n_microphones):
            raise ValueError("N deve ser >= 1 e M deve ser >= 2")
        if self.grid.resolution_m <= 0 or self.grid.sigma_m <= 0:
            raise ValueError("resolução e sigma da grade devem ser positivos")
        if self.dataset.train_count < 0 or self.dataset.validation_count < 0 or self.dataset.test_count < 0:
            raise ValueError("quantidades de cenas não podem ser negativas")
        if self.dataset.generation_workers < 1:
            raise ValueError("generation_workers deve ser >= 1")
        if self.dataset.map_audio_variant not in {"mixture", "clean", "enhanced"}:
            raise ValueError("dataset.map_audio_variant deve ser mixture, clean ou enhanced")
        if self.train.batch_size <= 0 or self.train.epochs <= 0:
            raise ValueError("batch_size e epochs devem ser positivos")
        if self.train.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("train.device deve ser auto, cpu ou cuda")
        if self.enhancement.max_microphones < max(sim.n_microphones):
            raise ValueError("enhancement.max_microphones deve cobrir M máximo")
        if self.enhancement.depth < 2 or self.enhancement.base_channels < 2:
            raise ValueError("Wave-U-Net exige depth >= 2 e base_channels >= 2")
        if self.enhancement.batch_size < 1 or self.enhancement.epochs < 1:
            raise ValueError("batch_size e epochs do Wave-U-Net devem ser positivos")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)


def _coerce_tuple(value: Any, item_type: type = float) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(item_type(v) for v in value)
    raise TypeError(f"esperada lista para configuração, recebido {type(value).__name__}")


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"parâmetro desconhecido: {key}")
        setattr(obj, key, value)
    return obj


def from_dict(values: dict[str, Any]) -> ProjectConfig:
    cfg = ProjectConfig()
    if "profile" in values:
        cfg.profile = str(values["profile"])
    sim_values = dict(values.get("simulation", {}))
    room_values = sim_values.pop("room", {})
    audio_values = sim_values.pop("audio", {})
    _merge_dataclass(cfg.simulation.room, room_values)
    _merge_dataclass(cfg.simulation.audio, audio_values)
    _merge_dataclass(cfg.simulation, sim_values)
    _merge_dataclass(cfg.grid, dict(values.get("grid", {})))
    _merge_dataclass(cfg.dataset, dict(values.get("dataset", {})))
    _merge_dataclass(cfg.train, dict(values.get("train", {})))
    _merge_dataclass(cfg.enhancement, dict(values.get("enhancement", {})))
    cfg.simulation.room.dimensions = _coerce_tuple(cfg.simulation.room.dimensions, float)
    cfg.simulation.n_sources = _coerce_tuple(cfg.simulation.n_sources, int)
    cfg.simulation.n_microphones = _coerce_tuple(cfg.simulation.n_microphones, int)
    cfg.simulation.geometries = _coerce_tuple(cfg.simulation.geometries, str)
    cfg.simulation.rt60_range = _coerce_tuple(cfg.simulation.rt60_range, float)
    cfg.simulation.snr_range_db = _coerce_tuple(cfg.simulation.snr_range_db, float)
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("PyYAML é necessário para carregar configurações YAML") from exc
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("o YAML de configuração deve conter um objeto na raiz")
    return from_dict(values)


def save_resolved_config(config: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.resolved_json() + "\n", encoding="utf-8")
