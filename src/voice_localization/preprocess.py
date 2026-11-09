"""Extração de mapas SRP-PHAT e alvos a partir dos WAVs das cenas."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from dataclasses import asdict
import json
import hashlib
import time

import numpy as np

from .audio import read_multichannel_audio
from .config import ProjectConfig
from .manifest import SceneRecord, dataset_id, read_jsonl, write_jsonl
from .spatial import make_grid, srp_phat_map, target_occupancy


def processing_fingerprint(config: ProjectConfig, map_variant: str = "mixture", audio_variant: str = "mixture") -> str:
    payload = {
        "room_dimensions": config.simulation.room.dimensions,
        "microphone_height": config.simulation.room.microphone_height,
        "sample_rate": config.simulation.audio.sample_rate,
        "stft_window": config.simulation.audio.stft_window,
        "stft_hop": config.simulation.audio.stft_hop,
        "fft_size": config.simulation.audio.fft_size,
        "grid_resolution": config.grid.resolution_m,
        "target_sigma": config.grid.sigma_m,
        "speed_of_sound": config.simulation.room.speed_of_sound,
        "map_variant": map_variant,
        "audio_variant": audio_variant,
        "enhancement": asdict(config.enhancement) if audio_variant == "enhanced" else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _manifest_path(config: ProjectConfig, split: str, variant: str | None = None) -> Path:
    variant = variant or config.dataset.map_variant
    suffix = "" if variant == "mixture" else f"_{variant}"
    return Path(config.dataset.manifest_root) / f"scenes_{split}{suffix}.jsonl"


def extract_maps_for_split(
    config: ProjectConfig,
    split: str,
    audio_variant: str | None = None,
    map_variant: str | None = None,
) -> Path:
    audio_variant = audio_variant or config.dataset.map_audio_variant
    if audio_variant not in {"mixture", "clean", "enhanced"}:
        raise ValueError("audio_variant deve ser mixture, clean ou enhanced")
    map_variant = map_variant or config.dataset.map_variant
    source_manifest = Path(config.dataset.manifest_root) / f"scenes_{split}.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(f"manifesto de cenas não encontrado: {source_manifest}")
    records = list(read_jsonl(source_manifest, SceneRecord))
    output_dir = Path(config.dataset.output_root) / "maps" / map_variant / split
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(config.simulation.room.dimensions, config.grid.resolution_m)
    timings: list[dict] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        audio_path = {
            "mixture": record.audio_path,
            "clean": record.clean_audio_path,
            "enhanced": record.enhanced_audio_path,
        }[audio_variant]
        if not audio_path:
            raise ValueError(f"cena sem caminho de áudio {audio_variant}: {record.scene_id}")
        audio = read_multichannel_audio(audio_path, config.simulation.audio.sample_rate)
        mic_xy = np.asarray(record.microphone_positions_xy, dtype=np.float32)
        mic_positions = np.column_stack(
            (mic_xy, np.full(len(mic_xy), config.simulation.room.microphone_height, dtype=np.float32))
        )
        srp_map, _ = srp_phat_map(
            audio,
            mic_positions,
            config.simulation.audio.sample_rate,
            config.simulation.room.dimensions,
            resolution_m=config.grid.resolution_m,
            speed_of_sound=config.simulation.room.speed_of_sound,
            window=config.simulation.audio.stft_window,
            hop=config.simulation.audio.stft_hop,
            fft_size=config.simulation.audio.fft_size,
        )
        positions = np.asarray(record.source_positions_xy, dtype=np.float32)
        target = target_occupancy(positions, grid, config.grid.sigma_m)
        path = output_dir / f"{record.scene_id}.npz"
        np.savez_compressed(
            path,
            srp_map=srp_map.astype(np.float32),
            target=target.astype(np.float32),
            xs=grid.xs,
            ys=grid.ys,
            source_positions_xy=positions,
            n_sources=np.int64(record.n_sources),
            microphone_positions_xy=mic_xy,
            processing_fingerprint=np.asarray(processing_fingerprint(config, map_variant, audio_variant)),
            transform_id=np.asarray(
                record.metadata.get("enhancement_checkpoint_sha256", "") if audio_variant == "enhanced" else ""
            ),
        )
        records[index] = replace(record, map_path=str(path.resolve()), target_path=str(path.resolve()))
        timings.append({"scene_id": record.scene_id, "seconds": time.perf_counter() - started})
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            elapsed = sum(item["seconds"] for item in timings)
            print(f"[{split}/{map_variant}] mapas: {index + 1}/{len(records)}; CPU acumulada {elapsed:.1f} s", flush=True)
    manifest_path = _manifest_path(config, split, map_variant)
    write_jsonl(records, manifest_path)
    (output_dir.parent / f"timings_{split}.json").write_text(
        json.dumps(
            {
                "split": split,
                "scene_count": len(timings),
                "total_seconds": float(sum(item["seconds"] for item in timings)),
                "scenes": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def validate_map_files(config: ProjectConfig, split: str) -> dict:
    manifest_path = _manifest_path(config, split)
    records = list(read_jsonl(manifest_path, SceneRecord))
    expected_shape = make_grid(config.simulation.room.dimensions, config.grid.resolution_m).shape
    failures: list[str] = []
    for record in records:
        if not record.map_path or not Path(record.map_path).exists():
            failures.append(f"{record.scene_id}: map_path ausente")
            continue
        with np.load(record.map_path) as data:
            for key in ("srp_map", "target", "xs", "ys", "source_positions_xy", "n_sources", "processing_fingerprint", "transform_id"):
                if key not in data:
                    failures.append(f"{record.scene_id}: chave {key} ausente")
            if data["srp_map"].shape != expected_shape or data["target"].shape != expected_shape:
                failures.append(f"{record.scene_id}: shape inesperado")
            if int(data["n_sources"]) != record.n_sources:
                failures.append(f"{record.scene_id}: N inconsistente")
            if not np.all(np.isfinite(data["srp_map"])) or not np.all(np.isfinite(data["target"])):
                failures.append(f"{record.scene_id}: mapa não finito")
            if str(data["processing_fingerprint"]) != processing_fingerprint(
                config, config.dataset.map_variant, config.dataset.map_audio_variant
            ):
                failures.append(f"{record.scene_id}: cache produzido por configuração diferente")
            expected_transform_id = (
                record.metadata.get("enhancement_checkpoint_sha256", "")
                if config.dataset.map_audio_variant == "enhanced"
                else ""
            )
            if str(data["transform_id"]) != expected_transform_id:
                failures.append(f"{record.scene_id}: mapa gerado por checkpoint Wave-U-Net diferente")
    if failures:
        raise ValueError("falhas nos mapas: " + "; ".join(failures[:10]))
    return {"split": split, "scene_count": len(records), "shape": expected_shape}


def fit_map_normalizer(config: ProjectConfig) -> Path:
    """Calcula média/desvio global somente sobre os mapas de treino."""
    manifest_path = _manifest_path(config, "train")
    records = list(read_jsonl(manifest_path, SceneRecord))
    if not records:
        raise ValueError("não há cenas de treino para ajustar o normalizador")
    total = 0
    sum_values = 0.0
    sum_squares = 0.0
    for record in records:
        if not record.map_path:
            raise ValueError(f"cena de treino sem mapa: {record.scene_id}")
        with np.load(record.map_path) as data:
            values = np.asarray(data["srp_map"], dtype=np.float64)
        total += values.size
        sum_values += float(np.sum(values))
        sum_squares += float(np.sum(values * values))
    mean = sum_values / total
    variance = max(sum_squares / total - mean * mean, 1e-12)
    path = Path(config.dataset.output_root) / "maps" / config.dataset.map_variant / "normalizer.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=np.float32(mean),
        std=np.float32(np.sqrt(variance)),
        train_scene_count=len(records),
        train_dataset_id=np.asarray(dataset_id(records)),
        processing_fingerprint=np.asarray(
            processing_fingerprint(config, config.dataset.map_variant, config.dataset.map_audio_variant)
        ),
    )
    return path


def apply_map_normalizer(config: ProjectConfig, split: str, normalizer_path: str | Path | None = None) -> None:
    path = Path(normalizer_path or (Path(config.dataset.output_root) / "maps" / config.dataset.map_variant / "normalizer.npz"))
    if not path.exists():
        raise FileNotFoundError(f"normalizador não encontrado: {path}; ajuste-o com dados de treino")
    with np.load(path) as values:
        mean = float(values["mean"])
        std = float(values["std"])
        stored_fingerprint = str(values["processing_fingerprint"]) if "processing_fingerprint" in values else None
    expected_fingerprint = processing_fingerprint(
        config, config.dataset.map_variant, config.dataset.map_audio_variant
    )
    if stored_fingerprint and stored_fingerprint != expected_fingerprint:
        raise ValueError("normalizador ajustado para uma configuração de processamento diferente")
    manifest_path = _manifest_path(config, split)
    records = list(read_jsonl(manifest_path, SceneRecord))
    for record in records:
        if not record.map_path:
            raise ValueError(f"cena sem map_path: {record.scene_id}")
        map_path = Path(record.map_path)
        with np.load(map_path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        arrays["srp_map"] = ((arrays["srp_map"].astype(np.float32) - mean) / std).astype(np.float32)
        np.savez_compressed(map_path, **arrays)
