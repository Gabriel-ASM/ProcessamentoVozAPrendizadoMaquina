"""Geração determinística de cenas multicanais e respectivos manifestos."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .audio import (
    add_noise_at_snr,
    anechoic_clean_mix,
    fit_window,
    normalize_active,
    read_audio,
    synthetic_speech,
)
from .config import ProjectConfig
from .geometry import microphone_array, sample_source_positions
from .manifest import AudioRecord, SceneRecord, dataset_id, read_jsonl, write_jsonl


GENERATOR_VERSION = "0.2.5"


@dataclass
class GeneratedSplit:
    split: str
    count: int
    manifest_path: Path


class SceneGenerator:
    def __init__(
        self,
        config: ProjectConfig,
        audio_records: list[AudioRecord] | None = None,
        synthetic_audio: bool = False,
    ) -> None:
        config.validate()
        self.config = config
        self.audio_records = audio_records or []
        self.synthetic_audio = synthetic_audio
        if not self.audio_records and not synthetic_audio:
            raise ValueError("audio_records é obrigatório quando synthetic_audio=False")

    @property
    def scene_root(self) -> Path:
        return Path(self.config.dataset.output_root) / "scenes"

    def generate_split(self, split: str, count: int) -> GeneratedSplit:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split desconhecido: {split}")
        if count < 0:
            raise ValueError("count não pode ser negativo")
        out_dir = self.scene_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[SceneRecord] = []
        existing_by_index: dict[int, SceneRecord] = {}
        manifest_path = Path(self.config.dataset.manifest_root) / f"scenes_{split}.jsonl"
        if manifest_path.exists():
            for existing in read_jsonl(manifest_path, SceneRecord):
                try:
                    existing_index = int(existing.scene_id.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                clean_present = not self.config.dataset.save_clean_targets or (
                    existing.clean_audio_path and Path(existing.clean_audio_path).exists()
                )
                if (
                    existing.metadata.get("simulation_fingerprint") == self.simulation_fingerprint
                    and existing.audio_path
                    and Path(existing.audio_path).exists()
                    and clean_present
                ):
                    existing_by_index[existing_index] = existing
        # A killed process may have finished individual WAV/JSON files before
        # the split manifest was committed. Recover valid scene sidecars so a
        # resumed full build does not regenerate those rooms and waveforms.
        for index in range(count):
            if index in existing_by_index:
                continue
            scene_id = f"{split}-{index:06d}"
            metadata_path = out_dir / f"{scene_id}.json"
            if not metadata_path.exists():
                continue
            try:
                existing = SceneRecord(**json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            expected_audio = (out_dir / f"{scene_id}.wav").resolve()
            expected_clean = (out_dir / f"{scene_id}.clean.wav").resolve()
            clean_present = not self.config.dataset.save_clean_targets or (
                existing.clean_audio_path
                and Path(existing.clean_audio_path).resolve() == expected_clean
                and expected_clean.exists()
            )
            if (
                existing.scene_id == scene_id
                and existing.split == split
                and existing.metadata.get("simulation_fingerprint") == self.simulation_fingerprint
                and existing.audio_path
                and Path(existing.audio_path).resolve() == expected_audio
                and expected_audio.exists()
                and clean_present
            ):
                existing_by_index[index] = existing
        pending = [index for index in range(count) if index not in existing_by_index]
        failures_path = Path(self.config.dataset.manifest_root) / f"generation_failures_{split}.jsonl"
        if pending and self.config.dataset.generation_workers > 1:
            with ProcessPoolExecutor(
                max_workers=self.config.dataset.generation_workers,
                initializer=_initialize_worker,
                initargs=(self.config, self.audio_records, self.synthetic_audio),
            ) as executor:
                try:
                    generated = executor.map(
                        _generate_scene_worker,
                        [(split, index, str(out_dir)) for index in pending],
                    )
                    recovered_count = len(existing_by_index)
                    for position, (index, result) in enumerate(zip(pending, generated), start=1):
                        existing_by_index[index] = result
                        if position % 500 == 0 or position == len(pending):
                            print(
                                f"[{split}] cenas válidas: {recovered_count + position}/{count}",
                                flush=True,
                            )
                except Exception as exc:
                    _append_generation_failure(failures_path, split, -1, exc)
                    raise
        else:
            for index in pending:
                try:
                    existing_by_index[index] = self._generate_scene(split, index, out_dir)
                    completed = len(existing_by_index)
                    if completed % 500 == 0 or completed == count:
                        print(f"[{split}] cenas válidas: {completed}/{count}", flush=True)
                except Exception as exc:
                    _append_generation_failure(failures_path, split, index, exc)
                    raise
        records = [existing_by_index[index] for index in range(count)]
        write_jsonl(records, manifest_path)
        metadata_path = Path(self.config.dataset.manifest_root) / f"generation_{split}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "scene_count": len(records),
                    "dataset_id": dataset_id(records),
                    "simulation_fingerprint": self.simulation_fingerprint,
                    "seed": self.config.dataset.seed,
                    "generator_version": GENERATOR_VERSION,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return GeneratedSplit(split, count, manifest_path)

    def generate_all(self) -> list[GeneratedSplit]:
        counts = {
            "train": self.config.dataset.train_count,
            "validation": self.config.dataset.validation_count,
            "test": self.config.dataset.test_count,
        }
        return [self.generate_split(split, count) for split, count in counts.items()]

    @property
    def simulation_fingerprint(self) -> str:
        pra_version = None
        if self.config.simulation.backend == "pyroomacoustics":
            try:
                import pyroomacoustics as pra

                pra_version = pra.__version__
            except ImportError:
                pra_version = None
        payload = {
            "profile": self.config.profile,
            "simulation": self.config.simulation,
            "dataset_seed": self.config.dataset.seed,
            "generator_version": GENERATOR_VERSION,
            "synthetic_audio": self.synthetic_audio,
            "audio_sources": sorted(
                (record.audio_id, record.speaker_id, record.split, record.path)
                for record in self.audio_records
            ),
            "pyroomacoustics_version": pra_version,
        }
        return hashlib.sha256(json.dumps(_as_serializable(payload), sort_keys=True).encode()).hexdigest()

    def _generate_scene(self, split: str, index: int, out_dir: Path) -> SceneRecord:
        seed = self.config.dataset.seed + _split_offset(split) + index
        rng = np.random.default_rng(seed)
        sim = self.config.simulation
        room = sim.room
        # Deterministic round-robin gives every pilot/full split balanced
        # coverage of N × M × geometry, rather than relying on random draws.
        condition_index = index
        n_sources = int(sim.n_sources[condition_index % len(sim.n_sources)])
        n_mics = int(sim.n_microphones[(condition_index // len(sim.n_sources)) % len(sim.n_microphones)])
        geometry_index = (condition_index // (len(sim.n_sources) * len(sim.n_microphones))) % len(sim.geometries)
        geometry = str(sim.geometries[geometry_index])
        rt60 = float(rng.uniform(*sim.rt60_range))
        snr_db = float(rng.uniform(*sim.snr_range_db))
        microphones = microphone_array(
            geometry,
            n_mics,
            room.dimensions,
            seed,
            margin=room.wall_margin,
            height=room.microphone_height,
        )
        source_positions = sample_source_positions(
            n_sources,
            room.dimensions,
            microphones,
            seed + 17,
            wall_margin=room.wall_margin,
            min_source_distance=room.min_source_distance,
            min_source_microphone_distance=room.min_source_microphone_distance,
            height=room.source_height,
        )
        sources, audio_ids, speaker_ids, starts = self._source_signals(
            n_sources, split, index, rng, sim.audio.sample_rate
        )
        if sim.backend == "pyroomacoustics":
            clean, rt60_realized, rt60_sabine_input = self._pyroom_mix(
                sources, source_positions, microphones, rt60, seed
            )
        else:
            clean = anechoic_clean_mix(
                sources,
                source_positions,
                microphones,
                sim.audio.sample_rate,
                room.speed_of_sound,
            )
            rt60_realized = 0.0
            rt60_sabine_input = None
        n_samples = round(sim.audio.sample_rate * sim.audio.duration_seconds)
        crop_start = round(sim.audio.sample_rate * sim.audio.context_seconds)
        clean = clean[:, crop_start : crop_start + n_samples]
        if clean.shape[1] < n_samples:
            clean = np.pad(clean, ((0, 0), (0, n_samples - clean.shape[1])))
        clean = clean.astype(np.float32)
        if not np.all(np.isfinite(clean)):
            raise ValueError(f"áudio não finito na cena {split}-{index:06d}")
        # Set SNR on the exact crop that is written to disk; speech power can
        # vary across the discarded context and the retained scene.
        mixed, snr_realized = add_noise_at_snr(clean, snr_db, rng)
        if not np.all(np.isfinite(mixed)):
            raise ValueError(f"mistura não finita na cena {split}-{index:06d}")
        scene_id = f"{split}-{index:06d}"
        audio_path = out_dir / f"{scene_id}.wav"
        clean_audio_path = out_dir / f"{scene_id}.clean.wav" if self.config.dataset.save_clean_targets else None
        _write_multichannel_wav(audio_path, mixed, sim.audio.sample_rate)
        if clean_audio_path is not None:
            _write_multichannel_wav(clean_audio_path, clean, sim.audio.sample_rate)
        record = SceneRecord(
            scene_id=scene_id,
            split=split,
            seed=seed,
            audio_ids=audio_ids,
            speaker_ids=speaker_ids,
            n_sources=n_sources,
            n_microphones=n_mics,
            geometry=geometry,
            room_dimensions=list(room.dimensions),
            source_positions_xy=source_positions[:, :2].tolist(),
            microphone_positions_xy=microphones[:, :2].tolist(),
            rt60_requested=rt60,
            snr_requested_db=snr_db,
            rt60_realized=rt60_realized,
            snr_realized_db=snr_realized,
            audio_path=str(audio_path.resolve()),
            clean_audio_path=str(clean_audio_path.resolve()) if clean_audio_path else None,
            generator_version=GENERATOR_VERSION,
            metadata={
                "source_window_starts": starts,
                "simulation_fingerprint": self.simulation_fingerprint,
                "rt60_sabine_input": rt60_sabine_input,
                "rt60_measurement": "Pyroomacoustics Room.measure_rt60(decay_db=30)" if rt60_sabine_input else "anechoic",
            },
        )
        metadata_path = out_dir / f"{scene_id}.json"
        temporary_metadata = metadata_path.with_suffix(".json.part")
        temporary_metadata.write_text(json.dumps(record.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary_metadata.replace(metadata_path)
        return record

    def _source_signals(
        self, count: int, split: str, scene_index: int, rng: np.random.Generator, sample_rate: int
    ) -> tuple[list[np.ndarray], list[str], list[str], list[int]]:
        duration = self.config.simulation.audio.duration_seconds + self.config.simulation.audio.context_seconds
        samples = round(duration * sample_rate)
        if self.synthetic_audio:
            signals = [synthetic_speech(int(rng.integers(0, 2**31 - 1)), sample_rate, duration) for _ in range(count)]
            return signals, [f"synthetic-{split}-{scene_index}-{i}" for i in range(count)], [f"synthetic-{split}-{i}" for i in range(count)], [0] * count
        by_speaker: dict[str, list[AudioRecord]] = {}
        for record in self.audio_records:
            by_speaker.setdefault(record.speaker_id, []).append(record)
        if len(by_speaker) < count:
            raise ValueError("o manifesto não contém falantes distintos suficientes para uma cena")
        selected_speakers = rng.choice(list(by_speaker), size=count, replace=False)
        signals: list[np.ndarray] = []
        audio_ids: list[str] = []
        speaker_ids: list[str] = []
        starts: list[int] = []
        for speaker in selected_speakers:
            speaker_records = by_speaker[str(speaker)]
            accepted = None
            attempts = max(8, min(64, 2 * len(speaker_records)))
            for _ in range(attempts):
                record = speaker_records[int(rng.integers(len(speaker_records)))]
                raw_signal = read_audio(record.path, sample_rate)
                raw_rms = float(np.sqrt(np.mean(np.square(raw_signal, dtype=np.float64)))) if raw_signal.size else 0.0
                if raw_rms <= self.config.simulation.audio.active_rms_floor:
                    continue
                signal = normalize_active(raw_signal)
                for _ in range(16):
                    window, start = fit_window(signal, samples, rng)
                    window_rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) if window.size else 0.0
                    if window_rms > self.config.simulation.audio.active_rms_floor:
                        accepted = (record, window, start)
                        break
                if accepted is not None:
                    break
            if accepted is None:
                raise ValueError(f"não foi possível encontrar recorte ativo para o falante {speaker}")
            record, window, start = accepted
            signals.append(window)
            audio_ids.append(record.audio_id)
            speaker_ids.append(record.speaker_id)
            starts.append(start)
        return signals, audio_ids, speaker_ids, starts

    def _pyroom_mix(
        self,
        sources: list[np.ndarray],
        source_positions: np.ndarray,
        microphones: np.ndarray,
        rt60: float,
        seed: int,
    ) -> tuple[np.ndarray, float | None, float]:
        try:
            import pyroomacoustics as pra
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Pyroomacoustics não está instalado; use backend anechoic somente para smoke"
            ) from exc
        sim = self.config.simulation
        # Calibrate the Sabine parameter to the realized RT60 measured on the
        # training pilot for the installed Pyroomacoustics version and room.
        sabine_rt60 = max(
            (rt60 - sim.rt60_calibration_intercept) / sim.rt60_calibration_slope,
            0.05,
        )
        # Seed before constructing the room: Pyroomacoustics can use its own
        # RNG while checking source and microphone positions, as well as while
        # simulating randomized image sources.
        global_state = np.random.get_state()
        pra_random = getattr(pra, "random", None)
        pra_rng_state = None
        if pra_random is not None and hasattr(pra_random, "get_rng"):
            pra_rng_state = copy.deepcopy(pra_random.get_rng().bit_generator.state)
        try:
            np.random.seed(seed)
            if pra_random is not None and hasattr(pra_random, "seed"):
                pra_random.seed(numpy=seed, libroom=seed)
            absorption, max_order = pra.inverse_sabine(sabine_rt60, list(sim.room.dimensions))
            room = pra.ShoeBox(
                list(sim.room.dimensions),
                fs=sim.audio.sample_rate,
                materials=pra.Material(absorption),
                max_order=max_order,
                use_rand_ism=True,
            )
            for position, signal in zip(source_positions, sources):
                room.add_source(position.tolist(), signal=signal)
            room.add_microphone_array(microphones.T)
            room.simulate()
            clean = np.asarray(room.mic_array.signals, dtype=np.float32)
            measured = np.asarray(room.measure_rt60(decay_db=30), dtype=np.float64)
        finally:
            if pra_rng_state is not None:
                pra_random.get_rng().bit_generator.state = pra_rng_state
            np.random.set_state(global_state)
        valid_rt60 = measured[np.isfinite(measured) & (measured > 0)]
        rt60_realized = float(np.median(valid_rt60)) if valid_rt60.size else None
        return clean, rt60_realized, float(sabine_rt60)


def _split_offset(split: str) -> int:
    return {"train": 0, "validation": 10_000_000, "test": 30_000_000}[split]


_WORKER_GENERATOR: SceneGenerator | None = None


def _initialize_worker(config: ProjectConfig, audio_records: list[AudioRecord], synthetic_audio: bool) -> None:
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = SceneGenerator(config, audio_records, synthetic_audio)


def _generate_scene_worker(payload: tuple[str, int, str]) -> SceneRecord:
    if _WORKER_GENERATOR is None:  # pragma: no cover - only reachable on broken process setup
        raise RuntimeError("worker de geração não inicializado")
    split, index, out_dir = payload
    return _WORKER_GENERATOR._generate_scene(split, index, Path(out_dir))


def _append_generation_failure(path: Path, split: str, index: int, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"split": split, "index": index, "error_type": type(error).__name__, "error": str(error)},
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_multichannel_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        temporary = path.with_suffix(path.suffix + ".part")
        sf.write(temporary, audio.T, sample_rate, format="WAV", subtype="FLOAT")
        temporary.replace(path)
        return
    except ImportError:
        pass
    import wave

    clipped = np.clip(audio.T, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    temporary = path.with_suffix(path.suffix + ".part")
    with wave.open(str(temporary), "wb") as wav:
        wav.setnchannels(audio.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    temporary.replace(path)


def _as_serializable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _as_serializable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_as_serializable(item) for item in value]
    if isinstance(value, list):
        return [_as_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _as_serializable(item) for key, item in value.items()}
    return value
