"""Manifestos JSONL e verificações de separação do corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class AudioRecord:
    audio_id: str
    path: str
    split: str
    speaker_id: str
    sample_rate: int
    duration_seconds: float
    transcript_path: str | None = None


@dataclass
class SceneRecord:
    scene_id: str
    split: str
    seed: int
    audio_ids: list[str]
    speaker_ids: list[str]
    n_sources: int
    n_microphones: int
    geometry: str
    room_dimensions: list[float]
    source_positions_xy: list[list[float]]
    microphone_positions_xy: list[list[float]]
    rt60_requested: float
    snr_requested_db: float
    rt60_realized: float | None = None
    snr_realized_db: float | None = None
    audio_path: str | None = None
    clean_audio_path: str | None = None
    enhanced_audio_path: str | None = None
    map_path: str | None = None
    target_path: str | None = None
    generator_version: str = "0.1.0"
    metadata: dict = field(default_factory=dict)

    @property
    def recipe_hash(self) -> str:
        payload = {
            "audio_ids": self.audio_ids,
            "speaker_ids": self.speaker_ids,
            "seed": self.seed,
            "n_sources": self.n_sources,
            "n_microphones": self.n_microphones,
            "geometry": self.geometry,
            "room_dimensions": self.room_dimensions,
            "source_positions_xy": self.source_positions_xy,
            "microphone_positions_xy": self.microphone_positions_xy,
            "rt60_requested": self.rt60_requested,
            "snr_requested_db": self.snr_requested_db,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_jsonl(records: Iterable[AudioRecord | SceneRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path, record_type: type = SceneRecord) -> Iterator:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = json.loads(line)
            try:
                yield record_type(**values)
            except TypeError as exc:
                raise ValueError(f"registro inválido em {path}:{line_no}: {exc}") from exc


def validate_scene_records(records: Iterable[SceneRecord], allowed_splits: set[str] | None = None) -> dict:
    records = list(records)
    if not records:
        raise ValueError("manifesto de cenas vazio")
    seen_ids: set[str] = set()
    seen_recipes: dict[str, str] = {}
    split_speakers: dict[str, set[str]] = {}
    for rec in records:
        if rec.scene_id in seen_ids:
            raise ValueError(f"scene_id duplicado: {rec.scene_id}")
        seen_ids.add(rec.scene_id)
        if allowed_splits and rec.split not in allowed_splits:
            raise ValueError(f"split inválido: {rec.split}")
        if rec.n_sources != len(rec.source_positions_xy) or rec.n_sources != len(rec.audio_ids):
            raise ValueError(f"N inconsistente na cena {rec.scene_id}")
        if rec.n_sources != len(rec.speaker_ids):
            raise ValueError(f"speaker_ids inconsistente na cena {rec.scene_id}")
        if rec.n_microphones != len(rec.microphone_positions_xy):
            raise ValueError(f"M inconsistente na cena {rec.scene_id}")
        width, length = rec.room_dimensions[:2]
        positions = rec.source_positions_xy + rec.microphone_positions_xy
        if any(not (0 <= x <= width and 0 <= y <= length) for x, y in positions):
            raise ValueError(f"posição fora da sala na cena {rec.scene_id}")
        if rec.recipe_hash in seen_recipes and seen_recipes[rec.recipe_hash] != rec.split:
            raise ValueError(f"receita duplicada entre partições: {rec.scene_id}")
        seen_recipes[rec.recipe_hash] = rec.split
        split_speakers.setdefault(rec.split, set()).update(rec.speaker_ids)
    all_splits = list(split_speakers)
    leakage: list[tuple[str, str, str]] = []
    for i, left in enumerate(all_splits):
        for right in all_splits[i + 1 :]:
            overlap = split_speakers[left] & split_speakers[right]
            leakage.extend((left, right, speaker) for speaker in sorted(overlap))
    if leakage:
        raise ValueError(f"falantes compartilhados entre partições: {leakage[:5]}")
    return {
        "scene_count": len(records),
        "counts_by_split": {split: sum(r.split == split for r in records) for split in split_speakers},
        "speakers_by_split": {split: len(speakers) for split, speakers in split_speakers.items()},
        "recipe_count": len(seen_recipes),
    }


def dataset_id(records: Iterable[SceneRecord]) -> str:
    """Identificador estável da receita, independente da ordem do JSONL."""
    hashes = sorted(record.recipe_hash for record in records)
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()
