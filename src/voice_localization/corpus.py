"""Descoberta e preparação do corpus LibriSpeech."""

from __future__ import annotations

from pathlib import Path
import re

from .audio import read_audio
from .manifest import AudioRecord, write_jsonl


_AUDIO_RE = re.compile(r"(?P<speaker>\d+)-(?P<chapter>\d+)-(?P<utterance>\d+)")


def prepare_corpus(root: str | Path, output_dir: str | Path, sample_rate: int = 16_000) -> dict[str, int]:
    root = Path(root)
    output_dir = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"corpus não encontrado: {root}")
    records_by_split: dict[str, list[AudioRecord]] = {"train": [], "validation": [], "test": []}
    source_dirs = {
        # A proposta fixa train-clean-100 para manter o corpus e o custo
        # reproduzíveis; train-clean-360 pode ser adicionado em experimento
        # posterior, mas não entra silenciosamente nesta versão.
        "train": [root / "train-clean-100"],
        "validation": [root / "dev-clean"],
        "test": [root / "test-clean"],
    }
    for split, directories in source_dirs.items():
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.flac")) + sorted(directory.rglob("*.wav")):
                match = _AUDIO_RE.search(path.stem)
                if not match:
                    continue
                relative_id = path.stem
                try:
                    data = read_audio(path, sample_rate)
                    duration = len(data) / sample_rate
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"não foi possível ler {path}: {exc}") from exc
                transcript = path.parent / f"{match.group('speaker')}-{match.group('chapter')}.trans.txt"
                records_by_split[split].append(
                    AudioRecord(
                        audio_id=relative_id,
                        path=str(path.resolve()),
                        split=split,
                        speaker_id=match.group("speaker"),
                        sample_rate=sample_rate,
                        duration_seconds=float(duration),
                        transcript_path=str(transcript.resolve()) if transcript.exists() else None,
                    )
                )
    if not records_by_split["train"] or not records_by_split["validation"] or not records_by_split["test"]:
        missing = [split for split, records in records_by_split.items() if not records]
        raise RuntimeError(f"partições ausentes ou vazias: {missing}")
    speakers = {split: {record.speaker_id for record in records} for split, records in records_by_split.items()}
    overlap = {a: sorted(speakers[a] & speakers[b]) for a in speakers for b in speakers if a < b}
    if any(overlap.values()):
        raise ValueError(f"falantes compartilhados entre partições: {overlap}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in records_by_split.items():
        write_jsonl(records, output_dir / f"audio_{split}.jsonl")
    return {split: len(records) for split, records in records_by_split.items()}


def load_audio_records(manifest_dir: str | Path, split: str) -> list[AudioRecord]:
    from .manifest import read_jsonl

    path = Path(manifest_dir) / f"audio_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"manifesto de áudio não encontrado: {path}")
    return list(read_jsonl(path, AudioRecord))
