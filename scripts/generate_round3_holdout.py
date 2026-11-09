"""Audit and generate the frozen 600-scene LibriSpeech test-other holdout."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voice_localization.config import load_config
from voice_localization.manifest import AudioRecord, SceneRecord, read_jsonl, write_jsonl
from voice_localization.simulation import SceneGenerator

OUT = ROOT / "artifacts" / "round3"
SOURCE_ROOT = ROOT / "data" / "full" / "downloads" / "test-other_extract" / "LibriSpeech" / "test-other"
RE_AUDIO = re.compile(r"(?P<speaker>\d+)-(?P<chapter>\d+)-(?P<utterance>\d+)")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"test-other extraído não encontrado: {SOURCE_ROOT}")
    old_speakers, old_audio_ids = set(), set()
    for split in ("train", "validation", "test"):
        for row in read_jsonl(ROOT / "data" / "full" / "manifests" / f"audio_{split}.jsonl", AudioRecord):
            old_speakers.add(row.speaker_id); old_audio_ids.add(row.audio_id)
    candidates = []
    for path in sorted(SOURCE_ROOT.rglob("*.flac")):
        match = RE_AUDIO.fullmatch(path.stem)
        if not match:
            continue
        candidates.append(AudioRecord(audio_id=path.stem, path=str(path.resolve()), split="test_other",
            speaker_id=match.group("speaker"), sample_rate=16000, duration_seconds=0.0,
            transcript_path=str((path.parent / f"{match.group('speaker')}-{match.group('chapter')}.trans.txt").resolve())))
    speakers = {r.speaker_id for r in candidates}
    audio_ids = {r.audio_id for r in candidates}
    overlap_speakers = sorted(speakers & old_speakers)
    overlap_audio = sorted(audio_ids & old_audio_ids)
    exclusive = [r for r in candidates if r.speaker_id not in old_speakers and r.audio_id not in old_audio_ids]
    exclusive_speakers = {r.speaker_id for r in exclusive}
    archive = ROOT / "data" / "full" / "downloads" / "test-other.tar.gz"
    result = {
        "corpus": "LibriSpeech test-other / OpenSLR 12",
        "archive_sha256": sha256(archive),
        "scanned_audio_files": len(candidates), "scanned_speakers": len(speakers),
        "existing_audio_files_train_validation_test": len(old_audio_ids),
        "existing_speakers_train_validation_test": len(old_speakers),
        "overlap_audio_ids": overlap_audio, "overlap_speaker_ids": overlap_speakers,
        "exclusive_audio_files": len(exclusive), "exclusive_speakers": len(exclusive_speakers),
        "speaker_and_audio_independent": not overlap_speakers and not overlap_audio,
    }
    if len(exclusive_speakers) < 3:
        raise RuntimeError("não há três falantes exclusivos para gerar cenas multifuente")
    # If the official quality split shares any speakers with the old corpus,
    # keep only exclusive speakers; never silently label overlap as independent.
    write_jsonl(exclusive, OUT / "manifests" / "test_other_audio_exclusive.jsonl")
    (OUT / "holdout_source_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("HOLDOUT_SOURCE_AUDIT", json.dumps(result), flush=True)
    return exclusive, result


def generate():
    audio_rows, audit_result = audit()
    if len(audio_rows) < 150:
        raise RuntimeError(f"quantidade insuficiente de áudios exclusivos: {len(audio_rows)}")
    cfg = load_config(ROOT / "configs" / "full.yaml")
    cfg.dataset.seed = 20261109
    cfg.dataset.output_root = str((ROOT / "data" / "round3_holdout").resolve())
    cfg.dataset.manifest_root = str((OUT / "manifests" / "holdout").resolve())
    cfg.dataset.generation_workers = 2
    cfg.dataset.train_count = 0; cfg.dataset.validation_count = 0; cfg.dataset.test_count = 600
    started = time.perf_counter()
    generator = SceneGenerator(cfg, audio_rows, synthetic_audio=False)
    generated = generator.generate_split("test", 600)
    scenes = list(read_jsonl(generated.manifest_path, SceneRecord))
    speaker_ids = {sp for row in scenes for sp in row.speaker_ids}
    used_audio_ids = {audio for row in scenes for audio in row.audio_ids}
    old_speakers, old_audio_ids = set(), set()
    for split in ("train", "validation", "test"):
        for row in read_jsonl(ROOT / "data" / "full" / "manifests" / f"audio_{split}.jsonl", AudioRecord):
            old_speakers.add(row.speaker_id); old_audio_ids.add(row.audio_id)
    if speaker_ids & old_speakers or used_audio_ids & old_audio_ids:
        raise RuntimeError("holdout gerado contém vazamento de áudio/falante")
    manifest_hash = sha256(generated.manifest_path)
    frozen = {"manifest_path": str(generated.manifest_path), "manifest_sha256": manifest_hash,
              "scene_count": len(scenes), "seed": 20261109, "used_speaker_count": len(speaker_ids),
              "used_audio_count": len(used_audio_ids), "speaker_overlap_count": 0, "audio_overlap_count": 0,
              "source_audit": audit_result, "frozen_before_inference": True,
              "generation_seconds": time.perf_counter()-started}
    (OUT / "holdout_manifest_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("HOLDOUT_MANIFEST_FROZEN", json.dumps(frozen), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "generate"))
    args = parser.parse_args()
    if args.mode == "audit": audit()
    else: generate()


if __name__ == "__main__":
    main()
