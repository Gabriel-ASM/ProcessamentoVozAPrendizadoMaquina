"""Fixa cópias imutáveis dos manifestos e da configuração da base completa."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_localization.config import load_config  # noqa: E402
from voice_localization.manifest import AudioRecord, SceneRecord, dataset_id, read_jsonl  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "full.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "full" / "frozen_dataset")
    args = parser.parse_args()
    config = load_config(args.config)
    source_root = _resolve(config.dataset.manifest_root)
    output = _resolve(args.output)
    manifest_names = [f"audio_{split}.jsonl" for split in ("train", "validation", "test")]
    manifest_names += [f"scenes_{split}.jsonl" for split in ("train", "validation", "test")]
    missing = [name for name in manifest_names if not (source_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"manifestos completos ausentes em {source_root}: {missing}")

    scene_records = {
        split: list(read_jsonl(source_root / f"scenes_{split}.jsonl", SceneRecord))
        for split in ("train", "validation", "test")
    }
    audio_records = {
        split: list(read_jsonl(source_root / f"audio_{split}.jsonl", AudioRecord))
        for split in ("train", "validation", "test")
    }
    dataset_ids = {split: dataset_id(records) for split, records in scene_records.items()}
    config_source = _resolve(args.config)
    normalizer_source = _resolve(config.dataset.output_root) / "maps" / "mixture" / "normalizer.npz"
    audit_source = ROOT / "artifacts" / "full" / "scene_audio_audit.json"
    if not normalizer_source.exists():
        raise FileNotFoundError(f"normalizador de mapas SRP ausente: {normalizer_source}")
    if not audit_source.exists():
        raise FileNotFoundError(f"auditoria completa ausente: {audit_source}")
    files_to_copy = {name: source_root / name for name in manifest_names}
    files_to_copy["configs/full.yaml"] = config_source
    if normalizer_source.exists():
        files_to_copy["maps/mixture/normalizer.npz"] = normalizer_source
    if audit_source.exists():
        files_to_copy["scene_audio_audit.json"] = audit_source

    freeze_path = output / "freeze.json"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen["config_sha256"] != _sha256(config_source):
            raise ValueError("configuração mudou depois do freeze da base")
        if frozen["scene_dataset_ids"] != dataset_ids:
            raise ValueError("receitas da base mudaram depois do freeze")
        for relative, expected_hash in frozen["files_sha256"].items():
            path = output / relative
            if not path.exists() or _sha256(path) != expected_hash:
                raise ValueError(f"arquivo congelado mudou ou sumiu: {path}")
        print(json.dumps({"output": str(output), "already_frozen": True, "scene_dataset_ids": dataset_ids}, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for relative, source in files_to_copy.items():
        if not source.exists():
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        file_hashes[relative] = _sha256(destination)
    summary = {
        "profile": config.profile,
        "config_sha256": _sha256(config_source),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_split": "test",
        "scene_counts": {split: len(records) for split, records in scene_records.items()},
        "audio_counts": {split: len(records) for split, records in audio_records.items()},
        "scene_dataset_ids": dataset_ids,
        "speaker_counts": {
            split: len({record.speaker_id for record in records}) for split, records in audio_records.items()
        },
        "files_sha256": file_hashes,
    }
    temporary = freeze_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(freeze_path)
    print(json.dumps({"output": str(output), "already_frozen": False, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
