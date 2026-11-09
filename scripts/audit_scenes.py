"""Audit scene manifests, multichannel WAV pairs, SNR, and split separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_localization.config import load_config
from voice_localization.manifest import SceneRecord, read_jsonl, validate_scene_records


def audit(config_path: Path, require_clean: bool = True) -> dict:
    config = load_config(config_path)
    manifest_root = Path(config.dataset.manifest_root)
    all_records: list[SceneRecord] = []
    split_rows: dict[str, list[SceneRecord]] = {}
    for split in ("train", "validation", "test"):
        records = list(read_jsonl(manifest_root / f"scenes_{split}.jsonl", SceneRecord))
        split_rows[split] = records
        all_records.extend(records)
    split_audit = validate_scene_records(all_records, {"train", "validation", "test"})

    sample_rate = config.simulation.audio.sample_rate
    expected_samples = round(sample_rate * config.simulation.audio.duration_seconds)
    failures: list[str] = []
    reports: dict[str, dict] = {}
    for split, records in split_rows.items():
        snr_values: list[float] = []
        rt60_values: list[float] = []
        channel_counts: set[int] = set()
        for record in records:
            paths = {"mixture": record.audio_path, "clean": record.clean_audio_path}
            if not paths["mixture"] or not Path(paths["mixture"]).is_file():
                failures.append(f"{record.scene_id}: missing mixture WAV")
                continue
            if require_clean and (not paths["clean"] or not Path(paths["clean"]).is_file()):
                failures.append(f"{record.scene_id}: missing clean target WAV")
                continue
            try:
                mixture, mixture_rate = sf.read(paths["mixture"], dtype="float32", always_2d=True)
                mixture = mixture.T
                if mixture_rate != sample_rate or mixture.shape != (record.n_microphones, expected_samples):
                    failures.append(f"{record.scene_id}: mixture rate/shape {mixture_rate}/{mixture.shape}")
                    continue
                if not np.all(np.isfinite(mixture)):
                    failures.append(f"{record.scene_id}: non-finite mixture")
                    continue
                channel_counts.add(int(mixture.shape[0]))
                if paths["clean"] and Path(paths["clean"]).is_file():
                    clean, clean_rate = sf.read(paths["clean"], dtype="float32", always_2d=True)
                    clean = clean.T
                    if clean_rate != sample_rate or clean.shape != mixture.shape:
                        failures.append(f"{record.scene_id}: clean target rate/shape differs")
                        continue
                    if not np.all(np.isfinite(clean)):
                        failures.append(f"{record.scene_id}: non-finite clean target")
                        continue
                    noise = mixture.astype(np.float64) - clean.astype(np.float64)
                    signal_power = float(np.mean(clean.astype(np.float64) ** 2))
                    noise_power = float(np.mean(noise**2))
                    measured_snr = 10.0 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))
                    snr_values.append(float(measured_snr))
                    if record.snr_realized_db is None or abs(measured_snr - record.snr_realized_db) > 0.1:
                        failures.append(f"{record.scene_id}: measured SNR differs from manifest")
                if record.rt60_realized is not None:
                    rt60_values.append(float(record.rt60_realized))
            except Exception as exc:
                failures.append(f"{record.scene_id}: {type(exc).__name__}: {exc}")
        reports[split] = {
            "scene_count": len(records),
            "microphone_counts": sorted(channel_counts),
            "measured_snr_db_range": [min(snr_values), max(snr_values)] if snr_values else None,
            "realized_rt60_seconds_range": [min(rt60_values), max(rt60_values)] if rt60_values else None,
        }
    report = {
        "config": str(config_path),
        "scene_integrity": split_audit,
        "splits": reports,
        "failures": failures,
    }
    if failures:
        raise ValueError(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing-clean", action="store_true")
    args = parser.parse_args()
    report = audit(args.config, require_clean=not args.allow_missing_clean)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
