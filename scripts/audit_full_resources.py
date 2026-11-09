"""Summarize full-corpus storage and recoverable stage timings."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def _tree_summary(path: Path) -> dict:
    total_bytes = 0
    file_count = 0
    extensions: dict[str, dict[str, int]] = {}
    mtimes: list[float] = []
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "bytes": 0}

    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
                total_bytes += stat.st_size
                file_count += 1
                mtimes.append(stat.st_mtime)
                extension = Path(entry.name).suffix.lower() or "[sem extensão]"
                group = extensions.setdefault(extension, {"file_count": 0, "bytes": 0})
                group["file_count"] += 1
                group["bytes"] += stat.st_size

    summary = {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "bytes": total_bytes,
        "gib": total_bytes / (1024**3),
        "extensions": extensions,
    }
    if mtimes:
        first, last = min(mtimes), max(mtimes)
        summary["first_file_mtime_utc"] = datetime.fromtimestamp(first, timezone.utc).isoformat()
        summary["last_file_mtime_utc"] = datetime.fromtimestamp(last, timezone.utc).isoformat()
        summary["file_mtime_span_seconds"] = last - first
    return summary


def _map_timings(map_root: Path) -> list[dict]:
    rows = []
    for variant_dir in sorted(path for path in map_root.iterdir() if path.is_dir()):
        for split in ("train", "validation", "test"):
            path = variant_dir / f"timings_{split}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "variant": variant_dir.name,
                "split": split,
                "scene_count": int(payload["scene_count"]),
                "total_seconds": float(payload["total_seconds"]),
                "mean_seconds_per_scene": float(payload["total_seconds"]) / max(int(payload["scene_count"]), 1),
                "timing_file": str(path),
            })
    return rows


def _generation_manifest_summaries(manifest_root: Path) -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        path = manifest_root / f"generation_{split}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "split": split,
            "scene_count": int(payload["scene_count"]),
            "dataset_id": payload["dataset_id"],
            "generator_version": payload.get("generator_version"),
            "simulation_fingerprint": payload.get("simulation_fingerprint"),
            "manifest": str(path),
            "wall_time_seconds_recorded": payload.get("total_seconds"),
        })
    return rows


def _distribution(values: list[float | None], edges: list[float], labels: list[str]) -> dict[str, int]:
    counts = Counter({label: 0 for label in labels})
    for value in values:
        if value is None:
            continue
        if value < edges[0]:
            counts[labels[0]] += 1
        elif value > edges[-1]:
            counts[labels[-1]] += 1
        else:
            for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
                is_last = index == len(edges) - 2
                if lower <= value < upper or (is_last and lower <= value <= upper):
                    counts[labels[index + 1]] += 1
                    break
    return dict(counts)


def _scene_distributions(manifest_root: Path) -> dict:
    reports = {}
    for split in ("train", "validation", "test"):
        path = manifest_root / f"scenes_{split}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        n_counts = Counter(str(record["n_sources"]) for record in records)
        m_counts = Counter(str(record["n_microphones"]) for record in records)
        geometry_counts = Counter(str(record["geometry"]) for record in records)
        combo_counts = Counter(
            f"N={record['n_sources']},M={record['n_microphones']},{record['geometry']}"
            for record in records
        )
        reports[split] = {
            "scene_count": len(records),
            "n_sources": dict(sorted(n_counts.items())),
            "n_microphones": dict(sorted(m_counts.items())),
            "geometry": dict(sorted(geometry_counts.items())),
            "n_m_geometry_combinations": dict(sorted(combo_counts.items())),
            "snr_requested_db_bins": _distribution(
                [float(record["snr_requested_db"]) for record in records],
                [10.0, 15.0, 20.0, 25.0, 30.0],
                ["<10", "10–15", "15–20", "20–25", "25–30", ">30"],
            ),
            "snr_realized_db_bins": _distribution(
                [record.get("snr_realized_db") for record in records],
                [10.0, 15.0, 20.0, 25.0, 30.0],
                ["<10", "10–15", "15–20", "20–25", "25–30", ">30"],
            ),
            "rt60_requested_s_bins": _distribution(
                [float(record["rt60_requested"]) for record in records],
                [0.2, 0.4, 0.6, 0.8],
                ["<0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", ">0.8"],
            ),
            "rt60_realized_s_bins": _distribution(
                [record.get("rt60_realized") for record in records],
                [0.2, 0.4, 0.6, 0.8],
                ["<0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", ">0.8"],
            ),
        }
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "full")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "full" / "resource_audit.json")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    artifact_root = args.artifact_root.resolve()

    corpus = _tree_summary(data_root / "LibriSpeech")
    manifests = _tree_summary(data_root / "manifests")
    downloads = _tree_summary(data_root / "downloads")
    scene_audio = {
        split: _tree_summary(data_root / "scenes" / split)
        for split in ("train", "validation", "test")
    }
    map_cache = {
        path.name: _tree_summary(path)
        for path in sorted((data_root / "maps").iterdir())
        if path.is_dir()
    }
    enhanced_audio = {
        path.name: _tree_summary(path)
        for path in sorted((data_root / "enhanced").iterdir())
        if path.is_dir()
    }
    measured_directories = [corpus, manifests, downloads, *scene_audio.values(), *map_cache.values(), *enhanced_audio.values()]
    storage = {
        "librispeech_corpus": corpus,
        "downloads": downloads,
        "manifests": manifests,
        "scene_audio_by_split": scene_audio,
        "map_cache_by_variant": map_cache,
        "enhanced_audio_by_variant": enhanced_audio,
        "full_data_root": {
            "path": str(data_root),
            "exists": data_root.exists(),
            "file_count": sum(item["file_count"] for item in measured_directories),
            "bytes": sum(item["bytes"] for item in measured_directories),
            "gib": sum(item["bytes"] for item in measured_directories) / (1024**3),
            "total_is_sum_of_disjoint_top_level_directories": True,
        },
    }
    disk = shutil.disk_usage(data_root)
    storage["volume"] = {
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "storage": storage,
        "generation": {
            "split_manifests": _generation_manifest_summaries(data_root / "manifests"),
            "scene_distributions": _scene_distributions(data_root / "manifests"),
            "scene_file_mtime_spans": {
                split: {
                    key: value
                    for key, value in scene_audio[split].items()
                    if key in ("first_file_mtime_utc", "last_file_mtime_utc", "file_mtime_span_seconds")
                }
                for split in ("train", "validation", "test")
            },
            "interpretation": (
                "The generation manifests do not store process wall time. File modification-time spans are retained "
                "as provenance only and must not be interpreted as measured command duration."
            ),
        },
        "measured_map_extraction_timings": _map_timings(data_root / "maps"),
        "training_and_inference_timings_csv": str(artifact_root / "final" / "report" / "processing_costs.csv"),
        "provenance": {
            "scene_manifest_root": str(data_root / "manifests"),
            "scene_audio_root": str(data_root / "scenes"),
            "map_root": str(data_root / "maps"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "full_data_bytes": storage["full_data_root"]["bytes"],
        "scene_audio_gib": {
            split: round(item["gib"], 3)
            for split, item in storage["scene_audio_by_split"].items()
        },
        "map_variants": list(storage["map_cache_by_variant"]),
        "generation_wall_time_recorded": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
