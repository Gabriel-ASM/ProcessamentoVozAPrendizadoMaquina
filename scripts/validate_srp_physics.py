"""Checagem de sala anecoica: atrasos de banda larga e coordenadas conhecidas."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from voice_localization.audio import anechoic_mix_with_target
from voice_localization.geometry import microphone_array
from voice_localization.metrics import decode_peaks, localization_metrics
from voice_localization.spatial import gcc_phat_pairs, srp_phat_map


def run() -> dict:
    sample_rate = 16_000
    room = (8.0, 6.0, 3.0)
    source_positions_xy = ((1.4, 1.1), (3.8, 2.7), (6.5, 4.7))
    rows = []
    for geometry in ("perimeter", "circle"):
        for microphone_count in (4, 6, 8):
            geometry_seed = 42 + microphone_count
            microphones = microphone_array(geometry, microphone_count, room, seed=geometry_seed)
            for position_index, position_xy in enumerate(source_positions_xy):
                seed = geometry_seed + position_index * 100
                rng = np.random.default_rng(seed)
                source = rng.normal(size=3 * sample_rate).astype(np.float32)
                source_xy = np.asarray([position_xy], dtype=np.float32)
                source_xyz = np.column_stack((source_xy, np.array([1.5], dtype=np.float32)))
                _, clean, _ = anechoic_mix_with_target(
                    [source], source_xyz, microphones, sample_rate, 100.0, 343.0, rng
                )
                correlations, pairs = gcc_phat_pairs(clean, window=512, hop=256, fft_size=1024)
                lags = np.arange(-512, 512)
                peak_lags = lags[np.argmax(correlations, axis=1)]
                expected = np.asarray(
                    [
                        (
                            np.linalg.norm(source_xyz[0] - microphones[i])
                            - np.linalg.norm(source_xyz[0] - microphones[j])
                        )
                        / 343.0
                        * sample_rate
                        for i, j in pairs
                    ]
                )
                map_values, grid = srp_phat_map(
                    clean,
                    microphones,
                    sample_rate,
                    room,
                    resolution_m=0.1,
                    window=512,
                    hop=256,
                    fft_size=1024,
                )
                estimate = decode_peaks(map_values, grid.xs, grid.ys, 1, suppression_radius_m=0.3)
                rows.append(
                    {
                        "geometry": geometry,
                        "microphones": microphone_count,
                        "source_xy": source_xy[0].tolist(),
                        "estimate_xy": estimate[0].tolist(),
                        "mean_abs_tdoa_error_samples": float(np.mean(np.abs(peak_lags - expected))),
                        "max_abs_tdoa_error_samples": float(np.max(np.abs(peak_lags - expected))),
                        "localization_error_m": localization_metrics(source_xy, estimate).mean_error_m,
                    }
                )
    return {
        "backend": "anechoic broadband point sources",
        "sample_rate": sample_rate,
        "room_dimensions_m": room,
        "sample_duration_seconds": 3,
        "grid_spacing_m": 0.1,
        "source_positions_xy": source_positions_xy,
        "geometry_count": 2,
        "microphone_counts": [4, 6, 8],
        "maximum_tdoa_error_expected_samples": 1.0,
        "maximum_localization_error_expected_m": 0.2,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = json.dumps(run(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)
