"""Generate one Streamlit demo scene in an isolated acoustics-only process."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_localization.config import load_config  # noqa: E402
from voice_localization.corpus import load_audio_records  # noqa: E402
from voice_localization.simulation import SceneGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sources", type=int, required=True)
    parser.add_argument("--n-microphones", type=int, required=True)
    parser.add_argument(
        "--geometry", choices=("perimeter", "circle", "circle_r6"), required=True
    )
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--rt60", type=float, required=True)
    parser.add_argument("--counter", type=int, required=True)
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "full.yaml")
    if args.geometry == "circle_r6":
        # Keep the Round 6 finalist out of the frozen training configuration.
        config.simulation.geometries = (*config.simulation.geometries, "circle_r6")
    if args.n_sources not in config.simulation.n_sources:
        parser.error("N fora do domínio do experimento")
    if args.n_microphones not in config.simulation.n_microphones:
        parser.error("M fora do domínio do experimento")
    if args.geometry not in config.simulation.geometries:
        parser.error("Geometria fora do domínio do experimento")
    if not config.simulation.snr_range_db[0] <= args.snr_db <= config.simulation.snr_range_db[1]:
        parser.error("SNR fora do domínio do experimento")
    if not config.simulation.rt60_range[0] <= args.rt60 <= config.simulation.rt60_range[1]:
        parser.error("RT60 fora do domínio do experimento")
    if args.counter < 0:
        parser.error("contador da sessão não pode ser negativo")

    config.simulation.snr_range_db = (args.snr_db, args.snr_db)
    config.simulation.rt60_range = (args.rt60, args.rt60)
    config.dataset.output_root = str(ROOT / "data" / "demo")
    config.dataset.manifest_root = str(ROOT / "data" / "demo" / "manifests")

    records = load_audio_records(ROOT / "data" / "full" / "manifests", "train")
    n_index = config.simulation.n_sources.index(args.n_sources)
    m_index = config.simulation.n_microphones.index(args.n_microphones)
    geometry_index = config.simulation.geometries.index(args.geometry)
    conditions_per_geometry = (
        len(config.simulation.n_sources) * len(config.simulation.n_microphones)
    )
    counter_offset = args.counter + (10_000 if args.geometry == "circle_r6" else 0)
    scene_index = (
        geometry_index * conditions_per_geometry
        + m_index * len(config.simulation.n_sources)
        + n_index
        + counter_offset * conditions_per_geometry * len(config.simulation.geometries)
    )
    generator = SceneGenerator(config, records, synthetic_audio=False)
    output_dir = Path(config.dataset.output_root) / "scenes" / "train"
    record = generator._generate_scene("train", scene_index, output_dir)
    print(record.scene_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
