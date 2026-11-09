"""Comandos de preparação, validação e execução do projeto."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from .config import load_config, save_resolved_config
from .corpus import load_audio_records, prepare_corpus
from .evaluation import evaluate_checkpoint, evaluate_srp
from .enhancement import enhance_split, train_wave_unet
from .manifest import SceneRecord, read_jsonl, validate_scene_records
from .preprocess import apply_map_normalizer, extract_maps_for_split, fit_map_normalizer, validate_map_files
from .simulation import SceneGenerator
from .training import smoke_train, train, validate_small_set_learning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-localization")
    sub = parser.add_subparsers(dest="command", required=True)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=True, help="caminho para configuração YAML")
    p = sub.add_parser("validate-config", parents=[config_parser])
    p.set_defaults(func=_validate_config)
    p = sub.add_parser("prepare-corpus")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.set_defaults(func=_prepare_corpus)
    p = sub.add_parser("generate-scenes", parents=[config_parser])
    p.add_argument("--synthetic", action="store_true", help="usar sinais sintéticos, apenas para smoke/piloto")
    p.add_argument("--split", choices=["train", "validation", "test"], default=None)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="sobrescreve o número de processos de geração")
    p.set_defaults(func=_generate)
    p = sub.add_parser("extract-maps", parents=[config_parser])
    p.add_argument("--split", choices=["train", "validation", "test"], default=None)
    p.add_argument("--variant", default=None, help="nome do cache e manifesto de mapas")
    p.add_argument("--audio-variant", choices=["mixture", "clean", "enhanced"], default=None)
    p.set_defaults(func=_extract_maps)
    p = sub.add_parser("validate-data", parents=[config_parser])
    p.add_argument("--split", choices=["train", "validation", "test"], default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--audio-variant", choices=["mixture", "clean", "enhanced"], default=None)
    p.set_defaults(func=_validate_data)
    p = sub.add_parser("evaluate-baseline", parents=[config_parser])
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--output", default=None)
    p.add_argument("--variant", default=None, help="cache SRP: mixture ou nome de uma variante")
    p.set_defaults(func=_evaluate_baseline)
    p = sub.add_parser("smoke-test", parents=[config_parser])
    p.set_defaults(func=_smoke_test)
    p = sub.add_parser("train", parents=[config_parser])
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--variant", default=None, help="variante de mapas que será treinada")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_train)
    p = sub.add_parser("validate-learning", parents=[config_parser])
    p.add_argument("--variant", default=None)
    p.add_argument("--scenes", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=_validate_learning)
    p = sub.add_parser("train-enhancer", parents=[config_parser])
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=_train_enhancer)
    p = sub.add_parser("enhance-audio", parents=[config_parser])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], required=True)
    p.set_defaults(func=_enhance_audio)
    p = sub.add_parser("evaluate", parents=[config_parser])
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--variant", default=None, help="variante de mapas usada na avaliação")
    p.add_argument("--output", default=None)
    p.set_defaults(func=_evaluate)
    return parser


def _load(args):
    config = load_config(args.config)
    save_resolved_config(config, Path(config.train.run_root) / config.profile / "config.json")
    return config


def _validate_config(args) -> int:
    config = _load(args)
    print(config.resolved_json())
    return 0


def _prepare_corpus(args) -> int:
    print(json.dumps(prepare_corpus(args.root, args.output, args.sample_rate), indent=2))
    return 0


def _generate(args) -> int:
    config = _load(args)
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("--workers deve ser >= 1")
        config.dataset.generation_workers = args.workers
    synthetic = args.synthetic or config.profile == "smoke"
    records = None if synthetic else load_audio_records(config.dataset.manifest_root, args.split or "train")
    generator = SceneGenerator(config, records, synthetic_audio=synthetic)
    if args.split:
        default_count = {
            "train": config.dataset.train_count,
            "validation": config.dataset.validation_count,
            "test": config.dataset.test_count,
        }[args.split]
        result = generator.generate_split(args.split, args.count if args.count is not None else default_count)
        print(result.manifest_path)
    else:
        # Audio records are split-specific; for real corpus, process each split separately.
        results = []
        for split, count in (("train", config.dataset.train_count), ("validation", config.dataset.validation_count), ("test", config.dataset.test_count)):
            split_records = None if synthetic else load_audio_records(config.dataset.manifest_root, split)
            results.append(SceneGenerator(config, split_records, synthetic_audio=synthetic).generate_split(split, count).manifest_path)
        print("\n".join(map(str, results)))
    return 0


def _extract_maps(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    if args.audio_variant:
        config.dataset.map_audio_variant = args.audio_variant
    splits = [args.split] if args.split else ["train", "validation", "test"]
    for split in splits:
        print(extract_maps_for_split(config, split))
    if args.split:
        if args.split == "train":
            normalizer = fit_map_normalizer(config)
            apply_map_normalizer(config, "train", normalizer)
            print(normalizer)
        else:
            apply_map_normalizer(config, args.split)
    else:
        normalizer = fit_map_normalizer(config)
        for split in ("train", "validation", "test"):
            apply_map_normalizer(config, split, normalizer)
        print(normalizer)
    return 0


def _validate_data(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    if args.audio_variant:
        config.dataset.map_audio_variant = args.audio_variant
    splits = [args.split] if args.split else ["train", "validation", "test"]
    reports = []
    all_records = []
    for split in splits:
        canonical = Path(config.dataset.manifest_root) / f"scenes_{split}.jsonl"
        all_records.extend(read_jsonl(canonical, SceneRecord))
        reports.append(validate_map_files(config, split))
    # A validação de vazamento precisa receber todas as partições de uma vez.
    reports.insert(0, validate_scene_records(all_records, {"train", "validation", "test"}))
    print(json.dumps(reports, indent=2))
    return 0


def _evaluate_baseline(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    print(json.dumps(evaluate_srp(config, args.split, args.output), indent=2))
    return 0


def _smoke_test(args) -> int:
    config = _load(args)
    config.dataset.train_count = max(config.dataset.train_count, 8)
    config.dataset.validation_count = max(config.dataset.validation_count, 4)
    config.dataset.test_count = max(config.dataset.test_count, 4)
    generator = SceneGenerator(config, synthetic_audio=True)
    for split, count in (("train", config.dataset.train_count), ("validation", config.dataset.validation_count), ("test", config.dataset.test_count)):
        generator.generate_split(split, count)
        extract_maps_for_split(config, split)
        validate_map_files(config, split)
    normalizer = fit_map_normalizer(config)
    for split in ("train", "validation", "test"):
        apply_map_normalizer(config, split, normalizer)
    print(json.dumps(evaluate_srp(config, "test"), indent=2))
    try:
        checkpoint = smoke_train(config)
        print(f"checkpoint: {checkpoint}")
    except RuntimeError as exc:
        if "PyTorch" in str(exc):
            print(f"treino smoke não executado: {exc}", file=sys.stderr)
        else:
            raise
    return 0


def _train(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    if args.seed is not None:
        config.train.seed = args.seed
    print(train(config, args.run_name, args.resume))
    return 0


def _train_enhancer(args) -> int:
    config = _load(args)
    if args.seed is not None:
        config.train.seed = args.seed
    print(train_wave_unet(config, args.run_name, args.resume))
    return 0


def _validate_learning(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    if args.seed is not None:
        config.train.seed = args.seed
    print(json.dumps(validate_small_set_learning(config, args.scenes, args.steps), indent=2))
    return 0


def _enhance_audio(args) -> int:
    config = _load(args)
    print(enhance_split(config, args.checkpoint, args.split))
    return 0


def _evaluate(args) -> int:
    config = _load(args)
    if args.variant:
        config.dataset.map_variant = args.variant
    if args.checkpoint:
        print(json.dumps(evaluate_checkpoint(config, args.checkpoint, args.split, args.output), indent=2))
    else:
        print(json.dumps(evaluate_srp(config, args.split), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
