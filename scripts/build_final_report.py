"""Resume avaliações finais congeladas e gera tabelas, figuras e relatório."""

from __future__ import annotations

import argparse
import csv
import hashlib
from inspect import signature
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHOD_FILES = {
    "SRP-PHAT": "srp_test.json",
    "CNN sobre mistura": "cnn_mixture_test.json",
    "Wave-U-Net + SRP-PHAT": "wave_srp_test.json",
    "Wave-U-Net + CNN": "cnn_wave_test.json",
}
METHODS = tuple(METHOD_FILES)
PAIRS = (
    ("SRP-PHAT", "CNN sobre mistura"),
    ("SRP-PHAT", "Wave-U-Net + SRP-PHAT"),
    ("SRP-PHAT", "Wave-U-Net + CNN"),
    ("CNN sobre mistura", "Wave-U-Net + CNN"),
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"resultado final ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(rows: list[dict]) -> dict[str, float | int]:
    errors = np.asarray([e for row in rows for e in row["source_errors_m"]], dtype=np.float64)
    times = [float(row["inference_seconds"]) for row in rows if row.get("inference_seconds") is not None]
    return {
        "scene_count": len(rows),
        "source_count": int(errors.size),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "within_0_5m": float(np.mean(errors <= 0.5)),
        "mean_inference_seconds": float(np.mean(times)) if times else float("nan"),
    }


def _load_runs(results_root: Path, seeds: list[int]) -> dict[int, dict[str, dict]]:
    runs: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        seed_dir = results_root / f"seed{seed}"
        runs[seed] = {
            method: _load_json(seed_dir / filename)
            for method, filename in METHOD_FILES.items()
        }
        expected = set(runs[seed][METHODS[0]]["rows"][i]["scene_id"] for i in range(len(runs[seed][METHODS[0]]["rows"])))
        for method in METHODS[1:]:
            observed = {str(row["scene_id"]) for row in runs[seed][method]["rows"]}
            if observed != expected:
                raise ValueError(f"{method} na seed {seed} não avaliou as mesmas cenas de SRP-PHAT")
    return runs


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _overall_table(runs: dict[int, dict[str, dict]], artifact_root: Path, map_root: Path) -> list[dict]:
    table: list[dict] = []
    for seed, methods in runs.items():
        wave_checkpoint = artifact_root / f"wave_seed{seed}" / "wave_unet" / "best.pt"
        if not wave_checkpoint.exists():
            raise FileNotFoundError(f"checkpoint Wave-U-Net da seed {seed} ausente: {wave_checkpoint}")
        wave_hash = hashlib.sha256(wave_checkpoint.read_bytes()).hexdigest()[:12]
        wave_variant = f"wave_{wave_hash}"
        wave_eval_path = artifact_root / "full" / "wave_unet" / f"evaluation_test_{wave_hash}.json"
        wave_map_timing = map_root / wave_variant / "timings_test.json"
        if not wave_eval_path.exists() or not wave_map_timing.exists():
            raise FileNotFoundError(f"diagnósticos Wave de teste ausentes para seed {seed}")
        wave_eval = json.loads(wave_eval_path.read_text(encoding="utf-8"))
        wave_map = json.loads(wave_map_timing.read_text(encoding="utf-8"))
        mixture_map_timing = map_root / "mixture" / "timings_test.json"
        if not mixture_map_timing.exists():
            raise FileNotFoundError(f"tempos de extração SRP da mistura ausentes: {mixture_map_timing}")
        mixture_map = json.loads(mixture_map_timing.read_text(encoding="utf-8"))
        for method, payload in methods.items():
            metrics = _metrics(payload["rows"])
            uses_wave = method.startswith("Wave-U-Net")
            map_timing = wave_map if uses_wave else mixture_map
            map_seconds = float(map_timing["total_seconds"]) / max(int(map_timing["scene_count"]), 1)
            wave_seconds = float(wave_eval["mean_wave_inference_seconds"]) if uses_wave else 0.0
            total_seconds = float(metrics["mean_inference_seconds"]) + map_seconds + wave_seconds
            table.append({
                "seed": seed,
                "method": method,
                **metrics,
                "mean_map_extraction_seconds": map_seconds,
                "mean_wave_inference_seconds": wave_seconds,
                "mean_total_pipeline_seconds": total_seconds,
            })
    for method in METHODS:
        seed_rows = [row for row in table if row["method"] == method]
        pooled = {"seed": "média ± DP", "method": method}
        for key in (
            "mean_error_m", "median_error_m", "p90_error_m", "within_0_5m",
            "mean_inference_seconds", "mean_map_extraction_seconds",
            "mean_wave_inference_seconds", "mean_total_pipeline_seconds",
        ):
            values = np.asarray([row[key] for row in seed_rows], dtype=np.float64)
            pooled[key] = f"{values.mean():.6f} ± {values.std(ddof=1):.6f}"
        pooled["scene_count"] = seed_rows[0]["scene_count"]
        pooled["source_count"] = seed_rows[0]["source_count"]
        table.append(pooled)
    return table


def _stratified_table(runs: dict[int, dict[str, dict]]) -> list[dict]:
    output: list[dict] = []
    for seed, methods in runs.items():
        for method, payload in methods.items():
            for field in ("n_sources", "n_microphones", "geometry", "snr_band", "rt60_band"):
                groups: dict[str, list[dict]] = {}
                for row in payload["rows"]:
                    groups.setdefault(str(row[field]), []).append(row)
                for level, rows in sorted(groups.items()):
                    output.append({
                        "seed": seed,
                        "method": method,
                        "stratum": field,
                        "level": level,
                        **_metrics(rows),
                    })
    return output


def _padded_error_array(
    runs: dict[int, dict[str, dict]], method: str, scene_ids: list[str], seeds: list[int] | None = None
) -> np.ndarray:
    seeds = list(runs) if seeds is None else seeds
    by_seed = []
    max_sources = max(
        len(row["source_errors_m"])
        for seed in seeds
        for row in runs[seed][method]["rows"]
    )
    for seed in seeds:
        rows = {str(row["scene_id"]): row for row in runs[seed][method]["rows"]}
        matrix = np.full((len(scene_ids), max_sources), np.nan, dtype=np.float64)
        for i, scene_id in enumerate(scene_ids):
            errors = np.asarray(rows[scene_id]["source_errors_m"], dtype=np.float64)
            matrix[i, : len(errors)] = errors
        by_seed.append(matrix)
    return np.stack(by_seed)


def _metric_vector(errors: np.ndarray) -> np.ndarray:
    values = errors[np.isfinite(errors)]
    return np.asarray(
        [values.mean(), np.median(values), np.percentile(values, 90), np.mean(values <= 0.5)],
        dtype=np.float64,
    )


def _metric_vectors_batched(errors: np.ndarray) -> np.ndarray:
    """Calculate bootstrap metrics for [seed, sample, scene, source] batches."""
    values = np.transpose(errors, (1, 0, 2, 3)).reshape(errors.shape[1], -1)
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    return np.column_stack((
        np.nansum(values, axis=1) / counts,
        np.nanmedian(values, axis=1),
        np.nanpercentile(values, 90, axis=1),
        np.sum(values <= 0.5, axis=1) / counts,
    ))


def _paired_bootstrap(runs: dict[int, dict[str, dict]], samples: int, seed: int) -> list[dict]:
    scene_ids = sorted(str(row["scene_id"]) for row in runs[next(iter(runs))][METHODS[0]]["rows"])
    names = ("mean_error_m", "median_error_m", "p90_error_m", "within_0_5m")
    output = []
    scopes: list[tuple[str | int, list[int]]] = [(scope_seed, [scope_seed]) for scope_seed in runs]
    scopes.append(("média entre seeds", list(runs)))
    for scope_index, (scope_name, scope_seeds) in enumerate(scopes):
        arrays = {
            method: _padded_error_array(runs, method, scene_ids, scope_seeds)
            for method in METHODS
        }
        points = {method: _metric_vector(arrays[method]) for method in METHODS}
        rng = np.random.default_rng(seed + scope_index)
        deltas_by_pair = {
            pair: np.empty((samples, len(names)), dtype=np.float64)
            for pair in PAIRS
        }
        batch_size = 128
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            indices = rng.integers(0, len(scene_ids), size=(stop - start, len(scene_ids)))
            metric_samples = {}
            for method in METHODS:
                sampled = arrays[method][:, indices, :]
                metric_samples[method] = _metric_vectors_batched(sampled)
            for reference, candidate in PAIRS:
                deltas_by_pair[(reference, candidate)][start:stop] = (
                    metric_samples[candidate] - metric_samples[reference]
                )
        for reference, candidate in PAIRS:
            deltas = deltas_by_pair[(reference, candidate)]
            point = points[candidate] - points[reference]
            for index, metric in enumerate(names):
                low, high = np.percentile(deltas[:, index], [2.5, 97.5])
                output.append({
                    "seed": scope_name,
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "delta_candidate_minus_reference": float(point[index]),
                    "ci_95_low": float(low),
                    "ci_95_high": float(high),
                    "scene_count": len(scene_ids),
                    "seed_count": len(scope_seeds),
                    "bootstrap_samples": samples,
                    "bootstrap_unit": "cena, agrupada entre fontes da cena",
                })
    return output


def _make_plots(runs: dict[int, dict[str, dict]], output_dir: Path, artifact_root: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#3c78a8", "#f58518", "#54a24b", "#e45756")
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    distributions = []
    for method in METHODS:
        distributions.append([
            error
            for methods in runs.values()
            for row in methods[method]["rows"]
            for error in row["source_errors_m"]
        ])
    label_argument = "tick_labels" if "tick_labels" in signature(ax.boxplot).parameters else "labels"
    box = ax.boxplot(distributions, showfliers=False, patch_artist=True, **{label_argument: METHODS})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Erro de localização por fonte (m)")
    ax.set_title("Distribuição dos erros no teste reservado")
    ax.tick_params(axis="x", labelrotation=15)
    fig.savefig(output_dir / "error_distribution.png", dpi=180)
    plt.close(fig)

    fields = (
        ("snr_band", "SNR solicitado", "Faixa (dB)"),
        ("rt60_band", "RT60 realizado", "Faixa (s)"),
        ("n_sources", "Número de fontes N", "N"),
        ("n_microphones", "Número de microfones M", "M"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (field, label, x_label) in zip(axes.flat, fields):
        first = runs[next(iter(runs))][METHODS[0]]["rows"]
        levels = sorted({str(row[field]) for row in first})
        for method, color in zip(METHODS, colors):
            means = []
            for level in levels:
                errors = [
                    error
                    for methods in runs.values()
                    for row in methods[method]["rows"]
                    if str(row[field]) == level
                    for error in row["source_errors_m"]
                ]
                means.append(float(np.mean(errors)))
            ax.plot(levels, means, marker="o", label=method, color=color)
        ax.set_title(label)
        ax.set_ylabel("Erro médio (m)")
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_dir / "performance_by_condition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for seed in runs:
        for method, ax in (("CNN sobre mistura", axes[0]), ("Wave-U-Net + CNN", axes[1])):
            folder = artifact_root / f"cnn_{'mixture' if method == 'CNN sobre mistura' else 'wave'}_seed{seed}"
            history_path = folder / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                ax.plot([r["epoch"] for r in history], [r["validation"]["mean_error_m"] for r in history], label=f"seed {seed}")
        history_path = artifact_root / f"wave_seed{seed}" / "wave_unet" / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            axes[2].plot([r["epoch"] for r in history], [r["validation_l1_normalized"] for r in history], label=f"seed {seed}")
    axes[0].set_title("CNN sobre a mistura")
    axes[1].set_title("CNN sobre mapas Wave")
    axes[2].set_title("Wave-U-Net — L1 de validação")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Erro médio de localização (m)")
    axes[2].set_ylabel("L1 normalizado")
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)


def _make_spatial_examples(runs: dict[int, dict[str, dict]], output_dir: Path, map_root: Path, artifact_root: Path) -> list[dict]:
    import matplotlib.pyplot as plt

    from voice_localization.visualization import plot_scene

    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 42 if 42 in runs else next(iter(runs))
    wave_checkpoint = artifact_root / f"wave_seed{seed}" / "wave_unet" / "best.pt"
    if not wave_checkpoint.exists():
        return []
    wave_hash = hashlib.sha256(wave_checkpoint.read_bytes()).hexdigest()[:12]
    wave_variant = f"wave_{wave_hash}"
    examples = []
    for method in METHODS:
        rows = runs[seed][method]["rows"]
        success_candidates = [row for row in rows if float(row["mean_error_m"]) <= 0.5]
        if success_candidates:
            success = min(success_candidates, key=lambda row: float(row["mean_error_m"]))
            success_rule = "menor erro médio entre cenas até 0,50 m"
        else:
            success = min(rows, key=lambda row: float(row["mean_error_m"]))
            success_rule = "menor erro médio disponível; nenhuma cena ficou até 0,50 m"
        failure = max(rows, key=lambda row: float(row["mean_error_m"]))
        variant = "mixture" if method in (METHODS[0], METHODS[1]) else wave_variant
        success_label = "Melhor cena ≤0,50 m" if success_candidates else "Menor erro disponível"
        for selection_label, rule, row in (
            (success_label, success_rule, success),
            ("Pior cena", "maior erro médio entre cenas", failure),
        ):
            scene_id = str(row["scene_id"])
            path = map_root / variant / "test" / f"{scene_id}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                srp_map = np.asarray(data["srp_map"], dtype=np.float32)
                xs = np.asarray(data["xs"], dtype=np.float32)
                ys = np.asarray(data["ys"], dtype=np.float32)
                microphones = np.asarray(data["microphone_positions_xy"], dtype=np.float32)
                reference = np.asarray(row["reference_xy"], dtype=np.float32)
            fig_path = output_dir / f"{method.lower().replace(' ', '_').replace('-', '_')}_{rule.split()[0]}_{scene_id}.png"
            plot_scene(
                fig_path,
                (8.0, 6.0, 3.0),
                microphones,
                reference,
                srp_map,
                xs,
                ys,
                predicted_xy=np.asarray(row["estimate_xy"], dtype=np.float32),
                title=f"{method}\n{scene_id} — {selection_label}: {float(row['mean_error_m']):.3f} m",
            )
            examples.append({"seed": seed, "method": method, "selection_rule": rule, "scene_id": scene_id, "mean_error_m": float(row["mean_error_m"]), "figure": str(fig_path)})
    return examples


def _wave_diagnostics(seeds: list[int], artifact_root: Path) -> list[dict]:
    output = []
    for seed in seeds:
        checkpoint = artifact_root / f"wave_seed{seed}" / "wave_unet" / "best.pt"
        environment_path = artifact_root / f"wave_seed{seed}" / "wave_unet" / "environment.json"
        if not checkpoint.exists():
            continue
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        evaluation_path = artifact_root / "full" / "wave_unet" / f"evaluation_test_{digest[:12]}.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
        environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
        output.append({
            "seed": seed,
            "wave_variant": f"wave_{digest[:12]}",
            "mean_mixture_si_sdr_db": evaluation.get("mean_mixture_si_sdr_db"),
            "mean_enhanced_si_sdr_db": evaluation.get("mean_enhanced_si_sdr_db"),
            "mean_si_sdr_change_db": evaluation.get("mean_si_sdr_change_db"),
            "mean_wave_inference_seconds": evaluation.get("mean_wave_inference_seconds"),
            "mean_total_seconds_per_scene": evaluation.get("mean_total_seconds_per_scene"),
            "peak_gpu_memory_mib": environment.get("peak_gpu_memory_mib"),
        })
    return output


def _processing_costs(seeds: list[int], artifact_root: Path, map_root: Path) -> list[dict]:
    output: list[dict] = []
    for variant in ("mixture",):
        for split in ("train", "validation", "test"):
            path = map_root / variant / f"timings_{split}.json"
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                output.append({
                    "seed": "shared",
                    "operation": "extração SRP-PHAT",
                    "variant": variant,
                    "split": split,
                    "scene_count": report["scene_count"],
                    "total_seconds": report["total_seconds"],
                    "mean_seconds_per_scene": report["total_seconds"] / max(report["scene_count"], 1),
                })
    for seed in seeds:
        checkpoint = artifact_root / f"wave_seed{seed}" / "wave_unet" / "best.pt"
        if not checkpoint.exists():
            continue
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]
        variant = f"wave_{digest}"
        for split in ("train", "validation", "test"):
            path = map_root / variant / f"timings_{split}.json"
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                output.append({
                    "seed": seed,
                    "operation": "extração SRP-PHAT",
                    "variant": variant,
                    "split": split,
                    "scene_count": report["scene_count"],
                    "total_seconds": report["total_seconds"],
                    "mean_seconds_per_scene": report["total_seconds"] / max(report["scene_count"], 1),
                })
            wave_eval_path = artifact_root / "full" / "wave_unet" / f"evaluation_{split}_{digest}.json"
            if wave_eval_path.exists():
                wave_eval = json.loads(wave_eval_path.read_text(encoding="utf-8"))
                scene_count = int(wave_eval["scene_count"])
                for operation, field in (
                    ("inferência Wave-U-Net", "mean_wave_inference_seconds"),
                    ("enhancement completo incluindo I/O e SI-SDR", "mean_total_seconds_per_scene"),
                ):
                    mean_seconds = float(wave_eval[field])
                    output.append({
                        "seed": seed,
                        "operation": operation,
                        "variant": variant,
                        "split": split,
                        "scene_count": scene_count,
                        "total_seconds": mean_seconds * scene_count,
                        "mean_seconds_per_scene": mean_seconds,
                    })
        history_path = artifact_root / f"wave_seed{seed}" / "wave_unet" / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            output.append({
                "seed": seed,
                "operation": "treino Wave-U-Net",
                "variant": variant,
                "split": "train/validation",
                "scene_count": None,
                "total_seconds": sum(float(row["seconds"]) for row in history),
                "mean_seconds_per_scene": None,
            })
        for run_name, label in ((f"cnn_mixture_seed{seed}", "treino CNN sobre mistura"), (f"cnn_wave_seed{seed}", "treino CNN sobre mapas Wave")):
            history_path = artifact_root / run_name / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                output.append({
                    "seed": seed,
                    "operation": label,
                    "variant": "mixture" if "mixture" in run_name else variant,
                    "split": "train/validation",
                    "scene_count": None,
                    "total_seconds": sum(float(row["seconds"]) for row in history),
                    "mean_seconds_per_scene": None,
                })
    return output


def _rt60_realization_note(artifact_root: Path) -> str | None:
    audit_path = artifact_root / "full" / "scene_audio_audit.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config_path = ROOT / audit.get("config", "configs/full.yaml")
    if not config_path.exists():
        return None

    from voice_localization.config import load_config
    from voice_localization.manifest import SceneRecord, read_jsonl

    config = load_config(config_path)
    lower, upper = config.simulation.rt60_range
    manifest_root = ROOT / config.dataset.manifest_root
    realized: list[float] = []
    for split in ("train", "validation", "test"):
        manifest_path = manifest_root / f"scenes_{split}.jsonl"
        if not manifest_path.exists():
            continue
        realized.extend(
            float(record.rt60_realized)
            for record in read_jsonl(manifest_path, SceneRecord)
            if record.rt60_realized is not None
        )
    if not realized:
        return None

    outside_count = sum(value < lower or value > upper for value in realized)
    ranges = audit.get("splits", {})
    observed = []
    split_labels = {"train": "treino", "validation": "validação", "test": "teste"}
    for split in ("train", "validation", "test"):
        values = ranges.get(split, {}).get("realized_rt60_seconds_range")
        if values:
            observed.append(f"{split_labels[split]}: {values[0]:.3f}–{values[1]:.3f} s")
    return (
        f"O RT60 solicitado foi {lower:.1f}–{upper:.1f} s. O RT60 realizado ficou fora dessa faixa em "
        f"{outside_count}/{len(realized)} cenas ({100 * outside_count / len(realized):.2f}%). "
        f"Faixas realizadas observadas — {'; '.join(observed)}."
    )


def _write_markdown(output_path: Path, overall: list[dict], paired: list[dict], runs: dict[int, dict[str, dict]], examples: list[dict], wave_diagnostics: list[dict], gate_path: Path, artifact_root: Path) -> None:
    lines = [
        "# Relatório final — localização de múltiplas fontes de voz",
        "",
        "## Protocolo",
        "",
        "Comparação no teste reservado depois de congelar modelos e regras na validação. Cada seed avalia as mesmas cenas; a incerteza pareada reamostra cenas inteiras e preserva juntas as fontes da mesma cena. O domínio é exclusivamente simulado: sala 8 × 6 × 3 m, microfones e fontes a 1,5 m, N=1–3, M=4/6/8, SNR solicitado 10–30 dB, RT60 solicitado 0,2–0,8 s e sincronização ideal.",
        "",
        "## Resultados agregados",
        "",
        "As linhas ‘média ± DP’ resumem as estimativas de cada seed; o erro é ponderado por fonte. ‘Avaliação com mapa pré-calculado’ mede leitura do mapa + decodificação (SRP) ou CNN + decodificação. ‘Extração SRP’ é somada para estimar a latência end-to-end de uma cena nova; a Wave-U-Net também é somada nos dois fluxos Wave. O detalhamento fica em `processing_costs.csv`.",
        "",
        "| Método | Erro médio (m) | Mediana (m) | P90 (m) | ≤0,50 m | Avaliação (ms/cena) | SRP (ms/cena) | Wave (ms/cena) | Fluxo total (ms/cena) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        if row["seed"] != "média ± DP":
            continue
        as_ms = lambda value: " ± ".join(f"{float(part) * 1000:.2f}" for part in str(value).split(" ± "))
        lines.append(
            f"| {row['method']} | {row['mean_error_m']} | {row['median_error_m']} | {row['p90_error_m']} | {row['within_0_5m']} | "
            f"{as_ms(row['mean_inference_seconds'])} | {as_ms(row['mean_map_extraction_seconds'])} | "
            f"{as_ms(row['mean_wave_inference_seconds'])} | {as_ms(row['mean_total_pipeline_seconds'])} |"
        )
    lines += ["", "## Comparações pareadas no teste", "", "Delta = candidato − referência; valor negativo favorece o candidato para erros e positivo favorece a proporção dentro de 0,50 m. O CSV também contém ICs separados por seed.", "", "| Seed/agregação | Referência | Candidato | Métrica | Delta | IC 95% |", "|---|---|---|---|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['seed']} | {row['reference']} | {row['candidate']} | {row['metric']} | {row['delta_candidate_minus_reference']:.4f} | [{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}] |")
    averaged_rows = [row for row in overall if row["seed"] == "média ± DP"]
    if averaged_rows:
        best_row = min(averaged_rows, key=lambda row: float(str(row["mean_error_m"]).split(" ± ")[0]))
        best_method = best_row["method"]
        if best_method == "SRP-PHAT":
            interpretation = "SRP-PHAT apresentou a menor estimativa média de erro entre os métodos avaliados."
        else:
            pooled = next((row for row in paired if row["seed"] == "média entre seeds" and row["reference"] == "SRP-PHAT" and row["candidate"] == best_method and row["metric"] == "mean_error_m"), None)
            if pooled and pooled["ci_95_high"] < 0:
                interpretation = f"{best_method} teve o menor erro médio e o IC pareado de 95% da diferença contra SRP-PHAT ficou abaixo de zero."
            elif pooled:
                interpretation = f"{best_method} teve o menor erro médio pontual, mas o IC pareado de 95% contra SRP-PHAT inclui zero; o resultado não sustenta afirmar uma melhoria clara."
            else:
                interpretation = f"{best_method} teve o menor erro médio pontual; consulte os ICs pareados e as tabelas estratificadas antes de interpretar o ganho."
        lines += ["", "## Interpretação dos resultados", "", interpretation, "A comparação deve ser lida junto das diferenças entre seeds e das faixas de N, M, geometria, SNR e RT60 em `metrics_stratified.csv`; uma média global não implica vantagem em todas as condições."]
    lines += ["", "## Diagnóstico de enhancement", "", "SI-SDR avalia a reconstrução de áudio e deve ser lido junto com os erros de localização.", "", "| Seed | SI-SDR mistura (dB) | SI-SDR aprimorado (dB) | Δ SI-SDR (dB) | Inferência Wave (ms/cena) | Tempo total Wave (s/cena) | Pico GPU (MiB) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in wave_diagnostics:
        lines.append(
            f"| {row['seed']} | {row.get('mean_mixture_si_sdr_db')} | {row.get('mean_enhanced_si_sdr_db')} | {row.get('mean_si_sdr_change_db')} | "
            f"{None if row.get('mean_wave_inference_seconds') is None else round(row['mean_wave_inference_seconds'] * 1000, 3)} | {row.get('mean_total_seconds_per_scene')} | {row.get('peak_gpu_memory_mib')} |"
        )
    lines += ["", "## Custos medidos", "", "Os custos por operação e split, incluindo extração de mapas e treinamento, estão em `processing_costs.csv`. As curvas de treino também reportam duração por época. A latência de localização na tabela principal é separada da extração SRP pré-calculada."]
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        lines += ["", "## Regra de inclusão da Wave-U-Net", "", f"Validação: `wave_branch_passes_validation_gate = {gate['wave_branch_passes_validation_gate']}`; média do delta pareado Wave-U-Net + SRP-PHAT menos SRP-PHAT = {gate['mean_wave_srp_delta_m_across_seeds']:.4f} m. Essa decisão foi registrada antes da avaliação no teste."]
    lines += ["", "## Figuras e limitações", "", "`error_distribution.png`, `performance_by_condition.png` e `learning_curves.png` mostram a dispersão e as curvas. Para cada método, os exemplos espaciais são escolhidos na seed 42 (ou na primeira disponível): menor erro médio entre as cenas até 0,50 m; se não houver nenhuma, menor erro disponível; e maior erro médio entre as cenas.", "", "Os resultados descrevem apenas o domínio simulado e não demonstram generalização para salas diferentes ou gravações reais. SI-SDR mede reconstrução de áudio e não substitui a métrica de localização; uma melhoria acústica isolada não prova que as pistas espaciais foram preservadas."]
    rt60_note = _rt60_realization_note(artifact_root)
    if rt60_note:
        lines += ["", rt60_note]
    lines += ["", f"Seeds avaliadas: {', '.join(map(str, runs))}. Cenas por seed e método: {len(runs[next(iter(runs))][METHODS[0]]['rows'])}. Exemplos espaciais salvos: {len(examples)}.", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "artifacts" / "final")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "final" / "report")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--map-root", type=Path, default=ROOT / "data" / "full" / "maps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--validation-gate", type=Path, default=ROOT / "artifacts" / "final" / "wave_validation_gate.json")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples precisa ser >= 100")
    runs = _load_runs(args.results_root, args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    overall = _overall_table(runs, args.artifact_root, args.map_root)
    stratified = _stratified_table(runs)
    paired = _paired_bootstrap(runs, args.bootstrap_samples, args.bootstrap_seed)
    _write_csv(args.output / "metrics_overall.csv", overall)
    _write_csv(args.output / "metrics_stratified.csv", stratified)
    _write_csv(args.output / "paired_test_bootstrap.csv", paired)
    wave_diagnostics = _wave_diagnostics(args.seeds, args.artifact_root)
    _write_csv(args.output / "wave_diagnostics.csv", wave_diagnostics)
    processing_costs = _processing_costs(args.seeds, args.artifact_root, args.map_root)
    _write_csv(args.output / "processing_costs.csv", processing_costs)
    _make_plots(runs, args.output, args.artifact_root)
    examples = _make_spatial_examples(runs, args.output / "spatial_examples", args.map_root, args.artifact_root)
    _write_csv(args.output / "spatial_examples.csv", examples)
    _write_markdown(args.output / "final_report.md", overall, paired, runs, examples, wave_diagnostics, args.validation_gate, args.artifact_root)
    print(json.dumps({"output": str(args.output), "seeds": args.seeds, "scene_count": len(runs[args.seeds[0]][METHODS[0]]["rows"]), "spatial_examples": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
