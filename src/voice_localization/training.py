"""Loop explícito de treino, validação, checkpoints e TensorBoard."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import numpy as np

from .config import ProjectConfig, save_resolved_config
from .dataset import make_torch_dataset
from .manifest import SceneRecord, dataset_id, read_jsonl
from .metrics import decode_peaks, localization_errors, localization_metrics
from .model import build_model
from .seed import seed_everything
from .visualization import plot_scene


def _device(config: ProjectConfig):
    import torch

    if config.train.device == "cpu":
        return torch.device("cpu")
    if config.train.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device=cuda, mas CUDA não está disponível")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collate(batch: list[dict]) -> dict:
    import torch

    return {
        "input": torch.stack([sample["input"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
        "n_sources": [int(sample["n_sources"]) for sample in batch],
        "positions": [sample["positions"].numpy() for sample in batch],
        "microphone_positions_xy": [sample["microphone_positions_xy"].numpy() for sample in batch],
        "xs": batch[0]["xs"].numpy(),
        "ys": batch[0]["ys"].numpy(),
        "scene_id": [sample["scene_id"] for sample in batch],
    }


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(path))
    except (ImportError, ModuleNotFoundError):
        return _NullWriter()


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _localization_batch_metrics(
    logits, batch: dict, tolerance_m: float = 0.50, suppression_radius_m: float = 0.30
) -> list[dict]:
    probabilities = logits.detach().sigmoid().cpu().numpy()[:, 0]
    output: list[dict] = []
    for probability, reference, n_sources in zip(probabilities, batch["positions"], batch["n_sources"]):
        estimate = decode_peaks(probability, batch["xs"], batch["ys"], n_sources, suppression_radius_m)
        row = localization_metrics(reference, estimate, tolerance_m).to_dict()
        row["source_errors_m"] = localization_errors(reference, estimate).tolist()
        row["reference_xy"] = np.asarray(reference, dtype=np.float32).tolist()
        row["estimate_xy"] = np.asarray(estimate, dtype=np.float32).tolist()
        output.append(row)
    return output


def _mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def _save_intermediate_figure(model, loader, device, output_path: Path, room_dimensions: tuple[float, float, float]) -> None:
    """Salva uma amostra fixa para acompanhar o aprendizado no próprio framework."""
    import torch

    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        logits = model(batch["input"].to(device))
    probability = logits[0, 0].sigmoid().cpu().numpy()
    estimate = decode_peaks(probability, batch["xs"], batch["ys"], batch["n_sources"][0])
    plot_scene(
        output_path,
        room_dimensions,
        batch["microphone_positions_xy"][0],
        batch["positions"][0],
        batch["input"][0, 0].numpy(),
        batch["xs"],
        batch["ys"],
        predicted_xy=estimate,
        title="Validação — previsão da CNN",
    )


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> tuple[float, dict]:
    import torch

    model.train(training)
    losses: list[float] = []
    metric_rows: list[dict] = []
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        metric_rows.extend(_localization_batch_metrics(logits, batch))
    return float(np.mean(losses)), {
        "mean_error_m": _mean_metric(metric_rows, "mean_error_m"),
        "median_error_m": _mean_metric(metric_rows, "median_error_m"),
        "p90_error_m": _mean_metric(metric_rows, "p90_error_m"),
        "within_tolerance": _mean_metric(metric_rows, "within_tolerance"),
    }


def _positive_weight(dataset) -> float:
    positives = 0.0
    total = 0.0
    for index in range(len(dataset)):
        target = dataset[index]["target"].numpy()
        positives += float(np.count_nonzero(target >= 0.5))
        total += float(target.size)
    if positives == 0 or positives == total:
        return 1.0
    return float((total - positives) / positives)


def train(config: ProjectConfig, run_name: str | None = None, resume: str | Path | None = None) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para executar o treinamento") from exc
    seed_everything(config.train.seed)
    suffix = "" if config.dataset.map_variant == "mixture" else f"_{config.dataset.map_variant}"
    train_manifest = Path(config.dataset.manifest_root) / f"scenes_train{suffix}.jsonl"
    validation_manifest = Path(config.dataset.manifest_root) / f"scenes_validation{suffix}.jsonl"
    if not train_manifest.exists() or not validation_manifest.exists():
        raise FileNotFoundError("manifestos de treino e validação são necessários")
    train_dataset = make_torch_dataset(train_manifest)
    validation_dataset = make_torch_dataset(validation_manifest)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("datasets de treino e validação não podem ser vazios")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=_collate,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=_collate,
        pin_memory=False,
    )
    device = _device(config)
    model = build_model().to(device)
    positive_weight = config.train.positive_weight or _positive_weight(train_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.train.scheduler_factor,
        patience=config.train.scheduler_patience,
    )
    run_name = run_name or config.profile
    run_dir = Path(config.train.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "config.json")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": str(device),
                "positive_weight": positive_weight,
                "train_dataset_id": dataset_id(list(read_jsonl(train_manifest, SceneRecord))),
                "validation_dataset_id": dataset_id(list(read_jsonl(validation_manifest, SceneRecord))),
                "seed": config.train.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    writer = _writer(run_dir / "tensorboard")
    best_error = float("inf")
    stale_epochs = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = list(state.get("history", []))
        start_epoch = int(state.get("epoch", 0)) + 1
        if history:
            best_error = min(float(row["validation"]["mean_error_m"]) for row in history)
        if "random_state" in state:
            torch.set_rng_state(state["random_state"])
    for epoch in range(start_epoch, config.train.epochs + 1):
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            validation_loss, validation_metrics = run_epoch(model, validation_loader, criterion, optimizer, device, False)
        scheduler.step(validation_metrics["mean_error_m"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"[{run_name}] época {epoch}/{config.train.epochs}: loss treino={train_loss:.4f}, "
            f"loss validação={validation_loss:.4f}, erro validação={validation_metrics['mean_error_m']:.3f} m, "
            f"{row['seconds']:.1f} s",
            flush=True,
        )
        for prefix, loss, metrics in (("train", train_loss, train_metrics), ("validation", validation_loss, validation_metrics)):
            writer.add_scalar(f"{prefix}/loss", loss, epoch)
            for key, value in metrics.items():
                writer.add_scalar(f"{prefix}/{key}", value, epoch)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "history": history,
            "random_state": torch.get_rng_state(),
        }
        torch.save(state, last_path)
        current_error = validation_metrics["mean_error_m"]
        if epoch == 1 or current_error <= best_error:
            _save_intermediate_figure(
                model,
                validation_loader,
                device,
                run_dir / f"validation_epoch_{epoch:03d}.png",
                config.simulation.room.dimensions,
            )
        if current_error < best_error:
            best_error = current_error
            stale_epochs = 0
            torch.save(state, best_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config.train.early_stopping_patience:
            break
    writer.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path


def smoke_train(config: ProjectConfig) -> Path:
    """Treina alguns passos para verificar gradientes e checkpoint."""
    original_epochs = config.train.epochs
    original_batch = config.train.batch_size
    config.train.epochs = 1
    config.train.batch_size = min(original_batch, 4)
    try:
        return train(config, run_name="smoke")
    finally:
        config.train.epochs = original_epochs
        config.train.batch_size = original_batch


def validate_small_set_learning(
    config: ProjectConfig, scenes: int = 8, steps: int = 500
) -> dict:
    """Verifica que a CNN consiga ajustar um subconjunto pequeno de treino."""
    import torch
    from torch.utils.data import DataLoader, Subset

    if scenes < 2 or steps < 1 or steps > 500:
        raise ValueError("use pelo menos duas cenas e entre 1 e 500 passos")
    seed_everything(config.train.seed)
    suffix = "" if config.dataset.map_variant == "mixture" else f"_{config.dataset.map_variant}"
    manifest = Path(config.dataset.manifest_root) / f"scenes_train{suffix}.jsonl"
    dataset = make_torch_dataset(manifest)
    if len(dataset) < scenes:
        raise ValueError(f"o conjunto contém {len(dataset)} cenas, mas foram solicitadas {scenes}")
    subset = Subset(dataset, list(range(scenes)))
    loader = DataLoader(
        subset,
        batch_size=min(config.train.batch_size, scenes),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    device = _device(config)
    model = build_model().to(device)
    positive_weight = _positive_weight(subset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=0.0)
    initial_loss, initial_metrics = run_epoch(model, loader, criterion, optimizer, device, False)
    batches = list(loader)
    for step in range(steps):
        batch = batches[step % len(batches)]
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final_loss, final_metrics = run_epoch(model, loader, criterion, optimizer, device, False)
    report = {
        "scene_count": scenes,
        "optimization_steps": steps,
        "device": str(device),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "initial_train_mean_error_m": initial_metrics["mean_error_m"],
        "final_train_mean_error_m": final_metrics["mean_error_m"],
        "loss_decreased": final_loss < initial_loss,
        "localization_error_decreased": final_metrics["mean_error_m"] < initial_metrics["mean_error_m"],
    }
    output_dir = Path(config.train.run_root) / config.profile / "learning_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report, "config": config.to_dict()}, output_dir / "overfit.pt")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
