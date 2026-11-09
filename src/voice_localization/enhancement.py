"""Wave-U-Net multicanal para redução de ruído sem descartar fontes."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .audio import read_multichannel_audio
from .config import ProjectConfig, save_resolved_config
from .manifest import SceneRecord, read_jsonl, write_jsonl
from .seed import seed_everything


def build_wave_unet(input_channels: int, output_channels: int, base_channels: int, depth: int):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para construir a Wave-U-Net") from exc

    class WaveUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.stem = nn.Conv1d(input_channels, base_channels, kernel_size=15, padding=7)
            channels = [base_channels * (2 ** (level + 1)) for level in range(depth)]
            previous = base_channels
            self.encoders = nn.ModuleList()
            for current in channels:
                self.encoders.append(nn.Conv1d(previous, current, kernel_size=15, stride=2, padding=7))
                previous = current
            self.bottleneck = nn.Conv1d(previous, previous, kernel_size=5, padding=2)
            self.decoders = nn.ModuleList()
            for skip_channels in [base_channels, *channels[:-1]][::-1]:
                self.decoders.append(
                    nn.Conv1d(previous + skip_channels, skip_channels, kernel_size=5, padding=2)
                )
                previous = skip_channels
            self.output = nn.Conv1d(base_channels, output_channels, kernel_size=1)
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, features):
            mixture = features[:, : self.output_channels]
            x = self.activation(self.stem(features))
            skips = [x]
            for index, layer in enumerate(self.encoders):
                x = self.activation(layer(x))
                if index < len(self.encoders) - 1:
                    skips.append(x)
            x = self.activation(self.bottleneck(x))
            for layer, skip in zip(self.decoders, reversed(skips)):
                x = functional.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
                x = self.activation(layer(torch.cat((x, skip), dim=1)))
            return mixture + self.output(x)

    return WaveUNet()


def _wave_features(record: SceneRecord, config: ProjectConfig) -> tuple[np.ndarray, np.ndarray, float]:
    if not record.audio_path or not record.clean_audio_path:
        raise ValueError(f"cena sem mistura ou alvo limpo: {record.scene_id}")
    mixture = read_multichannel_audio(record.audio_path, config.simulation.audio.sample_rate)
    target = read_multichannel_audio(record.clean_audio_path, config.simulation.audio.sample_rate)
    if mixture.shape != target.shape or mixture.shape[0] != record.n_microphones:
        raise ValueError(f"canais ou duração inconsistentes: {record.scene_id}")
    if record.n_microphones > config.enhancement.max_microphones:
        raise ValueError(f"M excede enhancement.max_microphones: {record.scene_id}")

    rms = max(float(np.sqrt(np.mean(mixture.astype(np.float64) ** 2))), 1e-6)
    max_mics = config.enhancement.max_microphones
    samples = mixture.shape[1]
    features = np.zeros((max_mics * 4, samples), dtype=np.float32)
    target_padded = np.zeros((max_mics, samples), dtype=np.float32)
    features[:max_mics] = 0.0
    features[: record.n_microphones] = mixture / rms
    features[max_mics : 2 * max_mics, : ] = 0.0
    features[max_mics : max_mics + record.n_microphones] = 1.0
    positions = np.asarray(record.microphone_positions_xy, dtype=np.float32)
    width, length = record.room_dimensions[:2]
    features[2 * max_mics : 2 * max_mics + record.n_microphones] = (
        positions[:, 0, None] / width
    )
    features[3 * max_mics : 3 * max_mics + record.n_microphones] = (
        positions[:, 1, None] / length
    )
    target_padded[: record.n_microphones] = target / rms
    return features, target_padded, rms


def _dataset(records: list[SceneRecord], config: ProjectConfig):
    import torch
    from torch.utils.data import Dataset

    class WaveSceneDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            features, target, _ = _wave_features(records[index], config)
            return torch.from_numpy(features), torch.from_numpy(target), records[index].n_microphones

    return WaveSceneDataset()


def _wave_model(config: ProjectConfig):
    return build_wave_unet(
        input_channels=4 * config.enhancement.max_microphones,
        output_channels=config.enhancement.max_microphones,
        base_channels=config.enhancement.base_channels,
        depth=config.enhancement.depth,
    )


def _sisdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SI-SDR médio por canal, preservando a imagem multicanal limpa como alvo."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = reference - np.mean(reference, axis=1, keepdims=True)
    estimate = estimate - np.mean(estimate, axis=1, keepdims=True)
    projection_scale = np.sum(estimate * reference, axis=1, keepdims=True) / np.maximum(
        np.sum(reference**2, axis=1, keepdims=True), 1e-12
    )
    projected = projection_scale * reference
    residual = estimate - projected
    ratios = np.sum(projected**2, axis=1) / np.maximum(np.sum(residual**2, axis=1), 1e-12)
    return float(np.mean(10.0 * np.log10(np.maximum(ratios, 1e-12))))


def train_wave_unet(
    config: ProjectConfig,
    run_name: str | None = None,
    resume: str | Path | None = None,
) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para treinar a Wave-U-Net") from exc

    seed_everything(config.train.seed)
    manifest_root = Path(config.dataset.manifest_root)
    train_records = list(read_jsonl(manifest_root / "scenes_train.jsonl", SceneRecord))
    validation_records = list(read_jsonl(manifest_root / "scenes_validation.jsonl", SceneRecord))
    if not train_records or not validation_records:
        raise ValueError("treino e validação precisam ter cenas")
    train_data, validation_data = _dataset(train_records, config), _dataset(validation_records, config)
    train_loader = DataLoader(train_data, batch_size=config.enhancement.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=config.enhancement.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if config.train.device == "auto" and torch.cuda.is_available() else config.train.device)
    if config.train.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi selecionada, mas não está disponível")
    model = _wave_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.enhancement.learning_rate)
    run_dir = Path(config.train.run_root) / (run_name or config.profile) / "wave_unet"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir / "config.json")
    environment = {
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_scenes": len(train_records),
        "validation_scenes": len(validation_records),
        "max_microphones": config.enhancement.max_microphones,
        "base_channels": config.enhancement.base_channels,
        "depth": config.enhancement.depth,
        "seed": config.train.seed,
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    def run_epoch(loader, training: bool) -> float:
        model.train(training)
        losses: list[float] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for features, targets, microphone_counts in loader:
                features, targets = features.to(device), targets.to(device)
                mask = torch.arange(config.enhancement.max_microphones, device=device)[None, :, None]
                mask = (mask < microphone_counts.to(device)[:, None, None]).to(targets.dtype)
                prediction = model(features)
                loss = (torch.abs(prediction - targets) * mask).sum() / (mask.sum() * targets.shape[-1])
                if not torch.isfinite(loss):
                    raise FloatingPointError("loss da Wave-U-Net não finita; treino interrompido")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    best_loss = float("inf")
    history: list[dict] = []
    start_epoch = 1
    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        if history:
            best_loss = min(float(row["validation_l1_normalized"]) for row in history)
        if "torch_random_state" in resume_state:
            torch.set_rng_state(resume_state["torch_random_state"].cpu())
        if device.type == "cuda" and resume_state.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_random_state_all"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config.enhancement.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(train_loader, True)
        validation_loss = run_epoch(validation_loader, False)
        row = {
            "epoch": epoch,
            "train_l1_normalized": train_loss,
            "validation_l1_normalized": validation_loss,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"[{run_name or config.profile}/wave_unet] época {epoch}/{config.enhancement.epochs}: "
            f"L1 treino={train_loss:.5f}, L1 validação={validation_loss:.5f}, {row['seconds']:.1f} s",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "history": history,
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        temporary_last = run_dir / "last.pt.tmp"
        torch.save(state, temporary_last)
        temporary_last.replace(run_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            temporary_best = run_dir / "best.pt.tmp"
            torch.save(state, temporary_best)
            temporary_best.replace(run_dir / "best.pt")
    if device.type == "cuda":
        environment["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return run_dir / "best.pt"


def enhance_split(config: ProjectConfig, checkpoint_path: str | Path, split: str) -> Path:
    import torch

    device = torch.device("cuda" if config.train.device == "auto" and torch.cuda.is_available() else config.train.device)
    if config.train.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    variant = f"wave_{checkpoint_digest[:12]}"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _wave_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    manifest_path = Path(config.dataset.manifest_root) / f"scenes_{split}.jsonl"
    records = list(read_jsonl(manifest_path, SceneRecord))
    output_dir = Path(config.dataset.output_root) / "enhanced" / variant / split
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(config.train.run_root)
        / config.profile
        / "wave_unet"
        / f"evaluation_{split}_{checkpoint_digest[:12]}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = report_path.with_suffix(".progress.jsonl")
    cached_rows: dict[str, dict] = {}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        cached_rows.update({str(row["scene_id"]): row for row in previous.get("rows", [])})
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    cached_rows[str(row["scene_id"])] = row
    rows: list[dict] = []
    from .simulation import _write_multichannel_wav

    with progress_path.open("a", encoding="utf-8") as progress:
        for index, record in enumerate(records):
            path = output_dir / f"{record.scene_id}.wav"
            cached = cached_rows.get(record.scene_id)
            if path.exists() and cached is not None:
                row = cached
            else:
                started = time.perf_counter()
                features, _, rms = _wave_features(record, config)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.no_grad():
                    prediction = model(torch.from_numpy(features[None]).to(device))[0, : record.n_microphones]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - inference_started
                waveform = prediction.float().cpu().numpy() * rms
                _write_multichannel_wav(path, waveform.astype(np.float32), config.simulation.audio.sample_rate)
                mixture = read_multichannel_audio(record.audio_path, config.simulation.audio.sample_rate)
                target = read_multichannel_audio(record.clean_audio_path, config.simulation.audio.sample_rate)
                row = {
                    "scene_id": record.scene_id,
                    "mixture_si_sdr_db": _sisdr(target, mixture),
                    "enhanced_si_sdr_db": _sisdr(target, waveform),
                    "inference_seconds": inference_seconds,
                    "seconds": time.perf_counter() - started,
                    "n_sources": record.n_sources,
                    "n_microphones": record.n_microphones,
                    "geometry": record.geometry,
                    "snr_requested_db": record.snr_requested_db,
                    "rt60_requested": record.rt60_requested,
                }
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.flush()
            rows.append(row)
            records[index] = SceneRecord(
                **{
                    **record.__dict__,
                    "enhanced_audio_path": str(path.resolve()),
                    "metadata": {**record.metadata, "enhancement_checkpoint_sha256": checkpoint_digest},
                }
            )
            if (index + 1) % 500 == 0 or index + 1 == len(records):
                print(f"[{split}/{variant}] áudios aprimorados: {index + 1}/{len(records)}", flush=True)
    write_jsonl(records, manifest_path)
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(
        json.dumps(
            {
                "split": split,
                "checkpoint_sha256": checkpoint_digest,
                "scene_count": len(rows),
                "mean_mixture_si_sdr_db": float(np.mean([row["mixture_si_sdr_db"] for row in rows])),
                "mean_enhanced_si_sdr_db": float(np.mean([row["enhanced_si_sdr_db"] for row in rows])),
                "mean_si_sdr_change_db": float(
                    np.mean([row["enhanced_si_sdr_db"] - row["mixture_si_sdr_db"] for row in rows])
                ),
                "mean_wave_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
                "mean_total_seconds_per_scene": float(np.mean([row["seconds"] for row in rows])),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_tmp.replace(report_path)
    return manifest_path
