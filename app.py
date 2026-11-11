"""Demonstração local das quatro linhas de localização em cenas simuladas."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

# Load PyTorch first so Windows reuses its OpenMP runtime instead of loading a
# second copy through the environment's MKL-backed NumPy libraries.
import torch
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Localização de fontes de voz", layout="wide")


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Carregando modelos congelados...")
def _load_models(seed: int, config_path: str):
    from voice_localization.config import load_config
    from voice_localization.enhancement import _wave_model
    from voice_localization.model import build_model

    config = load_config(config_path)
    device = torch.device("cuda" if config.train.device == "auto" and torch.cuda.is_available() else config.train.device)
    if config.train.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi selecionada, mas não está disponível")

    wave_path = ROOT / config.train.run_root / f"wave_seed{seed}" / "wave_unet" / "best.pt"
    mixture_cnn_path = ROOT / config.train.run_root / f"cnn_mixture_seed{seed}" / "best.pt"
    wave_cnn_path = ROOT / config.train.run_root / f"cnn_wave_seed{seed}" / "best.pt"
    missing = [str(path) for path in (wave_path, mixture_cnn_path, wave_cnn_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Checkpoints congelados ausentes:\n" + "\n".join(missing))

    wave_hash = _checkpoint_digest(wave_path)
    wave_variant = f"wave_{wave_hash[:12]}"
    models = {}
    wave = _wave_model(config).to(device)
    wave.load_state_dict(torch.load(wave_path, map_location=device, weights_only=False)["model"])
    wave.eval()
    models["wave"] = wave
    for key, path in (("cnn_mixture", mixture_cnn_path), ("cnn_wave", wave_cnn_path)):
        state = torch.load(path, map_location=device, weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(state.get("model", state))
        model.eval()
        models[key] = model

    round2_seed = seed if seed in (42, 43) else 42
    round2_path = ROOT / "artifacts" / "round2" / "confirm" / f"seed{round2_seed}" / "best.pt"
    if round2_path.exists():
        state = torch.load(round2_path, map_location=device, weights_only=False)
        round2_model = build_model((1, 2, 4, 8)).to(device)
        round2_model.load_state_dict(state["model"])
        round2_model.eval()
        models["cnn_round2"] = round2_model
        models["round2_seed"] = round2_seed

    normalizers = {}
    for variant in ("mixture", wave_variant):
        path = ROOT / config.dataset.output_root / "maps" / variant / "normalizer.npz"
        if not path.exists():
            raise FileNotFoundError(f"Normalizador de treino ausente: {path}")
        with np.load(path) as values:
            normalizers[variant] = (float(values["mean"]), float(values["std"]))
    return config, device, models, normalizers, wave_variant, wave_hash


def _normalize_map(srp_map: np.ndarray, normalizer: tuple[float, float]) -> np.ndarray:
    mean, std = normalizer
    return ((srp_map.astype(np.float32) - mean) / max(std, 1e-8)).astype(np.float32)


def _predict_cnn(model, device, srp_map: np.ndarray, n_sources: int, xs: np.ndarray, ys: np.ndarray, radius: float):
    from voice_localization.metrics import decode_peaks

    features = np.stack(
        (srp_map, np.full_like(srp_map, n_sources / 3.0, dtype=np.float32)), axis=0
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(features[None]).to(device))
    probability = logits[0, 0].sigmoid().cpu().numpy()
    estimate = decode_peaks(probability, xs, ys, n_sources, radius)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return estimate, elapsed, probability


def _render_scene_plots(record, results, xs, ys, methods):
    """Render maps in an isolated Matplotlib process, separate from PyTorch."""
    with tempfile.TemporaryDirectory(prefix="voz-demo-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = temporary_root / "scene.npz"
        output_dir = temporary_root / "plots"
        arrays = {
            "xs": np.asarray(xs, dtype=np.float32),
            "ys": np.asarray(ys, dtype=np.float32),
            "microphones": np.asarray(record.microphone_positions_xy, dtype=np.float32),
            "truth": np.asarray(record.source_positions_xy, dtype=np.float32),
            "room_dimensions": np.asarray(record.room_dimensions, dtype=np.float32),
        }
        method_specs = []
        for index, (result_key, label) in enumerate(methods):
            arrays[f"map_{index}"] = np.asarray(results[result_key]["display_map"], dtype=np.float32)
            arrays[f"estimate_{index}"] = np.asarray(results[result_key]["estimate"], dtype=np.float32)
            method_specs.append({"index": index, "result_key": result_key, "label": label})
        np.savez(input_path, **arrays)
        metadata_path = temporary_root / "methods.json"
        metadata_path.write_text(json.dumps(method_specs, ensure_ascii=False), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "render_demo_plots.py"),
                "--input", str(input_path),
                "--metadata", str(metadata_path),
                "--output-dir", str(output_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise RuntimeError(f"Falha ao renderizar os mapas da demonstração: {detail}")
        return {
            result_key: (output_dir / f"method_{index}.png").read_bytes()
            for index, result_key in enumerate(result_key for result_key, _ in methods)
        }


def _make_scene(n_sources: int, n_microphones: int, geometry: str, snr: float, rt60: float, counter: int):
    from voice_localization.config import load_config
    from voice_localization.manifest import SceneRecord

    config = load_config(ROOT / "configs" / "full.yaml")
    if geometry == "circle_r6":
        # This fixed orientation exactly reproduces the Round 6 finalist.
        config.simulation.geometries = (*config.simulation.geometries, "circle_r6")
    if n_sources not in config.simulation.n_sources:
        raise ValueError("N fora do domínio do experimento")
    if n_microphones not in config.simulation.n_microphones:
        raise ValueError("M fora do domínio do experimento")
    if geometry not in config.simulation.geometries:
        raise ValueError("Geometria fora do domínio do experimento")
    config.simulation.snr_range_db = (float(snr), float(snr))
    config.simulation.rt60_range = (float(rt60), float(rt60))
    config.dataset.output_root = str(ROOT / "data" / "demo")
    config.dataset.manifest_root = str(ROOT / "data" / "demo" / "manifests")
    # Encode N, M and geometry in the scene index so the generator's balanced
    # round-robin selector creates exactly the controls chosen in the UI.
    n_index = config.simulation.n_sources.index(n_sources)
    m_index = config.simulation.n_microphones.index(n_microphones)
    geometry_index = config.simulation.geometries.index(geometry)
    conditions_per_geometry = (
        len(config.simulation.n_sources) * len(config.simulation.n_microphones)
    )
    counter_offset = counter + (10_000 if geometry == "circle_r6" else 0)
    scene_index = (
        geometry_index * conditions_per_geometry
        + m_index * len(config.simulation.n_sources)
        + n_index
        + counter_offset * conditions_per_geometry * len(config.simulation.geometries)
    )
    output_dir = Path(config.dataset.output_root) / "scenes" / "train"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_demo_scene.py"),
        "--n-sources", str(n_sources),
        "--n-microphones", str(n_microphones),
        "--geometry", geometry,
        "--snr-db", str(snr),
        "--rt60", str(rt60),
        "--counter", str(counter),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(f"Falha ao simular a cena de demonstração: {detail}")
    metadata_path = output_dir / f"train-{scene_index:06d}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"O gerador terminou sem criar os metadados esperados: {metadata_path}")
    record = SceneRecord(**json.loads(metadata_path.read_text(encoding="utf-8")))
    return record, config


def _run_comparison(record, config, device, models, normalizers, wave_variant):
    from voice_localization.audio import read_multichannel_audio
    from voice_localization.enhancement import _sisdr, _wave_features
    from scipy.stats import rankdata
    from voice_localization.metrics import decode_local_peaks, decode_peaks, localization_metrics
    from voice_localization.spatial import srp_phat_map

    room = config.simulation.room
    audio_cfg = config.simulation.audio
    mic_xy = np.asarray(record.microphone_positions_xy, dtype=np.float32)
    microphones = np.column_stack(
        (mic_xy, np.full(len(mic_xy), room.microphone_height, dtype=np.float32))
    )
    mixture = read_multichannel_audio(record.audio_path, audio_cfg.sample_rate)
    wave_features, _, rms = _wave_features(record, config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wave_started = time.perf_counter()
    with torch.no_grad():
        enhanced = models["wave"](torch.from_numpy(wave_features[None]).to(device))[0, : record.n_microphones]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wave_seconds = time.perf_counter() - wave_started
    enhanced = enhanced.float().cpu().numpy() * rms

    srp_started = time.perf_counter()
    mixture_map, grid = srp_phat_map(
        mixture, microphones, audio_cfg.sample_rate, room.dimensions,
        resolution_m=config.grid.resolution_m, speed_of_sound=room.speed_of_sound,
        window=audio_cfg.stft_window, hop=audio_cfg.stft_hop, fft_size=audio_cfg.fft_size,
    )
    xs, ys = grid.xs, grid.ys
    mixture_srp_seconds = time.perf_counter() - srp_started
    mixture_normalized = _normalize_map(mixture_map, normalizers["mixture"])
    mix_srp_started = time.perf_counter()
    srp_estimate = decode_peaks(
        mixture_normalized, xs, ys, record.n_sources, config.grid.peak_suppression_radius_m
    )
    srp_decode_seconds = time.perf_counter() - mix_srp_started
    cnn_mix_estimate, cnn_mix_seconds, cnn_mix_map = _predict_cnn(
        models["cnn_mixture"], device, mixture_normalized, record.n_sources, xs, ys,
        config.grid.peak_suppression_radius_m,
    )
    round2_outputs = None
    if "cnn_round2" in models:
        round2_estimate, round2_seconds, round2_map = _predict_cnn(
            models["cnn_round2"], device, mixture_normalized, record.n_sources, xs, ys,
            0.60,
        )
        fused_srp = rankdata(mixture_normalized.ravel(), method="average").reshape(mixture_normalized.shape)
        fused_srp = fused_srp.astype(np.float32) / fused_srp.size
        fused_map = 0.75 * fused_srp + 0.25 * round2_map
        fused_estimate = decode_local_peaks(fused_map, xs, ys, record.n_sources, 0.60)
        round2_estimate = decode_local_peaks(round2_map, xs, ys, record.n_sources, 0.60)
        round2_outputs = (round2_estimate, round2_map, round2_seconds, fused_estimate, fused_map)

    enhanced_srp_started = time.perf_counter()
    enhanced_map, _ = srp_phat_map(
        enhanced, microphones, audio_cfg.sample_rate, room.dimensions,
        resolution_m=config.grid.resolution_m, speed_of_sound=room.speed_of_sound,
        window=audio_cfg.stft_window, hop=audio_cfg.stft_hop, fft_size=audio_cfg.fft_size,
    )
    enhanced_srp_seconds = time.perf_counter() - enhanced_srp_started
    enhanced_normalized = _normalize_map(enhanced_map, normalizers[wave_variant])
    wave_srp_started = time.perf_counter()
    wave_srp_estimate = decode_peaks(
        enhanced_normalized, xs, ys, record.n_sources, config.grid.peak_suppression_radius_m
    )
    wave_srp_decode_seconds = time.perf_counter() - wave_srp_started
    cnn_wave_estimate, cnn_wave_seconds, cnn_wave_map = _predict_cnn(
        models["cnn_wave"], device, enhanced_normalized, record.n_sources, xs, ys,
        config.grid.peak_suppression_radius_m,
    )

    reference = np.asarray(record.source_positions_xy, dtype=np.float32)
    methods = {
        "SRP-PHAT": (srp_estimate, mixture_map, mixture_srp_seconds + srp_decode_seconds),
        "CNN sobre mistura": (cnn_mix_estimate, cnn_mix_map, mixture_srp_seconds + cnn_mix_seconds),
        "Wave-U-Net + SRP-PHAT": (
            wave_srp_estimate, enhanced_map, wave_seconds + enhanced_srp_seconds + wave_srp_decode_seconds,
        ),
        "Wave-U-Net + CNN": (
            cnn_wave_estimate, cnn_wave_map, wave_seconds + enhanced_srp_seconds + cnn_wave_seconds,
        ),
    }
    if round2_outputs is not None:
        estimate, probability, latency, fused_estimate, fused_map = round2_outputs
        methods[f"CNN contexto amplo (seed {models['round2_seed']})"] = (estimate, probability, mixture_srp_seconds + latency)
        methods[f"SRP + CNN 25% (seed {models['round2_seed']})"] = (fused_estimate, fused_map, mixture_srp_seconds + latency)
    output = {}
    for name, (estimate, displayed_map, latency) in methods.items():
        metrics = localization_metrics(reference, estimate)
        output[name] = {
            "estimate": estimate,
            "mean_error_m": metrics.mean_error_m,
            "median_error_m": metrics.median_error_m,
            "p90_error_m": metrics.p90_error_m,
            "within_tolerance": metrics.within_tolerance,
            "latency_seconds": latency,
            "display_map": displayed_map,
        }
    target = read_multichannel_audio(record.clean_audio_path, audio_cfg.sample_rate)
    output["audio_metrics"] = {
        "mixture_si_sdr_db": _sisdr(target, mixture),
        "enhanced_si_sdr_db": _sisdr(target, enhanced),
        "wave_inference_seconds": wave_seconds,
        "mixture_srp_seconds": mixture_srp_seconds,
        "enhanced_srp_seconds": enhanced_srp_seconds,
        "cnn_mixture_seconds": cnn_mix_seconds,
        "cnn_wave_seconds": cnn_wave_seconds,
    }
    return output, (xs, ys)


st.title("Localização de múltiplas fontes de voz")
st.caption("Demonstração **simulada** com fala do LibriSpeech e sala virtual 8 × 6 × 3 m. Resultados não representam gravações reais.")
st.warning(
    "Na validação congelada, a Wave-U-Net não passou o critério de melhora da localização. "
    "Os dois fluxos Wave aparecem abaixo para comparação experimental, não como métodos recomendados."
)

with st.sidebar:
    st.header("Condições simuladas")
    selected_seed = st.selectbox("Seed de modelos congelados", [42, 43, 44], index=0)
    n_sources = st.selectbox("Fontes (N)", [1, 2, 3], index=1)
    n_microphones = st.selectbox("Microfones (M)", [4, 6, 8], index=2)
    geometry_options = ["circle_r6", "perimeter", "circle"]
    geometry_labels = {
        "circle_r6": "Círculo da Rodada 6 (orientação validada)",
        "perimeter": "Perímetro (baseline)",
        "circle": "Círculo com orientação aleatória",
    }
    geometry = st.selectbox(
        "Geometria",
        geometry_options,
        format_func=lambda value: geometry_labels[value],
    )
    if geometry == "circle_r6":
        st.caption(
            "Arranjo circular de raio 2 m reproduzido da Rodada 6. "
            "A redução de P90 foi confirmada para M=8; em M=4 e M=6, "
            "a evidência confirmatória foi de média, não de P90. "
            "A confirmação estatística foi para SRP-PHAT; CNN e Wave-U-Net "
            "não foram confirmadas separadamente nesta geometria."
        )
    snr = st.slider("SNR solicitado (dB)", 10.0, 30.0, 20.0, 1.0)
    rt60 = st.slider("RT60 solicitado (s)", 0.2, 0.8, 0.5, 0.05)
    generate = st.button("Gerar cena e comparar", type="primary", use_container_width=True)

if generate:
    try:
        config_path = str(ROOT / "configs" / "full.yaml")
        config, device, models, normalizers, wave_variant, wave_hash = _load_models(selected_seed, config_path)
        if "demo_counter" not in st.session_state:
            st.session_state.demo_counter = 0
        record, demo_config = _make_scene(
            n_sources, n_microphones, geometry, snr, rt60, st.session_state.demo_counter
        )
        st.session_state.demo_counter += 1
        results, (xs, ys) = _run_comparison(record, demo_config, device, models, normalizers, wave_variant)
        st.success(
            f"Cena {record.scene_id} gerada. SNR realizado: {record.snr_realized_db:.1f} dB; "
            f"RT60 medido: {record.rt60_realized if record.rt60_realized is not None else float('nan'):.2f} s."
        )
        st.caption(f"Checkpoint Wave-U-Net: SHA-256 {wave_hash}. Arquivo de fala selecionado entre as origens de treino; nenhum áudio real foi gravado nesta demonstração.")

        columns = st.columns(2)
        methods = (
            ("SRP-PHAT", "SRP-PHAT"),
            ("CNN sobre mistura", "CNN sobre mistura"),
            ("Wave-U-Net + SRP-PHAT", "Wave-U-Net + SRP-PHAT (experimental)"),
            ("Wave-U-Net + CNN", "Wave-U-Net + CNN (experimental)"),
            (f"CNN contexto amplo (seed {models['round2_seed']})", f"CNN contexto amplo (rodada 2, seed {models['round2_seed']})") if "cnn_round2" in models else None,
            (f"SRP + CNN 25% (seed {models['round2_seed']})", f"SRP + CNN 25% (rodada 2, seed {models['round2_seed']})") if "cnn_round2" in models else None,
        )
        methods = tuple(item for item in methods if item is not None)
        plot_images = _render_scene_plots(record, results, xs, ys, methods)
        for index, (result_key, label) in enumerate(methods):
            item = results[result_key]
            with columns[index % 2]:
                st.subheader(label)
                st.image(plot_images[result_key], width="stretch")
                st.metric("Erro médio por fonte", f"{item['mean_error_m']:.3f} m")
                st.write(
                    f"Mediana: {item['median_error_m']:.3f} m · P90: {item['p90_error_m']:.3f} m · "
                    f"Dentro de 0,50 m: {item['within_tolerance']:.0%} · "
                    f"Latência do fluxo: {item['latency_seconds'] * 1000:.1f} ms"
                )
        audio_metrics = results["audio_metrics"]
        st.subheader("Diagnóstico de enhancement")
        st.write(
            f"SI-SDR da mistura: {audio_metrics['mixture_si_sdr_db']:.2f} dB · "
            f"SI-SDR após Wave-U-Net: {audio_metrics['enhanced_si_sdr_db']:.2f} dB · "
            f"Inferência Wave-U-Net: {audio_metrics['wave_inference_seconds'] * 1000:.1f} ms · "
            f"SRP mistura: {audio_metrics['mixture_srp_seconds'] * 1000:.1f} ms · "
            f"SRP aprimorado: {audio_metrics['enhanced_srp_seconds'] * 1000:.1f} ms"
        )
        with st.expander("Configuração completa da cena"):
            st.json({
                "scene_id": record.scene_id,
                "seed_dados": record.seed,
                "n_sources": record.n_sources,
                "n_microphones": record.n_microphones,
                "geometry": record.geometry,
                "snr_requested_db": record.snr_requested_db,
                "snr_realized_db": record.snr_realized_db,
                "rt60_requested_s": record.rt60_requested,
                "rt60_realized_s": record.rt60_realized,
                "source_positions_xy_m": record.source_positions_xy,
                "microphone_positions_xy_m": record.microphone_positions_xy,
                "wave_checkpoint_sha256": wave_hash,
                "wave_map_variant": wave_variant,
            })
    except Exception as exc:
        st.error(str(exc))
        st.info("A demonstração depende da base completa, dos normalizadores de treino e dos checkpoints finais congelados.")
else:
    st.info("Selecione condições dentro do domínio estudado e gere uma cena para comparar as quatro linhas.")
