"""Leitura, normalização e janelas de áudio."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def read_audio(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Lê áudio mono em float32; usa SoundFile quando disponível e WAV PCM como fallback."""
    path = Path(path)
    try:
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=False)
        data = np.asarray(data, dtype=np.float32)
    except ImportError:
        data, rate = _read_wav_pcm(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if rate != sample_rate:
        data = resample_linear(data, rate, sample_rate)
    return ensure_finite(data.astype(np.float32))


def read_multichannel_audio(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Lê um WAV e retorna canais como ``(M, T)`` sem fazer downmix."""
    path = Path(path)
    try:
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        data = np.asarray(data, dtype=np.float32).T
    except ImportError:
        data, rate = _read_wav_pcm(path)
        data = np.asarray(data, dtype=np.float32).T
    if rate != sample_rate:
        data = np.stack([resample_linear(channel, rate, sample_rate) for channel in data])
    return ensure_finite(data.astype(np.float32))


def _read_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"PCM WAV com largura não suportada: {width} bytes")
    return data.reshape(-1, channels), rate


def resample_linear(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal
    if signal.size == 0:
        return signal.astype(np.float32)
    target_length = max(1, round(signal.size * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, signal.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def active_rms(signal: np.ndarray, floor: float = 1e-4) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if signal.size else 0.0
    return max(rms, floor)


def measured_rt60(rir: object, sample_rate: int) -> float | None:
    """Estima RT60 por integração de Schroeder de uma resposta impulsiva.

    ``rir`` pode ser o objeto aninhado produzido por ``room.rir`` ou um vetor
    1-D. Retorna ``None`` quando não há faixa dinâmica suficiente para uma
    regressão estável.
    """
    candidates: list[np.ndarray] = []

    def collect(value: object) -> None:
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                candidates.append(value.astype(np.float64))
            else:
                for row in value:
                    collect(row)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(rir)
    if not candidates:
        return None
    signal = max(candidates, key=lambda item: float(np.sum(item * item)))
    energy = signal * signal
    if not np.any(energy > 0):
        return None
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10.0 * np.log10(np.maximum(schroeder / np.max(schroeder), 1e-12))
    start_candidates = np.flatnonzero(schroeder_db <= -5.0)
    end_candidates = np.flatnonzero(schroeder_db <= -35.0)
    if len(start_candidates) == 0 or len(end_candidates) == 0:
        return None
    start = int(start_candidates[0])
    end = int(end_candidates[0])
    if end <= start:
        return None
    return float((end - start) / sample_rate * 2.0)


def normalize_active(signal: np.ndarray, target_rms: float = 0.1, floor: float = 1e-4) -> np.ndarray:
    return (signal / active_rms(signal, floor) * target_rms).astype(np.float32)


def fit_window(signal: np.ndarray, samples: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    if samples <= 0:
        raise ValueError("samples deve ser positivo")
    if signal.size < samples:
        output = np.zeros(samples, dtype=np.float32)
        output[: signal.size] = signal
        return output, 0
    rng = rng or np.random.default_rng()
    start = int(rng.integers(0, signal.size - samples + 1))
    return signal[start : start + samples].astype(np.float32), start


def mix_at_microphones(
    sources: list[np.ndarray],
    room: object,
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Simula com Pyroomacoustics e devolve ``(M, T)`` e SNR realizado."""
    try:
        import pyroomacoustics as pra
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pyroomacoustics é necessário para o backend acústico") from exc
    room_dim = list(room.dimensions)
    rt60 = float(getattr(room, "rt60", 0.2))
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    sim_room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=pra.Material(absorption),
        max_order=max_order,
        use_rand_ism=True,
    )
    for position, signal in zip(source_positions, sources):
        sim_room.add_source(position.tolist(), signal=signal)
    sim_room.add_microphone_array(microphone_positions.T)
    sim_room.simulate()
    clean = np.asarray(sim_room.mic_array.signals, dtype=np.float32)
    power = float(np.mean(clean**2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(max(noise_power, 1e-12)), clean.shape).astype(np.float32)
    mixed = clean + noise
    realized = 10.0 * np.log10(max(power, 1e-12) / max(float(np.mean(noise**2)), 1e-12))
    return mixed, float(realized)


def anechoic_mix(
    sources: list[np.ndarray],
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    snr_db: float,
    speed_of_sound: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Backend leve para smoke tests: atraso e atenuação de propagação direta."""
    mixed, _, realized = anechoic_mix_with_target(
        sources, source_positions, microphone_positions, sample_rate, snr_db, speed_of_sound, rng
    )
    return mixed, realized


def anechoic_mix_with_target(
    sources: list[np.ndarray],
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    snr_db: float,
    speed_of_sound: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retorna mistura ruidosa, alvo limpo multicanal e SNR realizado."""
    clean = anechoic_clean_mix(
        sources, source_positions, microphone_positions, sample_rate, speed_of_sound
    )
    mixed, realized = add_noise_at_snr(clean, snr_db, rng)
    return mixed, clean, realized


def anechoic_clean_mix(
    sources: list[np.ndarray],
    source_positions: np.ndarray,
    microphone_positions: np.ndarray,
    sample_rate: int,
    speed_of_sound: float,
) -> np.ndarray:
    """Retorna a soma multicanal anecoica, sem ruído, com atrasos de propagação."""
    samples = max(len(s) for s in sources)
    output = np.zeros((len(microphone_positions), samples + 4096), dtype=np.float32)
    for source, position in zip(sources, source_positions):
        for mic_index, mic in enumerate(microphone_positions):
            distance = float(np.linalg.norm(position - mic))
            delay = int(round(distance / speed_of_sound * sample_rate))
            attenuation = 1.0 / max(distance, 0.25)
            end = min(output.shape[1], delay + len(source))
            if end > delay:
                output[mic_index, delay:end] += source[: end - delay] * attenuation
    return output


def add_noise_at_snr(
    clean: np.ndarray, snr_db: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Mix white noise at the requested SNR and report it from saved-scale samples."""
    clean = np.asarray(clean, dtype=np.float32)
    clean64 = clean.astype(np.float64)
    clean_power = float(np.mean(clean64**2))
    noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(max(noise_power, 1e-12)), clean.shape).astype(np.float32)
    mixed = clean + noise
    effective_noise = mixed.astype(np.float64) - clean64
    realized = 10.0 * np.log10(
        max(clean_power, 1e-12) / max(float(np.mean(effective_noise**2)), 1e-12)
    )
    return mixed.astype(np.float32), float(realized)


def ensure_finite(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if not np.all(np.isfinite(array)):
        raise ValueError("áudio contém NaN ou infinito")
    return array


def synthetic_speech(seed: int, sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Sinal determinístico semelhante a fala para validar o pipeline sem corpus."""
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate * duration_seconds))
    t = np.arange(n, dtype=np.float32) / sample_rate
    envelope = np.maximum(0.0, np.sin(2 * np.pi * 2.2 * t)) ** 2
    frequencies = rng.choice([110, 140, 180, 220, 260], size=4, replace=False)
    signal = sum(np.sin(2 * np.pi * float(f) * t + rng.uniform(0, 2 * np.pi)) for f in frequencies)
    signal += 0.08 * rng.normal(size=n)
    return normalize_active((signal * envelope).astype(np.float32))
