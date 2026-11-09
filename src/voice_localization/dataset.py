"""Dataset PyTorch para mapas pré-calculados."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import SceneRecord, read_jsonl


def load_map_sample(record: SceneRecord) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not record.map_path:
        raise ValueError(f"cena sem map_path: {record.scene_id}")
    with np.load(record.map_path) as data:
        srp_map = np.asarray(data["srp_map"], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)
        xs = np.asarray(data["xs"], dtype=np.float32)
        ys = np.asarray(data["ys"], dtype=np.float32)
        positions = np.asarray(data["source_positions_xy"], dtype=np.float32)
        microphone_positions = np.asarray(data["microphone_positions_xy"], dtype=np.float32)
        n_sources = int(data["n_sources"])
    n_channel = np.full_like(srp_map, n_sources / 3.0, dtype=np.float32)
    features = np.stack((srp_map, n_channel), axis=0)
    return features, target[None, ...], n_sources, positions, xs, ys, microphone_positions


def make_torch_dataset(manifest_path: str | Path):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch é necessário para SceneMapDataset") from exc

    records = list(read_jsonl(manifest_path, SceneRecord))

    class SceneMapDataset(Dataset):
        def __init__(self) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            features, target, n_sources, positions, xs, ys, microphone_positions = load_map_sample(self.records[index])
            return {
                "input": torch.from_numpy(features),
                "target": torch.from_numpy(target),
                "n_sources": torch.tensor(n_sources, dtype=torch.long),
                "positions": torch.from_numpy(positions),
                "xs": torch.from_numpy(xs),
                "ys": torch.from_numpy(ys),
                "microphone_positions_xy": torch.from_numpy(microphone_positions),
                "scene_id": self.records[index].scene_id,
            }

    return SceneMapDataset()
