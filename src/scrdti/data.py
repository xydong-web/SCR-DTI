from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("smiles", "text", "protein", "labels", "split")


@dataclass
class FeatureArrays:
    smiles: np.ndarray
    text: np.ndarray
    protein: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    has_text: np.ndarray
    graph_context: np.ndarray | None = None
    graph_available: np.ndarray | None = None
    kge_score: np.ndarray | None = None
    kge_available: np.ndarray | None = None

    def subset(self, split_id: int) -> "FeatureArrays":
        mask = self.split.astype(np.int64) == int(split_id)
        return FeatureArrays(
            smiles=self.smiles[mask],
            text=self.text[mask],
            protein=self.protein[mask],
            labels=self.labels[mask],
            split=self.split[mask],
            has_text=self.has_text[mask],
            graph_context=None if self.graph_context is None else self.graph_context[mask],
            graph_available=(
                None if self.graph_available is None else self.graph_available[mask]
            ),
            kge_score=None if self.kge_score is None else self.kge_score[mask],
            kge_available=None if self.kge_available is None else self.kge_available[mask],
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def load_feature_npz(path: str | Path) -> FeatureArrays:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"Missing required NPZ fields {missing} in {path}")
        n = int(payload["labels"].shape[0])
        arrays = FeatureArrays(
            smiles=np.asarray(payload["smiles"], dtype=np.float32),
            text=np.asarray(payload["text"], dtype=np.float32),
            protein=np.asarray(payload["protein"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=np.float32).reshape(-1),
            split=np.asarray(payload["split"], dtype=np.int64).reshape(-1),
            has_text=(
                np.asarray(payload["has_text"], dtype=np.float32).reshape(-1)
                if "has_text" in payload
                else np.ones(n, dtype=np.float32)
            ),
            graph_context=(
                np.asarray(payload["graph_context"], dtype=np.float32)
                if "graph_context" in payload
                else None
            ),
            graph_available=(
                np.asarray(payload["graph_available"], dtype=np.float32).reshape(-1)
                if "graph_available" in payload
                else None
            ),
            kge_score=(
                np.asarray(payload["kge_score"], dtype=np.float32).reshape(-1)
                if "kge_score" in payload
                else None
            ),
            kge_available=(
                np.asarray(payload["kge_available"], dtype=np.float32).reshape(-1)
                if "kge_available" in payload
                else None
            ),
        )
    _validate_arrays(arrays, path)
    return arrays


def _validate_arrays(arrays: FeatureArrays, path: Path | None = None) -> None:
    n = len(arrays)
    label = str(path) if path is not None else "feature arrays"
    for name in ("smiles", "text", "protein"):
        value = getattr(arrays, name)
        if value.ndim != 2 or value.shape[0] != n:
            raise ValueError(f"{label}: {name} must be [N,D], got {value.shape}")
    for name in ("split", "has_text"):
        value = getattr(arrays, name)
        if value.shape != (n,):
            raise ValueError(f"{label}: {name} must be [N], got {value.shape}")
    if arrays.graph_context is not None and arrays.graph_context.shape[0] != n:
        raise ValueError(f"{label}: graph_context row count mismatch")
    for name in ("graph_available", "kge_score", "kge_available"):
        value = getattr(arrays, name)
        if value is not None and value.shape != (n,):
            raise ValueError(f"{label}: {name} must be [N], got {value.shape}")
    if not set(np.unique(arrays.split)).issubset({0, 1, 2}):
        raise ValueError(f"{label}: split values must be 0(train), 1(valid), or 2(test)")


class FeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: FeatureArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "smiles": torch.from_numpy(self.arrays.smiles[index]),
            "text": torch.from_numpy(self.arrays.text[index]),
            "protein": torch.from_numpy(self.arrays.protein[index]),
            "label": torch.tensor(self.arrays.labels[index], dtype=torch.float32),
            "has_text": torch.tensor(self.arrays.has_text[index], dtype=torch.float32),
        }
        if self.arrays.graph_context is not None:
            item["graph_context"] = torch.from_numpy(self.arrays.graph_context[index])
        if self.arrays.graph_available is not None:
            item["graph_available"] = torch.tensor(
                self.arrays.graph_available[index], dtype=torch.float32
            )
        if self.arrays.kge_score is not None:
            item["kge_score"] = torch.tensor(self.arrays.kge_score[index], dtype=torch.float32)
        if self.arrays.kge_available is not None:
            item["kge_available"] = torch.tensor(
                self.arrays.kge_available[index], dtype=torch.float32
            )
        return item


def make_synthetic_npz(
    output: str | Path,
    *,
    n: int = 96,
    seed: int = 7,
    smiles_dim: int = 12,
    text_dim: int = 12,
    protein_dim: int = 20,
    graph_dim: int = 2,
) -> Path:
    """Create a deterministic tiny dataset used by the release smoke tests."""

    if n < 24:
        raise ValueError("n must be at least 24 to create train/valid/test splits")
    rng = np.random.default_rng(seed)
    smiles = rng.normal(size=(n, smiles_dim)).astype(np.float32)
    text = (0.65 * smiles + 0.35 * rng.normal(size=(n, text_dim))).astype(np.float32)
    protein = rng.normal(size=(n, protein_dim)).astype(np.float32)
    graph = rng.normal(size=(n, graph_dim)).astype(np.float32)
    kge = rng.normal(size=n).astype(np.float32)
    interaction = (
        smiles[:, : min(smiles_dim, protein_dim)]
        * protein[:, : min(smiles_dim, protein_dim)]
    ).sum(axis=1)
    signal = interaction + 0.35 * graph[:, 0] + 0.20 * kge
    labels = (signal > np.median(signal)).astype(np.float32)
    order = rng.permutation(n)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    split = np.full(n, 2, dtype=np.int64)
    split[order[:n_train]] = 0
    split[order[n_train : n_train + n_valid]] = 1
    graph_available = (rng.random(n) > 0.15).astype(np.float32)
    kge_available = (rng.random(n) > 0.20).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        smiles=smiles,
        text=text,
        protein=protein,
        labels=labels,
        split=split,
        has_text=np.ones(n, dtype=np.float32),
        graph_context=graph,
        graph_available=graph_available,
        kge_score=kge,
        kge_available=kge_available,
    )
    return output

