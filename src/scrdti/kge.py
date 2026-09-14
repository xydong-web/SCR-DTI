from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class KGEIndex:
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]


class TorusE(nn.Module):
    """Compact TorusE scorer used by the strict fold-specific KGE workflow."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 256):
        super().__init__()
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.sigmoid(self.entity(heads))
        r = torch.sigmoid(self.relation(relations))
        t = torch.sigmoid(self.entity(tails))
        delta = torch.remainder(h + r - t, 1.0)
        distance = torch.minimum(delta.abs(), 1.0 - delta.abs()).sum(dim=-1)
        return -distance


def build_kge_index(triples: pd.DataFrame) -> KGEIndex:
    required = {"head", "relation", "tail"}
    missing = required.difference(triples.columns)
    if missing:
        raise ValueError(f"triples are missing columns: {sorted(missing)}")
    entities = sorted(set(triples["head"].astype(str)) | set(triples["tail"].astype(str)))
    relations = sorted(set(triples["relation"].astype(str)))
    return KGEIndex(
        entity_to_id={value: index for index, value in enumerate(entities)},
        relation_to_id={value: index for index, value in enumerate(relations)},
    )


def encode_triples(triples: pd.DataFrame, index: KGEIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heads = triples["head"].astype(str).map(index.entity_to_id).to_numpy(dtype=np.int64)
    relations = triples["relation"].astype(str).map(index.relation_to_id).to_numpy(dtype=np.int64)
    tails = triples["tail"].astype(str).map(index.entity_to_id).to_numpy(dtype=np.int64)
    return heads, relations, tails


def _negative_examples(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    *,
    num_entities: int,
    ratio: int,
    seed: int,
    entity_names: list[str],
    forbidden_pairs: set[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ratio < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    positive = {(int(h), int(r), int(t)) for h, r, t in zip(heads, relations, tails)}
    neg_h: list[int] = []
    neg_r: list[int] = []
    neg_t: list[int] = []
    for h, r, t in zip(heads, relations, tails):
        for _ in range(ratio):
            accepted = False
            for _attempt in range(256):
                if bool(rng.integers(0, 2)):
                    candidate_h = int(rng.integers(0, num_entities))
                    candidate_t = int(t)
                else:
                    candidate_h = int(h)
                    candidate_t = int(rng.integers(0, num_entities))
                candidate = (candidate_h, int(r), candidate_t)
                pair = (entity_names[candidate_h], entity_names[candidate_t])
                if candidate in positive or pair in forbidden_pairs:
                    continue
                neg_h.append(candidate_h)
                neg_r.append(int(r))
                neg_t.append(candidate_t)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("failed to sample a legal KGE negative after 256 attempts")
    return (
        np.asarray(neg_h, dtype=np.int64),
        np.asarray(neg_r, dtype=np.int64),
        np.asarray(neg_t, dtype=np.int64),
    )


def train_toruse(
    triples: pd.DataFrame,
    *,
    embedding_dim: int = 256,
    epochs: int = 5,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    negative_ratio: int = 1,
    seed: int = 42,
    forbidden_pairs: Iterable[tuple[str, str]] = (),
    device: str | torch.device = "cpu",
) -> tuple[TorusE, KGEIndex]:
    """Train a fold-specific TorusE model with head/tail corruption negatives."""

    torch.manual_seed(seed)
    index = build_kge_index(triples)
    heads, relations, tails = encode_triples(triples, index)
    entity_names = [""] * len(index.entity_to_id)
    for name, entity_id in index.entity_to_id.items():
        entity_names[entity_id] = name
    neg_h, neg_r, neg_t = _negative_examples(
        heads,
        relations,
        tails,
        num_entities=len(index.entity_to_id),
        ratio=negative_ratio,
        seed=seed,
        entity_names=entity_names,
        forbidden_pairs={(str(a), str(b)) for a, b in forbidden_pairs},
    )
    all_h = np.concatenate([heads, neg_h])
    all_r = np.concatenate([relations, neg_r])
    all_t = np.concatenate([tails, neg_t])
    labels = np.concatenate(
        [np.ones(len(heads), dtype=np.float32), np.zeros(len(neg_h), dtype=np.float32)]
    )
    dataset = TensorDataset(
        torch.from_numpy(all_h),
        torch.from_numpy(all_r),
        torch.from_numpy(all_t),
        torch.from_numpy(labels),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    device_obj = torch.device(device)
    model = TorusE(
        len(index.entity_to_id), len(index.relation_to_id), embedding_dim=embedding_dim
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _epoch in range(int(epochs)):
        model.train()
        for h, r, t, y in loader:
            h, r, t, y = h.to(device_obj), r.to(device_obj), t.to(device_obj), y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(h, r, t), y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model, index


@torch.no_grad()
def score_pairs(
    model: TorusE,
    index: KGEIndex,
    pairs: Iterable[tuple[str, str]],
    *,
    relation: str,
    device: str | torch.device = "cpu",
) -> list[float]:
    if relation not in index.relation_to_id:
        raise KeyError(f"unknown KGE relation: {relation}")
    device_obj = torch.device(device)
    model = model.to(device_obj).eval()
    relation_id = index.relation_to_id[relation]
    scores: list[float] = []
    for head_name, tail_name in pairs:
        if head_name not in index.entity_to_id or tail_name not in index.entity_to_id:
            scores.append(float("nan"))
            continue
        h = torch.tensor([index.entity_to_id[head_name]], device=device_obj)
        r = torch.tensor([relation_id], device=device_obj)
        t = torch.tensor([index.entity_to_id[tail_name]], device=device_obj)
        scores.append(float(model(h, r, t).item()))
    return scores


def save_kge_checkpoint(path: str | Path, model: TorusE, index: KGEIndex) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "entity_to_id": index.entity_to_id,
            "relation_to_id": index.relation_to_id,
            "embedding_dim": int(model.entity.embedding_dim),
        },
        path,
    )
    return path

