from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scrdti.config import ReleaseConfig
from scrdti.data import FeatureArrays, FeatureDataset, load_feature_npz
from scrdti.evaluate import binary_metrics
from scrdti.model import SCRDTI


@dataclass
class TrainResult:
    best_epoch: int
    best_validation_auprc: float
    checkpoint_path: str
    history: list[dict[str, float]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_forward(model: SCRDTI, batch: dict[str, torch.Tensor]):
    return model(
        batch["smiles"],
        batch["text"],
        batch["protein"],
        has_text=batch.get("has_text"),
        graph_context=batch.get("graph_context"),
        graph_available=batch.get("graph_available"),
        kge_score=batch.get("kge_score"),
        kge_available=batch.get("kge_available"),
    )


def _evaluate_loader(
    model: SCRDTI,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            output = _model_forward(model, batch)
            labels.append(batch["label"].detach().cpu().numpy())
            probabilities.append(output.probabilities.detach().cpu().numpy())
            logits.append(output.logits.detach().cpu().numpy())
    y = np.concatenate(labels) if labels else np.empty(0)
    p = np.concatenate(probabilities) if probabilities else np.empty(0)
    z = np.concatenate(logits) if logits else np.empty(0)
    return binary_metrics(y, p), y, p, z


def train_model(
    config: ReleaseConfig,
    arrays: FeatureArrays,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> TrainResult:
    _seed_everything(config.train.seed)
    device_obj = torch.device(device)
    train_arrays = arrays.subset(0)
    valid_arrays = arrays.subset(1)
    if len(train_arrays) == 0 or len(valid_arrays) == 0:
        raise ValueError("training requires non-empty train (0) and validation (1) splits")
    train_loader = DataLoader(
        FeatureDataset(train_arrays),
        batch_size=config.train.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        FeatureDataset(valid_arrays),
        batch_size=config.train.batch_size,
        shuffle=False,
    )
    model = SCRDTI(config.model).to(device_obj)
    if model.kge_refinement is not None and train_arrays.kge_score is not None:
        if train_arrays.kge_available is None:
            active_kge = np.ones(len(train_arrays), dtype=bool)
        else:
            active_kge = train_arrays.kge_available > 0
        if np.any(active_kge):
            training_scores = train_arrays.kge_score[active_kge].astype(np.float64)
            mean = float(training_scores.mean())
            std = float(training_scores.std(ddof=0))
            std = max(std, model.kge_refinement.norm_min_std)
            with torch.no_grad():
                model.kge_refinement.norm_mean.fill_(mean)
                model.kge_refinement.norm_std.fill_(std)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    pos_weight = None
    if config.train.pos_weight is not None:
        pos_weight = torch.tensor(float(config.train.pos_weight), device=device_obj)
    elif config.train.auto_pos_weight:
        positives = float((train_arrays.labels > 0.5).sum())
        negatives = float((train_arrays.labels <= 0.5).sum())
        if positives > 0 and negatives > 0:
            resolved = min(negatives / positives, float(config.train.max_pos_weight))
            pos_weight = torch.tensor(resolved, device=device_obj)
    best_score = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _batch_to_device(raw_batch, device_obj)
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                output.logits,
                batch["label"].float(),
                pos_weight=pos_weight,
            )
            loss.backward()
            if config.train.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        valid_metrics, _, _, _ = _evaluate_loader(model, valid_loader, device_obj)
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "valid_auprc": valid_metrics["auprc"],
            "valid_auroc": valid_metrics["auroc"],
        }
        history.append(record)
        score = valid_metrics["auprc"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if config.train.patience is not None and stale >= config.train.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_config": config.model.to_dict(),
            "train_config": config.train.to_dict(),
            "best_epoch": best_epoch,
            "best_validation_auprc": best_score,
            "release": "SCR-DTI 0.1.0",
        },
        output_path,
    )
    return TrainResult(
        best_epoch=best_epoch,
        best_validation_auprc=best_score,
        checkpoint_path=str(output_path),
        history=history,
    )


def train_from_npz(
    config: ReleaseConfig,
    data_path: str | Path,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> TrainResult:
    return train_model(config, load_feature_npz(data_path), output_path, device=device)


def load_release_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SCRDTI, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
        raise ValueError("Not an SCR-DTI release checkpoint")
    from scrdti.config import ModelConfig

    model = SCRDTI(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(torch.device(map_location))
    return model, payload


def predict_arrays(
    model: SCRDTI,
    arrays: FeatureArrays,
    *,
    split_id: int = 2,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    subset = arrays.subset(split_id)
    if len(subset) == 0:
        raise ValueError(f"split {split_id} is empty")
    device_obj = torch.device(device)
    model = model.to(device_obj)
    loader = DataLoader(FeatureDataset(subset), batch_size=batch_size, shuffle=False)
    _, labels, probabilities, logits = _evaluate_loader(model, loader, device_obj)
    return {"label": labels, "probability": probabilities, "logit": logits}


def result_to_json(result: TrainResult) -> str:
    return json.dumps(asdict(result), indent=2)

