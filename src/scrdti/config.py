from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Paper-facing SCR-DTI model configuration.

    The default dimensions reproduce the historical paper-checkpoint representation path.
    Smaller values can be used for CPU smoke tests or method development.
    """

    smiles_input_dim: int = 768
    text_input_dim: int = 768
    protein_input_dim: int = 1280
    projection_dim: int = 512
    projector_hidden_dim: int = 768
    projector_mid_dim: int = 512
    projector_dropout: float = 0.1

    pair_hidden_dim: int = 512
    pair_mid_dim: int = 256
    pair_dropout: float = 0.3

    use_hca: bool = True
    hca_num_heads: int = 4
    hca_hidden_dim: int = 512
    hca_dropout: float = 0.1

    use_graph_context: bool = True
    graph_context_dim: int = 2

    use_kge_refinement: bool = True
    kge_hidden_dim: int = 256
    kge_dropout: float = 0.2
    kge_initial_scale: float = 0.0
    kge_adapter_gate_bias: float = 4.0
    kge_norm_mean: float = 0.0
    kge_norm_std: float = 1.0
    kge_norm_min_std: float = 1e-6
    kge_residual_shrink: float = 0.2
    kge_residual_bound: float = 0.75
    confidence_margin_bias: float = 1.0
    confidence_margin_temperature: float = 1.0

    @classmethod
    def historical_checkpoint(cls) -> "ModelConfig":
        """Return the checkpoint-era architecture used by legacy mapping."""

        return cls()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    pos_weight: float | None = None
    auto_pos_weight: bool = True
    max_pos_weight: float = 12.0
    selection_metric: str = "auprc"
    patience: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReleaseConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: str | None = None
    regime: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "benchmark": self.benchmark,
            "regime": self.regime,
            "notes": self.notes,
        }


def _filter_dataclass_payload(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return payload


def load_config(path: str | Path) -> ReleaseConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    model_payload = payload.get("model", {}) or {}
    train_payload = payload.get("train", {}) or {}
    if not isinstance(model_payload, dict) or not isinstance(train_payload, dict):
        raise ValueError("'model' and 'train' must be mappings")
    model = ModelConfig(**_filter_dataclass_payload(ModelConfig, model_payload))
    train = TrainConfig(**_filter_dataclass_payload(TrainConfig, train_payload))
    known_top = {"model", "train", "benchmark", "regime", "notes"}
    unknown_top = sorted(set(payload) - known_top)
    if unknown_top:
        raise ValueError(f"Unknown top-level configuration fields: {unknown_top}")
    return ReleaseConfig(
        model=model,
        train=train,
        benchmark=payload.get("benchmark"),
        regime=payload.get("regime"),
        notes=payload.get("notes"),
    )

