from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from scrdti.model import SCRDTI


@dataclass(frozen=True)
class LegacyLoadReport:
    source_keys: int
    mapped_keys: int
    ignored_keys: int
    missing_public_keys: tuple[str, ...]
    unexpected_mapped_keys: tuple[str, ...]
    ignored_prefixes: tuple[str, ...]

    @property
    def core_complete(self) -> bool:
        return not self.missing_public_keys and not self.unexpected_mapped_keys


PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("semantic_backbone.reference.smiles_projector.", "smiles_projector."),
    ("semantic_backbone.reference.text_projector.", "text_projector."),
    ("semantic_backbone.reference.protein_projector.", "protein_projector."),
    ("binary_shared_paper_head.net.", "pair_head.net."),
    ("paper_binary_hierarchical_cross_attention.", "hca."),
    ("paper_binary_hierarchical_logit_head.", "hca_logit_head."),
    ("paper_binary_graph_logit_head.", "graph_context.logit_head."),
    ("paper_binary_kge_adapter.", "kge_refinement.adapter."),
    ("paper_binary_kge_calibration.", "kge_refinement.calibration."),
)

EXACT_MAP = {
    "paper_binary_kge_norm_mean": "kge_refinement.norm_mean",
    "paper_binary_kge_norm_std": "kge_refinement.norm_std",
}


def _state_dict_from_payload(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if payload and all(isinstance(key, str) for key in payload):
            tensor_values = [value for value in payload.values() if isinstance(value, torch.Tensor)]
            if tensor_values:
                return payload  # type: ignore[return-value]
    raise ValueError("Could not locate a model state dict in the legacy checkpoint")


def _map_key(key: str) -> str | None:
    if key in EXACT_MAP:
        return EXACT_MAP[key]
    for old_prefix, new_prefix in PREFIX_MAP:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return None


def _ignored_prefix(key: str) -> str:
    parts = key.split(".")
    if key.startswith("semantic_backbone.reference.ic50_classifier"):
        return "semantic_backbone.reference.ic50_classifier"
    return ".".join(parts[:2]) if len(parts) > 1 else parts[0]


def load_legacy_checkpoint(
    model: SCRDTI,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
    strict_core: bool = True,
) -> LegacyLoadReport:
    """Load the paper-relevant path from a historical paper checkpoint.

    Historical checkpoints include multitask/experimental scaffolding that is
    intentionally absent from the clean release. This function maps only weights
    that participate in the released SCR-DTI prediction path, validates shape and
    completeness, and reports ignored prefixes.
    """

    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    else:
        payload = checkpoint
    source = _state_dict_from_payload(payload)
    mapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    public_state = model.state_dict()
    for key, value in source.items():
        new_key = _map_key(str(key))
        if new_key is None:
            ignored.append(str(key))
            continue
        if new_key not in public_state:
            mapped[new_key] = value
            continue
        if tuple(public_state[new_key].shape) != tuple(value.shape):
            raise ValueError(
                f"Legacy tensor shape mismatch for {key!r} -> {new_key!r}: "
                f"checkpoint={tuple(value.shape)}, public={tuple(public_state[new_key].shape)}"
            )
        mapped[new_key] = value

    result = model.load_state_dict(mapped, strict=False)
    missing = tuple(sorted(result.missing_keys))
    unexpected = tuple(sorted(result.unexpected_keys))
    report = LegacyLoadReport(
        source_keys=len(source),
        mapped_keys=len(mapped),
        ignored_keys=len(ignored),
        missing_public_keys=missing,
        unexpected_mapped_keys=unexpected,
        ignored_prefixes=tuple(sorted({_ignored_prefix(key) for key in ignored})),
    )
    if strict_core and not report.core_complete:
        raise RuntimeError(
            "Legacy checkpoint did not fully populate the clean SCR-DTI prediction path: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return report

