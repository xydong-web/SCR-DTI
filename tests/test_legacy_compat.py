from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scrdti import ModelConfig, SCRDTI
from scrdti.compat import load_legacy_checkpoint


def _fake_legacy_from_public(model: SCRDTI) -> dict[str, object]:
    reverse_prefixes = (
        ("smiles_projector.", "semantic_backbone.reference.smiles_projector."),
        ("text_projector.", "semantic_backbone.reference.text_projector."),
        ("protein_projector.", "semantic_backbone.reference.protein_projector."),
        ("pair_head.net.", "binary_shared_paper_head.net."),
        ("hca.", "paper_binary_hierarchical_cross_attention."),
        ("hca_logit_head.", "paper_binary_hierarchical_logit_head."),
        ("graph_context.logit_head.", "paper_binary_graph_logit_head."),
        ("kge_refinement.adapter.", "paper_binary_kge_adapter."),
        ("kge_refinement.calibration.", "paper_binary_kge_calibration."),
    )
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "kge_refinement.norm_mean":
            state["paper_binary_kge_norm_mean"] = value.clone()
            continue
        if key == "kge_refinement.norm_std":
            state["paper_binary_kge_norm_std"] = value.clone()
            continue
        for public_prefix, legacy_prefix in reverse_prefixes:
            if key.startswith(public_prefix):
                state[legacy_prefix + key[len(public_prefix) :]] = value.clone()
                break
        else:  # pragma: no cover - protects against accidental public-key drift
            raise AssertionError(f"unmapped public key in fake legacy test: {key}")
    state["condition_encoder.task_embedding.weight"] = torch.randn(3, 512)
    state["paper_binary_interaction_trunk.0.weight"] = torch.randn(512, 512)
    return {"model": state}


def test_legacy_mapper_populates_clean_public_path() -> None:
    torch.manual_seed(13)
    source = SCRDTI(ModelConfig.historical_checkpoint())
    payload = _fake_legacy_from_public(source)
    target = SCRDTI(ModelConfig.historical_checkpoint())
    report = load_legacy_checkpoint(target, payload)
    assert report.core_complete
    assert report.ignored_keys == 2
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key]), key


@pytest.mark.skipif(
    not os.environ.get("SCRDTI_LEGACY_CHECKPOINT"),
    reason="set SCRDTI_LEGACY_CHECKPOINT to validate a real historical checkpoint",
)
def test_real_historical_checkpoint_maps_completely() -> None:
    checkpoint = Path(os.environ["SCRDTI_LEGACY_CHECKPOINT"])
    model = SCRDTI(ModelConfig.historical_checkpoint())
    report = load_legacy_checkpoint(model, checkpoint)
    assert report.core_complete
    assert report.source_keys == 206
    assert report.mapped_keys == len(model.state_dict())

