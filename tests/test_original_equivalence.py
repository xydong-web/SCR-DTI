from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from scrdti import ModelConfig, SCRDTI
from scrdti.compat import load_legacy_checkpoint


pytestmark = pytest.mark.skipif(
    not (os.environ.get("SCRDTI_ORIGINAL_ROOT") and os.environ.get("SCRDTI_LEGACY_CHECKPOINT")),
    reason="requires original research tree plus historical paper checkpoint",
)


def test_clean_public_model_matches_original_drugbank_binary_logits() -> None:
    original_root = Path(os.environ["SCRDTI_ORIGINAL_ROOT"]).resolve()
    checkpoint = Path(os.environ["SCRDTI_LEGACY_CHECKPOINT"]).resolve()
    sys.path.insert(0, str(original_root / "src"))
    try:
        from smhdti.cli.model_kwargs import build_task_conditioned_moe_kwargs
        from smhdti.data.task_schema import encode_dataset_name
        from smhdti.models.task_conditioned_moe import TaskConditionedMoEPairModel
        from smhdti.utils import load_yaml

        config_candidates = sorted(
            (original_root / "configs/train/evidti_topk_seed").glob(
                "*drugbank*hier_crossattn*trainseed1.yaml"
            )
        )
        if len(config_candidates) != 1:
            raise RuntimeError(
                "Expected exactly one historical DrugBank hierarchical-attention config, "
                f"found {len(config_candidates)}"
            )
        config_path = config_candidates[0]
        payload = load_yaml(config_path)
        kwargs = build_task_conditioned_moe_kwargs(payload)
        kwargs["paper_binary_use_interaction_trunk"] = False
        kwargs["paper_binary_use_shared_private_pair_repr"] = False
        original = TaskConditionedMoEPairModel(**kwargs).eval()
        source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        original.load_state_dict(source_payload["model"], strict=True)

        public = SCRDTI(ModelConfig.historical_checkpoint()).eval()
        report = load_legacy_checkpoint(public, source_payload)
        assert report.core_complete

        torch.manual_seed(2026)
        batch = 4
        smiles = torch.randn(batch, 768)
        text = torch.randn(batch, 768)
        hta = torch.zeros(batch, 768)
        protein = torch.randn(batch, 1280)
        ones = torch.ones(batch)
        zeros_long = torch.zeros(batch, dtype=torch.long)
        dataset_ids = torch.full(
            (batch,), encode_dataset_name("drugbank"), dtype=torch.long
        )
        with torch.no_grad():
            original_out = original(
                smiles,
                text,
                hta,
                protein,
                task_type_ids=zeros_long,
                measurement_type_ids=zeros_long,
                dataset_ids=dataset_ids,
                has_text=ones,
                has_real_text=ones,
                has_hta=torch.zeros(batch),
                has_real_hta=torch.zeros(batch),
                has_structure=torch.zeros(batch),
            )
            public_out = public(smiles, text, protein, has_text=ones)
        assert torch.allclose(original_out.binary_logits, public_out.logits, atol=1e-6, rtol=1e-6)
    finally:
        sys.path.remove(str(original_root / "src"))

