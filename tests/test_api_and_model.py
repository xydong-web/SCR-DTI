from __future__ import annotations

import torch

from scrdti import ModelConfig, SCRDTI


def tiny_config() -> ModelConfig:
    return ModelConfig(
        smiles_input_dim=12,
        text_input_dim=12,
        protein_input_dim=20,
        projection_dim=16,
        projector_hidden_dim=24,
        projector_mid_dim=16,
        projector_dropout=0.0,
        pair_hidden_dim=24,
        pair_mid_dim=12,
        pair_dropout=0.0,
        use_hca=True,
        hca_num_heads=4,
        hca_hidden_dim=24,
        hca_dropout=0.0,
        use_graph_context=True,
        graph_context_dim=2,
        use_kge_refinement=True,
        kge_hidden_dim=16,
        kge_dropout=0.0,
        kge_initial_scale=0.5,
        kge_residual_shrink=0.2,
        kge_residual_bound=0.75,
        confidence_margin_bias=1.0,
        confidence_margin_temperature=1.0,
    )


def test_import_and_synthetic_forward() -> None:
    torch.manual_seed(3)
    model = SCRDTI(tiny_config()).eval()
    batch = 5
    output = model(
        torch.randn(batch, 12),
        torch.randn(batch, 12),
        torch.randn(batch, 20),
        has_text=torch.ones(batch),
        graph_context=torch.randn(batch, 2),
        graph_available=torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32),
        kge_score=torch.randn(batch),
        kge_available=torch.tensor([1, 0, 1, 1, 0], dtype=torch.float32),
    )
    assert output.logits.shape == (batch,)
    assert output.probabilities.shape == (batch,)
    assert torch.all((output.probabilities >= 0) & (output.probabilities <= 1))
    assert output.pair_repr.shape == (batch, 16)


def test_graph_availability_mask_is_fail_closed() -> None:
    torch.manual_seed(5)
    config = tiny_config()
    config.use_kge_refinement = False
    model = SCRDTI(config).eval()
    with torch.no_grad():
        assert model.graph_context is not None
        model.graph_context.logit_head.weight.fill_(2.0)
    smiles = torch.randn(2, 12)
    text = torch.randn(2, 12)
    protein = torch.randn(2, 20)
    graph = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    masked = model(
        smiles,
        text,
        protein,
        graph_context=graph,
        graph_available=torch.tensor([1.0, 0.0]),
    )
    baseline = model(smiles, text, protein, graph_context=None)
    assert torch.allclose(masked.graph_logit[1], torch.zeros(()))
    assert torch.allclose(masked.logits[1], baseline.logits[1], atol=1e-7)
    assert not torch.allclose(masked.logits[0], baseline.logits[0])


def test_confidence_margin_gate_decreases_with_base_margin() -> None:
    model = SCRDTI(tiny_config()).eval()
    assert model.kge_refinement is not None
    gate = model.kge_refinement.confidence_gate(
        torch.tensor([0.0, 1.0, 4.0]), torch.ones(3)
    )
    assert gate[0] > gate[1] > gate[2]


def test_kge_residual_is_bounded_and_masked() -> None:
    torch.manual_seed(11)
    config = tiny_config()
    model = SCRDTI(config).eval()
    assert model.kge_refinement is not None
    with torch.no_grad():
        model.kge_refinement.adapter.base_scale.fill_(100.0)
    output = model(
        torch.randn(3, 12),
        torch.randn(3, 12),
        torch.randn(3, 20),
        kge_score=torch.tensor([100.0, -100.0, 50.0]),
        kge_available=torch.tensor([1.0, 1.0, 0.0]),
    )
    assert bool((output.kge_residual.abs() <= config.kge_residual_bound + 1e-6).all())
    assert torch.allclose(output.kge_residual[2], torch.zeros(()), atol=1e-7)

