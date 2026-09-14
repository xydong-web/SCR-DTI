from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from scrdti.config import ModelConfig


def _projection_mlp(
    input_dim: int,
    projection_dim: int,
    hidden_dim: int,
    mid_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Projector topology used by the paper-facing SCR-DTI representation path."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, mid_dim),
        nn.GELU(),
        nn.LayerNorm(mid_dim),
        nn.Linear(mid_dim, projection_dim),
    )


def build_pair_features(
    molecule_features: torch.Tensor,
    protein_features: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            molecule_features,
            protein_features,
            torch.abs(molecule_features - protein_features),
            molecule_features * protein_features,
        ],
        dim=-1,
    )


class HierarchicalCrossAttention(nn.Module):
    """Hierarchical pooled-token molecular-protein interaction module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        heads = max(int(num_heads), 1)
        self.smiles_to_protein = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.text_to_protein = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.fusion_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.type_embeddings = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.norm_smiles = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_protein = nn.LayerNorm(dim)
        self.norm_pair = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def _attend(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        out, _ = attn(query, key_value, key_value, need_weights=False)
        fused = norm(query + out)
        return norm(fused + self.ff(fused))

    def forward(
        self,
        smiles_features: torch.Tensor,
        text_features: torch.Tensor,
        protein_features: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        smiles_token = smiles_features.unsqueeze(1) + self.type_embeddings[0].view(1, 1, -1)
        text_token = text_features.unsqueeze(1) + self.type_embeddings[1].view(1, 1, -1)
        protein_token = protein_features.unsqueeze(1) + self.type_embeddings[2].view(1, 1, -1)
        updated_smiles = self._attend(
            smiles_token, protein_token, self.smiles_to_protein, self.norm_smiles
        )
        updated_text = self._attend(
            text_token, protein_token, self.text_to_protein, self.norm_text
        )
        if text_mask is not None:
            updated_text = updated_text * text_mask.float().view(-1, 1, 1)
        fusion_tokens = torch.cat(
            [
                updated_smiles + self.type_embeddings[0].view(1, 1, -1),
                updated_text + self.type_embeddings[1].view(1, 1, -1),
                protein_token + self.type_embeddings[2].view(1, 1, -1),
            ],
            dim=1,
        )
        pair_query = (
            (updated_smiles + protein_token) / 2.0
            + self.type_embeddings[3].view(1, 1, -1)
        )
        pair_token = self._attend(pair_query, fusion_tokens, self.fusion_attn, self.norm_pair)
        updated_protein = self.norm_protein(
            protein_token + 0.5 * (updated_smiles + updated_text)
        )
        return (
            updated_smiles.squeeze(1),
            updated_text.squeeze(1),
            updated_protein.squeeze(1),
            pair_token.squeeze(1),
        )


class PairSemanticHead(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mid_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(
        self,
        molecule_features: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([molecule_features, protein_features], dim=-1)).squeeze(-1)


class GraphContextSupport(nn.Module):
    """Independent additive support logit from graph-context features."""

    def __init__(self, graph_dim: int) -> None:
        super().__init__()
        self.logit_head = nn.Linear(graph_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.logit_head.weight)

    def forward(
        self,
        graph_context: torch.Tensor | None,
        available: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if graph_context is None:
            return torch.zeros_like(reference)
        graph_context = graph_context.to(device=reference.device, dtype=reference.dtype)
        return self.logit_head(graph_context).squeeze(-1) * available


def _zero_init_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class AuxiliaryScoreAdapter(nn.Module):
    def __init__(
        self,
        pair_feature_dim: int,
        hidden_dim: int,
        initial_scale: float,
        dropout: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        input_dim = pair_feature_dim + 2
        self.base_scale = nn.Parameter(torch.tensor(float(initial_scale), dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        _zero_init_last_linear(self.residual)
        _zero_init_last_linear(self.gate)
        last = self.gate[-1]
        assert isinstance(last, nn.Linear)
        nn.init.constant_(last.bias, gate_bias)

    def forward(
        self,
        pair_features: torch.Tensor,
        score: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            [pair_features, score.unsqueeze(-1), mask.unsqueeze(-1)], dim=-1
        )
        gate = torch.sigmoid(self.gate(inputs).squeeze(-1)) * mask
        residual = self.residual(inputs).squeeze(-1)
        return gate * (self.base_scale * score + residual), gate


class FusionCalibrationHead(nn.Module):
    def __init__(self, pair_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_feature_dim + 7, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        _zero_init_last_linear(self.net)

    def forward(
        self,
        pair_features: torch.Tensor,
        base_logits: torch.Tensor,
        kge_component: torch.Tensor,
        kge_gate: torch.Tensor,
        normalized_kge: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros_like(base_logits)
        inputs = torch.cat(
            [
                pair_features,
                base_logits.unsqueeze(-1),
                zero.unsqueeze(-1),
                kge_component.unsqueeze(-1),
                zero.unsqueeze(-1),
                kge_gate.unsqueeze(-1),
                zero.unsqueeze(-1),
                normalized_kge.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(inputs).squeeze(-1)


class ConfidenceMarginKGERefinement(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        pair_dim = int(config.projection_dim) * 4
        self.adapter = AuxiliaryScoreAdapter(
            pair_feature_dim=pair_dim,
            hidden_dim=config.kge_hidden_dim,
            initial_scale=config.kge_initial_scale,
            dropout=config.kge_dropout,
            gate_bias=config.kge_adapter_gate_bias,
        )
        self.calibration = FusionCalibrationHead(
            pair_feature_dim=pair_dim,
            hidden_dim=config.kge_hidden_dim,
            dropout=config.kge_dropout,
        )
        self.register_buffer("norm_mean", torch.tensor(float(config.kge_norm_mean)))
        self.register_buffer("norm_std", torch.tensor(float(config.kge_norm_std)))
        self.norm_min_std = float(config.kge_norm_min_std)
        self.residual_shrink = float(config.kge_residual_shrink)
        self.residual_bound = float(config.kge_residual_bound)
        self.margin_bias = float(config.confidence_margin_bias)
        self.margin_temperature = max(float(config.confidence_margin_temperature), 1e-6)

    def confidence_gate(
        self,
        base_logits: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.margin_bias - base_logits.abs() / self.margin_temperature
        return torch.sigmoid(logits) * available

    def bound_residual(self, residual: torch.Tensor) -> torch.Tensor:
        bound = max(self.residual_bound, 1e-8)
        return bound * torch.tanh(residual / bound)

    def forward(
        self,
        base_logits: torch.Tensor,
        molecule_features: torch.Tensor,
        protein_features: torch.Tensor,
        kge_score: torch.Tensor | None,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kge_score is None:
            zero = torch.zeros_like(base_logits)
            return base_logits, zero, zero
        kge_score = kge_score.to(device=base_logits.device, dtype=base_logits.dtype).view(-1)
        std = self.norm_std.to(base_logits).clamp_min(self.norm_min_std)
        normalized = (kge_score - self.norm_mean.to(base_logits)) / std
        pair_features = build_pair_features(molecule_features, protein_features)
        component, adapter_gate = self.adapter(pair_features, normalized, available)
        calibration = self.calibration(
            pair_features, base_logits, component, adapter_gate, normalized
        )
        candidate = (component + calibration) * available
        confidence_gate = self.confidence_gate(base_logits, available)
        post_gate = candidate * confidence_gate * self.residual_shrink
        bounded = self.bound_residual(post_gate)
        return base_logits + bounded, bounded, confidence_gate


@dataclass
class SCRDTIOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    pair_logit: torch.Tensor
    graph_logit: torch.Tensor
    kge_residual: torch.Tensor
    confidence_gate: torch.Tensor
    molecule_repr: torch.Tensor
    protein_repr: torch.Tensor
    pair_repr: torch.Tensor


class SCRDTI(nn.Module):
    """Paper-facing SCR-DTI predictor.

    Inputs are offline pooled molecular/text/protein embeddings. Optional graph
    context and KGE scores enter only through their explicitly separated paths.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.smiles_projector = _projection_mlp(
            cfg.smiles_input_dim,
            cfg.projection_dim,
            cfg.projector_hidden_dim,
            cfg.projector_mid_dim,
            cfg.projector_dropout,
        )
        self.text_projector = _projection_mlp(
            cfg.text_input_dim,
            cfg.projection_dim,
            cfg.projector_hidden_dim,
            cfg.projector_mid_dim,
            cfg.projector_dropout,
        )
        self.protein_projector = _projection_mlp(
            cfg.protein_input_dim,
            cfg.projection_dim,
            cfg.projector_hidden_dim,
            cfg.projector_mid_dim,
            cfg.projector_dropout,
        )
        self.hca = (
            HierarchicalCrossAttention(
                dim=cfg.projection_dim,
                num_heads=cfg.hca_num_heads,
                hidden_dim=cfg.hca_hidden_dim,
                dropout=cfg.hca_dropout,
            )
            if cfg.use_hca
            else None
        )
        self.pair_head = PairSemanticHead(
            dim=cfg.projection_dim,
            hidden_dim=cfg.pair_hidden_dim,
            mid_dim=cfg.pair_mid_dim,
            dropout=cfg.pair_dropout,
        )
        self.hca_logit_head = (
            nn.Sequential(
                nn.Linear(cfg.projection_dim, max(cfg.pair_hidden_dim, cfg.projection_dim)),
                nn.GELU(),
                nn.LayerNorm(max(cfg.pair_hidden_dim, cfg.projection_dim)),
                nn.Dropout(cfg.pair_dropout),
                nn.Linear(max(cfg.pair_hidden_dim, cfg.projection_dim), 1),
            )
            if cfg.use_hca
            else None
        )
        self.graph_context = (
            GraphContextSupport(cfg.graph_context_dim) if cfg.use_graph_context else None
        )
        self.kge_refinement = (
            ConfidenceMarginKGERefinement(cfg) if cfg.use_kge_refinement else None
        )

    @staticmethod
    def _mask(
        value: torch.Tensor | None,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, device=reference.device, dtype=reference.dtype)
        return value.to(device=reference.device, dtype=reference.dtype).view(-1)

    def project(
        self,
        smiles_emb: torch.Tensor,
        text_emb: torch.Tensor,
        protein_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smiles = F.normalize(self.smiles_projector(smiles_emb), dim=1)
        text = F.normalize(self.text_projector(text_emb), dim=1)
        protein = F.normalize(self.protein_projector(protein_emb), dim=1)
        return smiles, text, protein

    def forward(
        self,
        smiles_emb: torch.Tensor,
        text_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        *,
        has_text: torch.Tensor | None = None,
        graph_context: torch.Tensor | None = None,
        graph_available: torch.Tensor | None = None,
        kge_score: torch.Tensor | None = None,
        kge_available: torch.Tensor | None = None,
    ) -> SCRDTIOutput:
        smiles, text, protein = self.project(smiles_emb, text_emb, protein_emb)
        batch_size = smiles.shape[0]
        text_mask = self._mask(has_text, batch_size, smiles)
        pair_repr = F.normalize(smiles + protein, dim=-1)
        if self.hca is not None:
            smiles, text, protein, pair_repr = self.hca(
                smiles, text, protein, text_mask=text_mask
            )
        pair_logit = self.pair_head(smiles, protein)
        if self.hca_logit_head is not None:
            pair_logit = pair_logit + self.hca_logit_head(pair_repr).squeeze(-1)

        graph_mask = self._mask(graph_available, batch_size, pair_logit)
        if self.graph_context is None:
            graph_logit = torch.zeros_like(pair_logit)
        else:
            graph_logit = self.graph_context(
                graph_context, graph_mask, reference=pair_logit
            )
        base_logits = pair_logit + graph_logit

        kge_mask = self._mask(kge_available, batch_size, base_logits)
        if self.kge_refinement is None:
            logits = base_logits
            kge_residual = torch.zeros_like(base_logits)
            confidence_gate = torch.zeros_like(base_logits)
        else:
            logits, kge_residual, confidence_gate = self.kge_refinement(
                base_logits,
                smiles,
                protein,
                kge_score,
                kge_mask,
            )
        return SCRDTIOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            pair_logit=pair_logit,
            graph_logit=graph_logit,
            kge_residual=kge_residual,
            confidence_gate=confidence_gate,
            molecule_repr=smiles,
            protein_repr=protein,
            pair_repr=F.normalize(pair_repr, dim=-1),
        )

