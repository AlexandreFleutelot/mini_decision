"""Modèle complet : encodeur siamois partagé + projection du contexte + tête + porte."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mini_decision.config import ModelConfig
from mini_decision.model.encoders import CONTEXT, DESCRIPTION, Encoder, build_encoder
from mini_decision.model.heads import HEADS, ContextProjection, GatedOutput


@dataclass
class Batch:
    """Textes tokenisés : ids [B, L] et masque booléen [B, L] pour chaque champ."""

    ctx_ids: torch.Tensor
    ctx_mask: torch.Tensor
    true_ids: torch.Tensor
    true_mask: torch.Tensor
    false_ids: torch.Tensor
    false_mask: torch.Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


@dataclass
class Output:
    logit: torch.Tensor  # [B] ; P(vrai) = sigmoid(logit)
    s_true: torch.Tensor
    s_false: torch.Tensor
    gate: torch.Tensor
    z: torch.Tensor  # [B, k]


def _pad_to(ids: torch.Tensor, length: int, value: int) -> torch.Tensor:
    return nn.functional.pad(ids, (0, length - ids.shape[1]), value=value)


class DecisionModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.cfg, self.pad_id = cfg, pad_id
        self.encoder: Encoder = build_encoder(vocab_size, cfg, pad_id)
        d = self.encoder.d_model
        self.context_proj = ContextProjection(d, cfg.k)
        self.head = HEADS[cfg.head](d, cfg)
        self.output = GatedOutput(cfg.tau_init)

    def encode_descriptions(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Un seul passage d'encodeur pour les deux descriptions (concaténées sur le batch)."""
        length = max(batch.true_ids.shape[1], batch.false_ids.shape[1])
        ids = torch.cat([_pad_to(batch.true_ids, length, self.pad_id), _pad_to(batch.false_ids, length, self.pad_id)])
        mask = torch.cat([_pad_to(batch.true_mask, length, False), _pad_to(batch.false_mask, length, False)])
        h, _ = self.encoder(ids, mask, DESCRIPTION)
        return h.chunk(2)

    def forward(self, batch: Batch) -> Output:
        h_ctx, _ = self.encoder(batch.ctx_ids, batch.ctx_mask, CONTEXT)
        z = self.context_proj(h_ctx)
        h_true, h_false = self.encode_descriptions(batch)
        s_true, s_false = self.head.score(z, h_true), self.head.score(z, h_false)
        logit, gate = self.output(s_true, s_false)
        return Output(logit, s_true, s_false, gate, z)

    def set_rda_alpha(self, alpha: float) -> None:
        if hasattr(self.head, "alpha"):
            self.head.alpha.fill_(min(max(alpha, 0.0), 1.0))


class MLMHead(nn.Module):
    """Prédit le token d'origine à chaque position ; décodeur lié à l'embedding des tokens.

    Utilisée seulement pendant le pré-entraînement MLM (et la perte auxiliaire), puis jetée.
    """

    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        d = encoder.d_model
        self.transform = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.embedding = encoder.token_embedding  # poids partagés, pas copiés
        self.bias = nn.Parameter(torch.zeros(self.embedding.num_embeddings))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.transform(states) @ self.embedding.weight.T + self.bias


def build_model(vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> DecisionModel:
    return DecisionModel(vocab_size, cfg, pad_id)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_table(model: DecisionModel) -> dict[str, int]:
    """Exemple : {'encoder': 1_100_000, 'context_proj': 2_064, 'head': 4_144, 'output': 2, 'total': ...}."""
    parts = {name: count_parameters(m) for name, m in model.named_children()}
    parts["total"] = count_parameters(model)
    return parts
