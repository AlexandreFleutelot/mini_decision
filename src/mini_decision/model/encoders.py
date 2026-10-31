"""Encodeurs siamois : un texte (ids de tokens) -> un vecteur h ∈ R^d.

Le même encodeur sert pour le contexte et les descriptions ; un « type embedding »
(0 = contexte, 1 = description) lui dit lequel il lit. Trois barreaux d'échelle :
n-grammes hachés (sac de mots enrichi), CNN dilaté, transformer pre-LN.

Chaque encodeur renvoie (h [B, d], états par token [B, L, d]) : les états servent au MLM,
dont la tête de sortie est liée à `token_embedding`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mini_decision.config import ModelConfig

CONTEXT, DESCRIPTION = 0, 1


class AttentionPooling(nn.Module):
    """Moyenne pondérée des états, poids = softmax(⟨état, q⟩ / √d) sur les vrais tokens."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d))  # zéro au départ : moyenne simple

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = states @ self.query / math.sqrt(states.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        return torch.einsum("bl,bld->bd", weights, states)


class Encoder(nn.Module):
    d_model: int
    token_embedding: nn.Embedding

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, type_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        t = cfg.transformer
        self.d_model = t.d_model
        self.token_embedding = nn.Embedding(vocab_size, t.d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(cfg.max_len, t.d_model)
        self.type_embedding = nn.Embedding(2, t.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=t.d_model,
            nhead=t.n_heads,
            dim_feedforward=t.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN : plus stable sans warmup long
        )
        self.layers = nn.TransformerEncoder(layer, t.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(t.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool = AttentionPooling(t.d_model)

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, type_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(ids.shape[1], device=ids.device)
        types = torch.full_like(ids, type_id)
        x = self.token_embedding(ids) + self.position_embedding(positions) + self.type_embedding(types)
        x = self.layers(self.dropout(x), src_key_padding_mask=~mask)
        states = self.final_norm(x)
        return self.pool(states, mask), states


class CNNEncoder(Encoder):
    """Convolutions 1D dilatées résiduelles : champ récepteur 1+2·(1+2+4+8) = 31 tokens."""

    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        c = cfg.cnn
        if len(c.dilations) != c.n_layers:
            raise ValueError("cnn.dilations doit avoir n_layers éléments")
        self.d_model = c.d_model
        self.token_embedding = nn.Embedding(vocab_size, c.d_model, padding_idx=pad_id)
        self.type_embedding = nn.Embedding(2, c.d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(c.d_model) for _ in c.dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(c.d_model, c.d_model, c.kernel_size, dilation=d, padding="same")
            for d in c.dilations
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.final_norm = nn.LayerNorm(c.d_model)
        self.pool = AttentionPooling(c.d_model)

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, type_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = mask.unsqueeze(-1).float()
        x = self.token_embedding(ids) + self.type_embedding(torch.full_like(ids, type_id))
        for norm, conv in zip(self.norms, self.convs):
            y = conv((norm(x) * keep).transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(nn.functional.gelu(y))
        states = self.final_norm(x) * keep
        return self.pool(states, mask), states


class NgramEncoder(Encoder):
    """Unigrammes (table du vocabulaire) + n-grammes 2..n_max hachés dans n_buckets cases.

    État du token i = somme des embeddings des n-grammes qui finissent en i ; h = MLP(moyenne).
    Exemple : « ne pas payer » -> unigrammes ne, pas, payer + bigrammes (ne pas), (pas payer)...
    """

    PRIME = 1_000_003

    def __init__(self, vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        n = cfg.ngram
        self.n_max, self.n_buckets = n.n_max, n.n_buckets
        self.d_model = n.d_model
        self.token_embedding = nn.Embedding(vocab_size, n.d_model, padding_idx=pad_id)
        self.ngram_embedding = nn.Embedding(n.n_buckets, n.d_model)
        self.type_embedding = nn.Embedding(2, n.d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(n.d_model),
            nn.Linear(n.d_model, n.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(n.hidden, n.d_model),
        )
        self.final_norm = nn.LayerNorm(n.d_model)

    def ngram_ids(self, ids: torch.Tensor, mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Hache le n-gramme finissant en chaque position ; valide si ses n tokens sont réels."""
        h = torch.full_like(ids, n)  # le n entre dans le hash : bigramme ≠ trigramme
        valid = mask.clone()
        for j in range(n):
            shifted = torch.roll(ids, shifts=j, dims=1)
            shifted_mask = torch.roll(mask, shifts=j, dims=1)
            if j:
                shifted_mask[:, :j] = False
            h = (h * self.PRIME + shifted) % (2**31 - 1)
            valid &= shifted_mask
        return h % self.n_buckets, valid

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor, type_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.token_embedding(ids)
        for n in range(2, self.n_max + 1):
            bucket, valid = self.ngram_ids(ids, mask, n)
            states = states + self.ngram_embedding(bucket) * valid.unsqueeze(-1)
        keep = mask.unsqueeze(-1).float()
        states = states * keep
        mean = states.sum(1) / keep.sum(1).clamp(min=1.0)
        mean = mean + self.type_embedding.weight[type_id]
        return self.final_norm(mean + self.mlp(mean)), states


ENCODERS: dict[str, type[Encoder]] = {
    "transformer": TransformerEncoder,
    "cnn": CNNEncoder,
    "ngram": NgramEncoder,
}


def build_encoder(vocab_size: int, cfg: ModelConfig, pad_id: int = 0) -> Encoder:
    return ENCODERS[cfg.encoder](vocab_size, cfg, pad_id)
