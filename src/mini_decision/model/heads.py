"""Têtes de décision : un point z_c (contexte) face à deux descriptions (vrai / faux).

Tête QDA : chaque description d définit un nuage gaussien N(μ_d, P_d⁻¹) dans R^k, avec
P_d = L_d L_dᵀ (L_d triangulaire inférieure, diagonale > 0). Le score est la log-densité
(à une constante près) :

    s_d = −½‖L_dᵀ(z − μ_d)‖² + Σ log (L_d)_ii

Petit exemple (k = 1) : μ = 0, L = 2 (écart-type 0.5). Pour z = 0.5 : s = −½·1 + log 2 ≈ 0.19 ;
pour z = 3 : s = −½·36 + log 2 ≈ −17.3. Un point loin des deux nuages a deux scores très
négatifs : la porte g = sigmoid(max(s_vrai, s_faux) − τ) tombe à ~0 et P ≈ sigmoid(b) ≈ 0.5.

Tête dot (baseline) : s_d = ⟨z, e_d⟩ / √k, même porte et même sortie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from mini_decision.config import ModelConfig


class ContextProjection(nn.Module):
    """z_c = LayerNorm sans paramètres(Linear(h_c)) : ‖z_c‖² = k, centré."""

    def __init__(self, d: int, k: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, k)
        self.norm = nn.LayerNorm(k, elementwise_affine=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(h))


@dataclass
class Cloud:
    """Nuage d'une description : centre [B, k] et facteur de Cholesky de la précision [B, k, k]."""

    mu: torch.Tensor
    chol: torch.Tensor


def bounded_inverse_scale(raw: torch.Tensor, s_min: float, s_max: float) -> torch.Tensor:
    """Diagonale de L = 1 / s avec s ∈ [s_min, s_max] : précision bornée, jamais dégénérée."""
    return 1.0 / (s_min + (s_max - s_min) * torch.sigmoid(raw))


def lower_from_flat(diag: torch.Tensor, off: torch.Tensor | None, k: int) -> torch.Tensor:
    """Construit L [..., k, k] à partir de la diagonale et des k(k−1)/2 termes sous-diagonaux."""
    chol = torch.diag_embed(diag)
    if off is not None:
        rows, cols = torch.tril_indices(k, k, offset=-1, device=diag.device)
        chol = chol.clone()
        chol[..., rows, cols] = off
    return chol


def _logdet_spd(p: torch.Tensor) -> torch.Tensor:
    """log det d'une matrice symétrique définie positive via Cholesky (repli CPU pour MPS)."""
    device = p.device
    if device.type == "mps":
        p = p.cpu()
    chol = torch.linalg.cholesky(p)
    return (2 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)).to(device)


def gaussian_score(
    z: torch.Tensor,
    cloud: Cloud,
    shared: torch.Tensor | None = None,
    alpha: float = 1.0,
    diagonal: bool = False,
) -> torch.Tensor:
    """Log-densité (sans −k/2·log 2π) sous la précision RDA α·L Lᵀ + (1−α)·L₀ L₀ᵀ.

    diagonal=True (L et L₀ diagonales) : le log det se calcule terme à terme, sans Cholesky.
    """
    delta = z - cloud.mu
    own_quad = torch.einsum("bij,bi->bj", cloud.chol, delta).pow(2).sum(-1)
    own_logdet = 2 * torch.diagonal(cloud.chol, dim1=-2, dim2=-1).log().sum(-1)
    if shared is None or alpha >= 1.0:
        return -0.5 * own_quad + 0.5 * own_logdet
    shared_quad = (delta @ shared).pow(2).sum(-1)  # ‖L₀ᵀ δ‖² = ‖δᵀ L₀‖²
    quad = alpha * own_quad + (1 - alpha) * shared_quad
    if alpha <= 0.0:
        logdet = 2 * torch.diagonal(shared).log().sum().expand_as(quad)
    elif diagonal:
        d_own = torch.diagonal(cloud.chol, dim1=-2, dim2=-1).pow(2)
        d_shared = torch.diagonal(shared).pow(2)
        logdet = (alpha * d_own + (1 - alpha) * d_shared).log().sum(-1)
    else:
        precision = alpha * cloud.chol @ cloud.chol.transpose(-1, -2) + (1 - alpha) * shared @ shared.T
        logdet = _logdet_spd(precision)
    return -0.5 * quad + 0.5 * logdet


class QDAHead(nn.Module):
    def __init__(self, d: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.k, self.s_min, self.s_max = cfg.k, cfg.s_min, cfg.s_max
        self.full = cfg.cov == "full"
        n_off = self.k * (self.k - 1) // 2
        self.mu = nn.Linear(d, self.k)
        self.diag = nn.Linear(d, self.k)
        self.off = nn.Linear(d, n_off) if self.full else None
        if self.off is not None:  # part d'une précision diagonale
            nn.init.zeros_(self.off.weight)
            nn.init.zeros_(self.off.bias)
        # Nuage partagé L₀ (RDA) : diagonale bornée de la même façon, termes croisés si full.
        self.shared_diag = nn.Parameter(torch.zeros(self.k))
        self.shared_off = nn.Parameter(torch.zeros(n_off)) if self.full else None
        self.register_buffer("alpha", torch.tensor(1.0))  # piloté par l'entraînement : 0 -> 1

    def cloud(self, h: torch.Tensor) -> Cloud:
        diag = bounded_inverse_scale(self.diag(h), self.s_min, self.s_max)
        off = self.off(h) if self.off is not None else None
        return Cloud(self.mu(h), lower_from_flat(diag, off, self.k))

    def shared_chol(self) -> torch.Tensor:
        diag = bounded_inverse_scale(self.shared_diag, self.s_min, self.s_max)
        return lower_from_flat(diag, self.shared_off, self.k)

    def score(self, z: torch.Tensor, h_desc: torch.Tensor) -> torch.Tensor:
        return gaussian_score(
            z, self.cloud(h_desc), self.shared_chol(), float(self.alpha), diagonal=not self.full
        )


class DotHead(nn.Module):
    def __init__(self, d: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.k = cfg.k
        self.proj = nn.Linear(d, cfg.k)

    def score(self, z: torch.Tensor, h_desc: torch.Tensor) -> torch.Tensor:
        return (z * self.proj(h_desc)).sum(-1) / math.sqrt(self.k)


class GatedOutput(nn.Module):
    """P = sigmoid(g·(s_vrai − s_faux) + b), g = sigmoid(max(s_vrai, s_faux) − τ).

    Avec tau_init="auto", τ est fixé au premier batch d'entraînement à la moyenne de
    max(s_vrai, s_faux) : la porte démarre vers 0.5 au lieu d'être saturée.
    """

    def __init__(self, tau_init: float | str) -> None:
        super().__init__()
        auto = tau_init == "auto"
        self.tau = nn.Parameter(torch.tensor(0.0 if auto else float(tau_init)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("tau_ready", torch.tensor(not auto))

    def forward(self, s_true: torch.Tensor, s_false: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top = torch.maximum(s_true, s_false)
        if self.training and not bool(self.tau_ready):
            with torch.no_grad():
                self.tau.fill_(top.mean().item())
                self.tau_ready.fill_(True)
        gate = torch.sigmoid(top - self.tau)
        return gate * (s_true - s_false) + self.bias, gate


HEADS: dict[str, type[nn.Module]] = {"qda": QDAHead, "dot": DotHead}
