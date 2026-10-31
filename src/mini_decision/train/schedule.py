"""Optimiseur AdamW et planning du taux d'apprentissage : warmup linéaire puis cosinus."""

from __future__ import annotations

import math

import torch
from torch import nn


def warmup_cosine(step: int, total: int, warmup_frac: float) -> float:
    """Facteur multiplicatif du lr. Exemple (total 100, warmup 5 %) : pas 0 -> 0.2, pas 5 -> 1,
    pas 52 -> 0.5, pas 100 -> 0."""
    warmup = max(1, round(warmup_frac * total))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(
    module: nn.Module, lr: float, weight_decay: float, total_steps: int, warmup_frac: float
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Pas de weight decay sur les biais, normes, embeddings et paramètres scalaires."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() >= 2 and "embedding" not in name
        (decay if is_matrix else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, total_steps, warmup_frac)
    )
    return optimizer, scheduler
