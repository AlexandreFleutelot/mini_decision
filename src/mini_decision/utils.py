"""Petits utilitaires partagés : graines, choix du device, affichage de l'avancement."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    """`auto` choisit cuda, puis mps (Apple Silicon), puis cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Progress:
    """Ligne d'avancement périodique : fait/total, débit, temps restant, infos en plus.

    Exemple : « [verify] 450/3000 (15%) · 2.1/s · reste ~20 min · gardés 400 · 0.32 $ »
    """

    def __init__(
        self,
        label: str,
        total: int,
        log: Callable[[str], None],
        every_s: float = 5.0,
        extra: Callable[[], str] | None = None,
    ) -> None:
        self.label = label
        self.total = total
        self.log = log
        self.every_s = every_s
        self.extra = extra
        self.done = 0
        self.start = time.monotonic()
        self.last = 0.0

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.every_s or self.done >= self.total:
            self.last = now
            self.log(self.line(now))

    def line(self, now: float) -> str:
        elapsed = max(now - self.start, 1e-9)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        parts = [
            f"[{self.label}] {self.done}/{self.total} ({self.done / max(self.total, 1):.0%})",
            f"{rate:.1f}/s",
            f"écoulé {_duration(elapsed)}",
            f"reste ~{_duration(remaining)}",
        ]
        if self.extra is not None:
            parts.append(self.extra())
        return " · ".join(parts)


def _duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"
