"""Petits utilitaires partagés : graines, choix du device, affichage de l'avancement."""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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
        window_s: float = 60.0,
    ) -> None:
        self.label = label
        self.total = total
        self.log = log
        self.every_s = every_s
        self.extra = extra
        self.done = 0
        self.start = time.monotonic()
        self.last = 0.0
        self.window_s = window_s
        self.samples: deque[tuple[float, int]] = deque([(self.start, 0)])

    def update(self, n: int = 1) -> None:
        self.done += n
        now = time.monotonic()
        self.samples.append((now, self.done))
        while len(self.samples) > 2 and now - self.samples[1][0] > self.window_s:
            self.samples.popleft()
        if now - self.last >= self.every_s or self.done >= self.total:
            self.last = now
            self.log(self.line(now))

    def line(self, now: float) -> str:
        elapsed = max(now - self.start, 1e-9)
        # Débit récent (fenêtre glissante) : les éléments servis par le cache au début
        # ne doivent pas faire croire que la suite ira aussi vite.
        t0, d0 = self.samples[0]
        rate = (self.done - d0) / max(now - t0, 1e-9)
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


@contextmanager
def interruptible_pool(max_workers: int) -> Iterator[ThreadPoolExecutor]:
    """ThreadPoolExecutor qui s'arrête net sur Ctrl+C.

    Sans cela, Python attend la fin des requêtes HTTP en cours avant de quitter. Les réponses
    déjà reçues sont dans le cache disque (écritures atomiques) : relancer la commande reprend
    là où on s'était arrêté.
    """
    pool = ThreadPoolExecutor(max_workers)
    try:
        yield pool
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print(
            "\nInterrompu. Les appels terminés sont en cache : relancez pour reprendre.",
            file=sys.stderr,
        )
        os._exit(130)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
