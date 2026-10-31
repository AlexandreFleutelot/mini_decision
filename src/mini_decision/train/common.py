"""Outils communs aux boucles d'entraînement : journal JSONL, early stopping, checkpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

Log = Callable[[str], None]


class RunLog:
    """Écrit une ligne JSON par événement dans <run_dir>/log.jsonl et l'affiche."""

    def __init__(self, run_dir: Path, echo: Log = print) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "log.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.echo = echo
        self.start = time.monotonic()

    def write(self, event: str, **values: Any) -> None:
        record = {"event": event, "t": round(time.monotonic() - self.start, 1), **values}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        shown = " · ".join(f"{k} {_fmt(v)}" for k, v in values.items())
        self.echo(f"[{event}] {shown} · {record['t']:.0f} s")


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


class EarlyStopping:
    """Arrête après `patience` époques sans amélioration (plus bas = meilleur)."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Renvoie True si c'est le meilleur score jusqu'ici."""
        if value < self.best - 1e-6:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_state(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)
