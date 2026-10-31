"""Figures : courbe d'échelle (et diagramme de fiabilité au jalon 5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage : on écrit des PNG
import matplotlib.pyplot as plt


def scaling_plot(rows: list[dict], path: Path, title: str = "") -> None:
    """Log-loss, accuracy et ECE de validation en fonction du nombre d'exemples (axe log)."""
    n = [r["n_train"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(axes, ("log_loss", "accuracy", "ece"), ("log-loss", "accuracy (hors 0.5)", "ECE")):
        ax.plot(n, [r[key] for r in rows], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("exemples d'entraînement")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Courbe d'échelle {title}".strip())
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
