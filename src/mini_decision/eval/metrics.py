"""Métriques sur des probabilités P(vrai) face à des labels souples.

Exemple : p = [0.9, 0.2, 0.5], y = [1.0, 0.1, 0.5] (le 3e est hors sujet).
- accuracy : seulement là où y ≠ 0.5 -> (0.9 > 0.5) == (1.0 > 0.5) et (0.2 < 0.5) : 100 %.
- log-loss souple : −[y·log p + (1−y)·log(1−p)], minimale quand p = y.
- AUROC hors sujet : peut-on séparer les hors-sujet avec |p − 0.5| ? (1 = parfait).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def soft_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error : écart moyen |confiance − fréquence observée| par tranche."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def decided_accuracy(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
    sel = y != 0.5
    if not sel.any():
        return float("nan")
    return float(((p[sel] > threshold) == (y[sel] > 0.5)).mean())


def off_topic_auroc(p: np.ndarray, off_topic: np.ndarray) -> float:
    """AUROC « dans le sujet » vs « hors sujet » avec le score |p − 0.5|."""
    if off_topic.all() or not off_topic.any():
        return float("nan")
    return float(roc_auc_score(~off_topic, np.abs(p - 0.5)))


def all_metrics(
    p: np.ndarray, y: np.ndarray, kinds: list[str], n_bins: int = 15, threshold: float = 0.5
) -> dict[str, float]:
    kinds_arr = np.array(kinds)
    off = kinds_arr == "off_topic"
    cf = kinds_arr == "counterfactual"
    return {
        "n": float(len(p)),
        "log_loss": soft_log_loss(p, y),
        "accuracy": decided_accuracy(p, y, threshold),
        "brier": brier(p, y),
        "ece": ece(p, y, n_bins),
        "off_topic_auroc": off_topic_auroc(p, off),
        "counterfactual_accuracy": decided_accuracy(p[cf], y[cf], threshold) if cf.any() else float("nan"),
        "off_topic_mean_abs_dev": float(np.abs(p[off] - 0.5).mean()) if off.any() else float("nan"),
    }
