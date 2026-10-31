"""Entraînement supervisé : BCE sur labels souples, AdamW, warmup + cosinus, RDA, early stopping.

RDA : α (mélange nuage propre / nuage partagé) monte linéairement de 0 à 1 sur
`rda_alpha_end_frac` de l'entraînement. Le meilleur checkpoint et l'early stopping ne
comptent qu'à partir de α = 1 : le modèle gardé est toujours la QDA finale.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mini_decision.config import Config
from mini_decision.eval.metrics import all_metrics
from mini_decision.model import DecisionModel, MLMHead, build_model, parameter_table
from mini_decision.model.encoders import CONTEXT
from mini_decision.schemas import Example
from mini_decision.tokenizer import Tokenizer
from mini_decision.train.common import EarlyStopping, Log, RunLog, load_state, save_json
from mini_decision.train.data import Encoded, batches, encode_examples, pad, subset_by_pairs
from mini_decision.train.masking import IGNORE
from mini_decision.train.mlm import make_masker, mlm_dir
from mini_decision.train.schedule import make_optimizer


def run_dir(cfg: Config, suffix: str = "") -> Path:
    return cfg.paths.runs_dir / f"{cfg.run_name}{suffix}"


@torch.no_grad()
def predict(model: DecisionModel, items: list[Encoded], batch_size: int, pad_id: int, device: torch.device) -> np.ndarray:
    """P(vrai) pour chaque exemple, dans l'ordre."""
    model.eval()
    out = []
    for batch, _, _ in batches(items, batch_size, pad_id):
        out.append(torch.sigmoid(model(batch.to(device)).logit).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def evaluate(model: DecisionModel, items: list[Encoded], cfg: Config, pad_id: int, device: torch.device) -> dict[str, float]:
    p = predict(model, items, cfg.train.batch_size, pad_id, device)
    y = np.array([it.label for it in items])
    return all_metrics(p, y, [it.kind.value for it in items], cfg.eval.ece_bins, cfg.eval.decision_threshold)


def load_model(path: Path, cfg: Config, vocab_size: int, pad_id: int, device: torch.device) -> DecisionModel:
    model = build_model(vocab_size, cfg.model, pad_id).to(device)
    model.load_state_dict(load_state(path, device))
    return model.eval()


def _aux_mlm_loss(model: DecisionModel, head: MLMHead, masker, items: list[Encoded], rng: random.Random, pad_id: int, device: torch.device) -> torch.Tensor:
    masked = [masker(it.ctx, rng) for it in items]
    ids, mask = pad([m[0] for m in masked], pad_id)
    labels, _ = pad([m[1] for m in masked], IGNORE)
    _, states = model.encoder(ids.to(device), mask.to(device), CONTEXT)
    logits = head(states)
    return nn.functional.cross_entropy(logits.flatten(0, 1), labels.to(device).flatten(), ignore_index=IGNORE)


def train_supervised(
    cfg: Config,
    tok: Tokenizer,
    train_examples: list[Example],
    val_examples: list[Example],
    device: torch.device,
    out: Path,
    echo: Log = print,
) -> dict[str, Any]:
    t = cfg.train
    log = RunLog(out, echo)
    train_examples = subset_by_pairs(train_examples, t.max_train_examples, cfg.seed)
    train_items = encode_examples(train_examples, tok, cfg.model.max_len)
    val_items = encode_examples(val_examples, tok, cfg.model.max_len)

    model = build_model(tok.vocab_size, cfg.model, tok.pad_id).to(device)
    init = "aléatoire"
    if t.init_from_mlm:
        ckpt = mlm_dir(cfg) / "encoder.pt"
        if ckpt.exists():
            model.encoder.load_state_dict(load_state(ckpt, device))
            init = str(ckpt)
        else:
            echo(f"[train] ATTENTION : {ckpt} absent, encodeur initialisé au hasard (lancer pretrain-mlm)")
    aux_head = MLMHead(model.encoder).to(device) if t.aux_mlm_lambda > 0 else None
    masker = make_masker(tok, cfg) if aux_head is not None else None
    trainable = nn.ModuleDict({"model": model, **({"aux": aux_head} if aux_head else {})})

    steps_per_epoch = -(-len(train_items) // t.batch_size)
    total = steps_per_epoch * t.max_epochs
    alpha_end = max(1, round(t.rda_alpha_end_frac * total))
    optimizer, scheduler = make_optimizer(trainable, t.lr, t.weight_decay, total, t.warmup_frac)
    stopper = EarlyStopping(t.patience)
    rng = random.Random(cfg.seed)
    table = parameter_table(model)
    log.write("train_start", exemples=len(train_items), paires=len({e.pair_id for e in train_examples}),
              val=len(val_items), parametres=table["total"], init=init, device=str(device))
    save_json(out / "config.json", cfg.model_dump(mode="json"))

    step, best_metrics, best_epoch = 0, {}, 0
    for epoch in range(1, t.max_epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        running, n_batches = 0.0, 0
        for batch, labels, chunk in batches(train_items, t.batch_size, tok.pad_id, rng, t.swap_prob):
            model.set_rda_alpha(step / alpha_end)
            out_ = model(batch.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(out_.logit, labels.to(device))
            if aux_head is not None:
                loss = loss + t.aux_mlm_lambda * _aux_mlm_loss(model, aux_head, masker, chunk, rng, tok.pad_id, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), t.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1
            step += 1
        model.set_rda_alpha(step / alpha_end)  # α du modèle évalué et sauvegardé
        alpha = float(model.head.alpha) if hasattr(model.head, "alpha") else 1.0
        val = evaluate(model, val_items, cfg, tok.pad_id, device)
        # Tant que α < 1 le modèle n'est pas encore la QDA finale : ni sauvegarde ni arrêt.
        best = stopper.step(val["log_loss"]) if alpha >= 1.0 else False
        if best:
            torch.save(model.state_dict(), out / "best.pt")
            best_metrics, best_epoch = val, epoch
        log.write("train_epoch", epoque=epoch, train_loss=running / max(n_batches, 1), val_log_loss=val["log_loss"],
                  val_acc=val["accuracy"], val_ece=val["ece"], val_auroc_hs=val["off_topic_auroc"],
                  alpha=round(alpha, 3), tau=round(model.output.tau.item(), 2), meilleur=best)
        if stopper.should_stop:
            break
    summary = {
        "run": out.name,
        "n_train": len(train_items),
        "n_train_pairs": len({e.pair_id for e in train_examples}),
        "best_epoch": best_epoch,
        "init": init,
        "parameters": table,
        "val": best_metrics,
        "encoder": cfg.model.encoder,
        "head": cfg.model.head,
        "cov": cfg.model.cov,
    }
    save_json(out / "summary.json", summary)
    log.write("train_fin", meilleure_epoque=best_epoch, val_log_loss=best_metrics.get("log_loss", float("nan")),
              val_acc=best_metrics.get("accuracy", float("nan")), checkpoint=str(out / "best.pt"))
    return summary
