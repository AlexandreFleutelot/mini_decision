"""Pré-entraînement MLM de l'encodeur seul (tête liée à l'embedding, jetée ensuite).

On suit deux précisions sur la validation : sur tous les tokens masqués, et sur les mots
suréchantillonnés (négation, nombres) : c'est là que le MLM doit apprendre l'essentiel.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mini_decision.config import Config
from mini_decision.model.encoders import Encoder, build_encoder
from mini_decision.model.model import MLMHead
from mini_decision.tokenizer import Tokenizer
from mini_decision.train.common import EarlyStopping, Log, RunLog, save_json
from mini_decision.train.data import pad
from mini_decision.train.masking import IGNORE, WordMasker, group_words
from mini_decision.train.schedule import make_optimizer


def mlm_dir(cfg: Config) -> Path:
    return cfg.paths.runs_dir / f"{cfg.mlm.run_name}-{cfg.model.encoder}"


@dataclass
class MLMItem:
    ids: list[int]
    type_id: int


def make_masker(tok: Tokenizer, cfg: Config) -> WordMasker:
    pieces = [tok.id_to_piece(i) for i in range(tok.vocab_size)]
    return WordMasker(pieces, tok.special_ids, tok.mask_id, cfg.mlm)


def mlm_batches(
    items: list[MLMItem],
    masker: WordMasker,
    batch_size: int,
    pad_id: int,
    rng: random.Random,
    shuffle: bool,
):
    """Regroupe par type (un passage d'encodeur = un seul type) puis masque à la volée."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for type_id in (0, 1):
        idx = [i for i in order if items[i].type_id == type_id]
        for start in range(0, len(idx), batch_size):
            chunk = [items[i].ids for i in idx[start : start + batch_size]]
            masked = [masker(ids, rng) for ids in chunk]
            ids, mask = pad([m[0] for m in masked], pad_id)
            labels, _ = pad([m[1] for m in masked], IGNORE)
            yield ids, mask, labels, type_id, chunk


def oversampled_positions(masker: WordMasker, chunk: list[list[int]], length: int) -> torch.Tensor:
    flags = torch.zeros(len(chunk), length, dtype=torch.bool)
    for b, ids in enumerate(chunk):
        for word in group_words(ids, masker.pieces, masker.special_ids):
            if masker.is_oversampled(masker.word_text(ids, word)):
                flags[b, word] = True
    return flags


def mlm_loss(
    encoder: Encoder, head: MLMHead, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, type_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    _, states = encoder(ids, mask, type_id)
    logits = head(states)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
    return loss, logits


@torch.no_grad()
def evaluate_mlm(
    encoder: Encoder, head: MLMHead, items: list[MLMItem], masker: WordMasker, cfg: Config, pad_id: int, device: torch.device
) -> dict[str, float]:
    encoder.eval()
    head.eval()
    rng = random.Random(0)  # même masquage à chaque évaluation : courbes comparables
    total_loss = n_tok = n_ok = n_neg = n_neg_ok = 0.0
    for ids, mask, labels, type_id, chunk in mlm_batches(items, masker, cfg.mlm.batch_size, pad_id, rng, False):
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        loss, logits = mlm_loss(encoder, head, ids, mask, labels, type_id)
        target = labels != IGNORE
        ok = (logits.argmax(-1) == labels) & target
        n = target.sum().item()
        total_loss += loss.item() * n
        n_tok += n
        n_ok += ok.sum().item()
        neg = oversampled_positions(masker, chunk, ids.shape[1]).to(device) & target
        n_neg += neg.sum().item()
        n_neg_ok += (ok & neg).sum().item()
    return {
        "loss": total_loss / max(n_tok, 1),
        "acc": n_ok / max(n_tok, 1),
        "acc_negation_numbers": n_neg_ok / max(n_neg, 1),
    }


def pretrain_mlm(
    cfg: Config,
    tok: Tokenizer,
    train_texts: list[tuple[str, int]],
    val_texts: list[tuple[str, int]],
    device: torch.device,
    echo: Log = print,
) -> Path:
    """Entraîne l'encodeur en MLM ; sauve le meilleur dans <mlm_dir>/encoder.pt."""
    out = mlm_dir(cfg)
    log = RunLog(out, echo)
    encoder = build_encoder(tok.vocab_size, cfg.model, tok.pad_id).to(device)
    head = MLMHead(encoder).to(device)
    masker = make_masker(tok, cfg)
    train_items = [MLMItem(tok.encode(t, cfg.model.max_len), ty) for t, ty in train_texts]
    val_items = [MLMItem(tok.encode(t, cfg.model.max_len), ty) for t, ty in val_texts]
    m = cfg.mlm
    steps_per_epoch = sum(-(-sum(it.type_id == ty for it in train_items) // m.batch_size) for ty in (0, 1))
    total = steps_per_epoch * m.max_epochs
    params = nn.ModuleDict({"encoder": encoder, "head": head})
    optimizer, scheduler = make_optimizer(params, m.lr, m.weight_decay, total, m.warmup_frac)
    stopper = EarlyStopping(m.patience)
    rng = random.Random(cfg.seed)
    n_params = sum(p.numel() for p in params.parameters())
    log.write("mlm_start", textes=len(train_items), val=len(val_items), pas_par_epoque=steps_per_epoch,
              parametres=n_params, device=str(device))
    for epoch in range(1, m.max_epochs + 1):
        encoder.train()
        head.train()
        running = 0.0
        n_batches = 0
        for ids, mask, labels, type_id, _ in mlm_batches(train_items, masker, m.batch_size, tok.pad_id, rng, True):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            loss, _ = mlm_loss(encoder, head, ids, mask, labels, type_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params.parameters(), m.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1
        val = evaluate_mlm(encoder, head, val_items, masker, cfg, tok.pad_id, device)
        best = stopper.step(val["loss"])
        if best:
            torch.save(encoder.state_dict(), out / "encoder.pt")
        log.write("mlm_epoch", epoque=epoch, train_loss=running / max(n_batches, 1),
                  val_loss=val["loss"], val_acc=val["acc"],
                  val_acc_negation=val["acc_negation_numbers"], meilleur=best)
        if stopper.should_stop:
            break
    save_json(out / "summary.json", {"best_val_loss": stopper.best, "epochs": epoch, "encoder": cfg.model.encoder})
    log.write("mlm_fin", meilleure_val_loss=stopper.best, checkpoint=str(out / "encoder.pt"))
    return out / "encoder.pt"
