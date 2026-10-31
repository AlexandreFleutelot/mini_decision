"""Préparation des données d'entraînement : tokenisation une fois pour toutes, puis batches.

Chaque exemple devient trois listes d'ids (contexte, desc_vrai, desc_faux) + un label souple.
L'échange aléatoire des descriptions (swap_prob) apprend au modèle que la position ne porte
aucune information : (ctx, A, B, 0.9) devient (ctx, B, A, 0.1).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch

from mini_decision.model import Batch
from mini_decision.schemas import Example, Kind
from mini_decision.tokenizer import Tokenizer


@dataclass
class Encoded:
    ctx: list[int]
    desc_true: list[int]
    desc_false: list[int]
    label: float
    kind: Kind
    pair_id: str


def encode_examples(examples: list[Example], tok: Tokenizer, max_len: int) -> list[Encoded]:
    cache: dict[str, list[int]] = {}  # chaque description revient ~15 fois

    def enc(text: str) -> list[int]:
        if text not in cache:
            cache[text] = tok.encode(text, max_len)
        return cache[text]

    return [
        Encoded(enc(e.context), enc(e.desc_true), enc(e.desc_false), e.label, e.kind, e.pair_id)
        for e in examples
    ]


def subset_by_pairs(examples: list[Example], n: int | None, seed: int) -> list[Example]:
    """Garde des paires entières, tirées au hasard, jusqu'à atteindre ~n exemples.

    Les tailles emboîtées partagent le même ordre de paires : le jeu 10k est inclus dans le 30k.
    """
    if n is None or n >= len(examples):
        return examples
    by_pair: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        by_pair[e.pair_id].append(e)
    order = sorted(by_pair)
    random.Random(seed).shuffle(order)
    out: list[Example] = []
    for pid in order:
        if len(out) >= n:
            break
        out += by_pair[pid]
    return out


def pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, ids != pad_id


def collate(items: list[Encoded], pad_id: int, swap: list[bool] | None = None) -> tuple[Batch, torch.Tensor]:
    swap = swap or [False] * len(items)
    trues = [it.desc_false if s else it.desc_true for it, s in zip(items, swap)]
    falses = [it.desc_true if s else it.desc_false for it, s in zip(items, swap)]
    labels = torch.tensor([1 - it.label if s else it.label for it, s in zip(items, swap)])
    batch = Batch(*pad([it.ctx for it in items], pad_id), *pad(trues, pad_id), *pad(falses, pad_id))
    return batch, labels.float()


def batches(
    items: list[Encoded],
    batch_size: int,
    pad_id: int,
    rng: random.Random | None = None,
    swap_prob: float = 0.0,
) -> Iterator[tuple[Batch, torch.Tensor, list[Encoded]]]:
    """Batches mélangés si rng est donné (entraînement), dans l'ordre sinon (évaluation)."""
    order = list(range(len(items)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[start : start + batch_size]]
        swap = [rng.random() < swap_prob for _ in chunk] if rng is not None else None
        batch, labels = collate(chunk, pad_id, swap)
        yield batch, labels, chunk


def mlm_texts(
    examples: list[Example], unlabeled: list[str]
) -> list[tuple[str, int]]:
    """Textes du MLM avec leur type : contextes et non étiqueté -> 0, descriptions -> 1."""
    from mini_decision.model.encoders import CONTEXT, DESCRIPTION

    descs = sorted({d for e in examples for d in (e.desc_true, e.desc_false)})
    return (
        [(e.context, CONTEXT) for e in examples]
        + [(d, DESCRIPTION) for d in descs]
        + [(t, CONTEXT) for t in unlabeled]
    )
