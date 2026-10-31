"""Masquage MLM par mots entiers, avec suréchantillonnage des mots de négation et des nombres.

Un « mot » = un token SentencePiece qui commence par « ▁ » suivi de ses morceaux. On masque
round(mask_prob · nb_mots) mots (au moins 1). Les mots listés dans `oversample_words`
(« pas », « jamais », « n' »...) et ceux qui contiennent un chiffre ont un poids ×3 au tirage.
Chaque token masqué devient [MASK] (80 %), un token aléatoire (10 %) ou reste tel quel (10 %).

Exemple : « ▁Le ▁colis ▁n ' est ▁pas ▁arrivé » -> mots [Le] [colis] [n'est] [pas] [arrivé] ;
« n'est » (commence par n') et « pas » ont 3 fois plus de chances d'être masqués.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from mini_decision.config import MLMConfig

IGNORE = -100  # valeur ignorée par CrossEntropyLoss
WORD_START = "▁"
_PUNCT = string.punctuation + "«»“”’…"


def normalize_word(word: str) -> str:
    return word.replace("’", "'").lower().strip(_PUNCT.replace("'", ""))


def group_words(ids: list[int], pieces: list[str], special_ids: set[int]) -> list[list[int]]:
    """Positions des tokens regroupées par mot (les tokens spéciaux ne forment aucun mot)."""
    words: list[list[int]] = []
    for pos, i in enumerate(ids):
        if i in special_ids:
            continue
        if pieces[i].startswith(WORD_START) or not words or words[-1][-1] != pos - 1:
            words.append([pos])
        else:
            words[-1].append(pos)
    return words


@dataclass
class WordMasker:
    pieces: list[str]  # pieces[id] = morceau SentencePiece
    special_ids: set[int]
    mask_id: int
    cfg: MLMConfig

    def __post_init__(self) -> None:
        self.targets = {normalize_word(w) for w in self.cfg.oversample_words}
        self.prefixes = tuple(w for w in self.targets if w.endswith("'"))  # n', ...
        self.random_pool = [i for i in range(len(self.pieces)) if i not in self.special_ids]

    def word_text(self, ids: list[int], positions: list[int]) -> str:
        return normalize_word("".join(self.pieces[ids[p]] for p in positions).lstrip(WORD_START))

    def is_oversampled(self, word: str) -> bool:
        if word in self.targets or (self.prefixes and word.startswith(self.prefixes)):
            return True
        return self.cfg.oversample_numbers and any(ch.isdigit() for ch in word)

    def choose_words(self, ids: list[int], rng: random.Random) -> list[list[int]]:
        words = group_words(ids, self.pieces, self.special_ids)
        if not words:
            return []
        n = min(len(words), max(1, round(self.cfg.mask_prob * len(words))))
        weights = [
            self.cfg.oversample_factor if self.is_oversampled(self.word_text(ids, w)) else 1.0
            for w in words
        ]
        # Tirage pondéré sans remise (Efraimidis-Spirakis) : clé u^(1/w), on garde les n plus grandes.
        keys = [rng.random() ** (1.0 / w) for w in weights]
        chosen = sorted(range(len(words)), key=keys.__getitem__, reverse=True)[:n]
        return [words[i] for i in sorted(chosen)]

    def __call__(self, ids: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
        """Renvoie (ids d'entrée masqués, labels : id d'origine aux positions masquées, sinon −100)."""
        inputs, labels = list(ids), [IGNORE] * len(ids)
        for word in self.choose_words(ids, rng):
            for pos in word:
                labels[pos] = ids[pos]
                u = rng.random()
                if u < self.cfg.mask_token_prob:
                    inputs[pos] = self.mask_id
                elif u < self.cfg.mask_token_prob + self.cfg.random_token_prob:
                    inputs[pos] = rng.choice(self.random_pool)
                # sinon : token inchangé, mais toujours à prédire
        return inputs, labels
