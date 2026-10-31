import random

from mini_decision.config import load_config
from mini_decision.train.masking import IGNORE, WordMasker, group_words
from tests.conftest import ROOT

# Petit vocabulaire artificiel : ids 0-2 spéciaux, puis des morceaux SentencePiece.
PIECES = ["[PAD]", "[CLS]", "[MASK]", "▁le", "▁colis", "▁n", "'", "est", "▁pas", "▁arrivé",
          "▁facture", "▁de", "▁12", "0", "▁euros", "▁reçue", "▁hier", "▁client", "▁payée"]
ID = {p: i for i, p in enumerate(PIECES)}
SPECIAL = {0, 1, 2}


def masker(**overrides: object) -> WordMasker:
    sets = [f"mlm.{k}={v}" for k, v in overrides.items()]
    cfg = load_config(ROOT / "configs" / "base.yaml", sets).mlm
    return WordMasker(PIECES, SPECIAL, mask_id=2, cfg=cfg)


def ids(*pieces: str) -> list[int]:
    return [1] + [ID[p] for p in pieces]


def test_words_group_sentencepiece_pieces() -> None:
    seq = ids("▁le", "▁colis", "▁n", "'", "est", "▁pas", "▁arrivé")
    assert group_words(seq, PIECES, SPECIAL) == [[1], [2], [3, 4, 5], [6], [7]]
    m = masker()
    assert m.word_text(seq, [3, 4, 5]) == "n'est"
    assert m.is_oversampled("n'est") and m.is_oversampled("pas") and m.is_oversampled("120")
    assert not m.is_oversampled("colis")


def test_whole_words_are_masked_together() -> None:
    m = masker(mask_prob=1.0)
    seq = ids("▁le", "▁colis", "▁n", "'", "est", "▁pas", "▁arrivé")
    inputs, labels = m(seq, random.Random(0))
    assert labels[0] == IGNORE  # [CLS] jamais masqué
    assert labels[1:] == seq[1:]  # tous les mots choisis, en entier
    assert inputs[0] == 1


def test_masking_rate_and_replacement_split() -> None:
    m = masker()
    rng = random.Random(0)
    seq = ids("▁le", "▁colis", "▁client", "▁facture", "▁reçue", "▁hier", "▁arrivé")
    n_masked = n_mask_token = n_kept = 0
    for _ in range(3000):
        inputs, labels = m(seq, rng)
        for x, y, orig in zip(inputs, labels, seq):
            if y != IGNORE:
                n_masked += 1
                n_mask_token += x == 2
                n_kept += x == orig
    assert n_masked == 3000  # round(0.15 × 7) = 1 mot d'un token
    assert abs(n_mask_token / n_masked - 0.8) < 0.03
    assert abs(n_kept / n_masked - 0.1) < 0.03 + 1 / len(PIECES)


def test_negation_and_numbers_are_oversampled() -> None:
    m = masker()  # facteur 3
    rng = random.Random(0)
    # 10 mots dont « pas » et « 120 » ; 2 mots masqués par phrase (round(0.15 × 10) = 2).
    seq = ids("▁le", "▁colis", "▁pas", "▁arrivé", "▁facture", "▁de", "▁12", "0", "▁euros",
              "▁reçue", "▁hier")
    words = group_words(seq, PIECES, SPECIAL)
    counts = [0] * len(words)
    for _ in range(5000):
        _, labels = m(seq, rng)
        for w, positions in enumerate(words):
            counts[w] += labels[positions[0]] != IGNORE
    by_text = {m.word_text(seq, w): c for w, c in zip(words, counts)}
    plain = [c for t, c in by_text.items() if t not in ("pas", "120")]
    mean_plain = sum(plain) / len(plain)
    assert by_text["pas"] > 2.0 * mean_plain
    assert by_text["120"] > 2.0 * mean_plain

    neutral = masker(oversample_factor=1.0)
    counts_neutral = 0
    for _ in range(5000):
        _, labels = neutral(seq, rng)
        counts_neutral += labels[3] != IGNORE  # « pas »
    assert by_text["pas"] > 2.0 * counts_neutral
