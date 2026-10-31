"""Tokenizer SentencePiece (unigram) partagé par les contextes et les descriptions.

Ids fixes : [PAD]=0, <unk>=1, puis les tokens spéciaux restants dans l'ordre de la config
([CLS]=2, [MASK]=3, [CTX]=4, [DESC]=5). Pas de BOS/EOS : chaque séquence commence par [CLS].

Petit exemple : encode("Facture payée") -> [2, 812, 57, 1033] (ids illustratifs).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm
import torch

from mini_decision.config import Config, TokenizerConfig
from mini_decision.schemas import Example, RawPair, UnlabeledText, read_jsonl

PAD, UNK = "[PAD]", "<unk>"


def corpus_texts(cfg: Config) -> list[str]:
    """Toutes les phrases disponibles : contextes et descriptions des paires brutes + non étiqueté.

    On prend les paires brutes (et pas seulement train) : le tokenizer ne voit aucun label,
    il n'y a donc pas de fuite au sens supervisé.
    """
    texts: list[str] = []
    pairs_path = cfg.paths.raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        for p in read_jsonl(pairs_path, RawPair):
            texts += [p.desc_true, p.desc_false] + [c.text for c in p.contexts]
    elif (labeled := cfg.paths.processed_dir / "labeled.jsonl").exists():  # repli
        for e in read_jsonl(labeled, Example):
            texts += [e.context, e.desc_true, e.desc_false]
    unlabeled = cfg.paths.raw_dir / "unlabeled.jsonl"
    if unlabeled.exists():
        texts += [u.text for u in read_jsonl(unlabeled, UnlabeledText)]
    if not texts:
        raise FileNotFoundError(f"aucun texte dans {cfg.paths.raw_dir} : lancer datagen d'abord")
    return [t.strip() for t in texts if t.strip()]


def train_tokenizer(texts: Iterable[str], cfg: TokenizerConfig, prefix: Path) -> Tokenizer:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = prefix.with_suffix(".corpus.txt")
    corpus.write_text("\n".join(t.replace("\n", " ") for t in texts), encoding="utf-8")
    pad, *others = cfg.special_tokens  # le premier spécial est le padding
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type=cfg.model_type,
        vocab_size=cfg.vocab_size,
        character_coverage=1.0,  # garde tous les accents
        pad_id=0,
        pad_piece=pad,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=others,
        hard_vocab_limit=False,  # petit corpus de test : vocab_size est un plafond
        num_threads=4,
        minloglevel=2,
    )
    return Tokenizer(prefix.with_suffix(".model"))


class Tokenizer:
    def __init__(self, model_path: Path) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.sp.piece_to_id("[PAD]")
        self.cls_id = self.sp.piece_to_id("[CLS]")
        self.mask_id = self.sp.piece_to_id("[MASK]")
        self.unk_id = self.sp.unk_id()
        self.special_ids = {self.pad_id, self.cls_id, self.mask_id, self.unk_id} | {
            self.sp.piece_to_id(t) for t in ("[CTX]", "[DESC]")
        }

    @classmethod
    def from_prefix(cls, prefix: Path) -> Tokenizer:
        return cls(prefix.with_suffix(".model"))

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, max_len: int) -> list[int]:
        return [self.cls_id] + self.sp.encode(text)[: max_len - 1]

    def pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(i)

    def batch(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Renvoie (ids [B, L], mask [B, L] booléen, True = vrai token)."""
        seqs = [self.encode(t, max_len) for t in texts]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids, ids != self.pad_id
