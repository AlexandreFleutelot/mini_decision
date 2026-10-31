from pathlib import Path

from mini_decision.config import TokenizerConfig
from mini_decision.tokenizer import Tokenizer, train_tokenizer

TEXTS = [
    "La facture n° 1042 de 120 euros a été payée le 3 mars.",
    "Le colis n'est pas arrivé, le client demande un remboursement.",
    "Le ticket est fermé : aucune réponse du support depuis deux jours.",
    "Congés acceptés du 12 au 19 août, sauf avis contraire de la paie.",
    "L'abonnement est résilié sans frais, ni pénalité.",
] * 40


def test_train_and_encode(tmp_path: Path) -> None:
    cfg = TokenizerConfig(
        model_type="unigram",
        vocab_size=200,
        special_tokens=["[PAD]", "[CLS]", "[MASK]", "[CTX]", "[DESC]"],
    )
    tok = train_tokenizer(TEXTS, cfg, tmp_path / "spm")
    assert (tok.pad_id, tok.unk_id, tok.cls_id, tok.mask_id) == (0, 1, 2, 3)
    assert Tokenizer.from_prefix(tmp_path / "spm").vocab_size == tok.vocab_size <= 200

    ids, mask = tok.batch(["Le colis n'est pas arrivé, le client demande.", "Le colis"], max_len=8)
    assert ids.shape[0] == 2 and ids.shape[1] <= 8
    assert (ids[:, 0] == tok.cls_id).all()
    assert mask[0].all() and not mask[1].all()  # la 2e phrase est plus courte : padding
    assert (ids[~mask] == tok.pad_id).all()
    assert "é" in "".join(tok.pieces("payée"))  # les accents sont couverts
