"""Bout en bout sur un mini jeu synthétique : tokenizer -> MLM -> supervisé -> courbe d'échelle."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mini_decision.cli import app
from mini_decision.eval.metrics import all_metrics, ece
from mini_decision.schemas import Example, Kind, UnlabeledText, write_jsonl
from mini_decision.train.data import subset_by_pairs
from mini_decision.train.schedule import warmup_cosine
from tests.conftest import ROOT

ITEMS = ["colis", "facture", "contrat", "ticket", "congé", "abonnement", "remboursement", "devis"]


def make_examples(n_pairs: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for p in range(n_pairs):
        item = ITEMS[p % len(ITEMS)]
        d_true, d_false = f"le {item} {p} est validé", f"le {item} {p} est refusé"
        for _ in range(3):
            out.append(Example(pair_id=f"p_{p:05d}", context=f"Bonne nouvelle : le {item} {p} est bien validé.",
                               desc_true=d_true, desc_false=d_false, label=round(rng.uniform(0.85, 1.0), 2), kind=Kind.TRUE))
            out.append(Example(pair_id=f"p_{p:05d}", context=f"Hélas le {item} {p} n'est pas validé, il est refusé.",
                               desc_true=d_true, desc_false=d_false, label=round(rng.uniform(0.0, 0.15), 2), kind=Kind.FALSE))
        out.append(Example(pair_id=f"p_{p:05d}", context="La réunion d'équipe est déplacée à jeudi.",
                           desc_true=d_true, desc_false=d_false, label=0.5, kind=Kind.OFF_TOPIC))
    return out


def test_subset_by_pairs_is_nested_and_whole_pairs() -> None:
    examples = make_examples(20)
    small, big = subset_by_pairs(examples, 30, seed=1), subset_by_pairs(examples, 80, seed=1)
    assert {e.pair_id for e in small} <= {e.pair_id for e in big}
    assert len(small) % 7 == 0 and len(small) >= 30
    assert subset_by_pairs(examples, None, 1) == examples


def test_warmup_cosine() -> None:
    assert warmup_cosine(0, 100, 0.05) == pytest.approx(0.2)
    assert warmup_cosine(4, 100, 0.05) == pytest.approx(1.0)
    assert warmup_cosine(100, 100, 0.05) == pytest.approx(0.0)
    assert 0.45 < warmup_cosine(52, 100, 0.05) < 0.55


def test_metrics_small_example() -> None:
    p, y = np.array([0.9, 0.2, 0.5]), np.array([1.0, 0.1, 0.5])
    m = all_metrics(p, y, ["true", "false", "off_topic"])
    assert m["accuracy"] == 1.0 and m["off_topic_auroc"] == 1.0
    assert ece(np.array([0.8] * 10), np.array([1.0] * 8 + [0.0] * 2)) == pytest.approx(0.0)


@pytest.mark.parametrize("encoder", ["ngram", "transformer"])
def test_pipeline_end_to_end(tmp_path: Path, encoder: str) -> None:
    raw, processed, runs = tmp_path / "raw", tmp_path / "processed", tmp_path / "runs"
    write_jsonl(processed / "train.jsonl", make_examples(24, seed=0))
    write_jsonl(processed / "val.jsonl", make_examples(6, seed=1))
    write_jsonl(processed / "labeled.jsonl", make_examples(30))
    write_jsonl(raw / "unlabeled.jsonl", [UnlabeledText(text=f"Le dossier {i} n'est pas encore traité.") for i in range(50)])
    sets = []
    for kv in [f"paths.raw_dir={raw}", f"paths.processed_dir={processed}", f"paths.runs_dir={runs}",
               f"paths.tokenizer_prefix={processed}/tok/spm", "tokenizer.vocab_size=150",
               f"model.encoder={encoder}", "model.transformer.n_layers=1", "model.transformer.d_model=32",
               "model.transformer.d_ff=64", "model.ngram.d_model=32", "model.ngram.n_buckets=256",
               "mlm.max_epochs=2", "mlm.batch_size=32", "train.max_epochs=4", "train.batch_size=32",
               "train.aux_mlm_lambda=0.1", "train.scaling_sizes=[50, 100, 100000]", "device=cpu"]:
        sets += ["--set", kv]
    config = str(ROOT / "configs" / "debug.yaml")
    runner = CliRunner()
    for command in ["train-tokenizer", "pretrain-mlm", "train", "scaling"]:
        result = runner.invoke(app, [command, "--config", config, *sets])
        assert result.exit_code == 0, f"{command}\n{result.output}"
    assert (runs / f"mlm-{encoder}" / "encoder.pt").exists()
    summary = json.loads((runs / "debug" / "summary.json").read_text())
    assert summary["init"].endswith("encoder.pt")
    assert summary["n_train"] == 24 * 7
    assert 0 <= summary["val"]["log_loss"] < 5
    assert (runs / "debug-scaling" / "scaling.png").exists()
    assert "le dernier point utilise tout" in result.output  # 100000 > disponible
    assert (runs / f"debug-n{24 * 7}" / "summary.json").exists()
