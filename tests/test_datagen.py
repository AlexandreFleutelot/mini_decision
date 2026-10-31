from pathlib import Path

import pytest
from typer.testing import CliRunner

from mini_decision.cli import app
from mini_decision.config import load_config
from mini_decision.datagen.generate import clean_pair, plan_kinds
from mini_decision.datagen.split import split_by_pair
from mini_decision.datagen.verify import aggregate
from mini_decision.llm import CachedLLMClient
from mini_decision.schemas import (
    Example,
    GeneratedPair,
    Judgment,
    Kind,
    RawPair,
    UnlabeledBatch,
    read_jsonl,
)
from tests.conftest import ROOT, ScriptedLLMClient

CFG = load_config(ROOT / "configs" / "base.yaml")


def test_plan_kinds_matches_ratios() -> None:
    plan = plan_kinds(CFG.data)
    assert plan.total == CFG.data.contexts_per_pair == 15
    assert (plan.n_true, plan.n_false + plan.n_counterfactual, plan.n_off_topic) == (6, 6, 3)
    assert plan.n_counterfactual == 3


def test_clean_pair_drops_orphan_counterfactuals() -> None:
    pair = GeneratedPair.model_validate(
        {
            "desc_true": "a",
            "desc_false": "b",
            "contexts": [
                {"text": "vrai", "kind": "true"},
                {"text": "cf ok", "kind": "counterfactual", "source_index": 0},
                {"text": "cf orphelin", "kind": "counterfactual", "source_index": 7},
                {"text": "  ", "kind": "false"},
            ],
        }
    )
    assert [c.text for c in clean_pair(pair).contexts] == ["vrai", "cf ok"]


def _j(p: float, off: bool = False) -> Judgment:
    return Judgment(off_topic=off, p_true=p)


@pytest.mark.parametrize(
    ("kind", "judgments", "expected"),
    [
        (Kind.TRUE, [_j(0.9), _j(0.7)], (0.8, "ok")),
        (Kind.TRUE, [_j(0.9), _j(0.4)], (None, "désaccord")),
        (Kind.FALSE, [_j(0.9), _j(0.8)], (None, "contredit_le_kind")),
        (Kind.COUNTERFACTUAL, [_j(0.1), _j(0.2)], (0.15, "ok")),
        (Kind.TRUE, [_j(0.9), _j(0.5, off=True)], (None, "jugé_hors_sujet")),
        (Kind.OFF_TOPIC, [_j(0.5, True), _j(0.5, True)], (0.5, "ok")),
        (Kind.OFF_TOPIC, [_j(0.5, True), _j(0.9)], (None, "hors_sujet_contesté")),
    ],
)
def test_aggregate(kind: Kind, judgments: list[Judgment], expected: tuple) -> None:
    label, reason = aggregate(kind, judgments, CFG.data)
    assert reason == expected[1]
    assert label == pytest.approx(expected[0]) if expected[0] is not None else label is None


def _example(pair_id: str, i: int) -> Example:
    return Example(
        pair_id=pair_id, context=f"c{i}", desc_true="v", desc_false="f", label=0.9, kind="true"
    )


def test_split_never_shares_pair_ids() -> None:
    examples = [_example(f"p_{p:03d}", i) for p in range(100) for i in range(5)]
    splits = split_by_pair(examples, CFG.data.split, seed=0)
    ids = {name: {e.pair_id for e in items} for name, items in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert [len(ids[n]) for n in ("train", "val", "test")] == [80, 10, 10]
    assert sum(len(v) for v in splits.values()) == len(examples)


# --- Pipeline complet datagen -> verify -> split avec des LLM factices ----------------------


def _generator(prompt: str, schema: type, sample: int) -> dict:
    if schema is UnlabeledBatch:
        return {"texts": [f"Note interne {sample}-{k} : rien à signaler." for k in range(40)]}
    assert schema is GeneratedPair
    s = sample
    return {
        "desc_true": f"La facture F{s} est payée.",
        "desc_false": f"La facture F{s} reste impayée.",
        "contexts": [
            *[{"text": f"Le client {s} a réglé la facture F{s} ({k}).", "kind": "true"}
              for k in range(6)],
            *[{"text": f"Le client {s} n'a pas réglé F{s} ({k}).", "kind": "false"}
              for k in range(3)],
            *[{"text": f"Le client {s} n'a toujours pas réglé F{s} ({k}).",
               "kind": "counterfactual", "source_index": k} for k in range(3)],
            *[{"text": f"Réunion d'équipe lundi ({s}-{k}).", "kind": "off_topic"}
              for k in range(3)],
        ],
    }


def _judge(strict: bool):
    def answer(prompt: str, schema: type, sample: int) -> dict:
        if "Réunion" in prompt:
            return {"off_topic": True, "p_true": 0.5}
        if "n'a" in prompt:
            return {"off_topic": False, "p_true": 0.1 if strict else 0.2}
        # Le juge « laxiste » diverge sur le contexte (0) : il sera rejeté pour désaccord.
        return {"off_topic": False, "p_true": 0.4 if not strict and "(0)" in prompt else 0.9}

    return answer


def test_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"gen": _generator, "gen2": _generator, "j1": _judge(True), "j2": _judge(False)}

    def fake_make_client(cfg, cache_dir, tracker=None):  # noqa: ANN001, ANN202
        return CachedLLMClient(ScriptedLLMClient(cfg.model, answers[cfg.model]), cache_dir, tracker)

    monkeypatch.setattr("mini_decision.llm.make_client", fake_make_client)
    overrides = [
        f"paths.raw_dir={tmp_path / 'raw'}",
        f"paths.processed_dir={tmp_path / 'processed'}",
        f"paths.llm_cache_dir={tmp_path / 'cache'}",
        "llm.generator={provider: anthropic, model: gen}",
        "llm.judges=[{provider: anthropic, model: j1}, {provider: ollama, model: j2}]",
    ]
    args = ["--config", str(ROOT / "configs" / "debug.yaml")]
    for o in overrides:
        args += ["--set", o]
    runner = CliRunner()
    for command in ("datagen", "verify", "split"):
        result = runner.invoke(app, [command, *args])
        assert result.exit_code == 0, result.output

    pairs = read_jsonl(tmp_path / "raw" / "pairs.jsonl", RawPair)
    assert len(pairs) == 10
    labeled = read_jsonl(tmp_path / "processed" / "labeled.jsonl", Example)
    # 15 contextes par paire, dont 1 rejeté pour désaccord.
    assert len(labeled) == 10 * 14
    assert {e.label for e in labeled if e.kind is Kind.OFF_TOPIC} == {0.5}
    assert {e.label for e in labeled if e.kind is Kind.TRUE} == {0.9}
    stats = (tmp_path / "processed" / "stats.md").read_text(encoding="utf-8")
    assert "désaccord | 10" in stats
    assert "### counterfactual" in stats

    # Relancer datagen ne refait aucun appel : tout vient du cache.
    result = runner.invoke(app, ["datagen", *args])
    assert "0 nouveaux appels" in result.output


    # Un second générateur ajoute un bloc à la suite ; le bloc principal reste en cache.
    args2 = [*args, "--set", "llm.extra_generators=[{provider: openrouter, model: gen2, n_pairs: 4}]"]
    result = runner.invoke(app, ["datagen", "--skip-unlabeled", *args2])
    assert result.exit_code == 0, result.output
    assert "4 nouveaux appels" in result.output
    pairs = read_jsonl(tmp_path / "raw" / "pairs.jsonl", RawPair)
    assert [p.pair_id for p in pairs][-4:] == ["p_00010", "p_00011", "p_00012", "p_00013"]
    assert {p.generator for p in pairs} == {"gen", "gen2"}

    # probe : une paire générée puis un jugement, par modèle.
    answers["probe"] = lambda prompt, schema, sample: (
        _generator(prompt, schema, 0) if schema is GeneratedPair else {"off_topic": False, "p_true": 0.9}
    )
    result = runner.invoke(app, ["probe", "-m", "openrouter:probe", *args])
    assert result.exit_code == 0, result.output
    assert "15 contextes" in result.output and "p_vrai 0.90" in result.output
