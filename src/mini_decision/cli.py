"""Point d'entrée en ligne de commande : `uv run mini-decision <commande> --config configs/x.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mini_decision.config import Config, load_config
from mini_decision.llm import SpendTracker
from mini_decision.schemas import Example, RawPair, Rejected, UnlabeledText, read_jsonl, write_jsonl
from mini_decision.utils import get_device, seed_everything

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Fichier YAML de configuration.")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Surcharge clé=valeur, ex. model.head=dot.")
]


def _setup(config: Path, overrides: list[str] | None) -> Config:
    cfg = load_config(config, overrides)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    typer.echo(f"[config] {config} · run={cfg.run_name} · seed={cfg.seed} · device={device}")
    return cfg


def _tracker(cfg: Config) -> SpendTracker:
    return SpendTracker(cfg.paths.llm_cache_dir / "spend.json", cfg.llm.budget_usd)


def _spent(tracker: SpendTracker) -> None:
    typer.echo(f"[budget] {tracker.spent_usd:.4f} $ dépensés sur {tracker.budget_usd:.2f} $")


def _todo(step: str, milestone: int) -> None:
    typer.echo(f"[{step}] pas encore implémenté (jalon {milestone}).")


DEFAULT_CONFIG = Path("configs/base.yaml")


@app.command()
def datagen(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    skip_unlabeled: Annotated[bool, typer.Option(help="Ne génère que les paires.")] = False,
) -> None:
    """Génère paires, contextes et corpus non étiqueté via le LLM générateur."""
    from mini_decision.datagen.generate import generate_pairs, generate_unlabeled, plan_kinds
    from mini_decision.llm import make_client

    cfg = _setup(config, set_)
    tracker = _tracker(cfg)
    client = make_client(cfg.llm.generator, cfg.paths.llm_cache_dir, tracker)
    typer.echo(f"[datagen] {cfg.data.n_pairs} paires, contextes par paire : {plan_kinds(cfg.data)}")
    pairs = generate_pairs(cfg.data, client, cfg.llm.max_concurrency, typer.echo)
    n = write_jsonl(cfg.paths.raw_dir / "pairs.jsonl", pairs)
    n_contexts = sum(len(p.contexts) for p in pairs)
    typer.echo(f"[datagen] {n} paires, {n_contexts} contextes -> {cfg.paths.raw_dir}/pairs.jsonl")
    if not skip_unlabeled:
        n_texts = round(cfg.data.unlabeled_ratio * n_contexts)
        texts = generate_unlabeled(cfg.data, client, n_texts, cfg.llm.max_concurrency, typer.echo)
        write_jsonl(cfg.paths.raw_dir / "unlabeled.jsonl", (UnlabeledText(text=t) for t in texts))
        typer.echo(f"[datagen] {len(texts)} textes -> {cfg.paths.raw_dir}/unlabeled.jsonl")
    typer.echo(f"[cache] {client.hits} réutilisés, {client.misses} nouveaux appels")
    _spent(tracker)


@app.command()
def verify(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Fait juger chaque exemple par les LLM juges ; rejette les désaccords."""
    import json
    from collections import Counter

    from mini_decision.datagen.verify import candidates, judge_all
    from mini_decision.llm import make_client

    cfg = _setup(config, set_)
    tracker = _tracker(cfg)
    judges = [make_client(j, cfg.paths.llm_cache_dir, tracker) for j in cfg.llm.judges]
    pairs = read_jsonl(cfg.paths.raw_dir / "pairs.jsonl", RawPair)
    cands = candidates(pairs)
    typer.echo(f"[verify] {len(cands)} contextes, juges : {[j.model for j in judges]}")
    kept, rejected, spreads = judge_all(
        cands, judges, cfg.data, cfg.llm.max_concurrency, typer.echo
    )
    out = cfg.paths.processed_dir
    write_jsonl(out / "labeled.jsonl", kept)
    write_jsonl(out / "rejected.jsonl", rejected)
    (out / "judge_spreads.json").write_text(json.dumps(spreads), encoding="utf-8")
    reasons = dict(Counter(r.reason for r in rejected))
    typer.echo(f"[verify] gardés {len(kept)}, rejetés {len(rejected)} {reasons}")
    _spent(tracker)


@app.command()
def split(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Split train/val/test par pair_id + jeu de contrefactuels + contrôle de fuite."""
    import json

    from mini_decision.datagen.split import (
        build_report,
        counterfactual_set,
        leakage_report,
        split_by_pair,
    )

    cfg = _setup(config, set_)
    out = cfg.paths.processed_dir
    examples = read_jsonl(out / "labeled.jsonl", Example)
    splits = split_by_pair(examples, cfg.data.split, cfg.seed)
    for name, items in splits.items():
        write_jsonl(out / f"{name}.jsonl", items)
    write_jsonl(out / "test_counterfactual.jsonl", counterfactual_set(splits["test"]))
    leakage = leakage_report(splits, cfg)
    typer.echo("[split] " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    typer.echo(
        f"[fuite] descriptions seules {leakage['descs_only']:.1%}, "
        f"contexte seul {leakage['context_only']:.1%}"
    )
    if leakage["descs_only"] > cfg.data.leakage_max_accuracy:
        typer.secho("[fuite] FUITE DÉTECTÉE : corriger la génération.", fg="red")
    pairs = read_jsonl(cfg.paths.raw_dir / "pairs.jsonl", RawPair)
    report = build_report(
        splits,
        read_jsonl(out / "rejected.jsonl", Rejected),
        json.loads((out / "judge_spreads.json").read_text(encoding="utf-8")),
        leakage,
        {p.pair_id: p.subdomain for p in pairs},
        cfg,
    )
    (out / "stats.md").write_text(report, encoding="utf-8")
    typer.echo(f"[split] rapport -> {out}/stats.md")


@app.command("train-tokenizer")
def train_tokenizer(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Entraîne le tokenizer SentencePiece sur tout le corpus synthétique."""
    _setup(config, set_)
    _todo("train-tokenizer", 3)


@app.command("pretrain-mlm")
def pretrain_mlm(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Pré-entraîne l'encodeur en MLM (masquage par mots entiers)."""
    _setup(config, set_)
    _todo("pretrain-mlm", 4)


@app.command()
def train(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Entraînement supervisé (BCE sur labels souples)."""
    _setup(config, set_)
    _todo("train", 4)


@app.command()
def calibrate(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Temperature scaling sur le jeu de calibration (réel si présent, sinon val synthétique)."""
    _setup(config, set_)
    _todo("calibrate", 5)


@app.command()
def evaluate(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Métriques sur le test et les contrefactuels, diagramme de fiabilité."""
    _setup(config, set_)
    _todo("evaluate", 5)


@app.command("adversarial-round")
def adversarial_round(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Une ronde model-in-the-loop : erreurs confiantes -> nouveaux contextes -> juges."""
    _setup(config, set_)
    _todo("adversarial-round", 6)


@app.command()
def report(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Rapport Markdown comparatif de tous les runs."""
    _setup(config, set_)
    _todo("report", 5)


if __name__ == "__main__":
    app()
