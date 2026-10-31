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


def _tokenizer(cfg: Config):  # noqa: ANN202
    from mini_decision.tokenizer import Tokenizer

    path = cfg.paths.tokenizer_prefix.with_suffix(".model")
    if not path.exists():
        typer.secho(f"{path} absent : lancer train-tokenizer d'abord", fg="red")
        raise typer.Exit(1)
    return Tokenizer(path)


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
    extra = [
        (make_client(g, cfg.paths.llm_cache_dir, tracker), g.n_pairs or 0) for g in cfg.llm.extra_generators
    ]
    blocks = ", ".join(f"{c.model} {n}" for c, n in [(client, cfg.data.n_pairs), *extra])
    typer.echo(f"[datagen] paires : {blocks} · contextes par paire : {plan_kinds(cfg.data)}")
    pairs = generate_pairs(cfg.data, client, cfg.llm.max_concurrency, typer.echo, extra=extra)
    n = write_jsonl(cfg.paths.raw_dir / "pairs.jsonl", pairs)
    n_contexts = sum(len(p.contexts) for p in pairs)
    typer.echo(f"[datagen] {n} paires, {n_contexts} contextes -> {cfg.paths.raw_dir}/pairs.jsonl")
    if not skip_unlabeled:
        # Le non étiqueté suit le bloc principal seulement : ajouter des générateurs ne le
        # régénère pas (61k textes suffisent largement pour le MLM).
        main = sum(len(p.contexts) for p in pairs if int(p.pair_id[2:]) < cfg.data.n_pairs)
        n_texts = round(cfg.data.unlabeled_ratio * main)
        texts = generate_unlabeled(cfg.data, client, n_texts, cfg.llm.max_concurrency, typer.echo)
        write_jsonl(cfg.paths.raw_dir / "unlabeled.jsonl", (UnlabeledText(text=t) for t in texts))
        typer.echo(f"[datagen] {len(texts)} textes -> {cfg.paths.raw_dir}/unlabeled.jsonl")
    gens = [client, *(c for c, _ in extra)]
    typer.echo(f"[cache] {sum(c.hits for c in gens)} réutilisés, {sum(c.misses for c in gens)} nouveaux appels")
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


@app.command()
def probe(
    model: Annotated[
        list[str], typer.Option("--model", "-m", help="fournisseur:modèle, ex. openrouter:z-ai/glm-5.3-flash")
    ],
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    max_tokens: Annotated[int, typer.Option(help="max_tokens de la génération.")] = 16000,
) -> None:
    """Teste des modèles : une paire générée puis jugée par le modèle lui-même (durée, coût, validité)."""
    import time

    from mini_decision.config import LLMModelConfig
    from mini_decision.datagen.generate import _pair_prompt, clean_pair, plan_kinds
    from mini_decision.llm import LLMError, make_client
    from mini_decision.llm import prompts
    from mini_decision.schemas import GeneratedPair, Judgment

    cfg = _setup(config, set_)
    tracker = _tracker(cfg)
    plan = plan_kinds(cfg.data)
    system = prompts.SYSTEM_GENERATOR.format(language=cfg.data.language)
    for spec in model:
        provider, _, name = spec.partition(":")
        client = make_client(
            LLMModelConfig(provider=provider, model=name, max_tokens=max_tokens), cfg.paths.llm_cache_dir, tracker
        )
        before, t0 = tracker.spent_usd, time.monotonic()
        try:
            pair = clean_pair(client.complete_json(
                _pair_prompt(cfg.data, plan, cfg.data.subdomains[0], []), GeneratedPair, system=system, sample=-1
            ))
            t_gen = time.monotonic() - t0
            kinds = {k.value: sum(c.kind is k for c in pair.contexts) for k in {c.kind for c in pair.contexts}}
            ctx = pair.contexts[0]
            t1 = time.monotonic()
            judgment = client.complete_json(
                prompts.JUDGE.format(context=ctx.text, desc_true=pair.desc_true, desc_false=pair.desc_false),
                Judgment, system=prompts.SYSTEM_JUDGE, sample=-1,
            )
            t_judge = time.monotonic() - t1
        except LLMError as err:
            typer.secho(f"[probe] {spec} : ÉCHEC {err}", fg="red")
            continue
        mode = getattr(client.inner, "mode", "schema")
        typer.echo(
            f"[probe] {spec} · génération {t_gen:.1f} s, {len(pair.contexts)} contextes {kinds} · "
            f"jugement {t_judge:.1f} s (kind {ctx.kind.value} -> p_vrai {judgment.p_true:.2f}, "
            f"hors sujet {judgment.off_topic}) · mode {mode} · coût {tracker.spent_usd - before:.4f} $"
        )
        typer.echo(f"         VRAI : {pair.desc_true}\n         FAUX : {pair.desc_false}\n         ex. : {ctx.text}")
    _spent(tracker)


@app.command("train-tokenizer")
def train_tokenizer(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Entraîne le tokenizer SentencePiece sur tout le corpus synthétique."""
    from mini_decision.tokenizer import corpus_texts
    from mini_decision.tokenizer import train_tokenizer as train_spm

    cfg = _setup(config, set_)
    texts = corpus_texts(cfg)
    n_words = sum(len(t.split()) for t in texts)
    typer.echo(f"[tokenizer] {len(texts)} textes, {n_words} mots, vocab cible {cfg.tokenizer.vocab_size}")
    tok = train_spm(texts, cfg.tokenizer, cfg.paths.tokenizer_prefix)
    lengths = sorted(len(tok.encode(t, 10_000)) for t in texts)
    over = sum(n > cfg.model.max_len for n in lengths)
    typer.echo(
        f"[tokenizer] vocab {tok.vocab_size} -> {cfg.paths.tokenizer_prefix}.model · "
        f"tokens/texte médiane {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
        f"{over} textes tronqués à {cfg.model.max_len}"
    )
    example = "Le colis n'est pas arrivé, aucun remboursement avant le 12 mars."
    typer.echo(f"[tokenizer] exemple : {' '.join(tok.pieces(example))}")


@app.command()
def params(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Affiche le nombre de paramètres de chaque encodeur × tête (et de la config choisie)."""
    from mini_decision.config import load_config
    from mini_decision.model import build_model, parameter_table
    from mini_decision.tokenizer import Tokenizer

    cfg = _setup(config, set_)
    model_path = cfg.paths.tokenizer_prefix.with_suffix(".model")
    vocab = Tokenizer(model_path).vocab_size if model_path.exists() else cfg.tokenizer.vocab_size
    typer.echo(f"[params] vocab {vocab}")
    for encoder in ("ngram", "cnn", "transformer"):
        for head, cov in (("dot", "diag"), ("qda", "diag"), ("qda", "full")):
            sets = [*(set_ or []), f"model.encoder={encoder}", f"model.head={head}", f"model.cov={cov}"]
            table = parameter_table(build_model(vocab, load_config(config, sets).model))
            mark = " <- config" if (encoder, head, cov) == (cfg.model.encoder, cfg.model.head, cfg.model.cov) else ""
            detail = ", ".join(f"{k} {v:,}" for k, v in table.items() if k != "total")
            typer.echo(f"  {encoder:<11} {head}/{cov:<4} {table['total']:>10,}  ({detail}){mark}")


@app.command("pretrain-mlm")
def pretrain_mlm(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Pré-entraîne l'encodeur en MLM (masquage par mots entiers)."""
    from mini_decision.train.data import mlm_texts
    from mini_decision.train.mlm import pretrain_mlm as run_mlm

    cfg = _setup(config, set_)
    tok = _tokenizer(cfg)
    out = cfg.paths.processed_dir
    train = read_jsonl(out / "train.jsonl", Example)
    val = read_jsonl(out / "val.jsonl", Example)
    unlabeled_path = cfg.paths.raw_dir / "unlabeled.jsonl"
    unlabeled = [u.text for u in read_jsonl(unlabeled_path, UnlabeledText)] if unlabeled_path.exists() else []
    # Le non étiqueté sert à l'entraînement ; la validation MLM = contextes et descriptions de val.
    run_mlm(cfg, tok, mlm_texts(train, unlabeled), mlm_texts(val, []), get_device(cfg.device), typer.echo)


@app.command()
def train(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Entraînement supervisé (BCE sur labels souples)."""
    from mini_decision.train.supervised import run_dir, train_supervised

    cfg = _setup(config, set_)
    tok = _tokenizer(cfg)
    out = cfg.paths.processed_dir
    train_supervised(
        cfg,
        tok,
        read_jsonl(out / "train.jsonl", Example),
        read_jsonl(out / "val.jsonl", Example),
        get_device(cfg.device),
        run_dir(cfg),
        typer.echo,
    )


@app.command()
def scaling(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Courbe d'échelle : un entraînement par taille de train.scaling_sizes (paires emboîtées)."""
    import csv

    from mini_decision.eval.plots import scaling_plot
    from mini_decision.train.supervised import run_dir, train_supervised

    cfg = _setup(config, set_)
    tok = _tokenizer(cfg)
    out = cfg.paths.processed_dir
    train = read_jsonl(out / "train.jsonl", Example)
    val = read_jsonl(out / "val.jsonl", Example)
    rows = []
    sizes = sorted({min(size, len(train)) for size in cfg.train.scaling_sizes})
    if max(cfg.train.scaling_sizes) > len(train):
        typer.secho(f"[scaling] seulement {len(train)} exemples de train : le dernier point utilise tout", fg="yellow")
    for size in sizes:
        sized = cfg.model_copy(update={"train": cfg.train.model_copy(update={"max_train_examples": size})})
        summary = train_supervised(sized, tok, train, val, get_device(cfg.device), run_dir(cfg, f"-n{size}"), typer.echo)
        rows.append({"n_train": summary["n_train"], **summary["val"]})
    if not rows:
        typer.secho("[scaling] aucune taille réalisable : générer plus de données", fg="red")
        raise typer.Exit(1)
    dest = run_dir(cfg, "-scaling")
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scaling_plot(rows, dest / "scaling.png", title=cfg.run_name)
    for r in rows:
        typer.echo(
            f"[scaling] n={r['n_train']:>6} · log-loss {r['log_loss']:.4f} · acc {r['accuracy']:.3f} · "
            f"ECE {r['ece']:.4f} · AUROC hors sujet {r['off_topic_auroc']:.3f}"
        )
    typer.echo(f"[scaling] -> {dest}/scaling.csv, scaling.png")


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
