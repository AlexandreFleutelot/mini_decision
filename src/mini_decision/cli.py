"""Point d'entrée en ligne de commande : `uv run mini-decision <commande> --config configs/x.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mini_decision.config import Config, load_config
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


def _todo(step: str, milestone: int) -> None:
    typer.echo(f"[{step}] pas encore implémenté (jalon {milestone}).")


DEFAULT_CONFIG = Path("configs/base.yaml")


@app.command()
def datagen(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Génère paires, contextes et corpus non étiqueté via le LLM générateur."""
    _setup(config, set_)
    _todo("datagen", 2)


@app.command()
def verify(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Fait juger chaque exemple par les LLM juges ; rejette les désaccords."""
    _setup(config, set_)
    _todo("verify", 2)


@app.command()
def split(config: ConfigOpt = DEFAULT_CONFIG, set_: SetOpt = None) -> None:
    """Split train/val/test par pair_id + jeu de contrefactuels + contrôle de fuite."""
    _setup(config, set_)
    _todo("split", 2)


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
