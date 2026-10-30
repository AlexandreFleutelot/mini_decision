from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_decision.config import load_config, parse_override
from tests.conftest import ROOT


def test_base_config_loads(base_config_path: Path) -> None:
    cfg = load_config(base_config_path)
    assert cfg.model.encoder == "transformer"
    assert cfg.model.head == "qda"
    assert cfg.data.kind_ratios == {"true": 0.4, "false": 0.4, "off_topic": 0.2}
    assert len(cfg.data.subdomains) >= 2


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_every_variant_loads(path: Path) -> None:
    load_config(path)


def test_variant_inherits_and_overrides() -> None:
    cfg = load_config(ROOT / "configs" / "transformer_dot.yaml")
    assert cfg.model.head == "dot"
    assert cfg.model.k == 16  # hérité de base.yaml


def test_debug_inherits_two_levels() -> None:
    cfg = load_config(ROOT / "configs" / "debug.yaml")
    assert cfg.model.encoder == "ngram"
    assert cfg.data.n_pairs == 10


def test_cli_overrides(base_config_path: Path) -> None:
    cfg = load_config(base_config_path, ["model.head=dot", "train.lr=0.01"])
    assert cfg.model.head == "dot"
    assert cfg.train.lr == 0.01


def test_parse_override_nested() -> None:
    assert parse_override("a.b.c=3") == {"a": {"b": {"c": 3}}}


def test_invalid_value_rejected(base_config_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(base_config_path, ["model.head=svm"])


def test_unknown_key_rejected(base_config_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(base_config_path, ["model.typo=1"])
