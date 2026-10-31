import pytest
from typer.testing import CliRunner

from mini_decision.cli import app
from tests.conftest import ROOT

COMMANDS = [
    "datagen",
    "verify",
    "split",
    "train-tokenizer",
    "pretrain-mlm",
    "train",
    "calibrate",
    "evaluate",
    "adversarial-round",
    "report",
]
NOT_YET_IMPLEMENTED = COMMANDS[3:]


@pytest.mark.parametrize("command", NOT_YET_IMPLEMENTED)
def test_stub_command_runs(command: str) -> None:
    config = ROOT / "configs" / "debug.yaml"
    result = CliRunner().invoke(app, [command, "--config", str(config), "--set", "seed=1"])
    assert result.exit_code == 0, result.output
    assert "seed=1" in result.output


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    for command in COMMANDS:
        assert command in result.output
