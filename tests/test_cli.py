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
    "params",
    "scaling",
    "probe",
]
NOT_YET_IMPLEMENTED = ["calibrate", "evaluate", "adversarial-round", "report"]


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


def test_train_tokenizer_and_params(tmp_path) -> None:  # noqa: ANN001
    from mini_decision.schemas import UnlabeledText, write_jsonl

    raw = tmp_path / "raw"
    texts = [f"Le colis {i} n'est pas arrivé, facture de {i * 7} euros payée." for i in range(300)]
    write_jsonl(raw / "unlabeled.jsonl", [UnlabeledText(text=t) for t in texts])
    sets = ["--set", f"paths.raw_dir={raw}", "--set", f"paths.tokenizer_prefix={tmp_path}/tok/spm",
            "--set", "tokenizer.vocab_size=120"]
    config = str(ROOT / "configs" / "debug.yaml")
    result = CliRunner().invoke(app, ["train-tokenizer", "--config", config, *sets])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tok" / "spm.model").exists()
    result = CliRunner().invoke(app, ["params", "--config", config, *sets])
    assert result.exit_code == 0, result.output
    assert "transformer" in result.output and "<- config" in result.output
