import torch

from mini_decision.utils import get_device


def test_explicit_device() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_auto_device_is_available() -> None:
    assert get_device("auto").type in {"cuda", "mps", "cpu"}


def test_progress_logs_rate_eta_and_extra() -> None:
    from mini_decision.utils import Progress

    lines: list[str] = []
    progress = Progress("verify", 4, lines.append, every_s=0.0, extra=lambda: "0.10 $")
    progress.update(2)
    progress.update(2)
    assert len(lines) == 2
    assert lines[0].startswith("[verify] 2/4 (50%)")
    assert "reste ~" in lines[0] and lines[0].endswith("0.10 $")
    assert lines[1].startswith("[verify] 4/4 (100%)")
