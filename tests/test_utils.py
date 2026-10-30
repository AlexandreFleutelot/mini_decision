import torch

from mini_decision.utils import get_device


def test_explicit_device() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_auto_device_is_available() -> None:
    assert get_device("auto").type in {"cuda", "mps", "cpu"}
