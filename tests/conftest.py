from collections.abc import Callable
from pathlib import Path

import pytest

from mini_decision.llm.base import LLMClient, T, Usage

ROOT = Path(__file__).resolve().parents[1]


class FakeLLMClient(LLMClient):
    """Client factice : renvoie une réponse fixe et compte les appels réels."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = 0

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        self.calls += 1
        self.last_usage = Usage(input_tokens=1_000_000, output_tokens=200_000)
        return schema.model_validate(self.response)


class ScriptedLLMClient(LLMClient):
    """Client factice dont la réponse dépend du prompt, du schéma et du tirage."""

    provider = "fake"

    def __init__(self, model: str, answer: Callable[[str, type, int], dict]) -> None:
        self.model = model
        self.answer = answer
        self.calls = 0

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        self.calls += 1
        return schema.model_validate(self.answer(prompt, schema, sample))


@pytest.fixture
def base_config_path() -> Path:
    return ROOT / "configs" / "base.yaml"
