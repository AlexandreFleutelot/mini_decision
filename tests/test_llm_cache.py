from pathlib import Path

import pytest

from mini_decision.config import LLMModelConfig
from mini_decision.llm import BudgetExceeded, CachedLLMClient, SpendTracker, make_client
from mini_decision.schemas import Judgment
from tests.conftest import FakeLLMClient

RESPONSE = {"off_topic": False, "p_true": 0.9, "rationale": "payée"}


def test_second_call_hits_cache(tmp_path: Path) -> None:
    fake = FakeLLMClient(RESPONSE)
    client = CachedLLMClient(fake, tmp_path)
    first = client.complete_json("prompt", Judgment)
    second = client.complete_json("prompt", Judgment)
    assert first == second == Judgment.model_validate(RESPONSE)
    assert fake.calls == 1
    assert (client.hits, client.misses) == (1, 1)


def test_cache_survives_new_client(tmp_path: Path) -> None:
    CachedLLMClient(FakeLLMClient(RESPONSE), tmp_path).complete_json("p", Judgment)
    fake = FakeLLMClient({"off_topic": True, "p_true": 0.5})
    assert CachedLLMClient(fake, tmp_path).complete_json("p", Judgment).p_true == 0.9
    assert fake.calls == 0


def test_key_depends_on_prompt_system_and_sample(tmp_path: Path) -> None:
    fake = FakeLLMClient(RESPONSE)
    client = CachedLLMClient(fake, tmp_path)
    client.complete_json("p", Judgment)
    client.complete_json("p2", Judgment)
    client.complete_json("p", Judgment, system="s")
    client.complete_json("p", Judgment, sample=1)
    assert fake.calls == 4


def test_make_client_builds_every_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for provider, model in [
        ("anthropic", "claude-haiku-5-5"),
        ("openai", "gpt-5"),
        ("ollama", "gemma4"),
    ]:
        client = make_client(LLMModelConfig(provider=provider, model=model), tmp_path)
        assert isinstance(client, CachedLLMClient)
        assert (client.provider, client.model) == (provider, model)


def _paid_client(tmp_path: Path, budget: float) -> tuple[CachedLLMClient, FakeLLMClient]:
    # Chaque appel factice : 1M tokens en entrée à 0.10 $ + 0.2M en sortie à 0.50 $ = 0.20 $.
    fake = FakeLLMClient(RESPONSE)
    tracker = SpendTracker(tmp_path / "spend.json", budget_usd=budget)
    return CachedLLMClient(fake, tmp_path / "cache", tracker, 0.10, 0.50), fake


def test_spend_is_tracked_and_persisted(tmp_path: Path) -> None:
    client, _ = _paid_client(tmp_path, budget=10.0)
    client.complete_json("a", Judgment)
    client.complete_json("a", Judgment)  # cache : gratuit
    client.complete_json("b", Judgment)
    assert client.tracker.spent_usd == pytest.approx(0.40)
    assert SpendTracker(tmp_path / "spend.json", 10.0).spent_usd == pytest.approx(0.40)


def test_budget_blocks_new_paid_calls_but_not_cache(tmp_path: Path) -> None:
    client, fake = _paid_client(tmp_path, budget=0.30)
    client.complete_json("a", Judgment)
    client.complete_json("b", Judgment)  # 0.20 < 0.30 : autorisé, on passe à 0.40
    with pytest.raises(BudgetExceeded):
        client.complete_json("c", Judgment)
    assert fake.calls == 2
    client.complete_json("a", Judgment)  # toujours servi par le cache


def test_free_provider_ignores_budget(tmp_path: Path) -> None:
    fake = FakeLLMClient(RESPONSE)
    tracker = SpendTracker(tmp_path / "spend.json", budget_usd=0.0)
    client = CachedLLMClient(fake, tmp_path / "cache", tracker)  # prix 0 : modèle local
    client.complete_json("a", Judgment)
    assert tracker.spent_usd == 0.0


def test_ollama_retries_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from mini_decision.llm import LLMError
    from mini_decision.llm.providers import OllamaClient

    def reply(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content=content), prompt_eval_count=10, eval_count=5
        )

    outputs = [reply('{"off_topic": false, "p_true": 0.9, "rationale": "tronqu'), reply(
        '{"off_topic": false, "p_true": 0.9, "rationale": "ok"}'
    )]
    client = OllamaClient("gemma4:e4b")
    seeds: list[int] = []

    def fake_chat(**kwargs):  # noqa: ANN003, ANN202
        seeds.append(kwargs["options"]["seed"])
        return outputs.pop(0)

    monkeypatch.setattr(client._client, "chat", fake_chat)
    assert client.complete_json("p", Judgment, sample=2).rationale == "ok"
    assert seeds == [2000, 2001]  # graine différente à chaque essai

    monkeypatch.setattr(client._client, "chat", lambda **kw: reply("pas du json"))
    with pytest.raises(LLMError):
        client.complete_json("p", Judgment)


def test_ollama_falls_back_when_structured_output_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import ollama

    from mini_decision.llm.providers import OllamaClient

    formats: list = []

    def fake_chat(**kwargs):  # noqa: ANN003, ANN202
        formats.append(kwargs["format"])
        if isinstance(kwargs["format"], dict):
            raise ollama.ResponseError("structured output is unavailable", 501)
        assert "schéma" in kwargs["messages"][-1]["content"]  # schéma décrit dans le prompt
        return SimpleNamespace(
            message=SimpleNamespace(content='```json\n{"off_topic": true, "p_true": 0.5}\n```'),
            prompt_eval_count=1,
            eval_count=1,
        )

    client = OllamaClient("gemma4:e4b")
    monkeypatch.setattr(client._client, "chat", fake_chat)
    assert client.complete_json("p", Judgment).off_topic is True
    assert client.complete_json("q", Judgment).off_topic is True
    assert formats[0] != "json" and formats[1:] == ["json", "json"]  # bascule mémorisée
    assert client.mode == "json"
