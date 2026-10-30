"""Implémentations Anthropic et OpenAI de `LLMClient` (sorties JSON structurées)."""

from __future__ import annotations

import json

from mini_decision.llm.base import LLMClient, LLMError, T, Usage


class AnthropicClient(LLMClient):
    """Clé lue dans ANTHROPIC_API_KEY par le SDK."""

    provider = "anthropic"

    def __init__(self, model: str, max_tokens: int = 4000) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        self.last_usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"Réponse inexploitable (stop_reason={response.stop_reason})")
        if response.parsed_output is None:
            raise LLMError("Sortie structurée absente")
        return response.parsed_output


class OpenAIClient(LLMClient):
    """Clé lue dans OPENAI_API_KEY par le SDK."""

    provider = "openai"

    def __init__(self, model: str, max_tokens: int = 4000) -> None:
        import openai

        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI()

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        completion = self._client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
            max_completion_tokens=self.max_tokens,
        )
        if completion.usage is not None:
            self.last_usage = Usage(
                completion.usage.prompt_tokens, completion.usage.completion_tokens
            )
        message = completion.choices[0].message
        if message.refusal or message.parsed is None:
            raise LLMError(f"Réponse inexploitable : {message.refusal or 'sortie absente'}")
        return message.parsed


class OllamaClient(LLMClient):
    """Modèle local servi par Ollama (gratuit) ; sortie contrainte par le schéma JSON."""

    provider = "ollama"

    def __init__(self, model: str, max_tokens: int = 4000, host: str | None = None) -> None:
        import ollama

        self.model = model
        self.max_tokens = max_tokens
        self._client = ollama.Client(host=host)

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        response = self._client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),
            # La graine varie avec `sample` : plusieurs tirages restent reproductibles.
            options={"num_predict": self.max_tokens, "seed": sample},
        )
        self.last_usage = Usage(response.prompt_eval_count or 0, response.eval_count or 0)
        try:
            return schema.model_validate(json.loads(response.message.content or ""))
        except ValueError as err:
            raise LLMError(f"JSON invalide renvoyé par Ollama : {err}") from err
