"""Implémentations Anthropic, OpenAI et Ollama de `LLMClient` (sorties JSON structurées)."""

from __future__ import annotations

import json
import threading

from pydantic import BaseModel

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
        import anthropic

        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ) as err:  # transitoires, déjà retentées par le SDK : on saute cet exemple
            raise LLMError(f"Anthropic : {err}") from err
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
    """Modèle local servi par Ollama (gratuit).

    La sortie est contrainte par le schéma JSON quand le serveur le permet. Certains moteurs
    d'Ollama (par ex. le runner MLX) répondent 501 « structured output is unavailable » : on
    se rabat alors sur le mode JSON simple, puis sur du texte libre, avec le schéma décrit
    dans le prompt. Le mode retenu est mémorisé pour les appels suivants.
    """

    provider = "ollama"
    MODES = ("schema", "json", "text")

    def __init__(
        self, model: str, max_tokens: int = 4000, host: str | None = None, max_attempts: int = 3
    ) -> None:
        import ollama

        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.mode = "schema"
        self._client = ollama.Client(host=host)
        self._lock = threading.Lock()

    def _chat(self, messages: list[dict], schema: type[T], seed: int):  # noqa: ANN202
        import ollama

        while True:
            mode = self.mode
            fmt = {"schema": schema.model_json_schema(), "json": "json", "text": None}[mode]
            msgs = messages
            if mode != "schema":
                hint = "\n\n" + _format_hint(schema)
                last = {**messages[-1], "content": messages[-1]["content"] + hint}
                msgs = [*messages[:-1], last]
            try:
                return self._client.chat(
                    model=self.model,
                    messages=msgs,
                    format=fmt,
                    options={"num_predict": self.max_tokens, "seed": seed},
                )
            except ollama.ResponseError as err:
                if err.status_code == 501 and mode != "text":
                    with self._lock:
                        if self.mode == mode:  # un autre thread a peut-être déjà basculé
                            self.mode = self.MODES[self.MODES.index(mode) + 1]
                    continue
                raise LLMError(f"Ollama : {err.error} (HTTP {err.status_code})") from err
            except ConnectionError as err:
                raise LLMError(f"Ollama injoignable : {err}") from err

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        # Les petits modèles locaux renvoient parfois un JSON tronqué ou mal formé : on
        # retente avec une autre graine (reproductible, dérivée de `sample`).
        error: Exception | None = None
        for attempt in range(self.max_attempts):
            response = self._chat(messages, schema, seed=sample * 1000 + attempt)
            self.last_usage = Usage(response.prompt_eval_count or 0, response.eval_count or 0)
            try:
                data = _unwrap(_extract_json(response.message.content or ""))
                return schema.model_validate(data)
            except ValueError as err:
                error = err
        raise LLMError(f"JSON invalide renvoyé par Ollama ({self.max_attempts} essais) : {error}")


_TYPE_NAMES = {
    "boolean": "booléen",
    "number": "nombre",
    "integer": "entier",
    "string": "texte",
    "array": "liste",
    "object": "objet",
}


def _format_hint(schema: type[BaseModel]) -> str:
    """Consigne en clair (pas le schéma JSON brut, que les petits modèles recopient)."""
    props = schema.model_json_schema().get("properties", {})
    fields = ", ".join(
        f"« {name} » ({_TYPE_NAMES.get(spec.get('type', ''), 'valeur')})"
        for name, spec in props.items()
    )
    return (
        "Réponds uniquement par un objet JSON plat (pas un schéma, pas de texte autour) "
        f"avec exactement ces clés : {fields}."
    )


def _unwrap(data: dict) -> dict:
    """Si le modèle a imité un schéma ({"properties": {...valeurs...}}), en extrait les valeurs."""
    inner = data.get("properties")
    if isinstance(inner, dict) and not any(isinstance(v, dict) for v in inner.values()):
        return inner
    return data


def _extract_json(text: str) -> dict:
    """Premier objet JSON du texte (tolère un bloc ```json ... ``` ou du texte autour)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("aucun objet JSON dans la réponse")
    return json.loads(text[start : end + 1])
