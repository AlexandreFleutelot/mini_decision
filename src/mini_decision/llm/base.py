"""Interface commune des clients LLM : un prompt + un schéma pydantic -> un objet validé."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Appel LLM impossible à exploiter (refus, sortie tronquée, JSON invalide)."""


class BudgetExceeded(LLMError):
    """Le plafond de dépense configuré (llm.budget_usd) est atteint."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # coût annoncé par le fournisseur (OpenRouter), prioritaire


class LLMClient(ABC):
    provider: str
    model: str
    reports_cost: bool = False  # True : le fournisseur renvoie le coût réel de chaque appel

    # Tokens (et coût) du dernier appel réel *de ce thread* : les appels partent en parallèle,
    # un simple attribut serait écrasé par un autre thread avant que le cache ne le lise.
    @property
    def last_usage(self) -> Usage:
        return getattr(self._usage_local(), "usage", Usage())

    @last_usage.setter
    def last_usage(self, usage: Usage) -> None:
        self._usage_local().usage = usage

    def _usage_local(self) -> threading.local:
        return self.__dict__.setdefault("_usage_tls", threading.local())

    @abstractmethod
    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        """Renvoie la réponse du LLM validée par `schema`.

        `sample` distingue plusieurs tirages indépendants du même prompt (il entre dans la clé
        de cache ; il n'est pas envoyé au fournisseur).
        """
