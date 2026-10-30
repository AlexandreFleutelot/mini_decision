"""Interface commune des clients LLM : un prompt + un schéma pydantic -> un objet validé."""

from __future__ import annotations

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


class LLMClient(ABC):
    provider: str
    model: str
    last_usage: Usage = Usage()  # tokens du dernier appel réel, pour le suivi des coûts

    @abstractmethod
    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        """Renvoie la réponse du LLM validée par `schema`.

        `sample` distingue plusieurs tirages indépendants du même prompt (il entre dans la clé
        de cache ; il n'est pas envoyé au fournisseur).
        """
