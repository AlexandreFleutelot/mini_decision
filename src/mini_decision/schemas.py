"""Schémas pydantic : exemples du dataset et sorties structurées des LLM."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T", bound=BaseModel)


class Kind(str, Enum):
    TRUE = "true"
    FALSE = "false"
    COUNTERFACTUAL = "counterfactual"
    OFF_TOPIC = "off_topic"


class Example(BaseModel):
    """Une ligne du dataset (JSONL)."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    context: str = Field(min_length=1)
    desc_true: str = Field(min_length=1)
    desc_false: str = Field(min_length=1)
    label: float = Field(ge=0.0, le=1.0)
    kind: Kind
    round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _off_topic_is_half(self) -> Example:
        if self.kind is Kind.OFF_TOPIC and self.label != 0.5:
            raise ValueError("un exemple hors sujet doit avoir label = 0.5")
        return self


class UnlabeledText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


# --- Sorties structurées demandées aux LLM -------------------------------------------------


class GeneratedContext(BaseModel):
    text: str
    kind: Kind
    source_index: int | None = Field(
        default=None, description="Pour un contrefactuel : index du contexte vrai édité."
    )


class GeneratedPair(BaseModel):
    """Sortie du générateur : une paire de descriptions et ses contextes."""

    desc_true: str
    desc_false: str
    contexts: list[GeneratedContext]


class Judgment(BaseModel):
    """Sortie d'un juge pour un (contexte, desc_true, desc_false)."""

    off_topic: bool
    p_true: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class UnlabeledBatch(BaseModel):
    texts: list[str]


class AdversarialBatch(BaseModel):
    contexts: list[GeneratedContext]


# --- Fichiers intermédiaires ------------------------------------------------------------------


class RawPair(BaseModel):
    """Sortie de `datagen` (data/raw/pairs.jsonl), avant jugement."""

    pair_id: str
    subdomain: str
    generator: str = ""  # modèle qui a écrit la paire
    desc_true: str
    desc_false: str
    contexts: list[GeneratedContext]


class Rejected(BaseModel):
    """Exemple écarté par `verify`, avec la raison et l'avis des juges."""

    pair_id: str
    context: str
    kind: Kind
    reason: str
    judgments: list[Judgment]


# --- E/S JSONL ------------------------------------------------------------------------------


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
            n += 1
    return n
