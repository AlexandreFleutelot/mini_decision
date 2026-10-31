"""Cache disque des appels LLM : un appel déjà payé n'est jamais refait.

Clé = sha256 de (fournisseur, modèle, system, prompt, schéma, tirage). Chaque entrée est un
fichier JSON `<clé[:2]>/<clé>.json` qui contient aussi le prompt, pour pouvoir l'inspecter.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from pydantic import BaseModel

from mini_decision.llm.base import BudgetExceeded, LLMClient, T


class SpendTracker:
    """Dépense cumulée en $, persistée sur disque : le plafond vaut pour tous les runs."""

    def __init__(self, path: Path, budget_usd: float) -> None:
        self.path = Path(path)
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()  # les appels partent en parallèle
        if self.path.exists():
            self.spent_usd = json.loads(self.path.read_text(encoding="utf-8"))["spent_usd"]

    def check(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"Plafond atteint : {self.spent_usd:.2f} $ dépensés sur {self.budget_usd:.2f} $. "
                "Augmenter llm.budget_usd pour continuer."
            )

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"spent_usd": self.spent_usd}), encoding="utf-8")


def cache_key(
    provider: str, model: str, system: str, prompt: str, schema: type[BaseModel], sample: int
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema(),
            "sample": sample,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def spend_summary(clients: list[LLMClient]) -> str:
    """« 0.32 $ » si l'un des clients suit une dépense, sinon chaîne vide."""
    for client in clients:
        tracker = getattr(client, "tracker", None)
        if tracker is not None:
            return f"{tracker.spent_usd:.2f} $ / {tracker.budget_usd:.0f} $"
    return ""


class CachedLLMClient(LLMClient):
    """Enveloppe n'importe quel `LLMClient` avec un cache disque."""

    def __init__(
        self,
        inner: LLMClient,
        cache_dir: Path,
        tracker: SpendTracker | None = None,
        usd_per_m_in: float = 0.0,
        usd_per_m_out: float = 0.0,
    ) -> None:
        self.inner = inner
        self.tracker = tracker
        self.usd_per_m_in = usd_per_m_in
        self.usd_per_m_out = usd_per_m_out
        self.provider = inner.provider
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete_json(
        self, prompt: str, schema: type[T], *, system: str = "", sample: int = 0
    ) -> T:
        key = cache_key(self.provider, self.model, system, prompt, schema, sample)
        path = self._path(key)
        if path.exists():
            self.hits += 1
            entry = json.loads(path.read_text(encoding="utf-8"))
            return schema.model_validate(entry["response"])

        self.misses += 1
        paid = self.usd_per_m_in > 0 or self.usd_per_m_out > 0 or self.inner.reports_cost
        if self.tracker is not None and paid:
            self.tracker.check()
        result = self.inner.complete_json(prompt, schema, system=system, sample=sample)
        usage = self.inner.last_usage
        if usage.cost_usd is not None:
            usd = usage.cost_usd
        else:
            usd = (
                usage.input_tokens * self.usd_per_m_in + usage.output_tokens * self.usd_per_m_out
            ) / 1e6
        if self.tracker is not None and paid:
            self.tracker.add(usd)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "provider": self.provider,
            "model": self.model,
            "schema": schema.__name__,
            "sample": sample,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            "usd": usd,
            "system": system,
            "prompt": prompt,
            "response": result.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)  # écriture atomique : pas d'entrée corrompue si on interrompt
        return result
