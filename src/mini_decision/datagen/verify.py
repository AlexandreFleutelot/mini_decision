"""Labels souples : chaque contexte est jugé par plusieurs LLM, on garde la moyenne si
les juges sont d'accord, sinon on rejette l'exemple."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mini_decision.config import DataConfig
from mini_decision.llm import BudgetExceeded, LLMClient, LLMError, spend_summary
from mini_decision.llm import prompts
from mini_decision.schemas import Example, Judgment, Kind, RawPair, Rejected
from mini_decision.utils import Progress

Log = Callable[[str], None]

# Côté attendu du label selon le kind demandé au générateur (None : hors sujet).
EXPECTED_SIDE: dict[Kind, bool | None] = {
    Kind.TRUE: True,
    Kind.FALSE: False,
    Kind.COUNTERFACTUAL: False,
    Kind.OFF_TOPIC: None,
}


@dataclass(frozen=True)
class Candidate:
    pair_id: str
    context: str
    desc_true: str
    desc_false: str
    kind: Kind


def candidates(pairs: list[RawPair]) -> list[Candidate]:
    return [
        Candidate(p.pair_id, c.text.strip(), p.desc_true, p.desc_false, c.kind)
        for p in pairs
        for c in p.contexts
    ]


def too_long(text: str, cfg: DataConfig) -> bool:
    return len(text.split()) * cfg.tokens_per_word_estimate > cfg.max_context_tokens


def aggregate(kind: Kind, judgments: list[Judgment], cfg: DataConfig) -> tuple[float | None, str]:
    """Renvoie (label, "ok") ou (None, raison du rejet).

    Exemple avec deux juges à 0.9 et 0.7 sur un contexte « true » : écart 0.2 ≤ 0.3, label 0.8.
    Avec 0.9 et 0.4 : écart 0.5 > 0.3, rejet « désaccord ».
    """
    if kind is Kind.OFF_TOPIC:
        if all(j.off_topic for j in judgments):
            return cfg.off_topic_label, "ok"
        return None, "hors_sujet_contesté"
    if any(j.off_topic for j in judgments):
        return None, "jugé_hors_sujet"
    ps = [j.p_true for j in judgments]
    if max(ps) - min(ps) > cfg.judge_max_disagreement:
        return None, "désaccord"
    label = sum(ps) / len(ps)
    if (label > 0.5) != EXPECTED_SIDE[kind] or label == 0.5:
        return None, "contredit_le_kind"  # le générateur s'est trompé de côté
    return label, "ok"


def judge_all(
    cands: list[Candidate],
    judges: list[LLMClient],
    cfg: DataConfig,
    max_concurrency: int,
    log: Log = print,
) -> tuple[list[Example], list[Rejected], list[float]]:
    """Renvoie (gardés, rejetés, écarts max − min entre juges pour chaque contexte jugé)."""
    spreads: list[float] = []

    def one(c: Candidate) -> Example | Rejected | None:
        if too_long(c.context, cfg):
            return Rejected(
                pair_id=c.pair_id,
                context=c.context,
                kind=c.kind,
                reason="trop_long",
                judgments=[],
            )
        prompt = prompts.JUDGE.format(
            context=c.context, desc_true=c.desc_true, desc_false=c.desc_false
        )
        try:
            judgments = [
                j.complete_json(prompt, Judgment, system=prompts.SYSTEM_JUDGE) for j in judges
            ]
        except BudgetExceeded:
            raise
        except LLMError as err:
            log(f"  jugement ignoré ({c.pair_id}) : {err}")
            return None
        ps = [j.p_true for j in judgments]
        spreads.append(max(ps) - min(ps))
        label, reason = aggregate(c.kind, judgments, cfg)
        if label is None:
            return Rejected(
                pair_id=c.pair_id,
                context=c.context,
                kind=c.kind,
                reason=reason,
                judgments=judgments,
            )
        return Example(
            pair_id=c.pair_id,
            context=c.context,
            desc_true=c.desc_true,
            desc_false=c.desc_false,
            label=round(label, 4),
            kind=c.kind,
        )

    results: dict[int, Example | Rejected | None] = {}

    def extra() -> str:
        n_kept = sum(isinstance(r, Example) for r in results.values())
        return f"gardés {n_kept}, rejetés {len(results) - n_kept} · {spend_summary(judges)}"

    progress = Progress("verify", len(cands), log, extra=extra)
    with ThreadPoolExecutor(max_concurrency) as pool:
        futures = {pool.submit(one, c): i for i, c in enumerate(cands)}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                progress.update()
        except BudgetExceeded as err:
            log(f"Arrêt : {err}")
            for f in futures:
                f.cancel()
    ordered = [results[i] for i in sorted(results)]  # ordre stable, quel que soit le parallélisme
    kept = [r for r in ordered if isinstance(r, Example)]
    rejected = [r for r in ordered if isinstance(r, Rejected)]
    return kept, rejected, spreads
