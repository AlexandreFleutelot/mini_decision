"""Génération des paires (desc_true, desc_false) + contextes, et du corpus non étiqueté."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import as_completed
from dataclasses import dataclass

from mini_decision.config import DataConfig
from mini_decision.llm import BudgetExceeded, LLMClient, LLMError, spend_summary
from mini_decision.llm import prompts
from mini_decision.schemas import GeneratedPair, Kind, RawPair, UnlabeledBatch
from mini_decision.utils import Progress, interruptible_pool

Log = Callable[[str], None]


@dataclass(frozen=True)
class KindPlan:
    n_true: int
    n_false: int
    n_counterfactual: int
    n_off_topic: int

    @property
    def total(self) -> int:
        return self.n_true + self.n_false + self.n_counterfactual + self.n_off_topic


def plan_kinds(cfg: DataConfig) -> KindPlan:
    """Répartit `contexts_per_pair` selon les ratios ; le reste va aux vrais."""
    n = cfg.contexts_per_pair
    n_false_total = round(n * cfg.kind_ratios["false"])
    n_off_topic = round(n * cfg.kind_ratios["off_topic"])
    n_counterfactual = round(n_false_total * cfg.counterfactual_share_of_false)
    return KindPlan(
        n_true=n - n_false_total - n_off_topic,
        n_false=n_false_total - n_counterfactual,
        n_counterfactual=n_counterfactual,
        n_off_topic=n_off_topic,
    )


def clean_pair(pair: GeneratedPair) -> GeneratedPair:
    """Retire les contextes vides et les contrefactuels sans contexte vrai source valide."""
    true_indices = {i for i, c in enumerate(pair.contexts) if c.kind is Kind.TRUE}
    kept = [
        c
        for c in pair.contexts
        if c.text.strip()
        and (c.kind is not Kind.COUNTERFACTUAL or c.source_index in true_indices)
    ]
    return pair.model_copy(update={"contexts": kept})


def _pair_prompt(cfg: DataConfig, plan: KindPlan, subdomain: str, avoid: list[str]) -> str:
    return prompts.GENERATE_PAIR.format(
        domain=cfg.domain,
        subdomain=subdomain,
        n_contexts=plan.total,
        n_true=plan.n_true,
        n_false=plan.n_false,
        n_counterfactual=plan.n_counterfactual,
        n_off_topic=plan.n_off_topic,
        avoid="; ".join(avoid) if avoid else "(aucune)",
    )


def generate_pairs(
    cfg: DataConfig,
    client: LLMClient,
    max_concurrency: int,
    log: Log = print,
    avoid_window: int = 30,
    extra: list[tuple[LLMClient, int]] | None = None,
) -> list[RawPair]:
    """Génère `cfg.n_pairs` paires, réparties en tourniquet entre les sous-domaines.

    `extra` ajoute des blocs de paires faites par d'autres générateurs, à la suite :
    [(deepseek, 1000)] -> p_00000..p_02099 par le générateur principal, puis p_02100..p_03099
    par deepseek. Le bloc principal ne change pas : il reste servi par le cache.

    Les appels partent par vagues de `max_concurrency` ; chaque vague reçoit la liste des
    propositions déjà générées dans son sous-domaine (les `avoid_window` dernières), pour
    limiter les doublons. Une paire qui échoue est ignorée ; le plafond de dépense arrête tout
    en gardant ce qui est fait.
    """
    plan = plan_kinds(cfg)
    system = prompts.SYSTEM_GENERATOR.format(language=cfg.language)
    schedule = [client] * cfg.n_pairs
    for other, n in extra or []:
        schedule += [other] * n
    clients = list(dict.fromkeys(schedule)) or [client]
    total = len(schedule)
    done: list[RawPair] = []
    by_subdomain: dict[str, list[str]] = {s: [] for s in cfg.subdomains}
    seen: set[str] = set()
    n_duplicates = 0

    def one(i: int, avoid: list[str]) -> RawPair | None:
        subdomain = cfg.subdomains[i % len(cfg.subdomains)]
        prompt = _pair_prompt(cfg, plan, subdomain, avoid)
        try:
            gen = schedule[i]
            pair = clean_pair(gen.complete_json(prompt, GeneratedPair, system=system, sample=i))
        except BudgetExceeded:
            raise
        except LLMError as err:
            log(f"  paire {i} ignorée ({schedule[i].model}) : {err}")
            return None
        return RawPair(
            pair_id=f"p_{i:05d}",
            subdomain=subdomain,
            generator=gen.model,
            desc_true=pair.desc_true,
            desc_false=pair.desc_false,
            contexts=pair.contexts,
        )

    progress = Progress(
        "datagen paires", total, log, every_s=0.0, extra=lambda: spend_summary(clients)
    )
    with interruptible_pool(max_concurrency) as pool:
        for start in range(0, total, max_concurrency):
            wave = range(start, min(start + max_concurrency, total))
            avoids = [
                by_subdomain[cfg.subdomains[i % len(cfg.subdomains)]][-avoid_window:] for i in wave
            ]
            try:
                results = list(pool.map(one, wave, avoids))
            except BudgetExceeded as err:
                log(f"Arrêt : {err}")
                break
            for raw in results:
                if raw is None:
                    continue
                key = " ".join(raw.desc_true.lower().split())
                if key in seen:  # doublon exact d'une proposition déjà générée
                    n_duplicates += 1
                    continue
                seen.add(key)
                done.append(raw)
                by_subdomain[raw.subdomain].append(raw.desc_true)
            progress.update(len(wave))
    if n_duplicates:
        log(f"  {n_duplicates} paires en double écartées")
    return done


def generate_unlabeled(
    cfg: DataConfig, client: LLMClient, n_texts: int, max_concurrency: int, log: Log = print
) -> list[str]:
    """Corpus non étiqueté pour le MLM, par lots de `unlabeled_texts_per_call` textes."""
    n_calls = -(-n_texts // cfg.unlabeled_texts_per_call)
    system = prompts.SYSTEM_GENERATOR.format(language=cfg.language)

    def one(i: int) -> list[str]:
        prompt = prompts.GENERATE_UNLABELED.format(
            domain=cfg.domain,
            subdomain=cfg.subdomains[i % len(cfg.subdomains)],
            n=cfg.unlabeled_texts_per_call,
            language=cfg.language,
            batch=i,
        )
        try:
            return client.complete_json(prompt, UnlabeledBatch, system=system, sample=i).texts
        except BudgetExceeded:
            raise
        except LLMError as err:
            log(f"  lot non étiqueté {i} ignoré : {err}")
            return []

    batches: dict[int, list[str]] = {}
    progress = Progress(
        "datagen non étiqueté", n_calls, log, extra=lambda: spend_summary([client])
    )
    with interruptible_pool(max_concurrency) as pool:
        futures = {pool.submit(one, i): i for i in range(n_calls)}
        try:
            for future in as_completed(futures):
                batches[futures[future]] = future.result()
                progress.update()
        except BudgetExceeded as err:
            log(f"Arrêt : {err}")
            for f in futures:
                f.cancel()
    texts = [t.strip() for i in sorted(batches) for t in batches[i] if t.strip()]
    return texts[:n_texts]
