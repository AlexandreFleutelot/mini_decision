"""Génération des paires (desc_true, desc_false) + contextes, et du corpus non étiqueté."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from mini_decision.config import DataConfig
from mini_decision.llm import BudgetExceeded, LLMClient, LLMError
from mini_decision.llm import prompts
from mini_decision.schemas import GeneratedPair, Kind, RawPair, UnlabeledBatch

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
) -> list[RawPair]:
    """Génère `cfg.n_pairs` paires, réparties en tourniquet entre les sous-domaines.

    Les appels partent par vagues de `max_concurrency` ; chaque vague reçoit la liste des
    propositions déjà générées dans son sous-domaine (les `avoid_window` dernières), pour
    limiter les doublons. Une paire qui échoue est ignorée ; le plafond de dépense arrête tout
    en gardant ce qui est fait.
    """
    plan = plan_kinds(cfg)
    system = prompts.SYSTEM_GENERATOR.format(language=cfg.language)
    done: list[RawPair] = []
    by_subdomain: dict[str, list[str]] = {s: [] for s in cfg.subdomains}

    def one(i: int, avoid: list[str]) -> RawPair | None:
        subdomain = cfg.subdomains[i % len(cfg.subdomains)]
        prompt = _pair_prompt(cfg, plan, subdomain, avoid)
        try:
            pair = clean_pair(client.complete_json(prompt, GeneratedPair, system=system, sample=i))
        except BudgetExceeded:
            raise
        except LLMError as err:
            log(f"  paire {i} ignorée : {err}")
            return None
        return RawPair(
            pair_id=f"p_{i:05d}",
            subdomain=subdomain,
            desc_true=pair.desc_true,
            desc_false=pair.desc_false,
            contexts=pair.contexts,
        )

    with ThreadPoolExecutor(max_concurrency) as pool:
        for start in range(0, cfg.n_pairs, max_concurrency):
            wave = range(start, min(start + max_concurrency, cfg.n_pairs))
            avoids = [
                by_subdomain[cfg.subdomains[i % len(cfg.subdomains)]][-avoid_window:] for i in wave
            ]
            try:
                results = list(pool.map(one, wave, avoids))
            except BudgetExceeded as err:
                log(f"Arrêt : {err}")
                break
            for raw in results:
                if raw is not None:
                    done.append(raw)
                    by_subdomain[raw.subdomain].append(raw.desc_true)
            log(f"  {len(done)}/{cfg.n_pairs} paires")
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

    texts: list[str] = []
    try:
        with ThreadPoolExecutor(max_concurrency) as pool:
            for batch in pool.map(one, range(n_calls)):
                texts.extend(t.strip() for t in batch if t.strip())
    except BudgetExceeded as err:
        log(f"Arrêt : {err}")
    log(f"  {len(texts)} textes non étiquetés")
    return texts[:n_texts]
