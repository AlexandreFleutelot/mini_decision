"""Split par pair_id, jeu de contrefactuels, contrôle de fuite et statistiques."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean, median

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mini_decision.config import Config, SplitConfig
from mini_decision.schemas import Example, Kind, Rejected


def split_by_pair(
    examples: list[Example], ratios: SplitConfig, seed: int
) -> dict[str, list[Example]]:
    """80/10/10 sur les pair_id (jamais sur les exemples) : le test ne contient que des paires
    jamais vues à l'entraînement."""
    pair_ids = sorted({e.pair_id for e in examples})
    random.Random(seed).shuffle(pair_ids)
    n_train = round(len(pair_ids) * ratios.train)
    n_val = round(len(pair_ids) * ratios.val)
    assignment = {pid: "train" for pid in pair_ids[:n_train]}
    assignment |= {pid: "val" for pid in pair_ids[n_train : n_train + n_val]}
    assignment |= {pid: "test" for pid in pair_ids[n_train + n_val :]}
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for e in examples:
        splits[assignment[e.pair_id]].append(e)
    return splits


def counterfactual_set(test: list[Example]) -> list[Example]:
    return [e for e in test if e.kind is Kind.COUNTERFACTUAL]


def _baseline_accuracy(
    train: list[Example], val: list[Example], text: str, seed: int
) -> float:
    """Régression logistique sur n-grammes de caractères d'une partie seulement de l'entrée.

    text = "descs" : (desc_true, desc_false) sans contexte, le contrôle de fuite de la spec.
    text = "context" : le contexte sans les descriptions, pour détecter des artefacts de style
    (par ex. « tous les contextes faux contiennent "pas" »).
    """

    def view(e: Example) -> str:
        return f"{e.desc_true} || {e.desc_false}" if text == "descs" else e.context

    train = [e for e in train if e.kind is not Kind.OFF_TOPIC]
    val = [e for e in val if e.kind is not Kind.OFF_TOPIC]
    y_train = [e.label > 0.5 for e in train]
    if len(set(y_train)) < 2 or not val:
        return float("nan")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform([view(e) for e in train])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(x_train, y_train)
    pred = clf.predict(vec.transform([view(e) for e in val]))
    return sum(bool(p) == (e.label > 0.5) for p, e in zip(pred, val)) / len(val)


def leakage_report(splits: dict[str, list[Example]], cfg: Config) -> dict[str, float]:
    return {
        "descs_only": _baseline_accuracy(splits["train"], splits["val"], "descs", cfg.seed),
        "context_only": _baseline_accuracy(splits["train"], splits["val"], "context", cfg.seed),
    }


# --- Statistiques et rapport -----------------------------------------------------------------


def _length_table(examples: list[Example]) -> list[str]:
    words: dict[Kind, list[int]] = defaultdict(list)
    for e in examples:
        words[e.kind].append(len(e.context.split()))
    rows = ["| kind | n | mots (moy.) | mots (méd.) | min | max |", "|---|---|---|---|---|---|"]
    for kind in Kind:
        w = words.get(kind, [])
        if w:
            rows.append(
                f"| {kind.value} | {len(w)} | {mean(w):.1f} | {median(w):.0f} | {min(w)} | {max(w)} |"
            )
    return rows


def _histogram(values: list[float], edges: list[float]) -> list[str]:
    rows = ["| écart entre juges | n |", "|---|---|"]
    for lo, hi in zip(edges, edges[1:]):
        n = sum(lo <= v < hi or (hi == edges[-1] and v == hi) for v in values)
        rows.append(f"| [{lo:.1f}, {hi:.1f}{']' if hi == edges[-1] else '['} | {n} |")
    return rows


def build_report(
    splits: dict[str, list[Example]],
    rejected: list[Rejected],
    spreads: list[float],
    leakage: dict[str, float],
    subdomain_of: dict[str, str],
    cfg: Config,
    n_examples_per_kind: int = 10,
) -> str:
    kept = [e for split in splits.values() for e in split]
    n_candidates = len(kept) + len(rejected)
    by_kind = Counter(e.kind.value for e in kept)
    by_subdomain = Counter(subdomain_of[e.pair_id] for e in kept)
    reasons = Counter(r.reason for r in rejected)
    threshold = cfg.data.leakage_max_accuracy

    lines = [
        "# Données v0 : statistiques",
        "",
        f"- Paires : {len({e.pair_id for e in kept})} ; contextes candidats : {n_candidates} ; "
        f"gardés : {len(kept)} ({len(kept) / max(n_candidates, 1):.0%}).",
        "- Split (par pair_id) : "
        + ", ".join(
            f"{name} {len(v)} ex. / {len({e.pair_id for e in v})} paires"
            for name, v in splits.items()
        ),
        f"- Contrefactuels de test : {len(counterfactual_set(splits['test']))}.",
        "",
        "## Répartition",
        "",
        "| kind | n | part |",
        "|---|---|---|",
        *[f"| {k} | {n} | {n / len(kept):.0%} |" for k, n in by_kind.most_common()],
        "",
        "| sous-domaine | n |",
        "|---|---|",
        *[f"| {s} | {n} |" for s, n in by_subdomain.most_common()],
        "",
        "## Longueurs des contextes (mots)",
        "",
        *_length_table(kept),
        "",
        "## Rejets et désaccords",
        "",
        "| raison | n |",
        "|---|---|",
        *[f"| {r} | {n} |" for r, n in reasons.most_common()],
        "",
        *_histogram(spreads, [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
        "",
        "## Contrôle de fuite (accuracy sur le val, contextes pertinents)",
        "",
        f"- Descriptions seules : {leakage['descs_only']:.1%} "
        + ("**FUITE : au-dessus du seuil**" if leakage["descs_only"] > threshold else "(OK)"),
        f"- Contexte seul (artefacts de style) : {leakage['context_only']:.1%} "
        + ("**à surveiller**" if leakage["context_only"] > threshold else "(OK)"),
        f"- Seuil : {threshold:.0%}.",
        "",
        f"## {n_examples_per_kind} exemples par kind",
    ]
    rng = random.Random(cfg.seed)
    for kind in Kind:
        pool = [e for e in kept if e.kind is kind]
        lines += ["", f"### {kind.value}", ""]
        for e in rng.sample(pool, min(n_examples_per_kind, len(pool))):
            lines += [
                f"- **{e.label:.2f}** · {e.context}",
                f"  - vrai : {e.desc_true}",
                f"  - faux : {e.desc_false}",
            ]
    return "\n".join(lines) + "\n"
