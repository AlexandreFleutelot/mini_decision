"""Chargement des configs YAML (héritage + surcharges) et validation typée."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    adversarial_dir: Path
    real_calibration: Path
    llm_cache_dir: Path
    runs_dir: Path
    tokenizer_prefix: Path  # SentencePiece écrit <prefix>.model et <prefix>.vocab


class SplitConfig(_Strict):
    train: float
    val: float
    test: float

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("train + val + test doit valoir 1")
        return self


class DataConfig(_Strict):
    domain: str
    subdomains: list[str] = Field(min_length=1)
    language: str
    n_pairs: int
    contexts_per_pair: int
    kind_ratios: dict[Literal["true", "false", "off_topic"], float]
    counterfactual_share_of_false: float = Field(ge=0, le=1)
    max_context_tokens: int
    tokens_per_word_estimate: float  # filtre de longueur avant d'avoir le tokenizer
    unlabeled_texts_per_call: int
    judge_max_disagreement: float
    off_topic_label: float
    unlabeled_ratio: float
    split: SplitConfig
    leakage_max_accuracy: float


class LLMModelConfig(_Strict):
    provider: Literal["anthropic", "openai", "ollama"]
    model: str
    max_tokens: int = 4000
    usd_per_m_in: float = 0.0
    usd_per_m_out: float = 0.0
    host: str | None = None  # ollama uniquement


class LLMConfig(_Strict):
    generator: LLMModelConfig
    judges: list[LLMModelConfig] = Field(min_length=1)
    budget_usd: float
    max_concurrency: int = 8


class TokenizerConfig(_Strict):
    model_type: Literal["unigram", "bpe"]
    vocab_size: int
    special_tokens: list[str]


class TransformerConfig(_Strict):
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int


class CNNConfig(_Strict):
    n_layers: int
    d_model: int
    kernel_size: int
    dilations: list[int]


class NgramConfig(_Strict):
    n_max: int
    n_buckets: int
    d_model: int
    hidden: int


class ModelConfig(_Strict):
    encoder: Literal["ngram", "cnn", "transformer"]
    head: Literal["qda", "dot"]
    cov: Literal["diag", "full"]
    k: int
    s_min: float
    s_max: float
    tau_init: float | Literal["auto"]  # auto : moyenne de max(s_vrai, s_faux) au 1er batch
    max_len: int
    dropout: float
    transformer: TransformerConfig
    cnn: CNNConfig
    ngram: NgramConfig


class MLMConfig(_Strict):
    mask_prob: float
    oversample_factor: float
    oversample_words: list[str]
    oversample_numbers: bool
    mask_token_prob: float  # parmi les tokens masqués : part remplacée par [MASK]
    random_token_prob: float  # part remplacée par un token aléatoire (le reste est inchangé)
    lr: float
    weight_decay: float
    warmup_frac: float
    grad_clip: float
    batch_size: int
    max_epochs: int
    patience: int
    run_name: str  # checkpoint : <runs_dir>/<run_name>-<encodeur>/encoder.pt


class TrainConfig(_Strict):
    lr: float
    weight_decay: float
    warmup_frac: float
    batch_size: int
    grad_clip: float
    max_epochs: int
    patience: int
    aux_mlm_lambda: float
    rda_alpha_end_frac: float
    init_from_mlm: bool
    swap_prob: float  # échange desc_vrai/desc_faux (label -> 1 − label) : la position ne dit rien
    max_train_examples: int | None  # sous-échantillon par paires (courbe d'échelle) ; None = tout
    scaling_sizes: list[int]


class AdversarialConfig(_Strict):
    n_rounds: int
    error_threshold: float
    n_pairs_per_round: int
    errors_per_prompt: tuple[int, int]
    new_contexts_per_pair: int
    max_share: float


class EvalConfig(_Strict):
    ece_bins: int
    decision_threshold: float


class Config(_Strict):
    seed: int
    run_name: str
    device: Literal["auto", "cpu", "cuda", "mps"]
    paths: PathsConfig
    data: DataConfig
    llm: LLMConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    mlm: MLMConfig
    train: TrainConfig
    adversarial: AdversarialConfig
    eval: EvalConfig


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : les dicts sont fusionnés, le reste est remplacé."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return deep_merge(_load_raw(path.parent / parent), raw)


def parse_override(item: str) -> dict[str, Any]:
    """'model.head=dot' -> {'model': {'head': 'dot'}} (valeur interprétée en YAML)."""
    if "=" not in item:
        raise ValueError(f"Surcharge invalide (attendu clé=valeur) : {item!r}")
    dotted, value = item.split("=", 1)
    tree: dict[str, Any] = {}
    node = tree
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(value)
    return tree


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = _load_raw(Path(path))
    for item in overrides or []:
        raw = deep_merge(raw, parse_override(item))
    return Config.model_validate(raw)
