from pathlib import Path

from mini_decision.config import LLMModelConfig
from mini_decision.llm.base import BudgetExceeded, LLMClient, LLMError
from mini_decision.llm.cache import CachedLLMClient, SpendTracker, spend_summary


def make_client(
    cfg: LLMModelConfig, cache_dir: Path, tracker: SpendTracker | None = None
) -> CachedLLMClient:
    """Construit le client du fournisseur demandé, enveloppé dans le cache disque."""
    from mini_decision.llm.providers import AnthropicClient, OllamaClient, OpenAIClient

    inner: LLMClient
    if cfg.provider == "anthropic":
        inner = AnthropicClient(cfg.model, cfg.max_tokens)
    elif cfg.provider == "openai":
        inner = OpenAIClient(cfg.model, cfg.max_tokens)
    else:
        inner = OllamaClient(cfg.model, cfg.max_tokens, cfg.host)
    return CachedLLMClient(inner, cache_dir, tracker, cfg.usd_per_m_in, cfg.usd_per_m_out)


__all__ = [
    "BudgetExceeded",
    "CachedLLMClient",
    "LLMClient",
    "LLMError",
    "SpendTracker",
    "make_client",
    "spend_summary",
]
