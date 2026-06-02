"""Factory for creating LLM clients from configuration."""

from __future__ import annotations

from memini_ai.config import MeminiConfig, get_config
from memini_ai.llm.base import BaseLLMClient
from memini_ai.llm.ollama import OllamaClient
from memini_ai.llm.openai_compat import OpenAICompatibleClient
from memini_ai.utils.logger import logger

_providers: dict[str, BaseLLMClient] = {}

# Default base URLs for cloud providers
_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama-cloud": "https://ollama.com/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _normalize_provider(provider: str) -> str:
    """Normalize provider string to lowercase, stripping whitespace."""
    return provider.lower().strip()


def _derive_base_url(config: MeminiConfig) -> str:
    """Determine base URL for OpenAI-compatible providers.

    Priority:
    1. Explicit llm_base_url config field
    2. Derive base URL from llm_url (strip /api/generate if present)
    3. Provider default
    """
    if config.llm_base_url:
        return config.llm_base_url.rstrip("/")

    url = config.llm_url or ""
    # If llm_url contains known Ollama-native endpoint, strip it
    if "/api/generate" in url:
        return url.split("/api/generate")[0].rstrip("/")
    return url.rstrip("/")


def get_llm_client(config: MeminiConfig | None = None) -> BaseLLMClient:
    """Get or create a cached LLM client based on configuration.

    Backward compatibility:
    - llm_provider == "ollama" and llm_base_url is not set -> use llm_url as full endpoint
    - llm_provider == "ollama-cloud" or llm_base_url set -> use base_url with /chat/completions

    Args:
        config: MeminiConfig instance. Uses global config if None.

    Returns:
        BaseLLMClient instance appropriate for the configured provider.
    """
    if config is None:
        config = get_config()

    provider = _normalize_provider(config.llm_provider)
    cache_key = f"{provider}:{config.llm_model}:{config.llm_url}:{config.llm_base_url}:{config.llm_api_key}"

    if cache_key in _providers:
        return _providers[cache_key]

    model = config.llm_model or "llama3.2"

    if provider == "ollama":
        # For local Ollama, use llm_url as the full endpoint URL
        url = config.llm_url or "http://localhost:11434/api/generate"
        client: BaseLLMClient = OllamaClient(url=url, model=model)
    elif provider in ("ollama-cloud", "openai", "openrouter"):
        base_url = _derive_base_url(config)
        if not base_url:
            base_url = _DEFAULT_BASE_URLS.get(provider, "https://ollama.com/v1")
        client = OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            api_key=config.llm_api_key,
        )
    else:
        # Unknown provider: warn and fall back to treating as Ollama-like
        logger.warning(
            "unknown_llm_provider",
            provider=provider,
            fallback="ollama",
        )
        url = config.llm_url or "http://localhost:11434/api/generate"
        client = OllamaClient(url=url, model=model)

    _providers[cache_key] = client
    return client


def clear_llm_client_cache() -> None:
    """Close and clear all cached LLM clients."""
    for client in _providers.values():
        if hasattr(client, "close"):
            try:
                import asyncio

                asyncio.create_task(client.close())
            except Exception:
                pass
    _providers.clear()
