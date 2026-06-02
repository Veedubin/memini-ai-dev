"""Unified LLM client for memini-ai.

Supports multiple providers:
- ollama (local /api/generate)
- ollama-cloud (OpenAI-compatible via Ollama Cloud)
- openai (OpenAI /v1/chat/completions)
- openrouter (OpenRouter /v1/chat/completions)
"""

from memini_ai.llm.base import BaseLLMClient
from memini_ai.llm.factory import get_llm_client
from memini_ai.llm.ollama import OllamaClient
from memini_ai.llm.openai_compat import OpenAICompatibleClient

__all__ = ["BaseLLMClient", "get_llm_client", "OllamaClient", "OpenAICompatibleClient"]
