"""Ollama local client for /api/generate endpoint."""

from __future__ import annotations

import re

import httpx

from memini_ai.llm.base import BaseLLMClient

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_content(text: str) -> str:
    """Strip reasoning-model artifacts from generated content."""
    return _THINK_TAG_RE.sub("", text).strip()


class OllamaClient(BaseLLMClient):
    """LLM client for Ollama's native /api/generate endpoint."""

    def __init__(self, url: str, model: str) -> None:
        """Initialize Ollama client.

        Args:
            url: Full endpoint URL (e.g. http://localhost:11434/api/generate).
            model: Model name (e.g. "llama3.2").
        """
        self._url = url
        self._model = model
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        client = await self._get_client()
        response = await client.post(
            self._url,
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                # T-LLM-BACKEND-001: thinking models (e.g. qwen3.5) burn the
                # entire num_predict budget on hidden reasoning and return an
                # empty `response` field unless thinking is disabled.
                "think": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        content = str(data.get("response", "") or "")
        if not content:
            # Defense-in-depth: some ollama builds/models route output to a
            # separate `thinking` field even with think disabled.
            content = str(data.get("thinking", "") or "")
        return _clean_content(content)

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Fallback to single-prompt for Ollama — concat messages into prompt."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts)
        return await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
