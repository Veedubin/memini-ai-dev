"""OpenAI-compatible client for /v1/chat/completions."""

from __future__ import annotations

import httpx

from memini_ai.llm.base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    """LLM client for OpenAI-compatible APIs (Ollama Cloud, OpenAI, OpenRouter)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        """Initialize OpenAI-compatible client.

        Args:
            base_url: Base URL without trailing endpoint (e.g. https://ollama.com/v1).
            model: Model name (e.g. "ministral-3:14b-cloud").
            api_key: Optional Bearer token. If omitted, no Authorization header is sent.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._http_client = httpx.AsyncClient(
                timeout=120.0,
                headers=headers,
            )
        return self._http_client

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
