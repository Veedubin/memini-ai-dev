"""Tests for OllamaClient (T-LLM-BACKEND-001).

Thinking models served by ollama's /api/generate can return an empty
`response` field with all output in a separate `thinking` field, or inline
<think> tags. The client must disable thinking in the payload AND degrade
gracefully when either artifact appears anyway.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memini_ai.llm.ollama import OllamaClient


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


class _FakeHttpClient:
    """Captures the posted payload and replays a canned JSON body."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.payload: dict[str, Any] | None = None

    async def post(
        self, url: str, *, json: dict[str, Any] | None = None
    ) -> _FakeResponse:
        self.payload = json
        return _FakeResponse(self.data)


def _client_with(data: dict[str, Any]) -> tuple[OllamaClient, _FakeHttpClient]:
    client = OllamaClient(url="http://localhost:11434/api/generate", model="m")
    fake = _FakeHttpClient(data)
    client._http_client = fake  # noqa: SLF001  (bypass lazy real httpx)
    return client, fake


def test_payload_disables_thinking() -> None:
    client, fake = _client_with({"response": "ok"})
    out = asyncio.run(client.generate("prompt"))
    assert out == "ok"
    assert fake.payload is not None
    assert fake.payload["think"] is False
    assert fake.payload["stream"] is False
    assert fake.payload["options"]["num_predict"] == 2048


def test_response_field_preferred_over_thinking() -> None:
    client, _ = _client_with({"response": "answer", "thinking": "reasoning"})
    assert asyncio.run(client.generate("p")) == "answer"


def test_empty_response_falls_back_to_thinking_field() -> None:
    # The exact qwen3.5:9b failure mode from Session 65 diagnosis.
    client, _ = _client_with({"response": "", "thinking": "reasoned summary"})
    assert asyncio.run(client.generate("p")) == "reasoned summary"


def test_missing_response_with_thinking_only() -> None:
    client, _ = _client_with({"thinking": "fallback text"})
    assert asyncio.run(client.generate("p")) == "fallback text"


def test_inline_think_tags_stripped() -> None:
    client, _ = _client_with({"response": "<think>internal</think>Answer here"})
    assert asyncio.run(client.generate("p")) == "Answer here"


def test_multiline_think_block_stripped() -> None:
    client, _ = _client_with({"response": "<think>line1\nline2</think>\nFinal"})
    assert asyncio.run(client.generate("p")) == "Final"


def test_generate_chat_concatenates_messages() -> None:
    client, fake = _client_with({"response": "done"})
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    out = asyncio.run(client.generate_chat(messages))
    assert out == "done"
    assert fake.payload is not None
    assert "system: be brief" in fake.payload["prompt"]
    assert fake.payload["think"] is False
