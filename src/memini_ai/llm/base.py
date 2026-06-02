"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract interface for LLM clients across providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a single prompt string.

        Args:
            prompt: The prompt to send.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a chat messages array.

        Args:
            messages: List of dicts with "role" and "content" keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        ...
