"""Pre-Compression Extraction - Capture memories before context squeeze."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memini_ai.extractor import MemoryExtractor
    from memini_ai.memory.system import MemorySystem

import httpx

from memini_ai.config import get_config


@dataclass
class PrecompressResult:
    """Result of pre-compression extraction."""

    memories_created: list[str]
    context_captured: str
    extraction_count: int


class PrecompressExtractor:
    """Extract memories before context window compaction.

    Features:
    - Register callback for OpenCode context events
    - Capture current conversation context
    - Trigger extraction before context squeeze
    - Optional (PRECOMPRESS=false disables)
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self._memory_system = memory_system
        self._extractor = extractor
        self._config = get_config()
        self._context_buffer: list[dict[str, str]] = []
        self._enabled: bool | None = None
        self._http_client: httpx.AsyncClient | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if pre-compression extraction is enabled."""
        if self._enabled is None:
            self._enabled = self._config.precompress_enabled
        return self._enabled

    def register_context_event_handler(self, callback: Callable[..., object]) -> None:
        """Register callback to be called before context compaction.

        Args:
            callback: Async function that receives context dict with
                     'content', 'usage', 'remaining' keys.
        """
        # This would integrate with OpenCode's event system
        # Implementation depends on OpenCode event API documentation
        pass

    async def capture_and_extract(
        self,
        context_content: str,
    ) -> PrecompressResult:
        """Capture context and trigger extraction.

        Args:
            context_content: Current context window content.

        Returns:
            PrecompressResult with created memory IDs.
        """
        if not self.is_enabled:
            return PrecompressResult(
                memories_created=[],
                context_captured="",
                extraction_count=0,
            )

        # Store context in buffer
        self._context_buffer.append(
            {
                "role": "system",
                "content": context_content,
            }
        )

        # Build conversation text from buffer
        conversation = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in self._context_buffer
        )

        # Trigger extraction using existing extractor
        if self._extractor is not None:
            memory_ids = await self._extractor.trigger_extraction(conversation)

            return PrecompressResult(
                memories_created=memory_ids,
                context_captured=context_content[:500],  # Truncate for logging
                extraction_count=len(memory_ids),
            )

        return PrecompressResult(
            memories_created=[],
            context_captured="",
            extraction_count=0,
        )

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn to the buffer.

        Args:
            role: "user" or "agent"
            content: Turn content
        """
        self._context_buffer.append({"role": role, "content": content})
        # Keep buffer bounded at 20 turns
        if len(self._context_buffer) > 20:
            self._context_buffer = self._context_buffer[-20:]

    def clear_buffer(self) -> None:
        """Clear the conversation buffer."""
        self._context_buffer.clear()

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
