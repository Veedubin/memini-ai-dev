"""Auto-Extract - Automatic memory extraction after conversation turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from memini_ai.config import get_config
from memini_ai.memory.schema import MemoryEntry, MemorySourceType
from memini_ai.utils.logger import logger

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem

EXTRACTION_PROMPT = """
Analyze the following conversation and extract:

1. **Facts**: Objective information stated (preferences, constraints, requirements)
2. **Decisions**: Choices made with rationale
3. **Patterns**: Recurring behaviors or approaches
4. **Preferences**: User/agent stated preferences

Return JSON with keys: "facts", "decisions", "patterns", "preferences"
Each item should have "text" and "confidence" (0.0-1.0).

Conversation:
{conversation}

Return valid JSON only, no markdown or explanation.
"""


@dataclass
class ExtractedMemory:
    """Result of memory extraction."""

    text: str
    category: str  # "fact", "decision", "pattern", "preference"
    confidence: float
    source_memory_id: str | None = None


class ConversationTurnTracker:
    """Tracks conversation turns for auto-extract triggering."""

    def __init__(self, turns_before_extract: int = 5) -> None:
        self._turns_before_extract = turns_before_extract
        self._turn_count = 0
        self._conversation_buffer: list[dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn.

        Args:
            role: "user" or "agent"
            content: Turn content
        """
        self._turn_count += 1
        self._conversation_buffer.append({"role": role, "content": content})

    @property
    def should_extract(self) -> bool:
        """Check if extraction should trigger."""
        return self._turn_count >= self._turns_before_extract

    def get_conversation_text(self) -> str:
        """Get conversation as text for extraction."""
        return "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in self._conversation_buffer
        )

    def reset(self) -> None:
        """Reset turn counter after extraction."""
        self._turn_count = 0
        self._conversation_buffer.clear()


class MemoryExtractor:
    """Automatic memory extraction using LLM.

    Features:
    - Fire LLM pass after N conversation turns
    - Extract facts, decisions, patterns, preferences
    - Store automatically via existing add_memory flow
    - Optional (MEMINI_AUTO_EXTRACT=false disables)
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        llm_url: str | None = None,
    ) -> None:
        self._memory_system = memory_system
        self._config = get_config()
        self._llm_url = (
            llm_url or self._config.llm_url or "http://localhost:11434/api/generate"
        )
        self._llm_model = self._config.llm_model or "llama3.2"
        self._turn_tracker = ConversationTurnTracker(
            turns_before_extract=self._config.auto_extract_turns
        )
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for LLM calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def is_enabled(self) -> bool:
        """Check if auto-extract is enabled."""
        return self._config.auto_extract_enabled

    async def record_turn(self, role: str, content: str) -> None:
        """Record a conversation turn and trigger extraction if needed.

        Args:
            role: "user" or "agent"
            content: Turn content
        """
        if not self.is_enabled:
            return

        self._turn_tracker.add_turn(role, content)

        if self._turn_tracker.should_extract:
            await self._extract_and_store()
            self._turn_tracker.reset()

    async def _extract_and_store(self) -> list[str]:
        """Extract memories from conversation and store.

        Returns:
            List of created memory IDs.
        """
        if self._memory_system is None:
            return []

        conversation = self._turn_tracker.get_conversation_text()

        # Call LLM for extraction
        extracted = await self._extract_memories(conversation)

        # Store each extracted memory
        memory_ids = []
        for item in extracted:
            metadata = {
                "category": item.category,
                "confidence": item.confidence,
                "extracted_at": datetime.utcnow().isoformat(),
            }
            entry = MemoryEntry(
                text=item.text,
                sourceType=MemorySourceType.session,
                metadataJson=json.dumps(metadata),
            )
            try:
                memory_id = await self._memory_system.add_memory(entry)
                memory_ids.append(memory_id)
            except ValueError:
                # Duplicate - skip
                pass

        return memory_ids

    async def _extract_memories(self, conversation: str) -> list[ExtractedMemory]:
        """Call LLM to extract memories from conversation.

        Args:
            conversation: Conversation text.

        Returns:
            List of ExtractedMemory objects.
        """
        try:
            client = await self._get_http_client()

            response = await client.post(
                self._llm_url,
                json={
                    "model": self._llm_model,
                    "prompt": EXTRACTION_PROMPT.format(conversation=conversation),
                    "stream": False,
                },
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                return self._parse_extraction(text)
        except Exception:
            logger.warning("extraction_llm_call_failed", error=str(Exception))

        return []

    def _parse_extraction(self, raw_json: str) -> list[ExtractedMemory]:
        """Parse LLM extraction response.

        Args:
            raw_json: Raw JSON string from LLM.

        Returns:
            List of ExtractedMemory objects.
        """
        try:
            data = json.loads(raw_json)
            memories = []

            for category in ["facts", "decisions", "patterns", "preferences"]:
                for item in data.get(category, []):
                    text = item.get("text", "")
                    if text:
                        # Normalize category (remove trailing 's')
                        normalized_category = (
                            category[:-1] if category.endswith("s") else category
                        )
                        memories.append(
                            ExtractedMemory(
                                text=text,
                                category=normalized_category,
                                confidence=item.get("confidence", 0.5),
                            )
                        )

            return memories
        except json.JSONDecodeError:
            logger.warning("extraction_parse_failed", raw_length=len(raw_json))
            return []

    async def trigger_extraction(self, conversation: str | None = None) -> list[str]:
        """Manually trigger extraction.

        Args:
            conversation: Optional conversation text (uses buffer if not provided).

        Returns:
            List of created memory IDs.
        """
        if not self.is_enabled:
            return []

        if conversation:
            # Use provided conversation instead of buffer
            extracted = await self._extract_memories(conversation)
            memory_ids = []

            for item in extracted:
                metadata = {
                    "category": item.category,
                    "confidence": item.confidence,
                    "extracted_at": datetime.utcnow().isoformat(),
                }
                entry = MemoryEntry(
                    text=item.text,
                    sourceType=MemorySourceType.session,
                    metadataJson=json.dumps(metadata),
                )
                try:
                    if self._memory_system:
                        memory_id = await self._memory_system.add_memory(entry)
                        memory_ids.append(memory_id)
                except ValueError:
                    pass

            return memory_ids

        # Use conversation buffer
        conv_text = self._turn_tracker.get_conversation_text()
        if conv_text:
            result = await self._extract_and_store()
            self._turn_tracker.reset()
            return result

        return []
