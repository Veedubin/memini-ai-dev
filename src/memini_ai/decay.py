"""Memory Decay and Consolidation Engine for Memini-ai v3.0.

This module implements Phase 4A: Memory decay over time if not referenced,
and periodic consolidation of similar memories. All features are opt-in via
decay_enabled config (default false).

Features:
- DecayEngine: Applies decay to memories based on half-life and access patterns
- ConsolidationEngine: Finds and merges similar memories above similarity threshold
- Background tasks: Automatic decay loop and consolidation on configurable interval
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from memini_ai.config import get_config
from memini_ai.memory.schema import MemoryEntry, TrustLevel
from memini_ai.utils.logger import logger

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem

# Decay constants
DECAY_BASE_HALF_LIFE_DAYS = 90  # Default half-life in days
DECAY_MIN_RATE = 0.1  # Minimum decay rate multiplier
DECAY_MAX_RATE = 10.0  # Maximum decay rate multiplier
DECAY_DEFAULT_RATE = 1.0  # Normal decay rate multiplier

# Consolidation constants
DEFAULT_SIMILARITY_THRESHOLD = 0.92
MIN_CONSOLIDATION_SIMILARITY = 0.70

# Archive threshold for faded memories
FADE_THRESHOLD = 0.15  # Trust score below which memory is considered faded


@dataclass
class DecayStats:
    """Statistics for decay engine operation."""

    memories_processed: int = 0
    memories_decayed: int = 0
    memories_archived: int = 0
    last_run: datetime | None = None
    next_scheduled_run: datetime | None = None
    total_decay_events: int = 0


@dataclass
class ConsolidationStats:
    """Statistics for consolidation engine operation."""

    pairs_found: int = 0
    pairs_merged: int = 0
    memories_consolidated: int = 0
    last_run: datetime | None = None
    next_scheduled_run: datetime | None = None


@dataclass
class MemoryDecayInfo:
    """Decay information for a single memory."""

    memory_id: str
    text_preview: str  # First 50 chars of text
    current_decay_rate: float
    trust_score: float
    trust_level: TrustLevel
    last_accessed: datetime | None
    access_count: int
    days_until_archive: float | None
    is_fading: bool


@dataclass
class ConsolidationCandidate:
    """A pair of memories that could be consolidated."""

    memory_a: MemoryEntry
    memory_b: MemoryEntry
    similarity: float
    combined_text: str
    suggested_action: str = "merge"  # "merge", "supersede", "keep_separate"


class DecayEngine:
    """Memory decay engine with configurable half-life and access patterns.

    Decay applies to all memories that haven't been accessed within the decay
    interval. Memories with higher decay_rate decay faster. Memories with lower
    decay_rate (e.g., important memories) decay slower.

    The decay formula uses exponential decay based on half-life:
        decay_factor = 0.5 ^ (days_elapsed / half_life_days)

    This means a memory loses half its remaining trust every half_life_days
    if not accessed.
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        decay_rate: float = DECAY_DEFAULT_RATE,
    ) -> None:
        """Initialize DecayEngine.

        Args:
            memory_system: Optional MemorySystem instance for DB operations.
            decay_rate: Default decay rate multiplier (1.0 = normal).
        """
        self._memory_system = memory_system
        self._decay_rate = decay_rate
        self._decay_task: asyncio.Task[None] | None = None
        self._running = False
        self._stats = DecayStats()

    @property
    def is_enabled(self) -> bool:
        """Check if decay engine is enabled via config."""
        config = get_config()
        return config.decay_enabled

    @property
    def stats(self) -> DecayStats:
        """Get current decay statistics."""
        return self._stats

    def calculate_decay(
        self,
        trust_score: float,
        decay_rate: float,
        days_elapsed: float,
        half_life_days: int,
    ) -> float:
        """Calculate new trust score after decay.

        Uses exponential decay formula:
            new_score = old_score * 0.5 ^ (days_elapsed * decay_rate / half_life)

        Args:
            trust_score: Current trust score.
            decay_rate: Decay rate multiplier for this memory.
            days_elapsed: Number of days since last access.
            half_life_days: Half-life in days for base decay.

        Returns:
            New trust score after decay application.
        """
        if days_elapsed <= 0:
            return trust_score

        # Effective decay days = actual days * decay_rate
        effective_days = days_elapsed * decay_rate
        # Half-life formula: score * 0.5^(effective_days/half_life)
        decay_factor = 0.5 ** (effective_days / half_life_days)
        return trust_score * decay_factor

    def calculate_days_until_archive(
        self,
        trust_score: float,
        decay_rate: float,
        half_life_days: int,
        archive_threshold: float = FADE_THRESHOLD,
    ) -> float | None:
        """Calculate estimated days until memory reaches archive threshold.

        Args:
            trust_score: Current trust score.
            decay_rate: Decay rate multiplier.
            half_life_days: Half-life in days.
            archive_threshold: Threshold below which memory is archived.

        Returns:
            Estimated days until archive, or None if already below threshold.
        """
        if trust_score <= archive_threshold:
            return None

        # Solve for days in: archive_threshold = trust_score * 0.5^(days * rate / half_life)
        # log(archive_threshold/trust_score) = (days * rate / half_life) * log(0.5)
        # days = log(archive_threshold/trust_score) * half_life / (rate * log(0.5))
        ratio = archive_threshold / trust_score
        if ratio <= 0:
            return None

        days = math.log(ratio) * (half_life_days / (decay_rate * math.log(0.5)))
        return max(0, days)

    async def apply_decay(self, memory: MemoryEntry) -> float | None:
        """Apply decay to a single memory.

        Args:
            memory: MemoryEntry to apply decay to.

        Returns:
            New trust score after decay, or None if skipped/error.
        """
        if not self.is_enabled:
            return None

        config = get_config()
        half_life_days = config.decay_half_life_days

        # Calculate days since last access
        last_accessed = memory.last_accessed or memory.timestamp
        days_elapsed = (datetime.utcnow() - last_accessed).total_seconds() / 86400

        # Apply decay
        memory_decay_rate = memory.decay_rate if hasattr(memory, 'decay_rate') else self._decay_rate
        new_score = self.calculate_decay(
            memory.trust_score,
            memory_decay_rate,
            days_elapsed,
            half_life_days,
        )

        # Clamp to valid range
        new_score = max(0.0, min(1.0, new_score))

        return new_score

    async def process_memories(self) -> dict[str, Any]:
        """Process all active memories, applying decay to each.

        Returns:
            Dictionary with processed count, decayed count, archived count.
        """
        if not self.is_enabled or self._memory_system is None:
            return {"processed": 0, "decayed": 0, "archived": 0}

        config = get_config()
        processed = 0
        decayed = 0
        archived = 0

        try:
            all_memories = await self._memory_system.list_memories()

            for memory in all_memories:
                if memory.is_archived:
                    continue

                processed += 1
                old_score = memory.trust_score

                # Apply decay
                new_score = await self.apply_decay(memory)

                if new_score is not None and new_score != old_score:
                    decayed += 1
                    memory.trust_score = new_score

                    # Check for archive threshold
                    if new_score < config.trust_threshold_archive:
                        memory.is_archived = True
                        archived += 1

                    # Persist update
                    await self._update_memory_decay(
                        memory.id,
                        new_score,
                        memory.is_archived,
                    )

            # Update stats
            self._stats.memories_processed = processed
            self._stats.memories_decayed = decayed
            self._stats.memories_archived += archived
            self._stats.last_run = datetime.utcnow()
            self._stats.total_decay_events += decayed

        except Exception as e:
            logger.warning("decay_process_error", error=str(e))

        return {
            "processed": processed,
            "decayed": decayed,
            "archived": archived,
        }

    def _update_memory_decay(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        """Update decay fields in database (sync wrapper).

        Args:
            memory_id: Memory ID.
            trust_score: New trust score.
            is_archived: New archived status.
        """
        from memini_ai.memory.database import _client_cache, _get_collection_name

        config = get_config()
        collection_name = _get_collection_name(config.embedding_dim)

        if config.qdrant_url not in _client_cache:
            return

        client = _client_cache[config.qdrant_url]

        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=collection_name,
                payload={
                    "trustScore": trust_score,
                    "isArchived": is_archived,
                },
                points=[memory_id],
            )

    async def get_decay_status(self) -> dict[str, Any]:
        """Get decay status and statistics for all memories.

        Returns:
            Dictionary with decay stats and list of fading memories.
        """
        if not self.is_enabled:
            return {
                "enabled": False,
                "message": "Decay engine is disabled",
            }

        config = get_config()
        fading_memories: list[MemoryDecayInfo] = []

        if self._memory_system is not None:
            all_memories = await self._memory_system.list_memories()

            for memory in all_memories:
                if memory.is_archived:
                    continue

                last_accessed = memory.last_accessed or memory.timestamp
                days_elapsed = (datetime.utcnow() - last_accessed).total_seconds() / 86400

                # Determine trust level
                if memory.trust_score < 0.2:
                    level = TrustLevel.ARCHIVED
                elif memory.trust_score < 0.4:
                    level = TrustLevel.LOW
                elif memory.trust_score < 0.7:
                    level = TrustLevel.MEDIUM
                elif memory.trust_score < 0.8:
                    level = TrustLevel.HIGH
                else:
                    level = TrustLevel.PROMOTED

                # Calculate days until archive
                memory_decay_rate = memory.decay_rate if hasattr(memory, 'decay_rate') else self._decay_rate
                days_until = self.calculate_days_until_archive(
                    memory.trust_score,
                    memory_decay_rate,
                    config.decay_half_life_days,
                )

                # Check if fading
                is_fading = (
                    days_until is not None and days_until < 30
                ) or memory.trust_score < 0.3

                if is_fading:
                    fading_memories.append(MemoryDecayInfo(
                        memory_id=memory.id,
                        text_preview=memory.text[:50] + ("..." if len(memory.text) > 50 else ""),
                        current_decay_rate=memory_decay_rate,
                        trust_score=memory.trust_score,
                        trust_level=level,
                        last_accessed=last_accessed,
                        access_count=memory.retrieval_count if hasattr(memory, 'retrieval_count') else 0,
                        days_until_archive=days_until,
                        is_fading=is_fading,
                    ))

        return {
            "enabled": True,
            "half_life_days": config.decay_half_life_days,
            "stats": {
                "memories_processed": self._stats.memories_processed,
                "memories_decayed": self._stats.memories_decayed,
                "memories_archived": self._stats.memories_archived,
                "total_decay_events": self._stats.total_decay_events,
                "last_run": self._stats.last_run.isoformat() if self._stats.last_run else None,
            },
            "fading_count": len(fading_memories),
            "fading_memories": [
                {
                    "memory_id": m.memory_id,
                    "text_preview": m.text_preview,
                    "decay_rate": m.current_decay_rate,
                    "trust_score": m.trust_score,
                    "trust_level": m.trust_level.value,
                    "last_accessed": m.last_accessed.isoformat() if m.last_accessed else None,
                    "access_count": m.access_count,
                    "days_until_archive": m.days_until_archive,
                    "is_fading": m.is_fading,
                }
                for m in fading_memories[:20]  # Limit to 20
            ],
        }

    async def get_decay_info(self, memory_id: str) -> dict[str, Any] | None:
        """Get decay information for a specific memory.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            Dictionary with decay info, or None if not found.
        """
        if not self.is_enabled:
            return None

        if self._memory_system is None:
            return None

        memory = await self._memory_system.get_memory(memory_id)
        if memory is None:
            return None

        config = get_config()
        last_accessed = memory.last_accessed or memory.timestamp
        days_elapsed = (datetime.utcnow() - last_accessed).total_seconds() / 86400

        memory_decay_rate = memory.decay_rate if hasattr(memory, 'decay_rate') else self._decay_rate
        days_until = self.calculate_days_until_archive(
            memory.trust_score,
            memory_decay_rate,
            config.decay_half_life_days,
        )

        return {
            "memory_id": memory_id,
            "text_preview": memory.text[:50] + ("..." if len(memory.text) > 50 else ""),
            "current_decay_rate": memory_decay_rate,
            "effective_decay_rate": memory_decay_rate,
            "trust_score": memory.trust_score,
            "last_accessed": last_accessed.isoformat(),
            "days_elapsed": round(days_elapsed, 2),
            "days_until_archive": round(days_until, 2) if days_until else None,
            "half_life_days": config.decay_half_life_days,
            "is_fading": days_until is not None and days_until < 30,
        }


class ConsolidationEngine:
    """Consolidation engine for finding and merging similar memories.

    Consolidation identifies memory pairs with high similarity and either
    merges them into a single memory or marks one as superseding the other.
    This reduces redundancy and keeps memory store efficient.
    """

    def __init__(self, memory_system: MemorySystem | None = None) -> None:
        """Initialize ConsolidationEngine.

        Args:
            memory_system: Optional MemorySystem instance for DB operations.
        """
        self._memory_system = memory_system
        self._consolidation_task: asyncio.Task[None] | None = None
        self._running = False
        self._stats = ConsolidationStats()

    @property
    def is_enabled(self) -> bool:
        """Check if decay engine (which includes consolidation) is enabled."""
        config = get_config()
        return config.decay_enabled

    @property
    def stats(self) -> ConsolidationStats:
        """Get current consolidation statistics."""
        return self._stats

    async def find_similar_pairs(
        self,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[ConsolidationCandidate]:
        """Find all memory pairs above similarity threshold.

        Args:
            threshold: Minimum similarity score (0.0 to 1.0). Default 0.92.

        Returns:
            List of ConsolidationCandidate pairs.
        """
        if self._memory_system is None:
            return []

        candidates: list[ConsolidationCandidate] = []
        config = get_config()

        try:
            all_memories = await self._memory_system.list_memories()

            # Compare each pair (O(n^2) but acceptable for memory stores)
            for i, mem_a in enumerate(all_memories):
                if mem_a.is_archived:
                    continue

                for mem_b in all_memories[i + 1:]:
                    if mem_b.is_archived:
                        continue

                    # Calculate similarity based on vector cosine similarity
                    # For now, use a simplified text-based similarity
                    similarity = self._calculate_text_similarity(
                        mem_a.text,
                        mem_b.text,
                    )

                    if similarity >= threshold:
                        combined = self._combine_texts(mem_a.text, mem_b.text)
                        candidates.append(ConsolidationCandidate(
                            memory_a=mem_a,
                            memory_b=mem_b,
                            similarity=similarity,
                            combined_text=combined,
                            suggested_action="merge",
                        ))

        except Exception as e:
            logger.warning("consolidation_find_error", error=str(e))

        self._stats.pairs_found = len(candidates)
        return candidates

    def _calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity between two text strings.

        Uses Jaccard similarity on word tokens as a simple but effective
        approach. In production, this would use vector similarity from embeddings.

        Args:
            text_a: First text string.
            text_b: Second text string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Simple word-based Jaccard similarity
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _combine_texts(self, text_a: str, text_b: str) -> str:
        """Combine two memory texts into a single merged text.

        Uses simple concatenation with separator. In production, could use
        LLM to create a proper summary.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Combined text string.
        """
        # Simple merge: both texts concatenated
        # In future, could use LLM summarization
        separator = " [MERGED] "
        return text_a + separator + text_b

    async def consolidate_pair(
        self,
        candidate: ConsolidationCandidate,
    ) -> bool:
        """Consolidate a memory pair (merge or supersede).

        Args:
            candidate: ConsolidationCandidate with pair info.

        Returns:
            True if consolidation successful, False otherwise.
        """
        if self._memory_system is None:
            return False

        try:
            # Use the one with higher trust as the survivor
            if candidate.memory_a.trust_score >= candidate.memory_b.trust_score:
                survivor = candidate.memory_a
                superseded = candidate.memory_b
            else:
                survivor = candidate.memory_b
                superseded = candidate.memory_a

            # Update survivor with combined text
            survivor.text = candidate.combined_text

            # Update survivor's trust (average of both)
            new_trust = (survivor.trust_score + candidate.memory_b.trust_score) / 2
            survivor.trust_score = max(0.0, min(1.0, new_trust))

            # Mark superseded as archived
            superseded.is_archived = True

            # Persist changes
            await self._update_memory_on_consolidation(
                survivor.id,
                survivor.text,
                survivor.trust_score,
            )
            await self._archive_memory(superseded.id)

            self._stats.pairs_merged += 1
            self._stats.memories_consolidated += 1

            return True

        except Exception as e:
            logger.warning("consolidation_pair_error", memory_a=candidate.memory_a.id, error=str(e))
            return False

    def _update_memory_on_consolidation(
        self,
        memory_id: str,
        text: str,
        trust_score: float,
    ) -> None:
        """Update memory after consolidation."""
        from memini_ai.memory.database import _client_cache, _get_collection_name

        config = get_config()
        collection_name = _get_collection_name(config.embedding_dim)

        if config.qdrant_url not in _client_cache:
            return

        client = _client_cache[config.qdrant_url]

        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=collection_name,
                payload={
                    "text": text,
                    "trustScore": trust_score,
                },
                points=[memory_id],
            )

    def _archive_memory(self, memory_id: str) -> None:
        """Archive a memory (mark as deleted/archived)."""
        from memini_ai.memory.database import _client_cache, _get_collection_name

        config = get_config()
        collection_name = _get_collection_name(config.embedding_dim)

        if config.qdrant_url not in _client_cache:
            return

        client = _client_cache[config.qdrant_url]

        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=collection_name,
                payload={"isArchived": True},
                points=[memory_id],
            )

    async def run_consolidation(self) -> dict[str, Any]:
        """Run a full consolidation cycle.

        Finds similar pairs above threshold and consolidates them.

        Returns:
            Dictionary with consolidation results.
        """
        if not self.is_enabled:
            return {"consolidated": 0, "pairs_found": 0, "pairs_merged": 0}

        config = get_config()
        candidates = await self.find_similar_pairs(config.consolidation_similarity_threshold)

        merged = 0
        for candidate in candidates:
            success = await self.consolidate_pair(candidate)
            if success:
                merged += 1

        self._stats.last_run = datetime.utcnow()
        self._stats.pairs_merged = merged

        return {
            "pairs_found": len(candidates),
            "pairs_merged": merged,
            "memories_consolidated": merged,
        }

    async def list_fading_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        """List memories that are approaching archive threshold.

        Args:
            limit: Maximum number of results (default 20).

        Returns:
            List of fading memory info dictionaries.
        """
        if not self.is_enabled or self._memory_system is None:
            return []

        fading: list[dict[str, Any]] = []
        config = get_config()

        try:
            all_memories = await self._memory_system.list_memories()
            decay_engine = DecayEngine(self._memory_system)

            for memory in all_memories:
                if memory.is_archived:
                    continue

                # Calculate if fading
                memory_decay_rate = memory.decay_rate if hasattr(memory, 'decay_rate') else 1.0
                days_until = decay_engine.calculate_days_until_archive(
                    memory.trust_score,
                    memory_decay_rate,
                    config.decay_half_life_days,
                    FADE_THRESHOLD,
                )

                # Consider fading if days_until < 30 or trust < 0.3
                if (days_until is not None and days_until < 30) or memory.trust_score < 0.3:
                    last_accessed = memory.last_accessed or memory.timestamp

                    fading.append({
                        "memory_id": memory.id,
                        "text_preview": memory.text[:50] + ("..." if len(memory.text) > 50 else ""),
                        "trust_score": memory.trust_score,
                        "decay_rate": memory_decay_rate,
                        "last_accessed": last_accessed.isoformat(),
                        "days_until_archive": round(days_until, 2) if days_until else 0,
                        "access_count": memory.retrieval_count if hasattr(memory, 'retrieval_count') else 0,
                    })

            # Sort by days_until (most urgent first)
            fading.sort(key=lambda x: x["days_until_archive"] or 0)

        except Exception as e:
            logger.warning("fading_memories_error", error=str(e))

        return fading[:limit]


async def adjust_decay_rate(
    memory_system: MemorySystem,
    memory_id: str,
    decay_rate: float,
) -> dict[str, Any]:
    """Adjust the decay rate for a specific memory.

    Args:
        memory_system: MemorySystem instance.
        memory_id: ID of the memory to adjust.
        decay_rate: New decay rate (0.1 to 10.0).

    Returns:
        Dictionary with success status and updated decay info.
    """
    # Clamp decay rate to valid range
    decay_rate = max(DECAY_MIN_RATE, min(DECAY_MAX_RATE, decay_rate))

    memory = await memory_system.get_memory(memory_id)
    if memory is None:
        return {"success": False, "error": "Memory not found"}

    # Update decay rate on memory object
    if hasattr(memory, 'decay_rate'):
        memory.decay_rate = decay_rate
    else:
        # Add decay_rate attribute if not present
        memory.decay_rate = decay_rate

    # Persist to database
    from memini_ai.config import get_config
    from memini_ai.memory.database import _client_cache, _get_collection_name

    config = get_config()
    collection_name = _get_collection_name(config.embedding_dim)

    if config.qdrant_url in _client_cache:
        client = _client_cache[config.qdrant_url]
        with contextlib.suppress(Exception):
            client.set_payload(
                collection_name=collection_name,
                payload={"decayRate": decay_rate},
                points=[memory_id],
            )

    return {
        "success": True,
        "memory_id": memory_id,
        "decay_rate": decay_rate,
        "message": f"Decay rate adjusted to {decay_rate}",
    }
