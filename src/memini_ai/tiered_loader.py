"""Tiered Loading - Token-efficient memory loading at L0/L1/L2 granularity levels.

L0: Project summary (~100 tokens) - auto-injected at session start
L1: Key decisions (~2K tokens) - available for planning tasks
L2: Full memories - on demand via query_memories

This module provides TieredLoader class that generates and caches summaries
from memory entries at different trust thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from memini_ai.config import get_config
from memini_ai.memory.schema import (
    TRUST_THRESHOLD_PROMOTE,
    SummaryTier,
    TieredSummary,
)
from memini_ai.utils.logger import logger

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem


# L0 prompt - generates project summary from high-trust memories
L0_SUMMARY_PROMPT = """Generate a concise project summary (~100 tokens) from the following memories.

Focus on:
- What is this project about?
- Key technologies and patterns
- Important conventions and decisions

Memories:
{memories}

Return a single paragraph summary (under 100 tokens). Be extremely concise."""

# L1 prompt - generates key decisions from promoted memories
L1_SUMMARY_PROMPT = """Generate a structured summary of key decisions and patterns (~2K tokens) from the following memories.

Organize by category:
1. Architecture Decisions - major technical choices
2. Design Patterns - recurring code patterns
3. Conventions - coding standards and practices
4. Important Context - project-specific knowledge

Memories:
{memories}

Be comprehensive but concise. Use bullet points where appropriate."""


@dataclass
class TieredLoadingStats:
    """Statistics for tiered loading operations."""

    l0_generations: int = 0
    l1_generations: int = 0
    l0_cache_hits: int = 0
    l1_cache_hits: int = 0
    last_l0_generated: datetime | None = None
    last_l1_generated: datetime | None = None
    l0_token_count: int = 0
    l1_token_count: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "l0_generations": self.l0_generations,
            "l1_generations": self.l1_generations,
            "l0_cache_hits": self.l0_cache_hits,
            "l1_cache_hits": self.l1_cache_hits,
            "last_l0_generated": (
                self.last_l0_generated.isoformat() if self.last_l0_generated else None
            ),
            "last_l1_generated": (
                self.last_l1_generated.isoformat() if self.last_l1_generated else None
            ),
            "l0_token_count": self.l0_token_count,
            "l1_token_count": self.l1_token_count,
            "errors": self.errors,
        }


class TieredLoader:
    """Token-efficient tiered loading for memories.

    Provides L0/L1/L2 granularity levels:
    - L0: ~100 token project summary (auto-inject at session start)
    - L1: ~2K token key decisions (available for planning tasks)
    - L2: Full memories via query_memories (no changes needed)

    L0 uses high-trust memories (trust >= 0.5).
    L1 uses promoted memories (trust >= 0.8 from Trust Engine).

    Features:
    - Opt-in via TIERED_LOADING env var (default false)
    - TTL-based cache with configurable duration
    - Graceful degradation when disabled
    - Lazy HTTP client initialization
    """

    def __init__(self, memory_system: MemorySystem | None = None) -> None:
        """Initialize TieredLoader.

        Args:
            memory_system: Optional MemorySystem instance for memory operations.
        """
        self._memory_system = memory_system
        self._config = get_config()
        self._enabled: bool | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._l0_cache: TieredSummary | None = None
        self._l1_cache: TieredSummary | None = None
        self._stats = TieredLoadingStats()

    @property
    def is_enabled(self) -> bool:
        """Check if tiered loading is enabled via config."""
        if self._enabled is None:
            self._enabled = self._config.tiered_loading_enabled
        return self._enabled

    @property
    def stats(self) -> TieredLoadingStats:
        """Get tiered loading statistics."""
        return self._stats

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for LLM calls (lazy initialization)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _is_cache_stale(self, cache: TieredSummary | None, ttl: int) -> bool:
        """Check if cached summary is stale based on TTL.

        Args:
            cache: Cached TieredSummary or None.
            ttl: Time-to-live in seconds.

        Returns:
            True if cache is None or expired.
        """
        if cache is None:
            return True

        age = datetime.utcnow() - cache.generated_at
        return age.total_seconds() > ttl or cache.is_stale

    def _is_stale_by_timestamp(self, cache: TieredSummary | None) -> bool:
        """Check if cached summary has is_stale flag set.

        Args:
            cache: Cached TieredSummary or None.

        Returns:
            True if cache is None or is_stale is True.
        """
        if cache is None:
            return True
        return cache.is_stale

    async def _get_memories_above_trust(
        self,
        min_trust: float,
        max_memories: int = 100,
    ) -> list[tuple[str, str]]:
        """Get memory IDs and texts above trust threshold.

        Args:
            min_trust: Minimum trust score threshold.
            max_memories: Maximum number of memories to return.

        Returns:
            List of (memory_id, text) tuples.
        """
        if self._memory_system is None:
            return []

        try:
            all_memories = await self._memory_system.list_memories()

            # Filter by trust threshold and exclude archived
            filtered = [
                m
                for m in all_memories
                if m.trust_score >= min_trust and not m.is_archived
            ]

            # Sort by trust score descending
            filtered.sort(key=lambda m: m.trust_score, reverse=True)

            # Return up to max_memories
            return [(m.id, m.text) for m in filtered[:max_memories]]
        except Exception as e:
            logger.error("tiered_loader_get_memories_failed", error=str(e))
            return []

    async def _generate_summary(
        self,
        memories: list[tuple[str, str]],
        prompt_template: str,
        max_tokens: int,
    ) -> tuple[str, list[str]]:
        """Generate summary from memories using LLM.

        Args:
            memories: List of (memory_id, text) tuples.
            prompt_template: Prompt template with {memories} placeholder.
            max_tokens: Maximum tokens in response.

        Returns:
            Tuple of (summary_content, source_memory_ids).
        """
        if not memories:
            return "", []

        # Format memories for prompt
        memory_texts = [f"- {text}" for _, text in memories]
        memories_str = "\n".join(memory_texts[:50])  # Limit to 50 memories

        prompt = prompt_template.format(memories=memories_str)

        try:
            client = await self._get_http_client()
            llm_url = self._config.llm_url or "http://localhost:11434/api/generate"
            llm_model = self._config.llm_model or "llama3.2"

            response = await client.post(
                llm_url,
                json={
                    "model": llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens * 2},  # Allow some buffer
                },
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "").strip()

                return content, [mid for mid, _ in memories]
        except Exception as e:
            logger.error("tiered_loader_llm_call_failed", error=str(e))
            self._stats.errors += 1

        return "", [mid for mid, _ in memories]

    async def get_tier0(
        self,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get L0 project summary (~100 tokens).

        L0 uses high-trust memories (trust >= 0.5).
        Results are cached based on TIER0_CACHE_TTL (default 1 hour).

        Args:
            force_refresh: Force regeneration even if cache is valid.

        Returns:
            Dictionary with tier, content, token_count, cache_hit, and source_count.
            Returns error info if disabled or unavailable.
        """
        if not self.is_enabled:
            return {
                "tier": "L0",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "Tiered loading is disabled",
            }

        # Check cache first (unless force refresh)
        cache_ttl = self._config.tier0_cache_ttl
        if (
            not force_refresh
            and not self._is_cache_stale(self._l0_cache, cache_ttl)
            and self._l0_cache is not None
        ):
            self._stats.l0_cache_hits += 1
            return {
                "tier": "L0",
                "content": self._l0_cache.content,
                "token_count": self._l0_cache.token_count,
                "cache_hit": True,
                "source_count": len(self._l0_cache.source_memory_ids),
                "generated_at": self._l0_cache.generated_at.isoformat(),
            }

        # Get high-trust memories (trust >= 0.5)
        memories = await self._get_memories_above_trust(0.5, max_memories=50)

        if not memories:
            return {
                "tier": "L0",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "No high-trust memories available",
            }

        # Generate summary
        max_tokens = self._config.tier0_max_tokens or 100
        content, source_ids = await self._generate_summary(
            memories, L0_SUMMARY_PROMPT, max_tokens
        )

        if not content:
            self._stats.errors += 1
            # Return stale cache if available
            if self._l0_cache is not None:
                return {
                    "tier": "L0",
                    "content": self._l0_cache.content,
                    "token_count": self._l0_cache.token_count,
                    "cache_hit": False,
                    "source_count": len(self._l0_cache.source_memory_ids),
                    "generated_at": self._l0_cache.generated_at.isoformat(),
                    "error": "LLM call failed, returning stale cache",
                }
            return {
                "tier": "L0",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "LLM call failed",
            }

        # Create and cache summary
        self._l0_cache = TieredSummary(
            tier=SummaryTier.L0,
            content=content,
            source_memory_ids=source_ids,
            generated_at=datetime.utcnow(),
            token_count=len(content) // 4,
            is_stale=False,
        )

        self._stats.l0_generations += 1
        self._stats.l0_token_count = self._l0_cache.token_count
        self._stats.last_l0_generated = datetime.utcnow()

        return {
            "tier": "L0",
            "content": self._l0_cache.content,
            "token_count": self._l0_cache.token_count,
            "cache_hit": False,
            "source_count": len(source_ids),
            "generated_at": self._l0_cache.generated_at.isoformat(),
        }

    async def get_tier1(
        self,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get L1 key decisions summary (~2K tokens).

        L1 uses promoted memories (trust >= 0.8 from Trust Engine).
        Results are cached based on TIER1_CACHE_TTL (default 2 hours).

        Args:
            force_refresh: Force regeneration even if cache is valid.

        Returns:
            Dictionary with tier, content, token_count, cache_hit, and source_count.
            Returns error info if disabled or unavailable.
        """
        if not self.is_enabled:
            return {
                "tier": "L1",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "Tiered loading is disabled",
            }

        # Check cache first (unless force refresh)
        cache_ttl = self._config.tier1_cache_ttl
        if (
            not force_refresh
            and not self._is_cache_stale(self._l1_cache, cache_ttl)
            and self._l1_cache is not None
        ):
            self._stats.l1_cache_hits += 1
            return {
                "tier": "L1",
                "content": self._l1_cache.content,
                "token_count": self._l1_cache.token_count,
                "cache_hit": True,
                "source_count": len(self._l1_cache.source_memory_ids),
                "generated_at": self._l1_cache.generated_at.isoformat(),
            }

        # Get promoted memories (trust >= 0.8)
        memories = await self._get_memories_above_trust(
            TRUST_THRESHOLD_PROMOTE, max_memories=100
        )

        if not memories:
            return {
                "tier": "L1",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "No promoted memories available (trust >= 0.8 required)",
            }

        # Generate summary
        max_tokens = self._config.tier1_max_tokens or 2000
        content, source_ids = await self._generate_summary(
            memories, L1_SUMMARY_PROMPT, max_tokens
        )

        if not content:
            self._stats.errors += 1
            # Return stale cache if available
            if self._l1_cache is not None:
                return {
                    "tier": "L1",
                    "content": self._l1_cache.content,
                    "token_count": self._l1_cache.token_count,
                    "cache_hit": False,
                    "source_count": len(self._l1_cache.source_memory_ids),
                    "generated_at": self._l1_cache.generated_at.isoformat(),
                    "error": "LLM call failed, returning stale cache",
                }
            return {
                "tier": "L1",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "LLM call failed",
            }

        # Create and cache summary
        self._l1_cache = TieredSummary(
            tier=SummaryTier.L1,
            content=content,
            source_memory_ids=source_ids,
            generated_at=datetime.utcnow(),
            token_count=len(content) // 4,
            is_stale=False,
        )

        self._stats.l1_generations += 1
        self._stats.l1_token_count = self._l1_cache.token_count
        self._stats.last_l1_generated = datetime.utcnow()

        return {
            "tier": "L1",
            "content": self._l1_cache.content,
            "token_count": self._l1_cache.token_count,
            "cache_hit": False,
            "source_count": len(source_ids),
            "generated_at": self._l1_cache.generated_at.isoformat(),
        }

    def invalidate_cache(self, tier: SummaryTier | str | None = None) -> None:
        """Invalidate cached summaries.

        Args:
            tier: Specific tier to invalidate ("L0", "L1", "L2" or SummaryTier).
                  None or "all" invalidates all caches.
        """
        if tier is None or tier == "all":
            self._l0_cache = None
            self._l1_cache = None
            logger.info("tiered_loader_cache_invalidated", tier="all")
        elif tier == "L0" or tier == SummaryTier.L0:
            self._l0_cache = None
            logger.info("tiered_loader_cache_invalidated", tier="L0")
        elif tier == "L1" or tier == SummaryTier.L1:
            self._l1_cache = None
            logger.info("tiered_loader_cache_invalidated", tier="L1")

    def mark_stale(self, tier: SummaryTier | str | None = None) -> None:
        """Mark cached summaries as stale.

        Args:
            tier: Specific tier to mark ("L0", "L1", "L2" or SummaryTier).
                  None or "all" marks all caches as stale.
        """
        if tier is None or tier == "all":
            if self._l0_cache is not None:
                self._l0_cache.is_stale = True
            if self._l1_cache is not None:
                self._l1_cache.is_stale = True
            logger.info("tiered_loader_marked_stale", tier="all")
        elif tier == "L0" or tier == SummaryTier.L0:
            if self._l0_cache is not None:
                self._l0_cache.is_stale = True
            logger.info("tiered_loader_marked_stale", tier="L0")
        elif tier == "L1" or tier == SummaryTier.L1:
            if self._l1_cache is not None:
                self._l1_cache.is_stale = True
            logger.info("tiered_loader_marked_stale", tier="L1")

    async def get_summary_status(self) -> dict[str, Any]:
        """Get current status of cached summaries.

        Returns:
            Dictionary with L0 and L1 cache status including age, staleness, and token counts.
        """
        now = datetime.utcnow()
        l0_status: dict[str, Any] = {"cached": False}
        l1_status: dict[str, Any] = {"cached": False}

        if self._l0_cache is not None:
            age = (now - self._l0_cache.generated_at).total_seconds()
            l0_status = {
                "cached": True,
                "age_seconds": age,
                "is_stale": self._l0_cache.is_stale,
                "token_count": self._l0_cache.token_count,
                "source_count": len(self._l0_cache.source_memory_ids),
                "generated_at": self._l0_cache.generated_at.isoformat(),
            }

        if self._l1_cache is not None:
            age = (now - self._l1_cache.generated_at).total_seconds()
            l1_status = {
                "cached": True,
                "age_seconds": age,
                "is_stale": self._l1_cache.is_stale,
                "token_count": self._l1_cache.token_count,
                "source_count": len(self._l1_cache.source_memory_ids),
                "generated_at": self._l1_cache.generated_at.isoformat(),
            }

        return {
            "enabled": self.is_enabled,
            "l0": l0_status,
            "l1": l1_status,
            "stats": self._stats.to_dict(),
        }
