"""RRFDatabase — wraps two VectorDatabase backends and fuses query results via RRF.

v1.0.0 embedded-pgembed architecture (design section 6.2).

The ``RRFDatabase`` is a drop-in ``VectorDatabase`` subclass that composes
two underlying backends:

* **primary** — the authoritative store. All writes go here. If the
  primary fails, the error propagates to the caller (the primary is
  treated as the source of truth).
* **secondary** — an optional team/shared server used only to broaden
  read recall. Query operations fan out to both backends in parallel
  via ``asyncio.create_task`` and the ranked result lists are fused
  using Reciprocal Rank Fusion (``rrf_with_limit``).

Design properties (see design doc section 6.3 for the tie-breaking
matrix):

* Writes are delegated to the primary only. After a successful primary
  write, a fire-and-forget background task (Q3) also writes to the
  secondary. Failures in that background task are counted in
  ``self._team_write_failures`` and logged, never raised.
* Queries fan out to both backends concurrently. The primary task is
  awaited unconditionally — if it raises, the exception propagates.
  The secondary task is best-effort: on failure we log a warning and
  return primary-only results (graceful degradation).
* Tie-breaking on content conflicts: primary wins. We build a dict
  keyed by memory id populated from primary first, then use
  ``dict.setdefault`` for secondary entries so secondary only fills
  gaps the primary did not cover.
* No ``asyncio.gather`` is used for the fan-out — we want to await the
  primary independently of the secondary so a slow/dead secondary
  cannot delay primary results.
* No locks, no singleton state, no class-level mutable state.
"""

from __future__ import annotations

import asyncio
from typing import Any

from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.rrf import rrf_with_limit
from memini_ai.memory.schema import MemoryEntry, SearchFilter, SearchOptions
from memini_ai.utils.logger import logger

__all__ = ["RRFDatabase"]


class RRFDatabase(VectorDatabase):
    """Fuse two ``VectorDatabase`` backends via Reciprocal Rank Fusion.

    Writes go to the primary only (with an async fire-and-forget dual-write
    to the secondary). Reads fan out to both backends in parallel and the
    ranked result lists are merged with RRF. The primary is authoritative
    for tie-breaking; the secondary is best-effort and failures degrade
    gracefully to primary-only results.
    """

    def __init__(
        self,
        primary: VectorDatabase,
        secondary: VectorDatabase,
        k: int = 60,
    ) -> None:
        """Construct an RRF-fused database wrapper.

        Args:
            primary: Authoritative backend. Writes go here; primary query
                failures propagate to the caller.
            secondary: Best-effort secondary backend (typically a team
                server). Query failures degrade to primary-only results.
            k: RRF constant. Higher values flatten the contribution curve
                so lower ranks matter more. 60 is the canonical value from
                Cormack et al. SIGIR 2009.
        """
        self._primary = primary
        self._secondary = secondary
        self._k = k
        self._initialized = False
        self._dimension: int | None = None
        self._pool: Any = None
        self._team_write_failures: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize both backends.

        The primary is initialized unconditionally — a failure here
        propagates. The secondary is best-effort: if it cannot be
        reached at startup we log a warning and continue in
        primary-only mode.
        """
        if self._initialized:
            return
        await self._primary.initialize()
        try:
            await self._secondary.initialize()
        except Exception as e:
            logger.warning(
                "team_server_init_failed",
                error=str(e)[:200],
                message="Team server unavailable at startup, continuing in embedded-only mode",
            )
        # Mirror the primary's reported dimension so callers that
        # introspect RRFDatabase see the authoritative value.
        self._dimension = getattr(self._primary, "_dimension", None)
        self._pool = getattr(self._primary, "_pool", None)
        self._initialized = True

    async def close(self) -> None:
        """Close both backends. Secondary failures are logged, not raised."""
        await self._primary.close()
        try:
            await self._secondary.close()
        except Exception as e:
            logger.warning("team_server_close_failed", error=str(e)[:200])

    # ──────────────────────────────────────────────────────────────────
    # Writes — primary authoritative, secondary fire-and-forget (Q3)
    # ──────────────────────────────────────────────────────────────────

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory to the primary, then async dual-write to the secondary.

        The primary write is awaited and its returned id is the canonical
        id. The secondary write is dispatched as a fire-and-forget
        ``asyncio.create_task`` so the caller is never blocked by a slow
        team server. Secondary failures are counted in
        ``self._team_write_failures`` and logged.
        """
        await self.initialize()
        memory_id = await self._primary.add_memory(entry)
        asyncio.create_task(self._write_to_team(entry, memory_id))
        return memory_id

    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Batch add. Primary is awaited; secondary writes are fire-and-forget."""
        await self.initialize()
        written = await self._primary.add_memories(entries)
        for entry in written:
            asyncio.create_task(self._write_to_team(entry, entry.id))
        return written

    async def _write_to_team(self, entry: MemoryEntry, memory_id: str) -> None:
        """Best-effort dual-write to the secondary backend (Q3)."""
        try:
            await self._secondary.add_memory(entry)
        except Exception as e:
            self._team_write_failures += 1
            logger.warning(
                "team_write_failed",
                memory_id=memory_id,
                error=str(e)[:200],
            )

    async def delete_memory(self, memory_id: str) -> None:
        """Delete from the primary only. The secondary is eventually-consistent."""
        await self.initialize()
        await self._primary.delete_memory(memory_id)

    async def delete_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> int:
        """Delete by source path on the primary only."""
        await self.initialize()
        return await self._primary.delete_by_source_path(source_path, source_type)

    # ──────────────────────────────────────────────────────────────────
    # Reads — fan out + RRF fuse
    # ──────────────────────────────────────────────────────────────────

    async def get_memory(
        self,
        memory_id: str,
        include_archived: bool = False,
    ) -> MemoryEntry | None:
        """Get a memory by id. Primary is tried first; secondary is a fallback."""
        await self.initialize()
        result = await self._primary.get_memory(memory_id, include_archived)
        if result is not None:
            return result
        try:
            return await self._secondary.get_memory(memory_id, include_archived)
        except Exception as e:
            logger.warning(
                "team_server_get_failed",
                memory_id=memory_id,
                error=str(e)[:200],
            )
            return None

    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Query both backends in parallel and RRF-fuse the ranked results.

        The primary task is awaited unconditionally — if it raises, the
        exception propagates to the caller (the primary is
        authoritative). The secondary task is best-effort: on failure
        we log a warning and return primary-only results.

        Tie-breaking: when the same memory id appears in both result
        sets, the primary's ``MemoryEntry`` object wins (we populate
        the lookup dict from primary first, then ``setdefault`` for
        secondary entries so secondary only fills gaps).
        """
        await self.initialize()

        primary_task = asyncio.create_task(
            self._primary.query_memories(vector, options, collection_name),
        )
        secondary_task = asyncio.create_task(
            self._secondary.query_memories(vector, options, collection_name),
        )

        # Primary is required — propagate failures.
        results_primary = await primary_task

        # Secondary is best-effort.
        try:
            results_secondary = await secondary_task
        except Exception as e:
            logger.warning(
                "team_server_query_failed",
                error=str(e)[:200],
                fallback="embedded_only",
            )
            return results_primary

        # RRF fuse by memory id.
        ranked_ids: list[list[str]] = [
            [e.id for e in results_primary],
            [e.id for e in results_secondary],
        ]
        fused_ids = rrf_with_limit(ranked_ids, k=self._k, limit=options.top_k)

        # Re-hydrate MemoryEntry objects — primary wins on conflicts.
        entries_by_id: dict[str, MemoryEntry] = {e.id: e for e in results_primary}
        for entry in results_secondary:
            entries_by_id.setdefault(entry.id, entry)

        return [entries_by_id[mid] for mid in fused_ids if mid in entries_by_id]

    # ──────────────────────────────────────────────────────────────────
    # Pass-through read methods — delegate to primary
    # ──────────────────────────────────────────────────────────────────

    async def list_memories(
        self,
        filter: SearchFilter | None = None,
    ) -> list[MemoryEntry]:
        await self.initialize()
        return await self._primary.list_memories(filter)

    async def count_memories(self) -> int:
        await self.initialize()
        return await self._primary.count_memories()

    async def count_thoughts(self) -> int:
        await self.initialize()
        return await self._primary.count_thoughts()

    async def content_exists(self, content_hash: str) -> bool:
        await self.initialize()
        return await self._primary.content_exists(content_hash)

    async def get_entries_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> list[MemoryEntry]:
        await self.initialize()
        return await self._primary.get_entries_by_source_path(source_path, source_type)

    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        await self.initialize()
        return await self._primary.scroll_collection(collection_name, limit)

    async def get_collection_dimension(
        self,
        collection_name: str,
    ) -> int | None:
        await self.initialize()
        return await self._primary.get_collection_dimension(collection_name)

    async def update_trust_fields(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        await self.initialize()
        await self._primary.update_trust_fields(memory_id, trust_score, is_archived)

    async def increment_retrieval_count(self, memory_id: str) -> None:
        await self.initialize()
        await self._primary.increment_retrieval_count(memory_id)

    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> None:
        await self.initialize()
        await self._primary.set_payload(memory_id, payload)

    async def get_supersession_chain(
        self,
        memory_id: str,
        max_depth: int = 10,
    ) -> list[MemoryEntry]:
        await self.initialize()
        return await self._primary.get_supersession_chain(memory_id, max_depth)

    async def get_superseded_memory(
        self,
        memory_id: str,
    ) -> MemoryEntry | None:
        await self.initialize()
        return await self._primary.get_superseded_memory(memory_id)
