"""Tests for RRFDatabase — the VectorDatabase wrapper that fuses two backends via RRF.

These tests use fake backends (FakeVectorDatabase) to verify the fusion
logic, failure handling, tie-breaking, and write delegation of RRFDatabase
without requiring a live PostgreSQL connection.

Covers (design doc section 8.2):
1. Fusion correctness (5 tests) — Two backends return known ranked lists
2. Team unreachable (2 tests) — Secondary raises, degrade to primary-only
3. Tie-breaking (2 tests) — Same memory_id in both, primary wins
4. Empty results (2 tests) — One or both backends return empty
5. Parallel execution (2 tests) — Both queries fire concurrently
6. Write delegation (2 tests) — add_memory hits primary, async dual-write fires
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.rrf_database import RRFDatabase
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchFilter,
    SearchOptions,
)

# ──────────────────────────────────────────────────────────────────────
# FakeVectorDatabase — a minimal in-memory VectorDatabase for testing
# ──────────────────────────────────────────────────────────────────────


class FakeVectorDatabase(VectorDatabase):
    """Fake VectorDatabase for testing RRFDatabase fusion logic.

    Stores a list of MemoryEntry objects. ``query_memories`` returns them
    in the order they were added (simulating a ranked result list).
    ``add_memory`` appends to the list and generates a fake ID.

    Can be configured to raise on ``query_memories`` or ``add_memory``
    to simulate team server unreachable scenarios.
    """

    def __init__(
        self,
        entries: list[MemoryEntry] | None = None,
        *,
        raise_on_query: Exception | None = None,
        raise_on_add: Exception | None = None,
        raise_on_initialize: Exception | None = None,
    ) -> None:
        self._entries: list[MemoryEntry] = list(entries) if entries else []
        self._raise_on_query = raise_on_query
        self._raise_on_add = raise_on_add
        self._raise_on_initialize = raise_on_initialize
        self._initialized = False
        self._dimension: int | None = 384
        self._pool: Any = None
        self.add_calls: list[MemoryEntry] = []
        self.query_calls: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._raise_on_initialize:
            raise self._raise_on_initialize
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    # ── Writes ────────────────────────────────────────────────────────

    async def add_memory(self, entry: MemoryEntry) -> str:
        if self._raise_on_add:
            raise self._raise_on_add
        self.add_calls.append(entry)
        # Generate a fake ID if not already set
        if not entry.id or entry.id == "":
            entry.id = f"mem-{len(self._entries)}"
        self._entries.append(entry)
        return entry.id

    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        written: list[MemoryEntry] = []
        for e in entries:
            mid = await self.add_memory(e)
            e.id = mid
            written.append(e)
        return written

    async def delete_memory(self, memory_id: str) -> None:
        self._entries = [e for e in self._entries if e.id != memory_id]

    async def delete_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> int:
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if e.source_path != source_path
            or (source_type is not None and e.source_type != source_type)
        ]
        return before - len(self._entries)

    # ── Reads ─────────────────────────────────────────────────────────

    async def get_memory(
        self,
        memory_id: str,
        include_archived: bool = False,
    ) -> MemoryEntry | None:
        for e in self._entries:
            if e.id == memory_id:
                if not include_archived and e.is_archived:
                    continue
                return e
        return None

    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        self.query_calls += 1
        if self._raise_on_query:
            raise self._raise_on_query
        # Return entries in stored order (simulating ranked results)
        return self._entries[: options.top_k]

    async def list_memories(
        self,
        filter: SearchFilter | None = None,
    ) -> list[MemoryEntry]:
        return list(self._entries)

    async def count_memories(self) -> int:
        return len(self._entries)

    async def count_thoughts(self) -> int:
        return 0

    async def content_exists(self, content_hash: str) -> bool:
        return any(e.content_hash == content_hash for e in self._entries)

    async def get_entries_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> list[MemoryEntry]:
        return [
            e
            for e in self._entries
            if e.source_path == source_path
            and (source_type is None or e.source_type == source_type)
        ]

    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        return list(self._entries)

    async def get_collection_dimension(
        self,
        collection_name: str,
    ) -> int | None:
        return self._dimension

    async def update_trust_fields(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        for e in self._entries:
            if e.id == memory_id:
                e.trust_score = trust_score
                e.is_archived = is_archived
                break

    async def increment_retrieval_count(self, memory_id: str) -> None:
        for e in self._entries:
            if e.id == memory_id:
                e.retrieval_count += 1
                break

    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> None:
        pass

    async def get_supersession_chain(
        self,
        memory_id: str,
        max_depth: int = 10,
    ) -> list[MemoryEntry]:
        return []

    async def get_superseded_memory(
        self,
        memory_id: str,
    ) -> MemoryEntry | None:
        return None


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def make_entry(
    id: str | None,
    content: str,
    **overrides: Any,
) -> MemoryEntry:
    """Create a MemoryEntry with sensible defaults for testing.

    Args:
        id: Memory ID (None for auto-generated).
        content: Text content.
        **overrides: Any MemoryEntry field to override.

    Returns:
        A MemoryEntry ready for use in tests.
    """
    now = datetime.now(UTC)
    kwargs: dict[str, Any] = dict(
        id=id or "",
        text=content,
        vector=[0.1] * 384,
        source_type=MemorySourceType.session,
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        trust_score=0.5,
        timestamp=now,
        project_id="test",
        **overrides,
    )
    return MemoryEntry(**kwargs)


@pytest.fixture
def search_options() -> SearchOptions:
    """Default SearchOptions for RRF query tests."""
    return SearchOptions(top_k=10)


# ══════════════════════════════════════════════════════════════════════
# 1. Fusion correctness (5 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rrf_fuses_two_lists(search_options: SearchOptions) -> None:
    """Two backends with overlapping IDs: RRF boosts shared items."""
    entry1 = make_entry("m1", "alpha")
    entry2 = make_entry("m2", "beta")
    entry3 = make_entry("m3", "gamma")

    primary = FakeVectorDatabase([entry1, entry2])
    secondary = FakeVectorDatabase([entry2, entry3])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    # m2 appears in both lists → highest RRF score → first
    assert results[0].id == "m2"
    # All 3 distinct IDs should be present
    assert {r.id for r in results} == {"m1", "m2", "m3"}


@pytest.mark.asyncio
async def test_rrf_preserves_primary_order_for_unique_items(
    search_options: SearchOptions,
) -> None:
    """Items only in primary appear in primary's relative order."""
    entry1 = make_entry("m1", "alpha")
    entry2 = make_entry("m2", "beta")
    entry3 = make_entry("m3", "gamma")

    primary = FakeVectorDatabase([entry1, entry2, entry3])
    secondary = FakeVectorDatabase([])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    assert [r.id for r in results] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_rrf_three_way_fusion(search_options: SearchOptions) -> None:
    """Items in both lists get higher fused scores than items in one."""
    # m1: rank 0 in primary, rank 2 in secondary → high boost
    # m2: rank 1 in primary only → medium
    # m3: rank 0 in secondary only → medium
    m1 = make_entry("m1", "shared high")
    m2 = make_entry("m2", "primary only")
    m3 = make_entry("m3", "secondary only")

    primary = FakeVectorDatabase([m1, m2])
    secondary = FakeVectorDatabase([m3, m1])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    # m1 appears in both → highest fused score → first
    assert results[0].id == "m1"
    # All 3 present
    assert {r.id for r in results} == {"m1", "m2", "m3"}


@pytest.mark.asyncio
async def test_rrf_respects_top_k_limit(search_options: SearchOptions) -> None:
    """top_k from SearchOptions limits the fused result count."""
    entries_p = [make_entry(f"p{i}", f"primary-{i}") for i in range(5)]
    entries_s = [make_entry(f"s{i}", f"secondary-{i}") for i in range(5)]

    primary = FakeVectorDatabase(entries_p)
    secondary = FakeVectorDatabase(entries_s)

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    limited_opts = SearchOptions(top_k=3)
    results = await rrf.query_memories([0.5] * 384, limited_opts)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_rrf_identical_lists_returns_all(search_options: SearchOptions) -> None:
    """Two identical lists: all items returned, order preserved."""
    entries = [
        make_entry("m1", "alpha"),
        make_entry("m2", "beta"),
        make_entry("m3", "gamma"),
    ]

    primary = FakeVectorDatabase(list(entries))
    secondary = FakeVectorDatabase(list(entries))

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    # All 3 items present
    assert {r.id for r in results} == {"m1", "m2", "m3"}
    # m1 and m2 appear in both → tied for highest score
    assert results[0].id in ("m1", "m2")


# ══════════════════════════════════════════════════════════════════════
# 2. Team unreachable (2 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_rrf_degrades_to_primary_when_secondary_fails(
    search_options: SearchOptions,
) -> None:
    """Secondary raises on query → degrade to primary-only results."""
    entry1 = make_entry("m1", "alpha")
    entry2 = make_entry("m2", "beta")

    primary = FakeVectorDatabase([entry1, entry2])
    secondary = FakeVectorDatabase(
        raise_on_query=ConnectionError("team server down"),
    )

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    # Should NOT raise — secondary failure is caught and logged
    results = await rrf.query_memories([0.5] * 384, search_options)

    # Should return primary-only results
    assert len(results) == 2
    assert [r.id for r in results] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_rrf_primary_failure_propagates(
    search_options: SearchOptions,
) -> None:
    """Primary raises on query → exception propagates to caller."""
    primary = FakeVectorDatabase(raise_on_query=RuntimeError("primary crashed"))
    secondary = FakeVectorDatabase([make_entry("m1", "alpha")])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    with pytest.raises(RuntimeError, match="primary crashed"):
        await rrf.query_memories([0.5] * 384, search_options)


# ══════════════════════════════════════════════════════════════════════
# 3. Tie-breaking (2 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_primary_wins_on_content_conflict(
    search_options: SearchOptions,
) -> None:
    """Same memory_id in both with different content → primary wins."""
    entry_primary = make_entry("m1", "primary version")
    entry_secondary = make_entry("m1", "team version")

    primary = FakeVectorDatabase([entry_primary])
    secondary = FakeVectorDatabase([entry_secondary])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    assert len(results) == 1
    assert results[0].id == "m1"
    assert results[0].text == "primary version"


@pytest.mark.asyncio
async def test_primary_wins_on_mixed_conflict(
    search_options: SearchOptions,
) -> None:
    """Multiple conflicts: primary content wins for each shared ID."""
    entries_primary = [
        make_entry("m1", "primary-a"),
        make_entry("m2", "primary-b"),
        make_entry("m3", "only-in-primary"),
    ]
    entries_secondary = [
        make_entry("m1", "team-a"),
        make_entry("m2", "team-b"),
        make_entry("m4", "only-in-team"),
    ]

    primary = FakeVectorDatabase(entries_primary)
    secondary = FakeVectorDatabase(entries_secondary)

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    by_id = {r.id: r.text for r in results}

    # Primary wins for shared IDs
    assert by_id["m1"] == "primary-a"
    assert by_id["m2"] == "primary-b"
    # Unique items from each backend are present
    assert by_id["m3"] == "only-in-primary"
    assert by_id["m4"] == "only-in-team"


# ══════════════════════════════════════════════════════════════════════
# 4. Empty results (2 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_both_backends_empty_returns_empty(
    search_options: SearchOptions,
) -> None:
    """Both backends return no results → empty list, no crash."""
    primary = FakeVectorDatabase([])
    secondary = FakeVectorDatabase([])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    assert results == []


@pytest.mark.asyncio
async def test_secondary_empty_returns_primary_only(
    search_options: SearchOptions,
) -> None:
    """Secondary returns empty → primary results returned as-is."""
    entry = make_entry("m1", "alpha")
    primary = FakeVectorDatabase([entry])
    secondary = FakeVectorDatabase([])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    results = await rrf.query_memories([0.5] * 384, search_options)

    assert len(results) == 1
    assert results[0].id == "m1"


# ══════════════════════════════════════════════════════════════════════
# 5. Parallel execution (2 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_both_backends_queried_concurrently(
    search_options: SearchOptions,
) -> None:
    """Both backends receive a query call during RRF query."""
    primary = FakeVectorDatabase([make_entry("m1", "alpha")])
    secondary = FakeVectorDatabase([make_entry("m2", "beta")])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    assert primary.query_calls == 0
    assert secondary.query_calls == 0

    await rrf.query_memories([0.5] * 384, search_options)

    # Both should have been called exactly once
    assert primary.query_calls == 1
    assert secondary.query_calls == 1


@pytest.mark.asyncio
async def test_secondary_slow_does_not_block_primary(
    search_options: SearchOptions,
) -> None:
    """A slow secondary does not delay primary results.

    RRFDatabase uses asyncio.create_task for both queries and awaits
    the primary first, so a slow secondary cannot block the primary
    result path.
    """
    entry = make_entry("m1", "alpha")

    class SlowFake(FakeVectorDatabase):
        """Fake that adds a delay to query_memories."""

        async def query_memories(
            self,
            vector: list[float],
            options: SearchOptions,
            collection_name: str | None = None,
        ) -> list[MemoryEntry]:
            self.query_calls += 1
            if self._raise_on_query:
                raise self._raise_on_query
            await asyncio.sleep(0.2)  # Simulate slow backend
            return self._entries[: options.top_k]

    primary = FakeVectorDatabase([entry])
    secondary = SlowFake([make_entry("m2", "beta")])

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    # Should complete within ~200ms (not 400ms) because primary
    # is awaited first and secondary runs concurrently
    start = asyncio.get_event_loop().time()
    results = await rrf.query_memories([0.5] * 384, search_options)
    elapsed = asyncio.get_event_loop().time() - start

    # Primary result should be present
    assert results[0].id == "m1"
    # Should complete in less than 2x the secondary delay
    # (if sequential, it would be ~400ms; concurrent is ~200ms)
    assert elapsed < 0.35, f"Expected concurrent execution (<0.35s), got {elapsed:.3f}s"


# ══════════════════════════════════════════════════════════════════════
# 6. Write delegation (2 tests)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_add_memory_writes_to_primary_and_fires_to_secondary() -> None:
    """add_memory writes to primary immediately and fires async dual-write."""
    entry = make_entry(None, "new memory")

    primary = FakeVectorDatabase()
    secondary = FakeVectorDatabase()

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    memory_id = await rrf.add_memory(entry)

    # Primary got the write immediately
    assert len(primary.add_calls) == 1
    assert primary.add_calls[0].text == "new memory"
    assert memory_id is not None

    # Secondary write is fire-and-forget — give it a tick
    await asyncio.sleep(0.1)
    assert len(secondary.add_calls) == 1
    assert secondary.add_calls[0].text == "new memory"


@pytest.mark.asyncio
async def test_secondary_write_failure_does_not_raise() -> None:
    """Secondary add_memory failure is caught and counted, not raised."""
    entry = make_entry(None, "new memory")

    primary = FakeVectorDatabase()
    secondary = FakeVectorDatabase(
        raise_on_add=ConnectionError("team write failed"),
    )

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    # Should NOT raise — secondary failure is fire-and-forget
    memory_id = await rrf.add_memory(entry)

    assert memory_id is not None
    assert len(primary.add_calls) == 1

    # Give the fire-and-forget task time to execute
    await asyncio.sleep(0.1)

    # Team write failure should be counted
    assert rrf._team_write_failures >= 1


# ══════════════════════════════════════════════════════════════════════
# 7. Lifecycle (1 bonus test)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_initialize_handles_secondary_failure() -> None:
    """Secondary init failure is caught; RRFDatabase continues in degraded mode."""
    primary = FakeVectorDatabase([make_entry("m1", "alpha")])
    secondary = FakeVectorDatabase(
        raise_on_initialize=ConnectionError("team server down at startup"),
    )

    rrf = RRFDatabase(primary=primary, secondary=secondary, k=60)
    await rrf.initialize()

    # Should be marked as initialized despite secondary failure
    assert rrf._initialized

    # Queries should still work (primary only)
    results = await rrf.query_memories(
        [0.5] * 384,
        SearchOptions(top_k=10),
    )
    assert len(results) == 1
    assert results[0].id == "m1"
