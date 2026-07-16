"""Tests for multi-model add_memory routing.

Verifies that ``add_memory`` writes the embedding vector to the column
matching the model's dimensionality:

  * ``all-MiniLM-L6-v2``  → ``embedding``        (384-dim)
  * ``BAAI/bge-m3``       → ``embedding_bge_m3`` (1024-dim)

BGE-Large support was removed in v0.7.6 — only MiniLM and BGE-M3 are
supported. The BGE-Large migration script in
``archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py``
is kept as a reference example for users who want to do similar
migrations on their own.

The tests use the live PostgreSQL database at ``MEMINI_DB_URL`` (port 5434)
and clean up after themselves by deleting the inserted rows.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from memini_ai.memory.schema import MemoryEntry, MemorySourceType
from memini_ai.model.manager import BGE_M3_MODEL_ID, MINILM_MODEL_ID
from memini_ai.postgres.database import PostgresDatabase


def _make_vector(dim: int, seed: int = 0) -> list[float]:
    """Create a normalised random vector of the given dimension."""
    rng = np.random.RandomState(seed)
    vec = rng.rand(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# _entry_to_record unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestEntryToRecordColumnRouting:
    """Unit tests for _entry_to_record vector-column selection."""

    def test_minilm_writes_to_embedding(self) -> None:
        """MiniLM vectors go to the ``embedding`` key (backwards compat)."""
        db = PostgresDatabase.__new__(PostgresDatabase)
        vec = _make_vector(384, seed=1)
        entry = MemoryEntry(
            text="minilm test",
            vector=vec,
            embedding_model=MINILM_MODEL_ID,
            source_type=MemorySourceType.project,
        )
        record = db._entry_to_record(entry)
        assert record["embedding"] == vec
        # No bge columns should be populated
        assert "embedding_bge_m3" not in record

    def test_bge_m3_writes_to_embedding_bge_m3(self) -> None:
        """BGE-M3 vectors go to the ``embedding_bge_m3`` key."""
        db = PostgresDatabase.__new__(PostgresDatabase)
        vec = _make_vector(1024, seed=2)
        entry = MemoryEntry(
            text="bge-m3 test",
            vector=vec,
            embedding_model=BGE_M3_MODEL_ID,
            source_type=MemorySourceType.project,
        )
        record = db._entry_to_record(entry)
        assert record["embedding_bge_m3"] == vec
        # The generic ``embedding`` key must be present (None) for compat
        assert record["embedding"] is None

    def test_no_embedding_model_defaults_to_embedding(self) -> None:
        """When embedding_model is None, the vector goes to ``embedding``."""
        db = PostgresDatabase.__new__(PostgresDatabase)
        vec = _make_vector(384, seed=4)
        entry = MemoryEntry(
            text="legacy test",
            vector=vec,
            source_type=MemorySourceType.session,
        )
        record = db._entry_to_record(entry)
        assert record["embedding"] == vec
        assert record["embedding_model"] is None


# ---------------------------------------------------------------------------
# Integration tests against live PostgreSQL
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MEMINI_DB_URL"),
    reason="MEMINI_DB_URL not set — skipping live DB integration tests",
)
class TestAddMemoryLiveDB:
    """Integration tests that verify vectors land in the correct DB column."""

    @pytest.mark.asyncio
    async def test_bge_m3_vector_lands_in_bge_m3_column(
        self, pg_db: PostgresDatabase
    ) -> None:
        """BGE-M3 1024-dim vector must populate embedding_bge_m3, not embedding."""
        vec = _make_vector(1024, seed=42)
        entry = MemoryEntry(
            text="test_bge_m3_column_routing_2026_07_10",
            vector=vec,
            embedding_model=BGE_M3_MODEL_ID,
            source_type=MemorySourceType.project,
            source_path="multi-model-test",
        )
        mem_id = await pg_db.add_memory(entry)
        assert mem_id is not None

        try:
            async with pg_db._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT embedding, embedding_bge_m3, embedding_model "
                    "FROM memories WHERE id = $1",
                    mem_id,
                )
            assert row is not None, "Memory row not found after insert"
            assert row["embedding_model"] == BGE_M3_MODEL_ID
            # The 1024-dim vector must be in embedding_bge_m3
            assert row["embedding_bge_m3"] is not None, (
                "embedding_bge_m3 should be populated for BGE-M3"
            )
            # The 384-dim embedding column must be NULL (we didn't write to it)
            assert row["embedding"] is None, (
                "embedding column should be NULL for BGE-M3 entries"
            )
        finally:
            # Hard-delete the test row (bypasses soft-delete)
            async with pg_db._pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1", mem_id)

    @pytest.mark.asyncio
    async def test_minilm_vector_lands_in_embedding_column(
        self, pg_db: PostgresDatabase
    ) -> None:
        """MiniLM 384-dim vector must populate the default embedding column."""
        vec = _make_vector(384, seed=7)
        entry = MemoryEntry(
            text="test_minilm_column_routing_2026_07_10",
            vector=vec,
            embedding_model=MINILM_MODEL_ID,
            source_type=MemorySourceType.project,
            source_path="multi-model-test",
        )
        mem_id = await pg_db.add_memory(entry)
        assert mem_id is not None

        try:
            async with pg_db._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT embedding, embedding_bge_m3, embedding_model "
                    "FROM memories WHERE id = $1",
                    mem_id,
                )
            assert row is not None
            assert row["embedding_model"] == MINILM_MODEL_ID
            assert row["embedding"] is not None, (
                "embedding should be populated for MiniLM"
            )
            assert row["embedding_bge_m3"] is None
        finally:
            async with pg_db._pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1", mem_id)

    @pytest.mark.asyncio
    async def test_no_model_defaults_to_embedding_column(
        self, pg_db: PostgresDatabase
    ) -> None:
        """When embedding_model is None, the 384-dim vector goes to embedding."""
        vec = _make_vector(384, seed=11)
        entry = MemoryEntry(
            text="test_no_model_defaults_to_embedding_2026_07_10",
            vector=vec,
            source_type=MemorySourceType.session,
        )
        mem_id = await pg_db.add_memory(entry)
        assert mem_id is not None

        try:
            async with pg_db._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT embedding, embedding_bge_m3, embedding_model "
                    "FROM memories WHERE id = $1",
                    mem_id,
                )
            assert row is not None
            # embedding_model should be None (legacy path, no model tracking)
            assert row["embedding_model"] is None
            assert row["embedding"] is not None
            assert row["embedding_bge_m3"] is None
        finally:
            async with pg_db._pool.acquire() as conn:
                await conn.execute("DELETE FROM memories WHERE id = $1", mem_id)
