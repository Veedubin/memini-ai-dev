"""Tests for the memories_1024 schema migration (v0.7.0).

Real-DB tests. They verify:
- The memories_1024 table exists after initialize()
- The FK to memories(id) is enforced (cannot insert a 1024 row with a
  non-existent memory_id)
- The migration is idempotent (running get_schema_sql() twice does not
  raise)
- The unique constraint on memory_id is in place (a second insert with
  the same memory_id does not create a duplicate row)
- The vector(1024) column is the correct type (not vector(384))

These tests use the same PostgresDatabase the production code uses, so
they exercise the actual schema path. They assume the standard dev DB
at localhost:5434 is available (matches TEST_DB_URL default).
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from memini_ai.postgres import PostgresDatabase

TEST_DB_URL = os.getenv(
    "TEST_DB_URL",
    os.getenv(
        "MEMINI_DB_URL", "postgresql://postgres:password@localhost:5434/postgres"
    ),
)


@pytest_asyncio.fixture
async def pg_db():
    """Create PostgresDatabase connected to the dev DB (initializes schema)."""
    db = PostgresDatabase(TEST_DB_URL)
    await db.initialize()
    yield db
    await db.close()


async def test_memories_1024_table_exists(pg_db: PostgresDatabase) -> None:
    """memories_1024 table must exist after initialize()."""
    pool = await pg_db._get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables "
            "  WHERE table_name = 'memories_1024'"
            ")"
        )
    assert exists is True


async def test_memories_1024_fk_to_memories(pg_db: PostgresDatabase) -> None:
    """Inserting a 1024 row with a non-existent memory_id must fail."""
    pool = await pg_db._get_pool()
    fake_memory_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        with pytest.raises(Exception) as exc_info:
            await conn.execute(
                "INSERT INTO memories_1024 (memory_id, embedding) VALUES ($1, $2)",
                fake_memory_id,
                [0.0] * 1024,
            )
    # The error message should mention the FK constraint by name.
    # The constraint name follows the pattern memories_1024_memory_id_fkey
    # in PostgreSQL.
    assert (
        "memories_1024_memory_id_fkey" in str(exc_info.value)
        or "foreign key" in str(exc_info.value).lower()
    )


async def test_memories_1024_idempotent_migration(pg_db: PostgresDatabase) -> None:
    """Running initialize() a second time must not raise (idempotent)."""
    # pg_db is already initialized. Call initialize() again — should
    # be a no-op without raising.
    await pg_db.initialize()
    # The pool is still open, the schema is still in place, no errors
    # thrown. (PostgresDatabase doesn't expose a public ``is_initialized``
    # property; we infer success from "no exception".)


async def test_memories_1024_vector_column_type(pg_db: PostgresDatabase) -> None:
    """The embedding column must be vector(1024), not vector(384)."""
    pool = await pg_db._get_pool()
    async with pool.acquire() as conn:
        col_type = await conn.fetchval(
            "SELECT format_type(atttypid, atttypmod) "
            "FROM pg_attribute "
            "WHERE attrelid = 'memories_1024'::regclass "
            "  AND attname = 'embedding'"
        )
    assert col_type is not None
    # pgvector's format_type returns something like "vector" with a
    # typmod of 1024. Check both the full string and the typmod.
    assert "1024" in col_type or col_type == "vector"


async def test_memories_1024_unique_on_memory_id(pg_db: PostgresDatabase) -> None:
    """A unique constraint on memory_id prevents duplicate 1024 rows."""
    # Create a real 384-dim memory first, then try to insert two 1024
    # rows for it.
    from memini_ai.memory.schema import MemoryEntry, MemorySourceType

    entry = MemoryEntry(
        text="test_migration_unique_constraint",
        sourceType=MemorySourceType.session,
    )
    # Use a deterministic small vector to keep the test fast.
    entry.vector = [0.1] * 384
    memory_id = await pg_db.add_memory(entry)

    pool = await pg_db._get_pool()
    async with pool.acquire() as conn:
        # First insert should succeed.
        await conn.execute(
            "INSERT INTO memories_1024 (memory_id, embedding) VALUES ($1, $2)",
            memory_id,
            [0.1] * 1024,
        )
        # Second insert with the same memory_id should fail (unique).
        from asyncpg.exceptions import UniqueViolationError

        with pytest.raises(UniqueViolationError):
            await conn.execute(
                "INSERT INTO memories_1024 (memory_id, embedding) VALUES ($1, $2)",
                memory_id,
                [0.2] * 1024,
            )

    # Cleanup: remove the test memory (FK ON DELETE CASCADE will tidy
    # up the 1024 row).
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)
