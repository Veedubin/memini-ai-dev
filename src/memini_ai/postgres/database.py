"""PostgreSQL/pgvector implementation of VectorDatabase using asyncpg.

This module provides the PostgresDatabase class that implements the VectorDatabase
ABC using asyncpg for async PostgreSQL operations with pgvector for vector storage
and pgvectorscale's StreamingDiskANN index for high-performance similarity search.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

import asyncpg

from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchFilter,
    SearchOptions,
)
from memini_ai.postgres.queries import (
    DELETE_MEMORY,
    GET_MEMORY_BY_ID,
    GET_MEMORY_COUNT,
    INCREMENT_RETRIEVAL_COUNT,
    INSERT_MEMORY,
    SEARCH_MEMORIES_VECTOR,
    UPDATE_MEMORY_METADATA,
)
from memini_ai.postgres.schema import get_schema_sql

if TYPE_CHECKING:
    pass


class PostgresDatabase(VectorDatabase):
    """PostgreSQL/pgvector implementation of VectorDatabase.

    Uses asyncpg for async database operations and supports pgvector's
    StreamingDiskANN index for efficient vector similarity search.

    Connection Management:
        - Uses asyncpg connection pool for concurrent operations
        - Lazy connection establishment on first use
        - Schema initialization on first connection
    """

    def __init__(self, db_url: str, project_id: str | None = None) -> None:
        """Initialize PostgresDatabase.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/db).
            project_id: Optional project ID for isolation (unused in pgvector impl).
        """
        self._db_url = db_url
        self._project_id = project_id
        self._pool: asyncpg.Pool | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and create schema if needed."""
        if self._initialized:
            return

        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self._db_url,
                min_size=1,
                max_size=10,
            )

            # Initialize schema
            await self._ensure_schema()

            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PostgreSQL connection: {e}"
            ) from e

    async def _ensure_schema(self) -> None:
        """Create database schema if tables don't exist (idempotent)."""
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")

        schema_sql = get_schema_sql()

        async with self._pool.acquire() as conn:
            # Execute schema creation (IF NOT EXISTS makes it idempotent)
            await conn.execute(schema_sql)

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            await self.initialize()
        if self._pool is None:
            raise RuntimeError("Failed to create database pool")
        return self._pool

    def _entry_to_record(self, entry: MemoryEntry) -> dict[str, Any]:
        """Convert MemoryEntry to database record dict."""
        record: dict[str, Any] = {
            "id": entry.id or str(uuid.uuid4()),
            "text": entry.text,
            "source_type": entry.source_type.value
            if isinstance(entry.source_type, MemorySourceType)
            else entry.source_type,
        }

        # Convert vector to list for PostgreSQL
        if entry.vector is not None:
            import numpy as np

            if isinstance(entry.vector, np.ndarray):
                record["embedding"] = entry.vector.tolist()
            else:
                record["embedding"] = list(entry.vector)
        else:
            record["embedding"] = None

        record["content_hash"] = entry.content_hash or ""
        record["metadata"] = {}

        if entry.metadata_json:
            with contextlib.suppress(json.JSONDecodeError):
                record["metadata"] = json.loads(entry.metadata_json)

        return record

    def _row_to_memory(
        self,
        row: asyncpg.Record,
        score: float | None = None,
    ) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        data = {
            "id": str(row["id"]),
            "text": row["text"],
            "vector": list(row["embedding"]) if row["embedding"] else None,
            "source_type": row["source_type"],
            "content_hash": row.get("content_hash"),
            "metadata_json": (
                json.dumps(row["metadata"])
                if isinstance(row["metadata"], dict)
                else row["metadata"]
            ),
            "trust_score": row.get("trust_score", 0.5),
            "retrieval_count": row.get("retrieval_count", 0),
            "is_archived": row.get("is_archived", False),
            "last_accessed_at": row.get("last_accessed_at"),
            "score": score,
        }

        return MemoryEntry.model_validate(data)

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry.

        Args:
            entry: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.
        """
        await self.initialize()
        pool = await self._get_pool()

        record = self._entry_to_record(entry)
        memory_id = record["id"]

        async with pool.acquire() as conn:
            memory_id = await conn.fetchval(
                INSERT_MEMORY,
                memory_id,
                record["text"],
                record["embedding"],
                record["source_type"],
                record["content_hash"],
                json.dumps(record["metadata"]),
            )

        return str(memory_id)

    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Add multiple memory entries.

        Args:
            entries: List of MemoryEntry objects to add.

        Returns:
            List of MemoryEntry objects with IDs assigned.
        """
        if not entries:
            return []

        await self.initialize()
        pool = await self._get_pool()

        # Prepare all records
        records = []
        for entry in entries:
            record = self._entry_to_record(entry)
            records.append(record)

        async with pool.acquire() as conn, conn.transaction():
            for record in records:
                await conn.fetchval(
                    INSERT_MEMORY,
                    record["id"],
                    record["text"],
                    record["embedding"],
                    record["source_type"],
                    record["content_hash"],
                    json.dumps(record["metadata"]),
                )

        return entries

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_MEMORY_BY_ID, memory_id)
            if row is None:
                return None

            return self._row_to_memory(row)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID.

        Args:
            memory_id: ID of the memory entry to delete.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(DELETE_MEMORY, memory_id)

    async def delete_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> int:
        """Delete all memories with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            Number of deleted memories.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if source_type:
                query = """
                    UPDATE memories
                    SET is_archived = TRUE, updated_at = NOW()
                    WHERE source_path = $1 AND source_type = $2 AND is_archived = FALSE
                """
                await conn.execute(query, source_path, source_type)
            else:
                query = """
                    UPDATE memories
                    SET is_archived = TRUE, updated_at = NOW()
                    WHERE source_path = $1 AND is_archived = FALSE
                """
                await conn.execute(query, source_path)

            # Get affected row count
            count_query = """
                SELECT COUNT(*) FROM memories
                WHERE source_path = $1 AND is_archived = TRUE AND updated_at = NOW()
            """
            count = await conn.fetchval(count_query, source_path)
            return count or 0

    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Query memories using vector similarity.

        Args:
            vector: Query vector.
            options: Search options.
            collection_name: Optional collection override (unused in pgvector impl).

        Returns:
            List of matching MemoryEntry objects with scores.
        """
        await self.initialize()
        pool = await self._get_pool()

        # Convert vector to list format
        query_vector = vector if isinstance(vector, list) else list(vector)

        # Calculate threshold: pgvector <=> returns cosine distance (lower = better)
        # Convert similarity threshold to distance threshold: 1 - similarity = distance
        distance_threshold = 1.0 - options.threshold

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SEARCH_MEMORIES_VECTOR,
                query_vector,
                distance_threshold,
                options.top_k,
            )

            results = []
            for row in rows:
                distance = (
                    float(row["distance"]) if row["distance"] is not None else 0.0
                )
                score = 1.0 - distance
                results.append(self._row_to_memory(row, score=score))

            return results

    async def list_memories(
        self,
        filter: SearchFilter | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with optional filter.

        Args:
            filter: Optional search filter.

        Returns:
            List of MemoryEntry objects.
        """
        await self.initialize()
        pool = await self._get_pool()

        query = """
            SELECT id, text, embedding, source_type, content_hash, metadata,
                   trust_score, retrieval_count, is_archived, last_accessed_at
            FROM memories
            WHERE is_archived = FALSE
        """
        params: list[Any] = []

        if filter:
            if filter.source_type:
                query += " AND source_type = $1"
                params.append(
                    filter.source_type.value
                    if isinstance(filter.source_type, MemorySourceType)
                    else filter.source_type
                )

            if filter.session_id:
                query += f" AND metadata->>'session_id' = ${len(params) + 1}"
                params.append(filter.session_id)

            if filter.project_id:
                query += f" AND metadata->>'project_id' = ${len(params) + 1}"
                params.append(filter.project_id)

            if filter.since:
                query += f" AND created_at >= ${len(params) + 1}"
                params.append(filter.since)

        query += " ORDER BY created_at DESC LIMIT 100"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [self._row_to_memory(row) for row in rows]

    async def count_memories(self) -> int:
        """Count total memories.

        Returns:
            Number of memory entries.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_MEMORY_COUNT)
            return row["active"] if row else 0

    async def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists.

        Args:
            content_hash: SHA-256 hash to check.

        Returns:
            True if exists, False otherwise.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            query = """
                SELECT COUNT(*) FROM memories
                WHERE content_hash = $1 AND is_archived = FALSE
            """
            count = await conn.fetchval(query, content_hash)
            return (count or 0) > 0

    async def get_entries_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all entries with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            List of matching MemoryEntry objects.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if source_type:
                query = """
                    SELECT id, text, embedding, source_type, content_hash, metadata,
                           trust_score, retrieval_count, is_archived, last_accessed_at
                    FROM memories
                    WHERE source_path = $1 AND source_type = $2 AND is_archived = FALSE
                """
                rows = await conn.fetch(query, source_path, source_type)
            else:
                query = """
                    SELECT id, text, embedding, source_type, content_hash, metadata,
                           trust_score, retrieval_count, is_archived, last_accessed_at
                    FROM memories
                    WHERE source_path = $1 AND is_archived = FALSE
                """
                rows = await conn.fetch(query, source_path)

            return [self._row_to_memory(row) for row in rows]

    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Scroll through a collection.

        Args:
            collection_name: Collection to scroll (unused in pgvector impl).
            limit: Page size.

        Returns:
            List of MemoryEntry objects.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            query = """
                SELECT id, text, embedding, source_type, content_hash, metadata,
                       trust_score, retrieval_count, is_archived, last_accessed_at
                FROM memories
                WHERE is_archived = FALSE
                ORDER BY created_at DESC
                LIMIT $1
            """
            rows = await conn.fetch(query, limit)

            return [self._row_to_memory(row) for row in rows]

    async def get_collection_dimension(self, collection_name: str) -> int | None:
        """Get the dimension of a collection.

        Args:
            collection_name: Collection to check (unused in pgvector impl).

        Returns:
            Vector dimension (always 1024 for BGE-Large) or None if not supported.
        """
        # PostgreSQL pgvector always uses fixed dimension
        # Return 1024 for BGE-Large embeddings
        return 1024

    async def close(self) -> None:
        """Close the database connection."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False

    async def update_trust_fields(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        """Update trust fields for a memory entry.

        Args:
            memory_id: ID of the memory entry.
            trust_score: New trust score.
            is_archived: New archived status.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if is_archived:
                await conn.execute(
                    """
                    UPDATE memories
                    SET is_archived = TRUE, updated_at = NOW()
                    WHERE id = $1
                    """,
                    memory_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE memories
                    SET trust_score = $1, last_accessed_at = NOW()
                    WHERE id = $2
                    """,
                    trust_score,
                    memory_id,
                )

    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval count for a memory entry.

        Args:
            memory_id: ID of the memory entry.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(INCREMENT_RETRIEVAL_COUNT, memory_id)

    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Set payload fields for a memory entry.

        Args:
            memory_id: ID of the memory entry.
            payload: Dictionary of payload fields to set.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                UPDATE_MEMORY_METADATA,
                memory_id,
                json.dumps(payload),
            )


def create_postgres_database(
    db_url: str, project_id: str | None = None
) -> PostgresDatabase:
    """Factory function to create a PostgresDatabase instance.

    Args:
        db_url: PostgreSQL connection URL.
        project_id: Optional project ID for isolation.

    Returns:
        PostgresDatabase instance.
    """
    return PostgresDatabase(db_url=db_url, project_id=project_id)
