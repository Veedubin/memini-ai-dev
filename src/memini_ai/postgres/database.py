"""PostgreSQL/pgvector implementation of VectorDatabase using asyncpg.

This module provides the PostgresDatabase class that implements the VectorDatabase
ABC using asyncpg for async PostgreSQL operations with pgvector for vector storage.
It supports both pgvectorscale's StreamingDiskANN index (preferred) and pgvector's
HNSW index as a fallback when vectorscale is unavailable.

TLS/SSL Support:
    PostgreSQL connections can be secured with TLS by configuring db_sslmode and
    db_sslrootcert via environment variables DB_SSLMODE and DB_SSLROOTCERT, or
    through the MeminiConfig settings. When sslmode is 'require' or higher, an
    SSL context is created and passed to the asyncpg connection pool.
"""

from __future__ import annotations

import contextlib
import json
import ssl
import uuid
from typing import TYPE_CHECKING, Any

import asyncpg
from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

from memini_ai.config import get_config
from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchFilter,
    SearchOptions,
)
from memini_ai.postgres.driver import DatabaseDriver
from memini_ai.postgres.queries import (
    COUNT_BY_EMBEDDING_MODEL,
    COUNT_MEMORIES_1024,
    COUNT_THOUGHTS,
    DELETE_ENTITY,
    DELETE_MEMORY,
    DELETE_MEMORY_1024_BY_MEMORY_ID,
    DELETE_MEMORY_IMAGE,
    GET_ENTITIES,
    GET_ENTITIES_BY_TYPE,
    GET_ENTITIES_WITH_RELATIONSHIPS,
    GET_ENTITY_BY_ID,
    GET_ENTITY_STATS,
    GET_MEMORY_1024_BY_MEMORY_ID,
    GET_MEMORY_BY_ID,
    GET_MEMORY_BY_ID_INCLUDE_ARCHIVED,
    GET_MEMORY_COUNT,
    GET_SUPERSEDED_MEMORY,
    GET_SUPERSESSION_CHAIN,
    INCREMENT_RETRIEVAL_COUNT,
    INSERT_ENTITY_RELATIONSHIP,
    INSERT_MEMORY,
    INSERT_MEMORY_1024,
    INSERT_MEMORY_BGE_M3,
    INSERT_MEMORY_DELTA,
    INSERT_MEMORY_IMAGE,
    INSERT_MEMORY_WITH_MODEL,
    SEARCH_MEMORIES_1024_VECTOR,
    SEARCH_MEMORIES_BGE_M3,
    SEARCH_MEMORIES_IMAGE,
    SEARCH_MEMORIES_IMAGE_BY_SHA256,
    SEARCH_MEMORIES_MINILM,
    SEARCH_MEMORIES_VECTOR,
    UPDATE_MEMORY_IMAGE_TRUST,
    UPDATE_MEMORY_METADATA,
    UPSERT_ENTITY,
)
from memini_ai.postgres.schema import (
    SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN,
    SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW,
    SQL_CREATE_MEMORIES_IMAGE_INDEXES,
    SQL_CREATE_MEMORIES_IMAGE_TABLE,
    SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE,
    get_schema_sql,
)
from memini_ai.utils.logger import logger

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

    TLS/SSL:
        - Supports sslmode via DB_SSLMODE env var (default: prefer)
        - Supports CA cert path via DB_SSLROOTCERT env var
        - When sslmode is 'require' or higher, creates SSL context
        - Backward compatible: no SSL by default if not configured
    """

    # PostgreSQL sslmode values that require an SSL context
    _SSL_REQUIRED_MODES = {"require", "verify-ca", "verify-full"}

    def __init__(
        self,
        driver: DatabaseDriver,
        project_id: str | None = None,
        sslmode: str | None = None,
        sslrootcert: str | None = None,
    ) -> None:
        """Initialize PostgresDatabase.

        Args:
            driver: DatabaseDriver providing the connection URI and backend lifecycle.
            project_id: Optional project ID for isolation (unused in pgvector impl).
            sslmode: PostgreSQL SSL mode override. If None, reads from config/env.
            sslrootcert: Path to CA certificate for SSL verification.
                If None, reads from config/env.
        """
        self._driver = driver
        self._db_url: str | None = None  # resolved lazily in initialize()
        self._project_id = project_id
        self._pool: asyncpg.Pool | None = None
        self._initialized = False

        # Resolve SSL settings: explicit args > config > env > default
        config = get_config()
        self._sslmode = sslmode or config.db_sslmode
        self._sslrootcert = sslrootcert or config.db_sslrootcert
        self._dimension = config.embedding_dim

    async def initialize(self) -> None:
        """Initialize the database connection and create schema if needed.

        Creates an asyncpg connection pool with optional SSL support based on
        the configured db_sslmode. When sslmode is 'require' or higher, an
        SSL context is built and passed to the pool.

        Raises:
            RuntimeError: If connection pool creation or schema init fails.
        """
        if self._initialized:
            return

        try:
            # Resolve URI lazily from driver
            if self._db_url is None:
                self._db_url = await self._driver.get_uri()

            # Build pool kwargs
            pool_kwargs: dict[str, Any] = {
                "min_size": 1,
                "max_size": 10,
            }

            # Build SSL context if sslmode requires it
            ssl_context = self._build_ssl_context()
            if ssl_context is not None:
                pool_kwargs["ssl"] = ssl_context

            # Register pgvector type codec on every new connection via init callback.
            # This ensures all pool connections can bind Python lists to the vector
            # column type. Without this, only the first manually-acquired connection
            # has the codec, and subsequent pool connections fail with:
            #   "expected str, got list" when binding vector parameters.
            async def _init_conn(conn: asyncpg.Connection) -> None:
                await register_vector(conn)

            # Create connection pool with pgvector codec initializer
            self._pool = await asyncpg.create_pool(
                self._db_url,
                init=_init_conn,
                **pool_kwargs,
            )

            # Backend-specific initialization (e.g., EmbeddedPGDriver starts heartbeat)
            await self._driver.initialize()

            # Initialize schema
            await self._ensure_schema()

            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PostgreSQL connection: {e}"
            ) from e

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        """Build an SSL context based on the configured sslmode.

        Returns:
            An ssl.SSLContext if sslmode requires encryption, None otherwise.

        SSL mode behavior:
            - disable/allow: No SSL context (plaintext connection)
            - prefer: SSL context created but connection falls back to plaintext
            - require: SSL required, server identity not verified
            - verify-ca: SSL required, CA certificate verified
            - verify-full: SSL required, CA cert + hostname verified
        """
        if self._sslmode == "disable":
            return None

        if self._sslmode == "allow":
            # 'allow' means: try without SSL first, fall back to SSL on failure.
            # asyncpg doesn't support this negotiation natively, so we skip
            # SSL context creation. The pool will connect without SSL.
            return None

        if self._sslmode == "prefer":
            # 'prefer' means: try SSL first, fall back to plaintext.
            # asyncpg handles this via the ssl parameter — if we pass an
            # SSL context, it tries SSL first. However, for 'prefer' we
            # create a permissive context that doesn't verify the server cert,
            # matching libpq's prefer behavior.
            ctx = ssl.create_default_context()
            # Don't verify server cert for 'prefer' mode (matches libpq behavior)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx

        # sslmode in {require, verify-ca, verify-full}
        if self._sslrootcert:
            # Load custom CA certificate
            ctx = ssl.create_default_context(cafile=self._sslrootcert)
        else:
            # Use system default CA bundle
            ctx = ssl.create_default_context()

        if self._sslmode == "require":
            # 'require' encrypts but doesn't verify server identity
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        elif self._sslmode == "verify-ca":
            # 'verify-ca' verifies the CA but not the hostname
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_REQUIRED
        elif self._sslmode == "verify-full":
            # 'verify-full' verifies both CA and hostname
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED

        return ctx

    async def _detect_vectorscale(self) -> bool:
        """Check if pgvectorscale extension is available in the database.

        Returns:
            True if vectorscale extension can be created, False otherwise.
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vectorscale'"
            )
            return row is not None

    async def _ensure_schema(self) -> None:
        """Create database schema if tables don't exist (idempotent).

        Automatically detects whether pgvectorscale is available and uses
        StreamingDiskANN indexes when possible, falling back to HNSW indexes
        otherwise.
        """
        if self._pool is None:
            raise RuntimeError("Database pool not initialized")

        # Detect vectorscale availability and generate appropriate schema
        use_vectorscale = await self._detect_vectorscale()
        schema_sql = get_schema_sql(use_vectorscale=use_vectorscale)

        logger.info(
            "schema_initialization",
            use_vectorscale=use_vectorscale,
            index_type="diskann" if use_vectorscale else "hnsw",
        )

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
        """Convert MemoryEntry to database record dict.

        The vector is stored under a column-specific key (``embedding`` for
        MiniLM, ``embedding_bge_m3`` for BGE-M3) based on
        ``entry.embedding_model``.  For backwards compatibility, when the
        target column is ``embedding`` the vector is ALSO stored under the
        generic ``embedding`` key — matching the behaviour every caller
        relied on before v0.12.0.
        """
        from memini_ai.model.manager import MODEL_COLUMNS

        record: dict[str, Any] = {
            "id": entry.id or str(uuid.uuid4()),
            "text": entry.text,
            "source_type": entry.source_type.value
            if isinstance(entry.source_type, MemorySourceType)
            else entry.source_type,
        }

        # Convert vector to Python list for PostgreSQL pgvector via asyncpg
        # register_vector() (called in initialize) sets up proper type codec
        # so Python lists bind correctly to vector columns.
        vector_list: list[float] | None = None
        if entry.vector is not None:
            import numpy as np

            if isinstance(entry.vector, np.ndarray):
                vector_list = entry.vector.tolist()
            else:
                vector_list = list(entry.vector)

        # Determine the target column for the vector based on the embedding
        # model. Default to the 384-dim ``embedding`` column when no model is
        # specified (backwards compat with pre-v0.12.0 callers).
        target_column = "embedding"
        if entry.embedding_model is not None:
            target_column = MODEL_COLUMNS.get(entry.embedding_model, "embedding")

        record[target_column] = vector_list
        # Keep the legacy ``embedding`` key populated for backwards compat when
        # the target column IS ``embedding`` — many callers read
        # record["embedding"] directly.
        if target_column == "embedding":
            record["embedding"] = vector_list
        else:
            # Ensure the generic ``embedding`` key is still present (NULL) so
            # downstream code that unconditionally reads it doesn't KeyError.
            record["embedding"] = None

        record["content_hash"] = entry.content_hash or ""
        record["metadata"] = {}

        if entry.metadata_json:
            with contextlib.suppress(json.JSONDecodeError):
                record["metadata"] = json.loads(entry.metadata_json)

        # Delta model fields
        record["supersedes_id"] = entry.supersedes_id
        record["structured_fields"] = (
            json.dumps(entry.structured_fields)
            if entry.structured_fields is not None
            else None
        )
        record["change_ratio"] = entry.change_ratio
        record["created_at_ms"] = entry.created_at_ms
        record["embedding_model"] = entry.embedding_model

        return record

    def _row_to_memory(
        self,
        row: asyncpg.Record,
        score: float | None = None,
    ) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        # Parse vector from string format if needed (pgvector returns string '[0.1, 0.2, ...]')
        embedding = row["embedding"]
        if embedding is not None:
            if isinstance(embedding, str):
                # Parse string format '[0.1, 0.2, ...]' to list
                vector = json.loads(embedding)
            else:
                vector = list(embedding)
        else:
            vector = None

        data = {
            "id": str(row["id"]),
            "text": row["text"],
            "vector": vector,
            "source_type": row["source_type"],
            "content_hash": row.get("content_hash") or "",
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
            # Delta model fields
            "supersedes_id": row.get("supersedes_id"),
            "structured_fields": row.get("structured_fields"),
            "change_ratio": row.get("change_ratio", 1.0),
            "created_at_ms": row.get("created_at_ms"),
            "embedding_model": row.get("embedding_model"),
        }

        return MemoryEntry.model_validate(data)

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry.

        The vector is written to the column matching the model's
        dimensionality, selected via ``entry.embedding_model``:

          * ``all-MiniLM-L6-v2`` (or None) → ``embedding`` (384-dim)
          * ``BAAI/bge-m3`` → ``embedding_bge_m3`` (1024-dim)

        Args:
            entry: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.
        """
        from memini_ai.model.manager import MODEL_COLUMNS

        await self.initialize()
        pool = await self._get_pool()

        record = self._entry_to_record(entry)
        memory_id = record["id"]

        # Determine which column holds the vector for this entry
        target_column = "embedding"
        if entry.embedding_model is not None:
            target_column = MODEL_COLUMNS.get(entry.embedding_model, "embedding")
        vector_value = record.get(target_column)

        async with pool.acquire() as conn:
            # Use INSERT_MEMORY_DELTA if delta fields are present
            if (
                record.get("supersedes_id") is not None
                or record.get("structured_fields") is not None
                or record.get("change_ratio", 1.0) != 1.0
            ):
                # Delta inserts always go to the 384-dim embedding column
                # (delta model predates multi-model support)
                memory_id = await conn.fetchval(
                    INSERT_MEMORY_DELTA,
                    memory_id,
                    record["text"],
                    record["embedding"],
                    record["source_type"],
                    record["content_hash"],
                    json.dumps(record["metadata"]),
                    record["supersedes_id"],
                    record["structured_fields"],
                    record["change_ratio"],
                    record["created_at_ms"],
                )
            elif target_column == "embedding_bge_m3":
                # BGE-M3 (1024-dim) → write to embedding_bge_m3 column
                memory_id = await conn.fetchval(
                    INSERT_MEMORY_BGE_M3,
                    memory_id,
                    record["text"],
                    vector_value,
                    record["source_type"],
                    record["content_hash"],
                    json.dumps(record["metadata"]),
                    record["created_at_ms"],
                    record["embedding_model"],
                )
            elif record.get("embedding_model") is not None:
                # MiniLM with embedding_model tracking → INSERT_MEMORY_WITH_MODEL
                memory_id = await conn.fetchval(
                    INSERT_MEMORY_WITH_MODEL,
                    memory_id,
                    record["text"],
                    record["embedding"],
                    record["source_type"],
                    record["content_hash"],
                    json.dumps(record["metadata"]),
                    record["created_at_ms"],
                    record["embedding_model"],
                )
            else:
                # Legacy: no embedding_model, 384-dim embedding column
                memory_id = await conn.fetchval(
                    INSERT_MEMORY,
                    memory_id,
                    record["text"],
                    record["embedding"],
                    record["source_type"],
                    record["content_hash"],
                    json.dumps(record["metadata"]),
                    record["created_at_ms"],
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
                    record["created_at_ms"],
                )

        return entries

    async def get_memory(
        self,
        memory_id: str,
        include_archived: bool = False,
    ) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.
            include_archived: If True, include archived memories (default False).

        Returns:
            MemoryEntry if found, None otherwise.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if include_archived:
                row = await conn.fetchrow(GET_MEMORY_BY_ID_INCLUDE_ARCHIVED, memory_id)
            else:
                row = await conn.fetchrow(GET_MEMORY_BY_ID, memory_id, include_archived)
            if row is None:
                return None

            return self._row_to_memory(row)

    async def get_supersession_chain(
        self,
        memory_id: str,
        max_depth: int = 10,
    ) -> list[MemoryEntry]:
        """Get the full supersession chain for a memory.

        Args:
            memory_id: ID of the memory entry.
            max_depth: Maximum chain depth (default 10).

        Returns:
            List of MemoryEntry objects in the supersession chain (oldest first).
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(GET_SUPERSESSION_CHAIN, memory_id, max_depth)
            return [self._row_to_memory(row) for row in rows]

    async def get_superseded_memory(
        self,
        memory_id: str,
    ) -> MemoryEntry | None:
        """Get the memory that this memory supersedes (parent).

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry of the superseded memory if found, None otherwise.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_SUPERSEDED_MEMORY, memory_id)
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

        # Convert vector to Python list for asyncpg + pgvector binding
        # asyncpg natively binds Python lists to PostgreSQL arrays;
        # the SQL query casts $1::vector, which accepts array input.
        query_vector = vector if isinstance(vector, list) else list(vector)

        # Calculate threshold: pgvector <=> returns cosine distance (lower = better)
        # Convert similarity threshold to distance threshold: 1 - similarity = distance
        distance_threshold = 1.0 - options.threshold

        async with pool.acquire() as conn:
            async with conn.transaction():
                # When exact_search is True, disable the approximate DiskANN
                # index to guarantee exact nearest neighbor results.
                # Requires a transaction for SET LOCAL to take effect.
                if options.exact_search:
                    await conn.execute("SET LOCAL enable_indexscan = off")

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

    async def search_memories_rrf(
        self,
        query_vectors: dict[str, list[float]],
        options: SearchOptions,
        enabled_columns: dict[str, str] | None = None,
    ) -> list[MemoryEntry]:
        """Search memories across multiple model vector spaces and fuse via RRF.

        For each enabled model, runs top-k vector search in that model's column,
        collects the ranked results, then merges using Reciprocal Rank Fusion.

        Args:
            query_vectors: Dict mapping model name to query vector.
                e.g. {"all-MiniLM-L6-v2": [0.1, ...], "BAAI/bge-m3": [0.2, ...]}
            options: Search options (top_k, threshold, etc.).
            enabled_columns: Optional dict mapping model name to column name.
                If None, uses both: embedding, embedding_bge_m3.

        Returns:
            List of MemoryEntry objects ordered by fused RRF score.
        """
        await self.initialize()
        pool = await self._get_pool()

        if enabled_columns is None:
            enabled_columns = {
                "all-MiniLM-L6-v2": "embedding",
                "BAAI/bge-m3": "embedding_bge_m3",
            }

        # Map column names to their search SQL
        column_to_query = {
            "embedding": SEARCH_MEMORIES_MINILM,
            "embedding_bge_m3": SEARCH_MEMORIES_BGE_M3,
        }

        config = get_config()
        top_k_per_model = max(options.top_k * 2, config.rrf_top_k_per_model)
        distance_threshold = 1.0 - options.threshold

        # Collect ranked results from each model space
        # model_name -> list of (memory_id, row, distance)
        per_model_results: dict[str, list[tuple[str, asyncpg.Record, float]]] = {}

        async with pool.acquire() as conn:
            for model_name, column in enabled_columns.items():
                query_vec = query_vectors.get(model_name)
                if query_vec is None:
                    continue
                sql = column_to_query.get(column)
                if sql is None:
                    continue
                try:
                    rows = await conn.fetch(
                        sql,
                        query_vec,
                        distance_threshold,
                        top_k_per_model,
                    )
                except Exception:
                    # Column might not exist or vector is wrong dim — skip
                    continue
                per_model_results[model_name] = [
                    (str(row["id"]), row, float(row["distance"])) for row in rows
                ]

        if not per_model_results:
            return []

        # RRF fusion: build ranked ID lists
        ranked_lists: list[list[str]] = []
        all_entries: dict[str, MemoryEntry] = {}
        all_distances: dict[str, float] = {}

        for _model_name, results in per_model_results.items():
            ranked_ids: list[str] = []
            for mid, row, dist in results:
                ranked_ids.append(mid)
                if mid not in all_entries:
                    mem_entry = self._row_to_memory(row, score=1.0 - dist)
                    all_entries[mid] = mem_entry
                    all_distances[mid] = dist
                else:
                    # Keep the smallest (best) distance
                    if dist < all_distances.get(mid, float("inf")):
                        all_distances[mid] = dist
            ranked_lists.append(ranked_ids)

        # Use the RRF helper to fuse
        from memini_ai.memory.rrf import rrf_with_limit

        fused_ids = rrf_with_limit(ranked_lists, k=config.rrf_k, limit=options.top_k)

        # Return entries in fused order with RRF score as score
        result: list[MemoryEntry] = []
        for mid in fused_ids:
            entry: MemoryEntry | None = all_entries.get(mid)
            if entry is not None:
                # Score is the best cosine similarity across models
                best_sim = 1.0 - all_distances.get(mid, 1.0)
                result.append(entry.model_copy(update={"score": best_sim}))
        return result

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
                   trust_score, retrieval_count, is_archived, last_accessed_at,
                   supersedes_id, structured_fields, change_ratio, created_at_ms
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
                           trust_score, retrieval_count, is_archived, last_accessed_at,
                           supersedes_id, structured_fields, change_ratio, created_at_ms
                    FROM memories
                    WHERE source_path = $1 AND source_type = $2 AND is_archived = FALSE
                """
                rows = await conn.fetch(query, source_path, source_type)
            else:
                query = """
                    SELECT id, text, embedding, source_type, content_hash, metadata,
                           trust_score, retrieval_count, is_archived, last_accessed_at,
                           supersedes_id, structured_fields, change_ratio, created_at_ms
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
                       trust_score, retrieval_count, is_archived, last_accessed_at,
                       supersedes_id, structured_fields, change_ratio, created_at_ms
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
            Vector dimension from config (384 for MiniLM, 1024 for BGE-M3)
            or None if not supported.
        """
        # PostgreSQL pgvector always uses fixed dimension from config
        return self._dimension

    async def close(self) -> None:
        """Close the database connection."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        await self._driver.shutdown()

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

    # =========================================================================
    # Entity (Knowledge Graph) CRUD Operations
    # =========================================================================

    async def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        canonical_name: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert or update an entity in the knowledge graph.

        Args:
            entity_id: Unique entity ID.
            name: Surface form (as found in text).
            entity_type: Type classification (PERSON, ORGANIZATION, etc.).
            canonical_name: Canonical/normalised form.
            confidence: Extraction confidence 0.0-1.0.
            metadata: Optional metadata dict.

        Returns:
            The entity ID.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                UPSERT_ENTITY,
                entity_id,
                name,
                entity_type,
                canonical_name,
                confidence,
                json.dumps(metadata or {}),
            )
            return str(result)

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity dict if found, None otherwise.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_ENTITY_BY_ID, entity_id)
            if row is None:
                return None

            return {
                "id": str(row["id"]),
                "name": row["name"],
                "entity_type": row["entity_type"],
                "canonical_name": row["canonical_name"],
                "confidence": row["confidence"],
                "mention_count": row["mention_count"],
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
                "metadata": row["metadata"],
            }

    async def get_entities(
        self,
        limit: int = 1000,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all entities, optionally filtered by type.

        Args:
            limit: Maximum number of entities to return.
            entity_type: Optional entity type filter.

        Returns:
            List of entity dicts.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    GET_ENTITIES_BY_TYPE,
                    entity_type,
                    limit,
                )
            else:
                rows = await conn.fetch(GET_ENTITIES, limit)

            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "canonical_name": row["canonical_name"],
                    "confidence": row["confidence"],
                    "mention_count": row["mention_count"],
                    "first_seen_at": row["first_seen_at"],
                    "last_seen_at": row["last_seen_at"],
                    "metadata": row["metadata"],
                }
                for row in rows
            ]

    async def get_entities_with_relationships(
        self,
        limit: int = 1000,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Get entities and relationships for D3.js visualization.

        Args:
            limit: Maximum number of entities to return.

        Returns:
            Tuple of (nodes, edges) for D3.js force graph.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(GET_ENTITIES_WITH_RELATIONSHIPS, limit)

            # Build nodes and edges
            nodes_map: dict[str, dict[str, Any]] = {}
            edges: list[dict[str, Any]] = []

            for row in rows:
                entity_id = str(row["id"])
                if entity_id not in nodes_map:
                    nodes_map[entity_id] = {
                        "id": entity_id,
                        "name": row["canonical_name"] or row["name"],
                        "type": row["entity_type"],
                        "confidence": row["confidence"],
                        "group": self._get_entity_group(row["entity_type"]),
                    }

                if row["target_entity_id"]:
                    edges.append(
                        {
                            "source": entity_id,
                            "target": str(row["target_entity_id"]),
                            "relationship": row["relationship_type"],
                            "confidence": row["rel_confidence"],
                            "stroke": self._get_rel_color(row["relationship_type"]),
                        }
                    )

            return list(nodes_map.values()), edges

    async def get_entity_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dict with entity counts by type.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_ENTITY_STATS)
            return dict(row) if row else {}

    async def upsert_entity_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence: float = 1.0,
    ) -> str:
        """Insert or update an entity relationship.

        Args:
            source_entity_id: Source entity ID.
            target_entity_id: Target entity ID.
            relationship_type: Relationship type (SUPERSEDES, RELATED_TO, etc.).
            confidence: Relationship confidence 0.0-1.0.

        Returns:
            Relationship ID.
        """
        await self.initialize()
        pool = await self._get_pool()

        import uuid

        async with pool.acquire() as conn:
            rel_id = str(uuid.uuid4())
            result = await conn.fetchval(
                INSERT_ENTITY_RELATIONSHIP,
                rel_id,
                source_entity_id,
                target_entity_id,
                relationship_type,
                confidence,
            )
            return str(result)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships.

        Args:
            entity_id: Entity ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.fetchval(DELETE_ENTITY, entity_id)
            return result is not None

    def _get_entity_group(self, entity_type: str) -> int:
        """Map entity type to D3 group number for visualization."""
        group_map = {
            "PERSON": 1,
            "ORGANIZATION": 2,
            "CONCEPT": 3,
            "CODE": 4,
            "PROJECT": 5,
            "LOCATION": 6,
            "UNKNOWN": 0,
        }
        return group_map.get(entity_type, 0)

    def _get_rel_color(self, rel_type: str) -> str:
        """Get color for relationship type."""
        color_map = {
            "SUPERSEDES": "#e74c3c",
            "RELATED_TO": "#3498db",
            "CONTRADICTS": "#9b59b6",
            "DERIVED_FROM": "#27ae60",
        }
        return color_map.get(rel_type, "#95a5a6")

    # =============================================================================
    # Dual-Model RRF methods (v0.7.0)
    # =============================================================================
    #
    # The methods below operate on the memories_1024 sidecar table, which
    # holds 1024-dim embeddings for "elevated" memories. The 384-dim
    # memories table is always the source of truth — these are quality-boost
    # sidecars used in the AUTO mode of the dual-model RRF system.

    @staticmethod
    def _expand_384_to_1024(
        vector_384: list[float] | None,
        target_dim: int = 1024,
    ) -> list[float] | None:
        """Placeholder 384→1024 dimension expander.

        v0.7.0 ships with a deterministic placeholder: zero-pad the 384-dim
        vector up to 1024 dims and L2-normalize the result. This is NOT a
        real re-embedding — it's a stable stand-in so the
        memories_1024 table can be populated, queried, and RRF-fused without
        pulling in a second embedding model. A future version will swap this
        for an actual 1024-dim model call when the elevate tool is invoked.

        Args:
            vector_384: 384-dim source vector (Python list or None).
            target_dim: Target dimension (default 1024).

        Returns:
            1024-dim vector (zero-padded + L2-normalized), or None if input is None.
        """
        if vector_384 is None:
            return None
        if len(vector_384) >= target_dim:
            # Already big enough — truncate (shouldn't happen in practice)
            return list(vector_384[:target_dim])
        # Zero-pad to target_dim
        padded = list(vector_384) + [0.0] * (target_dim - len(vector_384))
        # L2 normalize so cosine distance behaves like cosine similarity
        import math

        norm = math.sqrt(sum(x * x for x in padded))
        if norm == 0.0:
            return padded
        return [x / norm for x in padded]

    async def add_memory_1024(
        self,
        memory_id: str,
        vector_1024: list[float],
        trust_score: float = 0.5,
        embedding_model: str = "placeholder-1024",
    ) -> str | None:
        """Insert (or no-op if already present) a 1024-dim sidecar for a memory.

        Idempotent: uses ON CONFLICT (memory_id) DO NOTHING. Re-elevating the
        same memory is a no-op (the original 1024 vector is preserved).

        Args:
            memory_id: UUID of the source 384-dim memory (must exist).
            vector_1024: 1024-dim embedding vector.
            trust_score: Trust score for the 1024 copy (default 0.5).
            embedding_model: Model name to record (default placeholder).

        Returns:
            The 1024-row id if inserted, None if already existed.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                INSERT_MEMORY_1024,
                memory_id,
                vector_1024,
                float(trust_score),
                embedding_model,
            )
        return str(row_id) if row_id else None

    async def query_memories_1024(
        self,
        vector_1024: list[float],
        threshold: float = 0.5,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search the 1024-dim sidecar table by vector similarity.

        Joins back to the 384-dim memories table so the returned MemoryEntry
        has full text/metadata. Results are ordered by cosine distance ASC.

        Args:
            vector_1024: 1024-dim query vector.
            threshold: Max cosine distance (default 0.5 = fairly strict).
            limit: Max results (default 10).

        Returns:
            List of MemoryEntry objects with `score` set to cosine distance.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SEARCH_MEMORIES_1024_VECTOR,
                vector_1024,
                float(threshold),
                int(limit),
            )

        results: list[MemoryEntry] = []
        for row in rows:
            # Reuse _row_to_memory for the joined-in memories row,
            # then attach the 1024 distance as the score.
            entry = self._row_to_memory(row, score=float(row["distance"]))
            results.append(entry)
        return results

    async def get_memory_1024_by_memory_id(
        self, memory_id: str
    ) -> dict[str, Any] | None:
        """Look up the 1024-dim sidecar for a specific memory.

        Args:
            memory_id: UUID of the 384-dim memory.

        Returns:
            Dict with keys (id, memory_id, embedding, elevated_at,
            elevated_from_dim, embedding_model, trust_score) or None if
            the memory has not been elevated.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(GET_MEMORY_1024_BY_MEMORY_ID, memory_id)

        if row is None:
            return None
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        elif embedding is not None:
            embedding = list(embedding)
        return {
            "id": str(row["id"]),
            "memory_id": str(row["memory_id"]),
            "embedding": embedding,
            "elevated_at": row["elevated_at"],
            "elevated_from_dim": row["elevated_from_dim"],
            "embedding_model": row["embedding_model"],
            "trust_score": float(row["trust_score"]),
        }

    async def elevate_memory_to_1024(
        self,
        memory_id: str,
        vector_1024: list[float] | None = None,
        trust_boost: float = 0.10,
    ) -> dict[str, Any]:
        """Promote a memory from 384-dim-only to also exist in 1024-dim space.

        Behavior:
            1. Verify the source 384-dim memory exists.
            2. If no 1024 vector provided, derive one from the 384-dim
               vector using `_expand_384_to_1024` (placeholder expansion).
            3. Insert into memories_1024 (idempotent).
            4. Boost trust_score on the 384-dim row by `trust_boost` (clamped
               to [0, 1]). The 1024-dim row stores the boosted value too.

        Args:
            memory_id: UUID of the 384-dim memory to elevate.
            vector_1024: Optional pre-computed 1024-dim embedding. If None,
                derived from the 384-dim vector via placeholder expansion.
            trust_boost: Amount to add to trust_score on elevate (default +0.10).

        Returns:
            Dict with keys:
                - memory_id: str
                - elevated: bool (False if already elevated)
                - trust_score: float (new boosted score)
                - vector_dim: int (always 1024 in v0.7.0)

        Raises:
            ValueError: If the source memory does not exist.
        """
        await self.initialize()
        pool = await self._get_pool()

        # Fetch the source 384-dim memory
        source = await self.get_memory(memory_id, include_archived=True)
        if source is None:
            raise ValueError(f"Memory {memory_id} not found")

        # Derive 1024 vector if not provided
        if vector_1024 is None:
            if source.vector is None:
                raise ValueError(
                    f"Memory {memory_id} has no 384-dim vector; cannot elevate"
                )
            vector_1024 = self._expand_384_to_1024(list(source.vector), 1024)
            if vector_1024 is None:
                raise ValueError(f"Memory {memory_id} vector expansion returned None")

        # Compute the new (boosted) trust score, clamped to [0, 1]
        new_trust = max(0.0, min(1.0, source.trust_score + trust_boost))

        async with pool.acquire() as conn, conn.transaction():
            # Idempotent insert into memories_1024
            inserted_id = await conn.fetchval(
                INSERT_MEMORY_1024,
                memory_id,
                vector_1024,
                new_trust,
                "placeholder-1024",
            )
            elevated = inserted_id is not None

            # Always bump trust on the 384-dim record (even if already elevated,
            # so re-elevation reaffirms importance). Capped at 1.0.
            await conn.execute(
                "UPDATE memories SET trust_score = $1, "
                "last_accessed_at = NOW() WHERE id = $2",
                new_trust,
                memory_id,
            )

        return {
            "memory_id": memory_id,
            "elevated": elevated,
            "trust_score": new_trust,
            "vector_dim": 1024,
        }

    async def count_memories_1024(self) -> int:
        """Count rows in the 1024-dim sidecar table.

        Returns:
            Number of memories that have been elevated to 1024-dim.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(COUNT_MEMORIES_1024)
        return int(row["count"]) if row else 0

    async def count_thoughts(self) -> int:
        """Count total thoughts in the ``thoughts`` table.

        Returns:
            Number of thought rows. Returns 0 if the table is empty or
            the database is uninitialized.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(COUNT_THOUGHTS)
        return int(row["count"]) if row else 0

    async def delete_memory_1024(self, memory_id: str) -> str | None:
        """Remove the 1024-dim sidecar for a memory (demote).

        Does NOT touch the 384-dim record. Idempotent — no error if the memory
        was never elevated.

        Args:
            memory_id: UUID of the 384-dim memory.

        Returns:
            The memory_id that was demoted, or None if there was no 1024 copy.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchval(DELETE_MEMORY_1024_BY_MEMORY_ID, memory_id)
        return str(row) if row else None

    # =========================================================================
    # Image Recall RRF methods (v0.8.0)
    # =========================================================================

    async def ensure_memories_image_table(self) -> None:
        """Create the ``memories_image`` table + indexes if absent.

        Idempotent: ``CREATE ... IF NOT EXISTS``. Uses DiskANN index when
        pgvectorscale is available, else HNSW. Also applies the extended
        ``source_type`` CHECK constraint (adds ``'image'``). Safe to call
        at startup regardless of ``MEMINI_IMAGE_SEARCH_ENABLED`` — the
        table exists but is empty/unqueried until the feature is enabled.
        This ensures videre-mcp can write image rows via the
        ``memini-vision`` library without memini-ai needing image search on.
        """
        await self.initialize()
        pool = await self._get_pool()
        use_vectorscale = await self._detect_vectorscale()
        image_index = (
            SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN
            if use_vectorscale
            else SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW
        )
        async with pool.acquire() as conn:
            await conn.execute(SQL_CREATE_MEMORIES_IMAGE_TABLE)
            await conn.execute(image_index)
            await conn.execute(SQL_CREATE_MEMORIES_IMAGE_INDEXES)
            await conn.execute(SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE)

    async def insert_image_memory(
        self,
        memory_id: str,
        embedding: list[float],
        image_path: str,
        image_sha256: str,
        mime_type: str,
        embedding_model: str = "placeholder-768",
        width: int | None = None,
        height: int | None = None,
        caption: str | None = None,
        file_size_bytes: int | None = None,
        trust_score: float = 0.5,
    ) -> str | None:
        """Insert (or no-op if already present) an image sidecar for a memory.

        Idempotent: uses ON CONFLICT (memory_id) DO NOTHING. The embedding
        must be 768-dim (ViT-B/32 zero-padded or ViT-L/14 native).

        Args:
            memory_id: UUID of the source memories row (must exist).
            embedding: 768-dim CLIP embedding vector.
            image_path: Absolute filesystem path to the stored image.
            image_sha256: SHA-256 hex digest of the image bytes.
            mime_type: MIME type (e.g. ``image/png``).
            embedding_model: CLIP model ID (default placeholder).
            width: Image width in pixels (optional).
            height: Image height in pixels (optional).
            caption: Optional human-readable caption.
            file_size_bytes: File size in bytes (optional).
            trust_score: Trust score for the image copy (default 0.5).

        Returns:
            The image-row id if inserted, None if already existed.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                INSERT_MEMORY_IMAGE,
                memory_id,
                embedding,
                embedding_model,
                image_path,
                image_sha256,
                mime_type,
                width,
                height,
                caption,
                file_size_bytes,
                float(trust_score),
            )
        return str(row_id) if row_id else None

    async def search_image_memories(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.9,
    ) -> list[MemoryEntry]:
        """Search the ``memories_image`` table by CLIP vector similarity.

        Joins back to the ``memories`` table so the returned MemoryEntry
        has full text/metadata. Results are ordered by cosine distance ASC.

        Args:
            query_embedding: 768-dim CLIP query vector.
            limit: Max results (default 10).
            threshold: Max cosine distance (default 0.9 = permissive; RRF
                re-ranks anyway).

        Returns:
            List of MemoryEntry objects with ``score`` set to cosine
            distance. The ``source_path`` field carries the image_path
            so callers can locate the image file on disk.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                SEARCH_MEMORIES_IMAGE,
                query_embedding,
                float(threshold),
                int(limit),
            )

        results: list[MemoryEntry] = []
        for row in rows:
            entry = self._row_to_memory(row, score=float(row["distance"]))
            # Attach image_path via source_path so the caller can locate the file
            image_path = row.get("image_path")
            if image_path is not None:
                entry.source_path = str(image_path)
            results.append(entry)
        return results

    async def get_image_by_sha256(self, image_sha256: str) -> dict[str, Any] | None:
        """Look up an image row by SHA-256 (idempotent re-insertion check).

        Args:
            image_sha256: SHA-256 hex digest of the image bytes.

        Returns:
            Dict with keys (id, memory_id, image_path, image_sha256,
            mime_type, caption, created_at) or None if not found.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(SEARCH_MEMORIES_IMAGE_BY_SHA256, image_sha256)
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "memory_id": str(row["memory_id"]),
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "mime_type": row["mime_type"],
            "caption": row["caption"],
            "created_at": row["created_at"],
        }

    async def delete_memory_image(self, memory_id: str) -> str | None:
        """Remove the image sidecar for a memory.

        Does NOT touch the parent memories row. Idempotent — no error if
        the memory had no image row.

        Args:
            memory_id: UUID of the parent memories row.

        Returns:
            The memory_id whose image was deleted, or None if there was
            no image row.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchval(DELETE_MEMORY_IMAGE, memory_id)
        return str(row) if row else None

    async def update_memory_image_trust(
        self, memory_id: str, trust_score: float
    ) -> str | None:
        """Update the trust score on an image row (trust engine integration).

        The canonical trust lives on ``memories.trust_score``; this is a
        denormalized cache so the image RRF arm can filter without a join.

        Args:
            memory_id: UUID of the parent memories row.
            trust_score: New trust score (0.0-1.0).

        Returns:
            The memory_id whose image trust was updated, or None if there
            was no image row.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchval(
                UPDATE_MEMORY_IMAGE_TRUST, memory_id, float(trust_score)
            )
        return str(row) if row else None

    async def count_by_embedding_model(self) -> dict[str, int]:
        """Count memories per embedding model (multi-model stats).

        Returns:
            Dict with keys: minilm_count, bge_m3_count,
            model_tracked_count.
        """
        await self.initialize()
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(COUNT_BY_EMBEDDING_MODEL)
        if row is None:
            return {
                "minilm_count": 0,
                "bge_m3_count": 0,
                "model_tracked_count": 0,
            }
        return {
            "minilm_count": int(row["minilm_count"]),
            "bge_m3_count": int(row["bge_m3_count"]),
            "model_tracked_count": int(row["model_tracked_count"]),
        }


def create_postgres_database(
    db_url: str,
    project_id: str | None = None,
    sslmode: str | None = None,
    sslrootcert: str | None = None,
) -> PostgresDatabase:
    """Factory function to create a PostgresDatabase instance.

    Args:
        db_url: PostgreSQL connection URL.
        project_id: Optional project ID for isolation.
        sslmode: PostgreSQL SSL mode override. If None, reads from config/env.
        sslrootcert: Path to CA certificate for SSL verification.

    Returns:
        PostgresDatabase instance.
    """
    from memini_ai.postgres.driver import ExternalPGDriver

    driver = ExternalPGDriver(db_url)
    return PostgresDatabase(
        driver=driver,
        project_id=project_id,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )
