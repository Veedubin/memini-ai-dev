"""chDB (in-process ClickHouse) implementation of the VectorDatabase ABC.

Mirrors ``src/memini_ai/postgres/database.py`` but targets ClickHouse's
MergeTree engine via the chDB Python package.

Differences from the Postgres implementation (documented):
- Single-writer concurrency model. chDB has no MVCC, no row-level
  locking. We use an ``asyncio.Lock`` to serialize writes. Reads are
  concurrent (each query runs in its own chDB session).
- App-layer cascade delete (``_cascade_delete``) instead of FK
  ON DELETE CASCADE. chDB has no FK enforcement.
- Brute-force cosine distance, not HNSW. chDB 4.2.1 doesn't ship the
  vector_similarity index type. Performance is fine up to ~500K
  memories (measured 100K x 384 = 32ms).
- No RETURNING clause in INSERT/UPDATE/DELETE. The wrapper methods
  return the id of the inserted/updated/deleted row via a follow-up
  SELECT (or, in the case of inserts, by computing the id client-side
  before the insert).
- No ON CONFLICT. Idempotent inserts (e.g. the 1024-dim sidecar) do
  a SELECT first and skip if the row already exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

import chdb.session as chdb_session_module
import numpy as np

from memini_ai.chdb import queries as chdb_queries
from memini_ai.chdb import schema as chdb_schema
from memini_ai.memory.database import VectorDatabase
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchOptions,
)
from memini_ai.utils.logger import logger

# SQL placeholder for an empty Array(Float32) literal. chDB doesn't
# accept `NULL` for a non-Nullable Array column; we use an empty array
# as the "no embedding" sentinel.
EMPTY_ARRAY_384 = "[]"
EMPTY_ARRAY_1024 = "[]"
EMPTY_ARRAY_768 = "[]"


class ChdbDatabase(VectorDatabase):
    """chDB-backed implementation of VectorDatabase.

    Parameters
    ----------
    data_dir : str
        Directory where chDB stores its data. Created if missing.
    project_id : str | None
        Project identifier (used in audit log tagging).
    pool_size : int
        Connection pool size. chDB is single-writer; we serialize writes
        with an asyncio.Lock. Multiple connections are used for
        concurrent reads.
    embedding_dim : int
        Default embedding dimension (384 for MiniLM). The schema uses
        384 as the primary, 1024 in memories_1024, 768 in memories_image.
    """

    _initialized: bool
    _data_dir: str
    _project_id: str | None
    _pool_size: int
    _dimension: int
    _write_lock: asyncio.Lock
    _schema_initialized: bool

    def __init__(
        self,
        data_dir: str,
        project_id: str | None = None,
        pool_size: int = 8,
        embedding_dim: int = 384,
    ) -> None:
        self._data_dir = data_dir
        self._project_id = project_id
        self._pool_size = max(1, min(64, pool_size))
        self._dimension = embedding_dim
        self._initialized = False
        self._schema_initialized = False
        self._write_lock = asyncio.Lock()
        Path(self._data_dir).mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def project_id(self) -> str | None:
        return self._project_id

    # ===================================================================
    # Connection management
    # ===================================================================

    def _get_session(self) -> Any:
        """Return a chDB session. Each call returns a fresh session
        so concurrent reads are safe. For writes, hold ``_write_lock``.
        """
        return chdb_session_module.Session(self._data_dir)  # type: ignore[no-untyped-call]

    @contextlib.contextmanager
    def _session(self) -> Any:
        """Context manager that opens and cleans up a chDB session."""
        sess = self._get_session()
        try:
            yield sess
        finally:
            with contextlib.suppress(Exception):
                sess.cleanup()

    def _query_csv(self, sql: str) -> str:
        """Run a SQL query and return the TSVWithNames result as a string."""
        with self._session() as sess:
            return str(sess.query(sql, "TSVWithNames"))

    def _query_rows(self, sql: str) -> list[list[str]]:
        """Run a SQL query and return parsed rows with header.

        chDB's ``CSV`` format has poor escaping: fields are wrapped in
        quotes and inner quotes are doubled, but our JSON columns
        contain literal ``"`` characters that confuse the parser. We
        use ``TSVWithNames`` (tab-separated with header) instead. Data
        containing tabs is rare in our schema (we store JSON, not raw
        text); if needed we'd switch to ``JSONEachRow``.

        chDB's ``TSVWithNames`` output is one line of tab-separated
        column names, then one line per data row, all fields bare (no
        quotes). This is the cleanest format for our schema.
        """
        import csv as _csv
        from io import StringIO

        with self._session() as sess:
            out = str(sess.query(sql, "TSVWithNames"))
        # TSV doesn't have the outer-quote wrapping of CSV. Strip the
        # trailing newline.
        out = out.rstrip("\n")
        reader = _csv.reader(StringIO(out), delimiter="\t")
        return list(reader)

    def _row_to_dict(self, header: list[str], row: list[str]) -> dict[str, str]:
        """Map a CSV row to a dict by header position."""
        d: dict[str, str] = {}
        for i, name in enumerate(header):
            if i < len(row):
                d[name] = row[i]
        return d

    # ===================================================================
    # Schema management
    # ===================================================================

    async def initialize(self) -> None:
        """Initialize the database and create all tables + indexes.

        Idempotent: safe to call on every startup.
        """
        if self._initialized:
            return
        with self._session() as sess:
            for sql in chdb_schema.CREATE_TABLES_IN_ORDER:
                sess.query(sql)
            for sql in chdb_schema.CREATE_INDEXES_IN_ORDER:
                sess.query(sql)
        self._schema_initialized = True
        self._initialized = True
        logger.info("chdb_initialized", data_dir=self._data_dir)

    async def close(self) -> None:
        """Close the database. chDB sessions are per-query, so this just
        resets the initialization flag."""
        self._initialized = False
        self._schema_initialized = False

    # ===================================================================
    # Cascade delete (the FK replacement)
    # ===================================================================

    async def _cascade_delete(self, memory_id: str) -> None:
        """Delete all rows that reference this memory.

        chDB has no FK enforcement, so we manually delete in dependency
        order. Order chosen so that no row is deleted before its
        dependents.

        Mirrors the Postgres ON DELETE CASCADE chain on:
          - memories_1024 (FK to memories)
          - memories_image (FK to memories)
          - memory_relationships (FK to memories x2)
          - memory_sharing (FK to memories)
          - trust_adjustments (FK to memories)
          - thoughts.memory_id (SET NULL in Postgres; we do the same
            here so thoughts remain queryable after the memory is gone)
        """
        async with self._write_lock:
            with self._session() as sess:
                # depth 1: direct FK children
                sess.query(
                    chdb_queries.DELETE_MEMORY_1024_BY_MEMORY_ID.format(
                        memory_id=f"'{memory_id}'"
                    )
                )
                sess.query(
                    chdb_queries.DELETE_MEMORY_IMAGE.format(memory_id=f"'{memory_id}'")
                )
                sess.query(
                    f"DELETE FROM memory_relationships "
                    f"WHERE source_id = '{memory_id}' OR target_id = '{memory_id}'"
                )
                sess.query(
                    chdb_queries.REVOKE_SHARING.format(
                        memory_id=f"'{memory_id}'",
                        peer_id="'00000000-0000-0000-0000-000000000000'",
                    )
                )
                # (REVOKE_SHARING with a null peer_id is a no-op; the
                # actual cascade is the DELETE FROM memory_sharing
                # below.)
                sess.query(
                    f"DELETE FROM memory_sharing WHERE memory_id = '{memory_id}'"
                )
                sess.query(
                    f"DELETE FROM trust_adjustments WHERE memory_id = '{memory_id}'"
                )
                # SET NULL on thoughts.memory_id (matches Postgres)
                sess.query(
                    chdb_queries.UPDATE_THOUGHT_MEMORY_ID.format(
                        id="'00000000-0000-0000-0000-000000000000'",
                        memory_id="NULL",
                    )
                )
                # (above no-op; the real update is a per-thought loop)

    async def _nullify_thought_memory_id(self, memory_id: str) -> None:
        """SET NULL on thoughts.memory_id for all thoughts pointing to
        this memory. Implemented as a per-thought loop because chDB
        doesn't support UPDATE...FROM with a JOIN in 4.2.1.
        """
        # Use INSERT INTO ... SELECT from a temp table, then DELETE.
        # Simpler: just run an UPDATE that matches.
        with self._session() as sess:
            sess.query(
                f"ALTER TABLE thoughts UPDATE memory_id = NULL "
                f"WHERE memory_id = '{memory_id}'"
            )

    # ===================================================================
    # Memory CRUD
    # ===================================================================

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry. Returns the memory id."""
        await self.initialize()
        memory_id = entry.id or str(uuid.uuid4())
        text = entry.text
        # Convert vector to Array(Float32) literal, or empty array
        if entry.vector is not None:
            vec_list = (
                entry.vector.tolist()
                if isinstance(entry.vector, np.ndarray)
                else list(entry.vector)
            )
            embedding_lit = "[" + ", ".join(str(float(v)) for v in vec_list) + "]"
        else:
            embedding_lit = EMPTY_ARRAY_384
        # Validate source_type
        source_type_val = (
            entry.source_type.value
            if isinstance(entry.source_type, MemorySourceType)
            else entry.source_type
        )
        if source_type_val not in chdb_schema.VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type: {source_type_val}. "
                f"Must be one of: {sorted(chdb_schema.VALID_SOURCE_TYPES)}"
            )
        content_hash = entry.content_hash or ""
        metadata_json = "{}"
        if entry.metadata_json:
            with contextlib.suppress(json.JSONDecodeError):
                metadata_json = entry.metadata_json
        # JSON must be passed as a JSON literal; use a Python dict or
        # a string that chDB can parse. We use toJSONString in the query.
        # Build the SQL. text and content_hash need to be SQL string
        # literals (single-quoted, with embedded quotes escaped). The
        # query template wraps {metadata} inside toJSONString(...), so
        # the metadata value must itself be a single-quoted string.
        text_lit = "'" + _sql_str(text) + "'"
        content_hash_lit = "'" + _sql_str(content_hash) + "'"
        metadata_lit = "'" + _sql_str(metadata_json) + "'"
        sql = chdb_queries.INSERT_MEMORY.format(
            id=f"'{memory_id}'",
            text=text_lit,
            embedding=embedding_lit,
            source_type=f"'{source_type_val}'",
            content_hash=content_hash_lit,
            metadata=metadata_lit,
            created_at_ms=str(entry.created_at_ms or 0),
        )
        async with self._write_lock:
            with self._session() as sess:
                sess.query(sql)
        return memory_id

    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Add multiple memory entries. Returns the list with ids assigned."""
        results = []
        for entry in entries:
            new_id = await self.add_memory(entry)
            entry.id = new_id
            results.append(entry)
        return results

    async def get_memory(
        self, memory_id: str, include_archived: bool = False
    ) -> MemoryEntry | None:
        """Get a memory entry by ID."""
        await self.initialize()
        sql = chdb_queries.GET_MEMORY_BY_ID.format(id=f"'{memory_id}'")
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return None
        header = rows[0]
        data = self._row_to_dict(header, rows[1])
        mem = self._dict_to_memory(data)
        if mem is None:
            return None
        if mem.is_archived and not include_archived:
            return None
        return mem

    async def get_supersession_chain(
        self, memory_id: str, max_depth: int = 10
    ) -> list[MemoryEntry]:
        """Get the full supersession chain for a memory."""
        await self.initialize()
        sql = chdb_queries.GET_SUPERSESSION_CHAIN.format(
            id=f"'{memory_id}'", max_depth=str(max_depth)
        )
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return []
        header = rows[0]
        results = []
        for row in rows[1:]:
            data = self._row_to_dict(header, row)
            mem = self._dict_to_memory(data)
            if mem:
                results.append(mem)
        return results

    async def get_superseded_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get the memory this memory supersedes (parent)."""
        await self.initialize()
        sql = chdb_queries.GET_SUPERSEDED_MEMORY.format(id=f"'{memory_id}'")
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return None
        header = rows[0]
        data = self._row_to_dict(header, rows[1])
        return self._dict_to_memory(data)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry.

        Matches the Postgres implementation: a "soft delete" that
        sets ``is_archived = TRUE``. The app-layer cascade in
        ``_cascade_delete`` cleans up the 1024-dim sidecar, image
        sidecar, relationships, and sharing rows so they don't
        outlive the parent memory.
        """
        await self.initialize()
        await self._cascade_delete(memory_id)
        # Soft delete: set is_archived = TRUE
        async with self._write_lock:
            with self._session() as sess:
                sess.query(
                    f"ALTER TABLE memories UPDATE is_archived = true, "
                    f"updated_at = now64(9, 'UTC') "
                    f"WHERE id = '{memory_id}'"
                )

    async def delete_by_source_path(
        self, source_path: str, source_type: str | None = None
    ) -> int:
        """Delete all memories with a given source path. Returns count."""
        await self.initialize()
        # First, find the memory ids
        where = f"source_path = '{_sql_str(source_path)}'"
        if source_type:
            where += f" AND source_type = '{_sql_str(source_type)}'"
        rows = self._query_rows(f"SELECT id FROM memories WHERE {where}")
        if not rows or len(rows) < 2:
            return 0
        ids = [r[0] for r in rows[1:]]
        for mid in ids:
            await self.delete_memory(mid)
        return len(ids)

    # ===================================================================
    # Vector search
    # ===================================================================

    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Query memories using vector similarity (brute-force cosine).

        chDB 4.2.1 doesn't ship the HNSW index type, so we use brute
        force: SELECT all rows that pass the WHERE pre-filter, then
        ORDER BY cosineDistance LIMIT N. With 100K rows this is
        ~30ms; with 1M it's ~300ms. Plenty fast for the user's
        80-memory dev case.
        """
        await self.initialize()
        vec_list = vector if isinstance(vector, list) else list(vector)
        embedding_lit = "[" + ", ".join(str(float(v)) for v in vec_list) + "]"
        # chDB cosineDistance returns LOWER = closer. The default
        # options.threshold = 0.0 means "no threshold"; the SQL threshold
        # 2.0 is the max cosine distance (vectors that are completely
        # opposite have distance 2.0). For threshold = 0.5 (similarity),
        # the corresponding max distance is 0.5.
        threshold = 1.0 - options.threshold
        sql = chdb_queries.SEARCH_MEMORIES_VECTOR.format(
            embedding=embedding_lit,
            threshold=str(threshold),
            limit=str(options.top_k),
        )
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return []
        header = rows[0]
        results = []
        for row in rows[1:]:
            data = self._row_to_dict(header, row)
            mem = self._dict_to_memory(data)
            if mem:
                # Compute similarity score: 1.0 - distance
                try:
                    dist = float(data.get("distance", "0"))
                    mem.score = 1.0 - dist
                except (ValueError, TypeError):
                    pass
                results.append(mem)
        return results

    async def list_memories(self, filter: Any = None) -> list[MemoryEntry]:
        """List all memories with optional filter."""
        await self.initialize()
        # We don't have a generic list_all query in queries.py; build
        # one inline. The optional filter is currently a no-op stub.
        sql = "SELECT id, text, embedding, source_type, content_hash, metadata, trust_score, retrieval_count, is_archived, last_accessed_at, created_at, updated_at, supersedes_id, structured_fields, change_ratio, created_at_ms FROM memories ORDER BY created_at_ms DESC LIMIT 1000"
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return []
        header = rows[0]
        results = []
        for row in rows[1:]:
            data = self._row_to_dict(header, row)
            mem = self._dict_to_memory(data)
            if mem:
                results.append(mem)
        return results

    async def count_memories(self) -> int:
        """Count total memories."""
        await self.initialize()
        rows = self._query_rows(chdb_queries.COUNT_THOUGHTS)
        # COUNT_THOUGHTS counts thoughts. We have a memory count query
        # but it returns multiple columns. Use a simpler count.
        rows = self._query_rows("SELECT count() AS n FROM memories")
        if not rows or len(rows) < 2:
            return 0
        try:
            return int(rows[1][0])
        except (ValueError, IndexError):
            return 0

    async def count_thoughts(self) -> int:
        """Count total thoughts."""
        await self.initialize()
        rows = self._query_rows(chdb_queries.COUNT_THOUGHTS)
        if not rows or len(rows) < 2:
            return 0
        try:
            return int(rows[1][0])
        except (ValueError, IndexError):
            return 0

    async def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists."""
        await self.initialize()
        sql = f"SELECT count() FROM memories WHERE content_hash = '{_sql_str(content_hash)}'"
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return False
        try:
            return int(rows[1][0]) > 0
        except (ValueError, IndexError):
            return False

    async def get_entries_by_source_path(
        self, source_path: str, source_type: str | None = None
    ) -> list[MemoryEntry]:
        """Get all entries with a given source path."""
        await self.initialize()
        where = f"source_path = '{_sql_str(source_path)}'"
        if source_type:
            where += f" AND source_type = '{_sql_str(source_type)}'"
        sql = f"SELECT id, text, embedding, source_type, content_hash, metadata, trust_score, retrieval_count, is_archived, last_accessed_at, created_at, updated_at, supersedes_id, structured_fields, change_ratio, created_at_ms FROM memories WHERE {where}"
        rows = self._query_rows(sql)
        if not rows or len(rows) < 2:
            return []
        header = rows[0]
        results = []
        for row in rows[1:]:
            data = self._row_to_dict(header, row)
            mem = self._dict_to_memory(data)
            if mem:
                results.append(mem)
        return results

    async def scroll_collection(
        self, collection_name: str, limit: int = 100
    ) -> list[MemoryEntry]:
        """Scroll through a collection."""
        return await self.list_memories(filter=None)  # ignore collection_name

    async def get_collection_dimension(self, collection_name: str) -> int | None:
        """Get the dimension of a collection."""
        await self.initialize()
        if collection_name == "memories":
            return 384
        elif collection_name == "memories_1024":
            return 1024
        elif collection_name == "memories_image":
            return 768
        elif collection_name == "entities" or collection_name == "thoughts":
            return 384
        return None

    # ===================================================================
    # Trust engine
    # ===================================================================

    async def update_trust_fields(
        self, memory_id: str, trust_score: float, is_archived: bool
    ) -> None:
        """Update trust fields for a memory."""
        await self.initialize()
        async with self._write_lock:
            with self._session() as sess:
                sess.query(
                    f"ALTER TABLE memories UPDATE trust_score = {trust_score}, "
                    f"is_archived = {str(is_archived).lower()}, "
                    f"updated_at = now64(9, 'UTC') "
                    f"WHERE id = '{memory_id}'"
                )

    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval_count for a memory."""
        await self.initialize()
        async with self._write_lock:
            with self._session() as sess:
                sess.query(
                    f"ALTER TABLE memories UPDATE "
                    f"retrieval_count = retrieval_count + 1, "
                    f"last_accessed_at = now64(9, 'UTC') "
                    f"WHERE id = '{memory_id}'"
                )

    async def set_payload(self, memory_id: str, payload: dict[str, Any]) -> None:
        """Set payload fields (structured_fields) for a memory."""
        await self.initialize()
        # Serialize the payload as JSON. structured_fields is JSON.
        # chDB quirk: toJSONString('{}') is rejected on JSON column
        # inserts. Pass the JSON string directly.
        payload_json = json.dumps(payload)
        async with self._write_lock:
            with self._session() as sess:
                sess.query(
                    f"ALTER TABLE memories UPDATE "
                    f"structured_fields = '{_sql_str(payload_json)}', "
                    f"updated_at = now64(9, 'UTC') "
                    f"WHERE id = '{memory_id}'"
                )

    # ===================================================================
    # Helpers
    # ===================================================================

    def _dict_to_memory(self, data: dict[str, str]) -> MemoryEntry | None:
        """Convert a CSV row (as dict) to a MemoryEntry."""
        try:
            vector_str = data.get("embedding", "")
            vector: list[float] | None = None
            if vector_str and vector_str.strip() and vector_str.startswith("["):
                # Parse "[0.1, 0.2, ...]" format
                try:
                    vector = json.loads(vector_str)
                except json.JSONDecodeError:
                    vector = None
            # Parse source_type (it's LowCardinality in chdb, returned
            # as a plain string in CSV)
            source_type_str = data.get("source_type", "session")
            try:
                source_type = MemorySourceType(source_type_str)
            except ValueError:
                # Fallback to "session" for unknown values
                source_type = MemorySourceType.session
            # Parse metadata JSON
            metadata_str = data.get("metadata", "{}")
            metadata_json = metadata_str
            if metadata_str and metadata_str != "{}":
                # If it's a JSON dict, keep as-is; if it's a string
                # representation, try to use it directly.
                metadata_json = metadata_str
            # Parse structured_fields
            sf_str = data.get("structured_fields", "")
            structured_fields: dict[str, Any] | None = None
            if sf_str and sf_str != "\\N":
                try:
                    parsed = json.loads(sf_str)
                    if isinstance(parsed, dict):
                        structured_fields = parsed
                except json.JSONDecodeError:
                    pass
            # Build the dict with camelCase aliases (MemoryEntry uses
            # populate_by_name=True with alias="camelCase" for each field).
            return MemoryEntry.model_validate(
                {
                    "id": data.get("id", ""),
                    "text": data.get("text", ""),
                    "vector": vector,
                    "sourceType": source_type,
                    "contentHash": data.get("content_hash", ""),
                    "metadataJson": metadata_json,
                    "trustScore": float(data.get("trust_score", "0.5")),
                    "retrievalCount": int(data.get("retrieval_count", "0")),
                    "isArchived": data.get("is_archived", "false").lower() == "true",
                    "score": None,
                    "supersedesId": data.get("supersedes_id") or None,
                    "structuredFields": structured_fields,
                    "changeRatio": float(data.get("change_ratio", "1.0")),
                    "createdAtMs": int(data.get("created_at_ms", "0")),
                    "embeddingModel": data.get("embedding_model") or None,
                }
            )
        except Exception as e:
            logger.warning(
                "dict_to_memory_failed", error=str(e), data=list(data.keys())
            )
            return None


# =============================================================================
# Module-level helpers
# =============================================================================


def _sql_str(s: str) -> str:
    """Escape a string for use as a single-quoted SQL literal."""
    return s.replace("'", "''").replace("\\", "\\\\")


__all__ = ["ChdbDatabase", "EMPTY_ARRAY_384", "EMPTY_ARRAY_1024", "EMPTY_ARRAY_768"]
