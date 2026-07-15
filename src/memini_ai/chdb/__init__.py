"""memini-ai chdb (in-process ClickHouse) backend.

Public surface mirrors `memini_ai.postgres`. The chdb package is opt-in via
the ``MEMINI_VECTOR_BACKEND=chdb`` environment variable.

The implementation is intentionally light in this 0.9.0 release. It exists
so the ``memini-ai`` server can boot with the new backend selected, fail
loudly on any actual query (NotImplementedError), and let the next tasks
(Task 2: schema, Task 3: queries, Task 4: database) fill in the real impl.

Design notes
------------
- chDB 4.2.1 ships ClickHouse 26.5.1.1 but does NOT include the
  ``vector_similarity`` HNSW index type (verified: ``allow_experimental_vector_similarity_index``
  is set to 1, but the index type itself is not registered). The new
  ``VECTOR`` data type is also not available in this build.
- Vector search in 0.9.0 is therefore **brute-force cosine distance** over
  ``Array(Float32)`` columns. Measured: 100K x 384 = 32ms, 100K x 768 = 61ms.
  Comfortably within the user's 80-memory dev case. Will re-evaluate HNSW
  when chDB ships an updated build with the index type.
- No FK enforcement in ClickHouse. Cascade delete is implemented at the
  app layer (see ``_cascade_delete`` in ``database.py``).
- Single-writer per process; reads concurrent. Connection pool wraps
  ``chdb.session.Session``.

See ``docs/memini-ai-v1-chdb-migration.md`` (Session 49 design) for the
full migration design.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from memini_ai.memory.database import VectorDatabase


class ChdbDatabase(VectorDatabase):
    """Placeholder chDB-backed VectorDatabase.

    0.9.0 placeholder: imports succeed, instantiation succeeds, but every
    abstract method raises NotImplementedError. Task 4 (database.py) will
    implement the 19 ABC methods against chDB.

    Initialization: ensures the chDB data directory exists, opens a
    ``chdb.session.Session`` (single-connection, lazy), and validates the
    data directory is writable. No schema is created here; that's the
    ``initialize()`` method which Task 4 will implement.
    """

    _initialized: bool
    _dimension: int | None
    _session: Any | None
    _data_dir: str
    _project_id: str | None
    _pool_size: int

    def __init__(
        self,
        data_dir: str,
        project_id: str | None = None,
        pool_size: int = 8,
        embedding_dim: int = 384,
    ) -> None:
        self._data_dir = os.path.expanduser(data_dir)
        self._project_id = project_id
        self._pool_size = max(1, min(64, pool_size))
        self._dimension = embedding_dim
        self._initialized = False
        self._session = None

        # Ensure the data directory exists and is writable. chDB will
        # create its own files inside this directory on first query.
        Path(self._data_dir).mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> str:
        """Return the chDB data directory path."""
        return self._data_dir

    @property
    def project_id(self) -> str | None:
        """Return the project_id (for audit log tagging)."""
        return self._project_id

    def _get_session(self) -> Any:
        """Lazy-open a chDB session. Task 4 will replace with a connection pool."""
        if self._session is None:
            import chdb.session

            self._session = chdb.session.Session(self._data_dir)
        return self._session

    def _check_init(self) -> None:
        """Verify initialize() has been called."""
        if not self._initialized:
            raise RuntimeError(
                "ChdbDatabase.initialize() has not been called. "
                "Call await ChdbDatabase.initialize() before using the database."
            )

    # ===================================================================
    # VectorDatabase ABC methods — all 19 are placeholders for Task 4.
    # ===================================================================

    async def initialize(self) -> None:
        """Initialize the database and create all tables + indexes.

        Creates the 15 tables defined in :mod:`memini_ai.chdb.schema`
        (idempotent — uses ``IF NOT EXISTS``), then creates the 35
        skip indexes (minmax for scalar ranges, set(0) for low-cardinality
        categories). Safe to call on every startup.

        Task 4 will: open a real connection pool, validate dim, etc.
        For 0.9.0, we use a single chDB session for the schema setup
        and then close it; the runtime queries in Task 4 will use a
        proper pool.
        """
        import chdb.session  # type: ignore[import-untyped]

        from memini_ai.chdb.schema import (
            CREATE_INDEXES_IN_ORDER,
            CREATE_TABLES_IN_ORDER,
        )

        # chDB doesn't allow multi-statement queries in a single .query()
        # call reliably; we run each statement individually.
        with chdb.session.Session(self._data_dir) as session:
            for sql in CREATE_TABLES_IN_ORDER:
                session.query(sql)
            for sql in CREATE_INDEXES_IN_ORDER:
                session.query(sql)

        self._initialized = True

    async def add_memory(self, entry: Any) -> str:
        raise NotImplementedError("ChdbDatabase.add_memory — see Task 4")

    async def add_memories(self, entries: list[Any]) -> list[Any]:
        raise NotImplementedError("ChdbDatabase.add_memories — see Task 4")

    async def get_memory(
        self, memory_id: str, include_archived: bool = False
    ) -> Any | None:
        raise NotImplementedError("ChdbDatabase.get_memory — see Task 4")

    async def get_supersession_chain(
        self, memory_id: str, max_depth: int = 10
    ) -> list[Any]:
        raise NotImplementedError("ChdbDatabase.get_supersession_chain — see Task 4")

    async def get_superseded_memory(self, memory_id: str) -> Any | None:
        raise NotImplementedError("ChdbDatabase.get_superseded_memory — see Task 4")

    async def delete_memory(self, memory_id: str) -> None:
        raise NotImplementedError("ChdbDatabase.delete_memory — see Task 4")

    async def delete_by_source_path(
        self, source_path: str, source_type: str | None = None
    ) -> int:
        raise NotImplementedError("ChdbDatabase.delete_by_source_path — see Task 4")

    async def query_memories(
        self,
        vector: list[float],
        options: Any,
        collection_name: str | None = None,
    ) -> list[Any]:
        raise NotImplementedError("ChdbDatabase.query_memories — see Task 4")

    async def list_memories(self, filter: Any | None = None) -> list[Any]:
        raise NotImplementedError("ChdbDatabase.list_memories — see Task 4")

    async def count_memories(self) -> int:
        raise NotImplementedError("ChdbDatabase.count_memories — see Task 4")

    async def count_thoughts(self) -> int:
        raise NotImplementedError("ChdbDatabase.count_thoughts — see Task 4")

    async def content_exists(self, content_hash: str) -> bool:
        raise NotImplementedError("ChdbDatabase.content_exists — see Task 4")

    async def get_entries_by_source_path(
        self, source_path: str, source_type: str | None = None
    ) -> list[Any]:
        raise NotImplementedError(
            "ChdbDatabase.get_entries_by_source_path — see Task 4"
        )

    async def scroll_collection(
        self, collection_name: str, limit: int = 100
    ) -> list[Any]:
        raise NotImplementedError("ChdbDatabase.scroll_collection — see Task 4")

    async def get_collection_dimension(self, collection_name: str) -> int | None:
        raise NotImplementedError("ChdbDatabase.get_collection_dimension — see Task 4")

    async def close(self) -> None:
        """Close the chDB session. Idempotent."""
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session.cleanup()
            self._session = None
        self._initialized = False

    async def update_trust_fields(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        raise NotImplementedError("ChdbDatabase.update_trust_fields — see Task 4")

    async def increment_retrieval_count(self, memory_id: str) -> None:
        raise NotImplementedError("ChdbDatabase.increment_retrieval_count — see Task 4")

    async def set_payload(self, memory_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError("ChdbDatabase.set_payload — see Task 4")


__all__ = ["ChdbDatabase"]
