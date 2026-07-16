"""Memory database layer - Vector database abstraction with PostgreSQL pgvector implementation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from memini_ai.config import MeminiConfig, get_config
from memini_ai.memory.schema import (
    MemoryEntry,
    SearchFilter,
    SearchOptions,
)


class VectorDatabase(ABC):
    """Abstract base class for vector database operations.

    Provides a common interface for different vector database backends
    (pgvector, etc.) to enable backend-agnostic memory operations.
    """

    _initialized: bool
    _dimension: int | None
    _pool: Any | None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database and create collections if needed."""
        ...

    @abstractmethod
    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry.

        Args:
            entry: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.
        """
        ...

    @abstractmethod
    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Add multiple memory entries.

        Args:
            entries: List of MemoryEntry objects to add.

        Returns:
            List of MemoryEntry objects with IDs assigned.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
            List of MemoryEntry objects in the supersession chain.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID.

        Args:
            memory_id: ID of the memory entry to delete.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
            collection_name: Optional collection override.

        Returns:
            List of matching MemoryEntry objects with scores.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def count_memories(self) -> int:
        """Count total memories.

        Returns:
            Number of memory entries.
        """
        ...

    @abstractmethod
    async def count_thoughts(self) -> int:
        """Count total thoughts in the thoughts table.

        Returns:
            Number of thought rows.
        """
        ...

    @abstractmethod
    async def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists.

        Args:
            content_hash: SHA-256 hash to check.

        Returns:
            True if exists, False otherwise.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Scroll through a collection.

        Args:
            collection_name: Collection to scroll.
            limit: Page size.

        Returns:
            List of MemoryEntry objects.
        """
        ...

    @abstractmethod
    async def get_collection_dimension(self, collection_name: str) -> int | None:
        """Get the dimension of a collection.

        Args:
            collection_name: Collection to check.

        Returns:
            Vector dimension or None if not found.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval count for a memory entry.

        Args:
            memory_id: ID of the memory entry.
        """
        ...

    @abstractmethod
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
        ...


def create_database(config: MeminiConfig | None = None) -> VectorDatabase:
    """Factory function to create a VectorDatabase instance.

    v1.0.0: Supports pgembed (embedded, default) and postgres-external
    (Docker/team server) backends.

    Q4 breaking change: vector_backend always wins. v0.8.2 users with
    MEMINI_DB_URL set must explicitly set MEMINI_VECTOR_BACKEND or run
    'memini-ai migrate'. See design section 5.4.
    """
    if config is None:
        config = get_config()

    # ── Q4: Refuse to start if v0.8.2 user with MEMINI_DB_URL set ──
    if (
        config.db_url or os.environ.get("MEMINI_DB_URL")
    ) and "MEMINI_VECTOR_BACKEND" not in os.environ:
        raise RuntimeError(
            "memini-ai v1.0.0: MEMINI_DB_URL is set but MEMINI_VECTOR_BACKEND is not.\n"
            "\n"
            "v1.0.0 defaults to 'pgembed' (embedded PostgreSQL) which IGNORES MEMINI_DB_URL.\n"
            "Your v0.8.2 setup connected to an external Postgres server via MEMINI_DB_URL.\n"
            "\n"
            "To keep using your external server:\n"
            "  export MEMINI_VECTOR_BACKEND=postgres-external\n"
            "\n"
            "To switch to embedded mode (and copy your data):\n"
            "  unset MEMINI_DB_URL\n"
            "  memini-ai migrate --from='<your old MEMINI_DB_URL>'\n"
            "\n"
            "Set MEMINI_VECTOR_BACKEND explicitly to suppress this error."
        )

    from memini_ai.postgres.driver import (
        DatabaseDriver,
        EmbeddedPGDriver,
        ExternalPGDriver,
    )

    if config.vector_backend == "pgembed":
        data_dir = Path(config.pgembed_data_dir).expanduser()
        driver: DatabaseDriver = EmbeddedPGDriver(data_dir)
    elif config.vector_backend == "postgres-external":
        db_url = config.db_url or os.environ.get("MEMINI_DB_URL", "")
        if not db_url:
            raise ValueError(
                "MEMINI_VECTOR_BACKEND=postgres-external but MEMINI_DB_URL is not set. "
                "Set MEMINI_DB_URL or switch to MEMINI_VECTOR_BACKEND=pgembed for embedded mode."
            )
        driver = ExternalPGDriver(db_url)
    else:
        raise ValueError(f"Unknown vector_backend: {config.vector_backend}")

    from memini_ai.postgres import PostgresDatabase

    db = PostgresDatabase(
        driver=driver,
        project_id=config.project_id,
        sslmode=config.db_sslmode,
        sslrootcert=config.db_sslrootcert,
    )

    # ── Optional: wrap in RRFDatabase for fusion mode ──
    if config.fusion_mode == "rrf" and config.team_db_url:
        from memini_ai.memory.rrf_database import RRFDatabase

        team_driver = ExternalPGDriver(config.team_db_url)
        team_db = PostgresDatabase(driver=team_driver, project_id=config.project_id)
        db = RRFDatabase(primary=db, secondary=team_db, k=config.rrf_k)

    return db
