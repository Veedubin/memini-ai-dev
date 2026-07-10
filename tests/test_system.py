"""Tests for MemorySystem - high-level coordinator integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.config import MeminiConfig
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchOptions,
    SearchStrategy,
)
from memini_ai.memory.system import MemorySystem, MemorySystemConfig


@pytest.fixture
def mock_db() -> MagicMock:
    """Create mock MemoryDatabase."""
    db = MagicMock()
    db.initialize = AsyncMock()
    db.add_memory = AsyncMock(return_value="new-memory-id")
    db.get_memory = AsyncMock(return_value=None)
    db.delete_memory = AsyncMock()
    db.query_memories = AsyncMock(return_value=[])
    db.list_memories = AsyncMock(return_value=[])
    db.count_memories = AsyncMock(return_value=0)
    db.content_exists = AsyncMock(return_value=False)
    db._initialized = True
    db._dimension = 1024
    db._project_id = None
    return db


@pytest.fixture
def mock_search() -> MagicMock:
    """Create mock MemorySearch."""
    search = MagicMock()
    search.query = AsyncMock(return_value=[])
    search.vector_only_search = AsyncMock(return_value=[])
    search.text_only_search = AsyncMock(return_value=[])
    search.parallel_search = AsyncMock(return_value=[])
    search.search_with_vector = AsyncMock(return_value=[])
    search.get_similar = AsyncMock(return_value=[])
    search.query_with_fallback_collection = AsyncMock(return_value=[])
    search._rrf_fusion = MagicMock(return_value=[])
    search.invalidate_bm25 = AsyncMock()
    return search


@pytest.fixture
def mock_embedding() -> MagicMock:
    """Create mock embedding result."""
    result = MagicMock()
    result.embedding = [0.1] * 1024
    result.model_id = "BAAI/bge-m3"
    result.device = "cpu"
    result.token_count = 50
    result.timestamp = 1234567890
    result.latency_ms = 10
    return result


@pytest.fixture
def mock_config() -> MeminiConfig:
    """Create mock MeminiConfig."""
    config = MagicMock()
    config.embedding_dim = 1024
    config.table_name = "memories"
    config.project_id = None
    config.query_collections = None
    return config


class TestMemorySystemConfig:
    """Tests for MemorySystemConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = MemorySystemConfig()
        assert config.project_id is None
        assert config.query_collections is None
        assert config.enable_cascade is True
        assert config.enable_deduplication is True

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = MemorySystemConfig(
            project_id="my-project",
            query_collections=["memories_1024", "memories_384"],
            enable_cascade=False,
            enable_deduplication=False,
        )
        assert config.project_id == "my-project"
        assert len(config.query_collections) == 2
        assert config.enable_cascade is False


class TestMemorySystemInit:
    """Tests for MemorySystem initialization."""

    def test_creates_db_and_search_if_none(self) -> None:
        """Should create database and search if not provided."""
        with patch("memini_ai.memory.system.create_database", return_value=MagicMock()):
            system = MemorySystem()
        assert system._db is not None
        assert system._search is not None

    def test_uses_provided_db_and_search(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should use provided database and search."""
        system = MemorySystem(db=mock_db, search=mock_search)
        assert system._db is mock_db
        assert system._search is mock_search

    def test_initialized_false_by_default(self) -> None:
        """Should not be initialized by default."""
        with patch("memini_ai.memory.system.create_database", return_value=MagicMock()):
            system = MemorySystem()
        assert system._initialized is False


class TestInitialize:
    """Tests for initialize method."""

    @pytest.mark.asyncio
    async def test_initializes_once(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should not reinitialize if already initialized."""
        system = MemorySystem(db=mock_db, search=mock_search)

        await system.initialize()
        assert system._initialized is True

        # Call again - should not error
        await system.initialize()
        assert system._initialized is True

    @pytest.mark.asyncio
    async def test_applies_project_id(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should store project ID in config."""
        config = MemorySystemConfig(project_id="my-project")
        system = MemorySystem(db=mock_db, search=mock_search, config=config)

        assert system._config.project_id == "my-project"


class TestIsInitialized:
    """Tests for is_initialized property."""

    def test_returns_false_when_not_initialized(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return False when not initialized."""
        system = MemorySystem(db=mock_db, search=mock_search)
        assert system.is_initialized is False

    @pytest.mark.asyncio
    async def test_returns_true_when_initialized(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return True when initialized."""
        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()
        assert system.is_initialized is True


class TestIsReady:
    """Tests for is_ready property."""

    def test_returns_false_when_not_initialized(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return False when not initialized."""
        system = MemorySystem(db=mock_db, search=mock_search)
        assert system.is_ready is False

    @pytest.mark.asyncio
    async def test_returns_true_when_db_ready(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return True when database is ready."""
        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()
        assert system.is_ready is True


class TestAddMemory:
    """Tests for add_memory method."""

    @pytest.mark.asyncio
    async def test_add_memory_generates_vector(
        self, mock_db: MagicMock, mock_search: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Should generate vector if not present."""
        system = MemorySystem(db=mock_db, search=mock_search)

        entry = MemoryEntry(
            text="Test memory",
            source_type=MemorySourceType.session,
            content_hash="abc123",
            # No vector
        )

        with (
            patch(
                "memini_ai.memory.system.generate_embedding",
                AsyncMock(return_value=mock_embedding),
            ),
            patch("memini_ai.memory.system.hash_content", return_value="abc123"),
        ):
            await system.add_memory(entry)

        assert entry.vector is not None

    @pytest.mark.asyncio
    async def test_add_memory_sets_hash(
        self, mock_db: MagicMock, mock_search: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Should set content hash if not present."""
        system = MemorySystem(db=mock_db, search=mock_search)

        entry = MemoryEntry(
            text="Test memory",
            source_type=MemorySourceType.session,
            # No content_hash - model validator auto-computes it
        )

        with patch(
            "memini_ai.memory.system.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            await system.add_memory(entry)

        # Hash is auto-computed by model validator
        assert entry.content_hash is not None
        assert len(entry.content_hash) > 0

    @pytest.mark.asyncio
    async def test_add_memory_checks_duplicate(
        self, mock_db: MagicMock, mock_search: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Should check for duplicate content."""
        mock_db.content_exists = AsyncMock(return_value=True)
        system = MemorySystem(db=mock_db, search=mock_search)

        entry = MemoryEntry(
            text="Duplicate content",
            source_type=MemorySourceType.session,
            content_hash="existing-hash",
        )

        with pytest.raises(ValueError, match="already exists"):
            await system.add_memory(entry)

    @pytest.mark.asyncio
    async def test_add_memory_calls_db(
        self, mock_db: MagicMock, mock_search: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Should call database add_memory."""
        system = MemorySystem(db=mock_db, search=mock_search)

        entry = MemoryEntry(
            text="Test memory",
            source_type=MemorySourceType.session,
            content_hash="abc123",
            vector=[0.1] * 1024,
        )

        with (
            patch(
                "memini_ai.memory.system.generate_embedding",
                AsyncMock(return_value=mock_embedding),
            ),
            patch("memini_ai.memory.system.hash_content", return_value="abc123"),
        ):
            await system.add_memory(entry)

        mock_db.add_memory.assert_called_once()


class TestGetMemory:
    """Tests for get_memory method."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_entry(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return memory from database."""
        expected = MemoryEntry(
            id="test-id",
            text="Test memory",
            source_type=MemorySourceType.session,
            content_hash="abc123",
        )
        mock_db.get_memory = AsyncMock(return_value=expected)

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        result = await system.get_memory("test-id")
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_memory_returns_none(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return None if not found."""
        mock_db.get_memory = AsyncMock(return_value=None)

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        result = await system.get_memory("nonexistent")
        assert result is None


class TestDeleteMemory:
    """Tests for delete_memory method."""

    @pytest.mark.asyncio
    async def test_delete_memory_calls_db(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should call database delete."""
        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.delete_memory("test-id")

        mock_db.delete_memory.assert_called_once_with("test-id")

    @pytest.mark.asyncio
    async def test_delete_memory_invalidates_bm25(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should invalidate BM25 cache after delete."""
        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.delete_memory("test-id")

        mock_search.invalidate_bm25.assert_called_once()


class TestQueryMemories:
    """Tests for query_memories method."""

    @pytest.mark.asyncio
    async def test_query_memories_calls_search(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should delegate to search."""
        mock_search.query = AsyncMock(return_value=[])

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.query_memories("test query")

        mock_search.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_memories_with_options(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should pass options to search."""
        mock_search.query = AsyncMock(return_value=[])

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        options = SearchOptions(top_k=10, strategy=SearchStrategy.PARALLEL)
        await system.query_memories("test query", options)

        mock_search.query.assert_called_once()


class TestSearchWithVector:
    """Tests for search_with_vector method."""

    @pytest.mark.asyncio
    async def test_search_with_vector_calls_search(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should delegate to search."""
        mock_search.search_with_vector = AsyncMock(return_value=[])

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.search_with_vector([0.1] * 1024)

        mock_search.search_with_vector.assert_called_once()


class TestGetSimilar:
    """Tests for get_similar method."""

    @pytest.mark.asyncio
    async def test_get_similar_calls_search(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should delegate to search."""
        mock_search.get_similar = AsyncMock(return_value=[])

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.get_similar("reference-id")

        mock_search.get_similar.assert_called_once()


class TestListMemories:
    """Tests for list_memories method."""

    @pytest.mark.asyncio
    async def test_list_memories_calls_db(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should delegate to database."""
        mock_db.list_memories = AsyncMock(return_value=[])

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        await system.list_memories()

        mock_db.list_memories.assert_called_once()


class TestGetStats:
    """Tests for get_stats method."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return statistics dictionary."""
        mock_db.count_memories = AsyncMock(return_value=42)

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        stats = await system.get_stats()

        assert "total_memories" in stats
        assert "dimension" in stats
        assert "collections" in stats
        assert "initialized" in stats
        assert "ready" in stats
        assert stats["total_memories"] == 42


class TestContentExists:
    """Tests for content_exists method."""

    @pytest.mark.asyncio
    async def test_content_exists_checks_hash(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should check content by hash."""
        mock_db.content_exists = AsyncMock(return_value=True)

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        with patch("memini_ai.memory.system.hash_content", return_value="test-hash"):
            result = await system.content_exists("test text")

        assert result is True
        mock_db.content_exists.assert_called_once_with("test-hash")

    @pytest.mark.asyncio
    async def test_content_exists_returns_false(
        self, mock_db: MagicMock, mock_search: MagicMock
    ) -> None:
        """Should return False when not found."""
        mock_db.content_exists = AsyncMock(return_value=False)

        system = MemorySystem(db=mock_db, search=mock_search)
        await system.initialize()

        with patch("memini_ai.memory.system.hash_content", return_value="test-hash"):
            result = await system.content_exists("nonexistent text")

        assert result is False
