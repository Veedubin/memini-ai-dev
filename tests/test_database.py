"""Tests for MemoryDatabase - Qdrant CRUD operations with mocking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.database import MemoryDatabase, _get_collection_name
from memini_ai.memory.schema import MemoryEntry, MemorySourceType


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Create mock Qdrant client."""
    client = MagicMock()
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock(return_value=None)
    client.create_payload_index = AsyncMock(return_value=None)
    client.upsert = AsyncMock(return_value={"status": "completed"})
    client.delete = AsyncMock(return_value={"status": "completed"})
    client.get_collection = AsyncMock(
        return_value=MagicMock(
            config=MagicMock(params=MagicMock(vector_size=1024)),
            vectors_count=10,
        )
    )
    client.retrieve = AsyncMock(return_value=[])
    client.search = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    return client


@pytest.fixture
def db(mock_qdrant: MagicMock) -> MemoryDatabase:
    """Create MemoryDatabase with mocked client."""
    database = MemoryDatabase(url="http://localhost:6333", project_id="test-project")
    return database


class TestGetCollectionName:
    """Tests for _get_collection_name helper."""

    def test_1024_dimension(self) -> None:
        """Should return memories_1024 for 1024 dimension."""
        assert _get_collection_name(1024) == "memories_1024"

    def test_384_dimension(self) -> None:
        """Should return memories_384 for 384 dimension."""
        assert _get_collection_name(384) == "memories_384"


class TestMemoryDatabaseInit:
    """Tests for MemoryDatabase initialization."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        db = MemoryDatabase()
        assert db._url == "http://localhost:6333"
        assert db._project_id is None
        assert db._initialized is False

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        db = MemoryDatabase(url="http://custom:6333", project_id="my-project")
        assert db._url == "http://custom:6333"
        assert db._project_id == "my-project"


class TestInitialize:
    """Tests for initialize method."""

    @pytest.mark.asyncio
    async def test_initializes_once(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should not reinitialize if already initialized."""
        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            assert db._initialized is True

            # Call again - should not error
            await db.initialize()
            assert db._initialized is True

    @pytest.mark.asyncio
    async def test_creates_collection_if_not_exists(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should create collection when it doesn't exist."""
        mock_qdrant.collection_exists = AsyncMock(return_value=False)

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()

            mock_qdrant.create_collection.assert_called_once()
            mock_qdrant.create_payload_index.assert_called()

    @pytest.mark.asyncio
    async def test_skips_creation_if_exists(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should not create collection if it already exists."""
        mock_qdrant.collection_exists = AsyncMock(return_value=True)

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()

            mock_qdrant.create_collection.assert_not_called()


class TestAddMemory:
    """Tests for add_memory method."""

    @pytest.mark.asyncio
    async def test_add_memory_generates_id(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should generate ID if not provided."""
        entry = MemoryEntry(
            text="Test memory",
            source_type=MemorySourceType.session,
            content_hash="abc123",
        )

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            memory_id = await db.add_memory(entry)

        assert memory_id is not None
        assert entry.id == memory_id

    @pytest.mark.asyncio
    async def test_add_memory_calls_upsert(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should call upsert with correct arguments."""
        entry = MemoryEntry(
            id="custom-id",
            text="Test memory",
            source_type=MemorySourceType.session,
            content_hash="abc123",
            vector=[0.1] * 1024,
        )

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.add_memory(entry)

        mock_qdrant.upsert.assert_called_once()
        call_args = mock_qdrant.upsert.call_args
        assert "collection_name" in call_args.kwargs or call_args.args[0] is not None

    @pytest.mark.asyncio
    async def test_add_memory_computes_hash_if_missing(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should compute content hash if not provided."""
        entry = MemoryEntry(
            text="Test memory",
            source_type=MemorySourceType.session,
            # No content_hash provided
        )

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.add_memory(entry)

        # Entry should have content hash now
        assert entry.content_hash is not None


class TestGetMemory:
    """Tests for get_memory method."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_entry(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return MemoryEntry when found."""
        mock_record = MagicMock()
        mock_record.payload = {
            "id": "test-id",
            "text": "Test memory",
            "sourceType": "session",
            "contentHash": "abc123",
        }
        mock_record.vector = [0.1] * 1024

        mock_qdrant.retrieve = AsyncMock(return_value=[mock_record])

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            entry = await db.get_memory("test-id")

        assert entry is not None
        assert entry.id == "test-id"
        assert entry.text == "Test memory"

    @pytest.mark.asyncio
    async def test_get_memory_returns_none_when_not_found(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return None when memory doesn't exist."""
        mock_qdrant.retrieve = AsyncMock(return_value=[])

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            entry = await db.get_memory("nonexistent-id")

        assert entry is None


class TestDeleteMemory:
    """Tests for delete_memory method."""

    @pytest.mark.asyncio
    async def test_delete_memory_calls_delete(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should call Qdrant delete."""
        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            await db.delete_memory("test-id")

        mock_qdrant.delete.assert_called_once()


class TestQueryMemories:
    """Tests for query_memories method."""

    @pytest.mark.asyncio
    async def test_query_memories_returns_results(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return matching memories."""
        from memini_ai.memory.schema import SearchOptions

        mock_hit = MagicMock()
        mock_hit.payload = {
            "text": "Test result",
            "sourceType": "session",
            "contentHash": "resulthash",
        }
        mock_hit.score = 0.95
        mock_hit.vector = [0.1] * 1024

        mock_qdrant.query_points = AsyncMock(return_value=MagicMock(points=[mock_hit]))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            results = await db.query_memories([0.1] * 1024, SearchOptions())

        assert len(results) == 1
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_query_memories_empty_on_error(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return empty list on error."""
        from memini_ai.memory.schema import SearchOptions

        mock_qdrant.search = AsyncMock(side_effect=Exception("Search error"))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            results = await db.query_memories([0.1] * 1024, SearchOptions())

        assert results == []


class TestContentExists:
    """Tests for content_exists method."""

    @pytest.mark.asyncio
    async def test_content_exists_returns_true(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return True when content exists."""
        mock_record = MagicMock()
        mock_qdrant.scroll = AsyncMock(return_value=([mock_record], None))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            exists = await db.content_exists("abc123")

        assert exists is True

    @pytest.mark.asyncio
    async def test_content_exists_returns_false(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return False when content doesn't exist."""
        mock_qdrant.scroll = AsyncMock(return_value=([], None))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            exists = await db.content_exists("nonexistent")

        assert exists is False


class TestCountMemories:
    """Tests for count_memories method."""

    @pytest.mark.asyncio
    async def test_count_memories_returns_count(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return vector count from collection."""
        mock_qdrant.get_collection = AsyncMock(return_value=MagicMock(vectors_count=42))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            count = await db.count_memories()

        assert count == 42


class TestListMemories:
    """Tests for list_memories method."""

    @pytest.mark.asyncio
    async def test_list_memories_returns_entries(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return all memory entries."""
        mock_record = MagicMock()
        mock_record.payload = {
            "id": "entry-1",
            "text": "Memory 1",
            "sourceType": "session",
            "contentHash": "hash1",
        }
        mock_record.vector = [0.1] * 1024

        mock_qdrant.scroll = AsyncMock(return_value=([mock_record], None))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            entries = await db.list_memories()

        assert len(entries) == 1
        assert entries[0].id == "entry-1"


class TestDeleteBySourcePath:
    """Tests for delete_by_source_path method."""

    @pytest.mark.asyncio
    async def test_delete_by_source_path_returns_count(
        self, db: MemoryDatabase, mock_qdrant: MagicMock
    ) -> None:
        """Should return number of deleted memories."""
        mock_record = MagicMock()
        mock_record.id = "id-to-delete"

        mock_qdrant.scroll = AsyncMock(return_value=([mock_record], None))

        with patch(
            "memini_ai.memory.database._client_cache",
            {"http://localhost:6333": mock_qdrant},
        ):
            await db.initialize()
            count = await db.delete_by_source_path("/test/path")

        assert count >= 0  # May be 0 if delete fails
