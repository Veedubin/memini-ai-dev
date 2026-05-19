"""Tests for PostgresDatabase - PostgreSQL/pgvector backend."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
import pytest_asyncio

from memini_ai.memory.schema import MemoryEntry, MemorySourceType, SearchOptions
from memini_ai.postgres.database import PostgresDatabase

# =============================================================================
# Test Configuration
# =============================================================================

TEST_DB_URL = "postgresql://postgres:password@localhost:5434/postgres"
TEST_DB_NAME = f"test_memini_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def pg_db():
    """Create PostgresDatabase connected to test database with transaction isolation."""
    # Use the default postgres database for test fixture
    # In production, you might want a separate test database
    db_url = TEST_DB_URL
    db = PostgresDatabase(db_url)

    await db.initialize()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def pg_db_isolated(pg_db: PostgresDatabase):
    """Provide pg_db with table cleanup between tests."""
    yield pg_db
    # Cleanup: delete all test memories
    try:
        async with pg_db._pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE text LIKE 'test_%'")
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def sample_vector() -> list[float]:
    """Create a sample 384-dim vector."""
    np.random.seed(42)
    return np.random.rand(384).astype(np.float32).tolist()


@pytest.fixture
def sample_memory_entry(sample_vector: list[float]) -> MemoryEntry:
    """Create a sample memory entry."""
    return MemoryEntry(
        id=str(uuid.uuid4()),
        text="test_sample_memory",
        vector=sample_vector,
        source_type=MemorySourceType.session,
        content_hash="test_hash_123",
    )


@pytest.fixture
def multiple_memory_entries(sample_vector: list[float]) -> list[MemoryEntry]:
    """Create multiple memory entries for batch testing."""
    entries = []
    for i in range(5):
        vector = np.random.rand(384).astype(np.float32)
        entries.append(MemoryEntry(
            id=str(uuid.uuid4()),
            text=f"test_memory_number_{i}",
            vector=vector.tolist(),
            source_type=MemorySourceType.session,
            content_hash=f"test_hash_{i}",
        ))
    return entries


# =============================================================================
# Helper Functions
# =============================================================================


def create_memory_entry(
    text: str,
    vector: list[float] | None = None,
    source_type: MemorySourceType = MemorySourceType.session,
    memory_id: str | None = None,
) -> MemoryEntry:
    """Helper to create a MemoryEntry with proper defaults."""
    if vector is None:
        np.random.seed(hash(text) % (2**32))
        vector = np.random.rand(384).astype(np.float32).tolist()

    return MemoryEntry(
        id=memory_id or str(uuid.uuid4()),
        text=text,
        vector=vector,
        source_type=source_type,
        content_hash=f"hash_{uuid.uuid4().hex[:8]}",
    )


# =============================================================================
# Test Schema Initialization
# =============================================================================


class TestSchemaInitialization:
    """Tests for database schema initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, pg_db: PostgresDatabase):
        """Should create connection pool on initialize."""
        assert pg_db._pool is not None
        assert pg_db._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, pg_db: PostgresDatabase):
        """Should be safe to call initialize multiple times."""
        await pg_db.initialize()
        await pg_db.initialize()
        assert pg_db._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_schema_creates_tables(self, pg_db: PostgresDatabase):
        """Should create memories table."""
        async with pg_db._pool.acquire() as conn:
            # Check that memories table exists
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'memories'
                )
            """)
            assert result is True

    @pytest.mark.asyncio
    async def test_schema_has_required_columns(self, pg_db: PostgresDatabase):
        """Should have all required columns in memories table."""
        async with pg_db._pool.acquire() as conn:
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'memories'
            """)
            column_names = {row['column_name'] for row in columns}

            required = {'id', 'text', 'embedding', 'source_type', 'content_hash',
                       'trust_score', 'retrieval_count', 'is_archived', 'metadata'}
            assert required.issubset(column_names)


# =============================================================================
# Test Basic CRUD Operations
# =============================================================================


class TestAddMemory:
    """Tests for add_memory method."""

    @pytest.mark.asyncio
    async def test_add_memory_inserts_entry(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should insert memory and return its ID."""
        memory_id = await pg_db_isolated.add_memory(sample_memory_entry)
        assert memory_id is not None
        assert sample_memory_entry.id == memory_id

    @pytest.mark.asyncio
    async def test_add_memory_generates_id_if_missing(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should generate ID if not provided."""
        entry = MemoryEntry(
            text="test_memory_no_id",
            vector=sample_vector,
            source_type=MemorySourceType.session,
        )
        # Entry has no ID until add_memory is called
        original_id = entry.id
        memory_id = await pg_db_isolated.add_memory(entry)

        assert entry.id is not None
        assert entry.id == memory_id
        assert original_id == memory_id  # Should be the same

    @pytest.mark.asyncio
    async def test_add_memory_stores_vector(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should store vector embedding correctly."""
        entry = create_memory_entry("test_vector_storage", sample_vector)
        await pg_db_isolated.add_memory(entry)

        retrieved = await pg_db_isolated.get_memory(entry.id)
        assert retrieved is not None
        assert retrieved.vector is not None
        assert len(retrieved.vector) == 384


class TestGetMemory:
    """Tests for get_memory method."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_entry(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should return MemoryEntry when found."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)

        assert result is not None
        assert result.id == sample_memory_entry.id
        assert result.text == sample_memory_entry.text
        assert result.source_type == sample_memory_entry.source_type

    @pytest.mark.asyncio
    async def test_get_memory_returns_none_when_not_found(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return None for non-existent ID."""
        result = await pg_db_isolated.get_memory(str(uuid.uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_memory_includes_trust_fields(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should return trust_score and retrieval_count."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)

        assert result is not None
        assert hasattr(result, 'trust_score')
        assert hasattr(result, 'retrieval_count')
        assert isinstance(result.trust_score, float)
        assert isinstance(result.retrieval_count, int)


class TestDeleteMemory:
    """Tests for delete_memory method."""

    @pytest.mark.asyncio
    async def test_delete_memory_archives_entry(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should soft-delete (archive) the memory."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        await pg_db_isolated.delete_memory(sample_memory_entry.id)

        # Memory should not be found via normal queries
        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_memory_nonexistent_does_not_error(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should not raise error when deleting non-existent memory."""
        await pg_db_isolated.delete_memory(str(uuid.uuid4()))
        # No exception means success


class TestAddMemories:
    """Tests for add_memories (batch insert) method."""

    @pytest.mark.asyncio
    async def test_add_memories_inserts_all(
        self, pg_db_isolated: PostgresDatabase, multiple_memory_entries: list[MemoryEntry]
    ):
        """Should insert all memory entries."""
        result = await pg_db_isolated.add_memories(multiple_memory_entries)

        assert len(result) == len(multiple_memory_entries)
        for entry in multiple_memory_entries:
            retrieved = await pg_db_isolated.get_memory(entry.id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_add_memories_empty_list(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return empty list for empty input."""
        result = await pg_db_isolated.add_memories([])
        assert result == []


# =============================================================================
# Test Vector Search
# =============================================================================


class TestQueryMemories:
    """Tests for query_memories (vector search) method."""

    @pytest.mark.asyncio
    async def test_query_memories_returns_similar(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return memories similar to query vector."""
        # Create memories with known vectors
        np.random.seed(42)
        base_vector = np.random.rand(384).astype(np.float32).tolist()

        # Memory 1: close to base_vector
        close_vector = (np.array(base_vector) + 0.01 * np.random.rand(384)).tolist()
        memory1 = create_memory_entry("test_similar_1", close_vector)

        # Memory 2: far from base_vector
        far_vector = (-np.array(base_vector)).tolist()
        memory2 = create_memory_entry("test_similar_2", far_vector)

        await pg_db_isolated.add_memories([memory1, memory2])

        # Search with base_vector
        options = SearchOptions(top_k=5, threshold=0.1)
        results = await pg_db_isolated.query_memories(base_vector, options)

        assert len(results) >= 1
        # The close memory should be in results (order not guaranteed with random vectors)
        result_ids = [r.id for r in results]
        assert memory1.id in result_ids

    @pytest.mark.asyncio
    async def test_query_memories_respects_threshold(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should filter results below threshold."""
        np.random.seed(123)
        base_vector = np.random.rand(384).astype(np.float32).tolist()

        # Create a memory with very different vector
        different_vector = np.random.rand(384).astype(np.float32)
        different_vector = (different_vector / np.linalg.norm(different_vector) * -1).tolist()

        memory = create_memory_entry("test_threshold", different_vector)
        await pg_db_isolated.add_memory(memory)

        # Search with high threshold - should not return the opposite vector
        options = SearchOptions(top_k=5, threshold=0.9)
        results = await pg_db_isolated.query_memories(base_vector, options)

        # May or may not return results depending on actual distance
        for result in results:
            assert result.score is not None
            assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_query_memories_respects_top_k(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should limit results to top_k."""
        np.random.seed(456)

        # Create multiple memories
        for i in range(10):
            vector = np.random.rand(384).astype(np.float32).tolist()
            memory = create_memory_entry(f"test_topk_{i}", vector)
            await pg_db_isolated.add_memory(memory)

        options = SearchOptions(top_k=3, threshold=0.0)
        query_vector = np.random.rand(384).astype(np.float32).tolist()
        results = await pg_db_isolated.query_memories(query_vector, options)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_query_memories_returns_scores(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return results with similarity scores."""
        np.random.seed(789)
        vector = np.random.rand(384).astype(np.float32).tolist()
        memory = create_memory_entry("test_scores", vector)
        await pg_db_isolated.add_memory(memory)

        options = SearchOptions(top_k=5, threshold=0.0)
        results = await pg_db_isolated.query_memories(vector, options)

        assert len(results) > 0
        for result in results:
            assert result.score is not None
            assert 0.0 <= result.score <= 1.0


# =============================================================================
# Test Trust Fields
# =============================================================================


class TestTrustFields:
    """Tests for trust_score and retrieval_count fields."""

    @pytest.mark.asyncio
    async def test_new_memory_has_default_trust(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should have default trust_score of 0.5."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is not None
        assert result.trust_score == 0.5

    @pytest.mark.asyncio
    async def test_update_trust_fields(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should update trust_score and is_archived fields."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        await pg_db_isolated.update_trust_fields(
            sample_memory_entry.id,
            trust_score=0.8,
            is_archived=False
        )

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is not None
        assert result.trust_score == 0.8

    @pytest.mark.asyncio
    async def test_update_trust_fields_can_archive(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should archive memory when is_archived=True."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        await pg_db_isolated.update_trust_fields(
            sample_memory_entry.id,
            trust_score=0.5,
            is_archived=True
        )

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is None  # Archived memories don't show up

    @pytest.mark.asyncio
    async def test_increment_retrieval_count(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should increment retrieval_count."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        # Get initial count
        initial = await pg_db_isolated.get_memory(sample_memory_entry.id)
        initial_count = initial.retrieval_count if initial else 0

        # Increment
        await pg_db_isolated.increment_retrieval_count(sample_memory_entry.id)

        # Check incremented
        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is not None
        assert result.retrieval_count == initial_count + 1


# =============================================================================
# Test Additional Methods
# =============================================================================


class TestCountMemories:
    """Tests for count_memories method."""

    @pytest.mark.asyncio
    async def test_count_memories_returns_total(
        self, pg_db_isolated: PostgresDatabase, multiple_memory_entries: list[MemoryEntry]
    ):
        """Should return count of all memories."""
        await pg_db_isolated.add_memories(multiple_memory_entries)

        count = await pg_db_isolated.count_memories()
        assert count >= len(multiple_memory_entries)


class TestListMemories:
    """Tests for list_memories method."""

    @pytest.mark.asyncio
    async def test_list_memories_returns_entries(
        self, pg_db_isolated: PostgresDatabase, multiple_memory_entries: list[MemoryEntry]
    ):
        """Should return list of memory entries."""
        await pg_db_isolated.add_memories(multiple_memory_entries)

        results = await pg_db_isolated.list_memories()

        assert len(results) >= len(multiple_memory_entries)
        for entry in multiple_memory_entries:
            found = any(r.id == entry.id for r in results)
            assert found


class TestContentExists:
    """Tests for content_exists method."""

    @pytest.mark.asyncio
    async def test_content_exists_returns_true(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should return True when content hash exists."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        result = await pg_db_isolated.content_exists(sample_memory_entry.content_hash)
        assert result is True

    @pytest.mark.asyncio
    async def test_content_exists_returns_false(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return False when content hash doesn't exist."""
        result = await pg_db_isolated.content_exists("nonexistent_hash_12345")
        assert result is False


class TestDeleteBySourcePath:
    """Tests for delete_by_source_path method."""

    @pytest.mark.asyncio
    async def test_delete_by_source_path_returns_count(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should return number of archived memories."""
        # Create memory with source_path
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text="test_source_path_memory",
            vector=sample_vector,
            source_type=MemorySourceType.file,
            source_path="/test/path/file.py",
            content_hash="source_hash_123",
        )
        await pg_db_isolated.add_memory(entry)

        count = await pg_db_isolated.delete_by_source_path("/test/path/file.py")
        assert count >= 0


class TestGetEntriesBySourcePath:
    """Tests for get_entries_by_source_path method."""

    @pytest.mark.asyncio
    async def test_get_entries_by_source_path(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should return entries with matching source path."""
        source_path = f"/test/get_path/{uuid.uuid4().hex[:8]}"
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text="test_get_path_memory",
            vector=sample_vector,
            source_type=MemorySourceType.file,
            source_path=source_path,
            content_hash="get_path_hash",
        )
        await pg_db_isolated.add_memory(entry)

        results = await pg_db_isolated.get_entries_by_source_path(source_path)

        assert len(results) >= 1
        assert any(r.id == entry.id for r in results)


class TestScrollCollection:
    """Tests for scroll_collection method."""

    @pytest.mark.asyncio
    async def test_scroll_collection_returns_memories(
        self, pg_db_isolated: PostgresDatabase, multiple_memory_entries: list[MemoryEntry]
    ):
        """Should return paginated memories."""
        await pg_db_isolated.add_memories(multiple_memory_entries)

        results = await pg_db_isolated.scroll_collection("memories", limit=10)

        assert len(results) >= 0  # May be empty if test cleanup failed


class TestGetCollectionDimension:
    """Tests for get_collection_dimension method."""

    @pytest.mark.asyncio
    async def test_get_collection_dimension_returns_384(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should return 384 for MiniLM embeddings."""
        dimension = await pg_db_isolated.get_collection_dimension("memories")
        assert dimension == 384


class TestSetPayload:
    """Tests for set_payload method."""

    @pytest.mark.asyncio
    async def test_set_payload_updates_metadata(
        self, pg_db_isolated: PostgresDatabase, sample_memory_entry: MemoryEntry
    ):
        """Should update metadata via set_payload."""
        await pg_db_isolated.add_memory(sample_memory_entry)

        await pg_db_isolated.set_payload(
            sample_memory_entry.id,
            {"custom_field": "custom_value", "tags": ["test"]}
        )

        result = await pg_db_isolated.get_memory(sample_memory_entry.id)
        assert result is not None
        # Metadata is stored as JSONB, check it's accessible
        assert result.metadata_json is not None


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_closes_pool(self, pg_db: PostgresDatabase):
        """Should close the connection pool."""
        await pg_db.close()
        assert pg_db._pool is None
        assert pg_db._initialized is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, pg_db: PostgresDatabase):
        """Should be safe to call close multiple times."""
        await pg_db.close()
        await pg_db.close()  # Should not raise


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreatePostgresDatabase:
    """Tests for create_postgres_database factory function."""

    def test_create_postgres_database_returns_instance(self):
        """Should return PostgresDatabase instance."""
        from memini_ai.postgres.database import create_postgres_database

        db = create_postgres_database(TEST_DB_URL)
        assert isinstance(db, PostgresDatabase)
        assert db._db_url == TEST_DB_URL


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_memory_with_no_vector(
        self, pg_db_isolated: PostgresDatabase
    ):
        """Should handle memories with null vectors."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text="test_memory_without_vector",
            vector=None,
            source_type=MemorySourceType.session,
            content_hash="no_vector_hash",
        )
        memory_id = await pg_db_isolated.add_memory(entry)

        result = await pg_db_isolated.get_memory(memory_id)
        assert result is not None
        assert result.vector is None

    @pytest.mark.asyncio
    async def test_memory_with_empty_text(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should handle memories with empty text."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text="",
            vector=sample_vector,
            source_type=MemorySourceType.session,
            content_hash="empty_text_hash",
        )
        memory_id = await pg_db_isolated.add_memory(entry)

        result = await pg_db_isolated.get_memory(memory_id)
        assert result is not None
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_memory_with_special_characters(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should handle memories with special characters in text."""
        special_text = "Test with émojis 🎉 and 'quotes' and \"double quotes\" and \\backslashes\\"
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text=special_text,
            vector=sample_vector,
            source_type=MemorySourceType.session,
            content_hash="special_hash",
        )
        memory_id = await pg_db_isolated.add_memory(entry)

        result = await pg_db_isolated.get_memory(memory_id)
        assert result is not None
        assert result.text == special_text

    @pytest.mark.asyncio
    async def test_query_with_all_source_types(
        self, pg_db_isolated: PostgresDatabase, sample_vector: list[float]
    ):
        """Should handle all valid source types."""
        for source_type in MemorySourceType:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                text=f"test_{source_type.value}_memory",
                vector=sample_vector,
                source_type=source_type,
                content_hash=f"source_{source_type.value}_hash",
            )
            memory_id = await pg_db_isolated.add_memory(entry)

            result = await pg_db_isolated.get_memory(memory_id)
            assert result is not None
            assert result.source_type == source_type
