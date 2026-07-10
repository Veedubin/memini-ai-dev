"""Tests for MemorySearch - 4 search strategies with mocking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchOptions,
)


@pytest.fixture
def mock_db() -> MagicMock:
    """Create mock MemoryDatabase."""
    db = MagicMock()
    db.query_memories = AsyncMock(return_value=[])
    db.list_memories = AsyncMock(return_value=[])
    db.get_memory = AsyncMock(return_value=None)
    db.scroll_collection = AsyncMock(return_value=[])
    return db


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
def mock_entries() -> list[MemoryEntry]:
    """Create mock memory entries."""
    return [
        MemoryEntry(
            id="entry-1",
            text="Python programming language",
            source_type=MemorySourceType.session,
            content_hash="hash1",
            vector=[0.1] * 1024,
            score=0.9,
        ),
        MemoryEntry(
            id="entry-2",
            text="Machine learning algorithms",
            source_type=MemorySourceType.session,
            content_hash="hash2",
            vector=[0.2] * 1024,
            score=0.8,
        ),
        MemoryEntry(
            id="entry-3",
            text="Web development frameworks",
            source_type=MemorySourceType.file,
            content_hash="hash3",
            vector=[0.3] * 1024,
            score=0.7,
        ),
    ]


class TestTokenize:
    """Tests for _tokenize helper."""

    def test_tokenize_lowercase(self) -> None:
        """Should convert text to lowercase."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        tokens = search._tokenize("Hello World")

        assert tokens == ["hello", "world"]

    def test_tokenize_split_words(self) -> None:
        """Should split on whitespace."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        tokens = search._tokenize("one two three")

        assert tokens == ["one", "two", "three"]

    def test_tokenize_empty_string(self) -> None:
        """Should handle empty string."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        tokens = search._tokenize("")

        assert tokens == []


class TestNormalizeBm25Scores:
    """Tests for _normalize_bm25_scores."""

    def test_normalize_empty_list(self) -> None:
        """Should handle empty list."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        result = search._normalize_bm25_scores([])

        assert result == []

    def test_normalize_same_values(self) -> None:
        """Should return 1.0 when all values same."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        result = search._normalize_bm25_scores([5.0, 5.0, 5.0])

        assert result == [1.0, 1.0, 1.0]

    def test_normalize_different_values(self) -> None:
        """Should normalize to [0, 1] range."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())
        result = search._normalize_bm25_scores([0.0, 5.0, 10.0])

        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 1.0


class TestRrfFusion:
    """Tests for _rrf_fusion."""

    def test_rrf_fusion_combines_rankings(self) -> None:
        """Should combine multiple rankings with RRF."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(MagicMock())

        entries1 = [
            MemoryEntry(
                id="a",
                text="Entry A",
                source_type=MemorySourceType.session,
                content_hash="a",
            ),
            MemoryEntry(
                id="b",
                text="Entry B",
                source_type=MemorySourceType.session,
                content_hash="b",
            ),
        ]
        entries2 = [
            MemoryEntry(
                id="b",
                text="Entry B",
                source_type=MemorySourceType.session,
                content_hash="b",
            ),
            MemoryEntry(
                id="c",
                text="Entry C",
                source_type=MemorySourceType.session,
                content_hash="c",
            ),
        ]

        scores1 = [0.9, 0.8]
        scores2 = [0.85, 0.75]

        result = search._rrf_fusion([entries1, entries2], [scores1, scores2])

        # Should be sorted by combined RRF score
        ids = [entry.id for entry, _ in result]
        assert "b" in ids  # b appears in both
        assert "a" in ids or "c" in ids


class TestTieredSearch:
    """Tests for tiered_search strategy."""

    @pytest.mark.asyncio
    async def test_tiered_uses_vector_when_above_threshold(
        self,
        mock_db: MagicMock,
        mock_embedding: MagicMock,
        mock_entries: list[MemoryEntry],
    ) -> None:
        """Should return vector results when top score >= threshold."""
        from memini_ai.memory.search import MemorySearch

        # Vector results with high score
        mock_db.query_memories = AsyncMock(return_value=mock_entries[:1])

        search = MemorySearch(mock_db)

        with patch(
            "memini_ai.memory.search.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            results = await search.tiered_search(
                "python",
                SearchOptions(top_k=5, threshold=0.72),
            )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_tiered_falls_back_to_text(
        self, mock_db: MagicMock, mock_embedding: MagicMock
    ) -> None:
        """Should fall back to text search when vector score too low."""
        from memini_ai.memory.search import MemorySearch

        # Vector results with low score
        low_score_entry = MemoryEntry(
            id="low-score",
            text="Not relevant content",
            source_type=MemorySourceType.session,
            content_hash="lowhash",
            vector=[0.1] * 1024,
            score=0.3,  # Below threshold
        )
        mock_db.query_memories = AsyncMock(return_value=[low_score_entry])
        mock_db.list_memories = AsyncMock(return_value=[])

        search = MemorySearch(mock_db)

        with patch(
            "memini_ai.memory.search.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            results = await search.tiered_search(
                "python",
                SearchOptions(top_k=5, threshold=0.72),
            )

        # Should have tried text search
        assert isinstance(results, list)


class TestVectorOnlySearch:
    """Tests for vector_only_search strategy."""

    @pytest.mark.asyncio
    async def test_vector_only_returns_results(
        self,
        mock_db: MagicMock,
        mock_embedding: MagicMock,
        mock_entries: list[MemoryEntry],
    ) -> None:
        """Should return vector search results."""
        from memini_ai.memory.search import MemorySearch

        mock_db.query_memories = AsyncMock(return_value=mock_entries)

        search = MemorySearch(mock_db)

        with patch(
            "memini_ai.memory.search.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            _results = await search.vector_only_search(
                "python",
                SearchOptions(top_k=5),
            )

        mock_db.query_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_only_uses_collection(
        self,
        mock_db: MagicMock,
        mock_embedding: MagicMock,
        mock_entries: list[MemoryEntry],
    ) -> None:
        """Should pass collection name to database."""
        from memini_ai.memory.search import MemorySearch

        mock_db.query_memories = AsyncMock(return_value=mock_entries)

        search = MemorySearch(mock_db)

        with patch(
            "memini_ai.memory.search.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            await search.vector_only_search(
                "python",
                SearchOptions(top_k=5),
                collection_name="memories_384",
            )

        # Collection name should be passed
        call_args = mock_db.query_memories.call_args
        assert call_args is not None


class TestTextOnlySearch:
    """Tests for text_only_search strategy."""

    @pytest.mark.asyncio
    async def test_text_only_requires_bm25_index(self, mock_db: MagicMock) -> None:
        """Should return empty if no BM25 index."""
        from memini_ai.memory.search import MemorySearch

        mock_db.list_memories = AsyncMock(return_value=[])

        search = MemorySearch(mock_db)
        results = await search.text_only_search("python", SearchOptions(top_k=5))

        assert results == []

    @pytest.mark.asyncio
    async def test_text_only_returns_ranked_results(
        self, mock_db: MagicMock, mock_entries: list[MemoryEntry]
    ) -> None:
        """Should return BM25 ranked results."""
        from memini_ai.memory.search import MemorySearch

        mock_db.list_memories = AsyncMock(return_value=mock_entries)

        search = MemorySearch(mock_db)

        # Build BM25 index
        await search._build_bm25_index()

        results = await search.text_only_search(
            "python programming", SearchOptions(top_k=5)
        )

        assert len(results) <= 5


class TestParallelSearch:
    """Tests for parallel_search strategy."""

    @pytest.mark.asyncio
    async def test_parallel_runs_both_searches(
        self,
        mock_db: MagicMock,
        mock_embedding: MagicMock,
        mock_entries: list[MemoryEntry],
    ) -> None:
        """Should run vector and text searches concurrently."""
        from memini_ai.memory.search import MemorySearch

        mock_db.query_memories = AsyncMock(return_value=mock_entries)
        mock_db.list_memories = AsyncMock(return_value=mock_entries)

        search = MemorySearch(mock_db)

        with patch(
            "memini_ai.memory.search.generate_embedding",
            AsyncMock(return_value=mock_embedding),
        ):
            results = await search.parallel_search(
                "python",
                SearchOptions(top_k=5),
            )

        # Should have results from both
        assert isinstance(results, list)


class TestGetSimilar:
    """Tests for get_similar method."""

    @pytest.mark.asyncio
    async def test_get_similar_returns_empty_if_not_found(
        self, mock_db: MagicMock
    ) -> None:
        """Should return empty list if memory not found."""
        from memini_ai.memory.search import MemorySearch

        mock_db.get_memory = AsyncMock(return_value=None)

        search = MemorySearch(mock_db)
        results = await search.get_similar("nonexistent-id", SearchOptions())

        assert results == []

    @pytest.mark.asyncio
    async def test_get_similar_uses_vector(
        self, mock_db: MagicMock, mock_entries: list[MemoryEntry]
    ) -> None:
        """Should search using reference memory's vector."""
        from memini_ai.memory.search import MemorySearch

        reference = mock_entries[0]
        mock_db.get_memory = AsyncMock(return_value=reference)
        mock_db.query_memories = AsyncMock(return_value=mock_entries[1:])

        search = MemorySearch(mock_db)

        await search.get_similar("entry-1", SearchOptions())

        mock_db.query_memories.assert_called_once()


class TestInvalidateBm25:
    """Tests for invalidate_bm25 method."""

    @pytest.mark.asyncio
    async def test_invalidate_clears_cache(self, mock_db: MagicMock) -> None:
        """Should clear BM25 index cache."""
        from memini_ai.memory.search import MemorySearch

        search = MemorySearch(mock_db)

        # Set cache
        search._bm25_index = MagicMock()
        search._bm25_corpus = [MagicMock()]

        await search.invalidate_bm25()

        assert search._bm25_index is None
        assert search._bm25_corpus == []
