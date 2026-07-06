"""Tests for memory schema types."""

from __future__ import annotations

from datetime import UTC, datetime

from memini_ai.memory.schema import (
    MEMORY_TABLE_NAME,
    MemoryEntry,
    MemorySourceType,
    SearchFilter,
    SearchOptions,
    SearchStrategy,
)


class TestMemorySourceType:
    """Tests for MemorySourceType enum."""

    def test_all_source_types_exist(self) -> None:
        """All expected source types should exist."""
        assert MemorySourceType.session == "session"
        assert MemorySourceType.file == "file"
        assert MemorySourceType.web == "web"
        assert MemorySourceType.boomerang == "boomerang"
        assert MemorySourceType.project == "project"

    def test_source_type_is_string_enum(self) -> None:
        """MemorySourceType should be a string enum for serialization."""
        assert isinstance(MemorySourceType.session, str)
        assert MemorySourceType.session.value == "session"


class TestMemoryEntry:
    """Tests for MemoryEntry model."""

    def test_create_minimal_entry(self) -> None:
        """Should create entry with minimal required fields."""
        entry = MemoryEntry(
            text="Test memory content",
            source_type=MemorySourceType.session,
            content_hash="abc123",
        )
        assert entry.text == "Test memory content"
        assert entry.source_type == MemorySourceType.session
        assert entry.content_hash == "abc123"

    def test_create_full_entry(self) -> None:
        """Should create entry with all fields."""
        now = datetime.now(UTC)
        entry = MemoryEntry(
            id="custom-id",
            text="Full memory",
            vector=[0.1, 0.2, 0.3],
            source_type=MemorySourceType.file,
            source_path="/path/to/file.py",
            timestamp=now,
            content_hash="fullhash",
            metadata_json='{"key": "value"}',
            session_id="session-123",
            project_id="project-456",
            score=0.95,
        )
        assert entry.id == "custom-id"
        assert entry.vector == [0.1, 0.2, 0.3]
        assert entry.source_path == "/path/to/file.py"
        assert entry.timestamp == now
        assert entry.metadata_json == '{"key": "value"}'
        assert entry.session_id == "session-123"
        assert entry.project_id == "project-456"
        assert entry.score == 0.95

    def test_default_values(self) -> None:
        """Should have sensible defaults for optional fields."""
        entry = MemoryEntry(
            text="Minimal",
            source_type=MemorySourceType.session,
            content_hash="hash123",
        )
        assert entry.id is not None  # Generated UUID
        assert entry.vector is None
        assert entry.source_path is None
        assert entry.metadata_json is None
        assert entry.session_id is None
        assert entry.project_id is None
        assert entry.score is None

    def test_id_is_uuid_format(self) -> None:
        """Default ID should be a valid UUID string."""
        import uuid as uuid_module

        entry = MemoryEntry(
            text="Test",
            source_type=MemorySourceType.session,
            content_hash="hash",
        )
        # Should not raise
        _uuid_obj = uuid_module.UUID(entry.id)

    def test_timestamp_default_is_reasonable(self) -> None:
        """Default timestamp should be recent (within last minute)."""
        entry = MemoryEntry(
            text="Test",
            source_type=MemorySourceType.session,
            content_hash="hash",
        )
        now = datetime.now(UTC)
        diff = abs(
            (
                entry.timestamp.replace(tzinfo=None) - now.replace(tzinfo=None)
            ).total_seconds()
        )
        assert diff < 60  # Within 1 minute

    def test_populate_by_name_with_camel_case_alias(self) -> None:
        """Should accept camelCase field names on input."""
        entry = MemoryEntry(
            text="Test",
            sourceType=MemorySourceType.project,  # camelCase input
            contentHash="hash",
            sessionId="my-session",  # camelCase input
        )
        assert entry.session_id == "my-session"

    def test_serialization_to_dict_with_aliases(self) -> None:
        """Should serialize to dict with proper alias names."""
        entry = MemoryEntry(
            text="Test",
            source_type=MemorySourceType.session,
            content_hash="hash",
            project_id="proj-1",
        )
        data = entry.model_dump(by_alias=True)
        assert data["sourceType"] == "session"
        assert data["sessionId"] is None  # Not set via snake_case
        assert data["projectId"] == "proj-1"
        assert "text" in data


class TestSearchFilter:
    """Tests for SearchFilter model."""

    def test_empty_filter(self) -> None:
        """Should create empty filter with all None values."""
        filter = SearchFilter()
        assert filter.source_type is None
        assert filter.session_id is None
        assert filter.since is None
        assert filter.project_id is None

    def test_filter_with_values(self) -> None:
        """Should create filter with specific values."""
        since = datetime(2024, 1, 1, tzinfo=UTC)
        filter = SearchFilter(
            source_type=MemorySourceType.file,
            session_id="session-123",
            since=since,
            project_id="project-abc",
        )
        assert filter.source_type == MemorySourceType.file
        assert filter.session_id == "session-123"
        assert filter.since == since
        assert filter.project_id == "project-abc"

    def test_populate_by_name_with_camel_case(self) -> None:
        """Should accept camelCase field names on input."""
        filter = SearchFilter(
            sourceType=MemorySourceType.boomerang,
            projectId="my-project",
        )
        assert filter.source_type == MemorySourceType.boomerang
        assert filter.project_id == "my-project"


class TestSearchStrategy:
    """Tests for SearchStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """All expected strategies should exist."""
        assert SearchStrategy.TIERED == "TIERED"
        assert SearchStrategy.VECTOR_ONLY == "VECTOR_ONLY"
        assert SearchStrategy.TEXT_ONLY == "TEXT_ONLY"
        assert SearchStrategy.PARALLEL == "PARALLEL"

    def test_is_string_enum(self) -> None:
        """SearchStrategy should be a string enum."""
        assert isinstance(SearchStrategy.TIERED, str)
        assert SearchStrategy.TIERED.value == "TIERED"


class TestSearchOptions:
    """Tests for SearchOptions model."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        options = SearchOptions()
        assert options.top_k == 5
        assert options.strategy == SearchStrategy.TIERED
        # v0.7.3: default threshold lowered from 0.72 to 0.0 — the
        # SQL-side cosine filter at 0.72 was rejecting most legitimate
        # semantic matches (MiniLM-L6-v2 similarities land in 0.4-0.7).
        # Ranking is now handled by RRF / parallel_search, and the
        # threshold is only an opt-in hard cutoff.
        assert options.threshold == 0.0
        assert options.filter is not None
        assert options.filter.source_type is None

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        options = SearchOptions(
            top_k=10,
            strategy=SearchStrategy.PARALLEL,
            threshold=0.85,
            filter=SearchFilter(source_type=MemorySourceType.file),
        )
        assert options.top_k == 10
        assert options.strategy == SearchStrategy.PARALLEL
        assert options.threshold == 0.85
        assert options.filter.source_type == MemorySourceType.file

    def test_populate_by_name_with_camel_case(self) -> None:
        """Should accept camelCase field names on input."""
        options = SearchOptions(
            topK=20,
            strategy=SearchStrategy.VECTOR_ONLY,
        )
        assert options.top_k == 20
        assert options.strategy == SearchStrategy.VECTOR_ONLY


class TestConstants:
    """Tests for module constants."""

    def test_memory_table_name(self) -> None:
        """MEMORY_TABLE_NAME should be 'memories'."""
        assert MEMORY_TABLE_NAME == "memories"
