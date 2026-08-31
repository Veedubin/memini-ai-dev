"""Tests for the v0.8.0 image-recall RRF fan-out arm.

These tests focus on the dispatch logic in ``MemorySystem._query_multi_model_rrf``
and the new config fields — they mock the database, search, and
``memini_vision`` layers to avoid requiring a live PostgreSQL
connection or a CLIP model download.

Covers (10 tests):
- ``test_memories_image_table_created`` — migration SQL is idempotent, table exists with correct columns
- ``test_image_search_disabled_by_default`` — config default is False
- ``test_image_search_enabled_3rd_fanout`` — when enabled, the 3rd arm is called and results are fused
- ``test_image_rrf_score_combines_all_three`` — RRF math is correct with 3 lists
- ``test_image_arm_disabled_no_behavior_change`` — when disabled, byte-for-byte identical to v0.7.9
- ``test_memories_image_fk_cascade`` — schema FK is ON DELETE CASCADE
- ``test_source_type_image_allowed`` — CHECK constraint allows 'image'
- ``test_source_type_image_rejected_other_values`` — CHECK constraint still rejects unknown values
- ``test_dual_model_rrf_with_no_image_results`` — text results only, image arm returns empty, RRF still works
- ``test_image_query_with_clip_text_tower_mocked`` — mock memini_vision.ImageQuery, verify it's called with the right args
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
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


@pytest.fixture(autouse=True)
def _reset_config_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    """Reset the module-level config singleton so each test reads fresh env vars.

    ``get_config()`` caches a ``MeminiConfig`` at module level; without this
    reset, the first test that constructs a config pins the env-var state
    for all subsequent tests, and ``monkeypatch.setenv`` has no effect.
    We also chdir to a temp directory so the project ``.env`` file is not
    read, and clear ``MEMINI_IMAGE_SEARCH_ENABLED`` so default-value
    assertions are reliable.
    """
    import memini_ai.config as cfg_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MEMINI_IMAGE_SEARCH_ENABLED", raising=False)
    old = cfg_mod._config
    cfg_mod._config = None
    yield
    cfg_mod._config = old


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock VectorDatabase that exposes the 1024 + image API surface."""
    db = MagicMock()
    db.add_memory = AsyncMock(return_value="mem-abc-123")
    db.add_memory_1024 = AsyncMock(return_value="1024-row-uuid")
    db.get_memory_1024_by_memory_id = AsyncMock(return_value=None)
    db.query_memories_1024 = AsyncMock(return_value=[])
    db.search_image_memories = AsyncMock(return_value=[])
    db._expand_384_to_1024 = MagicMock(
        side_effect=lambda v, dim: (
            [0.0] * dim if v is None else (list(v) + [0.0] * (dim - len(v)))
        )
    )
    db.content_exists = AsyncMock(return_value=False)
    db.initialize = AsyncMock()
    db._initialized = True
    return db


@pytest.fixture
def memory_system(mock_db: MagicMock) -> MemorySystem:
    """A MemorySystem with a mocked DB (no real connection)."""
    cfg = MemorySystemConfig(
        embedding_mode="auto",
        rrf_k=60,
        enable_deduplication=False,
    )
    system = MemorySystem(db=mock_db, config=cfg)
    system._initialized = True
    return system


def _make_entry(eid: str, text: str = "test") -> MemoryEntry:
    """Helper to build a minimal MemoryEntry."""
    return MemoryEntry(id=eid, text=text, sourceType=MemorySourceType.session)


# =============================================================================
# 1. Migration / schema tests
# =============================================================================


def test_memories_image_table_created() -> None:
    """Migration SQL is idempotent and defines the table with correct columns."""
    from memini_ai.postgres.schema import (
        SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN,
        SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW,
        SQL_CREATE_MEMORIES_IMAGE_INDEXES,
        SQL_CREATE_MEMORIES_IMAGE_TABLE,
        SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE,
        get_schema_sql,
    )

    table_sql = SQL_CREATE_MEMORIES_IMAGE_TABLE
    # All the columns from the spec
    for col in [
        "memories_image",
        "memory_id",
        "embedding",
        "vector(768)",
        "embedding_model",
        "image_path",
        "image_sha256",
        "mime_type",
        "width",
        "height",
        "caption",
        "file_size_bytes",
        "trust_score",
        "created_at",
        "ON DELETE CASCADE",
        "IF NOT EXISTS",
    ]:
        assert col in table_sql, f"Column/clause '{col}' missing from table SQL"

    # Idempotent: IF NOT EXISTS on the CREATE TABLE
    assert "CREATE TABLE IF NOT EXISTS memories_image" in table_sql

    # Index SQL is idempotent
    for idx_sql in (
        SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_DISKANN,
        SQL_CREATE_MEMORIES_IMAGE_EMBEDDING_INDEX_HNSW,
        SQL_CREATE_MEMORIES_IMAGE_INDEXES,
    ):
        assert "IF NOT EXISTS" in idx_sql

    # source_type CHECK includes 'image'
    check_sql = SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE
    assert "'image'" in check_sql
    assert "DROP CONSTRAINT IF EXISTS" in check_sql

    # get_schema_sql() includes the image table
    full = get_schema_sql(use_vectorscale=True)
    assert "memories_image" in full
    full_no_vs = get_schema_sql(use_vectorscale=False)
    assert "memories_image" in full_no_vs


def test_memories_image_fk_cascade() -> None:
    """Schema FK is ON DELETE CASCADE (image row goes when memory is deleted)."""
    from memini_ai.postgres.schema import SQL_CREATE_MEMORIES_IMAGE_TABLE

    assert (
        "REFERENCES memories(id) ON DELETE CASCADE" in SQL_CREATE_MEMORIES_IMAGE_TABLE
    )


def test_source_type_image_allowed() -> None:
    """CHECK constraint allows 'image' as a source_type."""
    from memini_ai.postgres.schema import SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE

    # The constraint should include all 7 values
    for val in [
        "'session'",
        "'file'",
        "'web'",
        "'boomerang'",
        "'project'",
        "'thought'",
        "'image'",
    ]:
        assert val in SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE, (
            f"'{val}' missing from CHECK constraint"
        )


def test_source_type_image_rejected_other_values() -> None:
    """CHECK constraint still rejects unknown source_type values.

    The constraint is a positive list — 'video', 'audio', 'random' are
    NOT in the allowed set, so inserting them would fail.
    """
    from memini_ai.postgres.schema import SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE

    # Verify the constraint is a positive allow-list (not a deny-list)
    assert "IN (" in SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE
    # The constraint does NOT mention these values
    for bad in ["'video'", "'audio'", "'random'"]:
        assert bad not in SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK_IMAGE


# =============================================================================
# 2. Config tests
# =============================================================================


def test_image_search_disabled_by_default() -> None:
    """Config default: image_search_enabled is False."""
    cfg = MeminiConfig()
    assert cfg.image_search_enabled is False
    assert cfg.image_clip_model == "clip-ViT-B-32"
    assert cfg.image_clip_device == "auto"
    assert cfg.image_dir == "~/.memini-ai/images"
    assert cfg.image_db_url == ""


def test_image_clip_model_validator_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid CLIP model name raises ValueError."""
    monkeypatch.setenv("MEMINI_IMAGE_CLIP_MODEL", "resnet-50")
    with pytest.raises(ValueError, match="Invalid image_clip_model"):
        MeminiConfig()


def test_image_clip_device_validator_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid device raises ValueError."""
    monkeypatch.setenv("MEMINI_IMAGE_CLIP_DEVICE", "tpu")
    with pytest.raises(ValueError, match="Invalid image_clip_device"):
        MeminiConfig()


# =============================================================================
# 3. RRF dispatch tests (image arm)
# =============================================================================


def test_image_arm_disabled_no_behavior_change(
    memory_system: MemorySystem,
    mock_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MEMINI_IMAGE_SEARCH_ENABLED is unset/false, the image arm is
    NOT called and the query path is byte-for-byte identical to v0.7.9.

    This is the critical backwards-compat test: text-only users see
    zero behavior change.
    """
    # Ensure image search is off
    monkeypatch.delenv("MEMINI_IMAGE_SEARCH_ENABLED", raising=False)

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        return [_make_entry("text-1"), _make_entry("text-2")]

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return [_make_entry("text-2"), _make_entry("text-3")]

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]
    memory_system._db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)  # type: ignore[method-assign]

    # Patch generate_embedding so we don't hit the real model
    async def fake_embed(text: str) -> MagicMock:
        m = MagicMock()
        m.embedding = [0.1] * 384
        return m

    with patch("memini_ai.memory.system.generate_embedding", new=fake_embed):
        result = asyncio.run(
            memory_system.query_memories(
                "hello world", SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
            )
        )

    # The image arm must NOT have been called
    mock_db.search_image_memories.assert_not_awaited()
    # Results come only from the 2 text lists
    ids = [e.id for e in result]
    assert set(ids).issubset({"text-1", "text-2", "text-3"})
    # text-2 appears in both lists → should be ranked first (boosted)
    assert ids[0] == "text-2"


def test_image_search_enabled_3rd_fanout(
    memory_system: MemorySystem,
    mock_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When MEMINI_IMAGE_SEARCH_ENABLED=true, the 3rd arm is called and
    results are fused into the final ranking."""
    monkeypatch.setenv("MEMINI_IMAGE_SEARCH_ENABLED", "true")

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        return [_make_entry("text-1"), _make_entry("text-2")]

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return [_make_entry("text-2"), _make_entry("text-3")]

    # The image arm returns a memory that's ONLY in the image list
    image_only_entries = [_make_entry("img-only-1")]
    mock_db.search_image_memories = AsyncMock(return_value=image_only_entries)

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]
    memory_system._db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)  # type: ignore[method-assign]

    async def fake_embed(text: str) -> MagicMock:
        m = MagicMock()
        m.embedding = [0.1] * 384
        return m

    # Patch memini_vision so the lazy import in _image_recall_arm works
    fake_clip = MagicMock()
    fake_clip.encode_text = MagicMock(return_value=[0.0] * 768)
    fake_query_obj = MagicMock()
    fake_query_obj.search_by_text = AsyncMock(return_value=[])
    with (
        patch("memini_ai.memory.system.generate_embedding", new=fake_embed),
        patch("memini_vision.ClipEmbedder", return_value=fake_clip),
        patch("memini_vision.ImageIndex", return_value=MagicMock()),
        patch("memini_vision.ImageQuery", return_value=fake_query_obj),
    ):
        result = asyncio.run(
            memory_system.query_memories(
                "hello world", SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
            )
        )

    # The image db helper should have been called (3rd arm activated)
    mock_db.search_image_memories.assert_awaited()
    # img-only-1 should appear in the results (it came from the image arm)
    result_ids = [e.id for e in result]
    assert "img-only-1" in result_ids


def test_image_rrf_score_combines_all_three() -> None:
    """RRF math is correct with 3 lists — a memory in all 3 lists gets
    the sum of all 3 contributions (the highest possible boost)."""
    from memini_ai.memory.rrf import reciprocal_rank_fusion

    list_384 = ["a", "b", "c"]
    list_1024 = ["b", "a", "d"]
    list_image = ["c", "b", "e"]

    fused = reciprocal_rank_fusion([list_384, list_1024, list_image])
    by_id = dict(fused)

    # "b" appears in all 3 lists:
    #   rank 1 in list_384 → 1/61
    #   rank 0 in list_1024 → 1/60
    #   rank 1 in list_image → 1/61
    expected_b = 1 / 61 + 1 / 60 + 1 / 61
    assert by_id["b"] == pytest.approx(expected_b)

    # "a" in 2 lists: rank 0 in 384 (1/60) + rank 1 in 1024 (1/61)
    assert by_id["a"] == pytest.approx(1 / 60 + 1 / 61)

    # "c" in 2 lists: rank 2 in 384 (1/62) + rank 0 in image (1/60)
    assert by_id["c"] == pytest.approx(1 / 62 + 1 / 60)

    # "e" only in image: rank 2 → 1/62
    assert by_id["e"] == pytest.approx(1 / 62)

    # "b" should be the top result (highest fused score)
    assert fused[0][0] == "b"


def test_dual_model_rrf_with_no_image_results(
    memory_system: MemorySystem,
    mock_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the image arm returns empty results, RRF still works with 2 lists.

    This simulates the common case where image search is enabled but the
    image table is empty (or no images match the query). The text RRF
    should proceed normally with 2 lists.
    """
    monkeypatch.setenv("MEMINI_IMAGE_SEARCH_ENABLED", "true")

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        return [_make_entry("text-1"), _make_entry("text-2")]

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return [_make_entry("text-3")]

    # Image arm returns empty
    mock_db.search_image_memories = AsyncMock(return_value=[])

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]
    memory_system._db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)  # type: ignore[method-assign]

    async def fake_embed(text: str) -> MagicMock:
        m = MagicMock()
        m.embedding = [0.1] * 384
        return m

    fake_clip = MagicMock()
    fake_clip.encode_text = MagicMock(return_value=[0.0] * 768)
    fake_query_obj = MagicMock()
    fake_query_obj.search_by_text = AsyncMock(return_value=[])
    with (
        patch("memini_ai.memory.system.generate_embedding", new=fake_embed),
        patch("memini_vision.ClipEmbedder", return_value=fake_clip),
        patch("memini_vision.ImageIndex", return_value=MagicMock()),
        patch("memini_vision.ImageQuery", return_value=fake_query_obj),
    ):
        result = asyncio.run(
            memory_system.query_memories(
                "hello world", SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
            )
        )

    # Results come from the 2 text lists only (image arm was empty)
    result_ids = {e.id for e in result}
    assert result_ids == {"text-1", "text-2", "text-3"}


def test_image_query_with_clip_text_tower_mocked(
    memory_system: MemorySystem,
    mock_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock memini_vision.ImageQuery, verify it's called with the right args
    (query text + fetch_k limit)."""
    monkeypatch.setenv("MEMINI_IMAGE_SEARCH_ENABLED", "true")

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        return []

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return []

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]
    memory_system._db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)  # type: ignore[method-assign]

    async def fake_embed(text: str) -> MagicMock:
        m = MagicMock()
        m.embedding = [0.1] * 384
        return m

    # Capture the args passed to ImageQuery.search_by_text
    captured: dict[str, object] = {}
    fake_clip = MagicMock()
    fake_clip.encode_text = MagicMock(return_value=[0.0] * 768)
    fake_query_obj = MagicMock()

    async def fake_search_by_text(text: str, limit: int = 10) -> list:
        captured["text"] = text
        captured["limit"] = limit
        return []

    fake_query_obj.search_by_text = fake_search_by_text

    with (
        patch("memini_ai.memory.system.generate_embedding", new=fake_embed),
        patch("memini_vision.ClipEmbedder", return_value=fake_clip),
        patch("memini_vision.ImageIndex", return_value=MagicMock()),
        patch("memini_vision.ImageQuery", return_value=fake_query_obj),
    ):
        asyncio.run(
            memory_system.query_memories(
                "terminal traceback python error",
                SearchOptions(topK=5, strategy=SearchStrategy.TIERED),
            )
        )

    # ImageQuery.search_by_text should have been called with the query text
    assert captured.get("text") == "terminal traceback python error"
    # The limit should be the over-fetch value (max(topK*2, topK+5) = 10)
    assert captured.get("limit") == 10


def test_image_arm_failure_is_swallowed(
    memory_system: MemorySystem,
    mock_db: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the image arm raises (CLIP model download fails, etc.), the
    text RRF proceeds with 2 lists instead of 3. Best-effort guarantee."""
    monkeypatch.setenv("MEMINI_IMAGE_SEARCH_ENABLED", "true")

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        return [_make_entry("text-1")]

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return [_make_entry("text-2")]

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]
    memory_system._db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)  # type: ignore[method-assign]

    async def fake_embed(text: str) -> MagicMock:
        m = MagicMock()
        m.embedding = [0.1] * 384
        return m

    # memini_vision import or ImageQuery construction fails
    with (
        patch("memini_ai.memory.system.generate_embedding", new=fake_embed),
        patch(
            "memini_vision.ClipEmbedder",
            side_effect=RuntimeError("CLIP download failed"),
        ),
    ):
        result = asyncio.run(
            memory_system.query_memories(
                "hello world", SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
            )
        )

    # Text RRF still returns results (image failure was swallowed)
    result_ids = {e.id for e in result}
    assert result_ids == {"text-1", "text-2"}
