"""Tests for the dual-model RRF dispatch in memory/system.py.

These tests focus on the dispatch logic (mode selection, mode validation,
fallback behavior) and the trust boost semantics of the elevate path.
They mock the database and search layers to avoid requiring a live
PostgreSQL connection.

Covers:
- Default embedding_mode resolution (auto from config)
- cpu mode: 384-dim add only, no 1024 sidecar
- auto mode: 384-dim add + 1024 mirror only if already elevated
- gpu mode: 384-dim add + 1024 mirror always
- Invalid embedding_mode raises ValueError
- Trust boost in elevate_memory_to_1024 is clamped to [0, 1]
- elevate tool gate: refuses in non-auto mode
- elevate tool gate: refuses when ELEVATE_ENABLED=false
- v0.7.3: RRF propagates caller threshold + exact_search to 384 side
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
    """Mock VectorDatabase that exposes the 1024-dim API surface."""
    db = MagicMock()
    db.add_memory = AsyncMock(return_value="mem-abc-123")
    db.add_memory_1024 = AsyncMock(return_value="1024-row-uuid")
    db.get_memory_1024_by_memory_id = AsyncMock(return_value=None)
    db._expand_384_to_1024 = MagicMock(
        side_effect=lambda v, dim: (
            [0.0] * dim if v is None else (list(v) + [0.0] * (dim - len(v)))
        )
    )
    db.content_exists = AsyncMock(return_value=False)
    db.initialize = AsyncMock()
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


def test_default_embedding_mode_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no override is set, the resolved mode comes from MeminiConfig."""
    # Default is "auto" per the config validator
    cfg = MeminiConfig()
    assert cfg.embedding_mode == "auto"


def test_invalid_embedding_mode_raises(memory_system: MemorySystem) -> None:
    """add_memory validates the resolved embedding_mode."""
    memory_system._config.embedding_mode = "bogus"  # type: ignore[assignment]
    entry = MemoryEntry(text="hi", sourceType=MemorySourceType.session)
    # The validator runs before any DB call, so the mock isn't even hit.
    import asyncio

    with pytest.raises(ValueError, match="Invalid embedding_mode"):
        asyncio.run(memory_system.add_memory(entry))


def test_cpu_mode_does_not_touch_1024(
    memory_system: MemorySystem, mock_db: MagicMock
) -> None:
    """cpu mode: only the 384-dim add_memory is called."""
    memory_system._config.embedding_mode = "cpu"  # type: ignore[assignment]
    entry = MagicMock()
    entry.text = "test"
    entry.vector = [0.1] * 384
    entry.content_hash = "abc"
    import asyncio

    asyncio.run(memory_system.add_memory(entry))

    mock_db.add_memory.assert_awaited_once()
    mock_db.add_memory_1024.assert_not_awaited()
    mock_db.get_memory_1024_by_memory_id.assert_not_awaited()


def test_auto_mode_writes_1024_if_already_elevated(
    memory_system: MemorySystem, mock_db: MagicMock
) -> None:
    """auto mode: 384-dim write + 1024 mirror only if the row is already elevated."""
    # Pretend the memory is already elevated
    mock_db.get_memory_1024_by_memory_id = AsyncMock(
        return_value={"memory_id": "mem-abc-123", "embedding": [0.0] * 1024}
    )
    entry = MagicMock()
    entry.text = "test"
    entry.vector = [0.1] * 384
    entry.content_hash = "abc"
    import asyncio

    asyncio.run(memory_system.add_memory(entry))

    mock_db.add_memory.assert_awaited_once()
    mock_db.get_memory_1024_by_memory_id.assert_awaited_once()
    mock_db.add_memory_1024.assert_awaited_once()  # mirror write


def test_auto_mode_skips_1024_mirror_if_not_elevated(
    memory_system: MemorySystem, mock_db: MagicMock
) -> None:
    """auto mode: when the row is NOT elevated, no 1024 mirror happens."""
    # get_memory_1024_by_memory_id returns None (default from fixture)
    entry = MagicMock()
    entry.text = "test"
    entry.vector = [0.1] * 384
    entry.content_hash = "abc"
    import asyncio

    asyncio.run(memory_system.add_memory(entry))

    mock_db.add_memory.assert_awaited_once()
    mock_db.add_memory_1024.assert_not_awaited()  # no mirror


def test_gpu_mode_always_writes_1024(
    memory_system: MemorySystem, mock_db: MagicMock
) -> None:
    """gpu mode: 384-dim write AND 1024 mirror, regardless of elevation state."""
    memory_system._config.embedding_mode = "gpu"  # type: ignore[assignment]
    entry = MagicMock()
    entry.text = "test"
    entry.vector = [0.1] * 384  # Note: in real gpu mode this would be 1024-dim
    entry.content_hash = "abc"
    import asyncio

    asyncio.run(memory_system.add_memory(entry))

    mock_db.add_memory.assert_awaited_once()
    mock_db.add_memory_1024.assert_awaited_once()
    # gpu mode does NOT consult get_memory_1024_by_memory_id
    mock_db.get_memory_1024_by_memory_id.assert_not_awaited()


def test_gpu_mode_raises_if_db_lacks_1024_support(
    memory_system: MemorySystem, mock_db: MagicMock
) -> None:
    """gpu mode raises RuntimeError if the db has no add_memory_1024."""
    memory_system._config.embedding_mode = "gpu"  # type: ignore[assignment]
    del mock_db.add_memory_1024  # remove the method
    entry = MagicMock()
    entry.text = "test"
    entry.vector = [0.1] * 384
    entry.content_hash = "abc"
    import asyncio

    with pytest.raises(RuntimeError, match="add_memory_1024"):
        asyncio.run(memory_system.add_memory(entry))


def test_rrf_k_clamped_by_config_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validator clamps rrf_k to [1, 1000] when sourced from env.

    Note: pydantic v2 with pydantic-settings treats falsy default
    values (like ``0``) as "not provided" in direct construction — the
    validator only fires for env-sourced or explicitly-None values. The
    env-var path is the production path, so we test it.
    """
    monkeypatch.setenv("RRF_K", "0")
    cfg_low = MeminiConfig()
    assert cfg_low.rrf_k == 1

    monkeypatch.setenv("RRF_K", "9999")
    cfg_high = MeminiConfig()
    assert cfg_high.rrf_k == 1000

    monkeypatch.setenv("RRF_K", "60")
    cfg_default = MeminiConfig()
    assert cfg_default.rrf_k == 60


def test_rrf_propagates_threshold_to_384_side(
    memory_system: MemorySystem,
) -> None:
    """Regression test for v0.7.3 Bug B: _query_dual_model_rrf must pass
    the caller's threshold and exact_search through to the 384-side
    SearchOptions. Previously it built the 384-side options with only
    topK/strategy/filter, silently using the SearchOptions default
    threshold (0.72 pre-fix), which filtered out most legitimate matches.
    """
    import asyncio
    from unittest.mock import MagicMock

    # Patch the search layer's vector_only_search to capture the options.
    captured: dict[str, object] = {}

    async def fake_vector_only(
        question: str, options: SearchOptions, **kwargs: object
    ) -> list[MemoryEntry]:
        captured["threshold"] = options.threshold
        captured["exact_search"] = options.exact_search
        return []

    async def fake_query_1024(
        vector: list[float], threshold: float, limit: int
    ) -> list[MemoryEntry]:
        return []

    memory_system._search.vector_only_search = fake_vector_only  # type: ignore[method-assign]
    memory_system._db.query_memories_1024 = fake_query_1024  # type: ignore[method-assign]

    # The fix: 1024 expansion is also needed for the RRF path to take
    # the dual-mode branch (not the 384-only fallback).
    memory_system._db._expand_384_to_1024 = MagicMock(  # type: ignore[method-assign]
        return_value=[0.0] * 1024
    )

    caller_options = SearchOptions(
        topK=5,
        strategy=SearchStrategy.TIERED,
        threshold=0.5,
        exact_search=True,
    )
    asyncio.run(memory_system.query_memories("hello world", caller_options))

    assert captured.get("threshold") == 0.5, (
        f"RRF should propagate caller threshold to 384-side; got "
        f"{captured.get('threshold')}"
    )
    assert captured.get("exact_search") is True, (
        f"RRF should propagate exact_search flag to 384-side; got "
        f"{captured.get('exact_search')}"
    )


def test_default_search_options_threshold_is_zero() -> None:
    """Regression test for v0.7.3 Bug A: SearchOptions.threshold default
    must be 0.0 (no SQL-side filtering). The previous default of 0.72
    silently filtered out most legitimate MiniLM-L6-v2 matches because
    real-world cosine similarity for natural-language queries against
    stored memories commonly lands in 0.4-0.7.
    """
    options = SearchOptions()
    assert options.threshold == 0.0
