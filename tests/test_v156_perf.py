"""v1.5.6 performance regression tests.

Covers the three approved perf fixes:
- Fix A (projection pushdown): search SQL no longer selects raw embedding
  vectors; ``_row_to_memory`` tolerates absent embedding key; add_memory
  post-write read-back prefers the lightweight ``memory_exists`` probe.
- Fix B (single-embed + parallel RRF): auto-mode RRF embeds the question
  exactly ONCE and runs the 384/1024 fan-out arms concurrently.
- Fix C (configurable timeout): ``MEMINI_OPERATION_TIMEOUT_MS`` config
  field with clamp validator; ``server._op_timeout()`` resolves it.

All tests are hermetic (mocked DB/search layers, no live PostgreSQL).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import memini_ai.config as config_module
from memini_ai.postgres.database import PostgresDatabase
from memini_ai.postgres.queries import (
    MEMORY_EXISTS_BY_ID,
    SEARCH_MEMORIES_1024_JOINED,
    SEARCH_MEMORIES_VECTOR,
)
from memini_ai.server import OPERATION_TIMEOUT, MCPServer, _op_timeout


def _select_columns(sql: str) -> list[str]:
    """Extract the SELECT column list from a simple single-SELECT query."""
    select_block = sql.split("FROM", 1)[0]
    columns = select_block.split("SELECT", 1)[1]
    # Strip comments and whitespace, split on commas not inside parens.
    lines = [ln.split("--")[0] for ln in columns.splitlines()]
    joined = " ".join(lines)
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in joined:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p]


class TestSqlProjection:
    """Fix A: search queries must NOT ship raw embedding vectors."""

    def test_search_memories_vector_excludes_embedding(self) -> None:
        cols = _select_columns(SEARCH_MEMORIES_VECTOR)
        assert all(c != "embedding" for c in cols), (
            f"SEARCH_MEMORIES_VECTOR must not select raw 'embedding' "
            f"(serialization waste); got columns: {cols}"
        )

    def test_search_memories_1024_joined_excludes_embedding(self) -> None:
        cols = _select_columns(SEARCH_MEMORIES_1024_JOINED)
        assert all("embedding" not in c.lower() for c in cols), (
            f"SEARCH_MEMORIES_1024_JOINED must not select m1024.embedding "
            f"(RRF only needs ids); got columns: {cols}"
        )

    def test_memory_exists_by_id_selects_id_only(self) -> None:
        cols = _select_columns(MEMORY_EXISTS_BY_ID)
        assert cols == ["id"], f"MEMORY_EXISTS_BY_ID must SELECT id only; got: {cols}"


class TestRowToMemoryTolerant:
    """Fix A: slim rows without an embedding key must not raise."""

    def _make_row(self) -> dict[str, object]:
        return {
            "id": "11111111-1111-1111-1111-111111111111",
            "text": "hello world",
            "source_type": "session",
            "trust_score": 0.5,
            "retrieval_count": 0,
            "is_archived": False,
            "metadata": {},
            "content_hash": "abc",
            "supersedes_id": None,
            "structured_fields": None,
            "change_ratio": 1.0,
            "created_at_ms": 1787000000000,
            "embedding_model": None,
        }

    def test_missing_embedding_key_yields_none_vector(self) -> None:
        db = PostgresDatabase.__new__(PostgresDatabase)
        entry = db._row_to_memory(self._make_row())  # type: ignore[arg-type]
        assert entry.vector is None

    def test_present_embedding_key_still_parsed(self) -> None:
        row = self._make_row()
        row["embedding"] = "[0.1, 0.2, 0.3]"  # pgvector text repr
        db = PostgresDatabase.__new__(PostgresDatabase)
        entry = db._row_to_memory(row)  # type: ignore[arg-type]
        assert entry.vector == pytest.approx([0.1, 0.2, 0.3])


class TestVectorOnlySearchSkipsEmbed:
    """Fix B: precomputed query_vector bypasses generate_embedding."""

    @pytest.mark.asyncio
    async def test_query_vector_skips_embedding(self) -> None:
        from memini_ai.memory.schema import SearchOptions, SearchStrategy
        from memini_ai.memory.search import MemorySearch

        fake_db = MagicMock()
        fake_db.query_memories = AsyncMock(return_value=[])

        engine = MemorySearch(fake_db)

        async def fail_embed(_q: str) -> object:
            raise AssertionError(
                "generate_embedding must not be called when query_vector is provided"
            )

        import memini_ai.memory.search as search_mod

        original = search_mod.generate_embedding
        search_mod.generate_embedding = fail_embed  # type: ignore[assignment]
        try:
            results = await engine.vector_only_search(
                "question",
                SearchOptions(topK=5, strategy=SearchStrategy.VECTOR_ONLY),
                query_vector=[0.1] * 384,
            )
        finally:
            search_mod.generate_embedding = original  # type: ignore[assignment]

        assert results == []
        fake_db.query_memories.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_without_query_vector_embeds(self) -> None:
        from memini_ai.memory.schema import SearchOptions, SearchStrategy
        from memini_ai.memory.search import MemorySearch

        fake_db = MagicMock()
        fake_db.query_memories = AsyncMock(return_value=[])
        engine = MemorySearch(fake_db)

        embed_calls: list[str] = []

        async def fake_embed(q: str) -> object:
            embed_calls.append(q)
            return SimpleNamespace(embedding=[0.2] * 384)

        import memini_ai.memory.search as search_mod

        original = search_mod.generate_embedding
        search_mod.generate_embedding = fake_embed  # type: ignore[assignment]
        try:
            await engine.vector_only_search(
                "question",
                SearchOptions(topK=5, strategy=SearchStrategy.VECTOR_ONLY),
            )
        finally:
            search_mod.generate_embedding = original  # type: ignore[assignment]

        assert embed_calls == ["question"]


class TestSingleEmbedRRF:
    """Fix B: auto-mode RRF embeds once and fans out concurrently."""

    @pytest.fixture
    def rrf_system(self) -> tuple[object, MagicMock]:
        """MemorySystem wired for the RRF branch with mocked DB."""
        from memini_ai.memory.system import MemorySystem, MemorySystemConfig

        db = MagicMock()
        db.initialize = AsyncMock()
        db.query_memories = AsyncMock(return_value=[])  # 384 arm result
        db.query_memories_1024 = AsyncMock(return_value=[])  # 1024 arm result
        db._expand_384_to_1024 = MagicMock(return_value=[0.0] * 1024)

        cfg = MemorySystemConfig(embedding_mode="auto", rrf_k=60)
        system = MemorySystem(db=db, config=cfg)
        system._initialized = True
        return system, db

    @pytest.mark.asyncio
    async def test_rrf_embeds_question_exactly_once(
        self, rrf_system: tuple[object, MagicMock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system, _db = rrf_system  # type: ignore[misc]

        embed_calls: list[str] = []

        async def counting_embed(q: str) -> object:
            embed_calls.append(q)
            return SimpleNamespace(embedding=[0.3] * 384)

        import memini_ai.memory.search as search_mod
        import memini_ai.memory.system as system_mod

        monkeypatch.setattr(system_mod, "generate_embedding", counting_embed)
        monkeypatch.setattr(search_mod, "generate_embedding", counting_embed)

        from memini_ai.memory.schema import SearchOptions, SearchStrategy

        options = SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
        await asyncio.wait_for(  # type: ignore[arg-type]
            system.query_memories("single embed question", options),
            timeout=10.0,
        )

        assert len(embed_calls) == 1, (
            f"RRF must embed the question exactly once per query "
            f"(previously twice); got {len(embed_calls)} calls"
        )

    @pytest.mark.asyncio
    async def test_rrf_passes_precomputed_vector_to_384_arm(
        self, rrf_system: tuple[object, MagicMock], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """vector_only_search must receive query_vector so IT doesn't re-embed."""
        system, _db = rrf_system  # type: ignore[misc]

        captured_kwargs: dict[str, object] = {}

        async def spy_vector_only(
            question: str, options: object, **kwargs: object
        ) -> list[object]:
            captured_kwargs.update(kwargs)
            return []

        async def counting_embed(q: str) -> object:
            return SimpleNamespace(embedding=[0.4] * 384)

        import memini_ai.memory.system as system_mod

        monkeypatch.setattr(system_mod, "generate_embedding", counting_embed)
        monkeypatch.setattr(system._search, "vector_only_search", spy_vector_only)

        from memini_ai.memory.schema import SearchOptions, SearchStrategy

        options = SearchOptions(topK=5, strategy=SearchStrategy.TIERED)
        await asyncio.wait_for(  # type: ignore[arg-type]
            system.query_memories("vec passthrough", options),
            timeout=10.0,
        )

        assert captured_kwargs.get("query_vector") == [0.4] * 384, (
            "RRF 384 arm must receive the precomputed query vector"
        )


class TestOperationTimeoutConfig:
    """Fix C: MEMINI_OPERATION_TIMEOUT_MS clamps + _op_timeout() resolution."""

    @staticmethod
    def _cfg_from_env(monkeypatch: pytest.MonkeyPatch, value: str) -> object:
        """Build a MeminiConfig reading only the given env override.

        Config fields are addressed via their MEMINI_* aliases (pydantic
        settings convention used throughout this repo's tests).
        """
        monkeypatch.setenv("MEMINI_OPERATION_TIMEOUT_MS", value)
        return config_module.MeminiConfig(_env_file=None)  # type: ignore[call-arg]

    def test_default_is_30s(self) -> None:
        cfg = self._cfg_from_env(pytest.MonkeyPatch(), "30000")  # type: ignore[arg-type]
        assert cfg.operation_timeout_ms == 30000  # type: ignore[attr-defined]

    def test_clamped_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._cfg_from_env(monkeypatch, "50")
        assert cfg.operation_timeout_ms == 1000  # type: ignore[attr-defined]

    def test_clamped_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._cfg_from_env(monkeypatch, "999999999")
        assert cfg.operation_timeout_ms == 600000  # type: ignore[attr-defined]

    def test_env_var_respected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("MEMINI_OPERATION_TIMEOUT_MS", "90000")
        saved = config_module._config
        config_module._config = None
        try:
            cfg = config_module.MeminiConfig(_env_file=None)  # type: ignore[call-arg]
            assert cfg.operation_timeout_ms == 90000
        finally:
            config_module._config = saved

    def test_op_timeout_reads_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMINI_OPERATION_TIMEOUT_MS", "60000")
        saved = config_module._config
        config_module._config = None
        try:
            assert _op_timeout() == pytest.approx(60.0)
        finally:
            config_module._config = saved

    def test_op_timeout_fallback_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import memini_ai.server as server_mod

        def boom() -> object:
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(server_mod, "get_config", boom)
        assert _op_timeout() == OPERATION_TIMEOUT

    def test_all_server_timeouts_use_helper(self) -> None:
        """No call site may pin the constant directly anymore."""
        import inspect

        import memini_ai.server as server_mod

        src = inspect.getsource(server_mod)
        body = src.split("def _op_timeout", 1)[1]
        assert "timeout=OPERATION_TIMEOUT" not in body, (
            "All asyncio.wait_for sites must use timeout=_op_timeout()"
        )


class TestReadbackPrefersMemoryExists:
    """Fix A: add_memory read-back must use the lightweight probe."""

    @pytest.mark.asyncio
    async def test_readback_uses_memory_exists_not_get_memory(self) -> None:
        server = MCPServer()
        mock_system = MagicMock()
        mock_system.is_ready = True
        mock_system.is_initialized = True
        mock_system.add_memory = AsyncMock(return_value="readback-id-1")
        mock_system.content_exists = AsyncMock(return_value=False)
        mock_system.memory_exists = AsyncMock(return_value="readback-id-1")
        mock_system.get_memory = AsyncMock(
            side_effect=AssertionError(
                "get_memory must not be called when memory_exists is available"
            )
        )
        server._memory_system = mock_system  # type: ignore[assignment]

        result = await asyncio.wait_for(
            server.add_memory("perf probe memory"), timeout=15.0
        )

        mock_system.memory_exists.assert_awaited_once_with("readback-id-1")
        mock_system.get_memory.assert_not_called()
        assert result["success"] is True
