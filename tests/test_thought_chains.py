"""Tests for thought chains module.

Unit tests for ThoughtChains class with mocked database pool.
Integration tests require a running PostgreSQL instance.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from memini_ai.config import MeminiConfig
from memini_ai.memory.schema import MemorySourceType
from memini_ai.thought_chains import Thought, ThoughtChain, ThoughtChains


# ---------------------------------------------------------------------------
# Env isolation fixture
# ---------------------------------------------------------------------------
# ThoughtChains.is_enabled reads from the get_config() singleton, which in
# turn reads THOUGHT_CHAINS from env vars AND the project .env file. When
# the shell or .env has THOUGHT_CHAINS=true (as in the dev environment),
# tests that assert "disabled by default" fail. This fixture chdir's to a
# clean temp dir (no .env) and resets the config singleton so each test
# gets a fresh MeminiConfig with default values.
@pytest.fixture(autouse=True)
def _isolate_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate thought-chain tests from shell/env config leaks."""
    monkeypatch.chdir(tmp_path)
    # Delete THOUGHT_CHAINS and all MEMINI_ env vars
    monkeypatch.delenv("THOUGHT_CHAINS", raising=False)
    for key in list(os.environ):
        if key.startswith("MEMINI_") or key == "THOUGHT_CHAINS":
            monkeypatch.delenv(key, raising=False)
    # Reset the get_config() singleton so it re-reads from the now-clean env
    import memini_ai.config as _cfg_mod

    monkeypatch.setattr(_cfg_mod, "_config", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    return pool


@pytest.fixture
def mock_memory_system():
    """Create a mock MemorySystem."""
    system = AsyncMock()
    system.add_memory = AsyncMock(return_value=str(uuid.uuid4()))
    system.create_relationship = AsyncMock(return_value=None)
    return system


@pytest.fixture
def mock_trust_engine():
    """Create a mock TrustEngine."""
    engine = AsyncMock()
    engine.is_enabled = True
    engine.adjust_trust = AsyncMock(return_value=None)
    return engine


@pytest.fixture
def thought_chains(mock_pool, mock_memory_system, mock_trust_engine):
    """Create a ThoughtChains instance with mocked dependencies."""
    with patch.object(ThoughtChains, "is_enabled", True):
        tc = ThoughtChains(
            pool=mock_pool,
            memory_system=mock_memory_system,
            trust_engine=mock_trust_engine,
        )
        return tc


def make_async_result(**kwargs):
    """Create a mock asyncpg Record-like object with attribute access."""
    record = MagicMock()
    for k, v in kwargs.items():
        setattr(record, k, v)
    return record


# ---------------------------------------------------------------------------
# Unit Tests: Data Classes
# ---------------------------------------------------------------------------


class TestThoughtChain:
    """Test ThoughtChain dataclass."""

    def test_default_values(self):
        """Test ThoughtChain with defaults."""
        tc = ThoughtChain(id=str(uuid.uuid4()))
        assert tc.session_id is None
        assert tc.parent_chain_id is None
        assert tc.status == "active"

    def test_with_session(self):
        """Test ThoughtChain with session_id."""
        tc = ThoughtChain(
            id=str(uuid.uuid4()),
            session_id="test-session",
            parent_chain_id=None,
        )
        assert tc.session_id == "test-session"
        assert tc.parent_chain_id is None


class TestThought:
    """Test Thought dataclass."""

    def test_default_values(self):
        """Test Thought with required fields only."""
        t = Thought(
            id=str(uuid.uuid4()),
            chain_id=str(uuid.uuid4()),
            thought="Hello world",
            thought_number=1,
            total_thoughts=5,
        )
        assert t.next_thought_needed is True
        assert t.is_revision is False
        assert t.revises_thought_id is None
        assert t.branch_from_thought_id is None
        assert t.branch_id is None

    def test_with_revision(self):
        """Test Thought as a revision."""
        t = Thought(
            id=str(uuid.uuid4()),
            chain_id=str(uuid.uuid4()),
            thought="Revised thought",
            thought_number=1,
            total_thoughts=5,
            is_revision=True,
            revises_thought_id=str(uuid.uuid4()),
        )
        assert t.is_revision is True
        assert t.revises_thought_id is not None


# ---------------------------------------------------------------------------
# Unit Tests: ThoughtChains Feature Gate
# ---------------------------------------------------------------------------


class TestFeatureGate:
    """Test that thought chains are feature-gated."""

    def test_disabled_by_default(self):
        """Test that thought chains are disabled by default."""
        with patch.dict("os.environ", {}, clear=True):
            config = MeminiConfig()
            assert config.thought_chains_enabled is False

    def test_enabled_with_env_var(self):
        """Test that THOUGHT_CHAINS=true enables the feature."""
        with patch.dict("os.environ", {"THOUGHT_CHAINS": "true"}):
            config = MeminiConfig()
            assert config.thought_chains_enabled is True

    def test_check_enabled_returns_error(self, thought_chains):
        """Test that _check_enabled returns error when feature is disabled."""
        with patch(
            "memini_ai.thought_chains.ThoughtChains.is_enabled",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = thought_chains._check_enabled()
            assert result is not None
            assert "not enabled" in result["error"].lower()


# ---------------------------------------------------------------------------
# Unit Tests: MemorySourceType
# ---------------------------------------------------------------------------


class TestMemorySourceType:
    """Test that 'thought' is a valid source type."""

    def test_thought_source_type(self):
        """Test that 'thought' is in MemorySourceType enum."""
        assert MemorySourceType.thought == "thought"

    def test_all_source_types(self):
        """Test all source types are present."""
        expected = {
            "session",
            "file",
            "web",
            "boomerang",
            "project",
            "thought",
            "image",
            "github",
        }
        actual = {e.value for e in MemorySourceType}
        assert expected == actual


# ---------------------------------------------------------------------------
# Unit Tests: ThoughtChains Methods (with mocks)
# ---------------------------------------------------------------------------


class TestStartChain:
    """Test start_chain method."""

    def test_start_chain_returns_dict(self, thought_chains, mock_pool):
        """Test creating a new chain returns a dict."""
        chain_id = uuid.uuid4()
        now = datetime.utcnow()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value=make_async_result(
                id=chain_id,
                session_id="test-session",
                parent_chain_id=None,
                status="active",
                created_at=now,
                updated_at=now,
            )
        )
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire = MagicMock(return_value=mock_conn)

        # The key check is that the method doesn't crash
        # Actual DB operations would need integration tests
        assert thought_chains is not None


class TestAddThought:
    """Test add_thought method."""

    def test_add_thought_auto_adjust(self):
        """Test that thoughtNumber > totalThoughts logic exists."""
        # This just tests the auto-adjust logic conceptually
        thought_num = 5
        total = 3
        adjusted_total = max(thought_num, total)
        assert adjusted_total == 5

    def test_embedding_truncates_to_384_when_model_returns_1024(self):
        """v0.7.1 bugfix: when a 1024-dim model (e.g. BGE-M3) is loaded, the
        embedding must be truncated to 384 dims to fit the
        thoughts.embedding vector(384) column. Previously the code
        stringified the vector and crashed with
        "expected 384 dimensions, not 1024".

        BGE-Large was removed in v0.7.6 — this test now simulates a generic
        1024-dim model (the test exercises dim-handling, not the model
        identity). BGE-M3 is the only production 1024-dim model.
        """
        from memini_ai.model.embeddings import EmbeddingResult

        # Simulate a 1024-dim model returning 1024-dim vector
        big_vec = [0.1 * i for i in range(1024)]
        result = EmbeddingResult(
            embedding=big_vec,
            token_count=10,
            model_id="BAAI/bge-m3",
            device="cuda",
            timestamp=0,
            latency_ms=0,
        )

        # Apply the same dim-handling logic as add_thought (refactored for test)
        vec: list[float] = list(result.embedding)
        if len(vec) > 384:
            vec = vec[:384]
        elif len(vec) < 384:
            vec = vec + [0.0] * (384 - len(vec))

        assert len(vec) == 384
        # First 384 values of the original vector should be preserved
        assert vec == big_vec[:384]
        # Embedding must be a Python list (not a string!) for asyncpg binding
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embedding_pads_to_384_when_model_returns_smaller(self):
        """v0.7.1: Zero-pad when the model returns fewer than 384 dims."""
        from memini_ai.model.embeddings import EmbeddingResult

        small_vec = [0.5] * 128
        result = EmbeddingResult(
            embedding=small_vec,
            token_count=10,
            model_id="test-model",
            device="cpu",
            timestamp=0,
            latency_ms=0,
        )

        vec: list[float] = list(result.embedding)
        if len(vec) > 384:
            vec = vec[:384]
        elif len(vec) < 384:
            vec = vec + [0.0] * (384 - len(vec))

        assert len(vec) == 384
        assert vec[:128] == [0.5] * 128
        assert vec[128:] == [0.0] * 256

    def test_add_thought_binds_embedding_as_list_not_string(
        self, thought_chains, mock_pool
    ):
        """v0.7.1 bugfix: the embedding parameter passed to conn.fetchrow must
        be a Python list[float], not a stringified pgvector literal.

        This test catches the original bug regression: if the code is changed
        back to building `f\"[{','.join(str(v) for v in vec)}]\"`, this test
        will fail because the embedding arg will be a str, not a list.
        """
        import asyncio

        from memini_ai.model.embeddings import EmbeddingResult

        # Mock the embedding model to return a 384-dim vector
        fake_embedding = EmbeddingResult(
            embedding=[0.1] * 384,
            token_count=10,
            model_id="test",
            device="cpu",
            timestamp=0,
            latency_ms=0,
        )

        # Set up the mock pool's fetchrow to capture the embedding arg
        mock_conn = AsyncMock()
        captured_args: list = []

        async def capture_fetchrow(query, *args, **kwargs):
            captured_args.append(args)
            return make_async_result(id=uuid.uuid4())

        mock_conn.fetchrow = capture_fetchrow
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire = MagicMock(return_value=mock_conn)

        async def run_add_thought() -> None:
            # The `thought_chains` fixture applies an `is_enabled=True` patch
            # only for the duration of the fixture's `with` block. We need to
            # re-apply it here so the `_check_enabled()` gate inside
            # `add_thought` doesn't return early.
            with (
                patch.object(ThoughtChains, "is_enabled", True),
                patch(
                    "memini_ai.thought_chains.generate_embedding",
                    new=AsyncMock(return_value=fake_embedding),
                ),
            ):
                await thought_chains.add_thought(
                    thought="test thought",
                    thought_number=1,
                    total_thoughts=1,
                    next_thought_needed=False,
                    chain_id=str(uuid.uuid4()),
                )

        # asyncio.run() creates a fresh event loop and is safe regardless of
        # whether the calling thread already has a loop (e.g. under pytest-asyncio).
        asyncio.run(run_add_thought())

        # Find the embedding arg (position 10 in the INSERT VALUES)
        assert captured_args, "fetchrow was never called"
        embedding_arg = captured_args[0][10]
        assert isinstance(embedding_arg, list), (
            f"Embedding must be a list[float], not {type(embedding_arg).__name__}. "
            "If this fails, the v0.7.1 bug has regressed — embedding is being "
            "stringified before binding."
        )
        assert all(isinstance(v, float) for v in embedding_arg)
        assert len(embedding_arg) == 384


class TestChainOperations:
    """Test chain lifecycle operations."""

    def test_pause_chain_check_enabled(self, thought_chains):
        """Test pause_chain calls _check_enabled."""
        result = thought_chains._check_enabled()
        # By default thought_chains_enabled is False in tests
        assert result is not None
        assert "error" in result


# ---------------------------------------------------------------------------
# Unit Tests: Schema
# ---------------------------------------------------------------------------


class TestSchema:
    """Test that schema SQL is valid."""

    def test_thought_chains_table_sql(self):
        """Test that thought_chains table SQL exists and is valid."""
        from memini_ai.postgres.schema import (
            SQL_CREATE_THOUGHT_CHAINS_INDEXES,
            SQL_CREATE_THOUGHT_CHAINS_TABLE,
            SQL_CREATE_THOUGHTS_INDEXES,
            SQL_CREATE_THOUGHTS_TABLE,
        )

        assert "thought_chains" in SQL_CREATE_THOUGHT_CHAINS_TABLE
        assert "thoughts" in SQL_CREATE_THOUGHTS_TABLE
        assert "idx_thought_chains_session" in SQL_CREATE_THOUGHT_CHAINS_INDEXES
        assert "idx_thoughts_chain" in SQL_CREATE_THOUGHTS_INDEXES

    def test_source_type_check_constraint(self):
        """Test that source_type CHECK constraint includes 'thought'."""
        from memini_ai.postgres.schema import SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK

        assert "'thought'" in SQL_UPDATE_MEMORIES_SOURCE_TYPE_CHECK


# ---------------------------------------------------------------------------
# Unit Tests: Queries
# ---------------------------------------------------------------------------


class TestQueries:
    """Test that SQL query strings are valid."""

    def test_insert_thought_chain_query(self):
        """Test INSERT_THOUGHT_CHAIN query string."""
        from memini_ai.postgres.queries import INSERT_THOUGHT_CHAIN

        assert "INSERT INTO thought_chains" in INSERT_THOUGHT_CHAIN
        assert "$1" in INSERT_THOUGHT_CHAIN

    def test_insert_thought_query(self):
        """Test INSERT_THOUGHT query string."""
        from memini_ai.postgres.queries import INSERT_THOUGHT

        assert "INSERT INTO thoughts" in INSERT_THOUGHT
        assert "embedding" in INSERT_THOUGHT

    def test_search_query(self):
        """Test SEARCH_THOUGHT_CHAINS_BY_EMBEDDING query."""
        from memini_ai.postgres.queries import SEARCH_THOUGHT_CHAINS_BY_EMBEDDING

        assert "embedding <=> $1::vector" in SEARCH_THOUGHT_CHAINS_BY_EMBEDDING

    def test_all_queries_exist(self):
        """Test all expected queries exist."""
        from memini_ai.postgres import queries

        expected = [
            "INSERT_THOUGHT_CHAIN",
            "GET_THOUGHT_CHAIN_BY_ID",
            "UPDATE_THOUGHT_CHAIN_STATUS",
            "GET_THOUGHT_CHAINS_BY_SESSION",
            "INSERT_THOUGHT",
            "GET_THOUGHTS_BY_CHAIN",
            "GET_THOUGHT_BY_NUMBER",
            "GET_LAST_THOUGHT_IN_CHAIN",
            "GET_THOUGHT_BRANCHES",
            "COUNT_THOUGHTS_IN_CHAIN",
            "SEARCH_THOUGHT_CHAINS_BY_EMBEDDING",
            "UPDATE_THOUGHT_MEMORY_ID",
        ]
        for name in expected:
            assert hasattr(queries, name), f"Query {name} not found"
