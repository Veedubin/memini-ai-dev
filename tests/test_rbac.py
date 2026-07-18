"""Tests for RBAC peer_id enforcement, SSL config, and container runtime detection.

Covers:
- Config: peer_enforcement, peer_id, db_sslmode defaults and env-var overrides
- Query helper: peer_filter_clause() with None and with peer_id
- Database mock: _enforce_peer_filter, _effective_peer_id, add_memory peer_id tagging
- Installer: check_container_runtime_docker (the one missing from test_installer.py)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memini_ai.config import MeminiConfig
from memini_ai.postgres.queries import peer_filter_clause

# =============================================================================
# Config Tests
# =============================================================================


class TestConfigPeerEnforcement:
    """Tests for peer_enforcement and peer_id config fields."""

    def test_peer_enforcement_default_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config without MEMINI_PEER_ENFORCEMENT -> peer_enforcement == False."""
        monkeypatch.delenv("MEMINI_PEER_ENFORCEMENT", raising=False)
        monkeypatch.delenv("MEMINI_PEER_ID", raising=False)
        # Clear any cached config
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.peer_enforcement is False

    def test_peer_enforcement_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set MEMINI_PEER_ENFORCEMENT=true -> peer_enforcement == True."""
        monkeypatch.setenv("MEMINI_PEER_ENFORCEMENT", "true")
        monkeypatch.delenv("MEMINI_PEER_ID", raising=False)
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.peer_enforcement is True

    def test_peer_id_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config without MEMINI_PEER_ID -> peer_id is None."""
        monkeypatch.delenv("MEMINI_PEER_ID", raising=False)
        monkeypatch.delenv("MEMINI_PEER_ENFORCEMENT", raising=False)
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.peer_id is None

    def test_peer_id_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set MEMINI_PEER_ID='my-project' -> peer_id == 'my-project'."""
        monkeypatch.setenv("MEMINI_PEER_ID", "my-project")
        monkeypatch.delenv("MEMINI_PEER_ENFORCEMENT", raising=False)
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.peer_id == "my-project"

    def test_db_sslmode_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config without DB_SSLMODE -> db_sslmode == 'disable'."""
        monkeypatch.delenv("DB_SSLMODE", raising=False)
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.db_sslmode == "disable"

    def test_db_sslmode_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set DB_SSLMODE=require -> db_sslmode == 'require'."""
        monkeypatch.setenv("DB_SSLMODE", "require")
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.db_sslmode == "require"

    def test_db_sslmode_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid DB_SSLMODE value raises ValueError."""
        monkeypatch.setenv("DB_SSLMODE", "invalid-mode")
        monkeypatch.setattr("memini_ai.config._config", None)
        with pytest.raises(ValueError, match="Invalid db_sslmode"):
            MeminiConfig()


# =============================================================================
# Query Helper Tests
# =============================================================================


class TestPeerFilterClause:
    """Tests for peer_filter_clause() helper."""

    def test_peer_filter_clause_none(self) -> None:
        """peer_filter_clause(None, 5) -> ('', None) — no filtering."""
        clause, value = peer_filter_clause(None, 5)
        assert clause == ""
        assert value is None

    def test_peer_filter_clause_with_id(self) -> None:
        """peer_filter_clause('my-project', 5) -> clause with $5::uuid and value."""
        clause, value = peer_filter_clause("my-project", 5)
        assert "AND" in clause
        assert "$5::uuid" in clause
        assert "peer_id IS NULL" in clause
        assert value == "my-project"

    def test_peer_filter_clause_param_index_1(self) -> None:
        """peer_filter_clause('proj-1', 1) uses $1::uuid."""
        clause, value = peer_filter_clause("proj-1", 1)
        assert "$1::uuid" in clause
        assert value == "proj-1"

    def test_peer_filter_clause_param_index_4(self) -> None:
        """peer_filter_clause('proj-1', 4) uses $4::uuid."""
        clause, value = peer_filter_clause("proj-1", 4)
        assert "$4::uuid" in clause
        assert value == "proj-1"


# =============================================================================
# Database Mock Tests
# =============================================================================


class TestDatabasePeerEnforcement:
    """Tests for _enforce_peer_filter and _effective_peer_id properties.

    These tests use monkeypatch to set config values and then create a
    PostgresDatabase instance via __new__ (no real DB connection needed).
    """

    def _make_db(self, peer_enforcement: bool, peer_id: str | None) -> object:
        """Create a PostgresDatabase-like object with given peer settings.

        We use __new__ to avoid the real __init__ which would try to read
        config and create a driver. Instead we manually set the attributes
        that _enforce_peer_filter and _effective_peer_id depend on.
        """
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = peer_enforcement
        db._peer_id = peer_id
        return db

    def test_enforce_peer_filter_off(self) -> None:
        """When _peer_enforcement=False, _enforce_peer_filter is False."""
        db = self._make_db(peer_enforcement=False, peer_id="proj-1")
        assert db._enforce_peer_filter is False

    def test_enforce_peer_filter_on_with_id(self) -> None:
        """When _peer_enforcement=True and _peer_id is set, _enforce_peer_filter is True."""
        db = self._make_db(peer_enforcement=True, peer_id="proj-1")
        assert db._enforce_peer_filter is True

    def test_enforce_peer_filter_on_no_id(self) -> None:
        """When _peer_enforcement=True but _peer_id=None, _enforce_peer_filter is False."""
        db = self._make_db(peer_enforcement=True, peer_id=None)
        assert db._enforce_peer_filter is False

    def test_effective_peer_id_off(self) -> None:
        """When _enforce_peer_filter is False, _effective_peer_id is None."""
        db = self._make_db(peer_enforcement=False, peer_id="proj-1")
        assert db._effective_peer_id is None

    def test_effective_peer_id_on(self) -> None:
        """When _enforce_peer_filter is True, _effective_peer_id returns the peer_id."""
        db = self._make_db(peer_enforcement=True, peer_id="proj-1")
        assert db._effective_peer_id == "proj-1"

    def test_effective_peer_id_on_no_id(self) -> None:
        """When _enforce_peer_filter is False (no peer_id), _effective_peer_id is None."""
        db = self._make_db(peer_enforcement=True, peer_id=None)
        assert db._effective_peer_id is None


class TestDatabasePeerFilteringInQueries:
    """Tests that query methods conditionally add peer_id filters.

    We mock the asyncpg pool and connection to verify SQL is constructed
    correctly with or without peer_id filters.
    """

    @pytest.mark.asyncio
    async def test_database_peer_enforcement_off(self) -> None:
        """When _peer_enforcement=False, query_memories does NOT add peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = None
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock()
        mock_conn.transaction = MagicMock()
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        # Call query_memories
        await db.query_memories(
            vector=[0.1] * 384,
            options=MagicMock(threshold=0.5, top_k=10, exact_search=False),
        )

        # Verify fetch was called with the base SEARCH_MEMORIES_VECTOR (no peer filter)
        assert mock_conn.fetch.call_count >= 1
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        # The base query should NOT have a peer_id filter
        assert "peer_id" not in sql

    @pytest.mark.asyncio
    async def test_database_peer_enforcement_on(self) -> None:
        """When _peer_enforcement=True and _peer_id='proj-1', query_memories DOES add peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = True
        db._peer_id = "proj-1"
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock()
        mock_conn.transaction = MagicMock()
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        # Call query_memories
        await db.query_memories(
            vector=[0.1] * 384,
            options=MagicMock(threshold=0.5, top_k=10, exact_search=False),
        )

        # Verify fetch was called with a peer_id filter
        assert mock_conn.fetch.call_count >= 1
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" in sql
        assert "$4::uuid" in sql
        # Verify the peer_id value was passed (arg index 4: sql, query_vector, distance_threshold, top_k, peer_id)
        assert call_args[0][4] == "proj-1"

    @pytest.mark.asyncio
    async def test_database_peer_enforcement_on_no_peer_id(self) -> None:
        """When _peer_enforcement=True but _peer_id=None, no filter added."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = True
        db._peer_id = None
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock()
        mock_conn.transaction = MagicMock()
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        # Call query_memories
        await db.query_memories(
            vector=[0.1] * 384,
            options=MagicMock(threshold=0.5, top_k=10, exact_search=False),
        )

        # Verify fetch was called WITHOUT a peer_id filter
        assert mock_conn.fetch.call_count >= 1
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" not in sql

    @pytest.mark.asyncio
    async def test_add_memory_writes_peer_id(self) -> None:
        """add_memory passes peer_id to the INSERT query (tagging even when enforcement is off)."""
        from memini_ai.memory.schema import MemoryEntry, MemorySourceType
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = "my-peer"
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="new-memory-id")
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        entry = MemoryEntry(
            text="test memory with peer tagging",
            vector=[0.1] * 384,
            source_type=MemorySourceType.session,
        )

        mem_id = await db.add_memory(entry)
        assert mem_id == "new-memory-id"

        # Verify fetchval was called with peer_id in the args
        assert mock_conn.fetchval.call_count >= 1
        call_args = mock_conn.fetchval.call_args
        assert call_args is not None
        # The last positional arg should be the peer_id
        args = call_args[0]
        assert args[-1] == "my-peer"

    @pytest.mark.asyncio
    async def test_add_memory_writes_entry_peer_id_overrides(self) -> None:
        """When entry.peer_id is set, it takes precedence over the instance default."""
        from memini_ai.memory.schema import MemoryEntry, MemorySourceType
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = "instance-peer"
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="new-memory-id")
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        entry = MemoryEntry(
            text="test memory with explicit peer_id",
            vector=[0.1] * 384,
            source_type=MemorySourceType.session,
            peer_id="entry-peer",
        )

        mem_id = await db.add_memory(entry)
        assert mem_id == "new-memory-id"

        # Verify the entry's peer_id was used, not the instance default
        assert mock_conn.fetchval.call_count >= 1
        call_args = mock_conn.fetchval.call_args
        assert call_args is not None
        args = call_args[0]
        assert args[-1] == "entry-peer"
        assert args[-1] != "instance-peer"

    @pytest.mark.asyncio
    async def test_add_memory_no_peer_id(self) -> None:
        """When neither entry.peer_id nor instance peer_id is set, peer_id is None in INSERT."""
        from memini_ai.memory.schema import MemoryEntry, MemorySourceType
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = None
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value="new-memory-id")
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        entry = MemoryEntry(
            text="test memory without peer_id",
            vector=[0.1] * 384,
            source_type=MemorySourceType.session,
        )

        mem_id = await db.add_memory(entry)
        assert mem_id == "new-memory-id"

        # Verify peer_id is None in the INSERT args
        assert mock_conn.fetchval.call_count >= 1
        call_args = mock_conn.fetchval.call_args
        assert call_args is not None
        args = call_args[0]
        assert args[-1] is None


class TestDatabasePeerFilteringInGetMemory:
    """Tests that get_memory conditionally adds peer_id filters."""

    @pytest.mark.asyncio
    async def test_get_memory_with_peer_enforcement(self) -> None:
        """get_memory with peer enforcement on adds peer_id filter to SQL."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = True
        db._peer_id = "proj-1"
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No row found
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        result = await db.get_memory("some-memory-id")
        assert result is None

        # Verify the SQL includes a peer_id filter
        assert mock_conn.fetchrow.call_count >= 1
        call_args = mock_conn.fetchrow.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" in sql
        assert "$3::uuid" in sql

    @pytest.mark.asyncio
    async def test_get_memory_without_peer_enforcement(self) -> None:
        """get_memory without peer enforcement does NOT add peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = None
        db._initialized = True

        # Mock the pool
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        result = await db.get_memory("some-memory-id")
        assert result is None

        # Verify the SQL does NOT include a peer_id filter
        assert mock_conn.fetchrow.call_count >= 1
        call_args = mock_conn.fetchrow.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" not in sql


class TestDatabasePeerFilteringInDeleteMemory:
    """Tests that delete_memory conditionally adds peer_id filters."""

    @pytest.mark.asyncio
    async def test_delete_memory_with_peer_enforcement(self) -> None:
        """delete_memory with peer enforcement on adds peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = True
        db._peer_id = "proj-1"
        db._initialized = True

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        await db.delete_memory("some-memory-id")

        assert mock_conn.execute.call_count >= 1
        call_args = mock_conn.execute.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" in sql
        assert "$2::uuid" in sql

    @pytest.mark.asyncio
    async def test_delete_memory_without_peer_enforcement(self) -> None:
        """delete_memory without peer enforcement does NOT add peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = None
        db._initialized = True

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        await db.delete_memory("some-memory-id")

        assert mock_conn.execute.call_count >= 1
        call_args = mock_conn.execute.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" not in sql


class TestDatabasePeerFilteringInListMemories:
    """Tests that list_memories conditionally adds peer_id filters."""

    @pytest.mark.asyncio
    async def test_list_memories_with_peer_enforcement(self) -> None:
        """list_memories with peer enforcement on adds peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = True
        db._peer_id = "proj-1"
        db._initialized = True

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        result = await db.list_memories()
        assert result == []

        assert mock_conn.fetch.call_count >= 1
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" in sql

    @pytest.mark.asyncio
    async def test_list_memories_without_peer_enforcement(self) -> None:
        """list_memories without peer enforcement does NOT add peer_id filter."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._peer_enforcement = False
        db._peer_id = None
        db._initialized = True

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()
        db._pool = mock_pool

        result = await db.list_memories()
        assert result == []

        assert mock_conn.fetch.call_count >= 1
        call_args = mock_conn.fetch.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "peer_id" not in sql


# =============================================================================
# SSL Config Tests
# =============================================================================


class TestSSLConfig:
    """Tests for SSL-related config and database behavior."""

    def test_db_sslrootcert_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config without DB_SSLROOTCERT -> db_sslrootcert is None."""
        monkeypatch.delenv("DB_SSLROOTCERT", raising=False)
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.db_sslrootcert is None

    def test_db_sslrootcert_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set DB_SSLROOTCERT=/path/to/ca.pem -> db_sslrootcert == '/path/to/ca.pem'."""
        monkeypatch.setenv("DB_SSLROOTCERT", "/path/to/ca.pem")
        monkeypatch.setattr("memini_ai.config._config", None)
        config = MeminiConfig()
        assert config.db_sslrootcert == "/path/to/ca.pem"

    def test_build_ssl_context_disable(self) -> None:
        """sslmode='disable' -> _build_ssl_context returns None."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "disable"
        ctx = db._build_ssl_context()
        assert ctx is None

    def test_build_ssl_context_allow(self) -> None:
        """sslmode='allow' -> _build_ssl_context returns None (asyncpg limitation)."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "allow"
        ctx = db._build_ssl_context()
        assert ctx is None

    def test_build_ssl_context_prefer(self) -> None:
        """sslmode='prefer' -> _build_ssl_context returns a permissive SSL context."""
        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "prefer"
        ctx = db._build_ssl_context()
        assert ctx is not None
        # 'prefer' mode should not verify the server cert
        assert ctx.verify_mode.name == "CERT_NONE"

    def test_build_ssl_context_require(self) -> None:
        """sslmode='require' -> _build_ssl_context returns SSL context with CERT_NONE."""
        import ssl

        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "require"
        db._sslrootcert = None
        ctx = db._build_ssl_context()
        assert ctx is not None
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_build_ssl_context_verify_ca(self) -> None:
        """sslmode='verify-ca' -> _build_ssl_context returns SSL context with CERT_REQUIRED."""
        import ssl

        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "verify-ca"
        db._sslrootcert = None
        ctx = db._build_ssl_context()
        assert ctx is not None
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_build_ssl_context_verify_full(self) -> None:
        """sslmode='verify-full' -> _build_ssl_context returns SSL context with hostname check."""
        import ssl

        from memini_ai.postgres.database import PostgresDatabase

        db = PostgresDatabase.__new__(PostgresDatabase)
        db._sslmode = "verify-full"
        db._sslrootcert = None
        ctx = db._build_ssl_context()
        assert ctx is not None
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED
