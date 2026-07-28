"""Tests for the FastMCP server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.server import MCPServer


@pytest.fixture
def mcp_server() -> MCPServer:
    """Create an MCPServer instance for testing."""
    return MCPServer()


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem."""
    mock = MagicMock()
    mock.is_ready = True
    mock.is_initialized = True
    mock.query_memories = AsyncMock(return_value=[])
    mock.add_memory = AsyncMock(return_value="test-memory-id-123")
    # Post-write read-back (v0.7.3): returns a truthy entry by default.
    mock.get_memory = AsyncMock(return_value=MagicMock(id="test-memory-id-123"))
    # _db used by get_status row-count probes (v0.7.3).
    mock._db = MagicMock()
    mock._db.count_memories = AsyncMock(return_value=42)
    mock._db.count_thoughts = AsyncMock(return_value=7)
    return mock


@pytest.fixture
def mock_indexer() -> MagicMock:
    """Create a mock ProjectIndexer."""
    mock = MagicMock()
    mock.is_running = False
    mock.search = AsyncMock(return_value=[])
    mock.get_file_contents = AsyncMock(return_value=None)
    mock.get_stats = MagicMock(
        return_value=MagicMock(
            files_indexed=0,
            chunks_created=0,
            bytes_processed=0,
            errors=0,
        )
    )
    return mock


class TestQueryMemories:
    """Tests for the query_memories tool."""

    @pytest.mark.asyncio
    async def test_query_memories_returns_empty_when_no_results(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Test query_memories returns empty results when no memories exist."""
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.query_memories(
            query="test query", limit=10, strategy="tiered"
        )

        assert result["count"] == 0
        assert result["memories"] == []
        assert result["strategy_used"] == "TIERED"

    @pytest.mark.asyncio
    async def test_query_memories_with_results(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Test query_memories returns results when memories exist."""
        from memini_ai.memory.schema import MemoryEntry, MemorySourceType

        # Create mock memory entries
        mock_entries = [
            MemoryEntry(
                id="mem-1",
                text="Test memory 1",
                sourceType=MemorySourceType.session,
            ),
            MemoryEntry(
                id="mem-2",
                text="Test memory 2",
                sourceType=MemorySourceType.file,
            ),
        ]
        mock_memory_system.query_memories = AsyncMock(return_value=mock_entries)
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.query_memories(
            query="test query", limit=10, strategy="tiered"
        )

        assert result["count"] == 2
        assert len(result["memories"]) == 2
        assert result["strategy_used"] == "TIERED"


class TestAddMemory:
    """Tests for the add_memory tool."""

    @pytest.mark.asyncio
    async def test_add_memory_success(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Test add_memory successfully adds a memory."""
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.add_memory(
            content="This is a test memory",
            sourceType="session",
            sourcePath="/test/path",
            metadata={"key": "value"},
        )

        assert result["success"] is True
        assert result["id"] == "test-memory-id-123"
        assert "added successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_add_memory_duplicate(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Test add_memory handles duplicate content."""
        mock_memory_system.add_memory = AsyncMock(
            side_effect=ValueError("Memory with this content already exists")
        )
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.add_memory(
            content="Duplicate content",
            sourceType="session",
        )

        assert result["success"] is False
        assert result["id"] == ""
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_add_memory_post_write_readback_failure(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Regression test for v0.7.3: when the post-write read-back returns
        None (write succeeded but read path can't see the row), the handler
        must surface a non-success response instead of claiming success.
        """
        # add_memory returns an id, but get_memory (read-back) returns None.
        mock_memory_system.add_memory = AsyncMock(return_value="ghost-id-xyz")
        mock_memory_system.get_memory = AsyncMock(return_value=None)
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.add_memory(
            content="vanishing memory",
            sourceType="session",
        )

        assert result["success"] is False
        assert result["id"] == "ghost-id-xyz"
        assert result["error"] == "post_write_readback_failed"
        assert "read-back" in result["message"].lower()


class TestSearchProject:
    """Tests for the search_project tool."""

    @pytest.mark.asyncio
    async def test_search_project_returns_empty(
        self, mcp_server: MCPServer, mock_indexer: MagicMock
    ) -> None:
        """Test search_project returns empty when no results."""
        mcp_server._indexer = mock_indexer

        result = await mcp_server.search_project(
            query="test", topK=20, fileTypes=None, paths=None
        )

        assert result["count"] == 0
        assert result["chunks"] == []


class TestIndexProject:
    """Tests for the index_project tool."""

    @pytest.mark.asyncio
    async def test_index_project_background_returns_job_id(
        self, mcp_server: MCPServer, mock_indexer: MagicMock
    ) -> None:
        """Test index_project in background mode returns job ID."""
        mcp_server._indexer = mock_indexer

        result = await mcp_server.index_project(path=".", force=False, background=True)

        assert result["success"] is True
        assert result["jobId"] is not None
        assert result["status"] == "running"
        assert "background" in result["message"].lower()


class TestGetFileContents:
    """Tests for the get_file_contents tool."""

    @pytest.mark.asyncio
    async def test_get_file_contents_not_found(
        self, mcp_server: MCPServer, mock_indexer: MagicMock
    ) -> None:
        """Test get_file_contents returns error when file not found."""
        mcp_server._indexer = mock_indexer

        result = await mcp_server.get_file_contents(
            filePath="/nonexistent/file.py", triggerIndex=False
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestGetStatus:
    """Tests for the get_status tool."""

    @pytest.mark.asyncio
    async def test_get_status_returns_component_status(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
        mock_indexer: MagicMock,
    ) -> None:
        """Test get_status returns status for all components."""
        mcp_server._memory_system = mock_memory_system
        mcp_server._indexer = mock_indexer

        result = await mcp_server.get_status()

        assert "memoryReady" in result
        assert "modelReady" in result
        assert "indexerReady" in result
        assert "initError" in result

    @pytest.mark.asyncio
    async def test_get_status_includes_row_counts(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """Regression test for v0.7.3: get_status must surface actual row
        counts so the agent can distinguish "table is empty" from
        "ready but no semantic match" (a 0 count with memoryReady=True
        is a contradiction the status must surface).
        """
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.get_status()

        assert "memoryCount" in result
        assert "thoughtsCount" in result
        assert isinstance(result["memoryCount"], int)
        assert isinstance(result["thoughtsCount"], int)
        assert result["memoryCount"] >= 0
        assert result["thoughtsCount"] >= 0
        # The mock fixture sets 42 / 7.
        assert result["memoryCount"] == 42
        assert result["thoughtsCount"] == 7

    @pytest.mark.asyncio
    async def test_get_status_count_failure_does_not_break(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """If count_memories raises, get_status must still return a valid
        dict (with memoryCount=0) — not crash the whole status call.
        """
        mock_memory_system._db.count_memories = AsyncMock(
            side_effect=RuntimeError("count failed")
        )
        mcp_server._memory_system = mock_memory_system

        result = await mcp_server.get_status()

        assert result["memoryReady"] is True
        assert result["memoryCount"] == 0


class TestHealthcheck:
    """Tests for the v0.7.3 healthcheck MCP tool."""

    @pytest.mark.asyncio
    async def test_healthcheck_pass(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """Write a marker, read it back, get a match → status=pass."""
        # Wire the mock so add_memory returns a known id and get_memory
        # returns a MagicMock whose .text attribute equals the marker.
        mcp_server._memory_system = mock_memory_system

        async def fake_add_memory(entry: object) -> str:
            return "hc-id-pass-001"

        async def fake_get_memory(memory_id: str) -> object:
            # Return a MagicMock with .text matching the marker the
            # healthcheck function wrote. We don't know the marker
            # ahead of time, so patch the add_memory call to record it.
            return None  # placeholder; overridden in body below

        mock_memory_system.add_memory = AsyncMock(side_effect=fake_add_memory)
        # Capture the marker passed to add_memory and have get_memory
        # echo it back.
        captured: dict[str, str] = {}

        async def capture_add(entry: object) -> str:
            # entry is a MemoryEntry with .text
            captured["marker"] = entry.text  # type: ignore[attr-defined]
            return "hc-id-pass-001"

        async def echo_get(memory_id: str) -> object:
            m = MagicMock()
            m.text = captured["marker"]
            return m

        mock_memory_system.add_memory = AsyncMock(side_effect=capture_add)
        mock_memory_system.get_memory = AsyncMock(side_effect=echo_get)

        result = await mcp_server.healthcheck()

        assert result["status"] == "pass"
        assert result["readbackMatch"] is True
        assert result["error"] is None
        assert result["memoryId"] == "hc-id-pass-001"
        assert result["writeLatencyMs"] is not None
        assert result["readLatencyMs"] is not None

    @pytest.mark.asyncio
    async def test_healthcheck_fail_on_readback_mismatch(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """If read-back returns None, status=fail with readback_mismatch."""
        mcp_server._memory_system = mock_memory_system
        mock_memory_system.add_memory = AsyncMock(return_value="hc-id-fail-001")
        mock_memory_system.get_memory = AsyncMock(return_value=None)

        result = await mcp_server.healthcheck()

        assert result["status"] == "fail"
        assert result["readbackMatch"] is False
        assert result["error"] == "readback_mismatch"
        assert result["memoryId"] == "hc-id-fail-001"


class TestGracefulDegradation:
    """Tests for graceful degradation when database is unavailable."""

    @pytest.mark.asyncio
    async def test_server_starts_without_database(self, mcp_server: MCPServer) -> None:
        """Test that server can be created even without database."""
        # Server creation should not raise
        assert mcp_server is not None

        # get_status should work even in degraded mode
        result = await mcp_server.get_status()

        # In degraded mode, memory should not be ready (no memory system initialized)
        assert result["memoryReady"] is False


# ---------------------------------------------------------------------------
# v0.7.7: get_status embedding observability fields
# ---------------------------------------------------------------------------


class TestGetStatusEmbeddingFields:
    """Tests for v0.7.7 get_status embedding observability fields."""

    @pytest.mark.asyncio
    async def test_get_status_includes_embedding_fields(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """get_status must include modelName, modelDimension,
        embeddingDimMismatch, embeddingDimExpected, embeddingDimActual.
        """
        mcp_server._memory_system = mock_memory_system

        # Mock ModelManager so the get_status code path can read the fields
        from memini_ai.model.manager import ModelManager

        mock_manager = MagicMock()
        mock_manager._model_id = "sentence-transformers/all-MiniLM-L6-v2"
        mock_manager._dimensions = 384
        mock_manager.has_dim_mismatch = False
        mock_manager.embedding_dim = 384
        mock_manager._model = MagicMock()
        mock_manager._model.get_embedding_dimension.return_value = 384
        mock_manager.get_dimensions.return_value = 384

        with patch.object(ModelManager, "get_instance", return_value=mock_manager):
            result = await mcp_server.get_status()

        assert "modelName" in result
        assert "modelDimension" in result
        assert "embeddingDimMismatch" in result
        assert "embeddingDimExpected" in result
        assert "embeddingDimActual" in result
        assert result["modelName"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert result["modelDimension"] == 384
        assert result["embeddingDimMismatch"] is False
        assert result["embeddingDimExpected"] == 384
        assert result["embeddingDimActual"] == 384

    @pytest.mark.asyncio
    async def test_get_status_reports_mismatch(
        self,
        mcp_server: MCPServer,
        mock_memory_system: MagicMock,
    ) -> None:
        """When has_dim_mismatch=True, get_status should surface it."""
        mcp_server._memory_system = mock_memory_system

        from memini_ai.model.manager import ModelManager

        mock_manager = MagicMock()
        mock_manager._model_id = "BAAI/bge-m3"
        mock_manager._dimensions = 1024
        mock_manager.has_dim_mismatch = True
        mock_manager.embedding_dim = 384
        mock_manager._model = MagicMock()
        mock_manager._model.get_embedding_dimension.return_value = 1024
        mock_manager.get_dimensions.return_value = 1024

        with patch.object(ModelManager, "get_instance", return_value=mock_manager):
            result = await mcp_server.get_status()

        assert result["embeddingDimMismatch"] is True
        assert result["embeddingDimExpected"] == 384
        assert result["embeddingDimActual"] == 1024
        assert result["modelName"] == "BAAI/bge-m3"
        assert result["modelDimension"] == 1024


# =============================================================================
# Phase 1 feature-activation: KG entity extraction hook (Layer A)
# =============================================================================


class TestKGAutoExtractHook:
    """Tests for the KG entity extraction hook in add_memory.

    Phase 1 feature-activation: when ``knowledge_graph_enabled`` is ON,
    ``add_memory`` must call ``extract_and_register_entities`` on the
    KG. When OFF, the hook must be silent. Any hook exception must
    never fail the add.
    """

    @pytest.mark.asyncio
    async def test_kg_hook_fires_when_enabled(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """When KG_ENABLED is True, extract_and_register_entities is called."""
        mcp_server._memory_system = mock_memory_system
        mock_kg = MagicMock()
        mock_kg.extract_and_register_entities = AsyncMock(return_value=[])
        mcp_server._knowledge_graph = mock_kg

        with patch(
            "memini_ai.server.get_config",
            return_value=MagicMock(
                knowledge_graph_enabled=True,
                rate_limit_per_minute=100,
                max_memory_content_size=102400,
                sanitize_content=False,
            ),
        ):
            result = await mcp_server.add_memory(
                content="Alice deployed PostgreSQL to the VPS.",
                sourceType="session",
            )

        assert result["success"] is True
        mock_kg.extract_and_register_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_kg_hook_silent_when_disabled(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """When KG_ENABLED is False, extract_and_register_entities is NOT called."""
        mcp_server._memory_system = mock_memory_system
        mock_kg = MagicMock()
        mock_kg.extract_and_register_entities = AsyncMock(return_value=[])
        mcp_server._knowledge_graph = mock_kg

        with patch(
            "memini_ai.server.get_config",
            return_value=MagicMock(
                knowledge_graph_enabled=False,
                rate_limit_per_minute=100,
                max_memory_content_size=102400,
                sanitize_content=False,
            ),
        ):
            result = await mcp_server.add_memory(
                content="Alice deployed PostgreSQL to the VPS.",
                sourceType="session",
            )

        assert result["success"] is True
        mock_kg.extract_and_register_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_kg_hook_exception_does_not_break_add(
        self, mcp_server: MCPServer, mock_memory_system: MagicMock
    ) -> None:
        """When extract_and_register_entities raises, add_memory still succeeds."""
        mcp_server._memory_system = mock_memory_system
        mock_kg = MagicMock()
        mock_kg.extract_and_register_entities = AsyncMock(
            side_effect=RuntimeError("kg DB connection lost")
        )
        mcp_server._knowledge_graph = mock_kg

        with patch(
            "memini_ai.server.get_config",
            return_value=MagicMock(
                knowledge_graph_enabled=True,
                rate_limit_per_minute=100,
                max_memory_content_size=102400,
                sanitize_content=False,
            ),
        ):
            result = await mcp_server.add_memory(
                content="Alice deployed PostgreSQL to the VPS.",
                sourceType="session",
            )

        assert result["success"] is True
        assert result["id"] == "test-memory-id-123"


# =============================================================================
# Phase 1 feature-activation: Multi-Peer auto-registration (Layer C)
# =============================================================================


class TestPeerAutoRegisterHook:
    """Tests for the multi-peer auto-registration startup hook.

    Phase 1 feature-activation: when ``multi_peer_enabled`` is ON,
    ``_init_memory_system`` must auto-register a default "owner" peer
    on startup if no peers exist. Idempotent: safe on every restart.
    """

    @pytest.mark.asyncio
    async def test_peer_auto_registers_when_enabled_and_empty(
        self, mcp_server: MCPServer
    ) -> None:
        """When multi_peer_enabled=True and no peers exist, register owner."""
        mock_system = MagicMock()
        mock_system.initialize = AsyncMock()
        mock_system.close = AsyncMock()
        mock_system._db = MagicMock()
        mock_system._db.count_memories = AsyncMock(return_value=0)
        mock_system._db._pool = None  # skip audit logger + thought chains

        mock_peer_mgr = MagicMock()
        mock_peer_mgr.list_peers = AsyncMock(return_value={"count": 0, "peers": []})
        mock_peer_mgr.add_peer = AsyncMock(
            return_value={"success": True, "peer_id": "owner"}
        )

        mock_config = MagicMock()
        mock_config.multi_peer_enabled = True
        mock_config.thought_chains_enabled = False

        with (
            patch("memini_ai.server.MemorySystem", return_value=mock_system),
            patch(
                "memini_ai.server.get_multi_peer_manager",
                return_value=mock_peer_mgr,
            ),
            patch("memini_ai.config.get_config", return_value=mock_config),
            patch(
                "memini_ai.model.manager.ModelManager.auto_detect_model",
                AsyncMock(return_value=False),
            ),
            patch("memini_ai.server.TrustEngine"),
            patch("memini_ai.server.MemoryGraph"),
            patch("memini_ai.server.KnowledgeGraph"),
            patch("memini_ai.server.MemoryExtractor"),
            patch("memini_ai.server.PrecompressExtractor"),
            patch("memini_ai.server.TieredLoader"),
            patch("memini_ai.server.UserModel"),
            patch("memini_ai.server.DecayEngine"),
            patch("memini_ai.server.ConsolidationEngine"),
        ):
            await mcp_server._init_memory_system()

        mock_peer_mgr.list_peers.assert_called_once()
        mock_peer_mgr.add_peer.assert_called_once()
        _call = mock_peer_mgr.add_peer.call_args
        assert _call.kwargs["peer_id"] == "owner"
        assert _call.kwargs["role"] == "owner"
        assert _call.kwargs["trust_level"] == 1.0

    @pytest.mark.asyncio
    async def test_peer_auto_register_idempotent_on_restart(
        self, mcp_server: MCPServer
    ) -> None:
        """When peers already exist, add_peer is NOT called (idempotent)."""
        mock_system = MagicMock()
        mock_system.initialize = AsyncMock()
        mock_system.close = AsyncMock()
        mock_system._db = MagicMock()
        mock_system._db.count_memories = AsyncMock(return_value=0)
        mock_system._db._pool = None

        mock_peer_mgr = MagicMock()
        # Simulate a restart: peers already present
        mock_peer_mgr.list_peers = AsyncMock(
            return_value={
                "count": 1,
                "peers": [{"peer_id": "owner", "name": "Default Owner"}],
            }
        )
        mock_peer_mgr.add_peer = AsyncMock()

        mock_config = MagicMock()
        mock_config.multi_peer_enabled = True
        mock_config.thought_chains_enabled = False

        with (
            patch("memini_ai.server.MemorySystem", return_value=mock_system),
            patch(
                "memini_ai.server.get_multi_peer_manager",
                return_value=mock_peer_mgr,
            ),
            patch("memini_ai.config.get_config", return_value=mock_config),
            patch(
                "memini_ai.model.manager.ModelManager.auto_detect_model",
                AsyncMock(return_value=False),
            ),
            patch("memini_ai.server.TrustEngine"),
            patch("memini_ai.server.MemoryGraph"),
            patch("memini_ai.server.KnowledgeGraph"),
            patch("memini_ai.server.MemoryExtractor"),
            patch("memini_ai.server.PrecompressExtractor"),
            patch("memini_ai.server.TieredLoader"),
            patch("memini_ai.server.UserModel"),
            patch("memini_ai.server.DecayEngine"),
            patch("memini_ai.server.ConsolidationEngine"),
        ):
            await mcp_server._init_memory_system()

        mock_peer_mgr.list_peers.assert_called_once()
        mock_peer_mgr.add_peer.assert_not_called()

    @pytest.mark.asyncio
    async def test_peer_auto_register_silent_when_disabled(
        self, mcp_server: MCPServer
    ) -> None:
        """When multi_peer_enabled=False, no peer registration happens."""
        mock_system = MagicMock()
        mock_system.initialize = AsyncMock()
        mock_system.close = AsyncMock()
        mock_system._db = MagicMock()
        mock_system._db.count_memories = AsyncMock(return_value=0)
        mock_system._db._pool = None

        mock_peer_mgr = MagicMock()
        mock_peer_mgr.list_peers = AsyncMock()
        mock_peer_mgr.add_peer = AsyncMock()

        mock_config = MagicMock()
        mock_config.multi_peer_enabled = False
        mock_config.thought_chains_enabled = False

        with (
            patch("memini_ai.server.MemorySystem", return_value=mock_system),
            patch(
                "memini_ai.server.get_multi_peer_manager",
                return_value=mock_peer_mgr,
            ),
            patch("memini_ai.config.get_config", return_value=mock_config),
            patch(
                "memini_ai.model.manager.ModelManager.auto_detect_model",
                AsyncMock(return_value=False),
            ),
            patch("memini_ai.server.TrustEngine"),
            patch("memini_ai.server.MemoryGraph"),
            patch("memini_ai.server.KnowledgeGraph"),
            patch("memini_ai.server.MemoryExtractor"),
            patch("memini_ai.server.PrecompressExtractor"),
            patch("memini_ai.server.TieredLoader"),
            patch("memini_ai.server.UserModel"),
            patch("memini_ai.server.DecayEngine"),
            patch("memini_ai.server.ConsolidationEngine"),
        ):
            await mcp_server._init_memory_system()

        mock_peer_mgr.list_peers.assert_not_called()
        mock_peer_mgr.add_peer.assert_not_called()

    @pytest.mark.asyncio
    async def test_peer_auto_register_skipped_when_manager_disabled(
        self, mcp_server: MCPServer
    ) -> None:
        """When multi_peer_enabled=True but manager.is_enabled=False (e.g.
        ``user_modeling_enabled=False``), the hook must NOT attempt
        registration — ``list_peers`` returns an error dict (count=0)
        which would trigger a doomed ``add_peer`` call. Regression for
        the missing ``is_enabled`` gate on the peer hook condition.
        """
        mock_system = MagicMock()
        mock_system.initialize = AsyncMock()
        mock_system.close = AsyncMock()
        mock_system._db = MagicMock()
        mock_system._db.count_memories = AsyncMock(return_value=0)
        mock_system._db._pool = None

        mock_peer_mgr = MagicMock()
        # Manager present but NOT enabled (user_modeling_enabled=False).
        # ``is_enabled`` is a property on the real class; on a MagicMock we
        # set it as a plain attribute (truthiness drives the gate condition).
        mock_peer_mgr.is_enabled = False
        mock_peer_mgr.list_peers = AsyncMock()
        mock_peer_mgr.add_peer = AsyncMock()

        mock_config = MagicMock()
        mock_config.multi_peer_enabled = True
        mock_config.thought_chains_enabled = False

        with (
            patch("memini_ai.server.MemorySystem", return_value=mock_system),
            patch(
                "memini_ai.server.get_multi_peer_manager",
                return_value=mock_peer_mgr,
            ),
            patch("memini_ai.config.get_config", return_value=mock_config),
            patch(
                "memini_ai.model.manager.ModelManager.auto_detect_model",
                AsyncMock(return_value=False),
            ),
            patch("memini_ai.server.TrustEngine"),
            patch("memini_ai.server.MemoryGraph"),
            patch("memini_ai.server.KnowledgeGraph"),
            patch("memini_ai.server.MemoryExtractor"),
            patch("memini_ai.server.PrecompressExtractor"),
            patch("memini_ai.server.TieredLoader"),
            patch("memini_ai.server.UserModel"),
            patch("memini_ai.server.DecayEngine"),
            patch("memini_ai.server.ConsolidationEngine"),
        ):
            await mcp_server._init_memory_system()

        mock_peer_mgr.list_peers.assert_not_called()
        mock_peer_mgr.add_peer.assert_not_called()
