"""Tests for the FastMCP server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

        result = await mcp_server.query_memories(query="test query", limit=10, strategy="tiered")

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

        result = await mcp_server.query_memories(query="test query", limit=10, strategy="tiered")

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


class TestSearchProject:
    """Tests for the search_project tool."""

    @pytest.mark.asyncio
    async def test_search_project_returns_empty(
        self, mcp_server: MCPServer, mock_indexer: MagicMock
    ) -> None:
        """Test search_project returns empty when no results."""
        mcp_server._indexer = mock_indexer

        result = await mcp_server.search_project(query="test", topK=20, fileTypes=None, paths=None)

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
        self, mcp_server: MCPServer, mock_memory_system: MagicMock, mock_indexer: MagicMock
    ) -> None:
        """Test get_status returns status for all components."""
        mcp_server._memory_system = mock_memory_system
        mcp_server._indexer = mock_indexer

        result = await mcp_server.get_status()

        assert "memoryReady" in result
        assert "modelReady" in result
        assert "indexerReady" in result
        assert "initError" in result


class TestGracefulDegradation:
    """Tests for graceful degradation when Qdrant is unavailable."""

    @pytest.mark.asyncio
    async def test_server_starts_without_qdrant(self, mcp_server: MCPServer) -> None:
        """Test that server can be created even without Qdrant."""
        # Server creation should not raise
        assert mcp_server is not None

        # get_status should work even in degraded mode
        result = await mcp_server.get_status()

        # In degraded mode, memory should not be ready (no memory system initialized)
        assert result["memoryReady"] is False
