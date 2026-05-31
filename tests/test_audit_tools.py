"""Tests for MCP Audit Tools - Phase 2.3 Security Audit Logging.

Tests cover the server-level MCP audit tools:
- log_audit_event tool
- get_audit_log tool
- get_security_summary tool
- Tool registration and integration
- Audit hooks in add_memory and adjust_trust
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.audit.logger import AuditLogger
from memini_ai.server import MCPServer

# =============================================================================
# Helper fixtures
# =============================================================================


@pytest.fixture
def mock_audit_logger() -> MagicMock:
    """Create a mock AuditLogger for testing."""
    logger = MagicMock(spec=AuditLogger)
    logger.log = MagicMock()
    logger.get_events = AsyncMock(return_value=[])
    logger.get_summary = AsyncMock(
        return_value={
            "total_events": 0,
            "critical_count": 0,
            "events_per_agent": {},
            "events_per_type": {},
            "severity_counts": {},
        }
    )
    logger.start = AsyncMock()
    logger.stop = AsyncMock()
    return logger


# =============================================================================
# MCP Tool Registration Tests
# =============================================================================


class TestAuditToolRegistration:
    """Tests verifying audit tools are registered."""

    def test_audit_tools_registered(self) -> None:
        """Test that all 3 audit tools are registered in _setup_tools."""
        server = MCPServer()
        # Check that methods exist
        assert hasattr(server, "log_audit_event")
        assert hasattr(server, "get_audit_log")
        assert hasattr(server, "get_security_summary")
        assert callable(server.log_audit_event)
        assert callable(server.get_audit_log)
        assert callable(server.get_security_summary)

    def test_audit_logger_attribute_exists(self) -> None:
        """Test that _audit_logger attribute exists on MCPServer."""
        server = MCPServer()
        assert hasattr(server, "_audit_logger")
        assert server._audit_logger is None


# =============================================================================
# log_audit_event Tests
# =============================================================================


class TestLogAuditEvent:
    """Tests for the log_audit_event MCP tool."""

    @pytest.mark.asyncio
    async def test_log_audit_event_no_logger(self) -> None:
        """Test log_audit_event when audit logger is not initialized."""
        server = MCPServer()
        server._audit_logger = None
        server._memory_system = None

        with patch.object(
            server,
            "_init_memory_system",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_mem_system = MagicMock()
            mock_mem_system._db = MagicMock()
            mock_mem_system._db._pool = None
            mock_init.return_value = mock_mem_system

            result = await server.log_audit_event(
                event_type="memory_mutation",
                description="Test event",
            )

        # Should return error since no pool available
        assert result["success"] is False
        assert (
            "database pool" in result["error"].lower()
            or "audit" in result["error"].lower()
        )

    @pytest.mark.asyncio
    async def test_log_audit_event_with_logger(self) -> None:
        """Test log_audit_event with a working audit logger."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.log = MagicMock()
        server._audit_logger = mock_logger

        result = await server.log_audit_event(
            event_type="memory_mutation",
            severity="info",
            description="Memory added",
            memory_id=str(uuid.uuid4()),
        )

        assert result["success"] is True
        mock_logger.log.assert_called_once()
        call_kwargs = mock_logger.log.call_args[1]
        assert call_kwargs["event_type"] == "memory_mutation"
        assert call_kwargs["description"] == "Memory added"

    @pytest.mark.asyncio
    async def test_log_audit_event_all_fields(self) -> None:
        """Test log_audit_event with all fields populated."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.log = MagicMock()
        server._audit_logger = mock_logger

        mem_id = str(uuid.uuid4())
        result = await server.log_audit_event(
            event_type="trust_adjustment",
            severity="warning",
            session_id="sess-123",
            peer_id="peer-456",
            agent_name="boomerang-coder",
            tool_name="adjust_trust",
            memory_id=mem_id,
            description="Trust adjusted",
            details={"old_score": 0.5, "new_score": 0.6},
            state_before={"trust_score": 0.5},
            state_after={"trust_score": 0.6},
            ip_address="127.0.0.1",
        )

        assert result["success"] is True
        call_kwargs = mock_logger.log.call_args[1]
        assert call_kwargs["session_id"] == "sess-123"
        assert call_kwargs["peer_id"] == "peer-456"
        assert call_kwargs["agent_name"] == "boomerang-coder"
        assert call_kwargs["memory_id"] == mem_id

    @pytest.mark.asyncio
    async def test_log_audit_event_invalid_event_type(self) -> None:
        """Test that log_audit_event handles invalid event type gracefully."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.log = MagicMock()
        server._audit_logger = mock_logger

        # The AuditLogger.log() will silently drop invalid event types
        _ = await server.log_audit_event(
            event_type="invalid_type",
            description="Test",
        )

        # Even with invalid type, the tool returns success (fire-and-forget)
        # The logger's log method will handle validation internally
        # The call was still made to logger.log()
        assert mock_logger.log.called


# =============================================================================
# get_audit_log Tests
# =============================================================================


class TestGetAuditLog:
    """Tests for the get_audit_log MCP tool."""

    @pytest.mark.asyncio
    async def test_get_audit_log_no_logger(self) -> None:
        """Test get_audit_log when audit logger is not initialized."""
        server = MCPServer()
        server._audit_logger = None
        server._memory_system = None

        # Without pool, should return error
        with patch.object(
            server,
            "_init_memory_system",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_mem_system = MagicMock()
            mock_mem_system._db = MagicMock()
            mock_mem_system._db._pool = None
            mock_init.return_value = mock_mem_system

            result = await server.get_audit_log(
                event_type="memory_mutation",
            )

        assert result["count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_audit_log_with_logger(self) -> None:
        """Test get_audit_log with working logger."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_events = AsyncMock(
            return_value=[
                {"event_type": "memory_mutation", "severity": "info"},
            ]
        )
        server._audit_logger = mock_logger

        result = await server.get_audit_log(event_type="memory_mutation")

        assert result["count"] == 1
        assert result["events"] == [
            {"event_type": "memory_mutation", "severity": "info"}
        ]

    @pytest.mark.asyncio
    async def test_get_audit_log_with_date_filters(self) -> None:
        """Test get_audit_log with start_time and end_time filters."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_events = AsyncMock(return_value=[])
        server._audit_logger = mock_logger

        result = await server.get_audit_log(
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-31T23:59:59+00:00",
        )

        assert result["count"] == 0
        # Verify get_events was called with filters
        call_args = mock_logger.get_events.call_args
        assert "start_time" in call_args[0][0]
        assert "end_time" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_audit_log_with_all_filters(self) -> None:
        """Test get_audit_log with all filter parameters."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_events = AsyncMock(return_value=[])
        server._audit_logger = mock_logger

        result = await server.get_audit_log(
            event_type="trust_adjustment",
            severity="warning",
            agent_name="boomerang-coder",
            session_id="sess-123",
            limit=50,
        )

        assert result["count"] == 0
        call_filters = mock_logger.get_events.call_args[0][0]
        assert call_filters["event_type"] == "trust_adjustment"
        assert call_filters["severity"] == "warning"
        assert call_filters["agent_name"] == "boomerang-coder"
        assert call_filters["session_id"] == "sess-123"


# =============================================================================
# get_security_summary Tests
# =============================================================================


class TestGetSecuritySummary:
    """Tests for the get_security_summary MCP tool."""

    @pytest.mark.asyncio
    async def test_get_security_summary_no_logger(self) -> None:
        """Test get_security_summary when audit logger is not initialized."""
        server = MCPServer()
        server._audit_logger = None
        server._memory_system = None

        with patch.object(
            server,
            "_init_memory_system",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_mem_system = MagicMock()
            mock_mem_system._db = MagicMock()
            mock_mem_system._db._pool = None
            mock_init.return_value = mock_mem_system

            result = await server.get_security_summary(hours=24)

        assert result["total_events"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_security_summary_with_logger(self) -> None:
        """Test get_security_summary with working logger."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_summary = AsyncMock(
            return_value={
                "total_events": 100,
                "critical_count": 5,
                "events_per_agent": {"boomerang": 60, "coder": 40},
                "events_per_type": {"memory_mutation": 50, "tool_invocation": 50},
                "severity_counts": {"info": 90, "warning": 5, "critical": 5},
            }
        )
        server._audit_logger = mock_logger

        result = await server.get_security_summary(hours=12)

        assert result["success"] is True
        assert result["total_events"] == 100
        assert result["critical_count"] == 5
        assert result["severity_counts"]["critical"] == 5
        mock_logger.get_summary.assert_called_once_with(12)

    @pytest.mark.asyncio
    async def test_get_security_summary_default_hours(self) -> None:
        """Test get_security_summary with default 24 hours."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_summary = AsyncMock(
            return_value={
                "total_events": 0,
                "critical_count": 0,
                "events_per_agent": {},
                "events_per_type": {},
                "severity_counts": {},
            }
        )
        server._audit_logger = mock_logger

        result = await server.get_security_summary()

        assert result["success"] is True
        mock_logger.get_summary.assert_called_once_with(24)

    @pytest.mark.asyncio
    async def test_get_security_summary_on_error(self) -> None:
        """Test get_security_summary handles errors gracefully."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        mock_logger.get_summary = AsyncMock(side_effect=Exception("DB timeout"))
        server._audit_logger = mock_logger

        result = await server.get_security_summary(hours=24)

        assert result["success"] is False
        assert result["total_events"] == 0
        assert "error" in result


# =============================================================================
# Audit Hook Integration Tests
# =============================================================================


class TestAuditHooks:
    """Tests verifying audit hooks are called in existing tools."""

    @pytest.mark.asyncio
    async def test_add_memory_logs_audit_event(self) -> None:
        """Test that add_memory creates an audit log entry."""
        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        server._audit_logger = mock_logger

        # Mock the memory system initialization
        mock_mem_system = AsyncMock()
        mock_mem_system.initialize = AsyncMock()
        mock_mem_system.add_memory = AsyncMock(return_value="test-memory-id")
        mock_mem_system.is_ready = True
        mock_mem_system._db = MagicMock()

        # Mock other required components
        server._memory_system = mock_mem_system
        server._trust_engine = MagicMock()
        server._memory_graph = MagicMock()
        server._knowledge_graph = MagicMock()
        server._extractor = MagicMock()
        server._precompress = MagicMock()
        server._tiered_loader = MagicMock()
        server._user_model = MagicMock()
        server._decay_engine = MagicMock()
        server._consolidation_engine = MagicMock()

        with patch("memini_ai.server.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.rate_limit_per_minute = 1000
            mock_config_obj.max_memory_content_size = 100000
            mock_config_obj.sanitize_content = False
            mock_config.return_value = mock_config_obj

            result = await server.add_memory(content="Test memory content")

        # Verify audit logger was called
        assert result["success"] is True
        mock_logger.log.assert_called_once()
        # log() is called as log(event_type, severity=..., ...)
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == "memory_mutation"  # First positional arg = event_type
        assert call_args[1]["memory_id"] == "test-memory-id"
        assert call_args[1]["tool_name"] == "add_memory"

    @pytest.mark.asyncio
    async def test_adjust_trust_logs_audit_event(self) -> None:
        """Test that adjust_trust creates an audit log entry."""
        from memini_ai.memory.schema import TrustSignal
        from memini_ai.trust_engine import TrustAdjustment

        server = MCPServer()
        mock_logger = MagicMock(spec=AuditLogger)
        server._audit_logger = mock_logger

        # Mock trust engine
        mock_trust = MagicMock()
        mock_trust.is_enabled = True
        mock_trust.adjust_trust = AsyncMock(
            return_value=TrustAdjustment(
                memory_id="mem-123",
                old_score=0.5,
                new_score=0.55,
                signal=TrustSignal.AGENT_USED,
                action="increased",
            )
        )
        server._trust_engine = mock_trust

        # Mock memory system
        mock_mem_system = AsyncMock()
        mock_mem_system.initialize = AsyncMock()
        mock_mem_system.is_ready = True
        server._memory_system = mock_mem_system

        result = await server.adjust_trust(memory_id="mem-123", signal="agent_used")

        assert result["success"] is True
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == "trust_adjustment"  # First positional arg
        assert call_args[1]["memory_id"] == "mem-123"
        assert call_args[1]["tool_name"] == "adjust_trust"
