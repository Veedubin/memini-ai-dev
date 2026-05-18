"""Tests for Pre-Compression Extraction feature (Phase 3.1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.precompress import PrecompressExtractor, PrecompressResult


class TestPrecompressResult:
    """Tests for PrecompressResult dataclass."""

    def test_create_precompress_result(self) -> None:
        """Should create PrecompressResult with all fields."""
        result = PrecompressResult(
            memories_created=["mem-1", "mem-2"],
            context_captured="test context",
            extraction_count=2,
        )
        assert result.memories_created == ["mem-1", "mem-2"]
        assert result.context_captured == "test context"
        assert result.extraction_count == 2

    def test_precompress_result_empty(self) -> None:
        """Should create empty PrecompressResult."""
        result = PrecompressResult(
            memories_created=[],
            context_captured="",
            extraction_count=0,
        )
        assert result.memories_created == []
        assert result.context_captured == ""
        assert result.extraction_count == 0


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem."""
    system = MagicMock()
    system.add_memory = AsyncMock(return_value="new-memory-id")
    return system


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Create a mock MemoryExtractor."""
    extractor = MagicMock()
    extractor.trigger_extraction = AsyncMock(return_value=["mem-1", "mem-2"])
    return extractor


@pytest.fixture
def mock_config_enabled() -> MagicMock:
    """Create a mock config with precompress enabled."""
    config = MagicMock()
    config.precompress_enabled = True
    return config


@pytest.fixture
def mock_config_disabled() -> MagicMock:
    """Create a mock config with precompress disabled."""
    config = MagicMock()
    config.precompress_enabled = False
    return config


class TestIsEnabled:
    """Tests for is_enabled property."""

    @pytest.mark.asyncio
    async def test_is_enabled_true(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled returns True when config enabled."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            assert extractor.is_enabled is True

    @pytest.mark.asyncio
    async def test_is_enabled_false(self, mock_config_disabled: MagicMock) -> None:
        """is_enabled returns False when config disabled."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            assert extractor.is_enabled is False

    @pytest.mark.asyncio
    async def test_is_enabled_cached(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled is cached after first call."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            first = extractor.is_enabled
            second = extractor.is_enabled
            assert first is second


class TestBufferManagement:
    """Tests for conversation buffer management."""

    @pytest.mark.asyncio
    async def test_add_turn_single(self, mock_config_enabled: MagicMock) -> None:
        """add_turn adds a single turn to buffer."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")

            assert len(extractor._context_buffer) == 1
            assert extractor._context_buffer[0]["role"] == "user"
            assert extractor._context_buffer[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_add_turn_multiple(self, mock_config_enabled: MagicMock) -> None:
        """add_turn adds multiple turns to buffer."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")
            extractor.add_turn("agent", "Hi there")
            extractor.add_turn("user", "How are you?")

            assert len(extractor._context_buffer) == 3

    @pytest.mark.asyncio
    async def test_add_turn_buffer_bounded_at_20(self, mock_config_enabled: MagicMock) -> None:
        """Buffer stays bounded at 20 turns."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()

            # Add 25 turns
            for i in range(25):
                extractor.add_turn("user", f"Turn {i}")

            # Buffer should be bounded to 20
            assert len(extractor._context_buffer) == 20
            # First turn should be removed
            assert extractor._context_buffer[0]["content"] == "Turn 5"  # Turns 0-4 removed
            # Last turn should be present
            assert extractor._context_buffer[-1]["content"] == "Turn 24"

    @pytest.mark.asyncio
    async def test_add_turn_exactly_20(self, mock_config_enabled: MagicMock) -> None:
        """Buffer works correctly at exactly 20 turns."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()

            for i in range(20):
                extractor.add_turn("user", f"Turn {i}")

            assert len(extractor._context_buffer) == 20

    @pytest.mark.asyncio
    async def test_clear_buffer(self, mock_config_enabled: MagicMock) -> None:
        """clear_buffer removes all turns."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")
            extractor.add_turn("agent", "Hi")

            extractor.clear_buffer()

            assert len(extractor._context_buffer) == 0

    @pytest.mark.asyncio
    async def test_clear_buffer_when_empty(self, mock_config_enabled: MagicMock) -> None:
        """clear_buffer handles empty buffer gracefully."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.clear_buffer()  # Should not raise

            assert len(extractor._context_buffer) == 0


class TestCaptureAndExtract:
    """Tests for capture_and_extract method."""

    @pytest.mark.asyncio
    async def test_capture_and_extract_disabled_returns_empty(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, capture_and_extract returns empty result."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            result = await extractor.capture_and_extract("any context")

            assert result.memories_created == []
            assert result.context_captured == ""
            assert result.extraction_count == 0

    @pytest.mark.asyncio
    async def test_capture_and_extract_with_extractor(
        self, mock_config_enabled: MagicMock, mock_extractor: MagicMock
    ) -> None:
        """When enabled with extractor, triggers extraction."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=mock_extractor)
            result = await extractor.capture_and_extract("test context")

            assert result.memories_created == ["mem-1", "mem-2"]
            assert result.context_captured == "test context"
            assert result.extraction_count == 2
            mock_extractor.trigger_extraction.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_and_extract_without_extractor(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """When enabled but no extractor, returns empty result."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=None)
            result = await extractor.capture_and_extract("test context")

            assert result.memories_created == []
            assert result.context_captured == ""
            assert result.extraction_count == 0

    @pytest.mark.asyncio
    async def test_capture_and_extract_truncates_context(
        self, mock_config_enabled: MagicMock, mock_extractor: MagicMock
    ) -> None:
        """Context is truncated to 500 chars for logging."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=mock_extractor)
            long_context = "x" * 1000
            result = await extractor.capture_and_extract(long_context)

            assert len(result.context_captured) == 500
            assert result.context_captured == "x" * 500

    @pytest.mark.asyncio
    async def test_capture_and_extract_stores_in_buffer(
        self, mock_config_enabled: MagicMock, mock_extractor: MagicMock
    ) -> None:
        """capture_and_extract stores context in buffer."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=mock_extractor)
            await extractor.capture_and_extract("test context")

            # Context should be stored with "system" role
            assert len(extractor._context_buffer) == 1
            assert extractor._context_buffer[0]["role"] == "system"
            assert extractor._context_buffer[0]["content"] == "test context"

    @pytest.mark.asyncio
    async def test_capture_and_extract_builds_conversation_from_buffer(
        self, mock_config_enabled: MagicMock, mock_extractor: MagicMock
    ) -> None:
        """capture_and_extract builds conversation string from buffer."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=mock_extractor)
            extractor.add_turn("user", "Hello")
            extractor.add_turn("agent", "Hi")

            await extractor.capture_and_extract("recent context")

            # Check that trigger_extraction was called with combined conversation
            call_args = mock_extractor.trigger_extraction.call_args[0][0]
            assert "user: Hello" in call_args
            assert "agent: Hi" in call_args
            assert "system: recent context" in call_args


class TestAddTurnTracking:
    """Tests for add_turn tracking."""

    @pytest.mark.asyncio
    async def test_add_turn_tracks_user_role(self, mock_config_enabled: MagicMock) -> None:
        """add_turn properly tracks user role."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")

            assert extractor._context_buffer[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_add_turn_tracks_agent_role(self, mock_config_enabled: MagicMock) -> None:
        """add_turn properly tracks agent role."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("agent", "I'm the agent")

            assert extractor._context_buffer[0]["role"] == "agent"

    @pytest.mark.asyncio
    async def test_add_turn_preserves_content(self, mock_config_enabled: MagicMock) -> None:
        """add_turn preserves full content."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            content = "This is a longer message with special chars: !@#$%"
            extractor.add_turn("user", content)

            assert extractor._context_buffer[0]["content"] == content


class TestDisabledNoOp:
    """Tests for disabled behavior (no-op)."""

    @pytest.mark.asyncio
    async def test_disabled_add_turn_still_works(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """add_turn still works when disabled (just doesn't extract)."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")  # Should not raise

            # Buffer still updated
            assert len(extractor._context_buffer) == 1

    @pytest.mark.asyncio
    async def test_disabled_clear_buffer_still_works(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """clear_buffer still works when disabled."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "Hello")
            extractor.clear_buffer()  # Should not raise

            assert len(extractor._context_buffer) == 0

    @pytest.mark.asyncio
    async def test_disabled_is_enabled_property(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """is_enabled returns False when disabled."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            assert extractor.is_enabled is False

    @pytest.mark.asyncio
    async def test_disabled_register_context_event_handler_does_not_raise(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """register_context_event_handler does not raise when disabled."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_disabled):
            extractor = PrecompressExtractor()
            callback = MagicMock()
            extractor.register_context_event_handler(callback)  # Should not raise


class TestRegisterContextEventHandler:
    """Tests for register_context_event_handler."""

    @pytest.mark.asyncio
    async def test_register_accepts_callback(self, mock_config_enabled: MagicMock) -> None:
        """register_context_event_handler accepts a callback."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            callback = MagicMock()
            extractor.register_context_event_handler(callback)  # Should not raise

    @pytest.mark.asyncio
    async def test_register_accepts_async_callback(self, mock_config_enabled: MagicMock) -> None:
        """register_context_event_handler accepts async callback."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            async_callback = AsyncMock()
            extractor.register_context_event_handler(async_callback)  # Should not raise


class TestClose:
    """Tests for HTTP client cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_client(self, mock_config_enabled: MagicMock) -> None:
        """close() properly closes HTTP client."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            extractor._http_client = mock_client

            await extractor.close()

            mock_client.aclose.assert_called_once()
            assert extractor._http_client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_config_enabled: MagicMock) -> None:
        """close() handles None client gracefully."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor._http_client = None

            # Should not raise
            await extractor.close()

            assert extractor._http_client is None


class TestEmptyContext:
    """Tests for empty context handling."""

    @pytest.mark.asyncio
    async def test_capture_and_extract_empty_string(
        self, mock_config_enabled: MagicMock, mock_extractor: MagicMock
    ) -> None:
        """capture_and_extract handles empty context string."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(extractor=mock_extractor)
            result = await extractor.capture_and_extract("")

            # Should still work with extractor
            assert result.extraction_count == 2

    @pytest.mark.asyncio
    async def test_add_turn_empty_content(self, mock_config_enabled: MagicMock) -> None:
        """add_turn handles empty content."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor()
            extractor.add_turn("user", "")  # Should not raise

            assert len(extractor._context_buffer) == 1
            assert extractor._context_buffer[0]["content"] == ""


class TestNoExtractorIntegration:
    """Tests when no extractor is provided."""

    @pytest.mark.asyncio
    async def test_no_extractor_returns_empty_result(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """When no extractor provided, capture_and_extract returns empty."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(memory_system=None, extractor=None)
            result = await extractor.capture_and_extract("test context")

            assert result.memories_created == []
            assert result.context_captured == ""
            assert result.extraction_count == 0

    @pytest.mark.asyncio
    async def test_with_memory_system_but_no_extractor(
        self, mock_config_enabled: MagicMock, mock_memory_system: MagicMock
    ) -> None:
        """With memory system but no extractor, still returns empty."""
        with patch("memini_ai.precompress.get_config", return_value=mock_config_enabled):
            extractor = PrecompressExtractor(memory_system=mock_memory_system, extractor=None)
            result = await extractor.capture_and_extract("test context")

            assert result.memories_created == []
            assert result.extraction_count == 0