"""Tests for Tiered Loading L0/L1/L2 feature (Phase 3.2)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.schema import SummaryTier, TieredSummary
from memini_ai.tiered_loader import (
    L0_SUMMARY_PROMPT,
    L1_SUMMARY_PROMPT,
    TieredLoader,
    TieredLoadingStats,
)


class TestTieredLoadingStats:
    """Tests for TieredLoadingStats dataclass."""

    def test_create_stats_defaults(self) -> None:
        """Should create TieredLoadingStats with default values."""
        stats = TieredLoadingStats()
        assert stats.l0_generations == 0
        assert stats.l1_generations == 0
        assert stats.l0_cache_hits == 0
        assert stats.l1_cache_hits == 0
        assert stats.last_l0_generated is None
        assert stats.last_l1_generated is None
        assert stats.l0_token_count == 0
        assert stats.l1_token_count == 0
        assert stats.errors == 0

    def test_create_stats_with_values(self) -> None:
        """Should create TieredLoadingStats with custom values."""
        now = datetime.utcnow()
        stats = TieredLoadingStats(
            l0_generations=5,
            l1_generations=3,
            l0_cache_hits=10,
            l1_cache_hits=8,
            last_l0_generated=now,
            last_l1_generated=now,
            l0_token_count=100,
            l1_token_count=2000,
            errors=1,
        )
        assert stats.l0_generations == 5
        assert stats.l1_generations == 3
        assert stats.l0_cache_hits == 10
        assert stats.l1_cache_hits == 8
        assert stats.last_l0_generated == now
        assert stats.last_l1_generated == now
        assert stats.l0_token_count == 100
        assert stats.l1_token_count == 2000
        assert stats.errors == 1

    def test_to_dict(self) -> None:
        """Should convert stats to dictionary."""
        now = datetime.utcnow()
        stats = TieredLoadingStats(
            l0_generations=2,
            l1_generations=1,
            last_l0_generated=now,
            last_l1_generated=now,
        )
        result = stats.to_dict()
        assert result["l0_generations"] == 2
        assert result["l1_generations"] == 1
        assert result["last_l0_generated"] == now.isoformat()
        assert result["last_l1_generated"] == now.isoformat()

    def test_to_dict_with_none_timestamps(self) -> None:
        """Should handle None timestamps in to_dict."""
        stats = TieredLoadingStats()
        result = stats.to_dict()
        assert result["last_l0_generated"] is None
        assert result["last_l1_generated"] is None


@pytest.fixture
def mock_config_enabled() -> MagicMock:
    """Create a mock config with tiered loading enabled."""
    config = MagicMock()
    config.tiered_loading_enabled = True
    config.tier0_cache_ttl = 3600  # 1 hour
    config.tier1_cache_ttl = 7200  # 2 hours
    config.tier0_max_tokens = 100
    config.tier1_max_tokens = 2000
    config.llm_url = "http://localhost:11434/api/generate"
    config.llm_model = "llama3.2"
    return config


@pytest.fixture
def mock_config_disabled() -> MagicMock:
    """Create a mock config with tiered loading disabled."""
    config = MagicMock()
    config.tiered_loading_enabled = False
    return config


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem with memories."""
    system = MagicMock()
    system.list_memories = AsyncMock(return_value=[])
    return system


@pytest.fixture
def mock_memories_high_trust() -> list[MagicMock]:
    """Create mock memories with high trust (>= 0.5 for L0)."""
    return [
        MagicMock(
            id="mem-1",
            text="Use async/await for all I/O operations",
            trust_score=0.9,
            is_archived=False,
        ),
        MagicMock(
            id="mem-2",
            text="Project uses Python 3.11+ type hints",
            trust_score=0.7,
            is_archived=False,
        ),
        MagicMock(
            id="mem-3",
            text="Use dataclasses for simple data structures",
            trust_score=0.5,
            is_archived=False,
        ),
    ]


@pytest.fixture
def mock_memories_promoted() -> list[MagicMock]:
    """Create mock memories with promoted trust (>= 0.8 for L1)."""
    return [
        MagicMock(
            id="promoted-1",
            text="Architecture: Use dependency injection for testability",
            trust_score=0.95,
            is_archived=False,
        ),
        MagicMock(
            id="promoted-2",
            text="Pattern: Repository pattern for data access",
            trust_score=0.88,
            is_archived=False,
        ),
    ]


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response from LLM."""
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(
        return_value={
            "response": "This project is a Python-based MCP memory server using async/await patterns and type hints throughout."
        }
    )
    return response


class TestIsEnabled:
    """Tests for is_enabled property."""

    @pytest.mark.asyncio
    async def test_is_enabled_true(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled returns True when config enabled."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()
            assert loader.is_enabled is True

    @pytest.mark.asyncio
    async def test_is_enabled_false(self, mock_config_disabled: MagicMock) -> None:
        """is_enabled returns False when config disabled."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_disabled):
            loader = TieredLoader()
            assert loader.is_enabled is False

    @pytest.mark.asyncio
    async def test_is_enabled_cached(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled is cached after first call."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()
            first = loader.is_enabled
            second = loader.is_enabled
            assert first is second


class TestStats:
    """Tests for stats property."""

    @pytest.mark.asyncio
    async def test_stats_initial(self, mock_config_enabled: MagicMock) -> None:
        """stats returns initial TieredLoadingStats."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()
            stats = loader.stats
            assert stats.l0_generations == 0
            assert stats.l1_generations == 0
            assert stats.l0_cache_hits == 0
            assert stats.l1_cache_hits == 0

    @pytest.mark.asyncio
    async def test_stats_updates_on_generation(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
    ) -> None:
        """stats updates after L0 generation."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Override _generate_summary directly instead of mocking HTTP client
            async def mock_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                return ("Generated L0 summary", [mid for mid, _ in memories])

            loader._generate_summary = mock_generate

            await loader.get_tier0()

            stats = loader.stats
            assert stats.l0_generations == 1
            assert stats.last_l0_generated is not None


class TestDisabledNoOp:
    """Tests for disabled behavior (no-op)."""

    @pytest.mark.asyncio
    async def test_get_tier0_disabled_returns_error(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, get_tier0 returns error info."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_disabled):
            loader = TieredLoader()
            result = await loader.get_tier0()

            assert result["tier"] == "L0"
            assert result["content"] is None
            assert result["cache_hit"] is False
            assert result["source_count"] == 0
            assert result["error"] == "Tiered loading is disabled"

    @pytest.mark.asyncio
    async def test_get_tier1_disabled_returns_error(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, get_tier1 returns error info."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_disabled):
            loader = TieredLoader()
            result = await loader.get_tier1()

            assert result["tier"] == "L1"
            assert result["content"] is None
            assert result["cache_hit"] is False
            assert result["source_count"] == 0
            assert result["error"] == "Tiered loading is disabled"


class TestTier0Generation:
    """Tests for L0 tier generation."""

    @pytest.mark.asyncio
    async def test_tier0_generation(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
        mock_http_response: MagicMock,
    ) -> None:
        """get_tier0 generates summary from high-trust memories."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_http_response)
            loader._http_client = mock_client

            result = await loader.get_tier0()

            assert result["tier"] == "L0"
            assert result["content"] is not None
            assert result["cache_hit"] is False
            assert result["source_count"] == 3
            assert "token_count" in result
            assert "generated_at" in result

    @pytest.mark.asyncio
    async def test_tier0_no_memories(
        self, mock_config_enabled: MagicMock, mock_memory_system: MagicMock
    ) -> None:
        """get_tier0 returns error when no high-trust memories."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            result = await loader.get_tier0()

            assert result["tier"] == "L0"
            assert result["content"] is None
            assert result["error"] == "No high-trust memories available"

    @pytest.mark.asyncio
    async def test_tier0_respects_trust_threshold(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """get_tier0 only uses memories with trust >= 0.5."""
        # Mix of memories with different trust scores
        memories = [
            MagicMock(id="high", text="High trust", trust_score=0.9, is_archived=False),
            MagicMock(id="medium", text="Medium trust", trust_score=0.6, is_archived=False),
            MagicMock(id="low", text="Low trust", trust_score=0.3, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured_memories: list[tuple[str, str]] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Override _generate_summary to capture input
            async def capture_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                captured_memories.extend(memories)
                return ("summary", [mid for mid, _ in memories])

            loader._generate_summary = capture_generate

            await loader.get_tier0()

            # Should only include high (0.9) and medium (0.6) - both >= 0.5
            assert len(captured_memories) == 2
            memory_ids = [mid for mid, _ in captured_memories]
            assert "high" in memory_ids
            assert "medium" in memory_ids
            assert "low" not in memory_ids


class TestTier1Generation:
    """Tests for L1 tier generation."""

    @pytest.mark.asyncio
    async def test_tier1_generation(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_promoted: list[MagicMock],
        mock_http_response: MagicMock,
    ) -> None:
        """get_tier1 generates summary from promoted memories."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_promoted)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_http_response)
            loader._http_client = mock_client

            result = await loader.get_tier1()

            assert result["tier"] == "L1"
            assert result["content"] is not None
            assert result["cache_hit"] is False
            assert result["source_count"] == 2
            assert "token_count" in result
            assert "generated_at" in result

    @pytest.mark.asyncio
    async def test_tier1_no_promoted_memories(
        self, mock_config_enabled: MagicMock, mock_memory_system: MagicMock
    ) -> None:
        """get_tier1 returns error when no promoted memories (trust >= 0.8)."""
        # Only memories below promotion threshold
        memories = [
            MagicMock(id="mem-1", text="Some memory", trust_score=0.5, is_archived=False),
            MagicMock(id="mem-2", text="Another memory", trust_score=0.7, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            result = await loader.get_tier1()

            assert result["tier"] == "L1"
            assert result["content"] is None
            assert result["error"] == "No promoted memories available (trust >= 0.8 required)"

    @pytest.mark.asyncio
    async def test_tier1_respects_promotion_threshold(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """get_tier1 only uses memories with trust >= 0.8."""
        memories = [
            MagicMock(id="promoted", text="Promoted", trust_score=0.9, is_archived=False),
            MagicMock(id="high", text="High", trust_score=0.75, is_archived=False),
            MagicMock(id="medium", text="Medium", trust_score=0.5, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured_memories: list[tuple[str, str]] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def capture_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                captured_memories.extend(memories)
                return ("summary", [mid for mid, _ in memories])

            loader._generate_summary = capture_generate

            await loader.get_tier1()

            # Should only include promoted (0.9) - >= 0.8
            assert len(captured_memories) == 1
            assert captured_memories[0][0] == "promoted"


class TestCacheHit:
    """Tests for cache hit behavior."""

    @pytest.mark.asyncio
    async def test_l0_cache_hit(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_http_response: MagicMock,
    ) -> None:
        """get_tier0 returns cached L0 without regeneration."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Pre-populate cache
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Cached L0 summary",
                source_memory_ids=["cached-1"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            # Track if generation was called
            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("new", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0()

            assert result["cache_hit"] is True
            assert result["content"] == "Cached L0 summary"
            assert generation_called is False
            assert loader.stats.l0_cache_hits == 1

    @pytest.mark.asyncio
    async def test_l1_cache_hit(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_http_response: MagicMock,
    ) -> None:
        """get_tier1 returns cached L1 without regeneration."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Pre-populate cache
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1,
                content="Cached L1 summary",
                source_memory_ids=["cached-1", "cached-2"],
                generated_at=datetime.utcnow(),
                token_count=500,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("new", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier1()

            assert result["cache_hit"] is True
            assert result["content"] == "Cached L1 summary"
            assert generation_called is False
            assert loader.stats.l1_cache_hits == 1


class TestForceRefresh:
    """Tests for force_refresh parameter."""

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
        mock_http_response: MagicMock,
    ) -> None:
        """force_refresh=True bypasses cache and regenerates."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Pre-populate cache
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Stale cached content",
                source_memory_ids=["old"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("Freshly generated", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0(force_refresh=True)

            assert result["cache_hit"] is False
            assert result["content"] == "Freshly generated"
            assert generation_called is True

    @pytest.mark.asyncio
    async def test_force_refresh_false_uses_cache(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """force_refresh=False (default) uses cache if valid."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Valid cached content",
                source_memory_ids=["cached"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("new", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0(force_refresh=False)

            assert result["cache_hit"] is True
            assert result["content"] == "Valid cached content"
            assert generation_called is False


class TestCacheStaleDetection:
    """Tests for cache staleness detection."""

    @pytest.mark.asyncio
    async def test_cache_stale_after_ttl(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
    ) -> None:
        """Cache is considered stale after TTL expires."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Create cache that is older than TTL (3600 seconds = 1 hour)
            old_time = datetime.utcnow() - timedelta(hours=2)
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Old cached content",
                source_memory_ids=["old"],
                generated_at=old_time,
                token_count=25,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("Freshly generated", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0()

            assert result["cache_hit"] is False
            assert generation_called is True

    @pytest.mark.asyncio
    async def test_cache_valid_within_ttl(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Cache is valid when within TTL."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Create fresh cache (just now)
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Fresh cached content",
                source_memory_ids=["fresh"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("new", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0()

            assert result["cache_hit"] is True
            assert generation_called is False

    @pytest.mark.asyncio
    async def test_mark_stale_marks_cache_stale(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """mark_stale() sets is_stale flag on cache."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Fresh cached content",
                source_memory_ids=["fresh"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            # Mark as stale
            loader.mark_stale("L0")

            assert loader._l0_cache.is_stale is True

    @pytest.mark.asyncio
    async def test_cache_valid_within_ttl(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Cache is valid when within TTL."""
        mock_memory_system.list_memories = AsyncMock(return_value=[])

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Create fresh cache (just now)
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Fresh cached content",
                source_memory_ids=["fresh"],
                generated_at=datetime.utcnow(),
                token_count=25,
                is_stale=False,
            )

            generation_called = False

            async def track_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                nonlocal generation_called
                generation_called = True
                return ("new", [])

            loader._generate_summary = track_generate

            result = await loader.get_tier0()

            assert result["cache_hit"] is True
            assert generation_called is False


class TestHighTrustFilter:
    """Tests for L0 high-trust filter (trust >= 0.5)."""

    @pytest.mark.asyncio
    async def test_l0_filters_by_trust_05(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """L0 includes only memories with trust >= 0.5."""
        memories = [
            MagicMock(id="trust-09", text="Very trusted", trust_score=0.9, is_archived=False),
            MagicMock(id="trust-05", text="Exactly 0.5", trust_score=0.5, is_archived=False),
            MagicMock(id="trust-04", text="Below 0.5", trust_score=0.4, is_archived=False),
            MagicMock(id="trust-02", text="Low trust", trust_score=0.2, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured: list[tuple[str, str]] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def capture(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                captured.extend(memories)
                return ("", [])

            loader._generate_summary = capture
            await loader.get_tier0()

            ids = [mid for mid, _ in captured]
            assert "trust-09" in ids
            assert "trust-05" in ids
            assert "trust-04" not in ids
            assert "trust-02" not in ids


class TestPromotedFilter:
    """Tests for L1 promoted filter (trust >= 0.8)."""

    @pytest.mark.asyncio
    async def test_l1_filters_by_trust_08(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """L1 includes only memories with trust >= 0.8."""
        memories = [
            MagicMock(id="promoted-09", text="Very promoted", trust_score=0.95, is_archived=False),
            MagicMock(id="promoted-08", text="Exactly 0.8", trust_score=0.8, is_archived=False),
            MagicMock(id="high-07", text="High but not promoted", trust_score=0.7, is_archived=False),
            MagicMock(id="medium-05", text="Medium trust", trust_score=0.5, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured: list[tuple[str, str]] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def capture(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                captured.extend(memories)
                return ("", [])

            loader._generate_summary = capture
            await loader.get_tier1()

            ids = [mid for mid, _ in captured]
            assert "promoted-09" in ids
            assert "promoted-08" in ids
            assert "high-07" not in ids
            assert "medium-05" not in ids


class TestExcludeArchived:
    """Tests for excluding archived memories."""

    @pytest.mark.asyncio
    async def test_excludes_archived_memories(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Archived memories are excluded from both L0 and L1."""
        memories = [
            MagicMock(id="active", text="Active memory", trust_score=0.9, is_archived=False),
            MagicMock(id="archived", text="Archived memory", trust_score=0.95, is_archived=True),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured: list[tuple[str, str]] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def capture(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                captured.extend(memories)
                return ("", [])

            loader._generate_summary = capture
            await loader.get_tier0()

            ids = [mid for mid, _ in captured]
            assert "active" in ids
            assert "archived" not in ids


class TestInvalidateCache:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_all(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """invalidate_cache(tier=None) clears all caches."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0, content="L0", source_memory_ids=[], generated_at=datetime.utcnow()
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1, content="L1", source_memory_ids=[], generated_at=datetime.utcnow()
            )

            loader.invalidate_cache()

            assert loader._l0_cache is None
            assert loader._l1_cache is None

    @pytest.mark.asyncio
    async def test_invalidate_l0_only(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """invalidate_cache(tier="L0") clears only L0 cache."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0, content="L0", source_memory_ids=[], generated_at=datetime.utcnow()
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1, content="L1", source_memory_ids=[], generated_at=datetime.utcnow()
            )

            loader.invalidate_cache("L0")

            assert loader._l0_cache is None
            assert loader._l1_cache is not None

    @pytest.mark.asyncio
    async def test_invalidate_l1_only(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """invalidate_cache(tier="L1") clears only L1 cache."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0, content="L0", source_memory_ids=[], generated_at=datetime.utcnow()
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1, content="L1", source_memory_ids=[], generated_at=datetime.utcnow()
            )

            loader.invalidate_cache("L1")

            assert loader._l0_cache is not None
            assert loader._l1_cache is None


class TestMarkStale:
    """Tests for marking cache as stale."""

    @pytest.mark.asyncio
    async def test_mark_stale_all(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """mark_stale(tier=None) marks all caches as stale."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0, content="L0", source_memory_ids=[], generated_at=datetime.utcnow(), is_stale=False
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1, content="L1", source_memory_ids=[], generated_at=datetime.utcnow(), is_stale=False
            )

            loader.mark_stale()

            assert loader._l0_cache.is_stale is True
            assert loader._l1_cache.is_stale is True

    @pytest.mark.asyncio
    async def test_mark_stale_l0_only(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """mark_stale(tier="L0") marks only L0 cache as stale."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0, content="L0", source_memory_ids=[], generated_at=datetime.utcnow(), is_stale=False
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1, content="L1", source_memory_ids=[], generated_at=datetime.utcnow(), is_stale=False
            )

            loader.mark_stale("L0")

            assert loader._l0_cache.is_stale is True
            assert loader._l1_cache.is_stale is False


class TestGetSummaryStatus:
    """Tests for get_summary_status method."""

    @pytest.mark.asyncio
    async def test_status_no_cache(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """get_summary_status returns correct format when no cache."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            status = await loader.get_summary_status()

            assert status["enabled"] is True
            assert status["l0"]["cached"] is False
            assert status["l1"]["cached"] is False
            assert "stats" in status

    @pytest.mark.asyncio
    async def test_status_with_cache(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """get_summary_status returns cache info when cached."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            now = datetime.utcnow()
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="L0 content",
                source_memory_ids=["mem-1", "mem-2"],
                generated_at=now,
                token_count=25,
                is_stale=False,
            )
            loader._l1_cache = TieredSummary(
                tier=SummaryTier.L1,
                content="L1 content",
                source_memory_ids=["mem-3"],
                generated_at=now,
                token_count=50,
                is_stale=False,
            )

            status = await loader.get_summary_status()

            assert status["l0"]["cached"] is True
            assert status["l0"]["token_count"] == 25
            assert status["l0"]["source_count"] == 2
            assert status["l0"]["is_stale"] is False

            assert status["l1"]["cached"] is True
            assert status["l1"]["token_count"] == 50
            assert status["l1"]["source_count"] == 1


class TestLLMCallFailure:
    """Tests for LLM call failure handling."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_stale_cache(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
    ) -> None:
        """On LLM failure, returns stale cache if available."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            # Pre-populate with stale cache
            loader._l0_cache = TieredSummary(
                tier=SummaryTier.L0,
                content="Stale but available cache",
                source_memory_ids=["cached"],
                generated_at=datetime.utcnow() - timedelta(hours=3),
                token_count=25,
                is_stale=False,
            )

            # Simulate LLM failure
            async def fail_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                return ("", [])  # Empty content = failure

            loader._generate_summary = fail_generate

            result = await loader.get_tier0()

            # Should return stale cache with warning
            assert result["content"] == "Stale but available cache"
            assert "error" in result
            assert "LLM call failed" in result["error"]
            assert result["cache_hit"] is False  # It's a fallback, not a hit

    @pytest.mark.asyncio
    async def test_llm_failure_increments_error_count(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_memories_high_trust: list[MagicMock],
    ) -> None:
        """LLM failure increments error count in stats."""
        mock_memory_system.list_memories = AsyncMock(return_value=mock_memories_high_trust)

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def fail_generate(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                return ("", [])

            loader._generate_summary = fail_generate

            initial_errors = loader.stats.errors
            await loader.get_tier0()

            assert loader.stats.errors > initial_errors


class TestClose:
    """Tests for HTTP client cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_client(self, mock_config_enabled: MagicMock) -> None:
        """close() properly closes HTTP client."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            loader._http_client = mock_client

            await loader.close()

            mock_client.aclose.assert_called_once()
            assert loader._http_client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_config_enabled: MagicMock) -> None:
        """close() handles None client gracefully."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()
            loader._http_client = None

            # Should not raise
            await loader.close()

            assert loader._http_client is None


class TestEmptyMemories:
    """Tests for empty memory handling."""

    @pytest.mark.asyncio
    async def test_generate_summary_empty_memories(
        self,
        mock_config_enabled: MagicMock,
    ) -> None:
        """_generate_summary returns empty when no memories."""
        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader()

            content, ids = await loader._generate_summary([], L0_SUMMARY_PROMPT, 100)

            assert content == ""
            assert ids == []


class TestPromptTemplates:
    """Tests for L0 and L1 prompt templates."""

    def test_l0_prompt_has_memories_placeholder(self) -> None:
        """L0_SUMMARY_PROMPT contains {memories} placeholder."""
        assert "{memories}" in L0_SUMMARY_PROMPT

    def test_l1_prompt_has_memories_placeholder(self) -> None:
        """L1_SUMMARY_PROMPT contains {memories} placeholder."""
        assert "{memories}" in L1_SUMMARY_PROMPT


class TestSortingByTrust:
    """Tests for sorting memories by trust score."""

    @pytest.mark.asyncio
    async def test_memories_sorted_by_trust_descending(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Memories are sorted by trust score descending."""
        # Only high (0.9) and medium (0.6) pass the trust >= 0.5 filter
        # low (0.3) is filtered out
        memories = [
            MagicMock(id="low", text="Low", trust_score=0.3, is_archived=False),
            MagicMock(id="high", text="High", trust_score=0.9, is_archived=False),
            MagicMock(id="medium", text="Medium", trust_score=0.6, is_archived=False),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        captured_order: list[float] = []

        with patch("memini_ai.tiered_loader.get_config", return_value=mock_config_enabled):
            loader = TieredLoader(memory_system=mock_memory_system)

            async def capture(
                memories: list[tuple[str, str]], prompt: str, max_tokens: int
            ) -> tuple[str, list[str]]:
                # Capture trust scores in order received
                for mid, text in memories:
                    mem = next(m for m in mock_memories if m.id == mid)
                    captured_order.append(mem.trust_score)
                return ("", [])

            mock_memories = memories

            loader._generate_summary = capture
            await loader.get_tier0()

            # Should be sorted descending: high (0.9), medium (0.6)
            # low (0.3) is filtered out by trust >= 0.5 filter
            assert captured_order == [0.9, 0.6]
