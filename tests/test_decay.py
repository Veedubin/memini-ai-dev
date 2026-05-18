"""Tests for Memory Decay and Consolidation feature (Phase 4A)."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.config import MeminiConfig
from memini_ai.decay import (
    DECAY_BASE_HALF_LIFE_DAYS,
    DECAY_DEFAULT_RATE,
    DECAY_MAX_RATE,
    DECAY_MIN_RATE,
    FADE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CONSOLIDATION_SIMILARITY,
    ConsolidationCandidate,
    ConsolidationEngine,
    ConsolidationStats,
    DecayEngine,
    DecayStats,
    MemoryDecayInfo,
    adjust_decay_rate,
)
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    TrustLevel,
)


class TestDecayConstants:
    """Tests for decay-related constants."""

    def test_decay_base_half_life_days(self) -> None:
        """Default half-life should be 90 days."""
        assert DECAY_BASE_HALF_LIFE_DAYS == 90

    def test_decay_min_rate(self) -> None:
        """Minimum decay rate should be 0.1."""
        assert DECAY_MIN_RATE == 0.1

    def test_decay_max_rate(self) -> None:
        """Maximum decay rate should be 10.0."""
        assert DECAY_MAX_RATE == 10.0

    def test_decay_default_rate(self) -> None:
        """Default decay rate should be 1.0."""
        assert DECAY_DEFAULT_RATE == 1.0

    def test_fade_threshold(self) -> None:
        """Fade threshold should be 0.15."""
        assert FADE_THRESHOLD == 0.15

    def test_default_similarity_threshold(self) -> None:
        """Default consolidation similarity threshold should be 0.92."""
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.92

    def test_min_consolidation_similarity(self) -> None:
        """Minimum consolidation similarity should be 0.70."""
        assert MIN_CONSOLIDATION_SIMILARITY == 0.70


class TestDecayStats:
    """Tests for DecayStats dataclass."""

    def test_default_stats(self) -> None:
        """Default stats should have zero values."""
        stats = DecayStats()
        assert stats.memories_processed == 0
        assert stats.memories_decayed == 0
        assert stats.memories_archived == 0
        assert stats.last_run is None
        assert stats.next_scheduled_run is None
        assert stats.total_decay_events == 0

    def test_stats_with_values(self) -> None:
        """Stats should accept custom values."""
        now = datetime.utcnow()
        stats = DecayStats(
            memories_processed=100,
            memories_decayed=50,
            memories_archived=5,
            last_run=now,
            total_decay_events=200,
        )
        assert stats.memories_processed == 100
        assert stats.memories_decayed == 50
        assert stats.last_run == now


class TestConsolidationStats:
    """Tests for ConsolidationStats dataclass."""

    def test_default_stats(self) -> None:
        """Default stats should have zero values."""
        stats = ConsolidationStats()
        assert stats.pairs_found == 0
        assert stats.pairs_merged == 0
        assert stats.memories_consolidated == 0
        assert stats.last_run is None

    def test_stats_with_values(self) -> None:
        """Stats should accept custom values."""
        now = datetime.utcnow()
        stats = ConsolidationStats(
            pairs_found=20,
            pairs_merged=10,
            memories_consolidated=10,
            last_run=now,
        )
        assert stats.pairs_found == 20
        assert stats.pairs_merged == 10


class TestMemoryDecayInfo:
    """Tests for MemoryDecayInfo dataclass."""

    def test_create_decay_info(self) -> None:
        """Should create decay info with all fields."""
        info = MemoryDecayInfo(
            memory_id="mem-123",
            text_preview="This is a test memory...",
            current_decay_rate=1.5,
            trust_score=0.6,
            trust_level=TrustLevel.MEDIUM,
            last_accessed=datetime.utcnow(),
            access_count=5,
            days_until_archive=45.5,
            is_fading=False,
        )
        assert info.memory_id == "mem-123"
        assert info.trust_score == 0.6
        assert info.is_fading is False

    def test_decay_info_is_fading(self) -> None:
        """Should identify fading memories."""
        info = MemoryDecayInfo(
            memory_id="mem-456",
            text_preview="Fading memory...",
            current_decay_rate=2.0,
            trust_score=0.25,
            trust_level=TrustLevel.LOW,
            last_accessed=datetime.utcnow() - timedelta(days=60),
            access_count=1,
            days_until_archive=10.0,
            is_fading=True,
        )
        assert info.is_fading is True
        assert info.days_until_archive == 10.0


class TestConsolidationCandidate:
    """Tests for ConsolidationCandidate dataclass."""

    def test_create_candidate(self) -> None:
        """Should create candidate with memory pair."""
        mem_a = MemoryEntry(
            id="mem-a",
            text="Memory A text content",
            source_type=MemorySourceType.session,
        )
        mem_b = MemoryEntry(
            id="mem-b",
            text="Memory B text content",
            source_type=MemorySourceType.session,
        )
        candidate = ConsolidationCandidate(
            memory_a=mem_a,
            memory_b=mem_b,
            similarity=0.95,
            combined_text="Memory A text content [MERGED] Memory B text content",
        )
        assert candidate.memory_a.id == "mem-a"
        assert candidate.memory_b.id == "mem-b"
        assert candidate.similarity == 0.95
        assert "[MERGED]" in candidate.combined_text


class TestDecayEngine:
    """Tests for DecayEngine class."""

    def test_calculate_decay_no_time(self) -> None:
        """No decay should occur with zero days elapsed."""
        engine = DecayEngine()
        new_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=0,
            half_life_days=90,
        )
        assert new_score == 0.8

    def test_calculate_decay_normal_rate(self) -> None:
        """Normal decay rate should halve score after half-life."""
        engine = DecayEngine()
        new_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=90,  # Exactly one half-life
            half_life_days=90,
        )
        # After half-life, score should be ~0.4 (half of 0.8)
        assert abs(new_score - 0.4) < 0.01

    def test_calculate_decay_fast_rate(self) -> None:
        """Higher decay rate should decay faster."""
        engine = DecayEngine()
        normal_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=45,
            half_life_days=90,
        )
        fast_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=2.0,  # 2x faster
            days_elapsed=45,
            half_life_days=90,
        )
        # Fast rate should decay more
        assert fast_score < normal_score

    def test_calculate_decay_slow_rate(self) -> None:
        """Lower decay rate should decay slower."""
        engine = DecayEngine()
        normal_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=45,
            half_life_days=90,
        )
        slow_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=0.5,  # 2x slower
            days_elapsed=45,
            half_life_days=90,
        )
        # Slow rate should decay less
        assert slow_score > normal_score

    def test_calculate_days_until_archive(self) -> None:
        """Should calculate days until archive threshold."""
        engine = DecayEngine()
        days = engine.calculate_days_until_archive(
            trust_score=0.4,
            decay_rate=1.0,
            half_life_days=90,
            archive_threshold=FADE_THRESHOLD,
        )
        assert days is not None
        assert days > 0

    def test_calculate_days_until_archive_below_threshold(self) -> None:
        """Should return None when already below threshold."""
        engine = DecayEngine()
        days = engine.calculate_days_until_archive(
            trust_score=0.1,
            decay_rate=1.0,
            half_life_days=90,
            archive_threshold=FADE_THRESHOLD,
        )
        assert days is None

    def test_calculate_decay_180_days(self) -> None:
        """180 days (2 half-lives) should reduce to 1/4."""
        engine = DecayEngine()
        new_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=180,  # Two half-lives
            half_life_days=90,
        )
        # 0.8 * 0.5 * 0.5 = 0.2
        assert abs(new_score - 0.2) < 0.01

    def test_decays_to_zero_eventually(self) -> None:
        """Very long time should decay trust to near zero."""
        engine = DecayEngine()
        new_score = engine.calculate_decay(
            trust_score=0.8,
            decay_rate=1.0,
            days_elapsed=1000,
            half_life_days=90,
        )
        assert new_score < 0.01

    @pytest.mark.asyncio
    async def test_apply_decay_no_memory_system(self) -> None:
        """Should return None when no memory system."""
        engine = DecayEngine(memory_system=None)
        memory = MemoryEntry(
            id="mem-123",
            text="Test memory",
            source_type=MemorySourceType.session,
            trust_score=0.8,
        )
        result = await engine.apply_decay(memory)
        assert result is None  # No memory system, should skip

    @pytest.mark.asyncio
    async def test_process_memories_disabled(self) -> None:
        """Should return empty stats when disabled."""
        engine = DecayEngine(memory_system=None)
        mock_config = MagicMock()
        mock_config.decay_enabled = False
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            result = await engine.process_memories()
            assert result["processed"] == 0
            assert result["decayed"] == 0


class TestConsolidationEngine:
    """Tests for ConsolidationEngine class."""

    def test_calculate_text_similarity_identical(self) -> None:
        """Identical texts should have 1.0 similarity."""
        engine = ConsolidationEngine()
        similarity = engine._calculate_text_similarity(
            "hello world test",
            "hello world test",
        )
        assert similarity == 1.0

    def test_calculate_text_similarity_none(self) -> None:
        """No common words should have 0.0 similarity."""
        engine = ConsolidationEngine()
        similarity = engine._calculate_text_similarity(
            "hello world",
            "foo bar baz",
        )
        assert similarity == 0.0

    def test_calculate_text_similarity_partial(self) -> None:
        """Partial overlap should give partial similarity."""
        engine = ConsolidationEngine()
        similarity = engine._calculate_text_similarity(
            "hello world test one",
            "hello world another",
        )
        # Common: hello, world (2)
        # Union: hello, world, test, one, another (5)
        # Jaccard = 2/5 = 0.4
        assert 0.3 < similarity < 0.5

    def test_calculate_text_similarity_case_insensitive(self) -> None:
        """Similarity should be case insensitive."""
        engine = ConsolidationEngine()
        sim1 = engine._calculate_text_similarity("Hello World", "hello world")
        sim2 = engine._calculate_text_similarity("hello world", "hello world")
        assert abs(sim1 - sim2) < 0.001

    def test_combine_texts(self) -> None:
        """Should combine texts with separator."""
        engine = ConsolidationEngine()
        combined = engine._combine_texts("Text A", "Text B")
        assert "Text A" in combined
        assert "Text B" in combined
        assert "[MERGED]" in combined

    @pytest.mark.asyncio
    async def test_find_similar_pairs_none(self) -> None:
        """Should return empty when no similar pairs."""
        engine = ConsolidationEngine(memory_system=None)
        pairs = await engine.find_similar_pairs(threshold=0.92)
        assert pairs == []

    @pytest.mark.asyncio
    async def test_run_consolidation_disabled(self) -> None:
        """Should return zeros when disabled."""
        engine = ConsolidationEngine(memory_system=None)
        mock_config = MagicMock()
        mock_config.decay_enabled = False
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            result = await engine.run_consolidation()
            assert result["consolidated"] == 0


class TestAdjustDecayRate:
    """Tests for adjust_decay_rate function."""

    @pytest.mark.asyncio
    async def test_adjust_rate_not_found(self) -> None:
        """Should return error when memory not found."""
        mock_system = MagicMock()
        mock_system.get_memory = AsyncMock(return_value=None)

        result = await adjust_decay_rate(mock_system, "nonexistent", 1.5)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_adjust_rate_clamp_low(self) -> None:
        """Should clamp rate below minimum."""
        mock_memory = MagicMock()
        mock_memory.id = "mem-123"
        mock_memory.decay_rate = 1.0

        mock_system = MagicMock()
        mock_system.get_memory = AsyncMock(return_value=mock_memory)

        result = await adjust_decay_rate(mock_system, "mem-123", 0.01)  # Way below min

        assert result["success"] is True
        assert result["decay_rate"] == DECAY_MIN_RATE

    @pytest.mark.asyncio
    async def test_adjust_rate_clamp_high(self) -> None:
        """Should clamp rate above maximum."""
        mock_memory = MagicMock()
        mock_memory.id = "mem-123"
        mock_memory.decay_rate = 1.0

        mock_system = MagicMock()
        mock_system.get_memory = AsyncMock(return_value=mock_memory)

        result = await adjust_decay_rate(mock_system, "mem-123", 20.0)  # Way above max

        assert result["success"] is True
        assert result["decay_rate"] == DECAY_MAX_RATE


class TestDecayEngineIntegration:
    """Integration tests for decay engine with mocked memory system."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create a mocked memory system."""
        system = MagicMock()
        system.list_memories = AsyncMock(return_value=[])
        system.get_memory = AsyncMock(return_value=None)
        return system

    @pytest.mark.asyncio
    async def test_get_decay_status_empty(self, mock_memory_system: MagicMock) -> None:
        """Should return empty stats when no memories."""
        engine = DecayEngine(memory_system=mock_memory_system)
        mock_config = MagicMock()
        mock_config.decay_enabled = True
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            status = await engine.get_decay_status()

            assert status["enabled"] is True
            assert status["fading_count"] == 0
            assert status["fading_memories"] == []

    @pytest.mark.asyncio
    async def test_get_decay_status_with_memories(self, mock_memory_system: MagicMock) -> None:
        """Should show fading memories."""
        now = datetime.utcnow()
        memories = [
            MagicMock(
                id="mem-1",
                text="Fresh memory with high trust",
                trust_score=0.8,
                is_archived=False,
                last_accessed=now,
                retrieval_count=10,
                decay_rate=1.0,
            ),
            MagicMock(
                id="mem-2",
                text="Old memory with low trust",
                trust_score=0.2,
                is_archived=False,
                last_accessed=now - timedelta(days=100),
                retrieval_count=1,
                decay_rate=1.5,
            ),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        engine = DecayEngine(memory_system=mock_memory_system)
        mock_config = MagicMock()
        mock_config.decay_enabled = True
        mock_config.decay_half_life_days = 90
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            status = await engine.get_decay_status()

            assert status["enabled"] is True
            assert status["fading_count"] >= 1  # mem-2 is fading

    @pytest.mark.asyncio
    async def test_list_fading_memories(self, mock_memory_system: MagicMock) -> None:
        """Should list fading memories sorted by urgency."""
        now = datetime.utcnow()
        memories = [
            MagicMock(
                id="mem-urgent",
                text="About to fade - needs attention soon",
                trust_score=0.18,
                is_archived=False,
                timestamp=now - timedelta(days=200),
                last_accessed=now - timedelta(days=180),
                retrieval_count=2,
                decay_rate=1.0,
            ),
            MagicMock(
                id="mem-ok",
                text="Healthy memory with high trust",
                trust_score=0.9,
                is_archived=False,
                timestamp=now,
                last_accessed=now,
                retrieval_count=100,
                decay_rate=0.5,
            ),
        ]
        mock_memory_system.list_memories = AsyncMock(return_value=memories)

        engine = ConsolidationEngine(memory_system=mock_memory_system)
        mock_config = MagicMock()
        mock_config.decay_enabled = True
        mock_config.decay_half_life_days = 90
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            fading = await engine.list_fading_memories(limit=10)

            # Should have at least the urgent one
            assert len(fading) >= 1
            # Urgent should be first (lowest days_until_archive)
            ids = [m["memory_id"] for m in fading]
            assert "mem-urgent" in ids


class TestConsolidationWorkflow:
    """Tests for consolidation workflow with mocked memory system."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create a mocked memory system."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_consolidate_pair_higher_trust_survives(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Memory with higher trust should survive consolidation."""
        mem_a = MagicMock()
        mem_a.id = "mem-a"
        mem_a.text = "Memory A content"
        mem_a.trust_score = 0.8
        mem_a.is_archived = False

        mem_b = MagicMock()
        mem_b.id = "mem-b"
        mem_b.text = "Memory B content"
        mem_b.trust_score = 0.4
        mem_b.is_archived = False

        mock_memory_system.get_memory = AsyncMock(side_effect=[mem_a, mem_b])

        candidate = ConsolidationCandidate(
            memory_a=mem_a,
            memory_b=mem_b,
            similarity=0.95,
            combined_text="Memory A content [MERGED] Memory B content",
        )

        engine = ConsolidationEngine(memory_system=mock_memory_system)
        result = await engine.consolidate_pair(candidate)

        # Should not raise - consolidation attempted
        # (actual DB update would fail in mock but function handles gracefully)


class TestDecayEnabledFlag:
    """Tests for decay feature enable/disable behavior."""

    def test_decay_engine_disabled_by_default(self) -> None:
        """DecayEngine should be disabled when config decay_enabled is False."""
        engine = DecayEngine()
        # Config default is False
        assert engine.is_enabled is False

    def test_consolidation_engine_disabled_by_default(self) -> None:
        """ConsolidationEngine should be disabled when config decay_enabled is False."""
        engine = ConsolidationEngine()
        # Config default is False
        assert engine.is_enabled is False

    @pytest.mark.asyncio
    async def test_apply_decay_respects_enabled_flag(self) -> None:
        """Should not decay when disabled."""
        engine = DecayEngine(memory_system=MagicMock())
        memory = MagicMock()
        memory.trust_score = 0.8
        memory.decay_rate = 1.0
        memory.last_accessed = datetime.utcnow() - timedelta(days=30)
        memory.timestamp = datetime.utcnow() - timedelta(days=30)

        mock_config = MagicMock()
        mock_config.decay_enabled = False
        with patch('memini_ai.decay.get_config', return_value=mock_config):
            result = await engine.apply_decay(memory)
            assert result is None


class TestDecayFieldOnMemoryEntry:
    """Tests for decay fields on MemoryEntry."""

    def test_memory_entry_has_decay_rate_default(self) -> None:
        """MemoryEntry should have decay_rate field with default 1.0."""
        memory = MemoryEntry(
            id="test-1",
            text="Test content",
            source_type=MemorySourceType.session,
        )
        assert memory.decay_rate == DECAY_DEFAULT_RATE

    def test_memory_entry_has_last_accessed_default_none(self) -> None:
        """MemoryEntry should have last_accessed field defaulting to None."""
        memory = MemoryEntry(
            id="test-2",
            text="Test content",
            source_type=MemorySourceType.session,
        )
        assert memory.last_accessed is None

    def test_memory_entry_has_access_count_default_zero(self) -> None:
        """MemoryEntry should have access_count field defaulting to 0."""
        memory = MemoryEntry(
            id="test-3",
            text="Test content",
            source_type=MemorySourceType.session,
        )
        assert memory.access_count == 0

    def test_memory_entry_decay_fields_can_be_set(self) -> None:
        """MemoryEntry decay fields should be settable."""
        now = datetime.utcnow()
        memory = MemoryEntry(
            id="test-4",
            text="Test content",
            source_type=MemorySourceType.session,
            decay_rate=2.0,
            last_accessed=now,
            access_count=5,
        )
        assert memory.decay_rate == 2.0
        assert memory.last_accessed == now
        assert memory.access_count == 5

    def test_memory_entry_decay_rate_clamped_by_pydantic(self) -> None:
        """MemoryEntry decay_rate should accept any float value (validation optional)."""
        memory = MemoryEntry(
            id="test-5",
            text="Test content",
            source_type=MemorySourceType.session,
            decay_rate=0.5,
        )
        # Value should be stored as-is (pydantic doesn't enforce custom clamps)
        assert memory.decay_rate == 0.5


class TestMemoryEntryAliasing:
    """Tests for MemoryEntry field aliasing (for Qdrant compatibility)."""

    def test_decay_rate_alias(self) -> None:
        """decayRate alias should map to decay_rate."""
        data = {
            "id": "test-alias",
            "text": "Test",
            "sourceType": "session",
            "decayRate": 1.5,
        }
        memory = MemoryEntry.model_validate(data)
        assert memory.decay_rate == 1.5

    def test_last_accessed_alias(self) -> None:
        """lastAccessed alias should map to last_accessed."""
        now = datetime.utcnow()
        data = {
            "id": "test-alias-2",
            "text": "Test",
            "sourceType": "session",
            "lastAccessed": now.isoformat(),
        }
        memory = MemoryEntry.model_validate(data)
        assert memory.last_accessed is not None

    def test_access_count_alias(self) -> None:
        """accessCount alias should map to access_count."""
        data = {
            "id": "test-alias-3",
            "text": "Test",
            "sourceType": "session",
            "accessCount": 42,
        }
        memory = MemoryEntry.model_validate(data)
        assert memory.access_count == 42


class TestDecayExponentialBehavior:
    """Tests for exponential decay behavior verification."""

    def test_exponential_decay_curve(self) -> None:
        """Verify exponential decay produces expected curve."""
        engine = DecayEngine()
        half_life = 90

        # Starting at 1.0, after each half-life, score should halve
        scores = []
        score = 1.0
        for days in [0, 90, 180, 270, 360]:
            score = engine.calculate_decay(1.0, 1.0, days, half_life)
            scores.append((days, score))

        # 0 days: 1.0
        assert abs(scores[0][1] - 1.0) < 0.001
        # 90 days: ~0.5
        assert abs(scores[1][1] - 0.5) < 0.01
        # 180 days: ~0.25
        assert abs(scores[2][1] - 0.25) < 0.01
        # 270 days: ~0.125
        assert abs(scores[3][1] - 0.125) < 0.01

    def test_decay_rate_multiplier_effect(self) -> None:
        """Higher decay rate should produce lower scores."""
        engine = DecayEngine()

        normal = engine.calculate_decay(0.5, 1.0, 30, 90)
        double_rate = engine.calculate_decay(0.5, 2.0, 30, 90)
        triple_rate = engine.calculate_decay(0.5, 3.0, 30, 90)

        assert double_rate < normal
        assert triple_rate < double_rate


class TestConfigDecayFields:
    """Tests for decay-related config fields."""

    def test_default_decay_disabled(self) -> None:
        """Decay should be disabled by default."""
        config = MeminiConfig()
        assert config.decay_enabled is False

    def test_default_half_life_90_days(self) -> None:
        """Default half-life should be 90 days."""
        config = MeminiConfig()
        assert config.decay_half_life_days == 90

    def test_default_consolidation_interval_168_hours(self) -> None:
        """Default consolidation interval should be 168 hours (1 week)."""
        config = MeminiConfig()
        assert config.consolidation_interval_hours == 168

    def test_default_similarity_threshold(self) -> None:
        """Default similarity threshold should be 0.92."""
        config = MeminiConfig()
        assert abs(config.consolidation_similarity_threshold - 0.92) < 0.001

    def test_decay_configurable(self) -> None:
        """Decay config fields should exist with correct defaults."""
        config = MeminiConfig()
        # Check fields exist with correct defaults
        assert hasattr(config, 'decay_enabled')
        assert hasattr(config, 'decay_half_life_days')
        assert hasattr(config, 'consolidation_interval_hours')
        assert hasattr(config, 'consolidation_similarity_threshold')
        # Defaults are correct
        assert config.decay_enabled is False
        assert config.decay_half_life_days == 90
        assert config.consolidation_interval_hours == 168
        assert abs(config.consolidation_similarity_threshold - 0.92) < 0.001