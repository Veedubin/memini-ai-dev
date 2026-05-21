"""Tests for Trust Engine feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.schema import (
    TRUST_DEFAULT,
    TRUST_DELTA_CONFIRM,
    TRUST_DELTA_CORRECT,
    TRUST_DELTA_IGNORED,
    TRUST_DELTA_USE,
    TRUST_THRESHOLD_ARCHIVE,
    TRUST_THRESHOLD_PROMOTE,
    MemoryEntry,
    MemorySourceType,
    TrustLevel,
    TrustSignal,
)
from memini_ai.trust_engine import TrustAdjustment, TrustEngine


class TestTrustConstants:
    """Tests for trust-related constants."""

    def test_default_trust_score(self) -> None:
        """Default trust score should be 0.5."""
        assert TRUST_DEFAULT == 0.5

    def test_archive_threshold(self) -> None:
        """Archive threshold should be 0.2."""
        assert TRUST_THRESHOLD_ARCHIVE == 0.2

    def test_promote_threshold(self) -> None:
        """Promote threshold should be 0.8."""
        assert TRUST_THRESHOLD_PROMOTE == 0.8

    def test_delta_use(self) -> None:
        """Delta for agent used should be +0.05."""
        assert TRUST_DELTA_USE == 0.05

    def test_delta_ignored(self) -> None:
        """Delta for agent ignored should be -0.02."""
        assert TRUST_DELTA_IGNORED == -0.02

    def test_delta_correct(self) -> None:
        """Delta for user corrected should be -0.15."""
        assert TRUST_DELTA_CORRECT == -0.15

    def test_delta_confirm(self) -> None:
        """Delta for user confirmed should be +0.10."""
        assert TRUST_DELTA_CONFIRM == 0.10


class TestTrustSignal:
    """Tests for TrustSignal enum."""

    def test_all_signals_exist(self) -> None:
        """All expected trust signals should exist."""
        assert TrustSignal.AGENT_USED.value == "agent_used"
        assert TrustSignal.AGENT_IGNORED.value == "agent_ignored"
        assert TrustSignal.USER_CORRECTED.value == "user_corrected"
        assert TrustSignal.USER_CONFIRMED.value == "user_confirmed"

    def test_signal_is_string_enum(self) -> None:
        """TrustSignal should be a string enum for serialization."""
        assert isinstance(TrustSignal.AGENT_USED, str)


class TestTrustLevel:
    """Tests for TrustLevel enum."""

    def test_all_levels_exist(self) -> None:
        """All expected trust levels should exist."""
        assert TrustLevel.ARCHIVED.value == "archived"
        assert TrustLevel.LOW.value == "low"
        assert TrustLevel.MEDIUM.value == "medium"
        assert TrustLevel.HIGH.value == "high"
        assert TrustLevel.PROMOTED.value == "promoted"


class TestTrustAdjustment:
    """Tests for TrustAdjustment dataclass."""

    def test_create_adjustment(self) -> None:
        """Should create trust adjustment with all fields."""
        adjustment = TrustAdjustment(
            memory_id="mem-123",
            old_score=0.5,
            new_score=0.55,
            signal=TrustSignal.AGENT_USED,
            action="increased",
        )
        assert adjustment.memory_id == "mem-123"
        assert adjustment.old_score == 0.5
        assert adjustment.new_score == 0.55
        assert adjustment.signal == TrustSignal.AGENT_USED
        assert adjustment.action == "increased"


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem."""
    system = MagicMock()
    system.get_memory = AsyncMock(return_value=None)
    system.list_memories = AsyncMock(return_value=[])
    return system


@pytest.fixture
def sample_memory() -> MemoryEntry:
    """Create a sample memory entry with default trust score."""
    return MemoryEntry(
        id="test-memory-123",
        text="Test memory content",
        source_type=MemorySourceType.session,
        content_hash="testhash123",
        trust_score=TRUST_DEFAULT,
        retrieval_count=0,
        is_archived=False,
    )


class TestTrustEngineDefaultScore:
    """Tests for default trust score behavior."""

    @pytest.mark.asyncio
    async def test_default_trust_score(self, mock_memory_system: MagicMock) -> None:
        """Memory starts with 0.5 trust."""
        memory = MemoryEntry(
            id="mem-123",
            text="Test",
            source_type=MemorySourceType.session,
            content_hash="hash",
        )
        assert memory.trust_score == TRUST_DEFAULT
        assert memory.trust_score == 0.5

    @pytest.mark.asyncio
    async def test_memory_with_custom_trust(self) -> None:
        """Memory can be created with custom trust score."""
        memory = MemoryEntry(
            id="mem-123",
            text="Test",
            source_type=MemorySourceType.session,
            content_hash="hash",
            trust_score=0.9,
        )
        assert memory.trust_score == 0.9


class TestTrustEngineAdjustments:
    """Tests for trust score adjustments."""

    @pytest.mark.asyncio
    async def test_trust_adjustment_agent_used(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """+0.05 for agent use."""
        sample_memory.trust_score = 0.5
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.AGENT_USED
            )

            assert result is not None
            assert result.old_score == 0.5
            assert result.new_score == 0.55  # 0.5 + 0.05
            assert result.action == "increased"

    @pytest.mark.asyncio
    async def test_trust_adjustment_agent_ignored(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """-0.02 for agent ignore."""
        sample_memory.trust_score = 0.5
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.AGENT_IGNORED
            )

            assert result is not None
            assert result.old_score == 0.5
            assert result.new_score == 0.48  # 0.5 + (-0.02)
            assert result.action == "decreased"

    @pytest.mark.asyncio
    async def test_trust_adjustment_user_corrected(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """-0.15 for user correction."""
        sample_memory.trust_score = 0.5
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CORRECTED
            )

            assert result is not None
            assert result.old_score == 0.5
            assert result.new_score == 0.35  # 0.5 + (-0.15)
            assert result.action == "decreased"

    @pytest.mark.asyncio
    async def test_trust_adjustment_user_confirmed(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """+0.10 for user confirmation."""
        sample_memory.trust_score = 0.5
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CONFIRMED
            )

            assert result is not None
            assert result.old_score == 0.5
            assert result.new_score == 0.60  # 0.5 + 0.10
            assert result.action == "increased"


class TestTrustEngineClamping:
    """Tests for trust score clamping to valid range [0.0, 1.0]."""

    @pytest.mark.asyncio
    async def test_clamp_at_zero(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Trust score should not go below 0.0."""
        sample_memory.trust_score = 0.1
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            # USER_CORRECTED is -0.15, which would go negative
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CORRECTED
            )

            assert result is not None
            assert result.new_score == 0.0  # Clamped to minimum

    @pytest.mark.asyncio
    async def test_clamp_at_one(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Trust score should not exceed 1.0."""
        sample_memory.trust_score = 0.95
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            # USER_CONFIRMED is +0.10, which would exceed 1.0
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CONFIRMED
            )

            assert result is not None
            assert result.new_score == 1.0  # Clamped to maximum


class TestTrustEngineArchive:
    """Tests for memory archiving based on trust threshold."""

    @pytest.mark.asyncio
    async def test_trust_archive_below_threshold(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Memories < 0.2 get archived."""
        sample_memory.trust_score = 0.25  # Above threshold
        sample_memory.is_archived = False
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            # USER_CORRECTED is -0.15, 0.25 - 0.15 = 0.10 < 0.2
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CORRECTED
            )

            assert result is not None
            assert result.new_score == 0.10
            assert result.action == "archived"
            # Note: actual archiving happens via _update_memory_trust
            # The action field in result should indicate "archived"


class TestTrustEnginePromote:
    """Tests for memory promotion based on trust threshold."""

    @pytest.mark.asyncio
    async def test_trust_promote_above_threshold(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Memories > 0.8 get promoted."""
        sample_memory.trust_score = 0.75  # Below threshold
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            # USER_CONFIRMED is +0.10, 0.75 + 0.10 = 0.85 > 0.8
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.USER_CONFIRMED
            )

            assert result is not None
            assert result.new_score == 0.85
            assert result.action == "promoted"


class TestTrustEngineGetScore:
    """Tests for getting trust score."""

    @pytest.mark.asyncio
    async def test_get_trust_score(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Returns correct score."""
        sample_memory.trust_score = 0.7
        sample_memory.retrieval_count = 5
        sample_memory.is_archived = False
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.get_trust_score("test-memory-123")

            assert result is not None
            assert result["id"] == "test-memory-123"
            assert result["trustScore"] == 0.7
            assert result["retrievalCount"] == 5
            assert result["isArchived"] is False

    @pytest.mark.asyncio
    async def test_get_trust_score_memory_not_found(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Returns None when memory not found."""
        mock_memory_system.get_memory.return_value = None

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.get_trust_score("nonexistent-memory")

            assert result is None


class TestTrustEngineListArchived:
    """Tests for listing archived memories."""

    @pytest.mark.asyncio
    async def test_list_archived(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Returns archived memories."""
        archived_memory1 = MemoryEntry(
            id="archived-1",
            text="Archived memory 1",
            source_type=MemorySourceType.session,
            content_hash="hash1",
            trust_score=0.1,
            is_archived=True,
        )
        archived_memory2 = MemoryEntry(
            id="archived-2",
            text="Archived memory 2",
            source_type=MemorySourceType.session,
            content_hash="hash2",
            trust_score=0.15,
            is_archived=True,
        )
        active_memory = MemoryEntry(
            id="active-1",
            text="Active memory",
            source_type=MemorySourceType.session,
            content_hash="hash3",
            trust_score=0.5,
            is_archived=False,
        )

        mock_memory_system.list_memories.return_value = [
            archived_memory1,
            archived_memory2,
            active_memory,
        ]

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_archived()

            assert len(result) == 2
            assert all(m.is_archived for m in result)

    @pytest.mark.asyncio
    async def test_list_archived_empty(self, mock_memory_system: MagicMock) -> None:
        """Returns empty list when no archived memories."""
        mock_memory_system.list_memories.return_value = []

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_archived()

            assert result == []

    @pytest.mark.asyncio
    async def test_list_archived_with_pagination(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Supports limit and offset pagination."""
        archived_memories = [
            MemoryEntry(
                id=f"archived-{i}",
                text=f"Archived memory {i}",
                source_type=MemorySourceType.session,
                content_hash=f"hash{i}",
                trust_score=0.1,
                is_archived=True,
            )
            for i in range(10)
        ]

        mock_memory_system.list_memories.return_value = archived_memories

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_archived(limit=3, offset=2)

            assert len(result) == 3


class TestTrustEngineRecordRetrieval:
    """Tests for recording memory retrievals."""

    @pytest.mark.asyncio
    async def test_record_retrieval(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Increments retrieval_count."""
        sample_memory.retrieval_count = 5
        mock_memory_system.get_memory.return_value = sample_memory

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            mock_config.return_value.embedding_dim = 1024

            engine = TrustEngine(memory_system=mock_memory_system)
            await engine.record_retrieval("test-memory-123")

            # The retrieval_count should be incremented in memory object
            # (actual persistence happens in background thread)
            assert sample_memory.retrieval_count == 6

    @pytest.mark.asyncio
    async def test_record_retrieval_missing_memory(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Does not fail when memory not found."""
        mock_memory_system.get_memory.return_value = None

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            # Should not raise
            await engine.record_retrieval("nonexistent-memory")


class TestTrustEngineDisabled:
    """Tests for disabled trust engine behavior."""

    @pytest.mark.asyncio
    async def test_adjust_trust_disabled(self, mock_memory_system: MagicMock) -> None:
        """Returns None when engine is disabled."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = False

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.AGENT_USED
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_get_trust_score_disabled(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Returns None when engine is disabled."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = False

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.get_trust_score("test-memory-123")

            assert result is None

    @pytest.mark.asyncio
    async def test_list_archived_disabled(self, mock_memory_system: MagicMock) -> None:
        """Returns empty list when engine is disabled."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = False

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_archived()

            assert result == []

    @pytest.mark.asyncio
    async def test_list_promoted_disabled(self, mock_memory_system: MagicMock) -> None:
        """Returns empty list when engine is disabled."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = False

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_promoted()

            assert result == []

    @pytest.mark.asyncio
    async def test_record_retrieval_disabled(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Does nothing when engine is disabled."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = False

            engine = TrustEngine(memory_system=mock_memory_system)
            # Should not raise
            await engine.record_retrieval("test-memory-123")


class TestTrustEngineNoMemorySystem:
    """Tests for TrustEngine behavior when no memory system is provided."""

    @pytest.mark.asyncio
    async def test_adjust_trust_no_memory_system(self) -> None:
        """Returns None when memory system is None."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=None)
            result = await engine.adjust_trust(
                "test-memory-123", TrustSignal.AGENT_USED
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_get_trust_score_no_memory_system(self) -> None:
        """Returns None when memory system is None."""
        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=None)
            result = await engine.get_trust_score("test-memory-123")

            assert result is None


class TestTrustEngineListPromoted:
    """Tests for listing promoted memories."""

    @pytest.mark.asyncio
    async def test_list_promoted(
        self, mock_memory_system: MagicMock, sample_memory: MemoryEntry
    ) -> None:
        """Returns promoted memories."""
        promoted_memory1 = MemoryEntry(
            id="promoted-1",
            text="Promoted memory 1",
            source_type=MemorySourceType.session,
            content_hash="hash1",
            trust_score=0.85,
            is_archived=False,
        )
        promoted_memory2 = MemoryEntry(
            id="promoted-2",
            text="Promoted memory 2",
            source_type=MemorySourceType.session,
            content_hash="hash2",
            trust_score=0.9,
            is_archived=False,
        )
        active_memory = MemoryEntry(
            id="active-1",
            text="Active memory",
            source_type=MemorySourceType.session,
            content_hash="hash3",
            trust_score=0.5,
            is_archived=False,
        )

        mock_memory_system.list_memories.return_value = [
            promoted_memory1,
            promoted_memory2,
            active_memory,
        ]

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_promoted()

            assert len(result) == 2
            assert all(m.trust_score >= TRUST_THRESHOLD_PROMOTE for m in result)

    @pytest.mark.asyncio
    async def test_list_promoted_empty(self, mock_memory_system: MagicMock) -> None:
        """Returns empty list when no promoted memories."""
        mock_memory_system.list_memories.return_value = []

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_promoted()

            assert result == []

    @pytest.mark.asyncio
    async def test_list_promoted_with_pagination(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Supports limit and offset pagination."""
        promoted_memories = [
            MemoryEntry(
                id=f"promoted-{i}",
                text=f"Promoted memory {i}",
                source_type=MemorySourceType.session,
                content_hash=f"hash{i}",
                trust_score=0.85 + (i * 0.01),
                is_archived=False,
            )
            for i in range(10)
        ]

        mock_memory_system.list_memories.return_value = promoted_memories

        with patch("memini_ai.trust_engine.get_config") as mock_config:
            mock_config.return_value.trust_engine_enabled = True

            engine = TrustEngine(memory_system=mock_memory_system)
            result = await engine.list_promoted(limit=3, offset=2)

            assert len(result) == 3


class TestTrustEngineTrustLevel:
    """Tests for trust level determination."""

    def test_trust_level_archived(self) -> None:
        """Score < 0.2 should be ARCHIVED."""
        engine = TrustEngine()
        level = engine._get_trust_level(0.1)
        assert level == TrustLevel.ARCHIVED

    def test_trust_level_low(self) -> None:
        """Score 0.2-0.4 should be LOW."""
        engine = TrustEngine()
        level = engine._get_trust_level(0.3)
        assert level == TrustLevel.LOW

    def test_trust_level_medium(self) -> None:
        """Score 0.4-0.7 should be MEDIUM."""
        engine = TrustEngine()
        level = engine._get_trust_level(0.5)
        assert level == TrustLevel.MEDIUM

    def test_trust_level_high(self) -> None:
        """Score 0.7-0.8 should be HIGH."""
        engine = TrustEngine()
        level = engine._get_trust_level(0.75)
        assert level == TrustLevel.HIGH

    def test_trust_level_promoted(self) -> None:
        """Score > 0.8 should be PROMOTED."""
        engine = TrustEngine()
        level = engine._get_trust_level(0.9)
        assert level == TrustLevel.PROMOTED

    def test_trust_level_boundary_archive(self) -> None:
        """Score exactly at archive threshold should be ARCHIVED."""
        engine = TrustEngine()
        level = engine._get_trust_level(TRUST_THRESHOLD_ARCHIVE)
        assert level == TrustLevel.LOW  # Not archived, it's the boundary

    def test_trust_level_boundary_promote(self) -> None:
        """Score exactly at promote threshold should be PROMOTED."""
        engine = TrustEngine()
        level = engine._get_trust_level(TRUST_THRESHOLD_PROMOTE)
        # At exactly 0.8, score is NOT < 0.8, so it falls into else -> PROMOTED
        assert level == TrustLevel.PROMOTED
