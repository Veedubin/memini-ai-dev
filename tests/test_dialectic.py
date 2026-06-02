"""Tests for Dialectic Engine feature (Phase 4D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.dialectic import (
    DialecticArgument,
    DialecticChallenge,
    DialecticEngine,
    DialecticHistory,
    DialecticResolution,
    DialecticSide,
    get_dialectic_engine,
)


class TestDialecticArgument:
    """Tests for DialecticArgument dataclass."""

    def test_create_argument_pro_a(self) -> None:
        """Should create a pro-A argument."""
        arg = DialecticArgument(
            memory_id="mem-a-123",
            side="pro_a",
            argument="This memory is correct because...",
            confidence=0.8,
            evidence=["evidence1", "evidence2"],
        )
        assert arg.memory_id == "mem-a-123"
        assert arg.side == "pro_a"
        assert arg.confidence == 0.8
        assert len(arg.evidence) == 2

    def test_create_argument_pro_b(self) -> None:
        """Should create a pro-B argument."""
        arg = DialecticArgument(
            memory_id="mem-b-456",
            side="pro_b",
            argument="Memory B is more reliable because...",
            confidence=0.6,
            evidence=["source corroboration"],
        )
        assert arg.memory_id == "mem-b-456"
        assert arg.side == "pro_b"
        assert arg.confidence == 0.6

    def test_default_evidence_empty(self) -> None:
        """Default evidence should be empty list."""
        arg = DialecticArgument(
            memory_id="mem-789",
            side="pro_a",
            argument="Test argument",
            confidence=0.5,
        )
        assert arg.evidence == []


class TestDialecticResolution:
    """Tests for DialecticResolution dataclass."""

    def test_create_resolution(self) -> None:
        """Should create dialectic resolution with all fields."""
        resolution = DialecticResolution(
            memory_a_id="mem-a-123",
            memory_b_id="mem-b-456",
            resolution="Memory A is preferred due to higher trust score",
            winner="A",
            reasoning="Memory A has trust 0.8 vs Memory B at 0.3",
            confidence=0.75,
        )
        assert resolution.memory_a_id == "mem-a-123"
        assert resolution.memory_b_id == "mem-b-456"
        assert resolution.winner == "A"
        assert resolution.confidence == 0.75

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        resolution = DialecticResolution(
            memory_a_id="mem-a",
            memory_b_id="mem-b",
        )
        assert resolution.pro_arguments == []
        assert resolution.con_arguments == []
        assert resolution.resolution == ""
        assert resolution.winner == "inconclusive"
        assert resolution.reasoning == ""
        assert resolution.confidence == 0.0

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        resolution = DialecticResolution(
            memory_a_id="mem-a-123",
            memory_b_id="mem-b-456",
            resolution="Test resolution",
            winner="A",
            confidence=0.7,
        )
        d = resolution.to_dict()
        assert d["memory_a_id"] == "mem-a-123"
        assert d["memory_b_id"] == "mem-b-456"
        assert d["winner"] == "A"
        assert "timestamp" in d


class TestDialecticChallenge:
    """Tests for DialecticChallenge dataclass."""

    def test_create_challenge(self) -> None:
        """Should create dialectic challenge."""
        challenge = DialecticChallenge(
            memory_id="mem-123",
            challenge_text="This memory seems incorrect because...",
            response="Good point, but consider that...",
            confidence_delta=-0.1,
        )
        assert challenge.memory_id == "mem-123"
        assert challenge.challenge_text == "This memory seems incorrect because..."
        assert challenge.response == "Good point, but consider that..."
        assert challenge.confidence_delta == -0.1

    def test_challenge_without_response(self) -> None:
        """Challenge can be created without response."""
        challenge = DialecticChallenge(
            memory_id="mem-456",
            challenge_text="Test challenge",
        )
        assert challenge.response is None
        assert challenge.confidence_delta == 0.0

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        challenge = DialecticChallenge(
            memory_id="mem-789",
            challenge_text="Challenge text",
            response="Response text",
            confidence_delta=-0.05,
        )
        d = challenge.to_dict()
        assert d["memory_id"] == "mem-789"
        assert d["challenge_text"] == "Challenge text"
        assert d["response"] == "Response text"
        assert d["confidence_delta"] == -0.05


class TestDialecticHistory:
    """Tests for DialecticHistory dataclass."""

    def test_create_history(self) -> None:
        """Should create empty dialectic history."""
        history = DialecticHistory(memory_id="mem-123")
        assert history.memory_id == "mem-123"
        assert history.notes == []
        assert history.challenges == []
        assert history.resolutions == []

    def test_add_note(self) -> None:
        """Should add dialectic note."""
        history = DialecticHistory(memory_id="mem-123")
        history.add_note("Initial analysis: memory seems reliable")
        history.add_note("Follow-up: user confirmed correctness")
        assert len(history.notes) == 2

    def test_add_challenge(self) -> None:
        """Should add challenge."""
        history = DialecticHistory(memory_id="mem-123")
        challenge = DialecticChallenge(
            memory_id="mem-123",
            challenge_text="Test challenge",
        )
        history.add_challenge(challenge)
        assert len(history.challenges) == 1

    def test_add_resolution(self) -> None:
        """Should add resolution."""
        history = DialecticHistory(memory_id="mem-123")
        resolution = DialecticResolution(
            memory_a_id="mem-a",
            memory_b_id="mem-b",
        )
        history.add_resolution(resolution)
        assert len(history.resolutions) == 1

    def test_to_dict(self) -> None:
        """Should serialize to dictionary."""
        history = DialecticHistory(memory_id="mem-123")
        history.add_note("Test note")
        d = history.to_dict()
        assert d["memory_id"] == "mem-123"
        assert d["notes"] == ["Test note"]
        assert d["challenges"] == []
        assert d["resolutions"] == []


class TestDialecticSide:
    """Tests for DialecticSide enum."""

    def test_pro_a_value(self) -> None:
        """Pro_A should have correct value."""
        assert DialecticSide.PRO_A.value == "pro_a"

    def test_pro_b_value(self) -> None:
        """Pro_B should have correct value."""
        assert DialecticSide.PRO_B.value == "pro_b"


class TestDialecticEngine:
    """Tests for DialecticEngine class."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        mock = MagicMock()
        mock.list_memories = AsyncMock(return_value=[])
        mock.get_memory = AsyncMock(return_value=None)
        return mock

    @pytest.fixture
    def engine(self, mock_memory_system: MagicMock) -> DialecticEngine:
        """Create dialectic engine with mock."""
        return DialecticEngine(memory_system=mock_memory_system)

    def test_is_enabled_disabled_by_default(self, engine: DialecticEngine) -> None:
        """Should be disabled by default."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = False
            mock_config.return_value.memory_graph_enabled = True
            mock_config.return_value.trust_engine_enabled = True
            mock_config.return_value.dialectic_llm_provider = "ollama"
            mock_config.return_value.dialectic_llm_model = "llama3"
            mock_config.return_value.llm_url = "http://localhost:11434/api/generate"
            assert engine.is_enabled is False

    def test_llm_provider_property(self, engine: DialecticEngine) -> None:
        """Should return LLM provider from config."""
        with patch.object(engine, "_config") as mock_config:
            mock_config.dialectic_llm_provider = "openai"
            assert engine.llm_provider == "openai"

    def test_llm_model_property(self, engine: DialecticEngine) -> None:
        """Should return LLM model from config."""
        with patch.object(engine, "_config") as mock_config:
            mock_config.dialectic_llm_model = "gpt-4"
            assert engine.llm_model == "gpt-4"


class TestDialecticEngineFindContradictions:
    """Tests for find_contradictions method."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system with memories."""
        from memini_ai.memory.schema import MemoryEntry, Relationship, RelationshipType

        # Create memories with CONTRADICTS relationship
        memory_a = MagicMock(spec=MemoryEntry)
        memory_a.id = "mem-a-123"
        memory_a.text = "Memory A says X is true"
        memory_a.trust_score = 0.8
        memory_a.relationships = [
            Relationship(
                target_id="mem-b-456",
                relationship_type=RelationshipType.CONTRADICTS,
                confidence=0.9,
            )
        ]

        memory_b = MagicMock(spec=MemoryEntry)
        memory_b.id = "mem-b-456"
        memory_b.text = "Memory B says X is false"
        memory_b.trust_score = 0.3

        mock = MagicMock()
        mock.list_memories = AsyncMock(return_value=[memory_a, memory_b])
        mock.get_memory = AsyncMock(return_value=memory_b)
        return mock

    @pytest.mark.asyncio
    async def test_find_contradictions_disabled(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Should return error when disabled."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = False
            mock_config.return_value.memory_graph_enabled = True
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.find_contradictions("test query")
            assert result[0].get("error") == "Dialectic engine disabled"

    @pytest.mark.asyncio
    async def test_find_contradictions_no_memory_graph(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Should return error when memory graph disabled."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            mock_config.return_value.memory_graph_enabled = False
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.find_contradictions("test query")
            assert "Memory graph disabled" in result[0].get("error", "")


class TestDialecticEngineResolve:
    """Tests for resolve_contradiction method."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        mock = MagicMock()
        mock.get_memory = AsyncMock(return_value=None)
        mock.query_memories = AsyncMock(return_value=[])
        mock.add_memory = AsyncMock(return_value="new-mem-id")
        mock.delete_memory = AsyncMock(return_value=True)
        return mock

    @pytest.mark.asyncio
    async def test_resolve_contradiction_disabled(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Should return None when disabled."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = False
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.resolve_contradiction("mem-a", "mem-b")
            assert result is None

    @pytest.mark.asyncio
    async def test_resolve_contradiction_no_memory_system(self) -> None:
        """Should return None when no memory system."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            engine = DialecticEngine(memory_system=None)
            result = await engine.resolve_contradiction("mem-a", "mem-b")
            assert result is None


class TestDialecticEngineChallenge:
    """Tests for challenge_memory method."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        from memini_ai.memory.schema import MemoryEntry

        memory = MagicMock(spec=MemoryEntry)
        memory.id = "mem-123"
        memory.text = "Test memory content"

        mock = MagicMock()
        mock.get_memory = AsyncMock(return_value=memory)
        mock.query_memories = AsyncMock(return_value=[])
        mock.add_memory = AsyncMock(return_value="new-mem-id")
        mock.delete_memory = AsyncMock(return_value=True)
        return mock

    @pytest.mark.asyncio
    async def test_challenge_disabled(self, mock_memory_system: MagicMock) -> None:
        """Should return None when disabled."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = False
            mock_config.return_value.trust_engine_enabled = False
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.challenge_memory("mem-123", "Test challenge")
            assert result is None

    @pytest.mark.asyncio
    async def test_challenge_memory_not_found(
        self, mock_memory_system: MagicMock
    ) -> None:
        """Should return None when memory not found."""
        mock_memory_system.get_memory = AsyncMock(return_value=None)
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            mock_config.return_value.trust_engine_enabled = False
            mock_config.return_value.dialectic_llm_provider = "ollama"
            mock_config.return_value.dialectic_llm_model = "llama3"
            mock_config.return_value.llm_url = "http://localhost:11434/api/generate"
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.challenge_memory("nonexistent", "Test challenge")
            assert result is None


class TestDialecticEngineHistory:
    """Tests for get_dialectic_history method."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        mock = MagicMock()
        mock.query_memories = AsyncMock(return_value=[])
        return mock

    @pytest.mark.asyncio
    async def test_history_disabled(self, mock_memory_system: MagicMock) -> None:
        """Should return None when disabled."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = False
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.get_dialectic_history("mem-123")
            assert result is None

    @pytest.mark.asyncio
    async def test_history_no_memory_system(self) -> None:
        """Should return None when no memory system."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            engine = DialecticEngine(memory_system=None)
            result = await engine.get_dialectic_history("mem-123")
            assert result is None

    @pytest.mark.asyncio
    async def test_history_empty(self, mock_memory_system: MagicMock) -> None:
        """Should return empty history when none stored."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            engine = DialecticEngine(memory_system=mock_memory_system)
            result = await engine.get_dialectic_history("mem-123")
            assert result is not None
            assert result["memory_id"] == "mem-123"


class TestGetDialecticEngine:
    """Tests for get_dialectic_engine singleton function."""

    def test_creates_singleton(self) -> None:
        """Should create singleton on first call."""
        # Reset module-level singleton
        import memini_ai.dialectic as dialectic_module

        dialectic_module._dialectic_engine = None

        engine1 = get_dialectic_engine(None)
        engine2 = get_dialectic_engine(None)
        assert engine1 is engine2

    def test_returns_same_instance(self) -> None:
        """Should return same instance on subsequent calls."""
        import memini_ai.dialectic as dialectic_module

        dialectic_module._dialectic_engine = None

        engine1 = get_dialectic_engine(None)
        engine2 = get_dialectic_engine(None)
        assert engine1 is engine2


class TestDialecticEngineClose:
    """Tests for close method."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_close_http_client(self, mock_memory_system: MagicMock) -> None:
        """Should close HTTP client."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            mock_config.return_value.dialectic_llm_provider = "ollama"
            mock_config.return_value.dialectic_llm_model = "llama3"
            mock_config.return_value.llm_url = "http://localhost:11434/api/generate"

            engine = DialecticEngine(memory_system=mock_memory_system)
            mock_client = AsyncMock()
            engine._http_client = mock_client

            await engine.close()
            assert engine._http_client is None


class TestDialecticArgumentsRoundTrip:
    """Tests for argument round-trip serialization."""

    def test_resolution_roundtrip(self) -> None:
        """Resolution should survive to_dict and creation."""
        original = DialecticResolution(
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            pro_arguments=[
                DialecticArgument(
                    memory_id="mem-a",
                    side="pro_a",
                    argument="A is correct",
                    confidence=0.7,
                    evidence=["source1"],
                )
            ],
            con_arguments=[
                DialecticArgument(
                    memory_id="mem-b",
                    side="pro_b",
                    argument="B is correct",
                    confidence=0.6,
                    evidence=["source2"],
                )
            ],
            resolution="A wins",
            winner="A",
            confidence=0.65,
        )

        d = original.to_dict()
        assert d["memory_a_id"] == "mem-a"
        assert d["memory_b_id"] == "mem-b"
        assert len(d["pro_arguments"]) == 1
        assert len(d["con_arguments"]) == 1
        assert d["winner"] == "A"

    def test_challenge_roundtrip(self) -> None:
        """Challenge should survive to_dict and creation."""
        original = DialecticChallenge(
            memory_id="mem-123",
            challenge_text="Test challenge",
            response="Test response",
            confidence_delta=-0.1,
        )

        d = original.to_dict()
        assert d["memory_id"] == "mem-123"
        assert d["challenge_text"] == "Test challenge"
        assert d["response"] == "Test response"
        assert d["confidence_delta"] == -0.1


class TestDialecticEngineLLMCalls:
    """Tests for LLM call functionality."""

    @pytest.fixture
    def mock_memory_system(self) -> MagicMock:
        """Create mock memory system."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_call_llm_failure(self) -> None:
        """Should return None on LLM failure."""
        with patch("memini_ai.dialectic.get_config") as mock_config:
            mock_config.return_value.dialectic_enabled = True
            mock_config.return_value.dialectic_llm_provider = "ollama"
            mock_config.return_value.dialectic_llm_model = "llama3"
            mock_config.return_value.llm_url = "http://localhost:11434/api/generate"

            engine = DialecticEngine(memory_system=None)
            engine._http_client = AsyncMock()
            engine._http_client.post = AsyncMock(side_effect=Exception("Network error"))

            result = await engine._call_llm("Test prompt")
            assert result is None


class TestDialecticAutoThreshold:
    """Tests for auto-threshold configuration."""

    def test_auto_threshold_default(self) -> None:
        """Default auto threshold should be 0.5."""
        from memini_ai.config import MeminiConfig

        config = MeminiConfig()
        assert config.dialectic_auto_threshold == 0.5

    def test_auto_threshold_valid_value(self) -> None:
        """Auto threshold should accept valid values."""
        from memini_ai.config import MeminiConfig

        config = MeminiConfig()
        config.dialectic_auto_threshold = 0.7
        assert config.dialectic_auto_threshold == 0.7
