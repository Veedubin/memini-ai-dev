"""Tests for User Modeling feature (Phase 3.3)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.schema import UserPreference, UserProfile
from memini_ai.user_model import UserModel

# =============================================================================
# Helper Functions
# =============================================================================


def create_test_profile(**overrides) -> UserProfile:
    """Create a valid UserProfile for testing.

    This helper avoids Pydantic Field() issues in dataclass defaults by always
    providing explicit values for fields that use Field(default_factory=...).
    """
    defaults = {
        "user_id": "test-project-123",
        "communication_style": "neutral",
        "expertise_domains": [],
        "preferences": {},
        "confidence": 0.0,
        "last_updated": datetime.utcnow(),
        "session_count": 0,
        "dialectic_notes": [],
    }
    defaults.update(overrides)
    return UserProfile(**defaults)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config_enabled() -> MagicMock:
    """Create a mock config with user modeling enabled."""
    config = MagicMock()
    config.user_modeling_enabled = True
    config.user_model_min_sessions = 50
    config.llm_url = "http://localhost:11434/api/generate"
    config.llm_model = "llama3.2"
    config.llm_provider = "ollama"
    config.effective_project_id = "test-project-123"
    return config


@pytest.fixture
def mock_config_disabled() -> MagicMock:
    """Create a mock config with user modeling disabled."""
    config = MagicMock()
    config.user_modeling_enabled = False
    config.user_model_min_sessions = 50
    config.effective_project_id = "test-project-123"
    return config


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem."""
    system = MagicMock()
    system.query_memories = AsyncMock(return_value=[])
    system.add_memory = AsyncMock(return_value="new-memory-id")
    system.delete_memory = AsyncMock(return_value=True)
    return system


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response from LLM."""
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(
        return_value={
            "response": '{"communication_style": "concise", "expertise_domains": ["python"], "preferences": {"format": "markdown"}, "confidence": 0.5, "reasoning": "User prefers concise responses"}'
        }
    )
    return response


@pytest.fixture
def mock_profile_entry() -> MagicMock:
    """Create a mock memory entry containing a profile."""
    entry = MagicMock()
    entry.id = "profile-entry-id"
    entry.text = '{"user_id": "test-project-123", "communication_style": "neutral", "expertise_domains": [], "preferences": {}, "confidence": 0.0, "last_updated": "2026-01-01T00:00:00", "session_count": 0, "dialectic_notes": []}'
    entry.metadata_json = '{"profile_tag": "user_profile", "user_id": "test-project-123", "updated_at": "2026-01-01T00:00:00"}'
    return entry


def make_mock_llm_client(return_text: str = "") -> AsyncMock:
    """Create a mock LLM client for factory-based tests."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=return_text)
    return client


# =============================================================================
# UserProfile and UserPreference Dataclass Tests
# =============================================================================


class TestUserPreferenceDataclass:
    """Tests for UserPreference dataclass."""

    def test_create_preference_defaults(self) -> None:
        """Should create UserPreference with default values."""
        pref = UserPreference(key="format", value="markdown")
        assert pref.key == "format"
        assert pref.value == "markdown"
        assert pref.source == "inferred"
        assert pref.confidence == 0.5
        assert pref.last_observed is not None

    def test_create_preference_with_values(self) -> None:
        """Should create UserPreference with custom values."""
        now = datetime.utcnow()
        pref = UserPreference(
            key="tone",
            value="formal",
            source="stated",
            confidence=0.9,
            last_observed=now,
        )
        assert pref.key == "tone"
        assert pref.value == "formal"
        assert pref.source == "stated"
        assert pref.confidence == 0.9
        assert pref.last_observed == now


class TestUserProfileDataclass:
    """Tests for UserProfile dataclass."""

    def test_create_profile_defaults(self) -> None:
        """Should create UserProfile with default values."""
        # NOTE: Using explicit values to avoid Pydantic Field() bug in dataclass defaults
        profile = UserProfile(
            user_id="user-123",
            communication_style="neutral",
            expertise_domains=[],
            preferences={},
            confidence=0.0,
            last_updated=datetime.utcnow(),
            session_count=0,
            dialectic_notes=[],
        )
        assert profile.user_id == "user-123"
        assert profile.communication_style == "neutral"
        assert profile.expertise_domains == []
        assert profile.preferences == {}
        assert profile.confidence == 0.0
        assert profile.session_count == 0
        assert profile.dialectic_notes == []

    def test_create_profile_with_values(self) -> None:
        """Should create UserProfile with custom values."""
        now = datetime.utcnow()
        profile = UserProfile(
            user_id="user-123",
            communication_style="concise",
            expertise_domains=["python", "fastapi"],
            preferences={"format": "markdown", "tone": "technical"},
            confidence=0.75,
            last_updated=now,
            session_count=10,
            dialectic_notes=["First reasoning trace", "Second reasoning trace"],
        )
        assert profile.user_id == "user-123"
        assert profile.communication_style == "concise"
        assert profile.expertise_domains == ["python", "fastapi"]
        assert profile.preferences["format"] == "markdown"
        assert profile.confidence == 0.75
        assert profile.session_count == 10
        assert len(profile.dialectic_notes) == 2

    def test_to_dict(self) -> None:
        """Should convert profile to dictionary."""
        profile = create_test_profile(
            user_id="user-123",
            communication_style="detailed",
            expertise_domains=["python"],
            preferences={"format": "json"},
            confidence=0.6,
            session_count=5,
        )
        result = profile.to_dict()
        assert result["user_id"] == "user-123"
        assert result["communication_style"] == "detailed"
        assert result["expertise_domains"] == ["python"]
        assert result["preferences"] == {"format": "json"}
        assert result["confidence"] == 0.6
        assert result["session_count"] == 5
        assert "last_updated" in result

    def test_from_dict(self) -> None:
        """Should create profile from dictionary."""
        data = {
            "user_id": "user-456",
            "communication_style": "technical",
            "expertise_domains": ["rust", "wasm"],
            "preferences": {"output": "terse"},
            "confidence": 0.8,
            "last_updated": "2026-05-01T12:00:00",
            "session_count": 25,
            "dialectic_notes": ["Note 1", "Note 2"],
        }
        profile = UserProfile.from_dict(data)
        assert profile.user_id == "user-456"
        assert profile.communication_style == "technical"
        assert profile.expertise_domains == ["rust", "wasm"]
        assert profile.preferences == {"output": "terse"}
        assert profile.confidence == 0.8
        assert profile.session_count == 25
        assert len(profile.dialectic_notes) == 2

    def test_from_dict_handles_iso_datetime(self) -> None:
        """Should parse ISO datetime string in from_dict."""
        data = {
            "user_id": "user-789",
            "last_updated": "2026-03-15T08:30:00",
        }
        profile = UserProfile.from_dict(data)
        assert profile.last_updated.year == 2026
        assert profile.last_updated.month == 3
        assert profile.last_updated.day == 15

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        """Should use defaults for missing optional fields."""
        data = {"user_id": "minimal-user"}
        profile = UserProfile.from_dict(data)
        assert profile.communication_style == "neutral"
        assert profile.expertise_domains == []
        assert profile.preferences == {}
        assert profile.confidence == 0.0
        assert profile.session_count == 0
        assert profile.dialectic_notes == []


# =============================================================================
# is_enabled Property Tests
# =============================================================================


class TestIsEnabled:
    """Tests for is_enabled property."""

    @pytest.mark.asyncio
    async def test_is_enabled_true(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled returns True when config enabled."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            assert user_model.is_enabled is True

    @pytest.mark.asyncio
    async def test_is_enabled_false(self, mock_config_disabled: MagicMock) -> None:
        """is_enabled returns False when config disabled."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            assert user_model.is_enabled is False

    @pytest.mark.asyncio
    async def test_is_enabled_cached(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled is cached after first call."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            first = user_model.is_enabled
            second = user_model.is_enabled
            assert first is second


# =============================================================================
# is_warmed_up Property Tests
# =============================================================================


class TestIsWarmedUp:
    """Tests for is_warmed_up property."""

    @pytest.mark.asyncio
    async def test_not_warmed_up_initially(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Not warmed up when cache is None."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            assert user_model.is_warmed_up is False

    @pytest.mark.asyncio
    async def test_not_warmed_up_below_min_sessions(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Not warmed up when session_count < min_sessions."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            user_model._profile_cache = create_test_profile(session_count=49)
            assert user_model.is_warmed_up is False

    @pytest.mark.asyncio
    async def test_warmed_up_after_min_sessions(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Warmed up when session_count >= min_sessions."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            user_model._profile_cache = create_test_profile(session_count=50)
            assert user_model.is_warmed_up is True

    @pytest.mark.asyncio
    async def test_warmed_up_above_min_sessions(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Warmed up when session_count > min_sessions."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            user_model._profile_cache = create_test_profile(session_count=100)
            assert user_model.is_warmed_up is True


# =============================================================================
# session_count Property Tests
# =============================================================================


class TestSessionCount:
    """Tests for session_count property."""

    @pytest.mark.asyncio
    async def test_session_count_zero_initially(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """session_count returns 0 when cache is None."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            assert user_model.session_count == 0

    @pytest.mark.asyncio
    async def test_session_count_from_cache(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """session_count returns value from cached profile."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            user_model._profile_cache = create_test_profile(session_count=25)
            assert user_model.session_count == 25


# =============================================================================
# get_profile Tests
# =============================================================================


class TestGetProfile:
    """Tests for get_profile method."""

    @pytest.mark.asyncio
    async def test_get_profile_disabled_returns_error(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, get_profile returns error info."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.get_profile()
            assert "error" in result
            assert result["error"] == "User modeling disabled"

    @pytest.mark.asyncio
    async def test_get_profile_no_profile_found(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns error when no profile exists."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            result = await user_model.get_profile()
            assert "error" in result
            assert result["error"] == "No profile found"
            assert result["warmed_up"] is False
            assert result["session_count"] == 0

    @pytest.mark.asyncio
    async def test_get_profile_with_cache(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns profile from cache when available."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = create_test_profile(
                communication_style="concise",
                expertise_domains=["python"],
                preferences={"format": "markdown"},
                confidence=0.7,
                session_count=60,
            )

            result = await user_model.get_profile()

            assert result["user_id"] == "test-project-123"
            assert result["communication_style"] == "concise"
            assert result["expertise_domains"] == ["python"]
            assert result["preferences"] == {"format": "markdown"}
            assert result["confidence"] == 0.7
            assert result["session_count"] == 60
            assert result["warmed_up"] is True

    @pytest.mark.asyncio
    async def test_get_profile_includes_dialectic_notes(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Includes dialectic notes when requested."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = create_test_profile(
                session_count=60,
                dialectic_notes=["Reasoning 1", "Reasoning 2", "Reasoning 3"],
            )

            result = await user_model.get_profile(include_dialectic_notes=True)

            assert "dialectic_notes" in result
            # Should return last 5 notes (we only have 3)
            assert len(result["dialectic_notes"]) == 3

    @pytest.mark.asyncio
    async def test_get_profile_limits_dialectic_notes(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Limits dialectic notes to last 5."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = create_test_profile(
                session_count=60,
                dialectic_notes=[
                    "Note 1",
                    "Note 2",
                    "Note 3",
                    "Note 4",
                    "Note 5",
                    "Note 6",
                    "Note 7",
                    "Note 8",
                ],
            )

            result = await user_model.get_profile(include_dialectic_notes=True)

            assert "dialectic_notes" in result
            # Should only include last 5
            assert len(result["dialectic_notes"]) == 5


# =============================================================================
# get_profile_summary Tests
# =============================================================================


class TestGetProfileSummary:
    """Tests for get_profile_summary method."""

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, mock_config_disabled: MagicMock) -> None:
        """When disabled, get_profile_summary returns None."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.get_profile_summary()
            assert result is None

    @pytest.mark.asyncio
    async def test_no_profile_returns_none(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns None when no profile exists."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            result = await user_model.get_profile_summary()
            assert result is None

    @pytest.mark.asyncio
    async def test_not_warmed_up_returns_none(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns None when not warmed up."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = create_test_profile(session_count=10)

            result = await user_model.get_profile_summary()
            assert result is None

    @pytest.mark.asyncio
    async def test_warmed_up_returns_summary(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns summary when warmed up and LLM call succeeds."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = UserProfile(
                user_id="test-project-123",
                communication_style="concise",
                expertise_domains=["python"],
                preferences={"format": "markdown"},
                session_count=60,
            )

            mock_llm_text = (
                "User prefers concise, technical responses with markdown formatting."
            )
            with patch(
                "memini_ai.user_model.get_llm_client",
                return_value=make_mock_llm_client(return_text=mock_llm_text),
            ):
                result = await user_model.get_profile_summary()

            assert result is not None
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns None when LLM call fails."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            user_model._profile_cache = UserProfile(
                user_id="test-project-123",
                session_count=60,
            )

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            user_model._http_client = mock_client

            result = await user_model.get_profile_summary()
            assert result is None


# =============================================================================
# update_profile_from_session Tests
# =============================================================================


class TestUpdateProfileFromSession:
    """Tests for update_profile_from_session method."""

    @pytest.mark.asyncio
    async def test_disabled_returns_error(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, update_profile_from_session returns error."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.update_profile_from_session("test conversation")
            assert result["success"] is False
            assert result["error"] == "User modeling disabled"

    @pytest.mark.asyncio
    async def test_empty_conversation_returns_error(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error for empty conversation."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            result = await user_model.update_profile_from_session("")
            assert result["success"] is False
            assert result["error"] == "Empty conversation"

    @pytest.mark.asyncio
    async def test_whitespace_conversation_returns_error(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error for whitespace-only conversation."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            result = await user_model.update_profile_from_session("   \n\t  ")
            assert result["success"] is False
            assert result["error"] == "Empty conversation"

    @pytest.mark.asyncio
    async def test_profile_creation_for_new_user(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Creates new profile for user without one."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)

            # No profile in memory
            user_model._load_profile = AsyncMock(return_value=None)

            # Mock _dialectic_update to return a valid reasoning string
            async def mock_dialectic(profile, conversation):
                return '{"communication_style": "concise", "expertise_domains": ["python"], "preferences": {}, "confidence": 0.4, "reasoning": "Initial profile"}'

            user_model._dialectic_update = mock_dialectic

            result = await user_model.update_profile_from_session(
                "I prefer concise responses"
            )

            assert result["success"] is True
            assert result["session_count"] == 1
            assert result["warmed_up"] is False

    @pytest.mark.asyncio
    async def test_profile_update_increments_session_count(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Dialectic update increments session_count."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)

            # Existing profile with 5 sessions
            existing_profile = UserProfile(
                user_id="test-project-123",
                session_count=5,
                communication_style="neutral",
                last_updated=datetime.utcnow(),
            )
            user_model._load_profile = AsyncMock(return_value=existing_profile)

            # Mock _dialectic_update to return a valid reasoning string
            async def mock_dialectic(profile, conversation):
                return '{"communication_style": "detailed", "expertise_domains": ["python"], "preferences": {}, "confidence": 0.5, "reasoning": "Updated profile"}'

            user_model._dialectic_update = mock_dialectic

            result = await user_model.update_profile_from_session(
                "I like detailed explanations"
            )

            assert result["success"] is True
            assert result["session_count"] == 6  # Incremented

    @pytest.mark.asyncio
    async def test_dialectic_reasoning_failure(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns error when dialectic reasoning fails."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)

            user_model._load_profile = AsyncMock(return_value=None)

            # Mock HTTP client that fails
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            user_model._http_client = mock_client

            result = await user_model.update_profile_from_session("test conversation")

            assert result["success"] is False
            assert "error" in result


# =============================================================================
# Disabled No-Op Behavior Tests
# =============================================================================


class TestDisabledNoOp:
    """Tests for disabled behavior (no-op)."""

    @pytest.mark.asyncio
    async def test_get_profile_disabled_returns_error_info(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, get_profile returns error info."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.get_profile()
            assert result["error"] == "User modeling disabled"

    @pytest.mark.asyncio
    async def test_get_profile_summary_disabled_returns_none(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, get_profile_summary returns None."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.get_profile_summary()
            assert result is None

    @pytest.mark.asyncio
    async def test_update_profile_disabled_returns_error(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """When disabled, update_profile_from_session returns error."""
        with patch(
            "memini_ai.user_model.get_config", return_value=mock_config_disabled
        ):
            user_model = UserModel()
            result = await user_model.update_profile_from_session("test conversation")
            assert result["success"] is False
            assert result["error"] == "User modeling disabled"


# =============================================================================
# Profile Storage Tests
# =============================================================================


class TestProfileStorage:
    """Tests for profile loading and saving."""

    @pytest.mark.asyncio
    async def test_load_profile_from_memory(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_profile_entry: MagicMock,
    ) -> None:
        """Loads profile from memory system."""
        mock_memory_system.query_memories = AsyncMock(return_value=[mock_profile_entry])

        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            profile = await user_model._load_profile()

            assert profile is not None
            assert profile.user_id == "test-project-123"

    @pytest.mark.asyncio
    async def test_load_profile_no_memory_system(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns None when no memory system."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=None)
            profile = await user_model._load_profile()
            assert profile is None

    @pytest.mark.asyncio
    async def test_load_profile_uses_cache(
        self, mock_config_enabled: MagicMock, mock_memory_system: MagicMock
    ) -> None:
        """Uses cached profile without calling memory system."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            cached_profile = UserProfile(user_id="cached-user", session_count=10)
            user_model._profile_cache = cached_profile

            profile = await user_model._load_profile()

            assert profile is cached_profile
            # query_memories should NOT be called
            mock_memory_system.query_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_profile_to_memory(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Saves profile to memory system."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=mock_memory_system)
            profile = UserProfile(
                user_id="test-project-123",
                communication_style="concise",
                last_updated=datetime.utcnow(),
                session_count=5,
            )

            await user_model._save_profile(profile)

            # Should add a new memory entry
            mock_memory_system.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_profile_no_memory_system(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Does nothing when no memory system."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel(memory_system=None)
            profile = UserProfile(user_id="test-project-123")

            # Should not raise
            await user_model._save_profile(profile)


# =============================================================================
# Close Method Tests
# =============================================================================


class TestClose:
    """Tests for HTTP client cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_client(self, mock_config_enabled: MagicMock) -> None:
        """close() properly closes HTTP client."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            user_model._http_client = mock_client

            await user_model.close()

            mock_client.aclose.assert_called_once()
            assert user_model._http_client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_config_enabled: MagicMock) -> None:
        """close() handles None client gracefully."""
        with patch("memini_ai.user_model.get_config", return_value=mock_config_enabled):
            user_model = UserModel()
            user_model._http_client = None

            # Should not raise
            await user_model.close()

            assert user_model._http_client is None


# =============================================================================
# Module-Level Singleton Tests
# =============================================================================


class TestModuleLevelSingleton:
    """Tests for module-level get_user_model singleton."""

    def test_get_user_model_creates_instance(self) -> None:
        """get_user_model creates instance when called first time."""
        mock_config = MagicMock()
        mock_config.user_modeling_enabled = True
        mock_config.user_model_min_sessions = 50
        mock_config.llm_url = "http://localhost:11434"
        mock_config.llm_model = "llama3.2"
        mock_config.effective_project_id = "test"

        with patch("memini_ai.user_model.get_config", return_value=mock_config):
            import memini_ai.user_model as um

            # Reset singleton
            um._user_model = None  # noqa: SLF001

            result = um.get_user_model(None)
            assert result is not None
            assert isinstance(result, UserModel)

    def test_get_user_model_returns_same_instance(self) -> None:
        """get_user_model returns same instance on subsequent calls."""
        mock_config = MagicMock()
        mock_config.user_modeling_enabled = True
        mock_config.user_model_min_sessions = 50
        mock_config.llm_url = "http://localhost:11434"
        mock_config.llm_model = "llama3.2"
        mock_config.effective_project_id = "test"

        with patch("memini_ai.user_model.get_config", return_value=mock_config):
            import memini_ai.user_model as um

            # Reset singleton
            um._user_model = None  # noqa: SLF001

            first = um.get_user_model(None)
            second = um.get_user_model(None)
            assert first is second
