"""Tests for Multi-Peer Profiles feature (Phase 4C)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memini_ai.memory.schema import MemorySourceType, UserProfile
from memini_ai.multi_peer import (
    DEFAULT_PEER_TRUST_LEVEL,
    MemoryPermission,
    MemorySharing,
    MultiPeerManager,
    PeerCollection,
    PeerProfile,
    PeerRole,
    get_multi_peer_manager,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_peer(**overrides) -> PeerProfile:
    """Create a valid PeerProfile for testing."""
    defaults = {
        "peer_id": "peer-123",
        "name": "Test Peer",
        "role": PeerRole.GUEST,
        "trust_level": 0.5,
        "shared_collections": [],
        "preferences": {},
        "created_at": datetime.utcnow(),
        "last_active": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return PeerProfile(**defaults)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config_enabled() -> MagicMock:
    """Create a mock config with multi-peer enabled."""
    config = MagicMock()
    config.multi_peer_enabled = True
    config.user_modeling_enabled = True
    config.effective_project_id = "test-project-123"
    return config


@pytest.fixture
def mock_config_disabled() -> MagicMock:
    """Create a mock config with multi-peer disabled."""
    config = MagicMock()
    config.multi_peer_enabled = False
    config.user_modeling_enabled = False
    config.effective_project_id = "test-project-123"
    return config


@pytest.fixture
def mock_config_user_modeling_only() -> MagicMock:
    """Create a mock config with user_modeling but multi_peer disabled."""
    config = MagicMock()
    config.multi_peer_enabled = False
    config.user_modeling_enabled = True
    config.effective_project_id = "test-project-123"
    return config


@pytest.fixture
def mock_memory_system() -> MagicMock:
    """Create a mock MemorySystem."""
    system = MagicMock()
    system.query_memories = AsyncMock(return_value=[])
    system.add_memory = AsyncMock(return_value="new-memory-id")
    system.delete_memory = AsyncMock(return_value=True)
    system.get_memory = AsyncMock(return_value=None)
    system.update_memory = AsyncMock(return_value=True)
    system.is_ready = True
    return system


@pytest.fixture
def mock_peer_entry() -> MagicMock:
    """Create a mock memory entry containing a peer profile."""
    entry = MagicMock()
    entry.id = "peer-entry-id"
    entry.text = '{"peer_id": "peer-456", "name": "Alice", "role": "collaborator", "trust_level": 0.7, "shared_collections": [], "preferences": {}, "created_at": "2026-01-01T00:00:00", "last_active": null, "metadata": {}}'
    entry.metadata_json = '{"profile_tag": "peer_profile", "peer_id": "peer-456", "updated_at": "2026-01-01T00:00:00"}'
    return entry


# =============================================================================
# PeerRole Enum Tests
# =============================================================================


class TestPeerRole:
    """Tests for PeerRole enum."""

    def test_peer_role_values(self) -> None:
        """PeerRole has correct values."""
        assert PeerRole.OWNER.value == "owner"
        assert PeerRole.COLLABORATOR.value == "collaborator"
        assert PeerRole.READONLY.value == "readonly"
        assert PeerRole.GUEST.value == "guest"

    def test_peer_role_from_string(self) -> None:
        """Can create PeerRole from string."""
        assert PeerRole("owner") == PeerRole.OWNER
        assert PeerRole("collaborator") == PeerRole.COLLABORATOR
        assert PeerRole("readonly") == PeerRole.READONLY
        assert PeerRole("guest") == PeerRole.GUEST


# =============================================================================
# MemoryPermission Enum Tests
# =============================================================================


class TestMemoryPermission:
    """Tests for MemoryPermission enum."""

    def test_memory_permission_values(self) -> None:
        """MemoryPermission has correct values."""
        assert MemoryPermission.PRIVATE.value == "private"
        assert MemoryPermission.SHARED.value == "shared"
        assert MemoryPermission.INHERITED.value == "inherited"

    def test_memory_permission_from_string(self) -> None:
        """Can create MemoryPermission from string."""
        assert MemoryPermission("private") == MemoryPermission.PRIVATE
        assert MemoryPermission("shared") == MemoryPermission.SHARED
        assert MemoryPermission("inherited") == MemoryPermission.INHERITED


# =============================================================================
# PeerProfile Dataclass Tests
# =============================================================================


class TestPeerProfileDataclass:
    """Tests for PeerProfile dataclass."""

    def test_create_peer_profile_defaults(self) -> None:
        """Should create PeerProfile with default values."""
        peer = PeerProfile(
            peer_id="peer-123",
            name="Test",
        )
        assert peer.peer_id == "peer-123"
        assert peer.name == "Test"
        assert peer.role == PeerRole.GUEST
        assert peer.trust_level == 0.5
        assert peer.shared_collections == []
        assert peer.preferences == {}
        assert peer.created_at is not None
        assert peer.last_active is None
        assert peer.metadata == {}

    def test_create_peer_profile_with_values(self) -> None:
        """Should create PeerProfile with custom values."""
        now = datetime.utcnow()
        peer = PeerProfile(
            peer_id="peer-456",
            name="Alice",
            role=PeerRole.COLLABORATOR,
            trust_level=0.8,
            shared_collections=["collab-1"],
            preferences={"format": "markdown"},
            created_at=now,
            last_active=now,
        )
        assert peer.peer_id == "peer-456"
        assert peer.name == "Alice"
        assert peer.role == PeerRole.COLLABORATOR
        assert peer.trust_level == 0.8
        assert peer.shared_collections == ["collab-1"]
        assert peer.preferences == {"format": "markdown"}

    def test_to_dict(self) -> None:
        """Should convert peer profile to dictionary."""
        peer = create_test_peer(
            peer_id="peer-789",
            name="Bob",
            role=PeerRole.OWNER,
            trust_level=0.9,
        )
        result = peer.to_dict()
        assert result["peer_id"] == "peer-789"
        assert result["name"] == "Bob"
        assert result["role"] == "owner"
        assert result["trust_level"] == 0.9

    def test_from_dict(self) -> None:
        """Should create PeerProfile from dictionary."""
        data = {
            "peer_id": "peer-abc",
            "name": "Charlie",
            "role": "readonly",
            "trust_level": 0.6,
            "shared_collections": ["collab-2"],
            "preferences": {"tone": "formal"},
            "created_at": "2026-05-01T12:00:00",
            "last_active": "2026-05-15T08:30:00",
            "metadata": {"notes": "test"},
        }
        peer = PeerProfile.from_dict(data)
        assert peer.peer_id == "peer-abc"
        assert peer.name == "Charlie"
        assert peer.role == PeerRole.READONLY
        assert peer.trust_level == 0.6
        assert peer.shared_collections == ["collab-2"]
        assert peer.preferences == {"tone": "formal"}
        assert peer.metadata == {"notes": "test"}

    def test_from_dict_handles_missing_optional_fields(self) -> None:
        """Should use defaults for missing optional fields."""
        data = {"peer_id": "minimal-peer"}
        peer = PeerProfile.from_dict(data)
        assert peer.peer_id == "minimal-peer"
        assert peer.name == "Unknown"
        assert peer.role == PeerRole.GUEST
        assert peer.trust_level == 0.5


# =============================================================================
# MemorySharing Dataclass Tests
# =============================================================================


class TestMemorySharing:
    """Tests for MemorySharing dataclass."""

    def test_create_memory_sharing(self) -> None:
        """Should create MemorySharing with default values."""
        sharing = MemorySharing(
            memory_id="mem-123",
            owner_peer_id="owner-456",
            target_peer_id="target-789",
            permission=MemoryPermission.SHARED,
        )
        assert sharing.memory_id == "mem-123"
        assert sharing.owner_peer_id == "owner-456"
        assert sharing.target_peer_id == "target-789"
        assert sharing.permission == MemoryPermission.SHARED
        assert sharing.shared_at is not None
        assert sharing.shared_by is None

    def test_memory_sharing_to_dict(self) -> None:
        """Should convert MemorySharing to dictionary."""
        sharing = MemorySharing(
            memory_id="mem-abc",
            owner_peer_id="owner-def",
            target_peer_id="target-ghi",
            permission=MemoryPermission.INHERITED,
            shared_by="owner-def",
        )
        result = sharing.to_dict()
        assert result["memory_id"] == "mem-abc"
        assert result["owner_peer_id"] == "owner-def"
        assert result["target_peer_id"] == "target-ghi"
        assert result["permission"] == "inherited"
        assert result["shared_by"] == "owner-def"


# =============================================================================
# PeerCollection Dataclass Tests
# =============================================================================


class TestPeerCollection:
    """Tests for PeerCollection dataclass."""

    def test_create_peer_collection(self) -> None:
        """Should create PeerCollection with default values."""
        collection = PeerCollection(
            collection_id="col-123",
            name="Test Collection",
            owner_peer_id="owner-456",
        )
        assert collection.collection_id == "col-123"
        assert collection.name == "Test Collection"
        assert collection.owner_peer_id == "owner-456"
        assert collection.shared_with == []
        assert collection.permission == MemoryPermission.PRIVATE
        assert collection.created_at is not None

    def test_peer_collection_to_dict(self) -> None:
        """Should convert PeerCollection to dictionary."""
        collection = PeerCollection(
            collection_id="col-abc",
            name="Shared Notes",
            owner_peer_id="owner-def",
            shared_with=["peer-1", "peer-2"],
            permission=MemoryPermission.SHARED,
        )
        result = collection.to_dict()
        assert result["collection_id"] == "col-abc"
        assert result["name"] == "Shared Notes"
        assert result["shared_with"] == ["peer-1", "peer-2"]
        assert result["permission"] == "shared"


# =============================================================================
# is_enabled Property Tests
# =============================================================================


class TestIsEnabled:
    """Tests for is_enabled property."""

    @pytest.mark.asyncio
    async def test_is_enabled_true(self, mock_config_enabled: MagicMock) -> None:
        """is_enabled returns True when both config flags are set."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            assert manager.is_enabled is True

    @pytest.mark.asyncio
    async def test_is_enabled_false_multi_peer_disabled(
        self, mock_config_user_modeling_only: MagicMock
    ) -> None:
        """is_enabled returns False when multi_peer_enabled is False."""
        with patch(
            "memini_ai.multi_peer.get_config", return_value=mock_config_user_modeling_only
        ):
            manager = MultiPeerManager()
            assert manager.is_enabled is False

    @pytest.mark.asyncio
    async def test_is_enabled_false_user_modeling_disabled(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """is_enabled returns False when user_modeling_enabled is False."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            assert manager.is_enabled is False


# =============================================================================
# Peer Management Tests
# =============================================================================


class TestListPeers:
    """Tests for list_peers method."""

    @pytest.mark.asyncio
    async def test_list_peers_disabled(self, mock_config_disabled: MagicMock) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.list_peers()
            assert "error" in result
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_peers_no_memory_system(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error when no memory system."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            result = await manager.list_peers()
            assert "error" in result

    @pytest.mark.asyncio
    async def test_list_peers_with_entries(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_peer_entry: MagicMock,
    ) -> None:
        """Returns peer list when entries found."""
        mock_memory_system.query_memories = AsyncMock(return_value=[mock_peer_entry])

        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            result = await manager.list_peers()

            assert result["count"] == 1
            assert len(result["peers"]) == 1
            assert result["peers"][0]["peer_id"] == "peer-456"


class TestAddPeer:
    """Tests for add_peer method."""

    @pytest.mark.asyncio
    async def test_add_peer_disabled(self, mock_config_disabled: MagicMock) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.add_peer(
                peer_id="new-peer",
                name="New Peer",
            )
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_add_peer_no_memory_system(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error when no memory system."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            result = await manager.add_peer(
                peer_id="new-peer",
                name="New Peer",
            )
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_add_peer_success(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Successfully adds a new peer."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            result = await manager.add_peer(
                peer_id="new-peer",
                name="New Peer",
                role="collaborator",
                trust_level=0.7,
            )

            assert result["success"] is True
            assert result["peer_id"] == "new-peer"
            assert result["peer"]["name"] == "New Peer"
            assert result["peer"]["role"] == "collaborator"

    @pytest.mark.asyncio
    async def test_add_peer_invalid_role(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns error for invalid role."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            result = await manager.add_peer(
                peer_id="new-peer",
                name="New Peer",
                role="invalid_role",
            )

            assert result["success"] is False
            assert "Invalid role" in result["error"]


# =============================================================================
# Peer Context Management Tests
# =============================================================================


class TestSwitchPeerContext:
    """Tests for switch_peer_context method."""

    @pytest.mark.asyncio
    async def test_switch_peer_context_disabled(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.switch_peer_context("peer-123")
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_switch_peer_context_peer_not_found(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error when peer not found."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            manager._peer_cache = {}  # Empty cache

            # Mock get_peer to return not found
            manager.get_peer = AsyncMock(return_value={"error": "Peer not found"})

            result = await manager.switch_peer_context("nonexistent-peer")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_switch_peer_context_success(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
        mock_peer_entry: MagicMock,
    ) -> None:
        """Successfully switches peer context."""
        mock_memory_system.query_memories = AsyncMock(return_value=[mock_peer_entry])

        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)

            # First add the peer to cache
            manager._peer_cache["peer-456"] = create_test_peer(peer_id="peer-456")

            result = await manager.switch_peer_context("peer-456")

            assert result["success"] is True
            assert result["peer_id"] == "peer-456"

    @pytest.mark.asyncio
    async def test_switch_peer_context_reset(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Successfully resets context to default."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            manager._current_peer_id = "peer-123"

            result = await manager.switch_peer_context(None)

            assert result["success"] is True
            assert result["peer_id"] is None
            assert "Switched to default context" in result["message"]


class TestGetCurrentContext:
    """Tests for get_current_context method."""

    @pytest.mark.asyncio
    async def test_get_current_context_disabled(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """Returns default context when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.get_current_context()

            assert result["is_enabled"] is False
            assert result["current_peer_id"] is None
            assert result["is_default"] is True

    @pytest.mark.asyncio
    async def test_get_current_context_active(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns current context when active."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            manager._current_peer_id = "peer-123"

            result = await manager.get_current_context()

            assert result["is_enabled"] is True
            assert result["current_peer_id"] == "peer-123"
            assert result["is_default"] is False


# =============================================================================
# Memory Sharing Tests
# =============================================================================


class TestShareMemory:
    """Tests for share_memory method."""

    @pytest.mark.asyncio
    async def test_share_memory_disabled(self, mock_config_disabled: MagicMock) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.share_memory(
                memory_id="mem-123",
                target_peer_id="peer-456",
            )
            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_share_memory_peer_not_found(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """Returns error when peer not found."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            manager.get_peer = AsyncMock(return_value={"error": "Peer not found"})

            result = await manager.share_memory(
                memory_id="mem-123",
                target_peer_id="nonexistent-peer",
            )
            assert result["success"] is False
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_share_memory_invalid_permission(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Returns error for invalid permission."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            manager._peer_cache["peer-456"] = create_test_peer(peer_id="peer-456")

            result = await manager.share_memory(
                memory_id="mem-123",
                target_peer_id="peer-456",
                permission="invalid",
            )
            assert result["success"] is False
            assert "Invalid permission" in result["error"]

    @pytest.mark.asyncio
    async def test_share_memory_success(
        self,
        mock_config_enabled: MagicMock,
        mock_memory_system: MagicMock,
    ) -> None:
        """Successfully shares memory."""
        mock_memory = MagicMock()
        mock_memory.metadata_json = None
        mock_memory_system.get_memory = AsyncMock(return_value=mock_memory)

        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager(memory_system=mock_memory_system)
            manager._peer_cache["peer-456"] = create_test_peer(peer_id="peer-456")

            result = await manager.share_memory(
                memory_id="mem-123",
                target_peer_id="peer-456",
                permission="shared",
            )

            assert result["success"] is True
            assert result["memory_id"] == "mem-123"
            assert result["target_peer_id"] == "peer-456"


# =============================================================================
# Get Peer Memories Tests
# =============================================================================


class TestGetPeerMemories:
    """Tests for get_peer_memories method."""

    @pytest.mark.asyncio
    async def test_get_peer_memories_disabled(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.get_peer_memories("peer-456")
            assert "error" in result
            assert result["count"] == 0


# =============================================================================
# Get Shared Memories Tests
# =============================================================================


class TestGetSharedMemories:
    """Tests for get_shared_memories method."""

    @pytest.mark.asyncio
    async def test_get_shared_memories_disabled(
        self, mock_config_disabled: MagicMock
    ) -> None:
        """Returns error when disabled."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_disabled):
            manager = MultiPeerManager()
            result = await manager.get_shared_memories()
            assert "error" in result
            assert result["count"] == 0


# =============================================================================
# current_peer_id Property Tests
# =============================================================================


class TestCurrentPeerId:
    """Tests for current_peer_id property."""

    def test_current_peer_id_initially_none(self, mock_config_enabled: MagicMock) -> None:
        """current_peer_id is None initially."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            assert manager.current_peer_id is None

    def test_current_peer_id_after_switch(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """current_peer_id reflects context switch."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            manager._current_peer_id = "peer-123"
            assert manager.current_peer_id == "peer-123"


# =============================================================================
# is_context_switched Property Tests
# =============================================================================


class TestIsContextSwitched:
    """Tests for is_context_switched property."""

    def test_not_context_switched_initially(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """is_context_switched is False initially."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            assert manager.is_context_switched is False

    def test_context_switched_after_switch(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """is_context_switched is True after context switch."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            manager = MultiPeerManager()
            manager._current_peer_id = "peer-123"
            assert manager.is_context_switched is True


# =============================================================================
# Module-Level Singleton Tests
# =============================================================================


class TestModuleLevelSingleton:
    """Tests for module-level get_multi_peer_manager singleton."""

    def test_get_multi_peer_manager_creates_instance(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """get_multi_peer_manager creates instance when called first time."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            from memini_ai.multi_peer import _multi_peer_manager

            # Reset singleton
            import memini_ai.multi_peer

            memini_ai.multi_peer._multi_peer_manager = None

            result = get_multi_peer_manager(None)
            assert result is not None
            assert isinstance(result, MultiPeerManager)

    def test_get_multi_peer_manager_returns_same_instance(
        self, mock_config_enabled: MagicMock
    ) -> None:
        """get_multi_peer_manager returns same instance on subsequent calls."""
        with patch("memini_ai.multi_peer.get_config", return_value=mock_config_enabled):
            from memini_ai.multi_peer import _multi_peer_manager

            # Reset singleton
            import memini_ai.multi_peer

            memini_ai.multi_peer._multi_peer_manager = None

            first = get_multi_peer_manager(None)
            second = get_multi_peer_manager(None)
            assert first is second