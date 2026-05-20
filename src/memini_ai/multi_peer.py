"""Multi-Peer Profiles - Support for multiple users with memory sharing.

Phase 4C extends Phase 3C User Modeling to support multiple peers with
shared memory capabilities. This enables collaborative memory scenarios
where multiple users can share memories with configurable permissions.

Features:
- Peer profiles with roles and trust levels
- Memory sharing with permissions (PRIVATE, SHARED, INHERITED)
- Peer context switching for multi-user scenarios
- Peer discovery and registration
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem

from memini_ai.config import get_config
from memini_ai.memory.schema import MemorySourceType, SearchFilter, SearchOptions
from memini_ai.utils.logger import logger

# Storage constants
PEER_COLLECTION_NAME = "peer_profiles"
PEER_MEMORY_TAG = "peer_profile"
SHARING_MEMORY_TAG = "memory_sharing"

# Default trust level for new peers
DEFAULT_PEER_TRUST_LEVEL = 0.5


class PeerRole(str, Enum):
    """Role of a peer in the system."""

    OWNER = "owner"  # Primary user (project owner)
    COLLABORATOR = "collaborator"  # Full read/write access
    READONLY = "readonly"  # Can only read shared memories
    GUEST = "guest"  # Limited access, can only see explicitly shared


class MemoryPermission(str, Enum):
    """Permission level for shared memories."""

    PRIVATE = "private"  # Only owner can see
    SHARED = "shared"  # Explicitly shared peers can see
    INHERITED = "inherited"  # Inherited from a shared collection


@dataclass
class PeerProfile:
    """Profile for a peer (user) in the multi-peer system.

    Tracks peer identity, role, trust level, shared collections,
    and preferences for memory sharing.
    """

    peer_id: str
    name: str
    role: PeerRole = PeerRole.GUEST
    trust_level: float = 0.5  # 0.0-1.0
    shared_collections: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "peer_id": self.peer_id,
            "name": self.name,
            "role": self.role.value,
            "trust_level": self.trust_level,
            "shared_collections": self.shared_collections,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerProfile:
        """Create PeerProfile from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        last_active = data.get("last_active")
        if isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)

        role = data.get("role", PeerRole.GUEST)
        if isinstance(role, str):
            role = PeerRole(role)

        return cls(
            peer_id=data["peer_id"],
            name=data.get("name", "Unknown"),
            role=role,
            trust_level=data.get("trust_level", 0.5),
            shared_collections=data.get("shared_collections", []),
            preferences=data.get("preferences", {}),
            created_at=created_at,
            last_active=last_active,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PeerCollection:
    """A shared collection of memories belonging to a peer."""

    collection_id: str
    name: str
    owner_peer_id: str
    shared_with: list[str] = field(default_factory=list)  # peer_ids with access
    permission: MemoryPermission = MemoryPermission.PRIVATE
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "owner_peer_id": self.owner_peer_id,
            "shared_with": self.shared_with,
            "permission": self.permission.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MemorySharing:
    """Record of a memory being shared with another peer."""

    memory_id: str
    owner_peer_id: str
    target_peer_id: str
    permission: MemoryPermission
    shared_at: datetime = field(default_factory=datetime.utcnow)
    shared_by: str | None = None  # peer_id who initiated the share

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "owner_peer_id": self.owner_peer_id,
            "target_peer_id": self.target_peer_id,
            "permission": self.permission.value,
            "shared_at": self.shared_at.isoformat(),
            "shared_by": self.shared_by,
        }


class MultiPeerManager:
    """Manager for multi-peer identities and memory sharing.

    Handles:
    - Peer profile storage and retrieval
    - Memory sharing with permission checks
    - Peer context management for multi-user scenarios
    - Trust level propagation between peers

    Requires user_modeling_enabled to be true for full functionality.
    When disabled, returns appropriate error messages.
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
    ) -> None:
        """Initialize multi-peer manager.

        Args:
            memory_system: Optional MemorySystem instance for storage.
        """
        self._memory_system = memory_system
        self._config = get_config()
        self._enabled: bool | None = None
        self._current_peer_id: str | None = None
        # In-memory cache of peer profiles
        self._peer_cache: dict[str, PeerProfile] = {}
        # Context stack for nested context switching
        self._context_stack: list[str] = []

    @property
    def is_enabled(self) -> bool:
        """Check if multi-peer features are enabled."""
        if self._enabled is None:
            # Multi-peer requires user modeling to be enabled
            self._enabled = (
                self._config.multi_peer_enabled and self._config.user_modeling_enabled
            )
        return self._enabled

    @property
    def current_peer_id(self) -> str | None:
        """Get the current active peer ID."""
        return self._current_peer_id

    @property
    def is_context_switched(self) -> bool:
        """Check if we're in a non-default peer context."""
        return self._current_peer_id is not None

    # =============================================================================
    # Peer Management
    # =============================================================================

    async def list_peers(self) -> dict[str, Any]:
        """List all known peers.

        Returns:
            Dictionary with count and list of peer profiles.
        """
        if not self.is_enabled:
            return {
                "error": "Multi-peer disabled. Enable with MULTI_PEER_ENABLED=true and USER_MODELING=true",
                "peers": [],
                "count": 0,
            }

        if self._memory_system is None:
            return {"error": "Memory system not initialized", "peers": [], "count": 0}

        try:
            # Search for all peer profile entries
            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=100, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"peer_profile {self._config.effective_project_id}", options
            )

            peers = []
            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("profile_tag") == PEER_MEMORY_TAG:
                            peer_data = json.loads(entry.text)
                            peer = PeerProfile.from_dict(peer_data)
                            # Update cache
                            self._peer_cache[peer.peer_id] = peer
                            peers.append(peer.to_dict())
                    except (json.JSONDecodeError, KeyError):
                        continue

            return {
                "count": len(peers),
                "peers": peers,
            }
        except Exception:
            logger.warning("multi_peer_list_peers_failed", error=str(Exception))
            return {"error": str(Exception), "peers": [], "count": 0}

    async def get_peer(self, peer_id: str) -> dict[str, Any]:
        """Get a specific peer profile.

        Args:
            peer_id: ID of the peer to retrieve.

        Returns:
            Dictionary with peer profile or error.
        """
        if not self.is_enabled:
            return {
                "error": "Multi-peer disabled. Enable with MULTI_PEER_ENABLED=true and USER_MODELING=true",
            }

        # Check cache first
        if peer_id in self._peer_cache:
            return {"peer": self._peer_cache[peer_id].to_dict()}

        if self._memory_system is None:
            return {"error": "Memory system not initialized"}

        try:
            # Search for specific peer profile
            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=20, filter=filter_opts)

            results = await self._memory_system.query_memories(
                f"peer_profile {peer_id}", options
            )

            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("profile_tag") == PEER_MEMORY_TAG:
                            peer_data = json.loads(entry.text)
                            peer = PeerProfile.from_dict(peer_data)
                            self._peer_cache[peer.peer_id] = peer
                            return {"peer": peer.to_dict()}
                    except (json.JSONDecodeError, KeyError):
                        continue

            return {"error": f"Peer not found: {peer_id}"}
        except Exception:
            logger.warning("multi_peer_get_peer_failed", error=str(Exception))
            return {"error": str(Exception)}

    async def add_peer(
        self,
        peer_id: str,
        name: str,
        role: str = "guest",
        trust_level: float = 0.5,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new peer.

        Args:
            peer_id: Unique identifier for the peer.
            name: Display name for the peer.
            role: Peer role - "owner", "collaborator", "readonly", "guest" (default "guest").
            trust_level: Trust level 0.0-1.0 (default 0.5).
            preferences: Optional preferences dictionary.

        Returns:
            Dictionary with success status and peer info.
        """
        if not self.is_enabled:
            return {
                "success": False,
                "error": "Multi-peer disabled. Enable with MULTI_PEER_ENABLED=true and USER_MODELING=true",
            }

        if self._memory_system is None:
            return {"success": False, "error": "Memory system not initialized"}

        # Parse role
        try:
            peer_role = PeerRole(role.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid role: {role}. Must be one of: owner, collaborator, readonly, guest",
            }

        # Clamp trust level
        trust_level = max(0.0, min(1.0, trust_level))

        try:
            # Create peer profile
            peer = PeerProfile(
                peer_id=peer_id,
                name=name,
                role=peer_role,
                trust_level=trust_level,
                preferences=preferences or {},
                created_at=datetime.utcnow(),
            )

            # Serialize peer
            peer_json = json.dumps(peer.to_dict())

            # Create metadata
            metadata = {
                "profile_tag": PEER_MEMORY_TAG,
                "peer_id": peer_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Create memory entry for peer profile
            from memini_ai.memory.schema import MemoryEntry

            entry = MemoryEntry(
                text=peer_json,
                sourceType=MemorySourceType.project,
                metadataJson=json.dumps(metadata),
            )

            # Add to memory
            memory_id = await self._memory_system.add_memory(entry)

            # Update cache
            self._peer_cache[peer_id] = peer

            logger.info(
                "multi_peer_added", peer_id=peer_id, name=name, role=peer_role.value
            )

            return {
                "success": True,
                "peer_id": peer_id,
                "memory_id": memory_id,
                "peer": peer.to_dict(),
            }
        except Exception as e:
            logger.warning("multi_peer_add_peer_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def update_peer(
        self,
        peer_id: str,
        name: str | None = None,
        role: str | None = None,
        trust_level: float | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing peer profile.

        Args:
            peer_id: ID of the peer to update.
            name: New display name (optional).
            role: New role - "owner", "collaborator", "readonly", "guest" (optional).
            trust_level: New trust level 0.0-1.0 (optional).
            preferences: New preferences dictionary (optional).

        Returns:
            Dictionary with success status and updated peer.
        """
        if not self.is_enabled:
            return {
                "success": False,
                "error": "Multi-peer disabled",
            }

        # Get existing peer
        peer = self._peer_cache.get(peer_id)
        if peer is None:
            result = await self.get_peer(peer_id)
            if "error" in result:
                return {"success": False, "error": result["error"]}
            peer = PeerProfile.from_dict(result["peer"])

        # Update fields
        if name is not None:
            peer.name = name
        if role is not None:
            peer.role = PeerRole(role.lower())
        if trust_level is not None:
            peer.trust_level = max(0.0, min(1.0, trust_level))
        if preferences is not None:
            peer.preferences = preferences
        peer.last_active = datetime.utcnow()

        # Save updated peer
        if self._memory_system is None:
            return {"success": False, "error": "Memory system not initialized"}

        try:
            peer_json = json.dumps(peer.to_dict())
            metadata = {
                "profile_tag": PEER_MEMORY_TAG,
                "peer_id": peer_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            from memini_ai.memory.schema import MemoryEntry

            entry = MemoryEntry(
                text=peer_json,
                sourceType=MemorySourceType.project,
                metadataJson=json.dumps(metadata),
            )

            memory_id = await self._memory_system.add_memory(entry)
            self._peer_cache[peer_id] = peer

            return {
                "success": True,
                "peer_id": peer_id,
                "memory_id": memory_id,
                "peer": peer.to_dict(),
            }
        except Exception:
            logger.warning("multi_peer_update_peer_failed", error=str(Exception))
            return {"success": False, "error": str(Exception)}

    # =============================================================================
    # Peer Context Management
    # =============================================================================

    async def switch_peer_context(self, peer_id: str | None = None) -> dict[str, Any]:
        """Switch the active peer context.

        When peer_id is provided, sets it as the current context. When None,
        switches back to the default (owner) context.

        Args:
            peer_id: Peer ID to switch to, or None for default context.

        Returns:
            Dictionary with success status and context info.
        """
        if not self.is_enabled:
            return {
                "success": False,
                "error": "Multi-peer disabled",
            }

        if peer_id is not None:
            # Verify peer exists
            result = await self.get_peer(peer_id)
            if "error" in result:
                return {"success": False, "error": result["error"]}

            # Push current context to stack if different
            if self._current_peer_id is not None and self._current_peer_id != peer_id:
                self._context_stack.append(self._current_peer_id)

            self._current_peer_id = peer_id
            logger.info("multi_peer_context_switched", peer_id=peer_id)

            return {
                "success": True,
                "peer_id": peer_id,
                "previous_peer_id": self._context_stack[-1]
                if self._context_stack
                else None,
                "stack_depth": len(self._context_stack),
            }
        else:
            # Reset to default context
            previous = self._current_peer_id
            self._current_peer_id = None
            self._context_stack.clear()
            logger.info("multi_peer_context_reset", previous_peer_id=previous)

            return {
                "success": True,
                "peer_id": None,
                "previous_peer_id": previous,
                "message": "Switched to default context",
            }

    async def get_current_context(self) -> dict[str, Any]:
        """Get the current peer context.

        Returns:
            Dictionary with current peer context info.
        """
        if not self.is_enabled:
            return {
                "is_enabled": False,
                "current_peer_id": None,
                "is_default": True,
            }

        return {
            "is_enabled": True,
            "current_peer_id": self._current_peer_id,
            "is_default": self._current_peer_id is None,
            "stack_depth": len(self._context_stack),
        }

    # =============================================================================
    # Memory Sharing
    # =============================================================================

    async def share_memory(
        self,
        memory_id: str,
        target_peer_id: str,
        permission: str = "shared",
    ) -> dict[str, Any]:
        """Share a memory with another peer.

        Args:
            memory_id: ID of the memory to share.
            target_peer_id: ID of the peer to share with.
            permission: Permission level - "shared", "inherited" (default "shared").

        Returns:
            Dictionary with success status and sharing details.
        """
        if not self.is_enabled:
            return {
                "success": False,
                "error": "Multi-peer disabled",
            }

        # Verify target peer exists
        peer_result = await self.get_peer(target_peer_id)
        if "error" in peer_result:
            return {
                "success": False,
                "error": f"Target peer not found: {target_peer_id}",
            }

        # Parse permission
        try:
            perm = MemoryPermission(permission.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid permission: {permission}. Must be 'shared' or 'inherited'",
            }

        # Determine owner peer_id
        owner_peer_id = self._current_peer_id or self._config.effective_project_id

        if self._memory_system is None:
            return {"success": False, "error": "Memory system not initialized"}

        try:
            # Create sharing record
            sharing = MemorySharing(
                memory_id=memory_id,
                owner_peer_id=owner_peer_id,
                target_peer_id=target_peer_id,
                permission=perm,
                shared_at=datetime.utcnow(),
                shared_by=owner_peer_id,
            )

            # Store sharing as metadata on the memory entry
            # We update the memory's metadata_json with sharing info
            memory = await self._memory_system.get_memory(memory_id)
            if memory is None:
                return {"success": False, "error": f"Memory not found: {memory_id}"}

            # Parse existing metadata
            metadata = {}
            if memory.metadata_json:
                with contextlib.suppress(json.JSONDecodeError):
                    metadata = json.loads(memory.metadata_json)

            # Add sharing info
            if "sharing" not in metadata:
                metadata["sharing"] = []
            metadata["sharing"].append(sharing.to_dict())
            metadata["peer_id"] = owner_peer_id  # Mark owner

            # Update memory via set_payload
            await self._memory_system.set_payload(
                memory_id, {"metadataJson": json.dumps(metadata)}
            )

            logger.info(
                "multi_peer_memory_shared",
                memory_id=memory_id,
                target_peer_id=target_peer_id,
                permission=permission,
            )

            return {
                "success": True,
                "memory_id": memory_id,
                "target_peer_id": target_peer_id,
                "permission": perm.value,
                "sharing": sharing.to_dict(),
            }
        except Exception as e:
            logger.warning("multi_peer_share_memory_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def get_peer_memories(
        self,
        peer_id: str,
        query: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Query another peer's memories (if we have access).

        Args:
            peer_id: ID of the peer whose memories to query.
            query: Search query string.
            limit: Maximum number of results (default 10).

        Returns:
            Dictionary with count and list of memories.
        """
        if not self.is_enabled:
            return {
                "error": "Multi-peer disabled",
                "memories": [],
                "count": 0,
            }

        # Verify peer exists
        peer_result = await self.get_peer(peer_id)
        if "error" in peer_result:
            return {"error": peer_result["error"], "memories": [], "count": 0}

        if self._memory_system is None:
            return {
                "error": "Memory system not initialized",
                "memories": [],
                "count": 0,
            }

        # Check if current user has access to this peer's memories
        current_peer = self._current_peer_id or self._config.effective_project_id
        if current_peer != peer_id:
            # Check if we're in the target peer's shared_with list
            peer = PeerProfile.from_dict(peer_result["peer"])
            if current_peer not in peer.shared_collections:
                return {
                    "error": f"No access to peer {peer_id}'s memories",
                    "memories": [],
                    "count": 0,
                }

        try:
            # Search memories with peer_id filter
            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=limit, filter=filter_opts)

            search_query = f"peer:{peer_id}"
            if query:
                search_query = f"{search_query} {query}"

            results = await self._memory_system.query_memories(search_query, options)

            # Filter to only memories shared with current user
            memories = []
            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        if metadata.get("peer_id") == peer_id:
                            # Check if shared with current user
                            sharing = metadata.get("sharing", [])
                            current_user = (
                                self._current_peer_id
                                or self._config.effective_project_id
                            )
                            has_access = any(
                                s.get("target_peer_id") == current_user
                                or s.get("owner_peer_id") == peer_id
                                for s in sharing
                            )
                            if has_access:
                                entry_dict = entry.model_dump(by_alias=True)
                                if entry_dict.get("timestamp"):
                                    entry_dict["timestamp"] = (
                                        entry.timestamp.isoformat()
                                    )
                                memories.append(entry_dict)
                    except json.JSONDecodeError:
                        continue

            return {
                "count": len(memories),
                "memories": memories,
                "peer_id": peer_id,
            }
        except Exception:
            logger.warning("multi_peer_get_peer_memories_failed", error=str(Exception))
            return {"error": str(Exception), "memories": [], "count": 0}

    async def get_shared_memories(self, limit: int = 20) -> dict[str, Any]:
        """Get all memories shared with the current peer context.

        Returns:
            Dictionary with count and list of shared memories.
        """
        if not self.is_enabled:
            return {
                "error": "Multi-peer disabled",
                "memories": [],
                "count": 0,
            }

        current_peer = self._current_peer_id or self._config.effective_project_id

        if self._memory_system is None:
            return {
                "error": "Memory system not initialized",
                "memories": [],
                "count": 0,
            }

        try:
            # Search for all memories with sharing metadata
            filter_opts = SearchFilter(sourceType=MemorySourceType.project)
            options = SearchOptions(topK=100, filter=filter_opts)

            results = await self._memory_system.query_memories("shared memory", options)

            memories = []
            for entry in results:
                if entry.metadata_json:
                    try:
                        metadata = json.loads(entry.metadata_json)
                        sharing = metadata.get("sharing", [])

                        # Check if shared with current peer
                        for s in sharing:
                            if s.get("target_peer_id") == current_peer:
                                entry_dict = entry.model_dump(by_alias=True)
                                if entry_dict.get("timestamp"):
                                    entry_dict["timestamp"] = (
                                        entry.timestamp.isoformat()
                                    )
                                entry_dict["sharing_permission"] = s.get("permission")
                                entry_dict["shared_by"] = s.get("shared_by")
                                entry_dict["owner_peer_id"] = s.get("owner_peer_id")
                                memories.append(entry_dict)
                                break
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Limit results
            memories = memories[:limit]

            return {
                "count": len(memories),
                "memories": memories,
                "current_peer_id": current_peer,
            }
        except Exception:
            logger.warning(
                "multi_peer_get_shared_memories_failed", error=str(Exception)
            )
            return {"error": str(Exception), "memories": [], "count": 0}

    async def revoke_sharing(
        self,
        memory_id: str,
        target_peer_id: str,
    ) -> dict[str, Any]:
        """Revoke memory sharing from a peer.

        Args:
            memory_id: ID of the memory.
            target_peer_id: ID of the peer to revoke access from.

        Returns:
            Dictionary with success status.
        """
        if not self.is_enabled:
            return {"success": False, "error": "Multi-peer disabled"}

        if self._memory_system is None:
            return {"success": False, "error": "Memory system not initialized"}

        try:
            memory = await self._memory_system.get_memory(memory_id)
            if memory is None:
                return {"success": False, "error": f"Memory not found: {memory_id}"}

            # Parse and update metadata
            metadata = {}
            if memory.metadata_json:
                with contextlib.suppress(json.JSONDecodeError):
                    metadata = json.loads(memory.metadata_json)

            # Remove sharing for target peer
            sharing = metadata.get("sharing", [])
            metadata["sharing"] = [
                s for s in sharing if s.get("target_peer_id") != target_peer_id
            ]

            # Update memory via set_payload
            await self._memory_system.set_payload(
                memory_id, {"metadataJson": json.dumps(metadata)}
            )

            logger.info(
                "multi_peer_sharing_revoked",
                memory_id=memory_id,
                target_peer_id=target_peer_id,
            )

            return {
                "success": True,
                "memory_id": memory_id,
                "target_peer_id": target_peer_id,
            }
        except Exception:
            logger.warning("multi_peer_revoke_sharing_failed", error=str(Exception))
            return {"success": False, "error": str(Exception)}


# Module-level singleton
_multi_peer_manager: MultiPeerManager | None = None


def get_multi_peer_manager(
    memory_system: MemorySystem | None = None,
) -> MultiPeerManager:
    """Get or create the global MultiPeerManager instance.

    Args:
        memory_system: Optional MemorySystem to use.

    Returns:
        MultiPeerManager instance.
    """
    global _multi_peer_manager
    if _multi_peer_manager is None:
        _multi_peer_manager = MultiPeerManager(memory_system=memory_system)
    return _multi_peer_manager
