"""Memory schema types - Pydantic models for memory-related data structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memini_ai.utils.hash import hash_content


class MemorySourceType(str, Enum):
    """Source type for memory entries."""

    session = "session"
    file = "file"
    web = "web"
    boomerang = "boomerang"
    project = "project"
    thought = "thought"


class TrustLevel(str, Enum):
    """Trust level classification for memory entries."""

    ARCHIVED = "archived"  # < 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.7
    HIGH = "high"  # 0.7 - 0.8
    PROMOTED = "promoted"  # > 0.8


class TrustSignal(str, Enum):
    """Feedback signals for trust adjustment."""

    AGENT_USED = "agent_used"  # Agent successfully used this memory
    AGENT_IGNORED = "agent_ignored"  # Agent ignored this memory
    USER_CORRECTED = "user_corrected"  # User corrected this memory
    USER_CONFIRMED = "user_confirmed"  # User confirmed this memory


class RelationshipType(str, Enum):
    """Types of relationships between memories."""

    SUPERSEDES = "SUPERSEDES"
    PARTIAL_UPDATE = "PARTIAL_UPDATE"
    RELATED_TO = "RELATED_TO"
    CONTRADICTS = "CONTRADICTS"
    DERIVED_FROM = "DERIVED_FROM"


# Decay engine constants
DECAY_BASE_HALF_LIFE_DAYS = 90
DECAY_MIN_RATE = 0.1
DECAY_MAX_RATE = 10.0
DECAY_DEFAULT_RATE = 1.0
FADE_THRESHOLD = 0.15

# Consolidation constants
DEFAULT_CONSOLIDATION_SIMILARITY_THRESHOLD = 0.92
MIN_CONSOLIDATION_SIMILARITY = 0.70


@dataclass
class Relationship:
    """A relationship between two memories."""

    target_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    source: str = "auto"  # "auto", "manual", "llm"

    @model_validator(mode="after")
    def clamp_confidence(self) -> Relationship:
        """Clamp confidence to valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self


# Trust engine constants
TRUST_THRESHOLD_ARCHIVE = 0.2
TRUST_THRESHOLD_PROMOTE = 0.8
TRUST_DEFAULT = 0.5
TRUST_DELTA_USE = 0.05
TRUST_DELTA_IGNORED = -0.02
TRUST_DELTA_CORRECT = -0.15
TRUST_DELTA_CONFIRM = 0.10

# User modeling constants
USER_MODEL_MIN_SESSIONS_DEFAULT = 50


@dataclass
class UserPreference:
    """A single user preference inferred or stated by the user."""

    key: str
    value: Any
    source: str = "inferred"  # "inferred", "stated", "observed"
    confidence: float = 0.5
    last_observed: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserProfile:
    """Persistent user profile with dialectic updates.

    Tracks user preferences, communication style, and expertise domains.
    Updated dialectically via LLM reasoning after each session.
    """

    user_id: str
    communication_style: str = "neutral"  # "concise", "detailed", "technical", "plain"
    expertise_domains: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # How confident we are in this profile
    last_updated: datetime = field(default_factory=datetime.utcnow)
    session_count: int = 0
    dialectic_notes: list[str] = field(default_factory=list)  # LLM reasoning traces

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "communication_style": self.communication_style,
            "expertise_domains": self.expertise_domains,
            "preferences": self.preferences,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
            "session_count": self.session_count,
            "dialectic_notes": self.dialectic_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Create UserProfile from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.utcnow()

        return cls(
            user_id=data["user_id"],
            communication_style=data.get("communication_style", "neutral"),
            expertise_domains=data.get("expertise_domains", []),
            preferences=data.get("preferences", {}),
            confidence=data.get("confidence", 0.0),
            last_updated=last_updated,
            session_count=data.get("session_count", 0),
            dialectic_notes=data.get("dialectic_notes", []),
        )


# Tiered loading constants
TIER0_DEFAULT_MAX_TOKENS = 100
TIER1_DEFAULT_MAX_TOKENS = 2000
TIER0_DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
TIER1_DEFAULT_CACHE_TTL = 7200  # 2 hours in seconds


class SummaryTier(str, Enum):
    """Summary tier levels for tiered loading.

    L0: Project summary (~100 tokens) - session start auto-inject
    L1: Key decisions (~2K tokens) - planning tasks
    L2: Full memories - on demand via query_memories
    """

    L0 = "L0"  # Project summary (~100 tokens)
    L1 = "L1"  # Key decisions (~2K tokens)


@dataclass
class TieredSummary:
    """A tiered summary generated from memories.

    Used for L0 and L1 tiered loading to provide token-efficient
    context at different granularity levels.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tier: SummaryTier = SummaryTier.L0
    content: str = ""
    source_memory_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = 0
    is_stale: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "tier": self.tier.value,
            "content": self.content,
            "source_memory_ids": self.source_memory_ids,
            "generated_at": self.generated_at.isoformat(),
            "token_count": self.token_count,
            "is_stale": self.is_stale,
        }


class MemoryEntry(BaseModel):
    """A single memory entry with vector embedding and metadata."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    vector: list[float] | None = None
    source_type: MemorySourceType = Field(alias="sourceType")
    source_path: str | None = Field(default=None, alias="sourcePath")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str = Field(
        default="",
        alias="contentHash",
        description="SHA-256 hash of content. Auto-computed if not provided.",
    )
    metadata_json: str | None = Field(default=None, alias="metadataJson")
    session_id: str | None = Field(default=None, alias="sessionId")
    project_id: str | None = Field(default=None, alias="projectId")
    score: float | None = None

    # Trust engine fields
    trust_score: float = Field(default=TRUST_DEFAULT, alias="trustScore")
    retrieval_count: int = Field(default=0, alias="retrievalCount")
    is_archived: bool = Field(default=False, alias="isArchived")

    # Memory graph fields
    relationships: list[Relationship] = Field(
        default_factory=list,
        alias="relationships",
        description="JSON list of relationships to other memories",
    )

    # Decay engine fields
    decay_rate: float = Field(
        default=DECAY_DEFAULT_RATE,
        alias="decayRate",
        description="Decay rate multiplier (1.0 = normal, higher = faster decay)",
    )
    last_accessed: datetime | None = Field(
        default=None,
        alias="lastAccessed",
        description="Last time this memory was accessed/retrieved",
    )
    access_count: int = Field(
        default=0,
        alias="accessCount",
        description="Number of times this memory was retrieved",
    )

    # Phase 4C: Multi-peer field - owner peer (null = primary user/owner context)
    peer_id: str | None = Field(
        default=None,
        alias="peerId",
        description="Owner peer ID for multi-peer memories (null = primary user)",
    )

    # Delta model fields for partial updates
    supersedes_id: str | None = Field(
        default=None,
        alias="supersedesId",
        description="ID of memory this partially updates (for PARTIAL_UPDATE relationships)",
    )

    structured_fields: dict[str, Any] | None = Field(
        default=None,
        alias="structuredFields",
        description="Key-value extracted fields for granular merge instead of full text replacement",
    )

    change_ratio: float = Field(
        default=1.0,
        alias="changeRatio",
        description="Fraction of content that is new/changed (0.0-1.0). 1.0 = full replacement, <1.0 = partial update",
    )

    created_at_ms: int = Field(
        default_factory=lambda: int(datetime.utcnow().timestamp() * 1000),
        alias="createdAtMs",
        description="Unix timestamp in milliseconds when this memory was created. Useful for temporal ordering and hierarchy.",
    )

    @model_validator(mode="after")
    def compute_content_hash_if_missing(self) -> MemoryEntry:
        """Auto-compute content_hash from text if not provided."""
        if not self.content_hash:
            self.content_hash = hash_content(self.text)
        return self


class SearchFilter(BaseModel):
    """Filter options for memory searches."""

    model_config = ConfigDict(populate_by_name=True)

    source_type: MemorySourceType | None = Field(default=None, alias="sourceType")
    session_id: str | None = Field(default=None, alias="sessionId")
    since: datetime | None = None
    project_id: str | None = Field(default=None, alias="projectId")


class SearchStrategy(str, Enum):
    """Search strategy for memory retrieval."""

    TIERED = "TIERED"
    VECTOR_ONLY = "VECTOR_ONLY"
    TEXT_ONLY = "TEXT_ONLY"
    PARALLEL = "PARALLEL"


class SearchOptions(BaseModel):
    """Options for memory search operations."""

    model_config = ConfigDict(populate_by_name=True)

    top_k: int = Field(default=5, alias="topK")
    strategy: SearchStrategy = SearchStrategy.TIERED
    threshold: float = 0.72
    filter: SearchFilter = Field(default_factory=SearchFilter)
    exact_search: bool = False


# Constants
MEMORY_TABLE_NAME = "memories"

SCHEMA_VERSION = 2
