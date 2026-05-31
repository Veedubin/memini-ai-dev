"""Trust Engine - Memory trust scoring with feedback signals and archive/promote thresholds."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memini_ai.config import get_config
from memini_ai.memory.schema import (
    TRUST_DELTA_CONFIRM,
    TRUST_DELTA_CORRECT,
    TRUST_DELTA_IGNORED,
    TRUST_DELTA_USE,
    TRUST_THRESHOLD_ARCHIVE,
    TRUST_THRESHOLD_PROMOTE,
    MemoryEntry,
    TrustLevel,
    TrustSignal,
)

if TYPE_CHECKING:
    from memini_ai.audit.logger import AuditLogger
    from memini_ai.memory.system import MemorySystem


@dataclass
class TrustAdjustment:
    """Result of a trust adjustment operation."""

    memory_id: str
    old_score: float
    new_score: float
    signal: TrustSignal
    action: str  # "increased", "decreased", "archived", "promoted"


class TrustEngine:
    """Memory trust scoring with feedback signals.

    Trust engine manages memory reliability scores based on usage feedback.
    Memories below ARCHIVE_THRESHOLD are automatically archived.
    Memories above PROMOTE_THRESHOLD are marked as promoted.
    All features are opt-in via TRUST_ENGINE env var (default false).
    """

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize TrustEngine.

        Args:
            memory_system: Optional MemorySystem instance for DB operations.
            audit_logger: Optional AuditLogger for logging trust adjustments.
        """
        self._memory_system = memory_system
        self._audit_logger = audit_logger
        self._enabled: bool | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if trust engine is enabled via config."""
        if self._enabled is None:
            config = get_config()
            self._enabled = config.trust_engine_enabled
        return self._enabled

    def _clamp_trust(self, score: float) -> float:
        """Clamp trust score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, score))

    def _get_trust_level(self, score: float) -> TrustLevel:
        """Get trust level from score."""
        if score < TRUST_THRESHOLD_ARCHIVE:
            return TrustLevel.ARCHIVED
        elif score < 0.4:
            return TrustLevel.LOW
        elif score < 0.7:
            return TrustLevel.MEDIUM
        elif score < TRUST_THRESHOLD_PROMOTE:
            return TrustLevel.HIGH
        else:
            return TrustLevel.PROMOTED

    def _get_delta(self, signal: TrustSignal) -> float:
        """Get trust delta for a signal."""
        delta_map = {
            TrustSignal.AGENT_USED: TRUST_DELTA_USE,
            TrustSignal.AGENT_IGNORED: TRUST_DELTA_IGNORED,
            TrustSignal.USER_CORRECTED: TRUST_DELTA_CORRECT,
            TrustSignal.USER_CONFIRMED: TRUST_DELTA_CONFIRM,
        }
        return delta_map.get(signal, 0.0)

    async def adjust_trust(
        self,
        memory_id: str,
        signal: TrustSignal,
    ) -> TrustAdjustment | None:
        """Adjust trust score for a memory based on feedback signal.

        Args:
            memory_id: ID of the memory entry.
            signal: TrustSignal feedback type.

        Returns:
            TrustAdjustment with old/new scores, or None if disabled/memory not found.
        """
        if not self.is_enabled:
            return None

        if self._memory_system is None:
            return None

        # Get current memory
        memory = await self._memory_system.get_memory(memory_id)
        if memory is None:
            return None

        old_score = memory.trust_score
        delta = self._get_delta(signal)
        new_score = self._clamp_trust(old_score + delta)

        # Determine action
        if new_score < TRUST_THRESHOLD_ARCHIVE and old_score >= TRUST_THRESHOLD_ARCHIVE:
            action = "archived"
        elif (
            new_score >= TRUST_THRESHOLD_PROMOTE and old_score < TRUST_THRESHOLD_PROMOTE
        ):
            action = "promoted"
        elif delta > 0:
            action = "increased"
        elif delta < 0:
            action = "decreased"
        else:
            action = "unchanged"

        # Update memory in background thread
        memory.trust_score = new_score
        if action == "archived":
            memory.is_archived = True

        # Persist update
        await asyncio.to_thread(
            self._update_memory_trust, memory_id, new_score, memory.is_archived
        )

        # Phase 2.3: Audit log for trust adjustment
        if self._audit_logger is not None:
            self._audit_logger.log(
                "trust_adjustment",
                severity="warning" if action == "archived" else "info",
                memory_id=memory_id,
                description=f"Trust {signal.value}: {old_score:.3f} -> {new_score:.3f} ({action})",
                details={
                    "signal": signal.value,
                    "old_score": old_score,
                    "new_score": new_score,
                    "action": action,
                },
                state_before={"trust_score": old_score},
                state_after={"trust_score": new_score},
            )

        return TrustAdjustment(
            memory_id=memory_id,
            old_score=old_score,
            new_score=new_score,
            signal=signal,
            action=action,
        )

    def _update_memory_trust(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        """Update trust fields in database (sync wrapper).

        Args:
            memory_id: Memory ID.
            trust_score: New trust score.
            is_archived: New archived status.
        """
        if self._memory_system is None:
            return

        with contextlib.suppress(Exception):
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._memory_system._db.update_trust_fields(
                    memory_id, trust_score, is_archived
                )
            )

    async def get_trust_score(self, memory_id: str) -> dict[str, object] | None:
        """Get trust score and level for a memory.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            Dictionary with id, trustScore, trustLevel, retrievalCount, isArchived,
            or None if not enabled/memory not found.
        """
        if not self.is_enabled:
            return None

        if self._memory_system is None:
            return None

        memory = await self._memory_system.get_memory(memory_id)
        if memory is None:
            return None

        return {
            "id": memory_id,
            "trustScore": memory.trust_score,
            "trustLevel": self._get_trust_level(memory.trust_score).value,
            "retrievalCount": memory.retrieval_count,
            "isArchived": memory.is_archived,
        }

    async def record_retrieval(self, memory_id: str) -> None:
        """Record that a memory was retrieved (increments retrieval_count).

        Args:
            memory_id: ID of the memory entry.
        """
        if not self.is_enabled:
            return

        if self._memory_system is None:
            return

        memory = await self._memory_system.get_memory(memory_id)
        if memory is None:
            return

        memory.retrieval_count += 1

        # Update in background
        await asyncio.to_thread(
            self._update_retrieval_count, memory_id, memory.retrieval_count
        )

    def _update_retrieval_count(self, memory_id: str, count: int) -> None:
        """Update retrieval count in database."""
        if self._memory_system is None:
            return

        with contextlib.suppress(Exception):
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._memory_system._db.set_payload(
                    memory_id, {"retrievalCount": count}
                )
            )

    async def list_archived(
        self, limit: int = 50, offset: int = 0
    ) -> list[MemoryEntry]:
        """List archived memories (trust below archive threshold).

        Args:
            limit: Maximum number of results (default 50).
            offset: Number of results to skip (default 0).

        Returns:
            List of archived MemoryEntry objects.
        """
        if not self.is_enabled:
            return []

        if self._memory_system is None:
            return []

        # Note: Filter doesn't have is_archived field, so we need to scroll and filter
        all_memories = await self._memory_system.list_memories()
        archived = [m for m in all_memories if m.is_archived]
        return archived[offset : offset + limit]

    async def list_promoted(
        self, limit: int = 50, offset: int = 0
    ) -> list[MemoryEntry]:
        """List promoted memories (trust above promote threshold).

        Args:
            limit: Maximum number of results (default 50).
            offset: Number of results to skip (default 0).

        Returns:
            List of promoted MemoryEntry objects.
        """
        if not self.is_enabled:
            return []

        if self._memory_system is None:
            return []

        all_memories = await self._memory_system.list_memories()
        promoted = [m for m in all_memories if m.trust_score >= TRUST_THRESHOLD_PROMOTE]
        return promoted[offset : offset + limit]
