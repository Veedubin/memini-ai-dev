"""Memory Graph - Relationship tracking and entity extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memini_ai.config import get_config

if TYPE_CHECKING:
    from memini_ai.memory.schema import MemoryEntry


@dataclass
class Entity:
    """Extracted entity from memory text."""

    name: str
    type: str  # "person", "place", "concept", "code", etc.
    mentions: int = 1


@dataclass
class RelationshipSummary:
    """Summary of relationships for a memory."""

    memory_id: str
    total_relationships: int
    by_type: dict[str, int]


class MemoryGraph:
    """Manages memory relationships and entity extraction.

    Features:
    - Track SUPERSEDES, RELATED_TO, CONTRADICTS, DERIVED_FROM relationships
    - Entity extraction when storing memories
    - Second-pass query for related memories
    - Optional (MEMINI_MEMORY_GRAPH=false disables)
    """

    def __init__(
        self,
        memory_system: Any = None,
        llm_client: Any = None,
    ) -> None:
        """Initialize MemoryGraph.

        Args:
            memory_system: Optional MemorySystem instance.
            llm_client: Optional httpx.AsyncClient for LLM calls.
        """
        self._memory_system = memory_system
        self._llm_client = llm_client
        self._config = get_config()
        self._initialized = False

    @property
    def is_enabled(self) -> bool:
        """Check if memory graph is enabled."""
        return self._config.memory_graph_enabled

    async def add_memory_with_relationships(
        self,
        entry: MemoryEntry,
    ) -> str:
        """Add memory and extract relationships.

        Args:
            entry: MemoryEntry to add.

        Returns:
            Memory ID.
        """
        if self._memory_system is None:
            raise ValueError("Memory system not available")

        if not self.is_enabled:
            result: str = await self._memory_system.add_memory(entry)
            return result

        # Add memory first
        memory_id: str = await self._memory_system.add_memory(entry)

        # Extract relationships if enabled
        if self._config.graph_relationship_suggestions:
            await self._extract_and_create_relationships(memory_id, entry.text)

        return memory_id

    async def _extract_and_create_relationships(
        self,
        memory_id: str,
        text: str,
    ) -> None:
        """Extract potential relationships from text and create them.

        Args:
            memory_id: ID of the source memory.
            text: Memory text content.
        """
        # Simple heuristic-based relationship extraction
        # Look for references to other memories based on content similarity
        if self._memory_system is None:
            return

        # Search for similar memories
        similar = await self._memory_system.query_memories(
            text,
            options=None,
        )

        # Create RELATED_TO relationships to similar memories
        for candidate in similar[:3]:  # Limit to top 3
            if candidate.id != memory_id and not self._relationship_exists(
                memory_id, candidate.id
            ):
                await self.create_relationship(
                    memory_id,
                    candidate.id,
                    "RELATED_TO",
                    confidence=0.5,
                )

    def _relationship_exists(self, source_id: str, target_id: str) -> bool:
        """Check if a relationship already exists.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.

        Returns:
            True if relationship exists.
        """
        # This is a placeholder - actual implementation would check DB
        return False

    async def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text using simple regex heuristics.

        Args:
            text: Memory text content.

        Returns:
            List of extracted Entity objects.
        """
        # Simple regex patterns for entity extraction
        patterns = {
            "person": r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Proper names
            "code": r"\b([a-z_]+|[A-Z_]+)\b",  # Variable/function names
            "file": r"([a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]+)",  # File paths
            "concept": r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",  # Multi-word concepts
        }

        seen: dict[str, Entity] = {}

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2:  # Filter short matches
                    key = f"{entity_type}:{match}"
                    if key in seen:
                        seen[key].mentions += 1
                    else:
                        seen[key] = Entity(name=match, type=entity_type, mentions=1)

        return list(seen.values())

    async def find_related_memories(
        self,
        memory_id: str,
        relationship_type: Any = None,
        limit: int = 10,
        include_archived: bool = True,
        max_chain_depth: int = 10,
    ) -> list[Any]:
        """Find memories related to given memory.

        For SUPERSEDES and PARTIAL_UPDATE relationships, will traverse the
        supersession chain including archived memories to find the full history.

        Args:
            memory_id: Reference memory ID.
            relationship_type: Optional filter by relationship type.
            limit: Maximum results.
            include_archived: Include archived memories for SUPERSEDES chains (default True).
            max_chain_depth: Maximum depth for supersession chain traversal (default 10).

        Returns:
            List of related MemoryEntry objects.
        """
        if self._memory_system is None:
            return []

        # Get the source memory (include archived for SUPERSEDES traversal)
        source = await self._memory_system.get_memory(memory_id, include_archived=True)
        if source is None:
            return []

        results: list[Any] = []
        seen_ids: set[str] = {memory_id}

        # Handle SUPERSEDES and PARTIAL_UPDATE relationships specially
        if relationship_type is None or relationship_type.value in (
            "SUPERSEDES",
            "PARTIAL_UPDATE",
        ):
            chain = await self._memory_system.get_supersession_chain(
                memory_id, max_chain_depth
            )
            for mem in chain:
                if mem.id not in seen_ids and len(results) < limit:
                    results.append(mem)
                    seen_ids.add(mem.id)

        # Find related memories through relationships field
        for rel in source.relationships:
            if (
                relationship_type is None or rel.relationship_type == relationship_type
            ) and rel.target_id not in seen_ids:
                memory = await self._memory_system.get_memory(
                    rel.target_id, include_archived=include_archived
                )
                if memory is not None and len(results) < limit:
                    results.append(memory)
                    seen_ids.add(memory.id)

        return results

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Any,
        confidence: float = 1.0,
    ) -> None:
        """Create a relationship between two memories.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            relationship_type: Type of relationship.
            confidence: Relationship confidence (0.0-1.0).
        """
        if self._memory_system is None:
            return

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Get source memory
        source = await self._memory_system.get_memory(source_id)
        if source is None:
            return

        # Create new relationship
        from memini_ai.memory.schema import Relationship

        new_rel = Relationship(
            target_id=target_id,
            relationship_type=relationship_type,
            confidence=confidence,
            source="auto",
        )

        # Add to source memory's relationships
        source.relationships.append(new_rel)

        # Update in database
        await self._update_memory_relationships(source_id, source.relationships)

    async def _update_memory_relationships(
        self,
        memory_id: str,
        relationships: list[Any],
    ) -> None:
        """Update relationships for a memory in the database.

        Args:
            memory_id: ID of the memory.
            relationships: List of Relationship objects.
        """
        if self._memory_system is None:
            return

        # Serialize relationships to JSON
        rel_json = json.dumps(
            [
                {
                    "targetId": r.target_id,
                    "relationshipType": r.relationship_type.value,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in relationships
            ]
        )

        # Update via database directly
        await self._memory_system._db.set_payload(
            memory_id,
            {"relationships": rel_json},
        )

    async def get_relationship_summary(
        self,
        memory_id: str,
    ) -> dict[str, Any]:
        """Get summary of all relationships for a memory.

        Args:
            memory_id: Memory ID.

        Returns:
            Dict with counts by relationship type.
        """
        if self._memory_system is None:
            return {
                "memoryId": memory_id,
                "totalRelationships": 0,
                "byType": {},
            }

        source = await self._memory_system.get_memory(memory_id)
        if source is None:
            return {
                "memoryId": memory_id,
                "totalRelationships": 0,
                "byType": {},
                "error": "Memory not found",
            }

        # Count by type
        by_type: dict[str, int] = {}
        for rel in source.relationships:
            type_key = rel.relationship_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        return {
            "memoryId": memory_id,
            "totalRelationships": len(source.relationships),
            "byType": by_type,
        }
