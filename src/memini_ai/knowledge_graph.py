"""Knowledge Graph - Entity extraction, inference, and formal KG querying.

Phase 4B upgrades the lightweight Memory Graph (Phase 2B) to a formal
knowledge graph with named entities, transitive inference, and SPARQL-lite querying.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memini_ai.config import MeminiConfig, get_config
from memini_ai.memory.schema import RelationshipType
from memini_ai.memory.system import MemorySystem
from memini_ai.postgres.database import PostgresDatabase
from memini_ai.utils.logger import logger


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PERSON = "PERSON"  # Human names
    ORGANIZATION = "ORGANIZATION"  # Companies, teams, agencies
    CONCEPT = "CONCEPT"  # Abstract ideas, theories
    CODE = "CODE"  # Function names, class names, variables
    PROJECT = "PROJECT"  # Project names, repositories
    LOCATION = "LOCATION"  # Places, addresses
    UNKNOWN = "UNKNOWN"  # Unclassified entities


@dataclass
class Entity:
    """A named entity extracted from memory content.

    Entities are the nodes in the knowledge graph. Each entity has a canonical
    form (the preferred name), type classification, and may have multiple surface
    forms (mentions found in text).
    """

    entity_id: str
    name: str  # Surface form (as found in text)
    entity_type: EntityType
    canonical_name: str  # Normalized/canonical form
    confidence: float = 1.0  # Extraction confidence 0.0-1.0
    mentions: list[str] = field(default_factory=list)  # All variations found

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entityId": self.entity_id,
            "name": self.name,
            "entityType": self.entity_type.value,
            "canonicalName": self.canonical_name,
            "confidence": self.confidence,
            "mentions": self.mentions,
        }


@dataclass
class KGQuery:
    """A formal query for the knowledge graph.

    Supports filtering by:
    - Specific entities (A and/or B)
    - Relationship types
    - Inference depth (transitive closure)
    """

    entity_a: str | None = None  # Entity ID or name filter
    entity_b: str | None = None  # Entity ID or name filter
    relationship_types: list[RelationshipType] | None = None  # Filter by rel types
    inference_depth: int = 1  # Transitive closure depth (1 = direct only)
    limit: int = 100  # Max results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entityA": self.entity_a,
            "entityB": self.entity_b,
            "relationshipTypes": (
                [r.value for r in self.relationship_types]
                if self.relationship_types
                else None
            ),
            "inferenceDepth": self.inference_depth,
            "limit": self.limit,
        }


@dataclass
class InferenceResult:
    """Result of an inference chain between two entities."""

    start_entity: str
    end_entity: str
    path: list[dict[str, Any]]  # List of {entity, relationship, next_entity}
    total_confidence: float  # Product of edge confidences
    depth: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "startEntity": self.start_entity,
            "endEntity": self.end_entity,
            "path": self.path,
            "totalConfidence": self.total_confidence,
            "depth": self.depth,
        }


@dataclass
class EntityGraphResult:
    """All connections to/from an entity."""

    entity_id: str
    entity_name: str
    incoming: list[dict[str, Any]] = field(default_factory=list)
    outgoing: list[dict[str, Any]] = field(default_factory=list)
    inferred: list[dict[str, Any]] = field(default_factory=list)  # Transitive

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entityId": self.entity_id,
            "entityName": self.entity_name,
            "incoming": self.incoming,
            "outgoing": self.outgoing,
            "inferred": self.inferred,
        }


class KnowledgeGraph:
    """Full knowledge graph with entity extraction and inference.

    Phase 4B upgrades the lightweight Memory Graph (Phase 2B) with:
    - Named entity extraction and canonicalization
    - Transitive inference (finding paths between entities)
    - SPARQL-lite query interface
    - Entity resolution (linking same entity across mentions)

    Features (all optional via KG_ENABLED config):
    - Entity extraction from memory text
    - Entity graph with typed relationships
    - Transitive closure for inference
    - SPARQL-lite query interface

    The KG builds on the Memory Graph relationships but adds:
    - Entity nodes (extracted named entities)
    - Entity-relationship edges
    - Inference engine for transitive closure

    Storage Backends:
    - MemorySystem (legacy): Entities stored as JSON in memory entries
    - PostgresDatabase (preferred): Entities stored directly in PostgreSQL
    """

    # Entity ID prefix
    ENTITY_PREFIX = "kg:entity:"

    def __init__(
        self,
        memory_system: MemorySystem | None = None,
        postgres_db: PostgresDatabase | None = None,
        config: MeminiConfig | None = None,
    ) -> None:
        """Initialize KnowledgeGraph.

        Args:
            memory_system: Optional MemorySystem instance for storage.
            postgres_db: Optional PostgresDatabase instance for direct PostgreSQL storage.
            config: Optional MeminiConfig instance.
        """
        self._memory_system = memory_system
        self._postgres_db = postgres_db
        self._config = config or get_config()
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Entity store: entity_id -> Entity
        self._entities: dict[str, Entity] = {}
        # Entity name index: canonical_name.lower() -> entity_id
        self._entity_name_index: dict[str, str] = {}
        # Entity-relationship store: source_id -> list of {target_id, rel_type, confidence}
        self._entity_relations: dict[str, list[dict[str, Any]]] = {}

    @property
    def is_enabled(self) -> bool:
        """Check if knowledge graph is enabled."""
        return self._config.knowledge_graph_enabled

    async def initialize(self) -> None:
        """Initialize the knowledge graph (load entities from storage)."""
        async with self._init_lock:
            if self._initialized:
                return

            if not self.is_enabled:
                logger.warning("knowledge_graph_disabled", status="skipping_init")
                return

            # Load entities from memory system if available
            if self._memory_system is not None:
                await self._load_entities_from_storage()

            self._initialized = True

    async def _load_entities_from_storage(self) -> None:
        """Load entities from persistent storage."""
        if self._memory_system is None:
            return

        # Search for entity memories
        try:
            results = await self._memory_system.query_memories(
                "kg:entity:",
                options=None,
            )
            # Filter to just entity memories
            for entry in results:
                if entry.text.startswith("kg:entity:"):
                    try:
                        entity_data = json.loads(entry.text[len("kg:entity:") :])
                        entity = Entity(
                            entity_id=entity_data["entityId"],
                            name=entity_data["name"],
                            entity_type=EntityType(entity_data["entityType"]),
                            canonical_name=entity_data["canonicalName"],
                            confidence=entity_data.get("confidence", 1.0),
                            mentions=entity_data.get("mentions", []),
                        )
                        self._entities[entity.entity_id] = entity
                        self._entity_name_index[
                            entity.canonical_name.lower()
                        ] = entity.entity_id
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.warning("entity_load_error", error=str(e))

    async def _save_entity_to_storage(self, entity: Entity) -> None:
        """Save an entity to persistent storage."""
        if self._memory_system is None:
            return

        try:
            entity_json = json.dumps(entity.to_dict())
            from memini_ai.memory.schema import MemoryEntry, MemorySourceType

            entry = MemoryEntry(
                text=f"kg:entity:{entity_json}",
                sourceType=MemorySourceType.boomerang,
                metadataJson=json.dumps({"entity_id": entity.entity_id}),
            )
            await self._memory_system.add_memory(entry)
        except Exception as e:
            logger.warning("entity_save_error", entity_id=entity.entity_id, error=str(e))

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    async def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        canonical_name: str | None = None,
        confidence: float = 1.0,
    ) -> Entity:
        """Create a new entity in the knowledge graph.

        Args:
            name: Surface form (as found in text).
            entity_type: Type classification.
            canonical_name: Canonical form (defaults to name if not provided).
            confidence: Extraction confidence 0.0-1.0.

        Returns:
            Created Entity instance.
        """
        # Generate entity ID
        entity_id = f"{self.ENTITY_PREFIX}{name.lower().replace(' ', '_')[:50]}"

        # Check for existing entity by canonical name
        canon = (canonical_name or name).lower()
        if canon in self._entity_name_index:
            existing_id = self._entity_name_index[canon]
            entity = self._entities[existing_id]
            # Update mentions
            if name not in entity.mentions:
                entity.mentions.append(name)
            entity.confidence = max(entity.confidence, confidence)
            return entity

        # Create new entity
        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            canonical_name=canonical_name or name,
            confidence=confidence,
            mentions=[name],
        )

        self._entities[entity_id] = entity
        self._entity_name_index[canon] = entity_id
        self._entity_relations[entity_id] = []

        # Persist to storage
        await self._save_entity_to_storage(entity)

        logger.info("entity_created", entity_id=entity_id, name=name, type=entity_type)
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity if found, None otherwise.
        """
        return self._entities.get(entity_id)

    async def get_entity_by_name(self, name: str) -> Entity | None:
        """Get entity by canonical name.

        Args:
            name: Canonical name to search for.

        Returns:
            Entity if found, None otherwise.
        """
        entity_id = self._entity_name_index.get(name.lower())
        if entity_id:
            return self._entities.get(entity_id)
        return None

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        """Search entities by name (fuzzy match).

        Args:
            query: Search query string.
            limit: Maximum results.

        Returns:
            List of matching Entity objects.
        """
        query_lower = query.lower()
        results: list[tuple[str, Entity]] = []

        for entity in self._entities.values():
            # Exact match on canonical name
            if query_lower == entity.canonical_name.lower():
                results.append((entity.canonical_name, entity))
            # Starts with query
            elif entity.canonical_name.lower().startswith(query_lower):
                results.append((f"0{entity.canonical_name}", entity))
            # Contains query
            elif query_lower in entity.canonical_name.lower():
                results.append((f"1{entity.canonical_name}", entity))
            # Contains in mentions
            else:
                for mention in entity.mentions:
                    if query_lower in mention.lower():
                        results.append((f"2{mention}", entity))
                        break

        # Sort by match quality and return top results
        results.sort(key=lambda x: x[0])
        return [e for _, e in results[:limit]]

    async def list_entities(
        self,
        entity_type: EntityType | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List all entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter.
            limit: Maximum results.

        Returns:
            List of Entity objects.
        """
        entities = list(self._entities.values())

        if entity_type is not None:
            entities = [e for e in entities if e.entity_type == entity_type]

        return entities[:limit]

    # =========================================================================
    # RELATIONSHIP MANAGEMENT (Entity-Entity)
    # =========================================================================

    async def create_entity_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
        confidence: float = 1.0,
    ) -> bool:
        """Create a relationship between two entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            rel_type: Type of relationship.
            confidence: Relationship confidence 0.0-1.0.

        Returns:
            True if relationship created, False if entities not found.
        """
        if source_id not in self._entities or target_id not in self._entities:
            return False

        # Initialize relations list if needed
        if source_id not in self._entity_relations:
            self._entity_relations[source_id] = []

        # Check for duplicate
        for rel in self._entity_relations[source_id]:
            if rel["target_id"] == target_id and rel["rel_type"] == rel_type:
                # Update confidence if higher
                if confidence > rel["confidence"]:
                    rel["confidence"] = confidence
                return True

        # Add relationship
        self._entity_relations[source_id].append({
            "target_id": target_id,
            "rel_type": rel_type,
            "confidence": confidence,
        })

        logger.info(
            "entity_relation_created",
            source=source_id,
            target=target_id,
            type=rel_type,
        )
        return True

    async def get_entity_relationships(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
    ) -> list[dict[str, Any]]:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity ID.
            rel_type: Optional relationship type filter.

        Returns:
            List of relationship dicts with target_id, rel_type, confidence.
        """
        if entity_id not in self._entity_relations:
            return []

        relations = self._entity_relations[entity_id]
        if rel_type is not None:
            relations = [r for r in relations if r["rel_type"] == rel_type]

        return relations

    def _get_entity_relationships_sync(
        self,
        entity_id: str,
        rel_type: RelationshipType | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous version of get_entity_relationships for internal use."""
        if entity_id not in self._entity_relations:
            return []

        relations = self._entity_relations[entity_id]
        if rel_type is not None:
            relations = [r for r in relations if r["rel_type"] == rel_type]

        return relations

    # =========================================================================
    # INFERENCE ENGINE (Transitive Closure)
    # =========================================================================

    async def find_inference_chain(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 3,
    ) -> InferenceResult | None:
        """Find a path between two entities using BFS.

        Args:
            start_entity_id: Starting entity ID.
            end_entity_id: Target entity ID.
            max_depth: Maximum traversal depth.

        Returns:
            InferenceResult if path found, None otherwise.
        """
        if start_entity_id not in self._entities or end_entity_id not in self._entities:
            return None

        # BFS to find shortest path
        queue: deque[tuple[str, list[dict[str, Any]], float]] = deque()
        queue.append((start_entity_id, [], 1.0))

        visited: set[str] = {start_entity_id}

        while queue:
            current, path, conf_product = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == end_entity_id:
                return InferenceResult(
                    start_entity=start_entity_id,
                    end_entity=end_entity_id,
                    path=path,
                    total_confidence=conf_product,
                    depth=len(path),
                )

            # Explore neighbors
            for rel in self._get_entity_relationships_sync(current):
                neighbor_id = rel["target_id"]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    new_path = path + [{
                        "entity": current,
                        "relationship": rel["rel_type"].value,
                        "next_entity": neighbor_id,
                    }]

                    new_conf = conf_product * rel["confidence"]
                    queue.append((neighbor_id, new_path, new_conf))

        return None

    async def get_inference_chains(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 3,
        max_results: int = 10,
    ) -> list[InferenceResult]:
        """Find multiple inference paths between two entities.

        Uses iterative deepening to find all paths up to max_depth.

        Args:
            start_entity_id: Starting entity ID.
            end_entity_id: Target entity ID.
            max_depth: Maximum path depth.
            max_results: Maximum number of paths to return.

        Returns:
            List of InferenceResult objects, sorted by confidence.
        """
        if start_entity_id not in self._entities or end_entity_id not in self._entities:
            return []

        all_paths: list[InferenceResult] = []

        # DFS to find all paths
        def dfs(
            current: str,
            end: str,
            path: list[dict[str, Any]],
            conf_product: float,
            depth: int,
            visited: set[str],
        ) -> None:
            if len(all_paths) >= max_results:
                return

            if depth > max_depth:
                return

            if current == end:
                all_paths.append(InferenceResult(
                    start_entity=start_entity_id,
                    end_entity=end_entity_id,
                    path=path.copy(),
                    total_confidence=conf_product,
                    depth=len(path),
                ))
                return

            for rel in self._get_entity_relationships_sync(current):
                neighbor_id = rel["target_id"]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    new_path = path + [{
                        "entity": current,
                        "relationship": rel["rel_type"].value,
                        "next_entity": neighbor_id,
                    }]

                    new_conf = conf_product * rel["confidence"]
                    dfs(neighbor_id, end, new_path, new_conf, depth + 1, visited)

                    visited.remove(neighbor_id)

        visited = {start_entity_id}
        dfs(start_entity_id, end_entity_id, [], 1.0, 0, visited)

        # Sort by confidence descending
        all_paths.sort(key=lambda x: x.total_confidence, reverse=True)

        return all_paths[:max_results]

    async def get_transitive_closure(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Get all entities reachable from given entity within depth.

        Args:
            entity_id: Starting entity ID.
            depth: Maximum traversal depth.

        Returns:
            List of {entity, depth, path} dicts.
        """
        if entity_id not in self._entities:
            return []

        results: list[dict[str, Any]] = []
        queue: deque[tuple[str, int, list[str]]] = deque()
        queue.append((entity_id, 0, []))
        visited: set[str] = {entity_id}

        while queue:
            current, current_depth, path = queue.popleft()

            if current_depth > 0:  # Don't include start entity in results
                results.append({
                    "entity_id": current,
                    "depth": current_depth,
                    "path": path + [current],
                })

            if current_depth >= depth:
                continue

            for rel in self._get_entity_relationships_sync(current):
                neighbor_id = rel["target_id"]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [current]
                    queue.append((neighbor_id, current_depth + 1, new_path))

        return results

    # =========================================================================
    # ENTITY GRAPH (All connections to/from an entity)
    # =========================================================================

    async def get_entity_graph(self, entity_id: str, depth: int = 1) -> EntityGraphResult | None:
        """Get all connections to/from an entity.

        Args:
            entity_id: Entity ID.
            depth: Include transitive connections up to this depth.

        Returns:
            EntityGraphResult with incoming, outgoing, and inferred connections.
        """
        if entity_id not in self._entities:
            return None

        entity = self._entities[entity_id]
        result = EntityGraphResult(entity_id=entity_id, entity_name=entity.canonical_name)

        # Get direct outgoing
        for rel in self._get_entity_relationships_sync(entity_id):
            target = self._entities.get(rel["target_id"])
            if target:
                result.outgoing.append({
                    "targetId": rel["target_id"],
                    "targetName": target.canonical_name,
                    "targetType": target.entity_type.value,
                    "relationship": rel["rel_type"].value,
                    "confidence": rel["confidence"],
                })

        # Get direct incoming (reverse lookup)
        for source_id, relations in self._entity_relations.items():
            for rel in relations:
                if rel["target_id"] == entity_id:
                    source = self._entities.get(source_id)
                    if source:
                        result.incoming.append({
                            "sourceId": source_id,
                            "sourceName": source.canonical_name,
                            "sourceType": source.entity_type.value,
                            "relationship": rel["rel_type"].value,
                            "confidence": rel["confidence"],
                        })

        # Get inferred (transitive) if depth > 1
        if depth > 1:
            transitive = await self.get_transitive_closure(entity_id, depth - 1)
            for item in transitive:
                inferred_entity = self._entities.get(item["entity_id"])
                if inferred_entity:
                    result.inferred.append({
                        "entityId": item["entity_id"],
                        "entityName": inferred_entity.canonical_name,
                        "entityType": inferred_entity.entity_type.value,
                        "depth": item["depth"],
                        "path": item["path"],
                    })

        return result

    # =========================================================================
    # SPARQL-LITE QUERY INTERFACE
    # =========================================================================

    async def query_kg(self, query: KGQuery) -> dict[str, Any]:
        """Execute a formal KG query.

        Supports:
        - Entity filtering (A and/or B)
        - Relationship type filtering
        - Transitive inference (inference_depth > 1)

        Args:
            query: KGQuery object with filters.

        Returns:
            Dictionary with results and metadata.
        """
        if not self.is_enabled:
            logger.warning("kg_query_disabled", status="kg_not_enabled")
            return {
                "success": False,
                "results": [],
                "error": "Knowledge graph is not enabled",
            }

        results: list[dict[str, Any]] = []

        # Direct entity lookup
        if query.entity_a:
            entity_a_id = await self._resolve_entity(query.entity_a)
            if entity_a_id:
                # Get all relationships for entity A
                relations = await self.get_entity_relationships(
                    entity_a_id,
                    rel_type=query.relationship_types[0] if query.relationship_types else None,
                )

                for rel in relations[: query.limit]:
                    target = self._entities.get(rel["target_id"])
                    if target:
                        result = {
                            "subject": entity_a_id,
                            "predicate": rel["rel_type"].value,
                            "object": rel["target_id"],
                            "objectName": target.canonical_name,
                            "objectType": target.entity_type.value,
                            "confidence": rel["confidence"],
                        }
                        results.append(result)

        elif query.entity_b:
            # Find all relationships pointing to entity B
            entity_b_id = await self._resolve_entity(query.entity_b)
            if entity_b_id:
                for source_id, relations in self._entity_relations.items():
                    for rel in relations:
                        if rel["target_id"] == entity_b_id and (
                            query.relationship_types is None or rel["rel_type"] in query.relationship_types
                        ):
                            source = self._entities.get(source_id)
                            if source:
                                result = {
                                    "subject": source_id,
                                    "predicate": rel["rel_type"].value,
                                    "object": entity_b_id,
                                    "subjectName": source.canonical_name,
                                    "subjectType": source.entity_type.value,
                                    "confidence": rel["confidence"],
                                }
                                results.append(result)

        else:
            # Full graph query - return all relationships
            for source_id, relations in self._entity_relations.items():
                source = self._entities.get(source_id)
                if source is None:
                    continue

                for rel in relations:
                    if query.relationship_types and rel["rel_type"] not in query.relationship_types:
                        continue

                    target = self._entities.get(rel["target_id"])
                    if target:
                        result = {
                            "subject": source_id,
                            "subjectName": source.canonical_name,
                            "subjectType": source.entity_type.value,
                            "predicate": rel["rel_type"].value,
                            "object": rel["target_id"],
                            "objectName": target.canonical_name,
                            "objectType": target.entity_type.value,
                            "confidence": rel["confidence"],
                        }
                        results.append(result)

        # Apply inference if depth > 1
        if query.inference_depth > 1 and query.entity_a and query.entity_b:
            start_id = await self._resolve_entity(query.entity_a)
            end_id = await self._resolve_entity(query.entity_b)

            if start_id and end_id:
                chains = await self.get_inference_chains(
                    start_id, end_id, max_depth=query.inference_depth - 1
                )
                # Convert to query results format
                for chain in chains:
                    results.append({
                        "inference": True,
                        "startEntity": chain.start_entity,
                        "endEntity": chain.end_entity,
                        "path": chain.path,
                        "totalConfidence": chain.total_confidence,
                        "depth": chain.depth,
                    })

        # Apply limit
        results = results[: query.limit]

        return {
            "success": True,
            "count": len(results),
            "results": results,
            "query": query.to_dict(),
        }

    async def _resolve_entity(self, identifier: str) -> str | None:
        """Resolve an entity identifier (ID or name) to entity ID.

        Args:
            identifier: Entity ID or canonical name.

        Returns:
            Entity ID if found, None otherwise.
        """
        # Direct ID lookup
        if identifier in self._entities:
            return identifier

        # Name lookup
        entity_id = self._entity_name_index.get(identifier.lower())
        if entity_id:
            return entity_id

        # Fuzzy search
        entities = await self.search_entities(identifier, limit=1)
        if entities:
            return entities[0].entity_id

        return None

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================

    async def extract_and_register_entities(self, text: str) -> list[Entity]:
        """Extract entities from text and register them in the KG.

        This is the main entry point for entity extraction. It uses regex
        patterns to find potential entities, then creates/updates them
        in the knowledge graph.

        Args:
            text: Text content to extract entities from.

        Returns:
            List of extracted Entity objects.
        """
        if not self.is_enabled:
            return []

        # Import here to avoid circular dependency
        from memini_ai.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        extracted = extractor.extract_from_text(text)

        # Register each extracted entity
        entities: list[Entity] = []
        for item in extracted:
            entity = await self.create_entity(
                name=item["name"],
                entity_type=item["type"],
                confidence=item["confidence"],
            )
            entities.append(entity)

        if entities:
            logger.info(
                "entities_extracted",
                count=len(entities),
                types=[e.entity_type.value for e in entities],
            )

        return entities

    # =========================================================================
    # MEMORY INTEGRATION
    # =========================================================================

    async def link_memory_to_entities(self, memory_id: str, text: str) -> list[str]:
        """Link a memory to extracted entities.

        Creates relationships between a memory ID and any entities found in text.

        Args:
            memory_id: Memory entry ID.
            text: Memory text content.

        Returns:
            List of entity IDs found in the text.
        """
        if not self.is_enabled:
            return []

        entities = await self.extract_and_register_entities(text)

        # Create memory -> entity relationships
        # Note: These are stored in the memory's relationship field
        # This allows tracing from memory to entities
        if self._memory_system and entities:
            memory = await self._memory_system.get_memory(memory_id)
            if memory:
                from memini_ai.memory.schema import Relationship

                for entity in entities:
                    # Create a relationship from memory to entity
                    # We encode the entity ID in the target_id for now
                    rel = Relationship(
                        target_id=entity.entity_id,
                        relationship_type=RelationshipType.RELATED_TO,
                        confidence=entity.confidence,
                        source="kg",
                    )
                    memory.relationships.append(rel)

                # Update memory
                import json as json_module

                rel_json = json_module.dumps([
                    {
                        "targetId": r.target_id,
                        "relationshipType": r.relationship_type.value,
                        "confidence": r.confidence,
                        "source": r.source,
                    }
                    for r in memory.relationships
                ])
                await self._memory_system._db.set_payload(memory_id, {"relationships": rel_json})

        return [e.entity_id for e in entities]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary with entity counts by type, relationship counts, etc.
        """
        type_counts: dict[str, int] = {}
        for entity in self._entities.values():
            type_key = entity.entity_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1

        rel_counts: dict[str, int] = {}
        for relations in self._entity_relations.values():
            for rel in relations:
                type_key = rel["rel_type"].value
                rel_counts[type_key] = rel_counts.get(type_key, 0) + 1

        return {
            "enabled": self.is_enabled,
            "total_entities": len(self._entities),
            "entity_types": type_counts,
            "total_relationships": sum(len(r) for r in self._entity_relations.values()),
            "relationship_types": rel_counts,
        }

    # =========================================================================
    # POSTGRESQL STORAGE (Direct PostgreSQL persistence for live visualization)
    # =========================================================================

    @property
    def has_postgres(self) -> bool:
        """Check if PostgreSQL backend is available."""
        return self._postgres_db is not None

    async def persist_to_postgres(self) -> dict[str, Any]:
        """Persist all in-memory entities and relationships to PostgreSQL.

        Returns:
            Dict with counts of persisted entities and relationships.
        """
        if not self._postgres_db:
            return {"success": False, "error": "PostgreSQL not configured"}

        try:
            entity_count = 0
            rel_count = 0

            # Persist all entities
            for entity in self._entities.values():
                await self._postgres_db.upsert_entity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    canonical_name=entity.canonical_name,
                    confidence=entity.confidence,
                    metadata={"mentions": entity.mentions},
                )
                entity_count += 1

            # Persist all relationships
            for source_id, relations in self._entity_relations.items():
                for rel in relations:
                    await self._postgres_db.upsert_entity_relationship(
                        source_entity_id=source_id,
                        target_entity_id=rel["target_id"],
                        relationship_type=rel["rel_type"].value,
                        confidence=rel["confidence"],
                    )
                    rel_count += 1

            logger.info("kg_persisted_to_postgres", entities=entity_count, relationships=rel_count)
            return {"success": True, "entities": entity_count, "relationships": rel_count}

        except Exception as e:
            logger.error("kg_persist_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def load_from_postgres(self) -> dict[str, Any]:
        """Load entities and relationships from PostgreSQL into memory.

        Returns:
            Dict with counts of loaded entities and relationships.
        """
        if not self._postgres_db:
            return {"success": False, "error": "PostgreSQL not configured"}

        try:
            # Load entities
            entities = await self._postgres_db.get_entities(limit=1000)
            for ent in entities:
                entity = Entity(
                    entity_id=ent["id"],
                    name=ent["name"],
                    entity_type=EntityType(ent["entity_type"]),
                    canonical_name=ent["canonical_name"],
                    confidence=ent["confidence"],
                    mentions=[],  # Metadata has mentions if any
                )
                self._entities[entity.entity_id] = entity
                self._entity_name_index[entity.canonical_name.lower()] = entity.entity_id
                self._entity_relations[entity.entity_id] = []

            # Load relationships
            _nodes, edges = await self._postgres_db.get_entities_with_relationships(limit=1000)
            for edge in edges:
                from memini_ai.memory.schema import RelationshipType

                rel_type = RelationshipType(edge["relationship"])
                source_id = edge["source"]
                target_id = edge["target"]

                if source_id in self._entity_relations:
                    # Check for existing
                    existing = False
                    for rel in self._entity_relations[source_id]:
                        if rel["target_id"] == target_id and rel["rel_type"] == rel_type:
                            existing = True
                            break
                    if not existing:
                        self._entity_relations[source_id].append({
                            "target_id": target_id,
                            "rel_type": rel_type,
                            "confidence": edge["confidence"],
                        })

            logger.info(
                "kg_loaded_from_postgres",
                entities=len(self._entities),
                relationships=sum(len(r) for r in self._entity_relations.values()),
            )
            return {
                "success": True,
                "entities": len(self._entities),
                "relationships": sum(len(r) for r in self._entity_relations.values()),
            }

        except Exception as e:
            logger.error("kg_load_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def to_postgres_json(self) -> dict[str, Any]:
        """Get D3.js visualization data directly from PostgreSQL.

        This is the primary method for the live visualization server.
        It fetches data directly from PostgreSQL without loading into memory first.

        Returns:
            Dict with nodes and edges arrays for D3.js.
        """
        if not self._postgres_db:
            return {"nodes": [], "edges": [], "error": "PostgreSQL not configured"}

        try:
            nodes, edges = await self._postgres_db.get_entities_with_relationships(limit=1000)
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error("kg_postgres_json_failed", error=str(e))
            return {"nodes": [], "edges": [], "error": str(e)}

    async def get_postgres_stats(self) -> dict[str, Any]:
        """Get entity statistics directly from PostgreSQL.

        Returns:
            Dict with entity counts by type.
        """
        if not self._postgres_db:
            return {"error": "PostgreSQL not configured"}

        try:
            return await self._postgres_db.get_entity_stats()
        except Exception as e:
            logger.error("kg_postgres_stats_failed", error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # D3.JS VISUALIZATION EXPORT
    # =========================================================================

    def to_d3_json(self, limit: int = 100) -> dict[str, Any]:
        """Export graph data in D3.js-compatible format for visualization.

        Args:
            limit: Maximum number of nodes to include.

        Returns:
            Dictionary with nodes and edges arrays for D3.js force graph.
        """
        if not self.is_enabled:
            return {"nodes": [], "edges": [], "error": "Knowledge graph is not enabled"}

        # Build nodes
        nodes: list[dict[str, Any]] = []
        for entity in list(self._entities.values())[:limit]:
            nodes.append({
                "id": entity.entity_id,
                "name": entity.canonical_name,
                "type": entity.entity_type.value,
                "confidence": entity.confidence,
                "mentions": len(entity.mentions),
                "group": self._get_entity_group(entity.entity_type),
            })

        # Build edges (relationships between entities)
        edges: list[dict[str, Any]] = []
        edge_count = 0
        for source_id, relations in self._entity_relations.items():
            for rel in relations:
                if source_id in self._entities and rel["target_id"] in self._entities:
                    edges.append({
                        "source": source_id,
                        "target": rel["target_id"],
                        "relationship": rel["rel_type"].value,
                        "confidence": rel["confidence"],
                        "stroke": self._get_rel_color(rel["rel_type"]),
                    })
                    edge_count += 1
                    if edge_count >= limit:
                        break
            if edge_count >= limit:
                break

        return {"nodes": nodes, "edges": edges}

    def _get_entity_group(self, entity_type: EntityType) -> int:
        """Map entity type to D3 group number for coloring."""
        group_map = {
            EntityType.PERSON: 1,
            EntityType.ORGANIZATION: 2,
            EntityType.CONCEPT: 3,
            EntityType.CODE: 4,
            EntityType.PROJECT: 5,
            EntityType.LOCATION: 6,
            EntityType.UNKNOWN: 0,
        }
        return group_map.get(entity_type, 0)

    def _get_rel_color(self, rel_type: RelationshipType) -> str:
        """Get color for relationship type."""
        color_map = {
            RelationshipType.SUPERSEDES: "#e74c3c",
            RelationshipType.RELATED_TO: "#3498db",
            RelationshipType.CONTRADICTS: "#9b59b6",
            RelationshipType.DERIVED_FROM: "#27ae60",
        }
        return color_map.get(rel_type, "#95a5a6")


def generate_visualization_html(graph_data: dict[str, Any]) -> str:
    """Generate self-contained HTML page with D3.js force-directed graph.

    Args:
        graph_data: Dict with nodes and edges arrays from to_d3_json().

    Returns:
        Complete HTML string with embedded D3.js visualization.
    """
    import json

    nodes_json = json.dumps(graph_data.get("nodes", []))
    edges_json = json.dumps(graph_data.get("edges", []))
    error_msg = graph_data.get("error", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --border-color: #2d3a5a;
        }}
        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-primary: #f5f5f5;
                --bg-secondary: #ffffff;
                --text-primary: #1a1a2e;
                --text-secondary: #4a4a6a;
                --border-color: #d0d0e0;
            }}
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ font-size: 1.25rem; font-weight: 600; }}
        .stats {{ font-size: 0.875rem; color: var(--text-secondary); }}
        #graph-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem;
        }}
        svg {{ background: var(--bg-secondary); border-radius: 8px; }}
        .node {{ cursor: pointer; stroke-width: 2px; }}
        .node:hover {{ stroke: #fff; stroke-width: 3px; }}
        .link {{ stroke-opacity: 0.6; }}
        .link:hover {{ stroke-opacity: 1; }}
        .link-label {{
            font-size: 10px;
            fill: var(--text-secondary);
            pointer-events: none;
        }}
        .tooltip {{
            position: absolute;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.75rem;
            font-size: 0.8rem;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 250px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .tooltip.visible {{ opacity: 1; }}
        .tooltip h3 {{ font-size: 0.9rem; margin-bottom: 0.5rem; }}
        .tooltip p {{ margin: 0.25rem 0; color: var(--text-secondary); }}
        .tooltip .type {{ display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 500;
        }}
        .legend {{
            position: absolute;
            bottom: 1rem;
            right: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            font-size: 0.75rem;
        }}
        .legend h4 {{ margin-bottom: 0.5rem; font-weight: 600; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
        .no-data {{
            text-align: center;
            padding: 4rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <header>
        <h1>Memory Graph Visualization</h1>
        <span class="stats" id="stats"></span>
    </header>
    <div id="graph-container"></div>
    <div class="tooltip" id="tooltip">
        <h3 id="tooltip-name"></h3>
        <p><span class="type" id="tooltip-type"></span></p>
        <p>Confidence: <span id="tooltip-confidence"></span></p>
        <p>Mentions: <span id="tooltip-mentions"></span></p>
        <p id="tooltip-rels"></p>
    </div>
    <div class="legend">
        <h4>Node Types</h4>
        <div class="legend-item"><div class="legend-color" style="background:#4a90d9"></div>Person</div>
        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>Organization</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>Concept</div>
        <div class="legend-item"><div class="legend-color" style="background:#e67e22"></div>Code</div>
        <div class="legend-item"><div class="legend-color" style="background:#f1c40f"></div>Project</div>
        <div class="legend-item"><div class="legend-color" style="background:#95a5a6"></div>Unknown</div>
        <h4 style="margin-top:0.75rem">Relationships</h4>
        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>Supersedes</div>
        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>Related To</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>Contradicts</div>
        <div class="legend-item"><div class="legend-color" style="background:#27ae60"></div>Derived From</div>
    </div>

    <script>
    const nodes = {nodes_json};
    const edges = {edges_json};
    const errorMsg = "{error_msg}";

    const typeColors = {{
        PERSON: "#4a90d9",
        ORGANIZATION: "#27ae60",
        CONCEPT: "#9b59b6",
        CODE: "#e67e22",
        PROJECT: "#f1c40f",
        LOCATION: "#1abc9c",
        UNKNOWN: "#95a5a6"
    }};

    if (nodes.length === 0) {{
        document.getElementById("graph-container").innerHTML =
            `<div class="no-data">{{${{errorMsg or "No graph data available."}}}}</div>`;
    }} else {{
        const container = document.getElementById("graph-container");
        const width = Math.min(container.clientWidth - 32, 1200);
        const height = Math.min(window.innerHeight - 120, 800);

        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const g = svg.append("g");

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);

        // Arrow marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-5 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M-5,-5L5,0L-5,5")
            .attr("fill", "#666");

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(40));

        const link = g.append("g")
            .selectAll("line")
            .data(edges)
            .join("line")
            .attr("class", "link")
            .attr("stroke", d => d.stroke || "#666")
            .attr("stroke-width", d => Math.max(1, d.confidence * 2))
            .attr("marker-end", "url(#arrowhead)");

        const node = g.append("g")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                }})
                .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
                .on("end", (event, d) => {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }}));

        node.append("circle")
            .attr("r", d => Math.max(8, Math.min(25, 8 + d.mentions * 2)))
            .attr("fill", d => typeColors[d.type] || typeColors.UNKNOWN)
            .attr("stroke", "#fff");

        node.append("text")
            .text(d => d.name.length > 15 ? d.name.substring(0, 12) + "..." : d.name)
            .attr("text-anchor", "middle")
            .attr("dy", d => Math.max(8, Math.min(25, 8 + d.mentions * 2)) + 15)
            .attr("fill", "var(--text-primary)")
            .attr("font-size", "11px");

        // Tooltip
        const tooltip = document.getElementById("tooltip");
        const tooltipName = document.getElementById("tooltip-name");
        const tooltipType = document.getElementById("tooltip-type");
        const tooltipConfidence = document.getElementById("tooltip-confidence");
        const tooltipMentions = document.getElementById("tooltip-mentions");
        const tooltipRels = document.getElementById("tooltip-rels");

        node.on("mouseover", (event, d) => {{
            tooltipName.textContent = d.name;
            tooltipType.textContent = d.type;
            tooltipType.style.background = typeColors[d.type] || typeColors.UNKNOWN;
            tooltipConfidence.textContent = (d.confidence * 100).toFixed(0) + "%";
            tooltipMentions.textContent = d.mentions;

            const relatedEdges = edges.filter(e => e.source.id === d.id || e.target.id === d.id);
            tooltipRels.textContent = relatedEdges.length + " relationships";
            tooltip.classList.add("visible");
        }})
        .on("mousemove", (event) => {{
            tooltip.style.left = (event.pageX + 15) + "px";
            tooltip.style.top = (event.pageY - 10) + "px";
        }})
        .on("mouseout", () => tooltip.classList.remove("visible"));

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        document.getElementById("stats").textContent =
            `${{nodes.length}} nodes, ${{edges.length}} edges`;
    }}
    </script>
</body>
</html>"""
