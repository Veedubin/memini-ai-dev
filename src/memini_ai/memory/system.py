"""Memory system coordinator - high-level API combining database and search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from memini_ai.memory.database import VectorDatabase, create_database
from memini_ai.memory.schema import (
    MemoryEntry,
    SearchFilter,
    SearchOptions,
    SearchStrategy,
)
from memini_ai.memory.search import MemorySearch
from memini_ai.model.embeddings import generate_embedding
from memini_ai.utils.hash import hash_content


@dataclass
class MemorySystemConfig:
    """Configuration for MemorySystem."""

    qdrant_url: str | None = None
    project_id: str | None = None
    query_collections: list[str] | None = None
    enable_cascade: bool = True
    enable_deduplication: bool = True


class MemorySystem:
    """High-level memory system coordinator.

    Combines database and search layers with lazy initialization,
    query cascade, multi-collection support, and content deduplication.
    """

    def __init__(
        self,
        db: VectorDatabase | None = None,
        search: MemorySearch | None = None,
        config: MemorySystemConfig | None = None,
    ) -> None:
        """Initialize MemorySystem.

        Args:
            db: Optional VectorDatabase instance.
            search: Optional MemorySearch instance.
            config: Optional MemorySystemConfig.
        """
        self._config = config or MemorySystemConfig()
        self._db = db or create_database()
        self._search = search or MemorySearch(self._db)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self, db_uri: str | None = None) -> None:
        """Initialize the memory system.

        Args:
            db_uri: Optional database URI override.
        """
        async with self._init_lock:
            if self._initialized:
                return

            # Apply config to Qdrant-specific attributes if present
            if hasattr(self._db, '_url') and self._config.qdrant_url:
                self._db._url = self._config.qdrant_url
            if hasattr(self._db, '_project_id') and self._config.project_id:
                self._db._project_id = self._config.project_id

            # Initialize database
            await self._db.initialize()

            self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self._initialized

    @property
    def is_ready(self) -> bool:
        """Check if system is ready for operations."""
        return self._initialized and self._db._initialized

    async def add_memory(
        self,
        input: MemoryEntry,
    ) -> str:
        """Add a memory entry.

        Args:
            input: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.

        Raises:
            ValueError: If content already exists and deduplication is enabled.
        """
        if not self._initialized:
            await self.initialize()

        # Check for duplicate content
        if self._config.enable_deduplication:
            content_hash = hash_content(input.text)
            if await self._db.content_exists(content_hash):
                raise ValueError("Memory with this content already exists")

        # Generate vector if not present
        if input.vector is None:
            embedding = await generate_embedding(input.text)
            input.vector = embedding.embedding

        # Set content hash
        if not input.content_hash:
            input.content_hash = hash_content(input.text)

        return await self._db.add_memory(input)

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        return await self._db.get_memory(memory_id)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry.

        Args:
            memory_id: ID of the memory entry to delete.
        """
        if not self._initialized:
            await self.initialize()

        await self._db.delete_memory(memory_id)
        # Invalidate BM25 cache
        await self._search.invalidate_bm25()

    async def query_memories(
        self,
        question: str,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Query memories with semantic search.

        Args:
            question: Query string.
            options: Optional search options.

        Returns:
            List of matching MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()

        # Get query collections
        collections = self._config.query_collections

        if collections and len(collections) > 1:
            # Multi-collection RRF
            return await self._multi_collection_search(question, collections, options)
        elif collections and len(collections) == 1:
            # Single collection with potential cascade
            results = await self._search.query(question, options)
            if not results and self._config.enable_cascade:
                # Try fallback collection
                fallback = self._get_fallback_collection(collections[0])
                if fallback:
                    results = await self._search.query_with_fallback_collection(
                        question, fallback, options
                    )
            return results
        else:
            # Default behavior with cascade
            results = await self._search.query(question, options)
            if not results and self._config.enable_cascade:
                # Try opposite dimension collection
                fallback = self._get_fallback_for_dimension()
                if fallback:
                    results = await self._search.query_with_fallback_collection(
                        question, fallback, options
                    )
            return results

    async def _multi_collection_search(
        self,
        question: str,
        collections: list[str],
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """Search across multiple collections with RRF fusion.

        Args:
            question: Query string.
            collections: List of collection names.
            options: Search options.

        Returns:
            Combined results from all collections.
        """
        # Search each collection
        search_tasks: list[Awaitable[list[MemoryEntry]]] = []
        for collection in collections:
            search_options = SearchOptions(
                topK=options.top_k,
                strategy=SearchStrategy.VECTOR_ONLY,
                filter=options.filter,
            )
            search_tasks.append(
                self._search.vector_only_search(
                    question,
                    search_options,
                    collection_name=collection,
                )
            )

        results_per_collection = await asyncio.gather(*search_tasks)

        # RRF fusion
        all_entries: list[list[MemoryEntry]] = []
        all_scores: list[list[float]] = []

        for results in results_per_collection:
            all_entries.append(results)
            all_scores.append([e.score or 0.0 for e in results])

        # Apply RRF
        fused = self._search._rrf_fusion(all_entries, all_scores)

        # Convert back to MemoryEntry objects with scores
        return [
            entry.model_copy(update={"score": score})
            for entry, score in fused[: options.top_k]
        ]

    def _get_fallback_collection(self, collection_name: str) -> str | None:
        """Get fallback collection for dimension cascade.

        Args:
            collection_name: Primary collection name.

        Returns:
            Fallback collection name or None.
        """
        if "1024" in collection_name:
            return collection_name.replace("1024", "384")
        elif "384" in collection_name:
            return collection_name.replace("384", "1024")
        return None

    def _get_fallback_for_dimension(self) -> str | None:
        """Get fallback collection based on current dimension.

        Returns:
            Fallback collection name or None.
        """
        dimension = self._db._dimension or 1024
        if dimension == 1024:
            return "memories_384"
        elif dimension == 384:
            return "memories_1024"
        return None

    async def search_with_vector(
        self,
        vector: list[float],
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Search with pre-computed vector.

        Args:
            vector: Pre-computed embedding vector.
            options: Optional search options.

        Returns:
            List of MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()
        return await self._search.search_with_vector(vector, options)

    async def get_similar(
        self,
        memory_id: str,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """Find memories similar to a given memory.

        Args:
            memory_id: ID of the reference memory.
            options: Optional search options.

        Returns:
            List of similar MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()
        return await self._search.get_similar(memory_id, options)

    async def list_memories(
        self,
        filter: SearchFilter | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with optional filter.

        Args:
            filter: Optional search filter.

        Returns:
            List of MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        return await self._db.list_memories(filter)

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with stats (count, dimension, etc.).
        """
        if not self._initialized:
            await self.initialize()

        count = await self._db.count_memories()
        dimension = self._db._dimension or 0
        collections = [f"memories_{dimension}"]

        return {
            "total_memories": count,
            "dimension": dimension,
            "collections": collections,
            "initialized": self._initialized,
            "ready": self.is_ready,
        }

    async def content_exists(self, text: str) -> bool:
        """Check if content with given text hash exists.

        Args:
            text: Text to check.

        Returns:
            True if content exists, False otherwise.
        """
        if not self._initialized:
            await self.initialize()

        content_hash = hash_content(text)
        return await self._db.content_exists(content_hash)

    # =============================================================================
    # TRUST ENGINE METHODS
    # =============================================================================

    async def update_memory_trust(
        self,
        memory_id: str,
        trust_score: float,
        is_archived: bool,
    ) -> None:
        """Update trust fields for a memory entry.

        Args:
            memory_id: ID of the memory entry.
            trust_score: New trust score.
            is_archived: New archived status.
        """
        if not self._initialized:
            await self.initialize()

        await self._db.update_trust_fields(memory_id, trust_score, is_archived)

    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval count for a memory entry.

        Args:
            memory_id: ID of the memory entry.
        """
        if not self._initialized:
            await self.initialize()

        await self._db.increment_retrieval_count(memory_id)

    # =============================================================================
    # MEMORY GRAPH METHODS
    # =============================================================================

    async def find_related_memories(
        self,
        memory_id: str,
        relationship_type: Any = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Find memories related to given memory.

        Args:
            memory_id: Reference memory ID.
            relationship_type: Optional filter by relationship type.
            limit: Maximum results.

        Returns:
            List of related MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        source = await self._db.get_memory(memory_id)
        if source is None:
            return []

        related_ids: list[str] = []
        for rel in source.relationships:
            if relationship_type is None or rel.relationship_type == relationship_type:
                related_ids.append(rel.target_id)

        results: list[MemoryEntry] = []
        for rel_id in related_ids[:limit]:
            memory = await self._db.get_memory(rel_id)
            if memory is not None:
                results.append(memory)

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
        if not self._initialized:
            await self.initialize()

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # Get source memory
        source = await self._db.get_memory(source_id)
        if source is None:
            return

        # Create new relationship
        from memini_ai.memory.schema import Relationship

        new_rel = Relationship(
            target_id=target_id,
            relationship_type=relationship_type,
            confidence=confidence,
            source="manual",
        )

        # Add to source memory's relationships
        source.relationships.append(new_rel)

        # Serialize and update
        rel_json = json.dumps(
            [
                {
                    "targetId": r.target_id,
                    "relationshipType": r.relationship_type.value,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in source.relationships
            ]
        )

        await self._db.set_payload(source_id, {"relationships": rel_json})

    async def get_relationship_summary(self, memory_id: str) -> dict[str, Any]:
        """Get summary of all relationships for a memory.

        Args:
            memory_id: Memory ID.

        Returns:
            Dict with counts by relationship type.
        """
        if not self._initialized:
            await self.initialize()

        source = await self._db.get_memory(memory_id)
        if source is None:
            return {
                "memoryId": memory_id,
                "totalRelationships": 0,
                "byType": {},
                "error": "Memory not found",
            }

        by_type: dict[str, int] = {}
        for rel in source.relationships:
            type_key = rel.relationship_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        return {
            "memoryId": memory_id,
            "totalRelationships": len(source.relationships),
            "byType": by_type,
        }
