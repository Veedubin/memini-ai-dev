"""Memory database layer - Vector database abstraction with Qdrant implementation."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from memini_ai.config import MeminiConfig, get_config
from memini_ai.memory.schema import (
    DEFAULT_QDRANT_URL,
    MEMORY_TABLE_NAME,
    QDRANT_HNSW_CONFIG,
    MemoryEntry,
    SearchFilter,
    SearchOptions,
)
from memini_ai.utils.hash import hash_content

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.conversions.common_types import PointId as QdrantPointId
    from qdrant_client.models import Condition as QdrantCondition
    from qdrant_client.models import Filter as QdrantFilterType


# Module-level client cache (singleton by URL)
_client_cache: dict[str, AsyncQdrantClient] = {}


def _get_collection_name(dimension: int) -> str:
    """Get collection name with dimension suffix."""
    return f"{MEMORY_TABLE_NAME}_{dimension}"


class VectorDatabase(ABC):
    """Abstract base class for vector database operations.

    Provides a common interface for different vector database backends
    (Qdrant, pgvector, etc.) to enable backend-agnostic memory operations.
    """

    _initialized: bool
    _dimension: int | None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database and create collections if needed."""
        ...

    @abstractmethod
    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry.

        Args:
            entry: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.
        """
        ...

    @abstractmethod
    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Add multiple memory entries.

        Args:
            entries: List of MemoryEntry objects to add.

        Returns:
            List of MemoryEntry objects with IDs assigned.
        """
        ...

    @abstractmethod
    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        ...

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID.

        Args:
            memory_id: ID of the memory entry to delete.
        """
        ...

    @abstractmethod
    async def delete_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> int:
        """Delete all memories with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            Number of deleted memories.
        """
        ...

    @abstractmethod
    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Query memories using vector similarity.

        Args:
            vector: Query vector.
            options: Search options.
            collection_name: Optional collection override.

        Returns:
            List of matching MemoryEntry objects with scores.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def count_memories(self) -> int:
        """Count total memories.

        Returns:
            Number of memory entries.
        """
        ...

    @abstractmethod
    async def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists.

        Args:
            content_hash: SHA-256 hash to check.

        Returns:
            True if exists, False otherwise.
        """
        ...

    @abstractmethod
    async def get_entries_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all entries with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            List of matching MemoryEntry objects.
        """
        ...

    @abstractmethod
    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Scroll through a collection.

        Args:
            collection_name: Collection to scroll.
            limit: Page size.

        Returns:
            List of MemoryEntry objects.
        """
        ...

    @abstractmethod
    async def get_collection_dimension(self, collection_name: str) -> int | None:
        """Get the dimension of a collection.

        Args:
            collection_name: Collection to check.

        Returns:
            Vector dimension or None if not found.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    async def update_trust_fields(
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
        ...

    @abstractmethod
    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval count for a memory entry.

        Args:
            memory_id: ID of the memory entry.
        """
        ...

    @abstractmethod
    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Set payload fields for a memory entry.

        Args:
            memory_id: ID of the memory entry.
            payload: Dictionary of payload fields to set.
        """
        ...


class QdrantDatabase(VectorDatabase):
    """Qdrant implementation of VectorDatabase with retry, project isolation, and dimension awareness.

    Client singleton caching at module level. Connection health validated before
    each operation. Exponential backoff retry (3 attempts, 1s base delay).
    """

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        project_id: str | None = None,
    ) -> None:
        """Initialize QdrantDatabase.

        Args:
            url: Qdrant server URL.
            project_id: Optional project ID for isolation.
        """
        self._url = url
        self._project_id = project_id
        self._dimension: int | None = None
        self._initialized = False
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Create collection if needed with proper indexing."""
        if self._initialized:
            return

        client = await self._get_client()
        config = get_config()
        self._dimension = config.embedding_dim

        collection_name = _get_collection_name(self._dimension)

        # Check if collection exists
        try:
            exists = await client.collection_exists(collection_name)
        except Exception:
            exists = False

        if not exists:
            # Create collection with HNSW config
            from qdrant_client.models import (
                Distance,
                HnswConfigDiff,
                VectorParams,
            )

            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=QDRANT_HNSW_CONFIG["m"],
                    ef_construct=QDRANT_HNSW_CONFIG["ef_construct"],
                    full_scan_threshold=QDRANT_HNSW_CONFIG["full_scan_threshold"],
                ),
            )

            # Create payload indexes
            await self._create_indexes(client, collection_name)

        self._initialized = True

    async def _create_indexes(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        """Create payload indexes on collection."""
        from qdrant_client.models import PayloadSchemaType

        # Index all metadata fields
        index_fields = [
            "sourceType",
            "sourcePath",
            "sessionId",
            "contentHash",
            "timestamp",
            "projectId",
            "isArchived",
            "relationships",
        ]

        for field in index_fields:
            with contextlib.suppress(Exception):
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client (singleton by URL)."""
        if self._url not in _client_cache:
            from qdrant_client import AsyncQdrantClient

            _client_cache[self._url] = AsyncQdrantClient(url=self._url)
        return _client_cache[self._url]

    async def _validate_connection(self) -> None:
        """Validate connection health before operations."""
        client = await self._get_client()
        try:
            # Quick health check
            await client.get_collections()
        except Exception as e:
            raise ConnectionError(f"Qdrant connection failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _retry_operation(
        self,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with retry on failure."""
        await self._validate_connection()
        return await operation(*args, **kwargs)

    def _entry_to_payload(self, entry: MemoryEntry) -> dict[str, Any]:
        """Convert MemoryEntry to Qdrant point payload."""
        payload: dict[str, Any] = {
            "text": entry.text,
            "sourceType": entry.source_type.value if entry.source_type else None,
            "sourcePath": entry.source_path,
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "contentHash": entry.content_hash,
            "metadataJson": entry.metadata_json,
            "sessionId": entry.session_id,
            "projectId": entry.project_id,
            "trustScore": entry.trust_score,
            "retrievalCount": entry.retrieval_count,
            "isArchived": entry.is_archived,
        }

        # Serialize relationships as JSON
        if entry.relationships:
            payload["relationships"] = json.dumps(
                [
                    {
                        "targetId": r.target_id,
                        "relationshipType": r.relationship_type.value,
                        "confidence": r.confidence,
                        "source": r.source,
                    }
                    for r in entry.relationships
                ]
            )

        return payload

    def _payload_to_entry(
        self,
        payload: dict[str, Any] | None,
        vector: list[float]
        | list[list[float]]
        | dict[str, list[float] | Any]
        | None = None,
        score: float | None = None,
    ) -> MemoryEntry:
        """Convert Qdrant payload to MemoryEntry."""
        if payload is None:
            payload = {}

        # Create a mutable copy with score injected
        entry_data = dict(payload)
        if score is not None:
            entry_data["score"] = score

        # Deserialize relationships from JSON
        if "relationships" in entry_data and isinstance(
            entry_data["relationships"], str
        ):
            try:
                rels_data = json.loads(entry_data["relationships"])
                from memini_ai.memory.schema import Relationship, RelationshipType

                rels = [
                    Relationship(
                        target_id=r["targetId"],
                        relationship_type=RelationshipType(r["relationshipType"]),
                        confidence=r.get("confidence", 1.0),
                        source=r.get("source", "auto"),
                    )
                    for r in rels_data
                ]
                entry_data["relationships"] = rels
            except (json.JSONDecodeError, KeyError, ValueError):
                # Invalid relationships data - use empty list
                entry_data["relationships"] = []

        # Use model_validate to properly handle populate_by_name
        return MemoryEntry.model_validate(entry_data)

    async def add_memory(self, entry: MemoryEntry) -> str:
        """Add a single memory entry.

        Args:
            entry: MemoryEntry to add.

        Returns:
            The ID of the added memory entry.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        # Generate ID if not present
        memory_id = entry.id or str(uuid.uuid4())
        entry.id = memory_id

        # If entry has no content hash, compute it
        if not entry.content_hash:
            entry.content_hash = hash_content(entry.text)

        # Upsert to Qdrant
        await self._retry_operation(
            client.upsert,
            collection_name=collection_name,
            points=[
                {
                    "id": memory_id,
                    "vector": entry.vector or [0.0] * (self._dimension or 1024),
                    "payload": self._entry_to_payload(entry),
                }
            ],
        )

        return memory_id

    async def add_memories(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Add multiple memory entries.

        Args:
            entries: List of MemoryEntry objects to add.

        Returns:
            List of MemoryEntry objects with IDs assigned.
        """
        if not entries:
            return []

        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        points = []
        for entry in entries:
            # Generate ID if not present
            memory_id = entry.id or str(uuid.uuid4())
            entry.id = memory_id

            # Compute content hash if missing
            if not entry.content_hash:
                entry.content_hash = hash_content(entry.text)

            points.append(
                {
                    "id": memory_id,
                    "vector": entry.vector or [0.0] * (self._dimension or 1024),
                    "payload": self._entry_to_payload(entry),
                }
            )

        if points:
            await self._retry_operation(
                client.upsert,
                collection_name=collection_name,
                points=points,
            )

        return entries

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry if found, None otherwise.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        try:
            result = await client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_vectors=True,
            )
            if result and len(result) > 0:
                record = result[0]
                return self._payload_to_entry(
                    record.payload,
                    vector=None,  # Vector handled via model_validate if needed
                )
        except Exception:
            pass

        return None

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID.

        Args:
            memory_id: ID of the memory entry to delete.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        await self._retry_operation(
            client.delete,
            collection_name=collection_name,
            points_selector=[memory_id],
        )

    async def delete_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> int:
        """Delete all memories with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            Number of deleted memories.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        # Build filter
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filter_conditions: list[QdrantCondition] = [
            FieldCondition(key="sourcePath", match=MatchValue(value=source_path))
        ]
        if source_type:
            filter_conditions.append(
                FieldCondition(key="sourceType", match=MatchValue(value=source_type))
            )

        filter_obj = Filter(must=filter_conditions)

        # Scroll and delete
        deleted_count = 0
        offset: QdrantPointId | None = None
        while True:
            try:
                result = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_obj,
                    limit=100,
                    offset=offset,
                    with_vectors=False,
                )
                # result is tuple[list[Record], PointId | None]
                records = result[0]
                if not records:
                    break

                ids_to_delete = [r.id for r in records]
                from qdrant_client.models import PointIdsList

                await client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=ids_to_delete),
                )
                deleted_count += len(ids_to_delete)

                # Get offset from result
                offset = result[1] if len(result) > 1 else None
                if offset is None:
                    break
            except Exception:
                break

        return deleted_count

    async def query_memories(
        self,
        vector: list[float],
        options: SearchOptions,
        collection_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Query memories using vector similarity.

        Args:
            vector: Query vector.
            options: Search options.
            collection_name: Optional collection override.

        Returns:
            List of matching MemoryEntry objects with scores.
        """
        await self.initialize()
        client = await self._get_client()
        collection = collection_name or _get_collection_name(self._dimension or 1024)

        # Build filter from options
        filter_obj = None
        if options.filter:
            filter_obj = self._build_filter_from_search_filter(options.filter)

        # Search using query_points
        try:
            result = await client.query_points(
                collection_name=collection,
                query=vector,
                limit=options.top_k,
                query_filter=filter_obj,
                with_vectors=False,
            )
            return [
                self._payload_to_entry(
                    hit.payload,
                    vector=hit.vector if hasattr(hit, "vector") else None,
                    score=hit.score if hasattr(hit, "score") else None,
                )
                for hit in result.points
            ]
        except Exception:
            return []

    def _build_filter_from_search_filter(
        self, filter: SearchFilter
    ) -> QdrantFilterType | None:
        """Build Qdrant filter from SearchFilter."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        conditions: list[QdrantCondition] = []

        if filter.source_type:
            conditions.append(
                FieldCondition(
                    key="sourceType",
                    match=MatchValue(value=filter.source_type.value),
                )
            )

        if filter.session_id:
            conditions.append(
                FieldCondition(
                    key="sessionId",
                    match=MatchValue(value=filter.session_id),
                )
            )

        if filter.project_id:
            conditions.append(
                FieldCondition(
                    key="projectId",
                    match=MatchValue(value=filter.project_id),
                )
            )

        if filter.since:
            # Convert datetime to timestamp float
            ts = filter.since.timestamp()
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(gte=ts),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

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
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        filter_obj = None
        if filter:
            filter_obj = self._build_filter_from_search_filter(filter)

        entries: list[MemoryEntry] = []
        offset: QdrantPointId | None = None
        while True:
            try:
                result = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_obj,
                    limit=100,
                    offset=offset,
                    with_vectors=False,
                )
                # result is tuple[list[Record], PointId | None]
                records = result[0]
                for record in records:
                    entries.append(
                        self._payload_to_entry(
                            record.payload,
                            vector=record.vector if hasattr(record, "vector") else None,
                        )
                    )

                offset = result[1] if len(result) > 1 else None
                if offset is None:
                    break
            except Exception:
                break

        return entries

    async def count_memories(self) -> int:
        """Count total memories.

        Returns:
            Number of memory entries.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        try:
            result = await client.get_collection(collection_name)
            # Use points_count for total point count
            return result.points_count or 0
        except Exception:
            return 0

    async def content_exists(self, content_hash: str) -> bool:
        """Check if content with given hash exists.

        Args:
            content_hash: SHA-256 hash to check.

        Returns:
            True if exists, False otherwise.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filter_obj = Filter(
            must=[
                FieldCondition(key="contentHash", match=MatchValue(value=content_hash))
            ]
        )

        try:
            result = await client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=1,
            )
            # result is tuple[list[Record], PointId | None]
            records = result[0]
            return len(records) > 0
        except Exception:
            return False

    async def get_entries_by_source_path(
        self,
        source_path: str,
        source_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all entries with a given source path.

        Args:
            source_path: Source path to match.
            source_type: Optional source type filter.

        Returns:
            List of matching MemoryEntry objects.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filter_conditions: list[QdrantCondition] = [
            FieldCondition(key="sourcePath", match=MatchValue(value=source_path))
        ]
        if source_type:
            filter_conditions.append(
                FieldCondition(key="sourceType", match=MatchValue(value=source_type))
            )

        filter_obj = Filter(must=filter_conditions)

        entries: list[MemoryEntry] = []
        offset: QdrantPointId | None = None
        while True:
            try:
                result = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_obj,
                    limit=100,
                    offset=offset,
                    with_vectors=False,
                )
                # result is tuple[list[Record], PointId | None]
                records = result[0]
                for record in records:
                    entries.append(
                        self._payload_to_entry(
                            record.payload,
                            vector=record.vector if hasattr(record, "vector") else None,
                        )
                    )

                offset = result[1] if len(result) > 1 else None
                if offset is None:
                    break
            except Exception:
                break

        return entries

    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Scroll through a collection.

        Args:
            collection_name: Collection to scroll.
            limit: Page size.

        Returns:
            List of MemoryEntry objects.
        """
        await self._validate_connection()
        client = await self._get_client()

        entries: list[MemoryEntry] = []
        offset: QdrantPointId | None = None
        while True:
            try:
                result = await client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_vectors=False,
                )
                # result is tuple[list[Record], PointId | None]
                records = result[0]
                for record in records:
                    entries.append(
                        self._payload_to_entry(
                            record.payload,
                            vector=record.vector if hasattr(record, "vector") else None,
                        )
                    )

                offset = result[1] if len(result) > 1 else None
                if offset is None:
                    break
            except Exception:
                break

        return entries

    async def get_collection_dimension(self, collection_name: str) -> int | None:
        """Get the dimension of a collection.

        Args:
            collection_name: Collection to check.

        Returns:
            Vector dimension or None if not found.
        """
        await self._validate_connection()
        client = await self._get_client()

        try:
            result = await client.get_collection(collection_name)
            if result.config and result.config.params:
                vectors_config = result.config.params
                # Handle both dict and object access patterns
                if hasattr(vectors_config, "vectors"):
                    vectors_dict = vectors_config.vectors
                    if isinstance(vectors_dict, dict):
                        first_vec = next(iter(vectors_dict.values()), None)
                        if first_vec is not None and hasattr(first_vec, "size"):
                            return first_vec.size
        except Exception:
            pass

        return None

    async def close(self) -> None:
        """Close the database connection."""
        if self._url in _client_cache:
            del _client_cache[self._url]
        self._client = None
        self._initialized = False

    async def update_trust_fields(
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
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        with contextlib.suppress(Exception):
            await client.set_payload(
                collection_name=collection_name,
                payload={
                    "trustScore": trust_score,
                    "isArchived": is_archived,
                },
                points=[memory_id],
            )

    async def increment_retrieval_count(self, memory_id: str) -> None:
        """Increment retrieval count for a memory entry.

        Args:
            memory_id: ID of the memory entry.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        # Get current count
        memory = await self.get_memory(memory_id)
        if memory is None:
            return

        new_count = memory.retrieval_count + 1

        with contextlib.suppress(Exception):
            await client.set_payload(
                collection_name=collection_name,
                payload={"retrievalCount": new_count},
                points=[memory_id],
            )

    async def set_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Set payload fields for a memory entry.

        Args:
            memory_id: ID of the memory entry.
            payload: Dictionary of payload fields to set.
        """
        await self.initialize()
        client = await self._get_client()
        collection_name = _get_collection_name(self._dimension or 1024)

        with contextlib.suppress(Exception):
            await client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[memory_id],
            )


def create_database(config: MeminiConfig | None = None) -> VectorDatabase:
    """Factory function to create a VectorDatabase instance.

    Checks MEMINI_DB_URL environment variable to determine backend type.
    Falls back to Qdrant if not set or if QDRANT_URL is specified.

    Args:
        config: Optional MeminiConfig instance. If not provided, uses get_config().

    Returns:
        VectorDatabase implementation instance (QdrantDatabase by default).

    Raises:
        ValueError: If database type is not recognized.
    """
    if config is None:
        config = get_config()

    # Check for database URL environment variable
    db_url = os.environ.get("MEMINI_DB_URL", "").lower()

    if not db_url:
        # Default to Qdrant using config's qdrant_url
        return QdrantDatabase(url=config.qdrant_url, project_id=config.project_id)

    if db_url.startswith("qdrant://") or db_url == "qdrant":
        # Qdrant backend
        return QdrantDatabase(url=config.qdrant_url, project_id=config.project_id)

    if (
        db_url.startswith("postgres://")
        or db_url.startswith("postgresql://")
        or db_url == "postgres"
        or db_url == "pgvector"
    ):
        # PostgreSQL/pgvector backend
        from memini_ai.postgres import PostgresDatabase

        return PostgresDatabase(
            db_url=os.environ.get("MEMINI_DB_URL", ""), project_id=config.project_id
        )

    # Unknown backend
    raise ValueError(
        f"Unknown database type: {db_url}. "
        "Supported backends: qdrant, postgres (future). "
        "Set MEMINI_DB_URL=qdrant or remove MEMINI_DB_URL to use Qdrant."
    )


# Backward compatibility alias
MemoryDatabase = QdrantDatabase
