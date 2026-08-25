"""Memory system coordinator - high-level API combining database and search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from memini_ai.config import get_config
from memini_ai.memory.database import VectorDatabase, create_database
from memini_ai.memory.rrf import rrf_with_limit
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    SearchFilter,
    SearchOptions,
    SearchStrategy,
)
from memini_ai.memory.search import MemorySearch
from memini_ai.model.embeddings import generate_embedding
from memini_ai.model.manager import ModelManager
from memini_ai.utils.hash import hash_content
from memini_ai.utils.logger import logger


@dataclass
class MemorySystemConfig:
    """Configuration for MemorySystem.

    The dual-model RRF fields (``embedding_mode``, ``rrf_k``) default to ``None``,
    which causes :class:`MemorySystem` to fall back to the global
    :class:`MeminiConfig` values. This keeps :class:`MemorySystem` testable in
    isolation while honoring the env-driven default at runtime.
    """

    project_id: str | None = None
    query_collections: list[str] | None = None
    enable_cascade: bool = True
    enable_deduplication: bool = True

    # Dual-model RRF (v0.7.0+). None → fall back to global MeminiConfig.
    embedding_mode: str | None = None
    rrf_k: int | None = None


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

    @property
    def _resolved_embedding_mode(self) -> str:
        """Resolve effective embedding_mode (config override > global MeminiConfig)."""
        if self._config.embedding_mode is not None:
            return self._config.embedding_mode
        return get_config().embedding_mode

    @property
    def _resolved_rrf_k(self) -> int:
        """Resolve effective RRF k (config override > global MeminiConfig)."""
        if self._config.rrf_k is not None:
            return self._config.rrf_k
        return get_config().rrf_k

    async def close(self) -> None:
        """Close the memory system and release resources.

        Closes the underlying database connection pool.
        After calling close(), the system should not be used further.
        """
        if self._initialized:
            await self._db.close()
            self._initialized = False

    async def add_memory(
        self,
        input: MemoryEntry,
    ) -> str:
        """Add a memory entry.

        Dispatches on ``embedding_mode`` (v0.7.0+):
            * ``"cpu"`` (default fallback): 384-dim write only — full
              backward compatibility, no 1024 sidecar.
            * ``"auto"``: 384-dim write (always), plus 1024-dim write if
              the active dimension is 1024 OR if the 384-dim row has
              already been elevated via :meth:`elevate_memory_to_1024`.
              In v0.7.0 the auto-mode path writes the 384 record first
              and defers 1024 sidecar creation to explicit elevation —
              the search path then fuses both stores via RRF.
            * ``"gpu"``: 1024-dim-only — caller is expected to have
              supplied a 1024-dim ``input.vector`` (or the configured
              embedder produces 1024). The 384-dim write is skipped.

        Args:
            input: MemoryEntry to add.

        Returns:
            The ID of the added memory entry (the 384-dim row's id in
            cpu/auto modes; the 1024-dim row's id in gpu mode if the
            underlying db does not have a 384 row — otherwise still the
            384 row id and a sidecar is written).

        Raises:
            ValueError: If content already exists and deduplication is enabled,
                or if the configured mode is invalid.
            RuntimeError: If ``embedding_mode == "gpu"`` but the underlying
                database does not expose 1024-dim methods (e.g. an in-memory
                mock).
        """
        if not self._initialized:
            await self.initialize()

        mode = self._resolved_embedding_mode
        if mode not in {"cpu", "auto", "gpu"}:
            raise ValueError(
                f"Invalid embedding_mode '{mode}'. Must be one of: cpu, auto, gpu"
            )

        # Check for duplicate content (only on the 384-dim store; the 1024
        # sidecar shares the 384-dim content_hash implicitly).
        if self._config.enable_deduplication:
            content_hash = hash_content(input.text)
            if await self._db.content_exists(content_hash):
                raise ValueError("Memory with this content already exists")

        # Generate 384-dim vector if not present. The MiniLM embedder is the
        # current model — same path used in v0.6.x.
        if input.vector is None:
            try:
                embedding = await generate_embedding(input.text)
                input.vector = embedding.embedding
            except Exception as e:
                # v0.7.7: EmbeddingDimMismatchError — the loaded model's dim
                # doesn't match the DB column. Store the memory with a NULL
                # embedding so the content is preserved; vector search will
                # be disabled until the mismatch is resolved.
                from memini_ai.model.manager import EmbeddingDimMismatchError

                if isinstance(e, EmbeddingDimMismatchError):
                    logger.warning(
                        "add_memory_embedding_dim_mismatch",
                        message=str(e),
                        content_prefix=input.text[:80],
                    )
                    input.vector = None
                else:
                    raise

        # Set content hash
        if not input.content_hash:
            input.content_hash = hash_content(input.text)

        # Phase 1 feature-activation: near-duplicate auto-SUPERSEDES.
        # When auto_relationship_detection is ON and we have a vector,
        # run a single vector-similarity query for near-duplicates
        # BEFORE the write so we can capture the target id. The
        # relationship is created AFTER the write returns memory_id.
        # Failure is isolated: any exception is logged and the write
        # proceeds normally (preserves v1.3.1 behavior when flag is OFF).
        # The flags live on the global MeminiConfig (env-driven), not
        # the MemorySystemConfig override, so read via get_config().
        _global_cfg = get_config()
        near_dup_target: str | None = None
        near_dup_similarity: float = 0.0
        if _global_cfg.auto_relationship_detection and input.vector is not None:
            try:
                _opts = SearchOptions(
                    topK=1,
                    strategy=SearchStrategy.VECTOR_ONLY,
                    threshold=_global_cfg.auto_relationship_similarity_threshold,
                )
                _results = await self._db.query_memories(list(input.vector), _opts)
                if _results:
                    _best = _results[0]
                    _sim = float(getattr(_best, "score", 0.0) or 0.0)
                    if _sim >= _global_cfg.auto_relationship_similarity_threshold:
                        near_dup_target = _best.id
                        near_dup_similarity = _sim
            except Exception:
                logger.warning(
                    "auto_relationship_near_dup_search_failed",
                    content_prefix=input.text[:80],
                )

        if mode == "cpu":
            # Legacy single-store path: only the 384-dim write.
            memory_id = await self._db.add_memory(input)
            await self._maybe_create_auto_relationship(
                memory_id, near_dup_target, near_dup_similarity
            )
            return memory_id

        if mode == "gpu":
            # 1024-dim-only path. The caller is expected to have set
            # input.vector to a 1024-dim vector (or run a 1024 embedder).
            # The 384-dim write would clobber the schema, so we skip it.
            # Use iscoroutinefunction to detect real 1024 support —
            # bare hasattr returns True for any MagicMock.
            add_1024 = getattr(self._db, "add_memory_1024", None)
            if add_1024 is None or not asyncio.iscoroutinefunction(add_1024):
                raise RuntimeError(
                    "embedding_mode='gpu' requires PostgresDatabase with "
                    "add_memory_1024 support; current db does not expose it."
                )
            # Write 384-dim row (the source-of-truth record) and then
            # mirror to the 1024 sidecar. gpu mode is "1024-only" at
            # search time, but the 384 row is still required because
            # every other component (trust, retrieval_count, etc.)
            # writes to the 384-dim ``memories`` table.
            memory_id = await self._db.add_memory(input)
            expand = getattr(self._db, "_expand_384_to_1024", None)
            vector_1024: list[float] | None = None
            if expand is not None and input.vector is not None:
                vector_1024 = expand(list(input.vector), 1024)
            await add_1024(memory_id, vector_1024)
            await self._maybe_create_auto_relationship(
                memory_id, near_dup_target, near_dup_similarity
            )
            return memory_id

        # mode == "auto": write 384-dim first (source of truth), then
        # mirror to 1024-dim if the underlying db supports it. In v0.7.0
        # we only mirror if the row has been explicitly elevated (see
        # elevate_memory_to_1024) — the auto-mode write is otherwise a
        # pure 384 write that participates in the RRF query fusion.
        memory_id = await self._db.add_memory(input)
        # Use ``asyncio.iscoroutinefunction`` to detect real 1024 support
        # — bare ``hasattr`` returns True for any MagicMock'd db, which
        # would crash the test suite (and is wrong for any non-1024 db
        # implementation).
        get_1024 = getattr(self._db, "get_memory_1024_by_memory_id", None)
        add_1024 = getattr(self._db, "add_memory_1024", None)
        if (
            get_1024 is not None
            and add_1024 is not None
            and asyncio.iscoroutinefunction(get_1024)
            and asyncio.iscoroutinefunction(add_1024)
        ):
            existing_1024 = await get_1024(memory_id)
            if existing_1024 is not None:
                # Already elevated: re-mirror with the current 384 vector
                # expanded to 1024. The DB helper expands the vector.
                expand = getattr(self._db, "_expand_384_to_1024", None)
                elevated_1024: list[float] | None = None
                if expand is not None and input.vector is not None:
                    elevated_1024 = expand(list(input.vector), 1024)
                await add_1024(memory_id, elevated_1024)
        await self._maybe_create_auto_relationship(
            memory_id, near_dup_target, near_dup_similarity
        )
        return memory_id

    async def _maybe_create_auto_relationship(
        self,
        memory_id: str,
        target_id: str | None,
        similarity: float,
    ) -> None:
        """Best-effort auto-SUPERSEDES relationship creation.

        Phase 1 feature-activation hook: if a near-duplicate was found
        in the pre-write vector search, create a SUPERSEDES
        relationship from the new memory to the existing one. Any
        failure is logged and swallowed — the write itself must never
        be affected.
        """
        if target_id is None or target_id == memory_id:
            return
        try:
            from memini_ai.memory.schema import RelationshipType

            await self.create_relationship(
                source_id=memory_id,
                target_id=target_id,
                relationship_type=RelationshipType.SUPERSEDES,
                confidence=similarity,
            )
            logger.info(
                "auto_relationship_created",
                source_id=memory_id,
                target_id=target_id,
                relationship_type="SUPERSEDES",
                similarity=similarity,
            )
        except Exception:
            logger.warning(
                "auto_relationship_failed",
                source_id=memory_id,
                target_id=target_id,
            )

    async def add_memory_by_text(
        self,
        text: str,
        source_type: MemorySourceType = MemorySourceType.session,
        source_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Add a memory entry by text with automatic embedding generation.

        Convenience method that creates a MemoryEntry and adds it.

        Args:
            text: The memory content text.
            source_type: Source type for the memory.
            source_path: Optional source path or URL.
            metadata: Optional metadata dict.
            session_id: Optional session ID.

        Returns:
            The ID of the added memory entry.

        Raises:
            ValueError: If content already exists and deduplication is enabled.
        """
        entry = MemoryEntry(
            text=text,
            sourceType=source_type,
            sourcePath=source_path,
        )
        if session_id:
            entry.session_id = session_id
        if metadata:
            import json

            entry.metadata_json = json.dumps(metadata)
        return await self.add_memory(entry)

    async def get_memory(
        self,
        memory_id: str,
        include_archived: bool = False,
    ) -> MemoryEntry | None:
        """Get a memory entry by ID.

        Args:
            memory_id: ID of the memory entry.
            include_archived: If True, include archived memories (default False).

        Returns:
            MemoryEntry if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        return await self._db.get_memory(memory_id, include_archived)

    async def memory_exists(self, memory_id: str) -> str | None:
        """Lightweight existence probe (v1.5.6 perf).

        Returns the memory id if the row exists, else None. Prefers a
        backend-native ``memory_exists`` (no vector fetch); falls back to
        :meth:`get_memory` for backends that don't implement it (e.g.
        RRFDatabase wrapper, test mocks).

        Args:
            memory_id: ID of the memory entry.

        Returns:
            The memory id if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        exists_fn = getattr(self._db, "memory_exists", None)
        if exists_fn is not None and asyncio.iscoroutinefunction(exists_fn):
            result: str | None = await exists_fn(memory_id)
            return result

        entry = await self.get_memory(memory_id)
        return entry.id if entry is not None else None

    async def get_supersession_chain(
        self,
        memory_id: str,
        max_depth: int = 10,
    ) -> list[MemoryEntry]:
        """Get the full supersession chain for a memory.

        Args:
            memory_id: ID of the memory entry.
            max_depth: Maximum chain depth (default 10).

        Returns:
            List of MemoryEntry objects in the supersession chain.
        """
        if not self._initialized:
            await self.initialize()

        return await self._db.get_supersession_chain(memory_id, max_depth)

    async def get_superseded_memory(
        self,
        memory_id: str,
    ) -> MemoryEntry | None:
        """Get the memory that this memory supersedes (parent).

        Args:
            memory_id: ID of the memory entry.

        Returns:
            MemoryEntry of the superseded memory if found, None otherwise.
        """
        if not self._initialized:
            await self.initialize()

        return await self._db.get_superseded_memory(memory_id)

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

        Dispatches on ``embedding_mode`` (v0.7.0+):
            * ``"cpu"``: legacy 384-dim search only (with the existing
              collection cascade).
            * ``"auto"``: dual-model RRF — issue a 384-dim search AND a
              1024-dim search in parallel, then fuse with the RRF
              algorithm. When ``MEMINI_IMAGE_SEARCH_ENABLED=true`` (v0.8.0+),
              a third CLIP image arm is added to the RRF fusion.
              ``query_collections`` overrides the dual-mode path (caller
              is in control of which collections to query).
            * ``"gpu"``: 1024-dim-only search (the 384-dim store is
              ignored).

        Args:
            question: Query string.
            options: Optional search options.

        Returns:
            List of matching MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        options = options or SearchOptions()
        mode = self._resolved_embedding_mode
        if mode not in {"cpu", "auto", "gpu"}:
            raise ValueError(
                f"Invalid embedding_mode '{mode}'. Must be one of: cpu, auto, gpu"
            )

        # v0.7.7: if the loaded model's dim doesn't match config.embedding_dim,
        # vector search would produce bad results (or crash the embedder).
        # Fall back to text-only search so the system stays usable.
        manager = ModelManager.get_instance()
        if manager.has_dim_mismatch:
            options.strategy = SearchStrategy.TEXT_ONLY

        # Get query collections
        collections = self._config.query_collections

        # Explicit collection control overrides mode dispatch — the caller
        # knows what they want, and that intent wins.
        if collections and len(collections) > 1:
            return await self._multi_collection_search(question, collections, options)
        if collections and len(collections) == 1:
            # Single collection with potential cascade
            results = await self._search.query(question, options)
            if not results and self._config.enable_cascade:
                fallback = self._get_fallback_collection(collections[0])
                if fallback:
                    results = await self._search.query_with_fallback_collection(
                        question, fallback, options
                    )
            return results

        # No explicit collections — dispatch on mode.
        if mode == "auto":
            return await self._query_multi_model_rrf(question, options)
        if mode == "gpu":
            return await self._query_gpu_1024(question, options)

        # mode == "cpu" (default): legacy 384-dim-only. The pre-v0.7
        # cascade-by-collection path is removed (PostgresDatabase has a
        # single ``memories`` table; cross-dim cascade is handled by
        # ``mode == "auto"``).
        return await self._search.query(question, options)

    async def _query_multi_model_rrf(
        self,
        question: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """Multi-model RRF query (auto mode): 384 + 1024 + (optional) image.

        Issues a 384-dim vector search and a 1024-dim vector search in
        parallel, then fuses the ranked lists using
        :func:`memini_ai.memory.rrf.reciprocal_rank_fusion`. Items that
        appear in both lists are boosted (their fused score is the sum
        of both contributions).

        v0.8.0: when ``MEMINI_IMAGE_SEARCH_ENABLED=true``, a **third**
        ranked list is added by calling
        ``memini_vision.ImageQuery.search_by_text`` (cross-modal CLIP
        search over the ``memories_image`` table). The three lists are
        passed to :func:`reciprocal_rank_fusion` — it already accepts
        variable-length inputs, so the 3-arm fusion is the same math
        with one extra contribution term. A memory that appears in both
        text and image lists gets both contributions summed — the
        natural boost for multi-modal agreement.

        When ``MEMINI_IMAGE_SEARCH_ENABLED=false`` (the default), the
        image arm is skipped entirely and this method is byte-for-byte
        identical to the v0.7.9 ``_query_dual_model_rrf`` behavior.

        Args:
            question: Query string.
            options: Search options.

        Returns:
            Top-K MemoryEntry objects ordered by fused RRF score.
        """
        query_1024 = getattr(self._db, "query_memories_1024", None)
        if query_1024 is None or not asyncio.iscoroutinefunction(query_1024):
            # Fall back to 384-only if the underlying db has no 1024 support.
            return await self._search.query(question, options)

        # Generate the 384-dim query vector and expand to 1024 in one shot,
        # so the two searches use the same source text.
        query_embedding = await generate_embedding(question)
        vector_384 = list(query_embedding.embedding)
        expand = getattr(self._db, "_expand_384_to_1024", None)
        vector_1024: list[float] | None = None
        if expand is not None:
            vector_1024 = expand(vector_384, 1024)
        if vector_1024 is None:
            # 1024 expansion is required for the 1024-side search; if the
            # db can't do it, fall back to 384-only.
            return await self._search.query(question, options)

        # Over-fetch so RRF has enough material to work with (otherwise
        # the two single-source top-Ks may be too small to benefit from
        # the fusion).
        fetch_k = max(options.top_k * 2, options.top_k + 5)

        # 384-dim search: reuse the existing search stack.
        # SearchOptions uses ``topK`` as its Python constructor arg
        # (pydantic Field ``alias="topK"``) — pass it positionally-by-keyword.
        search_options_384 = SearchOptions(
            topK=fetch_k,
            strategy=SearchStrategy.VECTOR_ONLY,
            threshold=options.threshold,
            exact_search=options.exact_search,
            filter=options.filter,
        )

        # v1.5.6 perf: run both fan-out arms CONCURRENTLY (the docstring
        # always claimed parallel but the awaits were sequential) and pass
        # the precomputed 384 vector into the 384-side search so the
        # question is embedded exactly ONCE per query (it was previously
        # embedded twice — once here, once inside vector_only_search).
        results_384, results_1024 = await asyncio.gather(
            self._search.vector_only_search(
                question, search_options_384, query_vector=vector_384
            ),
            query_1024(
                vector_1024,
                threshold=0.9,  # permissive — RRF will re-rank anyway
                limit=fetch_k,
            ),
        )

        # RRF fusion over memory IDs.
        ranked_ids: list[list[str]] = [
            [e.id for e in results_384],
            [e.id for e in results_1024],
        ]

        # v0.8.0: image fan-out arm (only when enabled).
        # When MEMINI_IMAGE_SEARCH_ENABLED is true, lazily import
        # memini_vision and call ImageQuery.search_by_text to get a 3rd
        # ranked list of memory_ids. The 3 lists are fused via the same
        # reciprocal_rank_fusion() — it already accepts variable-length
        # inputs. A memory with both a text AND image match gets both
        # contributions summed (multi-modal agreement boost).
        # When disabled (the default), this block is skipped entirely
        # and the query path is byte-for-byte identical to v0.7.9.
        image_entries: list[MemoryEntry] = []
        if get_config().image_search_enabled:
            try:
                image_entries = await self._image_recall_arm(question, fetch_k)
            except Exception:
                # Image search is best-effort: never let a vision failure
                # break the text query. Log and continue with text-only RRF.
                logger.warning("image_recall_arm_failed", question=question[:80])
        if image_entries:
            ranked_ids.append([e.id for e in image_entries])

        fused_ids = rrf_with_limit(
            ranked_ids, k=self._resolved_rrf_k, limit=options.top_k
        )

        # Re-hydrate MemoryEntry objects. Prefer the 384 copy (it has
        # the canonical text) and fall back to 1024 if the 384 row is
        # somehow missing.
        entries_by_id: dict[str, MemoryEntry] = {}
        for entry in results_384:
            entries_by_id[entry.id] = entry
        for entry in results_1024:
            entries_by_id.setdefault(entry.id, entry)
        # Image arm entries carry the memories text (joined in the SQL)
        # so they can serve as a fallback for IDs only in the image list.
        for entry in image_entries:
            entries_by_id.setdefault(entry.id, entry)
        return [entries_by_id[mid] for mid in fused_ids if mid in entries_by_id]

    async def _image_recall_arm(
        self,
        question: str,
        fetch_k: int,
    ) -> list[MemoryEntry]:
        """Third RRF fan-out arm: cross-modal CLIP image search (v0.8.0).

        Lazily imports :mod:`memini_vision` (so text-only users who never
        enable image search pay no import cost) and calls
        :meth:`memini_vision.ImageQuery.search_by_text` to get a ranked
        list of memory IDs whose associated images match the query text.
        The results are joined back to the ``memories`` table so the
        returned :class:`MemoryEntry` objects carry the canonical text.

        This method is ONLY called when ``MEMINI_IMAGE_SEARCH_ENABLED``
        is true. It is best-effort: any exception (CLIP model download
        failure, DB connection error, missing table) is caught by the
        caller and logged as a warning — the text RRF proceeds with 2
        lists instead of 3.

        Args:
            question: Query string (encoded via the CLIP text tower).
            fetch_k: Over-fetch limit (same as the text arms).

        Returns:
            List of :class:`MemoryEntry` from the image arm, ordered by
            ascending cosine distance. Empty list if the image table is
            empty or the search returns no matches.
        """
        # Lazy import — text-only users never pay this cost.
        # (memini_vision has a mypy override in pyproject.toml, so no
        # inline ignore is needed here.)
        from memini_vision import (
            ClipEmbedder,
            ImageIndex,
            ImageQuery,
        )

        config = get_config()
        # Resolve the DB URL for the image index (falls back to db_url).
        image_db_url = config.image_db_url or config.db_url
        embedder = ClipEmbedder(
            model_name=config.image_clip_model, device=config.image_clip_device
        )
        index = ImageIndex(image_db_url)
        query = ImageQuery(embedder, index)
        results = await query.search_by_text(question, limit=fetch_k)
        # Hydrate MemoryEntry objects from the memories table so the RRF
        # re-hydration step has the canonical text. We use the existing
        # search_image_memories path via the db if available; otherwise
        # fall back to get_memory per result (slower but correct).
        search_image = getattr(self._db, "search_image_memories", None)
        if search_image is not None and asyncio.iscoroutinefunction(search_image):
            # Re-run the search through the db helper to get joined rows.
            # The ImageQuery results already have memory_ids; we use the
            # db helper because it returns MemoryEntry objects directly.
            query_vec = embedder.encode_text(question)
            img_entries: list[MemoryEntry] = await search_image(
                query_vec, limit=fetch_k
            )
            return img_entries
        # Fallback: hydrate one-by-one via get_memory (no db helper).
        entries: list[MemoryEntry] = []
        for r in results:
            mem = await self._db.get_memory(r.memory_id, include_archived=False)
            if mem is not None:
                entries.append(mem)
        return entries

    async def _query_gpu_1024(
        self,
        question: str,
        options: SearchOptions,
    ) -> list[MemoryEntry]:
        """1024-dim-only query (gpu mode).

        Generates a 1024-dim query vector via the placeholder expansion
        of the 384-dim embedder (v0.7.0 has no real 1024 embedder
        wired in — a future version will swap in BGE-M3). Returns
        ``[]`` if the underlying db has no 1024 support.

        Args:
            question: Query string.
            options: Search options.

        Returns:
            Top-K MemoryEntry objects from the 1024 sidecar.
        """
        query_1024 = getattr(self._db, "query_memories_1024", None)
        if query_1024 is None or not asyncio.iscoroutinefunction(query_1024):
            return []
        query_embedding = await generate_embedding(question)
        vector_384 = list(query_embedding.embedding)
        expand = getattr(self._db, "_expand_384_to_1024", None)
        vector_1024: list[float] | None = None
        if expand is not None:
            vector_1024 = expand(vector_384, 1024)
        if vector_1024 is None:
            return []
        results: list[MemoryEntry] = await query_1024(
            vector_1024,
            threshold=0.9,
            limit=options.top_k,
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

        # Thought count is best-effort — not all backends implement it.
        thought_count = 0
        count_thoughts_fn = getattr(self._db, "count_thoughts", None)
        if count_thoughts_fn is not None and asyncio.iscoroutinefunction(
            count_thoughts_fn
        ):
            try:
                thought_count = int(await count_thoughts_fn())
            except Exception:
                thought_count = 0

        return {
            "total_memories": count,
            "total_thoughts": thought_count,
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

    async def count_thoughts(self) -> int:
        """Count total thoughts in the underlying store.

        Best-effort: returns 0 if the backend does not expose
        ``count_thoughts`` or the call raises.

        Returns:
            Number of thought rows (0 if unsupported).
        """
        if not self._initialized:
            await self.initialize()

        count_thoughts = getattr(self._db, "count_thoughts", None)
        if count_thoughts is None or not asyncio.iscoroutinefunction(count_thoughts):
            return 0
        try:
            return int(await count_thoughts())
        except Exception:
            return 0

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
        if not self._initialized:
            await self.initialize()

        await self._db.set_payload(memory_id, payload)

    # =============================================================================
    # MEMORY GRAPH METHODS
    # =============================================================================

    async def find_related_memories(
        self,
        memory_id: str,
        relationship_type: Any = None,
        limit: int = 10,
        include_archived: bool = True,
        max_chain_depth: int = 10,
    ) -> list[MemoryEntry]:
        """Find memories related to given memory.

        For SUPERSEDES and PARTIAL_UPDATE relationships, will traverse the
        supersession chain including archived memories.

        Args:
            memory_id: Reference memory ID.
            relationship_type: Optional filter by relationship type.
            limit: Maximum results.
            include_archived: Include archived memories for SUPERSEDES chains (default True).
            max_chain_depth: Maximum depth for supersession chain traversal (default 10).

        Returns:
            List of related MemoryEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        source = await self._db.get_memory(memory_id, include_archived=True)
        if source is None:
            return []

        results: list[MemoryEntry] = []
        seen_ids: set[str] = {memory_id}

        if relationship_type is None or str(relationship_type.value) in (
            "SUPERSEDES",
            "PARTIAL_UPDATE",
        ):
            chain = await self._db.get_supersession_chain(memory_id, max_chain_depth)
            for mem in chain:
                if mem.id not in seen_ids and len(results) < limit:
                    results.append(mem)
                    seen_ids.add(mem.id)

        for rel in source.relationships:
            if (
                relationship_type is None or rel.relationship_type == relationship_type
            ) and rel.target_id not in seen_ids:
                memory = await self._db.get_memory(rel.target_id, include_archived)
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
