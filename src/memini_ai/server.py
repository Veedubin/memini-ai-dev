"""FastMCP server for memini-ai v3.0 - MCP tools for memory operations."""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from datetime import UTC
from typing import Any, Literal

from fastmcp import FastMCP

from memini_ai.audit.logger import AuditLogger
from memini_ai.config import get_config
from memini_ai.decay import ConsolidationEngine, DecayEngine, adjust_decay_rate
from memini_ai.dialectic import DialecticEngine, get_dialectic_engine
from memini_ai.extractor import MemoryExtractor
from memini_ai.graph import MemoryGraph
from memini_ai.indexer.indexer import IndexerConfig, ProjectIndexer
from memini_ai.knowledge_graph import (
    KGQuery,
    KnowledgeGraph,
    generate_visualization_html,
)
from memini_ai.memory.schema import (
    MemoryEntry,
    MemorySourceType,
    RelationshipType,
    SearchOptions,
    SearchStrategy,
    TrustSignal,
)
from memini_ai.memory.system import MemorySystem
from memini_ai.multi_peer import MultiPeerManager, get_multi_peer_manager
from memini_ai.precompress import PrecompressExtractor
from memini_ai.rate_limiter import AsyncRateLimiter
from memini_ai.thought_chains import ThoughtChains
from memini_ai.tiered_loader import TieredLoader
from memini_ai.trust_engine import TrustEngine
from memini_ai.user_model import UserModel
from memini_ai.utils.logger import logger
from memini_ai.utils.sanitizer import (
    ContentTooLargeError,
    RateLimitExceededError,
    sanitize_content,
    validate_content_size,
)

# Operation timeout in seconds
OPERATION_TIMEOUT = 30.0


class MCPServer:
    """FastMCP server with memory and indexer tools.

    Provides 6 MCP tools:
    - query_memories: Semantic search over memories
    - add_memory: Add a new memory with deduplication
    - search_project: Search indexed project files
    - index_project: Trigger project indexing (background mode supported)
    - get_file_contents: Reconstruct file from indexed chunks
    - get_status: Get server component status

    Features:
    - Graceful degradation when database unavailable
    - Memory initialization with exponential backoff retry
    - Background job tracking for indexing
    - SIGINT/SIGTERM graceful shutdown
    """

    def __init__(self) -> None:
        """Initialize MCP server."""
        self._memory_system: MemorySystem | None = None
        self._indexer: ProjectIndexer | None = None
        self._trust_engine: TrustEngine | None = None
        self._memory_graph: MemoryGraph | None = None
        self._knowledge_graph: KnowledgeGraph | None = None
        self._extractor: MemoryExtractor | None = None
        self._precompress: PrecompressExtractor | None = None
        self._tiered_loader: TieredLoader | None = None
        self._user_model: UserModel | None = None
        self._decay_engine: DecayEngine | None = None
        self._consolidation_engine: ConsolidationEngine | None = None
        self._multi_peer_manager: MultiPeerManager | None = None
        self._dialectic_engine: DialecticEngine | None = None
        self._thought_chains: ThoughtChains | None = None
        self._audit_logger: AuditLogger | None = None
        self._init_error: str | None = None
        self._background_jobs: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._shutdown_in_progress: bool = False
        self._stdio_watch_task: asyncio.Task[None] | None = None
        # Phase 2.1: Rate limiter for add_memory
        config = get_config()
        self._rate_limiter = AsyncRateLimiter(
            max_requests=config.rate_limit_per_minute,
            window_seconds=60,
        )
        self._mcp = FastMCP("memini-ai")
        self._setup_tools()
        self._setup_signal_handlers()

    def _setup_tools(self) -> None:
        """Register all MCP tools."""
        self._mcp.add_tool(self.query_memories)
        self._mcp.add_tool(self.add_memory)
        self._mcp.add_tool(self.search_project)
        self._mcp.add_tool(self.index_project)
        self._mcp.add_tool(self.get_file_contents)
        self._mcp.add_tool(self.get_status)
        # v0.7.3: end-to-end write+read health probe
        self._mcp.add_tool(self.healthcheck)
        self._mcp.add_tool(self.get_trust_score)
        self._mcp.add_tool(self.adjust_trust)
        self._mcp.add_tool(self.list_archived)
        self._mcp.add_tool(self.find_related_memories)
        self._mcp.add_tool(self.create_relationship)
        self._mcp.add_tool(self.get_relationship_summary)
        self._mcp.add_tool(self.trigger_extraction)
        self._mcp.add_tool(self.preconpress_extraction)
        self._mcp.add_tool(self.get_tier0_summary)
        self._mcp.add_tool(self.get_tier1_summary)
        self._mcp.add_tool(self.get_user_profile)
        self._mcp.add_tool(self.update_user_profile)
        self._mcp.add_tool(self.get_decay_status)
        self._mcp.add_tool(self.trigger_consolidation)
        self._mcp.add_tool(self.list_fading_memories)
        self._mcp.add_tool(self.adjust_decay_rate)
        # Phase 4B: Knowledge Graph tools
        self._mcp.add_tool(self.query_kg)
        self._mcp.add_tool(self.extract_entities)
        self._mcp.add_tool(self.get_entity_graph)
        self._mcp.add_tool(self.get_inference_chain)
        self._mcp.add_tool(self.search_entities)
        self._mcp.add_tool(self.get_graph_visualization)
        # Phase 4C: Multi-Peer tools
        self._mcp.add_tool(self.list_peers)
        self._mcp.add_tool(self.add_peer)
        self._mcp.add_tool(self.switch_peer_context)
        self._mcp.add_tool(self.share_memory)
        self._mcp.add_tool(self.get_peer_memories)
        self._mcp.add_tool(self.get_shared_memories)
        # Phase 4D: Dialectic tools
        self._mcp.add_tool(self.find_contradictions)
        self._mcp.add_tool(self.resolve_contradiction)
        self._mcp.add_tool(self.get_dialectic_history)
        self._mcp.add_tool(self.challenge_memory)
        # Phase 5: Thought Chains tools
        self._mcp.add_tool(self.add_thought)
        self._mcp.add_tool(self.start_thought_chain)
        self._mcp.add_tool(self.get_thought_chain)
        self._mcp.add_tool(self.get_related_chains)
        self._mcp.add_tool(self.revise_thought)
        self._mcp.add_tool(self.branch_thought)
        self._mcp.add_tool(self.pause_thought_chain)
        self._mcp.add_tool(self.resume_thought_chain)
        self._mcp.add_tool(self.abandon_thought_chain)
        # Phase 2.3: Audit logging tools
        self._mcp.add_tool(self.log_audit_event)
        self._mcp.add_tool(self.get_audit_log)
        self._mcp.add_tool(self.get_security_summary)
        # v0.7.0: Dual-model RRF
        self._mcp.add_tool(self.elevate_memory_to_1024)
        # Kanban cards (GitHub triage poller integration)
        self._mcp.add_tool(self.kanban_add_card)
        self._mcp.add_tool(self.kanban_move_card)
        self._mcp.add_tool(self.kanban_list_cards)
        self._mcp.add_tool(self.kanban_get_card)

    def _setup_signal_handlers(self) -> None:
        """Set up SIGINT/SIGTERM handlers."""
        try:
            loop = asyncio.get_running_loop()

            async def shutdown_handler(sig_num: int) -> None:
                await self._shutdown()

            def signal_handler(sig: signal.Signals) -> None:
                def signal_callback(s: signal.Signals = sig) -> None:
                    if self._shutdown_in_progress:
                        logger.warning(
                            "shutdown_already_in_progress",
                            signal=s.name,
                        )
                        return
                    self._shutdown_in_progress = True
                    asyncio.create_task(shutdown_handler(s))

                loop.add_signal_handler(sig, signal_callback)

            for sig in (signal.SIGINT, signal.SIGTERM):
                signal_handler(sig)
        except (NotImplementedError, AttributeError, RuntimeError):
            # Windows or other platforms without signal handlers
            pass

    async def _init_memory_system(self) -> MemorySystem:
        """Initialize memory system with exponential backoff retry."""
        system = MemorySystem()

        # Exponential backoff retry for database connection
        max_attempts = 3
        base_delay = 1.0
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                await system.initialize()
                # v0.7.7: auto-detect new deployments (0 memories) and default
                # to BGE-M3. Only fires when auto_detect_model is True and
                # the model_name is still at the factory default.
                from memini_ai.model.manager import ModelManager

                count = await system._db.count_memories()
                overridden = await ModelManager.auto_detect_model(memory_count=count)
                if overridden:
                    # The model_name was changed — re-initialize the memory
                    # system so the new model_name is picked up by the
                    # ModelManager singleton on first acquire().
                    await system.close()
                    system = MemorySystem()
                    await system.initialize()
                    count = await system._db.count_memories()
                # Initialize trust engine with memory system
                self._trust_engine = TrustEngine(memory_system=system)
                # Initialize memory graph with memory system
                self._memory_graph = MemoryGraph(memory_system=system)
                # Initialize knowledge graph with memory system
                self._knowledge_graph = KnowledgeGraph(memory_system=system)
                # Initialize extractor with memory system
                self._extractor = MemoryExtractor(memory_system=system)
                # Initialize precompress extractor with memory system and extractor
                self._precompress = PrecompressExtractor(
                    memory_system=system,
                    extractor=self._extractor,
                )
                # Initialize tiered loader with memory system
                self._tiered_loader = TieredLoader(memory_system=system)
                # Initialize user model with memory system
                self._user_model = UserModel(memory_system=system)
                # Initialize decay and consolidation engines
                self._decay_engine = DecayEngine(memory_system=system)
                self._consolidation_engine = ConsolidationEngine(memory_system=system)
                # Initialize multi-peer manager
                self._multi_peer_manager = get_multi_peer_manager(memory_system=system)
                # Initialize thought chains (requires DB pool from postgres backend)
                from memini_ai.config import get_config as _get_config

                _config = _get_config()
                if (
                    _config.thought_chains_enabled
                    and hasattr(system._db, "_pool")
                    and system._db._pool is not None
                ):
                    self._thought_chains = ThoughtChains(
                        pool=system._db._pool,
                        memory_system=system,
                        trust_engine=self._trust_engine,
                    )
                # Phase 2.3: Initialize audit logger
                if hasattr(system._db, "_pool") and system._db._pool is not None:
                    self._audit_logger = AuditLogger(db_pool=system._db._pool)
                    await self._audit_logger.start()
                    logger.info("audit_logger_initialized")

                # Phase 1 feature-activation: Multi-Peer auto-registration
                # (Layer C, startup hook). When multi_peer_enabled is ON,
                # auto-register a default "owner" peer on server start.
                # Idempotent: list_peers() check ensures we only register
                # if no peers exist, so re-starts are safe. Failure is
                # isolated — never blocks server startup.
                if (
                    _config.multi_peer_enabled
                    and self._multi_peer_manager is not None
                    and self._multi_peer_manager.is_enabled
                ):
                    try:
                        peers = await self._multi_peer_manager.list_peers()
                        if peers.get("count", 0) == 0:
                            await self._multi_peer_manager.add_peer(
                                peer_id="owner",
                                name="Default Owner",
                                role="owner",
                                trust_level=1.0,
                            )
                            logger.info("peer_auto_registered", peer_id="owner")
                    except Exception:
                        logger.warning("peer_auto_register_failed")

                return system
            except Exception as e:
                last_error = e
                delay = base_delay * (2**attempt)
                logger.warning(
                    "memory_init_retry",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    delay=delay,
                    error=str(e),
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)

        # All retries failed - return system in degraded mode
        logger.error("memory_init_failed", error=str(last_error))
        self._init_error = str(last_error)
        return system

    async def _init_indexer(self) -> ProjectIndexer:
        """Initialize project indexer."""
        config = get_config()
        indexer_config = IndexerConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_file_size=config.max_file_size,
            db_path=".memini/indexer.db",
        )
        return ProjectIndexer(indexer_config)

    # =========================================================================
    # TOOL: query_memories
    # =========================================================================
    async def query_memories(
        self,
        query: str,
        limit: int = 10,
        strategy: str = "tiered",
    ) -> dict[str, Any]:
        """Query memories with semantic search.

        Args:
            query: Search query string.
            limit: Maximum number of results (default 10).
            strategy: Search strategy - "tiered", "vector_only", "text_only", or "parallel" (default "tiered").

        Returns:
            Dictionary with count, memories list, and strategy_used.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Map strategy string to enum
            strat_map = {
                "tiered": SearchStrategy.TIERED,
                "vector_only": SearchStrategy.VECTOR_ONLY,
                "text_only": SearchStrategy.TEXT_ONLY,
                "parallel": SearchStrategy.PARALLEL,
            }
            strat = strat_map.get(strategy.lower(), SearchStrategy.TIERED)

            options = SearchOptions(topK=limit, strategy=strat)
            results = await asyncio.wait_for(
                self._memory_system.query_memories(query, options),
                timeout=OPERATION_TIMEOUT,
            )

            # Convert to dict format
            memories = []
            for entry in results[:limit]:
                entry_dict = entry.model_dump(by_alias=True)
                # Convert datetime to ISO format
                if entry_dict.get("timestamp"):
                    entry_dict["timestamp"] = entry.timestamp.isoformat()
                memories.append(entry_dict)

            return {
                "count": len(memories),
                "memories": memories,
                "strategy_used": strat.value,
            }
        except TimeoutError:
            logger.error("query_memories_timeout", query=query)
            return {
                "count": 0,
                "memories": [],
                "strategy_used": strategy,
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("query_memories_error", error=str(e), query=query)
            return {
                "count": 0,
                "memories": [],
                "strategy_used": strategy,
                "error": str(e),
            }

    # =========================================================================
    # TOOL: add_memory
    # =========================================================================
    async def add_memory(
        self,
        content: str,
        sourceType: str = "manual",  # noqa: N803
        sourcePath: str | None = None,  # noqa: N803
        metadata: dict[str, Any] | None = None,
        peerId: str | None = None,  # noqa: N803
        supersedesId: str | None = None,  # noqa: N803
        structuredFields: dict[str, Any] | None = None,  # noqa: N803
        changeRatio: float = 1.0,  # noqa: N803
    ) -> dict[str, Any]:
        """Add a new memory entry with deduplication.

        Args:
            content: The memory content text.
            sourceType: Source type - "session", "file", "web", "boomerang", "project" (default "manual").
            sourcePath: Optional source path or URL.
            metadata: Optional metadata dictionary.
            peerId: Optional peer ID for rate limiting (defaults to "default").
            supersedesId: Optional ID of memory this partially updates (for PARTIAL_UPDATE relationships).
            structuredFields: Optional key-value fields for granular merge.
            changeRatio: Fraction of content that is new/changed (0.0-1.0). Default 1.0 = full replacement.

        Returns:
            Dictionary with success status, memory ID, and message.
        """
        config = get_config()

        # Phase 2.1: Rate limiting check
        effective_peer_id = peerId or sourcePath or "default"
        try:
            await self._rate_limiter.check_and_raise(effective_peer_id)
        except RateLimitExceededError as e:
            logger.warning(
                "add_memory_rate_limited",
                peer_id=effective_peer_id,
                limit=config.rate_limit_per_minute,
            )
            return {
                "success": False,
                "id": "",
                "message": str(e),
                "error": f"rate_limit_exceeded: {e}",
            }

        # Phase 2.1: Content size validation
        try:
            validate_content_size(content, config.max_memory_content_size)
        except ContentTooLargeError as e:
            logger.warning(
                "add_memory_content_too_large",
                content_size=e.content_size,
                max_size=e.max_size,
            )
            return {
                "success": False,
                "id": "",
                "message": str(e),
                "error": f"content_too_large: {e}",
            }

        # Phase 2.1: Content sanitization
        if config.sanitize_content:
            content = sanitize_content(content)

        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Parse source type
            try:
                src_type = MemorySourceType(sourceType)
            except ValueError:
                src_type = MemorySourceType.session

            # Create memory entry with delta fields
            entry = MemoryEntry(
                text=content,
                sourceType=src_type,
                sourcePath=sourcePath,
                supersedesId=supersedesId,
                structuredFields=structuredFields,
                changeRatio=changeRatio,
            )

            # Add metadata if provided
            import json

            if metadata:
                entry.metadata_json = json.dumps(metadata)

            # Try to add (may raise on duplicate)
            memory_id = await asyncio.wait_for(
                self._memory_system.add_memory(entry), timeout=OPERATION_TIMEOUT
            )

            # Phase 2.3 (v0.7.3): Post-write read-back verification.
            # Confirms the write is actually retrievable via the read
            # path. If get_memory returns None, the write was silently
            # lost (or the read path is broken) — surface it to the
            # caller instead of claiming success.
            try:
                verified = await asyncio.wait_for(
                    self._memory_system.get_memory(memory_id),
                    timeout=OPERATION_TIMEOUT,
                )
            except Exception as e:
                logger.error(
                    "add_memory_post_write_readback_failed",
                    memory_id=memory_id,
                    error=str(e),
                )
                verified = None
                # Preserve the readback exception for the error response
                _readback_exc = e
            else:
                _readback_exc = None

            if verified is None:
                readback_detail = (
                    f" (readback error: {_readback_exc})"
                    if _readback_exc is not None
                    else ""
                )
                logger.error(
                    "add_memory_post_write_readback_failed",
                    memory_id=memory_id,
                )
                return {
                    "success": False,
                    "id": memory_id,
                    "message": f"Write succeeded but read-back failed{readback_detail}",
                    "error": f"post_write_readback_failed{readback_detail}",
                }

            # Phase 2.3: Audit log for memory mutation
            if self._audit_logger is not None:
                self._audit_logger.log(
                    "memory_mutation",
                    severity="info",
                    tool_name="add_memory",
                    memory_id=memory_id,
                    description=f"Memory added: {content[:100]}",
                    details={
                        "source_type": sourceType,
                        "content_length": len(content),
                        "supersedes_id": supersedesId,
                        "readback_verified": True,
                    },
                )

            # Phase 1 feature-activation: KG entity extraction hook
            # (Layer A, synchronous). When knowledge_graph_enabled is
            # ON, extract entities from the memory text via the regex
            # EntityExtractor (zero LLM) and register them in the KG.
            # Failure is isolated: any exception is logged and the
            # add_memory response is unaffected.
            if config.knowledge_graph_enabled and self._knowledge_graph is not None:
                try:
                    await self._knowledge_graph.extract_and_register_entities(content)
                except Exception:
                    logger.warning(
                        "kg_auto_extract_failed",
                        memory_id=memory_id,
                    )

            return {
                "success": True,
                "id": memory_id,
                "message": "Memory added successfully",
            }
        except TimeoutError:
            logger.error("add_memory_timeout", content_length=len(content))
            return {
                "success": False,
                "id": "",
                "message": "Operation timed out",
                "error": "Operation timed out",
            }
        except ValueError as e:
            # Duplicate content
            return {"success": False, "id": "", "message": str(e), "error": str(e)}
        except Exception as e:
            logger.error("add_memory_error", error=str(e))
            return {"success": False, "id": "", "message": str(e), "error": str(e)}

    # =========================================================================
    # TOOL: search_project
    # =========================================================================
    async def search_project(
        self,
        query: str,
        topK: int = 20,  # noqa: N803
        fileTypes: list[str] | None = None,  # noqa: N803
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search indexed project files for matching chunks.

        Args:
            query: Search query string.
            topK: Maximum number of results (default 20).
            fileTypes: Optional list of file extensions to filter (e.g., [".py", ".ts"]).
            paths: Optional list of directory paths to filter.

        Returns:
            Dictionary with count and list of matching chunks.
        """
        try:
            if self._indexer is None:
                self._indexer = await asyncio.wait_for(
                    self._init_indexer(), timeout=OPERATION_TIMEOUT
                )

            # Run search
            results = await asyncio.wait_for(
                self._indexer.search(query, {"top_k": topK}), timeout=OPERATION_TIMEOUT
            )

            chunks: list[dict[str, Any]] = []
            for result in results[:topK]:
                chunks.append(
                    {
                        "path": result.path,
                        "content": result.content,
                        "chunk_index": result.chunk_index,
                        "total_chunks": result.total_chunks,
                        "score": result.score,
                    }
                )

            return {"count": len(chunks), "chunks": chunks}
        except TimeoutError:
            logger.error("search_project_timeout", query=query)
            return {"count": 0, "chunks": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("search_project_error", error=str(e), query=query)
            return {"count": 0, "chunks": [], "error": str(e)}

    # =========================================================================
    # TOOL: index_project
    # =========================================================================
    async def index_project(
        self,
        path: str | None = None,
        force: bool = False,
        background: bool = True,
    ) -> dict[str, Any]:
        """Trigger project indexing with optional background mode.

        Args:
            path: Directory path to index (default: current directory).
            force: Force re-indexing of all files (default False).
            background: Run indexing in background, return jobId immediately (default True).

        Returns:
            Dictionary with success status, jobId, status, message, and stats.
        """
        try:
            if self._indexer is None:
                self._indexer = await asyncio.wait_for(
                    self._init_indexer(), timeout=OPERATION_TIMEOUT
                )

            target_path = path or "."
            indexer = self._indexer

            if background:
                # Start background indexing task
                job_id = str(uuid.uuid4())

                async def _background_index() -> dict[str, Any]:
                    """Background indexing task."""
                    try:
                        if force:
                            indexer.clear_index()

                        # Start indexer if not running
                        if not indexer.is_running:
                            await indexer.set_root_path(target_path)
                            await indexer.start()

                        # Index directory
                        count = await indexer.index_directory(target_path)
                        stats = indexer.get_stats()

                        return {
                            "success": True,
                            "jobId": job_id,
                            "status": "completed",
                            "message": f"Indexed {count} files",
                            "stats": {
                                "files_indexed": stats.files_indexed,
                                "chunks_created": stats.chunks_created,
                                "bytes_processed": stats.bytes_processed,
                                "errors": stats.errors,
                            },
                        }
                    except Exception as e:
                        logger.error(
                            "background_index_error", job_id=job_id, error=str(e)
                        )
                        return {
                            "success": False,
                            "jobId": job_id,
                            "status": "failed",
                            "message": str(e),
                            "stats": None,
                        }

                task = asyncio.create_task(_background_index())
                self._background_jobs[job_id] = task

                return {
                    "success": True,
                    "jobId": job_id,
                    "status": "running",
                    "message": f"Indexing started in background (jobId: {job_id})",
                    "stats": None,
                }
            # Synchronous indexing
            if force:
                indexer.clear_index()

            if not indexer.is_running:
                await indexer.set_root_path(target_path)
                await indexer.start()

            count = await indexer.index_directory(target_path)
            stats = indexer.get_stats()

            return {
                "success": True,
                "jobId": None,
                "status": "completed",
                "message": f"Indexed {count} files",
                "stats": {
                    "files_indexed": stats.files_indexed,
                    "chunks_created": stats.chunks_created,
                    "bytes_processed": stats.bytes_processed,
                    "errors": stats.errors,
                },
            }
        except TimeoutError:
            logger.error("index_project_timeout", path=path)
            return {
                "success": False,
                "jobId": None,
                "status": "timeout",
                "message": "Operation timed out",
                "stats": None,
            }
        except Exception as e:
            logger.error("index_project_error", error=str(e), path=path)
            return {
                "success": False,
                "jobId": None,
                "status": "failed",
                "message": str(e),
                "stats": None,
            }

    # =========================================================================
    # TOOL: get_file_contents
    # =========================================================================
    async def get_file_contents(
        self,
        filePath: str,  # noqa: N803
        triggerIndex: bool = False,  # noqa: N803
    ) -> dict[str, Any]:
        """Reconstruct file contents from indexed chunks.

        Args:
            filePath: Path to the file to reconstruct.
            triggerIndex: Trigger indexing if file not found (default False).

        Returns:
            Dictionary with success status, filePath, content, chunks info, lineCount, and truncated.
        """
        try:
            if self._indexer is None:
                self._indexer = await asyncio.wait_for(
                    self._init_indexer(), timeout=OPERATION_TIMEOUT
                )

            result = await asyncio.wait_for(
                self._indexer.get_file_contents(filePath), timeout=OPERATION_TIMEOUT
            )

            if result is None:
                # File not in index
                if triggerIndex:
                    # Trigger indexing first
                    if not self._indexer.is_running:
                        await self._indexer.set_root_path(".")
                        await self._indexer.start()
                    await self._indexer.index_directory(".")

                # Try again
                result = await asyncio.wait_for(
                    self._indexer.get_file_contents(filePath), timeout=OPERATION_TIMEOUT
                )

            if result is None:
                return {
                    "success": False,
                    "filePath": filePath,
                    "content": "",
                    "chunks": [],
                    "lineCount": 0,
                    "truncated": False,
                    "error": "File not found in index",
                }

            # Count lines
            line_count = len(result.content.splitlines())

            # Truncate if too long (10k lines max)
            max_lines = 10000
            truncated = False
            content = result.content
            if line_count > max_lines:
                truncated = True
                content = "\n".join(result.content.splitlines()[:max_lines])

            chunks: list[dict[str, Any]] = []
            # Would need chunk info from indexer - for now return empty
            # TODO: Integrate with chunker to get proper chunk info

            return {
                "success": True,
                "filePath": filePath,
                "content": content,
                "chunks": chunks,
                "lineCount": line_count,
                "truncated": truncated,
            }
        except TimeoutError:
            logger.error("get_file_contents_timeout", filePath=filePath)
            return {
                "success": False,
                "filePath": filePath,
                "content": "",
                "chunks": [],
                "lineCount": 0,
                "truncated": False,
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("get_file_contents_error", error=str(e), filePath=filePath)
            return {
                "success": False,
                "filePath": filePath,
                "content": "",
                "chunks": [],
                "lineCount": 0,
                "truncated": False,
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_status
    # =========================================================================
    async def get_status(self) -> dict[str, Any]:
        """Get server status for all components.

        Returns a status report distinguishing **warming** (components still
        initializing, retry shortly) from **ready** (operational) from
        **error** (initialization failed) from **degraded** (ready but some
        optional components offline).

        T-STATUS-001 (2026-07-28): prior versions reported
        ``memoryReady: false`` + ``memoryCount: 0`` during the warm-up window
        (5-15s after server start) with no signal that the system was merely
        initializing rather than broken. Agents misread this as a
        misconfiguration. The response now carries three new fields:

        - ``status`` (``"warming" | "ready" | "degraded" | "error"``) — a
          single top-level state an agent can switch on.
        - ``warming`` (bool) — explicit ``true`` while components are still
          initializing and the caller should retry shortly.
        - ``warmingMessage`` (str | None) — human-readable explanation when
          ``warming`` is true (e.g. "memory subsystem still initializing;
          retry get_status in a few seconds").

        During warming, ``memoryCount`` / ``thoughtsCount`` /
        ``kanbanCardCount`` are reported as ``null`` (not ``0``) so an agent
        cannot misread a transient zero as "the database is empty". The
        legacy boolean fields (``memoryReady`` etc.) are retained for
        backward compatibility and remain honest — they are ``false`` while
        warming, ``true`` once ready.

        Returns:
            Dictionary with status, warming, warmingMessage, memoryReady,
            modelReady, indexerReady, memoryCount (int | None), and initError.
        """
        # ── Determine warming vs error vs ready state ────────────────────────
        # Warming = no init error recorded AND memory subsystem not yet ready
        # (either _memory_system is None because no tool has lazily triggered
        # _init_memory_system(), or it exists but the DB pool / embedding
        # model is still coming up).
        has_init_error = self._init_error is not None
        memory_ready = False
        if self._memory_system is not None:
            memory_ready = self._memory_system.is_ready
        memory_system_present = self._memory_system is not None

        is_warming = (not has_init_error) and (not memory_ready)
        warming_message: str | None = None
        if is_warming:
            if not memory_system_present:
                warming_message = (
                    "memory subsystem still initializing; retry get_status "
                    "in a few seconds (no tool has triggered lazy init yet)"
                )
            else:
                warming_message = (
                    "memory subsystem initializing (DB pool / embedding "
                    "model warming up); retry get_status in a few seconds"
                )

        # ── Compute top-level status ─────────────────────────────────────────
        if has_init_error:
            status: str = "error"
        elif is_warming:
            status = "warming"
        elif memory_ready:
            status = "ready"
        else:
            status = "degraded"

        # Best-effort row counts for observability (v0.7.3).
        # During warming the counts are unknown — report null (NOT 0) so an
        # agent cannot misread a transient zero as "the database is empty".
        memory_count: int | None = None
        thoughts_count: int | None = None
        kanban_count: int | None = None
        query_latency_ms: float | None = None
        if memory_ready and self._memory_system is not None:
            try:
                import time

                # Counts are known once ready; default to 0 on per-call
                # failure (the DB is reachable but this particular count
                # raised — surfacing 0 is honest because we cannot prove
                # the table is empty either, but we *are* ready).
                memory_count = 0
                thoughts_count = 0
                kanban_count = 0
                t0 = time.monotonic()
                memory_count = await asyncio.wait_for(
                    self._memory_system._db.count_memories(),
                    timeout=OPERATION_TIMEOUT,
                )
                # count_thoughts is best-effort — may not be implemented
                # on all backends.
                count_thoughts_fn = getattr(
                    self._memory_system._db, "count_thoughts", None
                )
                if count_thoughts_fn is not None and asyncio.iscoroutinefunction(
                    count_thoughts_fn
                ):
                    try:
                        thoughts_count = await asyncio.wait_for(
                            count_thoughts_fn(), timeout=OPERATION_TIMEOUT
                        )
                    except Exception:
                        thoughts_count = 0
                # count_kanban_cards is best-effort — only supported on
                # PostgresDatabase (the kanban_cards table is created at
                # startup via _ensure_schema).
                count_kanban_fn = getattr(
                    self._memory_system._db, "count_kanban_cards", None
                )
                if count_kanban_fn is not None and asyncio.iscoroutinefunction(
                    count_kanban_fn
                ):
                    try:
                        kanban_count = await asyncio.wait_for(
                            count_kanban_fn(), timeout=OPERATION_TIMEOUT
                        )
                    except Exception:
                        kanban_count = 0
                query_latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
            except Exception as e:
                logger.warning(
                    "get_status_count_failed",
                    error=str(e),
                )
                memory_count = 0
                thoughts_count = 0
                kanban_count = 0

        # Check model (always ready if we got here)
        model_ready = True
        model_name: str | None = None
        model_dimension: int | None = None
        embedding_dim_mismatch = False
        embedding_dim_expected: int | None = None
        embedding_dim_actual: int | None = None
        try:
            from memini_ai.model.manager import ModelManager

            manager = ModelManager.get_instance()
            model_ready = manager.get_dimensions() > 0
            # v0.7.7: expose model + dim mismatch details for observability.
            model_name = manager._model_id
            model_dimension = manager._dimensions
            embedding_dim_mismatch = manager.has_dim_mismatch
            embedding_dim_expected = manager.embedding_dim
            # actual_dim is the model's real output dim (may differ from
            # config.embedding_dim when there's a mismatch).
            if manager._model is not None:
                embedding_dim_actual = manager._model.get_embedding_dimension()
            elif model_dimension is not None:
                embedding_dim_actual = model_dimension
        except Exception:
            pass

        # Check indexer
        indexer_ready = self._indexer is not None and self._indexer.is_running

        # Check trust engine
        trust_ready = self._trust_engine is not None and self._trust_engine.is_enabled

        # Check memory graph
        graph_ready = self._memory_graph is not None and self._memory_graph.is_enabled

        # Check extractor
        extractor_ready = self._extractor is not None and self._extractor.is_enabled

        # Check precompress extractor
        precompress_ready = (
            self._precompress is not None and self._precompress.is_enabled
        )

        # Check tiered loader
        tiered_ready = (
            self._tiered_loader is not None and self._tiered_loader.is_enabled
        )

        # Check user model
        user_model_ready = self._user_model is not None and self._user_model.is_enabled

        # Check knowledge graph
        kg_ready = (
            self._knowledge_graph is not None and self._knowledge_graph.is_enabled
        )

        # Check multi-peer manager
        multi_peer_ready = (
            self._multi_peer_manager is not None and self._multi_peer_manager.is_enabled
        )

        # Check dialectic engine
        dialectic_ready = (
            self._dialectic_engine is not None and self._dialectic_engine.is_enabled
        )

        # Check thought chains
        thought_chains_ready = (
            self._thought_chains is not None and self._thought_chains.is_enabled
        )

        return {
            # T-STATUS-001: top-level state for agent switching + warming
            # signal so callers can distinguish "still starting" from
            # "broken". memoryReady=false alone was ambiguous during the
            # warm-up window.
            "status": status,
            "warming": is_warming,
            "warmingMessage": warming_message,
            "memoryReady": memory_ready,
            "modelReady": model_ready,
            "indexerReady": indexer_ready,
            "trustEngineReady": trust_ready,
            "memoryGraphReady": graph_ready,
            "knowledgeGraphReady": kg_ready,
            "extractorReady": extractor_ready,
            "precompressReady": precompress_ready,
            "tieredLoadingReady": tiered_ready,
            "userModelingReady": user_model_ready,
            "multiPeerReady": multi_peer_ready,
            "dialecticReady": dialectic_ready,
            "thoughtChainsReady": thought_chains_ready,
            "memoryCount": memory_count,
            "thoughtsCount": thoughts_count,
            "kanbanCardCount": kanban_count,
            "queryLatencyMs": query_latency_ms,
            "modelName": model_name,
            "modelDimension": model_dimension,
            "embeddingDimMismatch": embedding_dim_mismatch,
            "embeddingDimExpected": embedding_dim_expected,
            "embeddingDimActual": embedding_dim_actual,
            "initError": self._init_error,
        }

    # =========================================================================
    # TOOL: healthcheck  (v0.7.3)
    # =========================================================================
    async def healthcheck(self) -> dict[str, Any]:
        """End-to-end write + read health probe.

        Writes a marker memory, immediately reads it back, and reports
        whether the round-trip succeeded along with latency metrics.
        Use this to verify the memory subsystem is wired correctly
        (storage + read path) without running a real semantic query.

        Returns:
            Dictionary with ``status`` ("pass" | "fail"), ``memoryId``,
            ``writeLatencyMs``, ``readLatencyMs``, ``readbackMatch``,
            and optionally ``error``.
        """
        marker = f"healthcheck_marker_{uuid.uuid4().hex[:8]}"
        result: dict[str, Any] = {
            "status": "fail",
            "memoryId": None,
            "writeLatencyMs": None,
            "readLatencyMs": None,
            "readbackMatch": False,
            "error": None,
        }

        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # 1. Write the marker memory.
            import time

            t_write = time.monotonic()
            entry = MemoryEntry(
                text=marker,
                sourceType=MemorySourceType.session,
            )
            entry.metadata_json = '{"type":"healthcheck_marker"}'
            memory_id = await asyncio.wait_for(
                self._memory_system.add_memory(entry),
                timeout=OPERATION_TIMEOUT,
            )
            result["memoryId"] = memory_id
            result["writeLatencyMs"] = round((time.monotonic() - t_write) * 1000.0, 2)

            # 2. Read it back.
            t_read = time.monotonic()
            readback = await asyncio.wait_for(
                self._memory_system.get_memory(memory_id),
                timeout=OPERATION_TIMEOUT,
            )
            result["readLatencyMs"] = round((time.monotonic() - t_read) * 1000.0, 2)

            # 3. Compare.
            if readback is not None and getattr(readback, "text", None) == marker:
                result["readbackMatch"] = True
                result["status"] = "pass"
            else:
                result["error"] = "readback_mismatch"
                logger.error(
                    "healthcheck_readback_mismatch",
                    memory_id=memory_id,
                    marker=marker,
                )
        except TimeoutError:
            result["error"] = "timeout"
            logger.error("healthcheck_timeout")
        except Exception as e:
            result["error"] = str(e)
            logger.error("healthcheck_error", error=str(e))

        # 4. Audit-log critical failures.
        if result["status"] == "fail" and self._audit_logger is not None:
            self._audit_logger.log(
                "tool_invocation",
                severity="critical",
                tool_name="healthcheck",
                description=f"healthcheck failed: {result['error']}",
                details=result,
            )

        return result

    # =========================================================================
    # TOOL: get_trust_score
    # =========================================================================
    async def get_trust_score(self, memory_id: str) -> dict[str, Any]:
        """Get trust score and level for a memory entry.

        Args:
            memory_id: ID of the memory entry.

        Returns:
            Dictionary with id, trustScore, trustLevel, retrievalCount, isArchived.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._trust_engine is None:
                self._trust_engine = TrustEngine(memory_system=self._memory_system)

            result = await asyncio.wait_for(
                self._trust_engine.get_trust_score(memory_id),
                timeout=OPERATION_TIMEOUT,
            )

            if result is None:
                return {
                    "id": memory_id,
                    "trustScore": None,
                    "trustLevel": None,
                    "retrievalCount": None,
                    "isArchived": None,
                    "error": "Trust engine disabled or memory not found",
                }

            return result
        except TimeoutError:
            logger.error("get_trust_score_timeout", memory_id=memory_id)
            return {"error": "Operation timed out"}
        except Exception as e:
            logger.error("get_trust_score_error", error=str(e), memory_id=memory_id)
            return {"error": str(e)}

    # =========================================================================
    # TOOL: adjust_trust
    # =========================================================================
    async def adjust_trust(self, memory_id: str, signal: str) -> dict[str, Any]:
        """Adjust trust score for a memory based on feedback signal.

        Args:
            memory_id: ID of the memory entry.
            signal: Trust signal - "agent_used", "agent_ignored", "user_corrected", "user_confirmed".

        Returns:
            Dictionary with success status, memoryId, oldScore, newScore, action.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._trust_engine is None:
                self._trust_engine = TrustEngine(memory_system=self._memory_system)

            # Parse signal
            try:
                trust_signal = TrustSignal(signal.lower())
            except ValueError:
                return {
                    "success": False,
                    "memoryId": memory_id,
                    "oldScore": None,
                    "newScore": None,
                    "action": None,
                    "error": f"Invalid signal: {signal}",
                }

            result = await asyncio.wait_for(
                self._trust_engine.adjust_trust(memory_id, trust_signal),
                timeout=OPERATION_TIMEOUT,
            )

            if result is None:
                return {
                    "success": False,
                    "memoryId": memory_id,
                    "oldScore": None,
                    "newScore": None,
                    "action": None,
                    "error": "Trust engine disabled or memory not found",
                }

            # Phase 2.3: Audit log for trust adjustment
            if self._audit_logger is not None:
                self._audit_logger.log(
                    "trust_adjustment",
                    severity="info",
                    tool_name="adjust_trust",
                    memory_id=memory_id,
                    description=f"Trust {signal}: {result.old_score:.3f} -> {result.new_score:.3f}",
                    details={
                        "signal": signal,
                        "old_score": result.old_score,
                        "new_score": result.new_score,
                        "action": result.action,
                    },
                    state_before={"trust_score": result.old_score},
                    state_after={"trust_score": result.new_score},
                )

            return {
                "success": True,
                "memoryId": result.memory_id,
                "oldScore": result.old_score,
                "newScore": result.new_score,
                "signal": result.signal.value,
                "action": result.action,
            }
        except TimeoutError:
            logger.error("adjust_trust_timeout", memory_id=memory_id, signal=signal)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error(
                "adjust_trust_error", error=str(e), memory_id=memory_id, signal=signal
            )
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: list_archived
    # =========================================================================
    async def list_archived(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List archived memories (trust below archive threshold).

        Args:
            limit: Maximum number of results (default 50).
            offset: Number of results to skip (default 0).

        Returns:
            Dictionary with count, memories list.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._trust_engine is None:
                self._trust_engine = TrustEngine(memory_system=self._memory_system)

            results = await asyncio.wait_for(
                self._trust_engine.list_archived(limit, offset),
                timeout=OPERATION_TIMEOUT,
            )

            memories = []
            for entry in results:
                entry_dict = entry.model_dump(by_alias=True)
                if entry_dict.get("timestamp"):
                    entry_dict["timestamp"] = entry.timestamp.isoformat()
                memories.append(entry_dict)

            return {
                "count": len(memories),
                "memories": memories,
            }
        except TimeoutError:
            logger.error("list_archived_timeout", limit=limit, offset=offset)
            return {"count": 0, "memories": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("list_archived_error", error=str(e))
            return {"count": 0, "memories": [], "error": str(e)}

    # =========================================================================
    # TOOL: find_related_memories
    # =========================================================================
    async def find_related_memories(
        self,
        memoryId: str,  # noqa: N803
        relationshipType: str | None = None,  # noqa: N803
        limit: int = 10,
        includeArchived: bool = True,  # noqa: N803
        maxChainDepth: int = 10,  # noqa: N803
    ) -> dict[str, Any]:
        """Find memories related to a given memory.

        For SUPERSEDES and PARTIAL_UPDATE relationships, will traverse the
        supersession chain including archived memories to find the full history.

        Args:
            memoryId: ID of the reference memory.
            relationshipType: Optional filter by relationship type ("SUPERSEDES", "PARTIAL_UPDATE", "RELATED_TO", "CONTRADICTS", "DERIVED_FROM").
            limit: Maximum number of results (default 10).
            includeArchived: Include archived memories for SUPERSEDES chains (default True).
            maxChainDepth: Maximum depth for supersession chain traversal (default 10).

        Returns:
            Dictionary with count, memories list, and relationshipType used.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Parse relationship type if provided
            rel_type = None
            if relationshipType:
                try:
                    rel_type = RelationshipType(relationshipType.upper())
                except ValueError:
                    return {
                        "count": 0,
                        "memories": [],
                        "relationshipType": relationshipType,
                        "error": f"Invalid relationship type: {relationshipType}",
                    }

            results = await asyncio.wait_for(
                self._memory_system.find_related_memories(
                    memoryId, rel_type, limit, includeArchived, maxChainDepth
                ),
                timeout=OPERATION_TIMEOUT,
            )

            memories = []
            for entry in results[:limit]:
                entry_dict = entry.model_dump(by_alias=True)
                if entry_dict.get("timestamp"):
                    entry_dict["timestamp"] = entry.timestamp.isoformat()
                memories.append(entry_dict)

            return {
                "count": len(memories),
                "memories": memories,
                "relationshipType": relationshipType or "all",
            }
        except TimeoutError:
            logger.error("find_related_memories_timeout", memoryId=memoryId)
            return {"count": 0, "memories": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("find_related_memories_error", error=str(e), memoryId=memoryId)
            return {"count": 0, "memories": [], "error": str(e)}

    # =========================================================================
    # TOOL: create_relationship
    # =========================================================================
    async def create_relationship(
        self,
        sourceId: str,  # noqa: N803
        targetId: str,  # noqa: N803
        relationshipType: str,  # noqa: N803
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Create a relationship between two memories.

        Args:
            sourceId: ID of the source memory.
            targetId: ID of the target memory.
            relationshipType: Type of relationship - "SUPERSEDES", "PARTIAL_UPDATE", "RELATED_TO", "CONTRADICTS", "DERIVED_FROM".
            confidence: Relationship confidence 0.0-1.0 (default 1.0).

        Returns:
            Dictionary with success status, sourceId, targetId, relationshipType, and confidence.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Parse relationship type
            try:
                rel_type = RelationshipType(relationshipType.upper())
            except ValueError:
                return {
                    "success": False,
                    "sourceId": sourceId,
                    "targetId": targetId,
                    "relationshipType": None,
                    "confidence": None,
                    "error": f"Invalid relationship type: {relationshipType}",
                }

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            await asyncio.wait_for(
                self._memory_system.create_relationship(
                    sourceId, targetId, rel_type, confidence
                ),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "sourceId": sourceId,
                "targetId": targetId,
                "relationshipType": rel_type.value,
                "confidence": confidence,
            }
        except TimeoutError:
            logger.error(
                "create_relationship_timeout", sourceId=sourceId, targetId=targetId
            )
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error(
                "create_relationship_error",
                error=str(e),
                sourceId=sourceId,
                targetId=targetId,
            )
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_relationship_summary
    # =========================================================================
    async def get_relationship_summary(self, memoryId: str) -> dict[str, Any]:  # noqa: N803
        """Get summary of all relationships for a memory.

        Args:
            memoryId: ID of the memory entry.

        Returns:
            Dictionary with memoryId, totalRelationships, and byType counts.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            result = await asyncio.wait_for(
                self._memory_system.get_relationship_summary(memoryId),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_relationship_summary_timeout", memoryId=memoryId)
            return {"error": "Operation timed out"}
        except Exception as e:
            logger.error(
                "get_relationship_summary_error", error=str(e), memoryId=memoryId
            )
            return {"error": str(e)}

    # =========================================================================
    # TOOL: trigger_extraction
    # =========================================================================
    async def trigger_extraction(
        self,
        conversation: str | None = None,
    ) -> dict[str, Any]:
        """Manually trigger memory extraction.

        Args:
            conversation: Optional conversation text (uses buffer if not provided).

        Returns:
            Dictionary with count of extracted memories and their IDs.
        """
        try:
            if self._extractor is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._extractor = MemoryExtractor(memory_system=self._memory_system)

            memory_ids = await asyncio.wait_for(
                self._extractor.trigger_extraction(conversation),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "count": len(memory_ids),
                "memory_ids": memory_ids,
            }
        except TimeoutError:
            logger.error("trigger_extraction_timeout")
            return {
                "success": False,
                "count": 0,
                "memory_ids": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("trigger_extraction_error", error=str(e))
            return {"success": False, "count": 0, "memory_ids": [], "error": str(e)}

    # =========================================================================
    # TOOL: preconpress_extraction
    # =========================================================================
    async def preconpress_extraction(
        self,
        context_content: str | None = None,
    ) -> dict[str, Any]:
        """Capture context and extract memories before compaction.

        Args:
            context_content: Current context content (captures if not provided).

        Returns:
            Dictionary with count of extracted memories and their IDs.
        """
        try:
            if self._precompress is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._extractor = MemoryExtractor(memory_system=self._memory_system)
                self._precompress = PrecompressExtractor(
                    memory_system=self._memory_system,
                    extractor=self._extractor,
                )

            result = await asyncio.wait_for(
                self._precompress.capture_and_extract(context_content or ""),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "count": result.extraction_count,
                "memory_ids": result.memories_created,
                "context_captured": result.context_captured,
            }
        except TimeoutError:
            logger.error("preconpress_extraction_timeout")
            return {
                "success": False,
                "count": 0,
                "memory_ids": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("preconpress_extraction_error", error=str(e))
            return {"success": False, "count": 0, "memory_ids": [], "error": str(e)}

    # =========================================================================
    # TOOL: get_tier0_summary
    # =========================================================================
    async def get_tier0_summary(self, force_refresh: bool = False) -> dict[str, Any]:
        """Get L0 project summary (~100 tokens).

        L0 uses high-trust memories (trust >= 0.5) to generate a concise
        project summary suitable for session start auto-injection.

        Args:
            force_refresh: Force regeneration even if cache is valid (default False).

        Returns:
            Dictionary with tier, content, token_count, cache_hit, source_count,
            generated_at, and error (if any).
        """
        try:
            if self._tiered_loader is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._tiered_loader = TieredLoader(memory_system=self._memory_system)

            result = await asyncio.wait_for(
                self._tiered_loader.get_tier0(force_refresh),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_tier0_summary_timeout")
            return {
                "tier": "L0",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("get_tier0_summary_error", error=str(e))
            return {
                "tier": "L0",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_tier1_summary
    # =========================================================================
    async def get_tier1_summary(self, force_refresh: bool = False) -> dict[str, Any]:
        """Get L1 key decisions summary (~2K tokens).

        L1 uses promoted memories (trust >= 0.8) to generate a structured
        summary of key decisions and patterns for planning tasks.

        Args:
            force_refresh: Force regeneration even if cache is valid (default False).

        Returns:
            Dictionary with tier, content, token_count, cache_hit, source_count,
            generated_at, and error (if any).
        """
        try:
            if self._tiered_loader is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._tiered_loader = TieredLoader(memory_system=self._memory_system)

            result = await asyncio.wait_for(
                self._tiered_loader.get_tier1(force_refresh),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_tier1_summary_timeout")
            return {
                "tier": "L1",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("get_tier1_summary_error", error=str(e))
            return {
                "tier": "L1",
                "content": None,
                "token_count": 0,
                "cache_hit": False,
                "source_count": 0,
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_user_profile
    # =========================================================================
    async def get_user_profile(
        self,
        include_dialectic_notes: bool = False,
    ) -> dict[str, Any]:
        """Get user profile with preferences and style.

        Args:
            include_dialectic_notes: Include LLM reasoning traces (default False).

        Returns:
            Dictionary with user_id, communication_style, expertise_domains,
            preferences, confidence, last_updated, session_count, warmed_up,
            and optionally dialectic_notes.
        """
        try:
            if self._user_model is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._user_model = UserModel(memory_system=self._memory_system)

            result = await asyncio.wait_for(
                self._user_model.get_profile(include_dialectic_notes),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_user_profile_timeout")
            return {
                "error": "Operation timed out",
                "warmed_up": False,
                "session_count": 0,
            }
        except Exception as e:
            logger.error("get_user_profile_error", error=str(e))
            return {
                "error": str(e),
                "warmed_up": False,
                "session_count": 0,
            }

    # =========================================================================
    # TOOL: update_user_profile
    # =========================================================================
    async def update_user_profile(
        self,
        conversation: str | None = None,
    ) -> dict[str, Any]:
        """Update user profile dialectically after a session.

        Args:
            conversation: Optional conversation text to analyze.
                         If not provided, returns error.

        Returns:
            Dictionary with success status, session_count, reasoning,
            and warmed_up status.
        """
        try:
            if self._user_model is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._user_model = UserModel(memory_system=self._memory_system)

            if not conversation:
                return {
                    "success": False,
                    "error": "No conversation provided",
                    "session_count": 0,
                }

            result = await asyncio.wait_for(
                self._user_model.update_profile_from_session(conversation),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("update_user_profile_timeout")
            return {
                "success": False,
                "error": "Operation timed out",
                "session_count": 0,
            }
        except Exception as e:
            logger.error("update_user_profile_error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "session_count": 0,
            }

    # =========================================================================
    # TOOL: get_decay_status
    # =========================================================================
    async def get_decay_status(self) -> dict[str, Any]:
        """Get decay engine status and statistics.

        Returns:
            Dictionary with enabled status, decay stats, and list of fading memories.
        """
        try:
            if self._decay_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._decay_engine = DecayEngine(memory_system=self._memory_system)

            result = await asyncio.wait_for(
                self._decay_engine.get_decay_status(),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_decay_status_timeout")
            return {"enabled": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("get_decay_status_error", error=str(e))
            return {"enabled": False, "error": str(e)}

    # =========================================================================
    # TOOL: trigger_consolidation
    # =========================================================================
    async def trigger_consolidation(self, force: bool = False) -> dict[str, Any]:
        """Trigger memory consolidation manually.

        Consolidation finds similar memory pairs and merges them. When enabled,
        it runs automatically on a schedule. This tool allows manual triggering.

        Args:
            force: Run consolidation even if not scheduled (default False).

        Returns:
            Dictionary with pairs_found, pairs_merged, memories_consolidated, and error.
        """
        try:
            if self._consolidation_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._consolidation_engine = ConsolidationEngine(
                    memory_system=self._memory_system
                )

            if not self._consolidation_engine.is_enabled and not force:
                return {
                    "success": False,
                    "error": "Decay engine is not enabled. Set DECAY_ENABLED=true to use consolidation.",
                    "pairs_found": 0,
                    "pairs_merged": 0,
                }

            result = await asyncio.wait_for(
                self._consolidation_engine.run_consolidation(),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "pairs_found": result.get("pairs_found", 0),
                "pairs_merged": result.get("pairs_merged", 0),
                "memories_consolidated": result.get("memories_consolidated", 0),
            }
        except TimeoutError:
            logger.error("trigger_consolidation_timeout")
            return {"success": False, "error": "Operation timed out", "pairs_merged": 0}
        except Exception as e:
            logger.error("trigger_consolidation_error", error=str(e))
            return {"success": False, "error": str(e), "pairs_merged": 0}

    # =========================================================================
    # TOOL: list_fading_memories
    # =========================================================================
    async def list_fading_memories(self, limit: int = 20) -> dict[str, Any]:
        """List memories approaching archive threshold.

        Fading memories are those with low trust scores or that will reach
        archive threshold within 30 days based on their decay rate.

        Args:
            limit: Maximum number of results (default 20).

        Returns:
            Dictionary with fading_count and list of fading memories.
        """
        try:
            if self._consolidation_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._consolidation_engine = ConsolidationEngine(
                    memory_system=self._memory_system
                )

            fading = await asyncio.wait_for(
                self._consolidation_engine.list_fading_memories(limit),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "fading_count": len(fading),
                "fading_memories": fading,
            }
        except TimeoutError:
            logger.error("list_fading_memories_timeout")
            return {
                "fading_count": 0,
                "fading_memories": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("list_fading_memories_error", error=str(e))
            return {"fading_count": 0, "fading_memories": [], "error": str(e)}

    # =========================================================================
    # TOOL: adjust_decay_rate
    # =========================================================================
    async def adjust_decay_rate(
        self,
        memory_id: str,
        decay_rate: float,
    ) -> dict[str, Any]:
        """Adjust decay rate for a specific memory.

        Higher decay rates cause faster trust decay. Lower rates preserve
        memories longer. Useful for marking important memories as "sticky".

        Args:
            memory_id: ID of the memory to adjust.
            decay_rate: New decay rate (0.1 to 10.0, default 1.0 = normal).

        Returns:
            Dictionary with success status, memory_id, new decay_rate, and message.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Clamp decay rate to valid range
            decay_rate = max(0.1, min(10.0, decay_rate))

            result = await asyncio.wait_for(
                adjust_decay_rate(self._memory_system, memory_id, decay_rate),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("adjust_decay_rate_timeout", memory_id=memory_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("adjust_decay_rate_error", memory_id=memory_id, error=str(e))
            return {"success": False, "error": str(e)}

    # =========================================================================
    # v0.7.0: Dual-Model RRF — TOOL: elevate_memory_to_1024
    # =========================================================================
    async def elevate_memory_to_1024(
        self,
        memory_id: str,
        vector_1024: list[float] | None = None,
        trust_boost: float = 0.10,
    ) -> dict[str, Any]:
        """Promote a memory from 384-dim-only to also exist in 1024-dim space.

        Auto-mode only. Calling this tool in ``cpu`` or ``gpu`` mode raises
        a helpful error — elevation is meaningful only in the dual-model
        auto pipeline.

        Args:
            memory_id: UUID of the 384-dim memory to elevate.
            vector_1024: Optional pre-computed 1024-dim embedding. If None,
                the underlying DB helper derives one from the 384-dim
                vector via zero-pad + L2-normalize (placeholder expansion).
            trust_boost: Amount to add to the trust score on elevate
                (default 0.10, clamped to [0, 1]).

        Returns:
            Dictionary with keys:
                - memory_id (str)
                - elevated (bool) — True if newly inserted, False if already elevated
                - trust_score (float) — new boosted trust score
                - vector_dim (int) — always 1024 in v0.7.0
                - mode (str) — the embedding_mode that was active when called
        """
        try:
            # Auto-mode gate (per HANDOFF). The dispatch is intentionally
            # in ``add_memory`` and ``query_memories``; this tool is the
            # one explicit user-initiated path that mutates the 1024
            # sidecar, so it must opt-in to the same model.
            config = get_config()
            if config.embedding_mode != "auto":
                return {
                    "success": False,
                    "error": (
                        f"elevate_memory_to_1024 requires embedding_mode='auto' "
                        f"(current: {config.embedding_mode!r}). Set "
                        f"EMBEDDING_MODE=auto in the environment to enable."
                    ),
                    "current_mode": config.embedding_mode,
                }
            if not config.elevate_enabled:
                return {
                    "success": False,
                    "error": (
                        "elevate_memory_to_1024 is disabled "
                        "(ELEVATE_ENABLED=false). Set ELEVATE_ENABLED=true "
                        "to enable."
                    ),
                    "current_mode": config.embedding_mode,
                }

            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            # Clamp trust_boost to a safe range — same range as the DB helper.
            trust_boost = max(0.0, min(1.0, trust_boost))

            db = self._memory_system._db
            if not hasattr(db, "elevate_memory_to_1024"):
                return {
                    "success": False,
                    "error": "Underlying database does not support 1024-dim elevation",
                    "current_mode": config.embedding_mode,
                }

            result: dict[str, Any] = await asyncio.wait_for(
                db.elevate_memory_to_1024(
                    memory_id,
                    vector_1024=vector_1024,
                    trust_boost=trust_boost,
                ),
                timeout=OPERATION_TIMEOUT,
            )
            result["mode"] = config.embedding_mode
            result["success"] = True
            return result
        except TimeoutError:
            logger.error("elevate_memory_to_1024_timeout", memory_id=memory_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error(
                "elevate_memory_to_1024_error", memory_id=memory_id, error=str(e)
            )
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: kanban_add_card / kanban_move_card / kanban_list_cards / kanban_get_card
    # (GitHub triage poller integration)
    # =========================================================================
    #
    # Kanban cards are plain Postgres rows (NO pgvector). The wrapped
    # issue/PR text is separately embedded as a memory (source_type='github')
    # via add_memory; the optional memory_id FK links the card to that
    # embedded memory. These tools let the triage poller and agents
    # create, move, list, and inspect cards.

    async def kanban_add_card(
        self,
        card_id: str,
        repo: str,
        number: int,
        item_type: str,
        url: str,
        title: str,
        author: str | None = None,
        wrapped_text: str | None = None,
        draft: bool = False,
        memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a kanban card for a GitHub issue/PR (idempotent on re-poll).

        Uses ON CONFLICT (repo, number, item_type) DO NOTHING so re-polling
        the same issue/PR is a no-op. Returns the created card (or the
        existing card on conflict).

        Args:
            card_id: Stable card identifier (e.g. 'T-GH-001').
            repo: GitHub repo name (e.g. 'memini-ai-dev').
            number: GitHub issue/PR number.
            item_type: One of 'bug', 'feature', 'question', 'docs', 'pr', 'triage'.
            url: Full GitHub URL of the issue/PR.
            title: Issue/PR title.
            author: GitHub login of the author (optional).
            wrapped_text: Prompt-wrapped card body (optional).
            draft: True for PR drafts (default False).
            memory_id: UUID of the linked embedded memory (optional).

        Returns:
            Dictionary with the card fields and an ``inserted`` boolean
            (True if newly inserted, False if already existed).
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            db = self._memory_system._db
            if not hasattr(db, "add_kanban_card"):
                return {
                    "success": False,
                    "error": "Underlying database does not support kanban cards",
                }

            result: dict[str, Any] = await asyncio.wait_for(
                db.add_kanban_card(
                    card_id=card_id,
                    repo=repo,
                    number=number,
                    item_type=item_type,
                    url=url,
                    title=title,
                    author=author,
                    wrapped_text=wrapped_text,
                    draft=draft,
                    memory_id=memory_id,
                ),
                timeout=OPERATION_TIMEOUT,
            )
            result["success"] = True
            return result
        except TimeoutError:
            logger.error("kanban_add_card_timeout", card_id=card_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("kanban_add_card_error", card_id=card_id, error=str(e))
            return {"success": False, "error": str(e)}

    async def kanban_move_card(
        self,
        card_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Move a kanban card to a new status.

        Args:
            card_id: Card identifier (e.g. 'T-GH-001').
            status: New status — one of 'triage', 'todo', 'ready',
                'running', 'blocked', 'done', 'archived'.

        Returns:
            Dictionary with the updated card fields, or an error if the
            card was not found or the status was invalid.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            db = self._memory_system._db
            if not hasattr(db, "move_kanban_card"):
                return {
                    "success": False,
                    "error": "Underlying database does not support kanban cards",
                }

            result: dict[str, Any] | None = await asyncio.wait_for(
                db.move_kanban_card(card_id, status),
                timeout=OPERATION_TIMEOUT,
            )
            if result is None:
                return {
                    "success": False,
                    "error": f"Card {card_id!r} not found",
                }
            result["success"] = True
            return result
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except TimeoutError:
            logger.error("kanban_move_card_timeout", card_id=card_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("kanban_move_card_error", card_id=card_id, error=str(e))
            return {"success": False, "error": str(e)}

    async def kanban_list_cards(
        self,
        status: str | None = None,
        repo: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List kanban cards with optional status/repo filters.

        Args:
            status: Optional status filter (triage|todo|ready|running|blocked|done|archived).
            repo: Optional repo filter (e.g. 'memini-ai-dev').
            limit: Max results (default 100).

        Returns:
            Dictionary with a ``cards`` list of card dicts and ``count``.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            db = self._memory_system._db
            if not hasattr(db, "list_kanban_cards"):
                return {
                    "success": False,
                    "error": "Underlying database does not support kanban cards",
                }

            cards: list[dict[str, Any]] = await asyncio.wait_for(
                db.list_kanban_cards(status=status, repo=repo, limit=limit),
                timeout=OPERATION_TIMEOUT,
            )
            return {"success": True, "cards": cards, "count": len(cards)}
        except TimeoutError:
            logger.error("kanban_list_cards_timeout")
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("kanban_list_cards_error", error=str(e))
            return {"success": False, "error": str(e)}

    async def kanban_get_card(
        self,
        card_id: str,
    ) -> dict[str, Any]:
        """Get a single kanban card by its card_id.

        Args:
            card_id: Card identifier (e.g. 'T-GH-001').

        Returns:
            Dictionary with the card fields, or a not-found error.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            db = self._memory_system._db
            if not hasattr(db, "get_kanban_card"):
                return {
                    "success": False,
                    "error": "Underlying database does not support kanban cards",
                }

            result: dict[str, Any] | None = await asyncio.wait_for(
                db.get_kanban_card(card_id),
                timeout=OPERATION_TIMEOUT,
            )
            if result is None:
                return {
                    "success": False,
                    "error": f"Card {card_id!r} not found",
                }
            result["success"] = True
            return result
        except TimeoutError:
            logger.error("kanban_get_card_timeout", card_id=card_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("kanban_get_card_error", card_id=card_id, error=str(e))
            return {"success": False, "error": str(e)}

    # =========================================================================
    # GRACEFUL SHUTDOWN
    # =========================================================================
    async def _shutdown(self) -> None:
        """Graceful shutdown handler."""
        if self._shutdown_in_progress:
            logger.warning("shutdown_already_in_progress")
            return
        self._shutdown_in_progress = True

        logger.info("server_shutdown_started")

        # Cancel all background jobs
        for job_id, task in list(self._background_jobs.items()):
            if not task.done():
                task.cancel()
                logger.info("background_job_cancelled", job_id=job_id)

        # Cancel stdio watch task if running
        if self._stdio_watch_task is not None and not self._stdio_watch_task.done():
            self._stdio_watch_task.cancel()
            logger.info("stdio_watch_task_cancelled")

        # Stop indexer
        if self._indexer is not None and self._indexer.is_running:
            await self._indexer.stop()
            logger.info("indexer_stopped")

        # Close memory system (closes DB pool)
        if self._memory_system is not None and self._memory_system.is_initialized:
            await self._memory_system.close()
            logger.info("memory_system_closed")

        # Close thought chains DB pool if active
        # ThoughtChains shares the same pool as the memory system,
        # so closing memory_system above already closes it.
        # Just clear the reference.
        if self._thought_chains is not None:
            self._thought_chains = None
            logger.info("thought_chains_cleared")

        # Stop audit logger
        if self._audit_logger is not None:
            await self._audit_logger.stop()
            self._audit_logger = None
            logger.info("audit_logger_stopped")

        logger.info("server_shutdown_complete")

    # =========================================================================
    # TOOL: query_kg (Phase 4B)
    # =========================================================================
    async def query_kg(self, query: str) -> dict[str, Any]:
        """Execute a formal knowledge graph query.

        Args:
            query: JSON string with KGQuery fields:
                - entity_a: Optional entity ID or name to start from
                - entity_b: Optional entity ID or name to target
                - relationship_types: Optional list of relationship types
                - inference_depth: Transitive closure depth (default 1)
                - limit: Maximum results (default 100)

        Returns:
            Dictionary with success status, count, results, and query.
        """
        try:
            if self._knowledge_graph is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            # Parse query string to dict if provided
            if isinstance(query, str):
                import json as json_module

                query_dict = json_module.loads(query)
            else:
                query_dict = query

            # Build KGQuery object
            rel_types = None
            if query_dict.get("relationship_types"):
                rel_types = [
                    RelationshipType(rt.upper())
                    for rt in query_dict["relationship_types"]
                ]

            kg_query = KGQuery(
                entity_a=query_dict.get("entity_a"),
                entity_b=query_dict.get("entity_b"),
                relationship_types=rel_types,
                inference_depth=query_dict.get("inference_depth", 1),
                limit=query_dict.get("limit", 100),
            )

            result = await asyncio.wait_for(
                self._knowledge_graph.query_kg(kg_query),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("query_kg_timeout", query=query)
            return {"success": False, "results": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("query_kg_error", error=str(e), query=query)
            return {"success": False, "results": [], "error": str(e)}

    # =========================================================================
    # TOOL: extract_entities (Phase 4B)
    # =========================================================================
    async def extract_entities(self, memory_id: str) -> dict[str, Any]:
        """Extract entities from a specific memory.

        Args:
            memory_id: ID of the memory entry to extract entities from.

        Returns:
            Dictionary with success status, memory_id, and list of extracted entities.
        """
        try:
            if self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._knowledge_graph is None:
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            # Get the memory
            memory = await asyncio.wait_for(
                self._memory_system.get_memory(memory_id),
                timeout=OPERATION_TIMEOUT,
            )

            if memory is None:
                return {
                    "success": False,
                    "memory_id": memory_id,
                    "entities": [],
                    "error": "Memory not found",
                }

            # Extract and register entities
            entities = await asyncio.wait_for(
                self._knowledge_graph.extract_and_register_entities(memory.text),
                timeout=OPERATION_TIMEOUT,
            )

            # Link memory to entities
            await asyncio.wait_for(
                self._knowledge_graph.link_memory_to_entities(memory_id, memory.text),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "memory_id": memory_id,
                "entities": [e.to_dict() for e in entities],
                "count": len(entities),
            }
        except TimeoutError:
            logger.error("extract_entities_timeout", memory_id=memory_id)
            return {
                "success": False,
                "memory_id": memory_id,
                "entities": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("extract_entities_error", error=str(e), memory_id=memory_id)
            return {
                "success": False,
                "memory_id": memory_id,
                "entities": [],
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_entity_graph (Phase 4B)
    # =========================================================================
    async def get_entity_graph(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Get all connections to/from an entity.

        Args:
            entity_id: Entity ID to query.
            depth: Include transitive connections up to this depth (default 1).

        Returns:
            Dictionary with entity info, incoming, outgoing, and inferred connections.
        """
        try:
            if self._knowledge_graph is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._knowledge_graph.get_entity_graph(entity_id, depth),
                timeout=OPERATION_TIMEOUT,
            )

            if result is None:
                return {
                    "success": False,
                    "entity_id": entity_id,
                    "error": "Entity not found",
                }

            return {
                "success": True,
                **result.to_dict(),
            }
        except TimeoutError:
            logger.error("get_entity_graph_timeout", entity_id=entity_id)
            return {
                "success": False,
                "entity_id": entity_id,
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("get_entity_graph_error", error=str(e), entity_id=entity_id)
            return {
                "success": False,
                "entity_id": entity_id,
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_inference_chain (Phase 4B)
    # =========================================================================
    async def get_inference_chain(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Find inference paths between two entities.

        Args:
            start_entity: Starting entity ID or name.
            end_entity: Target entity ID or name.
            max_depth: Maximum path depth (default 3).

        Returns:
            Dictionary with success status, start/end entities, paths, and metadata.
        """
        try:
            if self._knowledge_graph is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            # Resolve entity names to IDs
            start_id = await self._knowledge_graph._resolve_entity(start_entity)
            end_id = await self._knowledge_graph._resolve_entity(end_entity)

            if not start_id:
                return {
                    "success": False,
                    "start_entity": start_entity,
                    "end_entity": end_entity,
                    "paths": [],
                    "error": f"Start entity not found: {start_entity}",
                }

            if not end_id:
                return {
                    "success": False,
                    "start_entity": start_entity,
                    "end_entity": end_entity,
                    "paths": [],
                    "error": f"End entity not found: {end_entity}",
                }

            # Find all paths
            chains = await asyncio.wait_for(
                self._knowledge_graph.get_inference_chains(
                    start_id, end_id, max_depth=max_depth
                ),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": [c.to_dict() for c in chains],
                "count": len(chains),
            }
        except TimeoutError:
            logger.error(
                "get_inference_chain_timeout",
                start=start_entity,
                end=end_entity,
            )
            return {
                "success": False,
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error(
                "get_inference_chain_error",
                error=str(e),
                start=start_entity,
                end=end_entity,
            )
            return {
                "success": False,
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": [],
                "error": str(e),
            }

    # =========================================================================
    # TOOL: search_entities (Phase 4B)
    # =========================================================================
    async def search_entities(
        self,
        name: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search for entities by name.

        Args:
            name: Entity name to search for.
            limit: Maximum results (default 10).

        Returns:
            Dictionary with success status, count, and list of matching entities.
        """
        try:
            if self._knowledge_graph is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            entities = await asyncio.wait_for(
                self._knowledge_graph.search_entities(name, limit=limit),
                timeout=OPERATION_TIMEOUT,
            )

            return {
                "success": True,
                "query": name,
                "entities": [e.to_dict() for e in entities],
                "count": len(entities),
            }
        except TimeoutError:
            logger.error("search_entities_timeout", name=name)
            return {
                "success": False,
                "query": name,
                "entities": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("search_entities_error", error=str(e), name=name)
            return {
                "success": False,
                "query": name,
                "entities": [],
                "error": str(e),
            }

    # =========================================================================
    # TOOL: get_graph_visualization (Phase 4B)
    # =========================================================================
    async def get_graph_visualization(self, limit: int = 100) -> str:
        """Get an HTML visualization of the knowledge graph.

        Returns a self-contained HTML page with a D3.js force-directed graph
        showing entities as nodes and relationships as edges.

        Args:
            limit: Maximum number of nodes to visualize (default 100).

        Returns:
            Complete HTML string with embedded D3.js visualization,
            or error message if KG is disabled or has no data.
        """
        try:
            if self._knowledge_graph is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._knowledge_graph = KnowledgeGraph(
                    memory_system=self._memory_system
                )

            await self._knowledge_graph.initialize()

            # Export graph data and generate HTML
            graph_data = self._knowledge_graph.to_d3_json(limit=limit)
            html = generate_visualization_html(graph_data)

            return html
        except TimeoutError:
            logger.error("get_graph_visualization_timeout")
            return "<html><body><p>Operation timed out</p></body></html>"
        except Exception as e:
            logger.error("get_graph_visualization_error", error=str(e))
            return f"<html><body><p>Error: {str(e)}</p></body></html>"

    # =========================================================================
    # TOOL: list_peers (Phase 4C)
    # =========================================================================
    async def list_peers(self) -> dict[str, Any]:
        """List all known peers.

        Returns:
            Dictionary with count and list of peer profiles.
        """
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.list_peers(),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("list_peers_timeout")
            return {"count": 0, "peers": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("list_peers_error", error=str(e))
            return {"count": 0, "peers": [], "error": str(e)}

    # =========================================================================
    # TOOL: add_peer (Phase 4C)
    # =========================================================================
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
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.add_peer(
                    peer_id=peer_id,
                    name=name,
                    role=role,
                    trust_level=trust_level,
                    preferences=preferences,
                ),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("add_peer_timeout", peer_id=peer_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("add_peer_error", error=str(e), peer_id=peer_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: switch_peer_context (Phase 4C)
    # =========================================================================
    async def switch_peer_context(self, peer_id: str | None = None) -> dict[str, Any]:
        """Switch the active peer context.

        When peer_id is provided, sets it as the current context. When None,
        switches back to the default (owner) context.

        Args:
            peer_id: Peer ID to switch to, or None for default context.

        Returns:
            Dictionary with success status and context info.
        """
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.switch_peer_context(peer_id),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("switch_peer_context_timeout", peer_id=peer_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("switch_peer_context_error", error=str(e), peer_id=peer_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: share_memory (Phase 4C)
    # =========================================================================
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
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.share_memory(
                    memory_id=memory_id,
                    target_peer_id=target_peer_id,
                    permission=permission,
                ),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("share_memory_timeout", memory_id=memory_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("share_memory_error", error=str(e), memory_id=memory_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_peer_memories (Phase 4C)
    # =========================================================================
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
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.get_peer_memories(
                    peer_id=peer_id,
                    query=query,
                    limit=limit,
                ),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_peer_memories_timeout", peer_id=peer_id)
            return {"count": 0, "memories": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("get_peer_memories_error", error=str(e), peer_id=peer_id)
            return {"count": 0, "memories": [], "error": str(e)}

    # =========================================================================
    # TOOL: get_shared_memories (Phase 4C)
    # =========================================================================
    async def get_shared_memories(self, limit: int = 20) -> dict[str, Any]:
        """Get all memories shared with the current peer context.

        Args:
            limit: Maximum number of results (default 20).

        Returns:
            Dictionary with count and list of shared memories.
        """
        try:
            if self._multi_peer_manager is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._multi_peer_manager = get_multi_peer_manager(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._multi_peer_manager.get_shared_memories(limit=limit),
                timeout=OPERATION_TIMEOUT,
            )

            return result
        except TimeoutError:
            logger.error("get_shared_memories_timeout")
            return {"count": 0, "memories": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("get_shared_memories_error", error=str(e))
            return {"count": 0, "memories": [], "error": str(e)}

    # =========================================================================
    # TOOL: find_contradictions (Phase 4D)
    # =========================================================================
    async def find_contradictions(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find memory pairs that contradict each other.

        Args:
            query: Optional query string to filter contradictions.
            limit: Maximum number of results (default 10).

        Returns:
            Dictionary with count and list of contradiction pairs.
        """
        try:
            if self._dialectic_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._dialectic_engine = get_dialectic_engine(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._dialectic_engine.find_contradictions(query, limit),
                timeout=OPERATION_TIMEOUT,
            )

            return {"count": len(result), "contradictions": result}
        except TimeoutError:
            logger.error("find_contradictions_timeout", query=query)
            return {"count": 0, "contradictions": [], "error": "Operation timed out"}
        except Exception as e:
            logger.error("find_contradictions_error", error=str(e), query=query)
            return {"count": 0, "contradictions": [], "error": str(e)}

    # =========================================================================
    # TOOL: resolve_contradiction (Phase 4D)
    # =========================================================================
    async def resolve_contradiction(
        self,
        memory_id_a: str,
        memory_id_b: str,
    ) -> dict[str, Any]:
        """Generate dialectic resolution for two contradictory memories.

        Args:
            memory_id_a: ID of the first memory.
            memory_id_b: ID of the second memory.

        Returns:
            Dictionary with dialectic resolution and arguments.
        """
        try:
            if self._dialectic_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._dialectic_engine = get_dialectic_engine(
                    memory_system=self._memory_system
                )

            resolution = await asyncio.wait_for(
                self._dialectic_engine.resolve_contradiction(memory_id_a, memory_id_b),
                timeout=OPERATION_TIMEOUT,
            )

            if resolution is None:
                return {
                    "success": False,
                    "memory_a_id": memory_id_a,
                    "memory_b_id": memory_id_b,
                    "error": "Resolution failed - dialectic may be disabled",
                }

            return {
                "success": True,
                "memory_a_id": resolution.memory_a_id,
                "memory_b_id": resolution.memory_b_id,
                "pro_arguments": [
                    {
                        "memory_id": a.memory_id,
                        "side": a.side,
                        "argument": a.argument,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in resolution.pro_arguments
                ],
                "con_arguments": [
                    {
                        "memory_id": a.memory_id,
                        "side": a.side,
                        "argument": a.argument,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in resolution.con_arguments
                ],
                "resolution": resolution.resolution,
                "winner": resolution.winner,
                "reasoning": resolution.reasoning,
                "confidence": resolution.confidence,
                "timestamp": resolution.timestamp.isoformat(),
            }
        except TimeoutError:
            logger.error("resolve_contradiction_timeout", memory_id_a=memory_id_a)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error(
                "resolve_contradiction_error",
                error=str(e),
                memory_id_a=memory_id_a,
                memory_id_b=memory_id_b,
            )
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_dialectic_history (Phase 4D)
    # =========================================================================
    async def get_dialectic_history(
        self,
        memory_id: str,
    ) -> dict[str, Any]:
        """Get dialectic history for a memory.

        Args:
            memory_id: ID of the memory.

        Returns:
            Dictionary with dialectic notes, challenges, and resolutions.
        """
        try:
            if self._dialectic_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._dialectic_engine = get_dialectic_engine(
                    memory_system=self._memory_system
                )

            result = await asyncio.wait_for(
                self._dialectic_engine.get_dialectic_history(memory_id),
                timeout=OPERATION_TIMEOUT,
            )

            if result is None:
                return {
                    "memory_id": memory_id,
                    "notes": [],
                    "challenges": [],
                    "resolutions": [],
                    "error": "Dialectic engine disabled",
                }

            return result
        except TimeoutError:
            logger.error("get_dialectic_history_timeout", memory_id=memory_id)
            return {
                "memory_id": memory_id,
                "notes": [],
                "challenges": [],
                "resolutions": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error(
                "get_dialectic_history_error", error=str(e), memory_id=memory_id
            )
            return {
                "memory_id": memory_id,
                "notes": [],
                "challenges": [],
                "resolutions": [],
                "error": str(e),
            }

    # =========================================================================
    # TOOL: challenge_memory (Phase 4D)
    # =========================================================================
    async def challenge_memory(
        self,
        memory_id: str,
        challenge_text: str,
    ) -> dict[str, Any]:
        """Submit a counter-argument challenge to a memory.

        Args:
            memory_id: ID of the memory to challenge.
            challenge_text: The challenge or counter-argument text.

        Returns:
            Dictionary with challenge details and response.
        """
        try:
            if self._dialectic_engine is None:
                if self._memory_system is None:
                    self._memory_system = await asyncio.wait_for(
                        self._init_memory_system(), timeout=OPERATION_TIMEOUT
                    )
                self._dialectic_engine = get_dialectic_engine(
                    memory_system=self._memory_system
                )

            challenge = await asyncio.wait_for(
                self._dialectic_engine.challenge_memory(memory_id, challenge_text),
                timeout=OPERATION_TIMEOUT,
            )

            if challenge is None:
                return {
                    "success": False,
                    "memory_id": memory_id,
                    "error": "Challenge failed - dialectic may be disabled",
                }

            return {
                "success": True,
                "memory_id": challenge.memory_id,
                "challenge_text": challenge.challenge_text,
                "response": challenge.response,
                "confidence_delta": challenge.confidence_delta,
                "timestamp": challenge.timestamp.isoformat(),
            }
        except TimeoutError:
            logger.error("challenge_memory_timeout", memory_id=memory_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("challenge_memory_error", error=str(e), memory_id=memory_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Thought Chains Helper
    # =========================================================================

    async def _get_thought_chains(self) -> ThoughtChains | dict[str, Any]:
        """Get or initialize ThoughtChains instance.

        Returns ThoughtChains if available, or error dict if not.
        """
        config = get_config()
        if not config.thought_chains_enabled:
            return {
                "error": (
                    "Thought chains not enabled. "
                    "Set THOUGHT_CHAINS=true or THOUGHT_CHAINS=1 in environment."
                ),
            }

        if self._thought_chains is not None:
            return self._thought_chains

        # Lazy initialization: need pool from postgres backend
        if self._memory_system is None:
            self._memory_system = await asyncio.wait_for(
                self._init_memory_system(), timeout=OPERATION_TIMEOUT
            )

        db = self._memory_system._db
        if not hasattr(db, "_pool") or db._pool is None:
            return {
                "error": (
                    "Thought chains require PostgreSQL with pgvector. "
                    "Database pool not available."
                ),
            }

        self._thought_chains = ThoughtChains(
            pool=db._pool,
            memory_system=self._memory_system,
            trust_engine=self._trust_engine,
        )
        return self._thought_chains

    # =========================================================================
    # TOOL: add_thought (Phase 5 - Thought Chains)
    # =========================================================================
    async def add_thought(
        self,
        thought: str,
        thoughtNumber: int,  # noqa: N803
        totalThoughts: int,  # noqa: N803
        nextThoughtNeeded: bool,  # noqa: N803
        isRevision: bool = False,  # noqa: N803
        revisesThought: int | None = None,  # noqa: N803
        branchFromThought: int | None = None,  # noqa: N803
        branchId: str | None = None,  # noqa: N803
        chain_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a thought to a reasoning chain.

        API-compatible with @modelcontextprotocol/server-sequential-thinking.
        Auto-creates chain if chain_id not provided. Stores thought in BOTH
        thoughts table AND memories table (sourceType="thought").

        Args:
            thought: The thought text.
            thoughtNumber: Current thought number.
            totalThoughts: Total expected thoughts.
            nextThoughtNeeded: Whether more thoughts are needed.
            isRevision: Whether this is a revision.
            revisesThought: Thought number being revised.
            branchFromThought: Thought number to branch from.
            branchId: Branch identifier.
            chain_id: Chain UUID (auto-created if None).
            session_id: Session identifier.

        Returns:
            Dictionary with thoughtNumber, totalThoughts, nextThoughtNeeded,
            chain_id, branches, thoughtHistoryLength.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.add_thought(
                    thought=thought,
                    thought_number=thoughtNumber,
                    total_thoughts=totalThoughts,
                    next_thought_needed=nextThoughtNeeded,
                    is_revision=isRevision,
                    revises_thought=revisesThought,
                    branch_from_thought=branchFromThought,
                    branch_id=branchId,
                    chain_id=chain_id,
                    session_id=session_id,
                ),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("add_thought_timeout")
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("add_thought_error", error=str(e))
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: start_thought_chain (Phase 5 - Thought Chains)
    # =========================================================================
    async def start_thought_chain(
        self,
        session_id: str | None = None,
        parent_chain_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new thought chain.

        Args:
            session_id: Optional session identifier.
            parent_chain_id: Optional parent chain ID for hierarchical chains.

        Returns:
            Dictionary with chain_id, session_id, status, created_at.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.start_chain(
                    session_id=session_id,
                    parent_chain_id=parent_chain_id,
                ),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("start_thought_chain_timeout")
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("start_thought_chain_error", error=str(e))
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_thought_chain (Phase 5 - Thought Chains)
    # =========================================================================
    async def get_thought_chain(self, chain_id: str) -> dict[str, Any]:
        """Retrieve a full thought chain with all thoughts organized by branch.

        Args:
            chain_id: UUID of the chain to retrieve.

        Returns:
            Dictionary with chain_id, session_id, status, thoughts, branchMap,
            thought_count.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.get_chain(chain_id),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("get_thought_chain_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("get_thought_chain_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_related_chains (Phase 5 - Thought Chains)
    # =========================================================================
    async def get_related_chains(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search for thought chains with similar reasoning to the query.

        Uses pgvector cosine similarity on thought embeddings.

        Args:
            query: Search query text.
            limit: Maximum number of results (default 10).

        Returns:
            Dictionary with count and chains list.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.get_related_chains(query=query, limit=limit),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("get_related_chains_timeout", query=query)
            return {
                "success": False,
                "count": 0,
                "chains": [],
                "error": "Operation timed out",
            }
        except Exception as e:
            logger.error("get_related_chains_error", error=str(e), query=query)
            return {"success": False, "count": 0, "chains": [], "error": str(e)}

    # =========================================================================
    # TOOL: revise_thought (Phase 5 - Thought Chains)
    # =========================================================================
    async def revise_thought(
        self,
        chain_id: str,
        thought_number: int,
        revised_thought: str,
    ) -> dict[str, Any]:
        """Create a revision of an existing thought.

        Args:
            chain_id: UUID of the chain.
            thought_number: Number of the thought to revise.
            revised_thought: New thought text.

        Returns:
            Dictionary with success, thought_id, chain_id, thought_number.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.revise_thought(
                    chain_id=chain_id,
                    thought_number=thought_number,
                    revised_thought=revised_thought,
                ),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("revise_thought_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("revise_thought_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: branch_thought (Phase 5 - Thought Chains)
    # =========================================================================
    async def branch_thought(
        self,
        chain_id: str,
        from_thought_number: int,
        branchId: str,  # noqa: N803
        thought: str,
        thoughtNumber: int,  # noqa: N803
        totalThoughts: int,  # noqa: N803
        nextThoughtNeeded: bool,  # noqa: N803
    ) -> dict[str, Any]:
        """Start a new branch from an existing thought.

        Args:
            chain_id: UUID of the chain.
            from_thought_number: Thought number to branch from.
            branchId: Branch identifier.
            thought: New thought text.
            thoughtNumber: Thought number in new branch.
            totalThoughts: Total thoughts expected in branch.
            nextThoughtNeeded: Whether more thoughts follow.

        Returns:
            Dictionary with success, thought_id, chain_id, branch_id,
            thought_number.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.branch_thought(
                    chain_id=chain_id,
                    from_thought_number=from_thought_number,
                    branch_id=branchId,
                    thought=thought,
                    thought_number=thoughtNumber,
                    total_thoughts=totalThoughts,
                    next_thought_needed=nextThoughtNeeded,
                ),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("branch_thought_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("branch_thought_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: pause_thought_chain (Phase 5 - Thought Chains)
    # =========================================================================
    async def pause_thought_chain(self, chain_id: str) -> dict[str, Any]:
        """Pause a thought chain.

        Args:
            chain_id: UUID of the chain to pause.

        Returns:
            Dictionary with success, chain_id, previous_status, new_status.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.pause_chain(chain_id),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("pause_thought_chain_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("pause_thought_chain_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: resume_thought_chain (Phase 5 - Thought Chains)
    # =========================================================================
    async def resume_thought_chain(self, chain_id: str) -> dict[str, Any]:
        """Resume a paused thought chain.

        Returns the last thought so the agent can continue reasoning from
        where it left off.

        Args:
            chain_id: UUID of the chain to resume.

        Returns:
            Dictionary with success, chain_id, previous_status, new_status,
            thought_count, last_thought.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.resume_chain(chain_id),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("resume_thought_chain_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("resume_thought_chain_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: abandon_thought_chain (Phase 5 - Thought Chains)
    # =========================================================================
    async def abandon_thought_chain(self, chain_id: str) -> dict[str, Any]:
        """Abandon a thought chain. Applies agent_ignored trust signal.

        Args:
            chain_id: UUID of the chain to abandon.

        Returns:
            Dictionary with success, chain_id, previous_status, new_status.
        """
        try:
            tc_or_error = await self._get_thought_chains()
            if isinstance(tc_or_error, dict):
                return tc_or_error

            return await asyncio.wait_for(
                tc_or_error.abandon_chain(chain_id),
                timeout=OPERATION_TIMEOUT,
            )
        except TimeoutError:
            logger.error("abandon_thought_chain_timeout", chain_id=chain_id)
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error("abandon_thought_chain_error", error=str(e), chain_id=chain_id)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: log_audit_event (Phase 2.3 - Audit Logging)
    # =========================================================================
    async def log_audit_event(
        self,
        event_type: str,
        severity: str = "info",
        session_id: str | None = None,
        peer_id: str | None = None,
        agent_name: str | None = None,
        tool_name: str | None = None,
        memory_id: str | None = None,
        description: str | None = None,
        details: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> dict[str, Any]:
        """Log a single audit event manually.

        Args:
            event_type: Type of event - one of: auth_failure, permission_change,
                config_modification, agent_execution, memory_mutation,
                tool_invocation, trust_adjustment.
            severity: Severity level - "info", "warning", or "critical" (default "info").
            session_id: Optional session ID.
            peer_id: Optional peer ID.
            agent_name: Optional agent name.
            tool_name: Optional tool name.
            memory_id: Optional memory ID.
            description: Optional description text.
            details: Optional details dictionary.
            state_before: Optional state before the event.
            state_after: Optional state after the event.
            ip_address: Optional IP address.

        Returns:
            Dictionary with success status and message.
        """
        try:
            if self._audit_logger is None and self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._audit_logger is None:
                return {
                    "success": False,
                    "error": "Audit logger not available (database pool required)",
                }

            self._audit_logger.log(
                event_type=event_type,
                severity=severity,
                session_id=session_id,
                peer_id=peer_id,
                agent_name=agent_name,
                tool_name=tool_name,
                memory_id=memory_id,
                description=description,
                details=details,
                state_before=state_before,
                state_after=state_after,
                ip_address=ip_address,
            )

            return {
                "success": True,
                "message": f"Audit event '{event_type}' logged successfully",
            }
        except Exception as e:
            logger.error("log_audit_event_error", error=str(e), event_type=event_type)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # TOOL: get_audit_log (Phase 2.3 - Audit Logging)
    # =========================================================================
    async def get_audit_log(
        self,
        event_type: str | None = None,
        severity: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query audit log with filters.

        Args:
            event_type: Filter by event type.
            severity: Filter by severity level.
            agent_name: Filter by agent name.
            session_id: Filter by session ID.
            start_time: Filter events after this ISO timestamp.
            end_time: Filter events before this ISO timestamp.
            limit: Maximum number of results (default 100).

        Returns:
            Dictionary with count and list of matching audit events.
        """
        try:
            if self._audit_logger is None and self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._audit_logger is None:
                return {
                    "count": 0,
                    "events": [],
                    "error": "Audit logger not available (database pool required)",
                }

            from datetime import datetime

            filters: dict[str, Any] = {}
            if event_type:
                filters["event_type"] = event_type
            if severity:
                filters["severity"] = severity
            if agent_name:
                filters["agent_name"] = agent_name
            if session_id:
                filters["session_id"] = session_id
            if start_time:
                filters["start_time"] = datetime.fromisoformat(start_time).replace(
                    tzinfo=UTC
                )
            if end_time:
                filters["end_time"] = datetime.fromisoformat(end_time).replace(
                    tzinfo=UTC
                )

            events = await self._audit_logger.get_events(filters, limit)

            return {"count": len(events), "events": events}
        except Exception as e:
            logger.error("get_audit_log_error", error=str(e))
            return {"count": 0, "events": [], "error": str(e)}

    # =========================================================================
    # TOOL: get_security_summary (Phase 2.3 - Audit Logging)
    # =========================================================================
    async def get_security_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get aggregated security metrics for last N hours.

        Args:
            hours: Number of hours to look back (default 24).

        Returns:
            Dictionary with total_events, critical_count, events_per_agent,
            events_per_type, and severity_counts.
        """
        try:
            if self._audit_logger is None and self._memory_system is None:
                self._memory_system = await asyncio.wait_for(
                    self._init_memory_system(), timeout=OPERATION_TIMEOUT
                )

            if self._audit_logger is None:
                return {
                    "total_events": 0,
                    "critical_count": 0,
                    "events_per_agent": {},
                    "events_per_type": {},
                    "severity_counts": {},
                    "error": "Audit logger not available (database pool required)",
                }

            summary = await self._audit_logger.get_summary(hours)

            return {"success": True, **summary}
        except Exception as e:
            logger.error("get_security_summary_error", error=str(e))
            return {
                "success": False,
                "total_events": 0,
                "critical_count": 0,
                "events_per_agent": {},
                "events_per_type": {},
                "severity_counts": {},
                "error": str(e),
            }

    async def _watch_stdio_eof(self) -> None:
        """Watch stdin for EOF and trigger graceful shutdown.

        When running in stdio transport mode, OpenCode communicates via
        stdin/stdout. If OpenCode closes its side of the pipe, stdin
        will return EOF (empty bytes). This coroutine monitors for that
        condition and triggers a clean shutdown.
        """
        try:
            loop = asyncio.get_running_loop()
            reader = asyncio.StreamReader()

            try:
                read_transport, _ = await loop.connect_read_pipe(
                    lambda: asyncio.StreamReaderProtocol(reader),
                    sys.stdin,
                )
            except (OSError, ValueError, RuntimeError):
                # If we can't connect to stdin pipe, just wait for signal
                logger.warning("stdio_watch_pipe_connect_failed")
                return

            try:
                # Read until EOF - when OpenCode closes pipe, this returns empty
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        logger.info("stdio_eof_detected")
                        break
            except asyncio.CancelledError:
                # Task was cancelled during shutdown - that's fine
                return
            finally:
                # Clean up the transport
                if hasattr(read_transport, "close"):
                    read_transport.close()

            # Trigger graceful shutdown
            await self._shutdown()
            sys.exit(0)
        except asyncio.CancelledError:
            # Expected during shutdown
            pass
        except Exception:
            # Unexpected error in stdio watcher - don't crash, just log
            logger.warning("stdio_watch_error")

    def run(
        self,
        transport: Literal[
            "stdio", "http", "sse", "streamable-http"
        ] = "streamable-http",
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """Run the MCP server.

        Args:
            transport: Transport type - "stdio", "streamable-http" or other FastMCP option
            host: Host to bind to (default: 127.0.0.1)
            port: Port to bind to (default: 8765)
        """
        if transport == "stdio":
            # Start stdio EOF watcher as a background task
            # This detects when OpenCode closes its side of the pipe
            try:
                loop = asyncio.get_running_loop()
                self._stdio_watch_task = loop.create_task(self._watch_stdio_eof())
                logger.info("stdio_eof_watcher_started")
            except RuntimeError:
                # No running loop yet, will be created by FastMCP
                self._stdio_watch_task = asyncio.ensure_future(self._watch_stdio_eof())
                logger.info("stdio_eof_watcher_started")

        self._mcp.run(transport=transport, host=host, port=port)


def create_server() -> tuple[MCPServer, FastMCP]:
    """Create and return both MCPServer instance and FastMCP server."""
    mcp_server = MCPServer()
    return mcp_server, mcp_server._mcp


# Module-level server instance
_server: MCPServer | None = None
_mcp: FastMCP | None = None


def _create_server() -> FastMCP:
    """Create FastMCP server (legacy module-level interface)."""
    global _server, _mcp
    _server, _mcp = create_server()
    return _mcp


# Create server instance at module level
server = _create_server()
