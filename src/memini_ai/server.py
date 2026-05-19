"""FastMCP server for memini-ai v3.0 - MCP tools for memory operations."""

from __future__ import annotations

import asyncio
import signal
import uuid
from typing import Any

from fastmcp import FastMCP

from memini_ai.config import get_config
from memini_ai.decay import DecayEngine, ConsolidationEngine, adjust_decay_rate
from memini_ai.dialectic import DialecticEngine, get_dialectic_engine
from memini_ai.extractor import MemoryExtractor
from memini_ai.graph import MemoryGraph
from memini_ai.indexer.indexer import IndexerConfig, ProjectIndexer
from memini_ai.knowledge_graph import (
    EntityType,
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
from memini_ai.tiered_loader import TieredLoader
from memini_ai.trust_engine import TrustEngine
from memini_ai.user_model import UserModel
from memini_ai.utils.logger import logger

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
    - Graceful degradation when Qdrant unavailable
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
        self._init_error: str | None = None
        self._background_jobs: dict[str, asyncio.Task[dict[str, Any]]] = {}
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

    def _setup_signal_handlers(self) -> None:
        """Set up SIGINT/SIGTERM handlers."""
        try:
            loop = asyncio.get_running_loop()

            async def shutdown_handler(sig_num: int) -> None:
                await self._shutdown()

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(shutdown_handler(s))
                )  # type: ignore[misc]
        except (NotImplementedError, AttributeError, RuntimeError):
            # Windows or other platforms without signal handlers
            pass

    async def _init_memory_system(self) -> MemorySystem:
        """Initialize memory system with exponential backoff retry."""
        system = MemorySystem()

        # Exponential backoff retry for Qdrant
        max_attempts = 3
        base_delay = 1.0
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                await system.initialize()
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
    ) -> dict[str, Any]:
        """Add a new memory entry with deduplication.

        Args:
            content: The memory content text.
            sourceType: Source type - "session", "file", "web", "boomerang", "project" (default "manual").
            sourcePath: Optional source path or URL.
            metadata: Optional metadata dictionary.

        Returns:
            Dictionary with success status, memory ID, and message.
        """
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

            # Create memory entry
            entry = MemoryEntry(
                text=content,
                sourceType=src_type,
                sourcePath=sourcePath,
            )

            # Add metadata if provided
            import json

            if metadata:
                entry.metadata_json = json.dumps(metadata)

            # Try to add (may raise on duplicate)
            memory_id = await asyncio.wait_for(
                self._memory_system.add_memory(entry), timeout=OPERATION_TIMEOUT
            )

            return {
                "success": True,
                "id": memory_id,
                "message": "Memory added successfully",
            }
        except TimeoutError:
            logger.error("add_memory_timeout", content_length=len(content))
            return {"success": False, "id": "", "message": "Operation timed out"}
        except ValueError as e:
            # Duplicate content
            return {"success": False, "id": "", "message": str(e)}
        except Exception as e:
            logger.error("add_memory_error", error=str(e))
            return {"success": False, "id": "", "message": str(e)}

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

        Returns:
            Dictionary with memoryReady, modelReady, indexerReady, and initError.
        """
        # Check memory system
        memory_ready = False
        if self._memory_system is not None:
            memory_ready = self._memory_system.is_ready

        # Check model (always ready if we got here)
        model_ready = True
        try:
            from memini_ai.model.manager import ModelManager

            manager = ModelManager.get_instance()
            model_ready = manager.get_dimensions() > 0
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
        kg_ready = self._knowledge_graph is not None and self._knowledge_graph.is_enabled

        # Check multi-peer manager
        multi_peer_ready = (
            self._multi_peer_manager is not None and self._multi_peer_manager.is_enabled
        )

        # Check dialectic engine
        dialectic_ready = (
            self._dialectic_engine is not None and self._dialectic_engine.is_enabled
        )

        return {
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
            "initError": self._init_error,
        }

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
    ) -> dict[str, Any]:
        """Find memories related to a given memory.

        Args:
            memoryId: ID of the reference memory.
            relationshipType: Optional filter by relationship type ("SUPERSEDES", "RELATED_TO", "CONTRADICTS", "DERIVED_FROM").
            limit: Maximum number of results (default 10).

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
                self._memory_system.find_related_memories(memoryId, rel_type, limit),
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
            relationshipType: Type of relationship - "SUPERSEDES", "RELATED_TO", "CONTRADICTS", "DERIVED_FROM".
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
            return {"fading_count": 0, "fading_memories": [], "error": "Operation timed out"}
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
    # GRACEFUL SHUTDOWN
    # =========================================================================
    async def _shutdown(self) -> None:
        """Graceful shutdown handler."""
        logger.info("server_shutdown_started")

        # Cancel all background jobs
        for job_id, task in list(self._background_jobs.items()):
            if not task.done():
                task.cancel()
                logger.info("background_job_cancelled", job_id=job_id)

        # Stop indexer
        if self._indexer is not None and self._indexer.is_running:
            await self._indexer.stop()
            logger.info("indexer_stopped")

        # Stop memory system (close db connections)
        if self._memory_system is not None and self._memory_system.is_initialized:
            # MemorySystem doesn't have a close method yet
            # but we log the shutdown
            logger.info("memory_system_shutdown")

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
                self._knowledge_graph = KnowledgeGraph(memory_system=self._memory_system)

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
            logger.error("get_dialectic_history_error", error=str(e), memory_id=memory_id)
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
            logger.error(
                "challenge_memory_error", error=str(e), memory_id=memory_id
            )
            return {"success": False, "error": str(e)}

    def run(
        self,
        transport: str = "streamable-http",
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """Run the MCP server.

        Args:
            transport: Transport type - "streamable-http" or "stdio"
            host: Host to bind to (default: 127.0.0.1)
            port: Port to bind to (default: 8765)
        """
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
