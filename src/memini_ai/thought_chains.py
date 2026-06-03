"""Thought Chains module - persistent reasoning chains with trust integration.

Provides API-compatible sequential thinking with branching, revision,
and deep integration into memini-ai's memory, trust, and knowledge graph systems.

Each thought is stored in BOTH the thoughts table (for structural queries)
AND the memories table (for semantic search, trust scoring, and tiered loading).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import asyncpg

from memini_ai.config import get_config
from memini_ai.memory.schema import MemoryEntry, MemorySourceType
from memini_ai.model.embeddings import generate_embedding
from memini_ai.utils.hash import hash_content
from memini_ai.utils.logger import logger

if TYPE_CHECKING:
    from memini_ai.memory.system import MemorySystem
    from memini_ai.trust_engine import TrustEngine


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThoughtChain:
    """A thought chain (reasoning session)."""

    id: str
    session_id: str | None = None
    parent_chain_id: str | None = None
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Thought:
    """A single thought within a thought chain."""

    id: str
    chain_id: str
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool = True
    is_revision: bool = False
    revises_thought_id: str | None = None
    branch_from_thought_id: str | None = None
    branch_id: str | None = None
    content_hash: str = ""
    memory_id: str | None = None
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# ThoughtChains class
# ---------------------------------------------------------------------------


class ThoughtChains:
    """Manage persistent thought chains with trust integration.

    Thought chains provide structured reasoning that integrates with
    memini-ai's memory system for semantic search, trust scoring,
    and knowledge graph operations.

    All features are gated behind the THOUGHT_CHAINS config flag.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        memory_system: MemorySystem | None = None,
        trust_engine: TrustEngine | None = None,
    ) -> None:
        """Initialize ThoughtChains.

        Args:
            pool: asyncpg connection pool.
            memory_system: Optional MemorySystem for dual storage.
            trust_engine: Optional TrustEngine for trust integration.
        """
        self._pool = pool
        self._memory_system = memory_system
        self._trust_engine = trust_engine

    @property
    def is_enabled(self) -> bool:
        """Check if thought chains are enabled via config."""
        return get_config().thought_chains_enabled

    def _check_enabled(self) -> dict[str, Any] | None:
        """Return error dict if thought chains are not enabled."""
        if not self.is_enabled:
            return {
                "error": (
                    "Thought chains not enabled. "
                    "Set THOUGHT_CHAINS=true or THOUGHT_CHAINS=1 in environment."
                ),
            }
        return None

    async def _check_table_exists(self) -> dict[str, Any] | None:
        """Return error dict if thought_chains table doesn't exist."""
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'thought_chains')"
            )
            if not result:
                return {
                    "error": (
                        "thought_chains table not found. "
                        "Run migration: python scripts/migrate_thought_chains.py"
                    ),
                }
        return None

    # -------------------------------------------------------------------
    # Chain Operations
    # -------------------------------------------------------------------

    async def start_chain(
        self,
        session_id: str | None = None,
        parent_chain_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new thought chain.

        Args:
            session_id: Optional session identifier.
            parent_chain_id: Optional parent chain ID for hierarchical chains.

        Returns:
            Dict with chain_id, session_id, created_at.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        table_error = await self._check_table_exists()
        if table_error:
            return table_error

        chain_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO thought_chains (id, session_id, parent_chain_id, status) "
                "VALUES ($1, $2, $3, 'active') "
                "RETURNING id, session_id, parent_chain_id, status, created_at",
                uuid.UUID(chain_id),
                session_id,
                uuid.UUID(parent_chain_id) if parent_chain_id else None,
            )

        return {
            "chain_id": str(row["id"]),
            "session_id": row["session_id"],
            "status": row["status"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def get_chain(self, chain_id: str) -> dict[str, Any]:
        """Retrieve a full thought chain with all thoughts organized by branch.

        Args:
            chain_id: UUID of the chain to retrieve.

        Returns:
            Dict with chain metadata, thoughts list, and branch map.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        table_error = await self._check_table_exists()
        if table_error:
            return table_error

        async with self._pool.acquire() as conn:
            # Get chain metadata
            chain_row = await conn.fetchrow(
                "SELECT id, session_id, parent_chain_id, status, created_at, updated_at "
                "FROM thought_chains WHERE id = $1",
                uuid.UUID(chain_id),
            )
            if not chain_row:
                return {"error": f"Chain {chain_id} not found"}

            # Get all thoughts in the chain
            thought_rows = await conn.fetch(
                "SELECT id, chain_id, thought, thought_number, total_thoughts, "
                "next_thought_needed, is_revision, revises_thought_id, "
                "branch_from_thought_id, branch_id, content_hash, memory_id, created_at "
                "FROM thoughts WHERE chain_id = $1 "
                "ORDER BY thought_number ASC, created_at ASC",
                uuid.UUID(chain_id),
            )

            # Get distinct branches (for future use in branch tracking)
            _ = await conn.fetch(
                "SELECT DISTINCT branch_id FROM thoughts "
                "WHERE chain_id = $1 AND branch_id IS NOT NULL "
                "ORDER BY branch_id",
                uuid.UUID(chain_id),
            )

            # Count thoughts
            thought_count = await conn.fetchval(
                "SELECT COUNT(*) FROM thoughts WHERE chain_id = $1",
                uuid.UUID(chain_id),
            )

        # Build thoughts list
        thoughts = []
        for row in thought_rows:
            thoughts.append(
                {
                    "id": str(row["id"]),
                    "thought": row["thought"],
                    "thoughtNumber": row["thought_number"],
                    "totalThoughts": row["total_thoughts"],
                    "nextThoughtNeeded": row["next_thought_needed"],
                    "isRevision": row["is_revision"],
                    "revisesThoughtId": str(row["revises_thought_id"])
                    if row["revises_thought_id"]
                    else None,
                    "branchFromThoughtId": str(row["branch_from_thought_id"])
                    if row["branch_from_thought_id"]
                    else None,
                    "branchId": row["branch_id"],
                    "memoryId": str(row["memory_id"]) if row["memory_id"] else None,
                    "createdAt": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                }
            )

        # Build branch map
        branch_map: dict[str, list[dict[str, Any]]] = {}
        for t in thoughts:
            bid = t["branchId"] or "main"
            if bid not in branch_map:
                branch_map[bid] = []
            branch_map[bid].append(t)

        return {
            "chain_id": str(chain_row["id"]),
            "session_id": chain_row["session_id"],
            "parent_chain_id": str(chain_row["parent_chain_id"])
            if chain_row["parent_chain_id"]
            else None,
            "status": chain_row["status"],
            "thoughts": thoughts,
            "branchMap": branch_map,
            "thought_count": thought_count,
            "created_at": chain_row["created_at"].isoformat()
            if chain_row["created_at"]
            else None,
            "updated_at": chain_row["updated_at"].isoformat()
            if chain_row["updated_at"]
            else None,
        }

    async def pause_chain(self, chain_id: str) -> dict[str, Any]:
        """Pause a thought chain.

        Args:
            chain_id: UUID of the chain to pause.

        Returns:
            Dict with success, chain_id, previous_status, new_status.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        # Get current status
        async with self._pool.acquire() as conn:
            current = await conn.fetchval(
                "SELECT status FROM thought_chains WHERE id = $1",
                uuid.UUID(chain_id),
            )
            if current is None:
                return {"error": f"Chain {chain_id} not found"}

            previous_status = current

            row = await conn.fetchrow(
                "UPDATE thought_chains SET status = 'paused', updated_at = NOW() "
                "WHERE id = $1 "
                "RETURNING id, status",
                uuid.UUID(chain_id),
            )

        return {
            "success": True,
            "chain_id": chain_id,
            "previous_status": previous_status,
            "new_status": row["status"] if row else "paused",
        }

    async def resume_chain(self, chain_id: str) -> dict[str, Any]:
        """Resume a paused thought chain.

        Args:
            chain_id: UUID of the chain to resume.

        Returns:
            Dict with success, chain_id, previous_status, new_status,
            thought_count, and last_thought.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        async with self._pool.acquire() as conn:
            current = await conn.fetchval(
                "SELECT status FROM thought_chains WHERE id = $1",
                uuid.UUID(chain_id),
            )
            if current is None:
                return {"error": f"Chain {chain_id} not found"}

            previous_status = current

            # Update status
            await conn.execute(
                "UPDATE thought_chains SET status = 'active', updated_at = NOW() "
                "WHERE id = $1",
                uuid.UUID(chain_id),
            )

            # Get last thought for continuity
            last_thought_row = await conn.fetchrow(
                "SELECT id, thought, thought_number, total_thoughts, "
                "next_thought_needed FROM thoughts "
                "WHERE chain_id = $1 "
                "ORDER BY thought_number DESC, created_at DESC LIMIT 1",
                uuid.UUID(chain_id),
            )

            thought_count = await conn.fetchval(
                "SELECT COUNT(*) FROM thoughts WHERE chain_id = $1",
                uuid.UUID(chain_id),
            )

        last_thought = None
        if last_thought_row:
            last_thought = {
                "thoughtNumber": last_thought_row["thought_number"],
                "totalThoughts": last_thought_row["total_thoughts"],
                "thought": last_thought_row["thought"],
                "nextThoughtNeeded": last_thought_row["next_thought_needed"],
            }

        return {
            "success": True,
            "chain_id": chain_id,
            "previous_status": previous_status,
            "new_status": "active",
            "thought_count": thought_count,
            "last_thought": last_thought,
        }

    async def abandon_chain(self, chain_id: str) -> dict[str, Any]:
        """Abandon a thought chain. Applies agent_ignored trust signal.

        Args:
            chain_id: UUID of the chain to abandon.

        Returns:
            Dict with success, chain_id, previous_status, new_status.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        async with self._pool.acquire() as conn:
            current = await conn.fetchval(
                "SELECT status FROM thought_chains WHERE id = $1",
                uuid.UUID(chain_id),
            )
            if current is None:
                return {"error": f"Chain {chain_id} not found"}

            previous_status = current

            await conn.execute(
                "UPDATE thought_chains SET status = 'abandoned', updated_at = NOW() "
                "WHERE id = $1",
                uuid.UUID(chain_id),
            )

            # Get all memory_ids for trust adjustment
            memory_ids = await conn.fetch(
                "SELECT memory_id FROM thoughts "
                "WHERE chain_id = $1 AND memory_id IS NOT NULL",
                uuid.UUID(chain_id),
            )

        # Apply agent_ignored trust signal to all thoughts
        if self._trust_engine and self._trust_engine.is_enabled:
            from memini_ai.memory.schema import TrustSignal

            for row in memory_ids:
                try:
                    await self._trust_engine.adjust_trust(
                        str(row["memory_id"]),
                        TrustSignal.AGENT_IGNORED,
                    )
                except Exception as e:
                    logger.warning(
                        "abandon_chain_trust_error",
                        memory_id=str(row["memory_id"]),
                        error=str(e),
                    )

        return {
            "success": True,
            "chain_id": chain_id,
            "previous_status": previous_status,
            "new_status": "abandoned",
        }

    # -------------------------------------------------------------------
    # Thought Operations
    # -------------------------------------------------------------------

    async def add_thought(
        self,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        next_thought_needed: bool,
        is_revision: bool = False,
        revises_thought: int | None = None,
        branch_from_thought: int | None = None,
        branch_id: str | None = None,
        chain_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a thought to a reasoning chain.

        API-compatible with @modelcontextprotocol/server-sequential-thinking.
        Auto-creates chain if chain_id not provided.
        Stores thought in BOTH thoughts table AND memories table.

        Args:
            thought: The thought text.
            thought_number: Current thought number.
            total_thoughts: Total expected thoughts.
            next_thought_needed: Whether more thoughts are needed.
            is_revision: Whether this is a revision.
            revises_thought: Thought number being revised.
            branch_from_thought: Thought number to branch from.
            branch_id: Branch identifier.
            chain_id: Chain UUID (auto-created if None).
            session_id: Session identifier (used when auto-creating chains).

        Returns:
            Dict with thoughtNumber, totalThoughts, nextThoughtNeeded,
            chain_id, branches, thoughtHistoryLength.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        table_error = await self._check_table_exists()
        if table_error:
            return table_error

        # Auto-adjust total_thoughts if thought_number exceeds it
        if thought_number > total_thoughts:
            total_thoughts = thought_number

        # Auto-create chain if not provided
        if chain_id is None:
            chain_result = await self.start_chain(session_id=session_id)
            if "error" in chain_result:
                return chain_result
            chain_id = chain_result["chain_id"]

        # Generate embedding for the thought text.
        # IMPORTANT (v0.7.1 bugfix): Pass a Python list[float] directly to asyncpg.
        # Earlier versions stringified the vector to a pgvector literal (e.g.
        # `"[0.1,0.2,...]"`) and passed it to `$N::vector`, which asyncpg could
        # not bind correctly — it threw
        # "expected 384 dimensions, not 1024" / "could not convert string to
        # float" errors at runtime. Passing the raw list lets the registered
        # pgvector codec (see `register_vector` in `postgres/database.py`) do
        # the binding, matching how `memory.add` already does it.
        #
        # Also handle the dimension-mismatch case: the thoughts table column is
        # hardcoded to `vector(384)` (see `postgres/schema.py`), but the
        # embedding model may return 1024-dim BGE-Large vectors if a GPU is
        # available (ModelManager prefers BGE-Large on CUDA). In that case we
        # truncate to the first 384 dims to avoid a `expected 384 dimensions,
        # not 1024` Postgres error. (An ideal long-term fix is to widen the
        # column to `vector` and store whatever dim the model emits, but that
        # requires a migration and an HNSW re-index.)
        embedding: list[float] | None = None
        try:
            embedding_result = await generate_embedding(thought)
            vec: list[float] = list(embedding_result.embedding)
            # Truncate or zero-pad to 384 dims to match the column.
            if len(vec) > 384:
                vec = vec[:384]
            elif len(vec) < 384:
                vec = vec + [0.0] * (384 - len(vec))
            embedding = vec
        except Exception as e:
            logger.warning("thought_embedding_failed", error=str(e))

        # Compute content hash
        content_hash = hash_content(thought)

        # Resolve revises_thought_id and branch_from_thought_id
        revises_thought_id: uuid.UUID | None = None
        branch_from_thought_id: uuid.UUID | None = None

        async with self._pool.acquire() as conn:
            if revises_thought is not None:
                row = await conn.fetchrow(
                    "SELECT id FROM thoughts "
                    "WHERE chain_id = $1 AND thought_number = $2 "
                    "ORDER BY created_at DESC LIMIT 1",
                    uuid.UUID(chain_id),
                    revises_thought,
                )
                if row:
                    revises_thought_id = row["id"]

            if branch_from_thought is not None:
                row = await conn.fetchrow(
                    "SELECT id FROM thoughts "
                    "WHERE chain_id = $1 AND thought_number = $2 "
                    "ORDER BY created_at DESC LIMIT 1",
                    uuid.UUID(chain_id),
                    branch_from_thought,
                )
                if row:
                    branch_from_thought_id = row["id"]

        # Store as memory first (dual storage)
        memory_id: str | None = None
        if self._memory_system is not None:
            try:
                entry = MemoryEntry(
                    text=thought,
                    sourceType=MemorySourceType.thought,
                    sourcePath=f"thought-chain:{chain_id}",
                )
                memory_id = await self._memory_system.add_memory(entry)
                logger.debug(
                    "thought_stored_as_memory",
                    thought_number=thought_number,
                    memory_id=memory_id,
                )
            except ValueError:
                # Duplicate content — try with slight modification
                try:
                    entry = MemoryEntry(
                        text=f"[t{thought_number}] {thought}",
                        sourceType=MemorySourceType.thought,
                        sourcePath=f"thought-chain:{chain_id}",
                    )
                    memory_id = await self._memory_system.add_memory(entry)
                except Exception as e:
                    logger.warning("thought_memory_duplicate", error=str(e))
            except Exception as e:
                logger.warning("thought_memory_store_failed", error=str(e))

        # Insert thought into thoughts table
        thought_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO thoughts "
                "(id, chain_id, thought, thought_number, total_thoughts, "
                "next_thought_needed, is_revision, revises_thought_id, "
                "branch_from_thought_id, branch_id, embedding, content_hash, memory_id) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, "
                "$11, $12, $13) "  # $11 binds list[float] via pgvector codec (v0.7.1 fix)
                "RETURNING id",
                uuid.UUID(thought_id),
                uuid.UUID(chain_id),
                thought,
                thought_number,
                total_thoughts,
                next_thought_needed,
                is_revision,
                revises_thought_id,
                branch_from_thought_id,
                branch_id,
                embedding,
                content_hash,
                uuid.UUID(memory_id) if memory_id else None,
            )

            # Update memory_id on the thought if it was created after
            if memory_id and row:
                pass  # Already set during insert

            # Get current branches
            branch_rows = await conn.fetch(
                "SELECT DISTINCT branch_id FROM thoughts "
                "WHERE chain_id = $1 AND branch_id IS NOT NULL "
                "ORDER BY branch_id",
                uuid.UUID(chain_id),
            )

            # Count total thoughts
            thought_count = await conn.fetchval(
                "SELECT COUNT(*) FROM thoughts WHERE chain_id = $1",
                uuid.UUID(chain_id),
            )

        # Create SUPERSEDES relationship for revisions
        if is_revision and revises_thought_id and memory_id and self._memory_system:
            try:
                from memini_ai.memory.schema import RelationshipType

                old_memory_row = await self._get_thought_memory_id(
                    str(revises_thought_id),
                )
                if old_memory_row and old_memory_row.get("memory_id"):
                    await self._memory_system.create_relationship(
                        memory_id,
                        str(old_memory_row["memory_id"]),
                        RelationshipType.SUPERSEDES,
                    )
            except Exception as e:
                logger.warning("thought_supersedes_failed", error=str(e))

        # Create DERIVED_FROM relationship for branches
        if branch_from_thought_id and memory_id and self._memory_system:
            try:
                from memini_ai.memory.schema import RelationshipType

                parent_memory_row = await self._get_thought_memory_id(
                    str(branch_from_thought_id),
                )
                if parent_memory_row and parent_memory_row.get("memory_id"):
                    await self._memory_system.create_relationship(
                        memory_id,
                        str(parent_memory_row["memory_id"]),
                        RelationshipType.DERIVED_FROM,
                    )
            except Exception as e:
                logger.warning("thought_branch_relationship_failed", error=str(e))

        # Auto-extract entities if knowledge graph is available
        if self._memory_system is not None:
            try:
                # Entity extraction happens via the memory system
                # The add_memory above already stores it; KG extraction
                # is handled by the knowledge graph when enabled
                pass
            except Exception as e:
                logger.warning("thought_entity_extraction_failed", error=str(e))

        branches = [row["branch_id"] for row in branch_rows]

        return {
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": next_thought_needed,
            "chain_id": chain_id,
            "branches": branches,
            "thoughtHistoryLength": thought_count,
        }

    async def revise_thought(
        self,
        chain_id: str,
        thought_number: int,
        revised_thought: str,
    ) -> dict[str, Any]:
        """Create a revision of an existing thought.

        Creates a new thought that supersedes the old one, with is_revision=True
        and revises_thought_id pointing to the original.

        Args:
            chain_id: UUID of the chain.
            thought_number: Number of the thought to revise.
            revised_thought: New thought text.

        Returns:
            Dict with success, thought_id, chain_id, thought_number.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        # Get the original thought's total_thoughts for the new one
        async with self._pool.acquire() as conn:
            original = await conn.fetchrow(
                "SELECT id, thought_number, total_thoughts FROM thoughts "
                "WHERE chain_id = $1 AND thought_number = $2 "
                "ORDER BY created_at DESC LIMIT 1",
                uuid.UUID(chain_id),
                thought_number,
            )

        if not original:
            return {"error": f"Thought {thought_number} not found in chain {chain_id}"}

        new_total = original["total_thoughts"]

        result = await self.add_thought(
            thought=revised_thought,
            thought_number=thought_number,
            total_thoughts=new_total,
            next_thought_needed=True,
            is_revision=True,
            revises_thought=thought_number,
            chain_id=chain_id,
        )

        return {
            "success": True,
            "thought_id": result.get("thought_id"),
            "chain_id": chain_id,
            "thought_number": thought_number,
        }

    async def branch_thought(
        self,
        chain_id: str,
        from_thought_number: int,
        branch_id: str,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        next_thought_needed: bool,
    ) -> dict[str, Any]:
        """Create a new branch from an existing thought.

        Args:
            chain_id: UUID of the chain.
            from_thought_number: Thought number to branch from.
            branch_id: Branch identifier.
            thought: New thought text.
            thought_number: Thought number in new branch.
            total_thoughts: Total thoughts expected in branch.
            next_thought_needed: Whether more thoughts follow.

        Returns:
            Dict with success, thought_id, chain_id, branch_id, thought_number.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        result = await self.add_thought(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            branch_from_thought=from_thought_number,
            branch_id=branch_id,
            chain_id=chain_id,
        )

        return {
            "success": True,
            "thought_id": result.get("thought_id"),
            "chain_id": chain_id,
            "branch_id": branch_id,
            "thought_number": thought_number,
        }

    async def get_related_chains(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search for thought chains with similar reasoning.

        Uses pgvector cosine similarity on thought embeddings.

        Args:
            query: Search query text.
            limit: Maximum number of results.

        Returns:
            Dict with count and chains list.
        """
        enabled_error = self._check_enabled()
        if enabled_error:
            return enabled_error

        table_error = await self._check_table_exists()
        if table_error:
            return table_error

        # Generate embedding for the query.
        # (v0.7.1 fix: was building a stringified pgvector literal, which
        # asyncpg could not bind correctly. Pass list[float] directly so the
        # registered pgvector codec handles the binding.)
        try:
            embedding_result = await generate_embedding(query)
            query_vec: list[float] = list(embedding_result.embedding)
            if len(query_vec) > 384:
                query_vec = query_vec[:384]
            elif len(query_vec) < 384:
                query_vec = query_vec + [0.0] * (384 - len(query_vec))
            embedding_pg = query_vec
        except Exception as e:
            logger.error("thought_search_embedding_failed", error=str(e))
            return {
                "count": 0,
                "chains": [],
                "error": f"Embedding generation failed: {e}",
            }

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "WITH ranked_thoughts AS ("
                "  SELECT t.id, t.chain_id, t.thought, t.branch_id, "
                "         t.embedding <=> $1::vector as distance "
                "  FROM thoughts t "
                "  JOIN thought_chains tc ON t.chain_id = tc.id "
                "  WHERE t.embedding IS NOT NULL AND tc.status = 'active' "
                "  ORDER BY t.embedding <=> $1::vector "
                "  LIMIT $2 "
                ") "
                "SELECT rt.chain_id, tc.session_id, rt.thought as snippet, "
                "       rt.distance as score, "
                "       (SELECT COUNT(*) FROM thoughts WHERE chain_id = rt.chain_id) as thought_count "
                "FROM ranked_thoughts rt "
                "JOIN thought_chains tc ON rt.chain_id = tc.id "
                "GROUP BY rt.chain_id, tc.session_id, rt.thought, rt.distance, rt.thought_count "
                "ORDER BY MIN(rt.distance) ASC "
                "LIMIT $3",
                embedding_pg,
                limit * 3,  # Get more rows before deduplication
                limit,
            )

        chains = []
        for row in rows:
            chains.append(
                {
                    "chain_id": str(row["chain_id"]),
                    "session_id": row["session_id"],
                    "snippet": row["snippet"],
                    "score": round(float(row["score"]), 4)
                    if row["score"] is not None
                    else None,
                    "thought_count": row["thought_count"],
                }
            )

        return {
            "count": len(chains),
            "chains": chains,
        }

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    async def _get_thought_memory_id(self, thought_id: str) -> dict[str, Any] | None:
        """Get the memory_id for a thought by its UUID.

        Args:
            thought_id: UUID of the thought.

        Returns:
            Row dict or None.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT memory_id FROM thoughts WHERE id = $1",
                uuid.UUID(thought_id),
            )
        if row:
            return {"memory_id": str(row["memory_id"]) if row["memory_id"] else None}
        return None
