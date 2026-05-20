# Design: Thought Chains — Converging Sequential Thinking into memini-ai-dev

> **Author**: boomerang-architect (deepseek-v4-pro:cloud)  
> **Date**: 2026-05-19  
> **Status**: ✅ **APPROVED & IMPLEMENTED** (see Sprints 1-5 below)
> **Target**: memini-ai-dev v0.3.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Schema Design](#2-schema-design)
3. [New MCP Tools](#3-new-mcp-tools)
4. [Integration Points (Boomerang-v3 Orchestrator)](#4-integration-points)
5. [Trust Engine Integration](#5-trust-engine-integration)
6. [Migration / Deprecation Path](#6-migration--deprecation-path)
7. [Effort Estimate](#7-effort-estimate)
8. [Open Questions for User](#8-open-questions-for-user)

---

## 1. Executive Summary

### Problem

The `@modelcontextprotocol/server-sequential-thinking` MCP server is a simple in-memory scratchpad with no persistence, no search, no trust scoring, no multi-session awareness, and no concurrent chain support. Meanwhile, `memini-ai-dev` already has PostgreSQL+pgvector storage, semantic search (4 strategies), trust scoring, a knowledge graph, tiered loading (L0/L1/L2), contradiction detection, and multi-peer sharing.

The goal is to subsume sequential thinking into memini-ai-dev as a first-class feature, deprecating the external MCP server entirely.

### Solution

Add a `thought_chains` module to memini-ai-dev that:

1. Stores thought chains in PostgreSQL with full branching, revision, and multi-chain support
2. Exposes MCP tools that are **API-compatible** with the original `sequentialthinking` tool for easy migration
3. Stores thoughts as both `thoughts` table rows AND `memories` table entries (sourceType=`thought`) – giving semantic search, trust scoring, and tiered summary participation for free
4. Provides additional power tools: `get_related_chains` (semantic search), `pause/resume/abandon`, `get_thought_chain`
5. Integrates with the existing trust engine so high-quality reasoning chains are promoted to L1 summaries

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **API compatibility first** | The `add_thought` tool accepts the EXACT same 9 parameters as the original `sequentialthinking` tool, with one addition: optional `chain_id` |
| **Dual storage (hybrid)** | Each thought lives in BOTH a dedicated `thoughts` table (structural integrity) AND the `memories` table (semantic search, trust, graph) |
| **Opt-in** | All new functionality gated behind `THOUGHT_CHAINS` env var (default: `false`), following existing memini-ai pattern |
| **Graceful degradation** | If the `thought_chains` table doesn't exist, tools return descriptive errors; agents can fall back to old server during transition |
| **Multi-peer** | Thought chains inherit memini-ai's existing peer-aware architecture – other agents can query your reasoning |

---

## 2. Schema Design

### 2.1 new PostgreSQL Tables

#### `thought_chains`

```sql
CREATE TABLE IF NOT EXISTS thought_chains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),              -- Links to the agent session
    parent_chain_id UUID REFERENCES thought_chains(id) ON DELETE SET NULL,
                                          -- NULL for root chains, set for sub-agent chains
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'abandoned')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_thought_chains_session ON thought_chains(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_thought_chains_parent ON thought_chains(parent_chain_id) WHERE parent_chain_id IS NOT NULL;
CREATE INDEX idx_thought_chains_status ON thought_chains(status) WHERE status = 'active';
```

#### `thoughts`

```sql
CREATE TABLE IF NOT EXISTS thoughts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id UUID NOT NULL REFERENCES thought_chains(id) ON DELETE CASCADE,
    
    -- Original sequential-thinking fields
    thought TEXT NOT NULL,
    thought_number INTEGER NOT NULL CHECK (thought_number >= 1),
    total_thoughts INTEGER NOT NULL CHECK (total_thoughts >= 1),
    next_thought_needed BOOLEAN NOT NULL,
    
    -- Revision support
    is_revision BOOLEAN DEFAULT FALSE,
    revises_thought_id UUID REFERENCES thoughts(id) ON DELETE SET NULL,
    
    -- Branching support
    branch_from_thought_id UUID REFERENCES thoughts(id) ON DELETE SET NULL,
    branch_id VARCHAR(255),
    
    -- memini-ai additions
    embedding vector(384),                -- 384-dim MiniLM embedding for semantic search
    content_hash VARCHAR(64),             -- SHA-256 of the thought text for deduplication
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
                                          -- Links to the corresponding memory entry
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_thoughts_chain ON thoughts(chain_id);
CREATE INDEX idx_thoughts_embedding ON thoughts USING diskann (embedding vector_cosine_ops);
CREATE INDEX idx_thoughts_branch ON thoughts(branch_id) WHERE branch_id IS NOT NULL;
CREATE INDEX idx_thoughts_revises ON thoughts(revises_thought_id) WHERE revises_thought_id IS NOT NULL;
CREATE INDEX idx_thoughts_memory ON thoughts(memory_id) WHERE memory_id IS NOT NULL;
```

#### Diagram

```
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐
│  thought_chains  │       │     thoughts      │       │   memories   │
│                  │       │                   │       │              │
│ id            ◄──┼───────┤ chain_id          │       │ id ◄─────────┼── memory_id FK
│ session_id       │       │ id                │       │ text         │
│ parent_chain_id◄─┼── self-ref               │       │ source_type  │  = "thought"
│ status           │       │ thought           │       │ embedding    │
│ created_at       │       │ thought_number    │       │ trust_score  │
│ updated_at       │       │ total_thoughts    │       │ content_hash │
└─────────────────┘       │ next_thought_needed│       │ metadata     │
                          │ is_revision        │       └──────────────┘
                          │ revises_thought_id ◄┼── self-ref
                          │ branch_from_thought_id ◄ self-ref
                          │ branch_id          │
                          │ embedding          │
                          │ content_hash       │
                          │ memory_id          │
                          └───────────────────┘
```

### 2.2 Schema Updates to Existing Tables

#### memories table

No schema changes needed. The `source_type` column already supports values `['session', 'file', 'web', 'boomerang', 'project']`. We add `'thought'` to this enum in application-layer validation (no ALTER TABLE needed since it's a CHECK constraint with listed values).

**Migration needed**: `ALTER TABLE memories ADD CONSTRAINT ...` to add `'thought'` to the source_type CHECK constraint. See [Section 6](#6-migration--deprecation-path).

#### MemoryEntry Pydantic model

In `src/memini_ai/memory/schema.py`, add `'thought'` to the `sourceType` Literal type:

```python
sourceType: Literal["session", "file", "web", "boomerang", "project", "thought"] = "manual"
```

### 2.3 Why Hybrid Storage?

| Question | If stored ONLY in `thoughts` table | If stored ONLY in `memories` table | **Hybrid (both)** |
|----------|-------------------------------------|-------------------------------------|-------------------|
| Semantic search | ❌ Need custom search | ✅ pgvector already exists | ✅ Free via memories |
| Trust scoring | ❌ Need custom implementation | ✅ Built-in | ✅ Free via memories |
| Knowledge graph | ❌ Need custom extraction | ✅ Entity extraction works | ✅ Free via memories |
| Tiered summaries | ❌ Need custom prompts | ✅ L0/L1 auto-include | ✅ Free via memories |
| Structural query (chain, branches) | ✅ Clean relational model | ❌ No native chain grouping | ✅ thoughts table |
| Revision tracking | ✅ FK self-reference | ❌ Would need metadata parsing | ✅ thoughts table |
| Deduplication | ❌ Manual | ✅ content_hash | ✅ Both layers |
| Storage overhead | 1 row per thought | 1 row per thought | 2 rows per thought (~negligible) |

The hybrid approach gives us the best of both with minimal overhead. A typical thought is ~200 characters, so double storage is <1KB per thought.

---

## 3. New MCP Tools

### 3.1 Tool Catalog (9 tools)

| # | Tool Name | Parameters | Returns | Description |
|---|-----------|-----------|---------|-------------|
| 1 | `add_thought` | `thought: str, thoughtNumber: int, totalThoughts: int, nextThoughtNeeded: bool, isRevision: bool=False, revisesThought: int\|None, branchFromThought: int\|None, branchId: str\|None, chain_id: str\|None, session_id: str\|None` | `{thoughtNumber, totalThoughts, nextThoughtNeeded, chain_id, branches[], thoughtHistoryLength}` | **API-compatible** with original `sequentialthinking`. Auto-creates chain if `chain_id` not provided. |
| 2 | `start_thought_chain` | `session_id: str=None, parent_chain_id: str=None` | `{chain_id, session_id, created_at}` | Explicitly create a new chain. Returns chain_id for subsequent `add_thought` calls. |
| 3 | `get_thought_chain` | `chain_id: str` | `{chain_id, session_id, status, thoughts[], branchMap{}, thought_count}` | Retrieve full chain with all thoughts, organized by branch. |
| 4 | `get_related_chains` | `query: str, limit: int=10` | `{count, chains[{chain_id, session_id, snippet, score, thought_count}]}` | Semantic search across thought embeddings. Finds chains with similar reasoning. |
| 5 | `revise_thought` | `chain_id: str, thought_number: int, revised_thought: str` | `{success, thought_id, chain_id, thought_number}` | Mark a thought as revised by a new thought. Creates the revision thought. |
| 6 | `branch_thought` | `chain_id: str, from_thought_number: int, branch_id: str, thought: str, thoughtNumber: int, totalThoughts: int, nextThoughtNeeded: bool` | `{success, thought_id, chain_id, branch_id, thought_number}` | Start a new branch from an existing thought. |
| 7 | `pause_thought_chain` | `chain_id: str` | `{success, chain_id, previous_status, new_status}` | Mark a chain as paused (e.g., context window eviction). |
| 8 | `resume_thought_chain` | `chain_id: str` | `{success, chain_id, previous_status, new_status, thought_count, last_thought}` | Resume a paused chain. Returns the last thought for continuity. |
| 9 | `abandon_thought_chain` | `chain_id: str` | `{success, chain_id, previous_status, new_status}` | Mark a chain as abandoned. |

### 3.2 Detailed Tool Schemas

#### `add_thought` (Primary Tool — API-Compatible)

```python
@mcp.tool()
async def add_thought(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool = False,
    revisesThought: int | None = None,
    branchFromThought: int | None = None,
    branchId: str | None = None,
    chain_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Add a thought to a reasoning chain. API-compatible with the original 
    @modelcontextprotocol/server-sequential-thinking tool.
    
    If chain_id is not provided, a new chain is automatically created.
    The thought is stored both as a structural thought AND as a semantic memory
    entry (sourceType="thought") for search, trust scoring, and tiered summaries.
    
    Returns:
        thoughtNumber: Echo of input thought number
        totalThoughts: Echo of input (auto-adjusted if thoughtNumber > totalThoughts)
        nextThoughtNeeded: Echo of input
        chain_id: UUID of the chain (useful after auto-creation on first call)
        branches: List of active branch IDs in this chain
        thoughtHistoryLength: Total number of thoughts ever added to this chain
    """
```

**Key behaviors**:
- If `chain_id` is `None`, auto-creates a new chain (returns `chain_id` in response)
- If `thoughtNumber > totalThoughts`, auto-adjusts `totalThoughts` to match (matching original behavior)
- The thought text is embedded and stored in BOTH the `thoughts` table and the `memories` table
- The `memory_id` FK is set to the newly created memory entry
- Auto-extracts entities from the thought text into the knowledge graph

#### `get_related_chains`

```python
@mcp.tool()
async def get_related_chains(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Search for thought chains with similar reasoning to the query.
    Uses pgvector cosine similarity on thought embeddings.
    
    Returns chains ranked by relevance, with a snippet of the best-matching thought.
    
    Returns:
        count: Number of matching chains
        chains: List of {chain_id, session_id, snippet, score, thought_count}
    """
```

#### `resume_thought_chain`

```python
@mcp.tool()
async def resume_thought_chain(
    chain_id: str,
) -> dict:
    """
    Resume a paused thought chain. Returns the last thought so the agent
    can continue reasoning from where it left off.
    
    Returns:
        success: Whether the resume succeeded
        chain_id: The chain ID
        previous_status: The status before resume (should be "paused")
        new_status: "active"
        thought_count: Number of thoughts in the chain
        last_thought: {thoughtNumber, totalThoughts, thought, nextThoughtNeeded}
    """
```

### 3.3 Tool Mapping to Existing memini-ai Concepts

| New Tool | memini-ai Feature Used | Benefit |
|----------|----------------------|---------|
| `add_thought` | `MemorySystem.add_memory()` for memory entry, `EntityExtractor` for KG | Semantic search, trust, graph |
| `get_related_chains` | `MemorySystem.query_memories()` with `strategy="vector_only"` on `sourceType="thought"` | Finds similar reasoning |
| `revise_thought` | `MemoryGraph.create_relationship()` with `SUPERSEDES` | Tracks revisions |
| `pause_thought_chain` | `TrustEngine.record_retrieval()` to prevent decay | Prevent premature archiving |
| All tools | `TrustEngine` via memory entries | Automatic trust scoring |

---

## 4. Integration Points (Boomerang-v3 Orchestrator)

### 4.1 Current Flow → New Flow

**Current flow**:
```
User request → Query memini-ai → Call sequential-thinking_sequentialthinking → Plan → Delegate
```

**New flow**:
```
User request → Query memini-ai → Call memini-ai-dev_start_thought_chain (or add_thought) 
→ Plan → Delegate
```

### 4.2 When to Auto-Start a Thought Chain

The orchestrator currently detects complexity via regex and sets `suggestions.useSequentialThinking`. This logic stays the same – it just changes which tool it tells the LLM to call.

| Trigger | Tool to Call |
|---------|-------------|
| Complex task detected | `memini-ai-dev_start_thought_chain` followed by `memini-ai-dev_add_thought` |
| Planning phase | `memini-ai-dev_add_thought` with existing `chain_id` |
| Before dispatching sub-agent | Pass `chain_id` in ContextPackage |

### 4.3 Chain Hierarchy

```
Session
  └── Orchestrator Chain (parent_chain_id: NULL)
        ├── Architect Sub-Chain (parent_chain_id: orchestrator's chain_id)
        ├── Coder Sub-Chain (parent_chain_id: orchestrator's chain_id)
        └── Tester Sub-Chain (parent_chain_id: orchestrator's chain_id)
```

Each sub-agent gets its own chain linked to the parent. This enables:
- Tracing which agent produced which reasoning
- Independent trust scoring per agent
- Querying `get_related_chains` across all chains in a session

### 4.4 ContextPackage Changes

Add `thinkingChainId` to the orchestrator's `ContextPackage`:

```typescript
interface ContextPackage {
  // ... existing fields
  thinkingChainId?: string;  // UUID of the orchestrator's thought chain
  injectedContext?: {         // From context buffer middleware
    relatedChains: Array<{chain_id: string; snippet: string; score: number}>
  };
}
```

### 4.5 How the Orchestrator Uses Related Past Chains

Before planning, the orchestrator can call `memini-ai-dev_get_related_chains` with the user's request as the query. This returns similar past reasoning, which gets injected into the ContextPackage. The planning LLM then sees "Here's how we solved a similar problem last time."

This is a **new capability** that the old sequential-thinking server could never provide.

### 4.6 Changes Needed in Boomerang-v3 Code

| File | Change |
|------|--------|
| `.opencode/agents/*.md` (15 files) | Remove `"sequential-thinking_*": allow` line (already covered by `memini-ai-dev_*`) |
| `.opencode/agents/boomerang.md` | Change Step 2 instruction to use `memini-ai-dev_add_thought` |
| `.opencode/agents/boomerang-architect.md` | Change instruction to use `memini-ai-dev_add_thought` |
| `.opencode/agents/boomerang-coder.md` | Change instruction to use `memini-ai-dev_add_thought` |
| `.opencode/skills/boomerang-orchestrator/SKILL.md` | Update protocol description and tool names |
| `.opencode/skills/boomerang-architect/SKILL.md` | Update tool references |
| `.opencode/skills/boomerang-coder/SKILL.md` | Update tool references |
| `boomerang-v3/AGENTS.md` | Update 8-step protocol to reflect new tool |
| `boomerang-v3/README.md` | Update MCP server configuration example |
| `boomerang-v3/src/orchestrator.ts` | Add `thinkingChainId` to ContextPackage |
| `boomerang-v3/packages/opencode-plugin/src/orchestrator.ts` | Add `thinkingChainId` to ContextPackage |
| `boomerang-v3/packages/opencode-plugin/src/types.ts` | Add `thinkingChainId` to types |

---

## 5. Trust Engine Integration

### 5.1 How Thoughts Get Scored

Every thought creates a `memories` row with `sourceType="thought"` and `trustScore=0.5` (default). The trust engine automatically handles:

| Event | Trust Adjustment | Applied To |
|-------|-----------------|------------|
| Chain completed successfully | `agent_used` (+0.05) | All thoughts in chain |
| User confirms outcome | `user_confirmed` (+0.10) | Most recent thought in chain |
| Chain abandoned | `agent_ignored` (-0.02) | All thoughts in chain |
| User corrects outcome | `user_corrected` (-0.15) | Chain's "conclusion" thought |
| Agent frequently retrieves chain | `retrieval_count` increments | Each retrieval |

### 5.2 Promotion to L1 Tiered Summary

When a thought chain's **average trust score** exceeds 0.8 (the `PROMOTED` threshold), it becomes eligible for L1 summaries. The L1 summary prompt can now include a "Reasoning Patterns" section:

```
## Reasoning Patterns
- Solved X via Y approach (session S1, confidence: 0.85)
- Avoided dead-end Z in refactoring (session S2, confidence: 0.78)
```

### 5.3 Implementation

A new method on `ThoughtChains` module:

```python
async def _adjust_chain_trust(self, chain_id: str, signal: str) -> None:
    """Apply trust signal to all thoughts in a chain."""
    thoughts = await self._get_chain_thoughts(chain_id)
    for thought in thoughts:
        if thought.memory_id:
            await self.trust_engine.feedback(thought.memory_id, signal)
```

Called automatically when:
- `get_thought_chain` is called (records retrieval)
- `pause_thought_chain` (protects from decay)
- `abandon_thought_chain` (applies negative signal)

---

## 6. Migration / Deprecation Path

### 6.1 Phase 1: Add memini-ai Capability (This Implementation)

- Add `thought_chains` module to memini-ai-dev
- Add new tables via migration script
- Register new MCP tools
- Keep old `sequential-thinking` MCP server running in parallel

### 6.2 Phase 2: Soft Transition

- Add `memini-ai-dev_add_thought` tool permission to all agents (covered by existing `memini-ai-dev_*` wildcard)
- Update agent prompts to use the new tool
- Keep old `sequential-thinking_*: allow` permission for backward compatibility
- Both tools work side-by-side

### 6.3 Phase 3: Hard Cutover

- Remove `"sequential-thinking": {...}` from user's `opencode.json` MCP server config
- Remove `"sequential-thinking_*": allow` from all 15 agent files
- Remove `@modelcontextprotocol/server-sequential-thinking` from any dependencies

### 6.4 Phase 4: Cleanup

- Remove old server dependency from `package.json` if present
- Archive documentation references to old server

### 6.5 Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| `add_thought` called, `thought_chains` table doesn't exist | Return error: `{"error": "thought_chains table not found. Run migration: python -m memini_ai.migrate"}` |
| Old `sequential-thinking` server still configured | Both work simultaneously; agent prompt determines which is used |
| No `THOUGHT_CHAINS` env var set | Feature is disabled; tools return "not enabled" error |

### 6.6 Migration Script

`scripts/migrate_thought_chains.py`:
```python
"""Create thought_chains and thoughts tables with indexes."""
# Uses existing asyncpg connection from config
# Idempotent: uses IF NOT EXISTS
# Run: python scripts/migrate_thought_chains.py
```

Also need to update the memories CHECK constraint:
```sql
-- Add 'thought' to allowed source_type values
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_source_type_check;
ALTER TABLE memories ADD CONSTRAINT memories_source_type_check 
    CHECK (source_type IN ('session', 'file', 'web', 'boomerang', 'project', 'thought'));
```

---

## 7. Effort Estimate

### 7.1 Sprints

| Sprint | Description | Files | Est. LoC | Specialists |
|--------|-------------|-------|----------|-------------|
| **Sprint 1: Core Module** | Schema, config, ThoughtChains class | 5 files | ~650 lines | `boomerang-coder` |
| **Sprint 2: MCP Tools** | 9 new tools in server.py | 1 file (server.py) | ~250 lines | `boomerang-coder` |
| **Sprint 3: Tests** | Unit + integration tests | 1 test file | ~500 lines | `boomerang-tester` |
| **Sprint 4: Boomerang Integration** | Update agent/skill/docs files | ~20 files | ~200 lines | `boomerang-writer` + `boomerang-coder` |
| **Sprint 5: Quality** | Lint, typecheck, review | all changed files | ~50 lines | `boomerang-linter` |

**Total**: ~1,650 lines of new code, ~200 lines of modifications, ~25 files changed, ~4-5 hours with parallel execution.

### 7.2 File-Level Breakdown

#### New Files (memini-ai-dev)

| File | LoC | Description |
|------|-----|-------------|
| `src/memini_ai/thought_chains.py` | ~400 | ThoughtChains class: CRUD, search, trust integration |
| `tests/test_thought_chains.py` | ~500 | Test suite: unit + integration with real PG |
| `scripts/migrate_thought_chains.py` | ~50 | Migration script |
| `docs/DESIGN-thought-chains.md` | ~500 | This document |

#### Modified Files (memini-ai-dev)

| File | Added LoC | Description |
|------|-----------|-------------|
| `src/memini_ai/postgres/schema.py` | ~80 | CREATE TABLE statements for thought_chains, thoughts + indexes |
| `src/memini_ai/postgres/queries.py` | ~150 | SQL query methods for thought chain operations |
| `src/memini_ai/server.py` | ~250 | 9 new MCP tools + tool registration |
| `src/memini_ai/config.py` | ~20 | `THOUGHT_CHAINS` env var + config fields |
| `src/memini_ai/memory/schema.py` | ~10 | Add 'thought' to sourceType Literal |
| `src/memini_ai/memory/system.py` | ~20 | Expose add_memory_by_text() for ThoughtChains |

#### Modified Files (boomerang-v3)

| File | Action |
|------|--------|
| `.opencode/agents/*.md` (15 files) | Remove `sequential-thinking_*` permission line |
| `.opencode/agents/boomerang.md` | Update Step 2 instruction |
| `.opencode/agents/boomerang-architect.md` | Update instruction |
| `.opencode/agents/boomerang-coder.md` | Update instruction |
| `.opencode/skills/boomerang-orchestrator/SKILL.md` | Update protocol |
| `.opencode/skills/boomerang-architect/SKILL.md` | Update reference |
| `.opencode/skills/boomerang-coder/SKILL.md` | Update reference |
| `boomerang-v3/AGENTS.md` | Update protocol |
| `boomerang-v3/README.md` | Update MCP config example |
| `boomerang-v3/src/orchestrator.ts` | Add thinkingChainId |
| `boomerang-v3/packages/opencode-plugin/src/orchestrator.ts` | Add thinkingChainId |
| `boomerang-v3/packages/opencode-plugin/src/types.ts` | Add thinkingChainId to types |

---

## 8. Open Questions for User

These need answers before implementation begins. Please weigh in on each:

### Q1: Scope — Which agents use thought chains?

**Options:**
- A) **Orchestrator only** — only the orchestrator uses thought chains; sub-agents do their own reasoning without tools
- B) **Orchestrator + Architect + Coder** — matches current pattern (these 3 have explicit instructions)
- C) **All agents** — every agent can start and manage their own thought chains

**Recommendation**: **Option B**. This matches current behavior. Architects and coders benefit most from structured reasoning history. Other agents (linter, git, tester) have simpler tasks.

### Q2: Storage — Hybrid or thoughts-only?

**Options:**
- A) **Hybrid** — thoughts stored in BOTH `thoughts` table AND `memories` table (recommended)
- B) **thoughts-only** — dedicated table only, no memory integration
- C) **memories-only** — don't create a `thoughts` table, use memories with metadata to represent chains

**Recommendation**: **Option A (Hybrid)**. The storage overhead is negligible (~1KB per thought) and the benefits (semantic search, trust scoring, tiered summaries, knowledge graph) are enormous. This is the key innovation over the original server.

### Q3: API Design — Compatible or new?

**Options:**
- A) **Compatible** — `add_thought` mimics original `sequentialthinking` with auto-create chains. Extra tools for power users.
- B) **New API** — separate `start_chain` + `add_thought` required. More explicit but breaks all agent prompts.

**Recommendation**: **Option A (Compatible)**. The `add_thought` tool auto-creates chains, so existing prompts only need to change the tool name. Power users get additional tools for advanced use. Least friction migration.

### Q4: Abandoned chains — Preserve or delete?

**Options:**
- A) **Preserve** — mark as `status='abandoned'`, keep all data forever
- B) **Archive** — move to archived state after N days of abandonment
- C) **Delete** — remove abandoned chains after N days (save storage)

**Recommendation**: **Option A (Preserve)**. We never throw away data. The trust engine will naturally demote abandoned chains, and auto-decay will eventually make them low-trust. But we can always query them later.

### Q5: Chain sharing — Per-agent or shared?

**Options:**
- A) **Shared** — orchestrator creates one chain, all sub-agents append to it
- B) **Hierarchical** — orchestrator chain with parent_chain_id links to sub-agent chains
- C) **Per-agent** — each agent gets its own isolated chain, no linking

**Recommendation**: **Option B (Hierarchical)**. This gives clean isolation per agent while maintaining traceability. You can query "show me all reasoning in this session" by following parent_chain_id links, or zoom into just the coder's chain.

### Q6: Feature gate — Opt-in or always-on?

**Options:**
- A) **Opt-in** — `THOUGHT_CHAINS` env var must be set to enable (matching existing pattern)
- B) **Always-on** — feature is always enabled, no env var needed

**Recommendation**: **Option A (Opt-in)**. This follows the established pattern in memini-ai (all features are opt-in via env vars). However, given that this is a core feature, we could set it as enabled-by-default in the shipped config. Let me know your preference.

---

*End of design document. Ready for user review and approval.*
