# Dual-Model RRF Memory Architecture

**Date:** 2026-06-01
**Status:** DESIGN — Ready for implementation
**Verified data state:** 76 memories, all 384-dim, intact

## Problem

User wants BOTH 384-dim MiniLM (CPU, fast) AND 1024-dim BGE-Large (GPU, high quality) in the same memory system, with RRF fusion over both. The system must default to 384 for cost/speed, with explicit promotion to 1024 for important memories.

## Current Bugs Being Fixed

1. `config.py` defaults `embedding_dim=1024` but schema is `vector(384)`. → Mismatch
2. `get_collection_dimension()` was hardcoded to 1024 → Fixed in June 1 patch
3. `memory/system.py` has dual-dim cascade stubs pointing to non-existent `memories_384`/`memories_1024` tables → Phantom code
4. No actual 1024 storage path → RRF cannot work

## Solution: Two-Table Layout

### Schema (chosen option A: separate tables, FK-linked)

```sql
-- Existing 384-dim table — UNCHANGED. 76 existing memories preserved as-is.
memories (
  id UUID PRIMARY KEY,
  text TEXT,
  embedding vector(384),     -- 384-dim only, NULL for 1024-only elevated memories
  source_type, trust_score, metadata, etc.   -- existing columns preserved
)

-- NEW: 1024-dim elevated memories
memories_1024 (
  id UUID PRIMARY KEY,
  memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,  -- FK to 384 row
  text TEXT NOT NULL,
  embedding vector(1024),     -- BGE-Large 1024-dim
  source_type VARCHAR(50) NOT NULL,  -- mirrors memories.source_type
  trust_score FLOAT DEFAULT 0.5,
  metadata JSONB DEFAULT '{}',
  peer_id UUID,
  content_hash VARCHAR(64) UNIQUE,
  elevated_at TIMESTAMPTZ DEFAULT NOW(),
  elevated_by VARCHAR(100) DEFAULT 'auto',  -- 'auto' | 'user' | 'tool:elevate_memory_to_1024'
  created_at TIMESTAMPTZ DEFAULT NOW()
)

CREATE INDEX idx_memories_1024_embedding ON memories_1024 USING diskann (embedding vector_cosine_ops);
CREATE INDEX idx_memories_1024_memory_id ON memories_1024(memory_id);
CREATE INDEX idx_memories_1024_content_hash ON memories_1024(content_hash);
```

### Why option A (not B/C)

- **A preserves data:** Existing 76 384-dim memories are untouched. `ALTER TABLE ADD COLUMN` for option B is riskier on a populated table.
- **A is faster:** No NULL checks in hot path, vector index is single-dim
- **A is portable:** Works with any future dimension addition (just add another table)
- **A keeps RRF clean:** Two homogeneous queries UNION ALL'd + RRF fused

## MEMINI_MODE Routing

### Env var: `EMBEDDING_MODE` (alias) / `MEMINI_MODE` (prefix)

| Mode | add_memory | query_memories | elevate tool | Background extractor |
|------|-----------|----------------|--------------|---------------------|
| `cpu` | 384 only (MiniLM CPU) | 384 only | DISABLED (hidden from MCP) | DISABLED (skipped) |
| `auto` | 384 default (MiniLM) | 384 + 1024 RRF | ENABLED | ENABLED |
| `gpu` | 1024 only (BGE-Large GPU) | 1024 only | DISABLED (already 1024) | DISABLED |

Default: `auto` (matches user's stated preference — "leave it on auto").

### Config (config.py additions)

```python
embedding_mode: str = Field(default="auto", alias="EMBEDDING_MODE")  # cpu|auto|gpu
elevate_enabled: bool = Field(default=True, alias="ELEVATE_ENABLED")  # AUTO only
```

### `add_memory` dispatch

```python
async def add_memory(self, entry: MemoryEntry) -> str:
    if self._mode == "cpu":
        # 384 only
        await self._insert_memories_384(entry)
    elif self._mode == "gpu":
        # 1024 only — embedding must be 1024 or re-embed
        await self._insert_memories_1024(entry)
    else:  # auto
        # 384 default, marked elevatable
        await self._insert_memories_384(entry, elevatable=True)
    return memory_id
```

### `query_memories` dispatch with RRF

```python
async def query_memories(self, q: str, k: int = 10) -> list[MemoryEntry]:
    if self._mode == "cpu":
        return await self._query_384_only(q, k)
    elif self._mode == "gpu":
        return await self._query_1024_only(q, k)
    else:  # auto: dual RRF
        results_384 = await self._query_384_only(q, k * 2)  # 2x for fusion
        results_1024 = await self._query_1024_only(q, k * 2)
        fused = reciprocal_rank_fusion(results_384, results_1024, k=60)
        return fused[:k]
```

### RRF Algorithm (k=60)

```python
def reciprocal_rank_fusion(list_a, list_b, k_const=60):
    """Fuse two ranked lists. k=60 is the standard RRF constant."""
    scores = {}
    for rank, item in enumerate(list_a, start=1):
        scores[item.id] = scores.get(item.id, 0) + 1 / (k_const + rank)
    for rank, item in enumerate(list_b, start=1):
        scores[item.id] = scores.get(item.id, 0) + 1 / (k_const + rank)
    # Sort by fused score, return MemoryEntry objects
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [lookup(i) for i in sorted_ids]
```

When the same memory_id exists in BOTH tables (an elevated memory), RRF naturally boosts it.

## elevate_memory_to_1024 Tool (AUTO only)

### MCP signature
```python
@mcp.tool(name="elevate_memory_to_1024")
async def elevate(memory_id: str) -> str:
    """Promote a 384-dim memory to 1024-dim for higher fidelity. AUTO mode only."""
```

### Implementation
1. Read memory by `id` from `memories` table (verify it has 384 embedding)
2. Re-embed `text` with BGE-Large 1024-dim (GPU if available, else CPU slower)
3. Check `memories_1024` for existing row with same `memory_id` (no-op if exists)
4. INSERT INTO `memories_1024` with `elevated_by='tool:elevate_memory_to_1024'`
5. Boost trust_score by +0.1 in BOTH tables (slight promotion bump)
6. Return `{memory_id, elevated: true, elevation_id, model: 'BGE-Large'}`

### Gating
Tool is registered only when `embedding_mode == 'auto'`. In `cpu` mode, the tool is omitted from the MCP server's tool list (not just erroring out). In `gpu` mode, it's also omitted (everything is already 1024).

## Background Chat-Log Extractor

### Model
- **CPU:** all-MiniLM-L6-v2 (384-dim, already loaded, 1-2K token context fine)
- **Trigger:** Chat logs appended to `~/.memini-ai/chat_logs/*.jsonl` (file watcher)
- **Chunking:** Sliding window of 1.5K tokens, 200-token overlap, "finish that paragraph" boundary detection
- **Extraction:** Per chunk, prompt the model (or use zero-shot NER) to extract:
  - Entities (PERSON, ORG, CONCEPT, CODE, PROJECT, LOCATION — 6 types from KG)
  - Context (decisions, action items, open questions)
  - Per-extraction: `{chunk_id, entity, context, source_chunk_hash}`
- **Storage:** Insert as `MemoryEntry` with `source_type='auto_extracted'`, `trust_score=0.3` (low — auto)
- **Cadence:** Async worker, 1 chunk per 5 seconds, runs only when `embedding_mode != 'cpu'`

### Why MiniLM for background extraction
- Already in memory, no model swap cost
- 384-dim is fine for entity/concept embeddings (not storing the full text, just refs)
- "We have models small enough on CPU that could read in 1-2K tokens" — MiniLM qualifies
- Background extraction is offline enrichment, not query-path critical

## Trust Score Weighting

| Source | Trust Score | Reasoning |
|--------|-------------|-----------|
| User-uploaded memory | 0.5 (default) + boost | Manual creation = high signal |
| Thought chain entry | 0.5 + boost | Reasoning, contextual |
| `user_confirmed` signal | +0.10 boost | Explicit validation |
| Agent-written memory | 0.5 (default) | Standard |
| Elevated 1024 memory | +0.10 trust + 1024 storage | Promoted = important |
| Auto-extracted from chat log | 0.3 (low) | Bulk, not curated |
| `agent_ignored` signal | -0.05 penalty | Retrieved but unused |
| `user_corrected` signal | -0.10 penalty | User pushed back |

RRF uses position rank, not raw trust, so the trust score filters and sorts after fusion. Order of operations in `query_memories`:
1. Search 384 table → top N
2. Search 1024 table → top N (if mode=auto or gpu)
3. RRF fuse → ranked list of (memory_id, fused_score)
4. Apply trust filter (≥ 0.3) and sort by `fused_score * trust_score`
5. Return top k

## Migration Plan (zero data loss)

```sql
-- Step 1: Create new 1024 table (safe, no ALTER on existing)
CREATE TABLE IF NOT EXISTS memories_1024 (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  embedding vector(1024),
  source_type VARCHAR(50) NOT NULL,
  trust_score FLOAT DEFAULT 0.5,
  metadata JSONB DEFAULT '{}',
  peer_id UUID,
  content_hash VARCHAR(64) UNIQUE,
  elevated_at TIMESTAMPTZ DEFAULT NOW(),
  elevated_by VARCHAR(100) DEFAULT 'auto',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 2: Indexes (IF NOT EXISTS for safety)
CREATE INDEX IF NOT EXISTS idx_memories_1024_embedding ON memories_1024 USING diskann (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memories_1024_memory_id ON memories_1024(memory_id);
CREATE INDEX IF NOT EXISTS idx_memories_1024_content_hash ON memories_1024(content_hash);

-- Step 3: Existing 76 384-dim memories: UNCHANGED
-- (no migration needed, no backfill)
```

## Quality Gates

### New tests (pytest)

```python
# tests/test_dual_model_mode.py
def test_cpu_mode_writes_384_only()
def test_cpu_mode_hides_elevate_tool()
def test_cpu_mode_skips_background_extractor()
def test_auto_mode_default_384_writes()
def test_auto_mode_elevate_tool_registered()
def test_auto_mode_query_uses_rrf()
def test_gpu_mode_writes_1024_only()
def test_gpu_mode_hides_elevate_tool()

# tests/test_rrf.py
def test_rrf_fuses_two_lists()
def test_rrf_boosts_elevated_memories()
def test_rrf_k_constant_default_60()
def test_rrf_empty_lists()

# tests/test_migration.py
def test_existing_76_memories_still_queryable()
def test_memories_1024_table_created()
def test_elevate_inserts_into_1024_table()
def test_elevate_does_not_duplicate_existing_1024()
```

### Quality gate commands
```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev
ruff check src/ tests/  # 0 errors
mypy src/  # 0 errors
pytest tests/ -v  # all pass, including new tests
```

## Implementation Order

1. **Config** — Add `embedding_mode` field, fix default `embedding_dim` to 384
2. **Migration** — Create `memories_1024` table + indexes (additive, safe)
3. **Storage** — `add_memory` mode dispatch in `system.py`
4. **Query** — `query_memories` mode dispatch with RRF
5. **Tool** — `elevate_memory_to_1024` MCP tool with mode gating
6. **Background** — Async chat-log extractor worker (small CPU model)
7. **Tests** — All quality gates pass
8. **Verification** — Direct DB query confirms 76 existing memories still present
