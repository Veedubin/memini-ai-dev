# Memini-ai Context Document

> **Project**: Memini-ai v0.7.0 (formerly Super-Memory-TS)
> **Meaning**: "I remember" in Latin (pronounced "meh-mee-nee")
> **Goal**: Local-first semantic memory server, MCP-compatible, Python-based
> **Last Updated**: 2026-06-02 (v0.7.0 DUAL-MODEL RRF RELEASED ✅ — commit `18f37ed`, tag `v0.7.0` pushed. 15/15 steps done, 763 tests passing, 83 memories intact.)

---

## Project Origin

User is building Memini-ai v3.0 as a rewrite of Super-Memory-TS (TypeScript) to Python. Reasons:
1. User prefers Python over TypeScript
2. "Never seemed to work as well as it did in V1 in Python"
3. Wants to rename to avoid confusion with other "super-memory" products
4. Will use FastMCP if possible (instead of @modelcontextprotocol/sdk)

---

## Key Architectural Decisions

### Keep from Super-Memory-TS
- Qdrant backend (HNSW indexing)
- BGE-Large embeddings (1024-dim) - "the 1024 dim model path"
- MiniLM-L6-v2 (384-dim) as CPU fallback
- MCP server with same 5 tools
- Sub-10ms query latency
- Local-first, privacy-preserving

### Add: Memory Kernel v3.0
Based on Reality Check analysis, build these 3 features FIRST:

1. **Auto-Extract**: After every N turns (default 5), fire LLM pass to extract facts/decisions/preferences automatically
2. **Memory Graph**: Track relationships (SUPERSEDES, RELATED_TO, CONTRADICTS, DERIVED_FROM) via JSON field in Qdrant payload
3. **Trust Engine**: Every memory starts at trust=0.5, adjust based on agent behavior feedback

### Tier 2 (add later)
- Pre-compression extraction (capture before context squeeze)
- Tiered loading L0/L1/L2 (token-efficient context)
- User modeling (Honcho-style)

---

## The Three Problems We're Solving

1. **"I have to manually remember to save memories"** → Auto-Extract
2. **"My memories are just a pile of vectors with no structure"** → Memory Graph
3. **"Bad memories poison my context"** → Trust Engine

---

## Six Capability Layers (Long-term Vision)

| Layer | Description | Inspired By |
|-------|-------------|-------------|
| 1. Foundation | Qdrant/HNSW, BGE-Large, 5 MCP tools | Super-Memory-TS |
| 2. Extraction | Auto LLM-based extraction | Mem0, ByteRover |
| 3. Knowledge Graph | Lightweight relationships | Hindsight, Supermemory |
| 4. User Modeling | Dialectic peer profiles | Honcho |
| 5. Trust & Decay | Asymmetric scoring, contradiction detection | Holographic, RetainDB |
| 6. Tiered Loading | L0/L1/L2 abstraction levels | OpenViking |

**All layers optional and independently enableable.**

---

## Current Status: Phase 1 Complete ✅

**Implemented**: Foundation phase of Memini-ai v3.0 (Python rewrite of Super-Memory-TS)

### What's Implemented
- Full Python port of Super-Memory-TS v2.6.5 architecture
- 23 source modules in `src/memini_ai/`
- 201 passing tests, 10 skipped (integration tests)
- 6 MCP tools (including get_status bonus)
- Async-first with asyncio.to_thread()

### What's Next
**Phase 2: Memory Kernel Features** (Auto-Extract, Memory Graph, Trust Engine)

---

## Super-Memory-TS v2.6.5 Architecture (to port)

### Directory Structure
```
src/
├── memory/
│   ├── index.ts          # MemorySystem class
│   ├── schema.ts         # MemoryEntry, SearchOptions types
│   ├── database.ts       # Qdrant operations
│   └── search.ts         # TIERED/VECTOR_ONLY/TEXT_ONLY/PARALLEL
├── model/
│   ├── index.ts
│   ├── embeddings.ts     # generateEmbedding(), generateEmbeddings()
│   └── types.ts
├── project-index/
│   ├── index.ts
│   ├── indexer.ts        # File indexing with SHA-256
│   ├── chunker.ts        # Semantic chunking
│   ├── watcher.ts        # File watching (chokidar)
│   ├── file-tracker.ts
│   ├── snapshot.ts
│   ├── pause-controller.ts
│   └── constants.ts
├── server.ts             # MCP server
├── config.ts            # Environment + JSON config
└── utils/
    ├── logger.ts
    ├── hash.ts          # SHA-256
    └── errors.ts
```

### MemoryEntry Schema
```typescript
interface MemoryEntry {
  id: string;              // UUID
  text: string;            // Content
  vector: Float32Array;    // 1024-dim embedding
  sourceType: MemorySourceType;  // session, file, web, boomerang, project
  sourcePath?: string;
  timestamp: Date;
  contentHash: string;     // SHA-256 for deduplication
  metadataJson?: string;
  sessionId?: string;
  projectId?: string;
  score?: number;          // Qdrant similarity score
}
```

### 6 MCP Tools (Implemented)
1. `query_memories` - Semantic search over memories
2. `add_memory` - Store a new memory entry
3. `search_project` - Search indexed project files
4. `index_project` - Trigger project indexing
5. `get_file_contents` - Reconstruct file from indexed chunks
6. `get_status` - Component readiness status (bonus)

### Search Strategies
- **TIERED** (default): MiniLM primary + BGE fallback, threshold 0.72
- **PARALLEL**: Dual-tier with RRF fusion
- **VECTOR_ONLY**: Pure semantic similarity
- **TEXT_ONLY**: Fuse.js keyword matching

### Dependencies to Port
| TS | Python |
|----|--------|
| @modelcontextprotocol/sdk | fastmcp |
| @qdrant/js-client-rest | qdrant-client |
| @xenova/transformers | transformers + torch |
| chokidar | watchdog |
| fuse.js | rank_bm25 / Whoosh |
| xxhash-wasm | xxhash |
| zod | pydantic |

---

## Key Insights from Research

### From memory_providers_dim01.md (Hermes Comparison)
- 8 providers: Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory
- Hermes uses Python ABC protocol, NOT MCP-compatible directly
- Each provider isolation via $HERMES_HOME/ paths

### From memory_providers_dim02.md (Vendor Profiles)
- **Hindsight**: Best benchmarks (91.4-94.6% LongMemEval), TEMPR 4-strategy retrieval
- **Honcho**: Only provider built for user modeling, dialectic reasoning
- **OpenViking**: 80-90% token reduction via L0/L1/L2 tiered loading
- **Holographic**: Zero dependencies, HRR algebra, trust scoring (+0.05/-0.10)
- **Mem0**: Server-side LLM extraction, Apache 2.0, 51.4K stars

### From memory_providers_dim03.md (Super-Memory-TS Analysis)
Current gaps (12 total):
1. No automatic memory extraction
2. No knowledge graph
3. No user modeling
4. No memory synthesis
5. No tiered context loading
6. No trust scoring
7. No contradiction detection
8. No temporal awareness
9. No memory decay/consolidation
10. No context fencing
11. No pre-compression hooks
12. No multi-modal memory types

### From memory_providers_dim04.md (OpenCode Compatibility)
- Super-Memory-TS IS MCP-compatible (runs as stdio/HTTP MCP server)
- Hermes protocol is Python-only ABC, needs MCP bridge for OpenCode
- Mem0, Hindsight, Honcho all have MCP bridges

### From memory_providers_insight.md (Recommendations)
Recommended vNext architecture: 6-layer hybrid
- Keep MCP, local-first, Qdrant/HNSW
- Add auto-extraction (Mem0/ByteRover style)
- Add lightweight knowledge graph
- Add trust scoring + contradiction detection
- Add tiered loading L0/L1/L2
- All optional, backward compatible

### From reality_check.md (Build Priorities)
**DO NOT build all 6 layers at once!**

Build order:
1. **Memory Kernel** (4-6 weeks, ~2500 lines):
   - Auto-Extract
   - Memory Graph
   - Trust Engine
2. **Tier 2** (2-3 months each, after kernel stable):
   - Pre-Compression Extraction
   - Tiered Loading L0/L1/L2
3. **Tier 3** (maybe never):
   - Full KG, memory decay, multi-peer, dialectic

Every feature MUST be independently optional - can be disabled.

---

## MCP Command Reference

User mentioned "1024 dim model path" but couldn't remember the MCP command. The relevant commands for super-memory-ts were:
- `query_memories` - search memories
- `add_memory` - store memory
- `search_project` - search indexed code
- `index_project` - trigger indexing
- `get_file_contents` - reconstruct files

For Memini-ai, we may add:
- `get_trust_score` - get memory trust
- `adjust_trust` - update trust based on feedback
- `list_archived` - list archived low-trust memories

---

## Version History

| Version | Language | Notes |
|---------|----------|-------|
| **v0.7.0** | **Python** | **Dual-model RRF RELEASED (2026-06-02). 384+1024 tables, MEMINI_MODE routing (cpu/auto/gpu), RRF k=60, `elevate_memory_to_1024` MCP tool (auto-mode gated). Commit `18f37ed`, tag `v0.7.0` pushed. 763 tests passing, 83 memories preserved. 36 MCP tools total.** |
| v0.6.0 | Python | Modular cloud LLM (factory/provider pattern), 740/740 tests, tag `v0.6.0` pushed |
| v1.0 | Python | Original implementation (user prefers this) |
| v2.0 | TypeScript | Super-Memory-TS (current npm package) |
| v3.0 | Python | Memini-ai (rewriting v2.0 in Python) |

---

## Session 2026-06-02 (Session 4) — v0.7.0 Implementation 5/15 Done

### Pre-Implementation State Verified
- **83 memories** in `memories` table (was 80 in Session 3, +3 from this session's testing) — all 384-dim, all active
- **`memories_1024` table EXISTS but is EMPTY** (FK to memories.id, vector(1024), 4 indexes)
- Schema: `vector(384)` on `memories.embedding`, intact
- Working tree dirty on: `config.py` (validators), `schema.py` (table), `queries.py` (6 new constants), `database.py` (6 new methods + helper), NEW `memory/rrf.py`

### v0.7.0 Implementation (RELEASED 2026-06-02, commit `18f37ed`)

**All 15 implementation steps done in 2 sessions (4 + 5).**

| # | Step | File | Status |
|---|------|------|--------|
| 1 | `config.py` 3 validators | `src/memini_ai/config.py` | DONE |
| 2 | `memories_1024` table + indexes | `src/memini_ai/postgres/schema.py` | DONE (applied to live DB) |
| 3 | 6 new 1024 query constants | `src/memini_ai/postgres/queries.py` | DONE |
| 4 | `memory/rrf.py` NEW | `src/memini_ai/memory/rrf.py` | DONE (created) |
| 5 | 6 new DB methods + `_expand_384_to_1024` | `src/memini_ai/postgres/database.py` | DONE |
| 6 | `memory/system.py` cpu/auto/gpu dispatch + delete dead `_get_fallback_for_dimension()` | `src/memini_ai/memory/system.py` | DONE |
| 7 | `server.py` `elevate_memory_to_1024` MCP tool (auto-mode gated) | `src/memini_ai/server.py` | DONE |
| 8 | 3 new test files (23 tests) + 1 test_config fix | `tests/test_rrf.py`, `tests/test_dual_model.py`, `tests/test_schema_migration.py`, `tests/test_config.py` | DONE |
| 9 | `.env.example` 5 new env vars | `.env.example` | DONE |
| 10 | `.opencode/opencode.json` `EMBEDDING_MODE=auto` | `.opencode/opencode.json` (root) | DONE |
| 11 | Quality gates: ruff, mypy, pytest (763 passing) | — | DONE |
| 12 | Zero-data-loss verification: `SELECT COUNT(*) FROM memories = 83` before/after | — | DONE |
| 13 | `pyproject.toml` 0.6.0 → 0.7.0 | `pyproject.toml` | DONE |
| 14 | Commit + tag `v0.7.0` + push to GitHub | — | DONE |
| 15 | Update docs: AGENTS.md, CONTEXT.md, TASKS.md, HANDOFF.md, CHANGELOG.md, README.md | — | DONE (this file) |

### CRITICAL: Agent-Blocker Fix Applied (Restart Required)

**The HANDOFF's "tag-sweep complete" claim was inaccurate.** Session 3 reported 246 active agent files clean, but missed the **project-level `node_modules/@veedubin/boomerang-v3` install** (15 stale files), which is what OpenCode was actually loading. Also missed 3 files in `boomerang-v3/.opencode/agents/` and several in `neuralgentics`, `Super-Memory`, `boomerang`, and the npm cache.

**Session 4 fixed 47+ files across 6 locations**, including the project npm install that was the actual culprit. All model names verified against Ollama Cloud API (10/10 valid).

**⚠️ OpenCode restart STILL REQUIRED.** The running TUI (PID 307190) has old config cached in process memory. The user must exit and restart OpenCode for the fix to take effect. PID 307190 cannot be killed from inside this session (it owns this very session).

**Saved to memini-ai memory:** `b8b42742-e4e1-4a2a-a1a1-afd85e597f59`

### Process State (Awareness for Next Session)
- **PID 307190** — OpenCode TUI (this session's parent). Will need to be killed+restarted by user.
- **PID 250631** — Different OpenCode TUI in `/home/jcharles/Projects/reverse_engineering` (irrelevant to this project).
- **PostgreSQL on port 5434** — running, healthy, 83 memories at 384-dim, `memories_1024` table exists and is empty.
- **Ollama Cloud** — API key verified working, 40 models available.

### Resume Commands
```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev
# Verify state
git status -s
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories"  # expect 83
# Run quality gates
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest tests/ -v
# Final commit + tag (after steps 6-15 done)
git add -A && git commit -m "Release v0.7.0: Dual-model RRF"
git tag v0.7.0 -m "v0.7.0"
git push origin main && git push origin v0.7.0
```

---

### Pre-Implementation State Verified
- **80 memories** in `memories` table (all 384-dim, all active) — **NOT 76 as previous handoff said** (4 new since then)
- Schema: `vector(384)` on `memories.embedding`, intact
- Current `embedding_dim: int = 1024` default is WRONG (config bug) — needs 384
- Working tree: `config.py` partially modified (fields added, validators missing)

### Ollama Cloud API Key (USER-PROVIDED, BURN-OK)
```
LLM_PROVIDER=ollama-cloud
LLM_API_KEY=YOUR_OLLAMA_CLOUD_API_KEY
LLM_BASE_URL=https://ollama.com/v1
LLM_MODEL=ministral-3:8b   (small model for tests)
```
Verified working: `/api/tags` lists 40+ models, `/v1/chat/completions` responds. **User explicitly said: "if we need to burn it, we can."**

### v0.7.0 Design (Complete, see `docs/design/dual-model-rrf-architecture.md`)

**Two-table schema (additive, zero data loss):**
```
memories          (existing, 384-dim, 80 rows — UNTOUCHED)
memories_1024     (new, 1024-dim, FK memory_id → memories.id)
```

**Three operating modes via `EMBEDDING_MODE` env:**
| Mode | Write | Query | Use case |
|------|-------|-------|----------|
| `cpu` | 384 only | 384 only | Low-memory/legacy |
| `auto` (default) | 384 (and optional 1024 via elevate) | 384 + 1024, RRF fused | Production |
| `gpu` | 1024 only | 1024 only | High-fidelity |

**RRF k=60** fuses 384 and 1024 ranked lists. Same memory appearing in both naturally boosts.

**`elevate_memory_to_1024` MCP tool** (AUTO mode only): promotes 384-dim memory to 1024-dim, re-embeds with BGE-Large (placeholder expansion in v0.7.0), trust +0.10, idempotent (no-op if exists).

### Tag-Sweep Achievement
Across the **entire `/home/jcharles`** (not just MCP-Servers):
- 246 active agent `.md` files: 0 dirty
- 24 active `opencode.json`/`.jsonc`: 0 dirty  
- 139 files fixed in 2 sed passes (139 first, then 3 with multi-colon model names like `devstral-2:123b-cloud`)
- `qwen3.5` mapped to `qwen3.5:397b` (only `qwen3.5*` model in current API)
- Intentionally untouched: anti-pattern docs, upstream `opencode-base`, runtime state cache, session diffs, `dot-config-old`

---

## Important Files to Reference

1. Super-Memory-TS source: `/node_modules/@veedubin/super-memory-ts/dist/`
2. Research docs: `memini-ai-dev/docs/research/`
3. Memory Report: `memini-ai-dev/docs/memory_report.agent.final.md`
4. Reality Check: `memini-ai-dev/docs/reality_check.md`
5. This file: `memini-ai-dev/CONTEXT.md`
6. Tasks: `memini-ai-dev/TASKS.md`
7. **Dual-Model RRF Design**: `memini-ai-dev/docs/design/dual-model-rrf-architecture.md`
8. **Previous handoff**: `memini-ai-dev/HANDOFF.md`
