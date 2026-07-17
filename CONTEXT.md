# Memini-ai Context Document

> **Project**: Memini-ai v1.0.3 (formerly Super-Memory-TS)
> **Meaning**: "I remember" in Latin (pronounced "meh-mee-nee")
> **Goal**: Local-first semantic memory server, MCP-compatible, Python-based
> **Last Updated**: 2026-07-16 (Session 52). 10 versions released since v0.7.6: **v0.7.7** (BGE-M3 opt-in + `MEMINI_AUTO_DETECT_MODEL` + `MEMINI_STRICT_EMBEDDING_DIM` + `docs/upgrading-embeddings.md`), **v0.7.8** (comprehensive 8-area audit by boomerang-architect + boomerang-tester; 13 doc fixes including CRITICAL README `MEMINI_EMBEDDING_DIM=1024` missing; BM25 punctuation guard; step-limit 10x bump across 61 agent files; full reports at `docs/audits/v0.7.7-{audit,validation}.md`), **v0.7.9** (data-leak rule formalization in AGENTS.md after v0.7.8 19MB pg_dump near-miss), **v0.8.0** (image-recall RRF fan-out arm via `memini-vision` CLIP text tower over new `memories_image` table; backward-compatible zero-behavior-change for text-only users; design doc `docs/design/vision-memory-architecture.md` 30KB), **v0.8.1** (CI re-trigger; no code changes), **v0.8.2** (SECURITY: `detect-secrets` baseline + pre-commit hook + CI scan; came same day as Session 47 Ollama Cloud API key leak in 3 public repos), **v1.0.0 MAJOR** (embedded `pgembed` (in-process Postgres 17 + pgvector + vectorscale + pg_textsearch) as default backend, multi-process server sharing, RRF fusion across embedded + team via `RRFDatabase` wrapper, CLI `memini-ai {init,status,stop,migrate}`, 4 new env vars, design doc `docs/design/v1.0.0-embedded-pgembed-architecture.md` 76KB; **breaking: `MEMINI_VECTOR_BACKEND` must be set if `MEMINI_DB_URL` is set**), **v1.0.1 + v1.0.2** (6 bugs in `memini-ai migrate` fixed in both the standalone script and the CLI command), **v1.0.3** (docs catchup patch: HANDOFF/AGENTS/TASKS/CONTEXT refreshed to v1.0.2 reality + new CRITICAL `MEMINI_VECTOR_BACKEND` section in AGENTS.md + `uv.lock` regenerated to match `pyproject.toml`; no code changes). **DB server verified working 2026-07-16** (in-process `MCPServer.healthcheck()`: `status=pass, readbackMatch=True`; live `memini-postgres` on port 5434: 986 memories + 519 thoughts, all 13 tables present, 100% healthy). 809 tests pass (v0.7.8 baseline). ruff/mypy clean. See `CHANGELOG.md` for per-release detail and `HANDOFF.md` for session-by-session records.

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
| **v1.0.2** | **Python** | **migrate CLI fix (6 bugs). 809 tests pass, ruff/mypy clean. Commit `b050806`, tag `v1.0.2` pushed.** |
| **v1.0.1** | **Python** | **migrate script fix (6 bugs). 809 tests pass, ruff/mypy clean. Commit `63cfb8a`, tag `v1.0.1` pushed.** |
| **v1.0.0** | **Python** | **MAJOR: Embedded pgembed backend (default), Python 3.12+, 13 tables. 809 tests pass, ruff/mypy clean. Commit `74b81cf`, tag `v1.0.0` pushed.** |
| **v0.8.2** | **Python** | **SECURITY: detect-secrets baseline + CI scan. 812+13 tests pass, ruff/mypy clean. Commit `ed7e3ba`, tag `v0.8.2` pushed.** |
| **v0.8.1** | **Python** | **CI re-trigger for `memini-vision` dependency (no code changes). Commit `241e471`, tag `v0.8.1` pushed.** |
| **v0.8.0** | **Python** | **Image-Recall RRF fan-out arm, CLIP text tower, `memories_image` table. 799 tests pass, ruff/mypy clean. Commit `25eb3aa`, tag `v0.8.0` pushed.** |
| **v0.7.9** | **Python** | **Data-leak rule followup, `.gitignore` updates, 809 tests pass. Commit `2c71c2a`, tag `v0.7.9` pushed.** |
| **v0.7.8** | **Python** | **Audit-driven doc rewrite (13 fixes), 809 tests pass. Commit `9408a87`, tag `v0.7.8` pushed.** |
| **v0.7.7** | **Python** | **BGE-M3 opt-in, 2 new env vars, 807 tests pass. Commit `fa8223e`, tag `v0.7.7` pushed.** |
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

---

## v0.7.5 + v0.7.6 (Session 39 + 40) — Multi-Model RRF bugfixes + BGE-Large removal

### v0.7.5 (Session 39) — Multi-Model RRF bugfix
- 3 latent bugs in v0.7.0's multi-model feature fixed:
  1. `ModelManager._load_model()` was constrained by `embedding_dim` instead of `config.model_name` — BGE-M3 unreachable as active model
  2. `add_memory` wrote 1024-dim vectors to the 384-dim `embedding` column — silent data loss for BGE-M3/BGE-Large writes
  3. RRF `COLUMN_TO_MODEL` used short name `'all-MiniLM-L6-v2'` but `ModelManager` expects full HF name — would crash on MiniLM
- Fixes: model_name-driven selection with alias support, multi-model column routing, full-HF-name RRF column mapping
- 824 tests passing (was 763 in v0.7.0, +47 new tests across all multi-model fixes)
- Live DB verification: BGE-M3 with `MEMINI_MODEL_NAME=BAAI/bge-m3` loads, produces 1024-dim vectors, writes to `embedding_bge_m3` column
- All 3 model spaces (MiniLM, BGE-M3, BGE-Large) populated for ~800 memories in `memini-postgres` (port 5434)
- RRF search returns results from all 3 spaces

### v0.7.6 (Session 40) — BGE-Large removal
- BGE-Large (`BAAI/bge-large-en-v1.5`) support removed. Was added in v0.7.0 but not used in production.
- `embedding_bge_large vector(1024)` column dropped from `memories` table (migration 000007)
- `BGE_LARGE_MODEL_ID` / `BGE_LARGE_DIM` constants removed
- `INSERT_MEMORY_BGE_LARGE` / `SEARCH_MEMORIES_BGE_LARGE` query constants removed
- `embedding_bge_large` removed from `COLUMN_TO_MODEL` / `MODEL_TO_DIM` / `enabled_models`
- 40 BGE-Large tests removed (was 824, now 784)
- BGE-Large migration script kept as reference: `archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py`
- The supported models are now exactly two:
  - **MiniLM-L6-v2** (384-dim, default) — fast, small, CPU-friendly
  - **BGE-M3** (1024-dim, optional GPU upgrade) — higher precision, GPU-friendly
- Live DB state: 821 memories, 819 with MiniLM, 800 with BGE-M3, 0 with BGE-Large (column dropped)
- The canonical migration story is now: **MiniLM → BGE-M3** for users who get a GPU

### The "GPU upgrade path" (MiniLM → BGE-M3)
1. Set `MEMINI_MODEL_NAME=BAAI/bge-m3` in `.env`
2. Install `sentence-transformers` with the `[gpu]` extra (`uv pip install sentence-transformers[gpu]`)
3. Run migration script: `python archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py`
4. Verify: `SELECT COUNT(*) FROM memories WHERE embedding_bge_m3 IS NOT NULL;`
5. Set `MEMINI_ENABLE_RRF=true` for RRF search across both columns

The MiniLM column is never touched — both vectors coexist for RRF.

## v0.7.7 + v0.7.8 + v0.7.9

### v0.7.7: BGE-M3 Opt-In (2026-07-10)
- **2 new env vars**: `MEMINI_AUTO_DETECT_MODEL` (default `true`; new deployments with 0 memories auto-upgrade to BGE-M3 1024-dim; existing users keep MiniLM) and `MEMINI_STRICT_EMBEDDING_DIM` (default `false`; dim mismatch logs WARNING + degrades to text-only instead of raising RuntimeError).
- **Fixed BM25 `text_only_search` empty-corpus `ZeroDivisionError`** (3 guards in `_build_bm25_index`, `text_only_search`, `text_search_collection`).
- **Fixed `get_sentence_embedding_dimension` deprecation warning** → `get_embedding_dimension` (sentence-transformers 3.x).
- **Fixed 4 pre-existing test failures** via `autouse=True` `_isolate_env` fixture.
- **`get_status` now reports**: `modelName`, `modelDimension`, `embeddingDimMismatch`, `embeddingDimExpected`, `embeddingDimActual`.
- **New `docs/upgrading-embeddings.md`**: 4-step migration recipe + rollback + FAQ.
- **807 tests pass** (+23 net new), ruff/mypy clean.
- **OpenCode restart required** to load v0.7.7 code.
- **Commit**: `fa8223e`, **tag**: `v0.7.7`.

### v0.7.8: Audit-Driven Doc Rewrite (2026-07-10)
- **8-area audit by boomerang-architect + boomerang-tester**: 13 findings (1 CRITICAL, 4 HIGH, 6 MEDIUM, 2 LOW). All 13 fixed.
- **CRITICAL fix**: README "Enabling Multi-Model" example was missing `MEMINI_EMBEDDING_DIM=1024`, would have silently degraded to text-only.
- **HIGH fixes**: `.env.example` missing 6 v0.7.7 env vars, README env var table missing same, `upgrading-embeddings.md` referenced non-existent `sentence-transformers[gpu]` pip extra, migration script path was wrong.
- **2 minor code fixes**: BM25 punctuation-only query guard, `get_sentence_embedding_dimension` deprecation in migration script.
- **Process fix**: bumped `steps: N` frontmatter 10x across all 61 agent `.md` files (50→500, 40→400, 30→300) — sub-agents hitting 50-step limits.
- **809 tests pass**, ruff/mypy clean. 6 files changed, 1 directory moved (archives/ into memini-ai-dev/).
- **Full reports**: `docs/audits/v0.7.7-audit.md` (225 lines, 13 findings) and `docs/audits/v0.7.7-validation.md` (135 lines, 70 probes).
- **Commit**: `9408a87`, **tag**: `v0.7.8`.
- **Data-leak near-miss**: 19MB memory text + 3.2MB pg_dump almost committed before being caught. Required the next session (v0.7.9) to formalize the rule.

### v0.7.9: Data-Leak Rule Followup (2026-07-11)
- **Adds critical "Never Commit Memory Data" rule to `AGENTS.md`** as follow-up to the v0.7.8 near-miss.
- **Pre-commit inspection pattern**: `find <dir> -type f | xargs file | grep -iE "text|json|sql|archive"` and `du -sh <dir>/*` before `git add`.
- **`.gitignore` updates**: `*.dump`, `*.jsonl`, `archives/memini-migration-backup.jsonl`, `archives/memini-migration-to-bge-large-backup.jsonl`, `archives/memini-postgres-pre-migration.dump`.
- **`uv.lock` refresh** to match v0.7.8 `pyproject.toml`.
- **809 tests pass**, 3 skipped, 0 failed. ruff/mypy clean.
- **Commits**: `2c71c2a` (docs/agents rule) + `cb6fe6b` (v0.7.9 release).

## v0.8.0 + v0.8.1 + v0.8.2

### v0.8.0: Image-Recall RRF Fan-Out Arm (2026-07-13)
- **Image Recall RRF fan-out arm**: when `MEMINI_IMAGE_SEARCH_ENABLED=true`, `query_memories` adds a 3rd RRF fan-out arm that calls `memini-vision.ImageQuery.search_by_text` (CLIP text tower over the `memories_image` table) and fuses with the existing 384-dim MiniLM + 1024-dim BGE-M3 via the unchanged `reciprocal_rank_fusion()` (k=60).
- **Image arm is best-effort**: any CLIP failure is caught, logged, and text RRF proceeds with 2 lists.
- **`_query_dual_model_rrf` renamed to `_query_multi_model_rrf`** (handles 2 OR 3 models).
- **New `memories_image` table** (migration `000008_add_memories_image.sql`): 768-dim CLIP image embeddings, 1:1 FK to `memories.id` ON DELETE CASCADE. `vector(768)` accommodates both ViT-B/32 (zero-padded) and ViT-L/14 (native). Created at memini-ai startup REGARDLESS of whether image search enabled.
- **`source_type='image'` added to CHECK constraint.**
- **5 new env vars**: `MEMINI_IMAGE_SEARCH_ENABLED` (default `false`), `MEMINI_IMAGE_CLIP_MODEL` (default `clip-ViT-B-32`), `MEMINI_IMAGE_CLIP_DEVICE` (default `auto`), `MEMINI_IMAGE_DIR` (default `~/.memini-ai/images`), `MEMINI_IMAGE_DB_URL`.
- **`[vision]` optional dep**: `vision = ["memini-vision>=0.1.0"]`. `memini_vision` import is lazy.
- **Text-only users see ZERO behavior change. Backward compatible.**
- **799 tests pass**, 3 skipped, 10 pre-existing Keras 3 / tf-keras env failures.
- **ruff clean, mypy**: 1 pre-existing numpy stub error on Python 3.14.
- **Design doc**: `docs/design/vision-memory-architecture.md` (30KB).
- **Commits**: `25eb3aa` (design doc) + `15ad805` (v0.8.0 implementation).

### v0.8.1: CI Re-Trigger (2026-07-13)
- **Pure CI re-trigger** for `memini-vision` dependency that wasn't yet on PyPI when v0.8.0 published.
- **No code changes from v0.8.0.** This release is purely a CI re-run.
- **Original v0.8.0 tag preserved** on origin (failed publish attempt).
- **Commits**: `241e471` (v0.8.1 CHANGELOG entry) + `705fc36` (test trigger) + `52e8350` (v0.8.1 release).

### v0.8.2: Security (2026-07-13)
- **SECURITY**: adds `detect-secrets` baseline + CI scan to prevent API key/secret leaks in commit history.
- **Pre-commit hook + GitHub Actions workflow** run `detect-secrets` on every push.
- **812 + 13 tests pass**, ruff/mypy clean.
- **Background**: An Ollama Cloud API key was leaked in the public Git history of `boomerang-v3`, `neuralgentics`, and `memini-ai-dev` on 2026-07-13. The key appeared in `.opencode/opencode.json`, `scripts/install-boomerang.js`, `.env.example`, and `HANDOFF.md`. It was rotated, then `git-filter-repo` rewrote history to replace it with `YOUR_OLLAMA_CLOUD_API_KEY` placeholder. All 3 repos force-pushed.

## v1.0.0 + v1.0.1 + v1.0.2

### v1.0.0: Embedded pgembed Backend (2026-07-16)
- **MAJOR: Embedded PostgreSQL is now the default backend.** v0.8.2 used external Postgres. The new `pgembed` driver starts an in-process Postgres 17 server on first query. No Docker required.
- **`MEMINI_VECTOR_BACKEND` must be set explicitly** if you have `MEMINI_DB_URL` configured (v0.8.2 users will get a `RuntimeError` on startup with clear remediation).
- **Python 3.12+ required** (was 3.11+). pgembed 0.2.0 requires Python 3.12+.
- **`PostgresDatabase.__init__` now takes a `driver` parameter** instead of `db_url` (internal; users go through `create_database()` which is unchanged).
- **Data dir location changed** from `~/.memini-ai/pgembed/` to `~/.local/share/memini-ai/pgembed/data` (XDG Base Directory spec).
- **Driver pattern**: `DatabaseDriver` Protocol with `EmbeddedPGDriver` + `ExternalPGDriver` implementations.
- **Multi-process server sharing**: 1 embedded Postgres shared by all memini-ai processes on same machine. Cooperative heartbeat (1s ping, 2s timeout, 5s drain grace).
- **RRF fusion across embedded + team server** via `RRFDatabase` wrapper. Writes go to primary (embedded) only; reads fan out to both backends, fuse via RRF.
- **CLI commands**: `memini-ai init`, `memini-ai status`, `memini-ai stop`, `memini-ai migrate`.
- **4 new env vars**: `MEMINI_VECTOR_BACKEND`, `MEMINI_PGEMBED_DATA_DIR`, `MEMINI_TEAM_DB_URL`, `MEMINI_FUSION_MODE`.
- **100% backward compatible** with v0.8.2 if `MEMINI_VECTOR_BACKEND=postgres-external` is set.
- **Design doc**: `docs/design/v1.0.0-embedded-pgembed-architecture.md` (76KB).
- **Commit**: `74b81cf` (feature) + `8c7b9f7` (release) + `c795931` (merge).

### v1.0.1: migrate script fix (2026-07-16)
- **6 bugs fixed** in `scripts/migrate_external_to_embedded.py`:
  1. Used system pg_dump/pg_restore (pg18) instead of pgembed's pg17. Now: `pg_dump` from system PATH (>= source version), `pg_restore` from pgembed (matches target).
  2. `parse_db_url` didn't extract `?host=` for Unix socket URIs (was in `.query`, not `.hostname`).
  3. Didn't pre-install `vector`+`vectorscale` extensions on target before restore.
  4. Didn't exclude `timescaledb`+`timescaledb_toolkit` from dump (not in pgembed).
  5. `request_explicit_shutdown()` is sync, not async — `await` was crashing.
  6. Spot-check column was `content` not `text`.
- **Added `--dry-run` flag**, post-restore verification (per-table row counts, random memory spot-check, diskann index existence).
- **Commit**: `63cfb8a` + `9b4d456` (release).

### v1.0.2: migrate CLI fix (2026-07-16)
- **6 bugs in `src/memini_ai/cli.py::_migrate()`** (same as v1.0.1 but in the CLI command, not the standalone script). CLI brought to parity with the standalone script.
- **Commit**: `b050806` (fix) + `ad30e2c` (release).

### Next steps (Session 41+ backlog)
1. **Restart OpenCode TUI** to load the v0.7.6 code (running TUIs have cached pre-v0.7.5 code)
2. **Verify tier0/tier1 summaries** end-to-end on the new code (Session 12 E2E skipped these)
3. **Fix the text-only search path** — `text_only_search` in `src/memini_ai/memory/search.py` relies on a lazy BM25 index that may return 0 if hydration is incomplete
4. **Make tests env-isolated** — 4 pre-existing failures in `test_config.py` / `test_thought_chains.py` from shell env vars should use `monkeypatch` fixtures
5. **Update other `opencode.json` files** — 10+ other projects reference memini-ai-dev but don't have the new `MEMINI_MODEL_NAME` / `MEMINI_ENABLE_RRF` env vars yet (only `boomerang-v3/.opencode/opencode.json` and root `MCP-Servers/.opencode/opencode.json` were updated in Session 39)
6. **Bump boomerang-v3 version** to reflect the memini-ai v0.7.6 dependency update
