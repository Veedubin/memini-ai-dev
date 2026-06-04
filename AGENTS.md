# Memini-ai Agent Context

## ⚡ CRITICAL: memini-ai Memory Protocol (MUST FOLLOW)

All agents **MUST** interact with memini-ai at every step:
1. **Query FIRST** — Call `memini-ai-dev_query_memories` before starting work
2. **Save DURING** — Call `memini-ai-dev_add_memory` after every meaningful decision
3. **Preserve CONTEXT** — Save important context; query it back when continuing work

Failure to use memini-ai causes context loss, duplicate work, and wasted tokens.


## Project-Specific Context
This is memini-ai-dev — a Python-based semantic memory server with PostgreSQL/pgvector backend. Key facts:
- Language: Python 3.11+
- Framework: FastMCP (MCP server with 36 tools, including the v0.7.0 `elevate_memory_to_1024`)
- Database: PostgreSQL with pgvector + pgvectorscale
- Embeddings: BGE-Large (1024-dim, via placeholder expansion) / MiniLM-L6-v2 (384-dim, default)
- Dual-model RRF: cpu/auto/gpu modes via `EMBEDDING_MODE` env, 1024-dim sidecar in `memories_1024` table, RRF k=60 (Cormack SIGIR 2009)
- All features are independently optional via environment variables

## Quality Gate Commands (copy-pasteable)
```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev
ruff check src/
mypy src/
python -m pytest tests/ -v
```

## Environment Variables (key ones)
| Variable | Description | Default |
|-----------|-------------|---------|
| MEMINI_DB_URL | PostgreSQL connection | Set via `.env` (see `.env.example`) |
| MEMINI_EMBEDDING_DIM | 384 or 1024 | 384 |
| MEMINI_EMBEDDING_MODE | cpu / auto / gpu dispatch (v0.7.0+) | auto |
| MEMINI_ELEVATE_ENABLED | Enable `elevate_memory_to_1024` MCP tool (v0.7.0+) | true |
| RRF_K | Reciprocal Rank Fusion k constant (v0.7.0+) | 60 |
| MEMINI_TRUST_ENGINE | Enable trust scoring | false |
| MEMINI_MEMORY_GRAPH | Enable memory graph | false |
| MEMINI_AUTO_EXTRACT | Enable auto-extraction | false |
| MEMINI_TIERED_LOADING | Enable tiered loading | false |
| MEMINI_KG_ENABLED | Enable knowledge graph | false |
| MEMINI_MULTI_PEER_ENABLED | Enable multi-peer | false |
| MEMINI_DIALECTIC_ENABLED | Enable dialectic reasoning | false |
| THOUGHT_CHAINS | Enable persistent thought chains | false |

## Review Notes
- **2026-06-04 (Session 11)**: **v0.7.2 PATCH METADATA RELEASED** ✅ — No code changes from v0.7.1. CHANGELOG entry documents the Session 10 health-check verification (206 memories at 384-dim, 766 tests passing, MCP server end-to-end working) and **corrects the stale Session 9 "memory server broken" diagnosis**. Companion release to `@veedubin/boomerang-v3@0.5.3` (which ships the same `minimax-m3` model-registration fix in the published npm `opencode.json`). Commit `6fda0ba` on `main`, tag `v0.7.2` (`b98ef3a`) pushed to `VeeDubin/memini-ai-dev`. Quality gates green: ruff 0, mypy 0 (53 source files), 766/766 tests pass, in-process E2E verified. CI will publish to PyPI within 2-5 min via trusted publishing. **OpenCode TUI restart still required** (3 live TUIs at PID 917732, 1160224, 1162490).
- **2026-06-03 (Session 6)**: **v0.7.1 BUGFIX RELEASED** ✅ — `add_thought` MCP-call vector-injection error fixed. **Root cause**: `src/memini_ai/thought_chains.py::add_thought` was building a stringified pgvector literal (`f"[{','.join(str(v) for v in vec)}]"`) and passing it to asyncpg as `$11::vector`. asyncpg cannot bind a stringified literal to a `vector` type — it expects `list[float]` (handled by `pgvector.asyncpg.register_vector`). Secondary issue: 1024-dim BGE-Large model would have crashed the `vector(384)` column even with correct binding. **Fix**: pass `list[float]` directly (matches how `memory.add` does it), drop the `::vector` cast, truncate/pad to 384 dims. **3 new tests** in `tests/test_thought_chains.py::TestAddThought` including a key regression test that captures the actual arg passed to `conn.fetchrow` and asserts it's a `list`, not a `str`. **766 tests passing** (was 763, +3), ruff+mypy clean, in-process E2E verified. **Boomerang Protocol step 2 (Thought Chains) is now fully functional over MCP stdio.**
- **2026-06-02 (Session 5)**: **v0.7.0 DUAL-MODEL RRF RELEASED** ✅ — All 15 implementation steps done in a single session (orchestrator file-level parallel edits; Task tool still blocked by cached agent configs). Commit `18f37ed` on `main`, tag `v0.7.0` pushed to `VeeDubin/memini-ai-dev`. **763 tests passing (740 baseline + 23 new), ruff + mypy clean, 83 memories preserved (zero data loss).** New: `memory/system.py` cpu/auto/gpu dispatch with defensive `asyncio.iscoroutinefunction` guards (MagicMock-safe), `_query_dual_model_rrf` and `_query_gpu_1024` private methods, deleted dead `_get_fallback_for_dimension()`. New `elevate_memory_to_1024` MCP tool (auto-mode gated at call time). 3 new test files: `test_rrf.py` (10), `test_dual_model.py` (8), `test_schema_migration.py` (5). 3 pre-existing ruff issues in `test_dialectic.py` / `test_extractor.py` / `test_input_validation.py` also cleaned up. **OpenCode restart STILL REQUIRED** for `task` dispatch to work (PID 307190 has cached `ollama-cloud/<model>:<tag>-cloud` agent config).
- **2026-06-02 (Session 4)**: **v0.7.0 DUAL-MODEL RRF — 5/15 STEPS DONE** — Session 4 continued v0.7.0 implementation. **Step 1 (config validators), Step 2 (memories_1024 table), Step 3 (6 new 1024 query constants), Step 4 (memory/rrf.py NEW), Step 5 (6 new database methods + _expand_384_to_1024 helper) — ALL COMPLETE**. Migration applied to live DB; **83 memories at 384-dim verified intact** (was 80 in Session 3, +3 from this session's testing). Working tree dirty on `config.py`, `schema.py`, `queries.py`, `database.py` + new file `memory/rrf.py`. `ruff + mypy` clean for all 5 modified/created files. **Agent-blocker fix:** All 47+ agent `.md` files across 6 locations (root, boomerang-v3, neuralgentics, Super-Memory, boomerang, plus the critical `node_modules/@veedubin/boomerang-v3` install and the npm cache) corrected from `ollama-cloud/<model>:<tag>-cloud` → `ollama/<model>:<tag>`. Ollama Cloud API confirmed all 10 model names exist. **OpenCode restart STILL REQUIRED** — running process (PID 307190) has old config cached. Saved to memini-ai memory `b8b42742-e4e1-4a2a-a1a1-afd85e597f59`. See `TASKS.md` v0.7.0 Implementation Status table for remaining 10 steps.
- **2026-06-01**: **v0.7.0 IMPLEMENTATION STARTED** — Dual-model RRF work in progress. Step 1 of 14 done in `src/memini_ai/config.py` (embedding_dim 1024→384, 5 new fields added: `embedding_mode`, `elevate_enabled`, `rrf_k`, `auto_extract_log_dir`, `auto_extract_interval_seconds`; field validators PENDING). Design: `docs/design/dual-model-rrf-architecture.md`. **80 memories at 384-dim verified intact** (4 added since last handoff's "76"). Working tree dirty. Restart OpenCode before relying on `task` dispatch (ProviderModelNotFoundError from cached `ollama-cloud/<model>:cloud` agent configs).
- **2026-05-19**: **memini-ai-dev v0.3.1 RELEASED** — Documentation refreshed, stale version references updated. pyproject.toml bumped from v0.3.0 → v0.3.1.
- **2026-05-19**: **memini-ai-dev v0.3.0 RELEASED** — Thought chains persistent reasoning with branching/revision, PostgreSQL schema, 9 MCP tools, exact_search for DiskANN. ruff: 0 errors, mypy: 0 errors, pytest: 704/704 passed. Tag `v0.3.0` pushed.
- **2026-05-19**: **memini-ai-dev v0.2.8 RELEASED** — Ruff formatting pass across 30 files. Tag `v0.2.8` pushed.
- **2026-05-19**: **memini-ai-dev v0.2.7 RELEASED** — PostgreSQL schema fixes: IF NOT EXISTS, vector parsing, 384-dim vectors. Tag `v0.2.7` pushed.

## Key Reference Files
| File | Purpose |
|------|---------|
| TASKS.md | 5-phase task breakdown |
| HANDOFF.md | Session handoff notes |
| CONTEXT.md | Architecture decisions |
| README.md | Installation and usage |
