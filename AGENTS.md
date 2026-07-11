# Memini-ai Agent Context

## Provider Configuration (Ollama Cloud & Alternatives)

All projects in this workspace ship with **Ollama Cloud** as the default
LLM provider. To switch to a different provider — local Ollama, Docker
Model Runner, OpenAI, Anthropic, Google, OpenRouter, or any
OpenAI-compatible endpoint — see:

> **`~/Projects/MCP-Servers/docs/providers.md`** — the canonical
> provider-switching guide. Covers 5 recipes (local Ollama, Docker
> Model Runner, the Big Three, OpenRouter, custom endpoints), a
> quick-reference for just changing which Ollama Cloud model each
> agent uses, a 6-step migration checklist, and a troubleshooting
> table for the common `ProviderModelNotFoundError`,
> `Provider not found`, and `401 Unauthorized` errors.

If you only want to swap which model each agent uses (and the model
already exists in `provider.ollama.models`), the guide shows a `sed`
one-liner that does it in seconds.

## MCP Servers

This project integrates 12 MCP servers for specialized tooling. The configuration was fixed this session to ensure all 12 servers are wired in.

| Server | Purpose |
|--------|---------|
| memini-ai-dev | Python semantic memory + knowledge graph + tiered loading (PRIMARY) |
| markitdown | Convert files (PDF/DOCX/HTML) to Markdown |
| duckdb | In-memory SQL via DuckDB |
| redis | Redis key-value access on localhost |
| playwright | Browser automation / web scraping |
| calculator | Math evaluation |
| prefect | Prefect workflow orchestration |
| mlflow-mcp | MLflow experiment tracking + model registry |
| doc2png | Document to PNG rendering |
| github-mcp | GitHub repo/issue/PR operations (needs GH_TOKEN) |
| videre-mcp | Vision: screenshot, OCR, image description (Florence-2 / PaddleOCR) |
| searxng | Web search via SearXNG metasearch |

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
- Embeddings: MiniLM-L6-v2 (384-dim, default) / BGE-M3 (1024-dim, optional GPU upgrade). BGE-Large support was removed in v0.7.6.
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

- **2026-07-10 (Session 42)**: **v0.7.8 AUDIT + DOC REWRITE PATCH RELEASED** ✅ — Comprehensive 8-area audit (correctness, config, MCP tools, tests, migration, security, performance, docs) by boomerang-architect + boomerang-tester. **Code is correct** (70/70 live probes pass, 0 real bugs). **Docs had 13 real problems** — 1 CRITICAL (README "Enabling Multi-Model" example was missing `MEMINI_EMBEDDING_DIM=1024`, would have silently degraded users to text-only search), 4 HIGH (`.env.example` missing 6 v0.7.7 env vars, README env var table missing same, `upgrading-embeddings.md` referenced non-existent `sentence-transformers[gpu]` pip extra, migration script path was wrong), 6 MEDIUM, 2 LOW. All 13 fixed: README rewritten (tool count 35+→52, added 24 missing tools, regenerated architecture tree from actual file layout, added 6 env vars to Core Settings table), `.env.example` got v0.7.7 section, `upgrading-embeddings.md` Step 2 replaced with correct torch CUDA install, `archives/` moved INTO `memini-ai-dev/`, CHANGELOG v0.7.6 `enabled_models` inaccuracy corrected. **2 minor code fixes**: BM25 punctuation-only query guard (returns `[]` for queries with no alphabetic chars, fixes an audit overstatement), and `get_sentence_embedding_dimension` deprecation in migration script. **Process fix**: bumped `steps: N` frontmatter 10x across all 61 agent `.md` files (50→500, 40→400, 30→300) — sub-agents were hitting 50-step limits on legitimate long-running tasks. 809 tests pass, 0 failed, 3 skipped, ruff/mypy clean. Full reports at `docs/audits/v0.7.7-audit.md` and `docs/audits/v0.7.7-validation.md`.
- **2026-07-10 (Session 41)**: **v0.7.7 BGE-M3 OPT-IN PATCH RELEASED** ✅ — Non-breaking. Two new env vars: `MEMINI_AUTO_DETECT_MODEL` (default `true`; new deployments with 0 memories auto-upgrade to BGE-M3 1024-dim; existing users keep MiniLM) and `MEMINI_STRICT_EMBEDDING_DIM` (default `false`; dim mismatch logs WARNING + degrades to text-only instead of raising RuntimeError). Defense-in-depth dim assertion is now opt-in for safety. Fixed BM25 `text_only_search` empty-corpus `ZeroDivisionError` (3 guards in `_build_bm25_index`, `text_only_search`, `text_search_collection`). Fixed `get_sentence_embedding_dimension` deprecation warning (renamed to `get_embedding_dimension` in sentence-transformers 3.x). Fixed 4 pre-existing test failures via `autouse=True` `_isolate_env` fixture (pydantic-settings reads `.env` via `env_file=".env"`, so `monkeypatch.delenv` alone is not enough). `get_status` now reports `modelName`, `modelDimension`, `embeddingDimMismatch`, `embeddingDimExpected`, `embeddingDimActual`. New `docs/upgrading-embeddings.md` (4-step migration recipe: Backup → GPU/CPU setup → Run migration script → Update env vars + restart + verify; rollback; new-deployment guidance; FAQ). 807 tests pass (+23 net new), 0 failed, 3 skipped, ruff/mypy clean. **OpenCode restart required** to load v0.7.7 code in MCP server.
- **2026-07-10 (Session 40)**: **v0.7.6 BGE-LARGE REMOVAL** ✅ — BGE-Large support removed. The supported models are now exactly two: **MiniLM-L6-v2 (384-dim, default)** and **BGE-M3 (1024-dim, optional GPU upgrade)**. BGE-Large was added in v0.7.0 alongside BGE-M3 as a "high-precision 1024-dim option" but turned out not to be used in production. **What was removed**: `embedding_bge_large vector(1024)` column (dropped from live `memini-postgres` via migration 000007, 821 memories preserved, 819 MiniLM + 800 BGE-M3), `BGE_LARGE_MODEL_ID`/`BGE_LARGE_DIM` constants, `INSERT_MEMORY_BGE_LARGE`/`SEARCH_MEMORIES_BGE_LARGE` queries, BGE-Large entries in `COLUMN_TO_MODEL`/`MODEL_DIMS`/`enabled_models`. **What was kept**: the BGE-Large migration script at `archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py` stays as a reference example for users who want to do similar migrations on their own (e.g. swap to a different 1024-dim model, or upgrade from MiniLM to a custom model). **The MiniLM → BGE-M3 upgrade path is the canonical migration story** (script at `archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py`): start with MiniLM (fast, small, CPU-friendly), get a GPU, then migrate the existing memories to BGE-M3 (higher precision, GPU-friendly) without losing the original MiniLM data. **784 tests passing** (-40 from removing BGE-Large tests; 4 pre-existing env-var failures documented and unchanged), ruff+mypy clean, live DB verified. Backwards-incompatible at the schema level (column dropped), backwards-compatible at the API level (callers passing BGE-Large model_id get a clear `ValueError`). No new env vars.
- **2026-07-10 (Session 39)**: **v0.7.5 MULTI-MODEL RRF BUGFIX RELEASED** ✅ — Found and fixed 3 latent bugs that prevented the v0.7.0 multi-model RRF feature from actually working. (1) `ModelManager._load_model()` was constrained by `embedding_dim` instead of `config.model_name`, so BGE-M3 was unreachable. (2) `add_memory` wrote 1024-dim vectors to the 384-dim `embedding` column — silent data loss for BGE-M3/BGE-Large writes. (3) RRF `COLUMN_TO_MODEL` used short name `'all-MiniLM-L6-v2'` but `ModelManager` expects full HF name. **Fixes**: model_name-driven selection with alias support, multi-model column routing (new `INSERT_MEMORY_BGE_M3` / `INSERT_MEMORY_BGE_LARGE` queries), and full-HF-name RRF column mapping. **824 tests passing** (+47 new), ruff+mypy clean, live DB verification: BGE-M3 with `MEMINI_MODEL_NAME=BAAI/bge-m3` loads, produces 1024-dim vectors, writes to `embedding_bge_m3` column. All 3 model spaces now populated for 800 memories in `memini-postgres` (port 5434). RRF search returns results from all 3 spaces. **OpenCode TUI restart required** to pick up the new `MEMINI_MODEL_NAME` and `MEMINI_ENABLE_RRF` env vars in the MCP config.

- **2026-07-09**: **Database naming quirk — INVESTIGATED, NO CHANGE NEEDED (Option D)** — The `memini-postgres` container (port 5434) has 4 databases: `postgres`, `memini`, `template0`, `template1`. All memini-ai data (773 memories + 11 other tables = 12 total) lives in the **`postgres`** database. The **`memini`** database is **completely empty** (0 tables) — it is a vestigial artifact: the container was created with `POSTGRES_DB=memini` (confirmed via `podman inspect`), which caused Postgres to auto-create a `memini` database at init time. But the application has ALWAYS connected to `postgres` via `MEMINI_DB_URL=postgresql://postgres:password@localhost:5434/postgres` (in `.env`, line 7). The DB name is parsed purely from the URL path component in `memory/database.py:307` (`db_url = config.db_url or os.environ.get("MEMINI_DB_URL", "")`); there is NO hardcoded default and NO reference to the `memini` DB name anywhere in `src/` (verified with `grep -rn '5434/memini'` → 0 matches). Live `get_status` confirms `memoryCount: 773`, exactly matching `SELECT count(*) FROM memories` on the `postgres` DB. **Resolution chosen: Option D (document the quirk, no change).** Options A (migrate data to `memini` DB) and C (rename `postgres` → `memini`) were rejected as unnecessary data-migration risk for a purely cosmetic naming mismatch. The `memini` database can be safely dropped in the future if desired (it is unused), but per the container-deletion policy it is left in place unless the user explicitly approves. The `.env` and `.env.example` both correctly point at `postgres` and require no change.
- **2026-07-06 (Session 12)**: **v0.7.3 READ-PATH THRESHOLD BUGFIX RELEASED** ✅ — `query_memories` was returning 0 results for all natural-language queries because the default `SearchOptions.threshold = 0.72` (`src/memini_ai/memory/schema.py`) is unrealistically tight for MiniLM-L6-v2 384-dim cosine similarity (real matches land at sim 0.4-0.7, dist 0.3-0.6). Compounded by `_query_dual_model_rrf` (`src/memini_ai/memory/system.py:456-460`) NOT propagating the caller's `threshold` to the 384-side `SearchOptions`. **Fix**: lowered default to `0.0`; RRF now propagates `threshold=options.threshold` and `exact_search=options.exact_search`. **The 2026-07-06 diagnostic writeup's "writes are silently dropped" conclusion was incorrect at the storage layer** — the exact UUIDs from the report are present in the `postgres` database with valid 384-dim embeddings; the bug was purely on the read path. The 2026-06-11 "offline" review note is also stale — the `memini-postgres` container has been up 13+ hours. The active DB is `postgres` (per `MEMINI_DB_URL=postgresql://postgres:password@localhost:5434/postgres`), NOT the separate empty `memini` database. **Observability added**: `get_status` now returns `memoryCount` + `thoughtsCount` + `queryLatencyMs`. `add_memory` does a post-write read-back (returns `error="post_write_readback_failed"` if the row is gone). New `healthcheck` MCP tool (write+read round-trip with PASS/FAIL). 777 tests passing (was 766, +11), ruff+mypy clean, in-process E2E verified: `query_memories("Inversion Audit Program Wave 0 1 COMPLETE")` now returns 5 results (was 0 pre-fix). OpenCode TUI restart required to load the new code (PID of running TUI to be recorded in commit). 4 pre-existing test failures in `test_config.py` / `test_thought_chains.py` are caused by `MEMINI_PROJECT_ID=reverse_engineering` and `THOUGHT_CHAINS=true` being set in the active shell — not regressions, present on `main` before the fix.
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
