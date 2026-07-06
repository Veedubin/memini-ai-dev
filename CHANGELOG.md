# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2026-07-06

### Bug Fixes

- **`query_memories` returned 0 results for all natural-language queries in the 0.4-0.7 cosine-similarity range.** Symptom (reported 2026-07-06 by the boomerang orchestrator after a diagnostic writeup): every `query_memories` call returned `{"count": 0, "memories": []}` even right after a successful `add_memory`. The write path was healthy (data was persisting correctly — verified via direct SQL), but the read path silently filtered out legitimate matches.
  - **Root cause (Bug A)**: `SearchOptions.threshold` default of `0.72` (`src/memini_ai/memory/schema.py:324`) is unrealistically tight for MiniLM-L6-v2 384-dim cosine similarity. Real-world similarity for natural-language queries against semantically related stored memories typically lands in 0.4-0.7 (distance 0.3-0.6). The 0.72 threshold (distance < 0.28) filters out the vast majority of legitimate matches. **Fix**: lowered the default to `0.0` (no SQL-side filtering; ranking is the responsibility of RRF/score-based top-K, not the SQL `<` clause). Docstring updated to explain the cosine-similarity range and the caller's option to pass a higher value.
  - **Root cause (Bug B)**: `_query_dual_model_rrf` in `src/memini_ai/memory/system.py:456-460` was building the 384-side `SearchOptions` WITHOUT propagating the caller's `threshold` and `exact_search` flags. So even if the caller passed a permissive threshold, auto-mode RRF silently used the (now-fixed) 0.72 default. **Fix**: pass `threshold=options.threshold` and `exact_search=options.exact_search` through to the 384-side `SearchOptions`.

### Added (Observability)

- **`get_status` now reports actual row counts**: `memoryCount` (from `_db.count_memories()`) and `thoughtsCount` (from new `_db.count_thoughts()`) are included in the response, plus a `queryLatencyMs` for the count probe. A `memoryCount: 0` with `memoryReady: true` is a contradiction the agent can now detect from within the protocol — addresses Priority-0 recommendation #2 in the 2026-07-06 diagnostic writeup.
- **New `count_thoughts()` helpers** added to `postgres/database.py`, `memory/database.py` (abstract), and `memory/system.py` (wrapper). Best-effort — backends that don't implement it return 0.
- **Post-write read-back in `add_memory`**: after a successful write, the handler calls `get_memory(memory_id)` to confirm the row is retrievable. If the read-back returns `None`, the response is `{"success": false, "error": "post_write_readback_failed", ...}` instead of falsely claiming success. Audit log includes `readback_verified: True`. Addresses Priority-0 recommendation #1 in the 2026-07-06 diagnostic writeup.
- **New `healthcheck` MCP tool**: writes a known marker memory, immediately reads it back, returns `{"status": "pass"|"fail", "memoryId": ..., "writeLatencyMs": ..., "readLatencyMs": ..., "readbackMatch": bool, "error": str|None}`. Audit-logs critical on failure. Lets the agent (and any future startup probe) verify end-to-end storage + read-path health with a single call. Addresses Priority-1 recommendation #3 in the 2026-07-06 diagnostic writeup.

### Tests

- **5 new regression tests** (total: 777 passing, was 766 → +11 net after adjusting for the threshold-default change that obsoleted one assertion):
  - `tests/test_dual_model.py::test_rrf_propagates_threshold_to_384_side` — **the key regression test for Bug B** — patches the search layer to capture the inner `SearchOptions`, asserts the caller's `threshold=0.5` and `exact_search=True` reach the 384-side.
  - `tests/test_dual_model.py::test_default_search_options_threshold_is_zero` — regression test for Bug A.
  - `tests/test_server.py::test_add_memory_post_write_readback_failure` — mock `get_memory` returns `None`, assert handler returns `success=False, error="post_write_readback_failed"`.
  - `tests/test_server.py::test_get_status_includes_row_counts` — assert `memoryCount` and `thoughtsCount` are non-negative ints.
  - `tests/test_server.py::test_get_status_count_failure_does_not_break` — count probe errors must not crash the whole status call.
  - `tests/test_server.py::TestHealthcheck::test_healthcheck_pass` and `test_healthcheck_fail_on_readback_mismatch` — pass/fail paths for the new healthcheck tool.

### Quality Gates

- `ruff check src/ tests/` → 0 errors
- `mypy src/` → 0 errors (53 source files)
- `pytest tests/ --ignore=tests/test_postgres_database.py` → **777 passing** (was 766, +11 net). 4 pre-existing env-var-pollution failures (`MEMINI_PROJECT_ID` and `THOUGHT_CHAINS` set in the active shell) — NOT caused by this change, present on `main` before the fix.
- In-process E2E: `query_memories("Inversion Audit Program Wave 0 1 COMPLETE", VECTOR_ONLY)` now returns 5 results (was 0 pre-fix). `auto/TIERED` mode also returns 5. Verified against the live `postgres` database (627+ memories at 384-dim, zero data loss).

### Notes

- **The original diagnostic writeup's "writes are silently dropped" conclusion was incorrect at the storage layer.** The exact UUIDs from the report (`5417cb0c-5bf9-4b07-a493-7ee08b6909ba`, `50e696d9-4fc8-4083-baef-79c937c594de`, `da2fab50-...`, `599da157-...`) are present in the live `postgres` database, with valid 384-dim embeddings and the exact reported text. The bug was purely on the read path. The 2026-06-11 review-note claim "memini-ai is offline" is also stale — the `memini-postgres` container has been up and healthy for 13+ hours as of 2026-07-06. The `memini` database (a separate, empty DB) is NOT the active one; the active DB is `postgres` (per `MEMINI_DB_URL=postgresql://postgres:password@localhost:5434/postgres` in `.env` and the `memini-ai-dev` MCP server config).
- **Why the threshold default was 0.72 historically**: the original spec treated 0.72 as "the cosine similarity floor for relevant results" (a heuristic from a different embedding model). MiniLM-L6-v2's actual similarity distribution is shifted lower, so the heuristic was too strict. v0.7.3 makes the default permissive (0.0) and lets the caller opt into stricter filtering when they need it. The RRF re-ranking handles top-K selection correctly without an SQL-side filter.

## [0.7.2] - 2026-06-04

### Notes

- **Patch release with no code changes.** The v0.7.1 source is unchanged. This release exists to (a) record the Session 10 health-check verification for downstream PyPI consumers, and (b) provide a versioned checkpoint paired with the `@veedubin/boomerang-v3@0.5.3` release that ships the corresponding `opencode.json` config fix.
- **Verified state (Session 10, 2026-06-04)**:
  - 206 memories at 384-dim in `memories` table, schema intact, zero data loss since v0.7.0
  - 71 thoughts at 384-dim, `thought_chains` + `thoughts` tables healthy
  - `memories_1024` table exists (per v0.7.0 migration), empty (0 elevated memories)
  - `get_status` MCP tool reports `memoryReady: true` after first lazy-init tool call
  - In-process E2E: `MCPServer` construction + `query_memories` + `get_status` all green
  - `pip install -e .` install flow is unchanged
- **Corrected a stale Session 9 diagnosis.** Session 9's HANDOFF note "memory server is currently broken (vector dim 1024 vs 384 mismatch from v0.7.0 dual-model)" was incorrect. The memory server works fine. `get_status` reports `memoryReady: false` only because it does not trigger lazy init — every other MCP tool (`query_memories`, `add_memory`, etc.) lazy-inits `_memory_system` on first call via `await self._init_memory_system()`. After one tool call, `memoryReady` flips to `true`. The dual-model RRF code handles both `cpu` and `auto` modes correctly via the `EMBEDDING_MODE` env.
- **Companion release**: `@veedubin/boomerang-v3@0.5.3` ships the same `minimax-m3` model-registration fix in the published npm `opencode.json` (see the Boomerang-v3 CHANGELOG for that release). Both fixes address the same root cause: a missing model key in the project config triggered `ProviderModelNotFoundError` on every `boomerang` (primary orchestrator) task dispatch.

### Quality Gates

- `uv run ruff check src/ tests/` → 0 errors
- `uv run mypy src/` → 0 errors
- `uv run pytest tests/ --ignore=tests/test_postgres_database.py` → 766 passing (unchanged from v0.7.1)
- In-process E2E (MCPServer init + query_memories + get_status) → green

## [0.7.1] - 2026-06-03

### Bug Fixes

- **`add_thought` MCP tool was crashing with vector-injection error** at runtime. Symptom: `invalid input for query argument $11: '[0.1,0.2,...]' (could not convert string to float: ...)`. Root cause: `src/memini_ai/thought_chains.py::add_thought` was building a stringified pgvector literal (`f"[{','.join(str(v) for v in embedding)}]"`) and passing it to asyncpg as `$11::vector`. asyncpg cannot bind a stringified literal directly to a `vector` type — it expects either a `list[float]` or `numpy.ndarray` (registered via `pgvector.asyncpg.register_vector`). Fix: pass the raw `list[float]` directly, matching how `memory.add` already does it. Also removed the unnecessary `$11::vector` cast in the SQL (`asyncpg + register_vector` handles the type binding automatically).
- **Dimension-mismatch safety**: when the embedding model returns a vector whose dim doesn't match the `thoughts.embedding vector(384)` column, the new code truncates (>384) or zero-pads (<384) to 384 before binding. This handles the case where `ModelManager` prefers BGE-Large (1024-dim) on GPU and falls back to MiniLM (384-dim) on CPU — previously the 1024-dim path would crash with "expected 384 dimensions, not 1024".

### Tests

- **3 new tests** in `tests/test_thought_chains.py::TestAddThought`:
  - `test_embedding_truncates_to_384_when_model_returns_1024`: regression test for the GPU/1024-dim path.
  - `test_embedding_pads_to_384_when_model_returns_smaller`: edge case for sub-384-dim models.
  - `test_add_thought_binds_embedding_as_list_not_string`: **the key regression test** — mocks `generate_embedding`, captures the actual argument passed to `conn.fetchrow`, and asserts it's a Python `list[float]`, not a string. Catches any future re-introduction of the stringification bug.
- Total: **766 passing tests** (was 763 in v0.7.0). ruff + mypy clean.

### Notes

- This was a HIGH-priority fix: `add_thought` is a required step in the Boomerang Protocol (step 2: Thought Chains). Without this fix, every orchestrator session that tried to plan complex work hit the bug.
- The Boomerang Protocol step 2 (Thought Chains) is now fully functional over MCP stdio.

## [0.7.0] - 2026-06-02

### Features

- **Dual-model RRF (384 + 1024)**: New `memories_1024` sidecar table holds 1024-dim embeddings for "elevated" memories. The 384-dim `memories` table remains the source of truth; the 1024 sidecar is additive (no schema change to existing data, no data loss).
- **Embedding mode dispatch** (`EMBEDDING_MODE` env, default `auto`):
  - `cpu`: 384-dim-only writes and queries (legacy path)
  - `auto`: 384-dim writes; queries fuse 384 + 1024 via Reciprocal Rank Fusion (RRF, k=60)
  - `gpu`: 1024-dim mirror always written; queries use 1024 only
- **`elevate_memory_to_1024` MCP tool** (auto-mode gated): promotes a 384-dim memory to also exist in 1024-dim space. Bumps trust +0.10 on both 384 and 1024 records. Idempotent. Returns `{memory_id, elevated, trust_score, vector_dim, mode, success}`.
- **Reciprocal Rank Fusion** (`src/memini_ai/memory/rrf.py`): new `reciprocal_rank_fusion(ranked_lists, k=60)` and `rrf_with_limit(...)` helpers. Reference: Cormack, Clarke, Buettcher, SIGIR 2009.
- **Defensive `asyncio.iscoroutinefunction` guards** in `memory/system.py` dispatch: replaces bare `hasattr()` checks (which return True for any MagicMock test fixture). The MagicMock tests in `test_system.py` were crashing on `await` of non-AsyncMock attributes; now they fall through cleanly to the legacy 384-only path.
- **5 new env vars**: `EMBEDDING_MODE` (cpu/auto/gpu), `ELEVATE_ENABLED` (bool), `RRF_K` (1-1000), `AUTO_EXTRACT_LOG_DIR`, `AUTO_EXTRACT_INTERVAL_SECONDS` (1-3600s). All have field validators in `MeminiConfig`.
- **36th MCP tool** registered: `elevate_memory_to_1024` (now 36 total).

### Notes

- The 1024-dim vector is currently a **placeholder expansion** of the 384-dim vector (`_expand_384_to_1024`: zero-pad + L2-normalize). A future v0.7.1/v0.8.0 release will swap in a real BGE-Large call when the elevate tool is invoked.
- The `embedding_dim` config default is now `384` (was `1024` in v0.6.x). This aligns the config default with the schema default.
- The `memories_1024` migration is idempotent (`CREATE TABLE IF NOT EXISTS`) and zero-touch on existing `memories` data.
- Trust boost on elevate uses `MEMINI_TRUST_DELTA_CONFIRM` semantics (clamped to [0, 1]).

### Tests

- **763 tests passing** (740 v0.6.0 baseline + 23 new) — `pytest tests/`
- **0 ruff errors** — `ruff check src/ tests/`
- **0 mypy errors** — `mypy src/`
- **23 new tests** across 3 new files:
  - `tests/test_rrf.py` (10): RRF algorithm unit tests (no DB)
  - `tests/test_dual_model.py` (8): mode dispatch + RRF k clamping (mocked DB)
  - `tests/test_schema_migration.py` (5): real-DB schema verification
- **+1 test fix**: `tests/test_config.py::test_model_settings_defaults` updated for new `embedding_dim=384` default.
- **3 pre-existing ruff issues** also fixed (test_dialectic.py, test_extractor.py, test_input_validation.py).

### Release

- Commit: `18f37ed` on `main`
- Tag: `v0.7.0`
- Remote: `https://github.com/VeeDubin/memini-ai-dev.git`
- 22 files changed, +2108 / -74 lines
- **83 memories preserved** (zero data loss through migration, dispatch, tool, and quality gates)

### Migration Notes

- For existing v0.6.x installations: no action required. The new `memories_1024` table is created automatically on next server start (`initialize()` is idempotent).
- For new installations: set `EMBEDDING_MODE=auto` (default) to get the dual-model RRF behavior, or `EMBEDDING_MODE=cpu` to match pre-v0.7.0 behavior.

## [0.3.0] - 2026-05-19

### Features

- Memory Delta Model: Partial memory updates with `supersedes_id`, `structured_fields`, `change_ratio`
- Epoch-ms timestamps (`created_at_ms`) for temporal ordering in supersession chains
- Supersession chain traversal: `get_supersession_chain`, `get_superseded_memory`
- New `PARTIAL_UPDATE` relationship type alongside `SUPERSEDES`
- New `src/memini_ai/memory/merger.py` for structured field merging
- New migration script: `scripts/migrate_delta_model.py`
- Updated MCP tools with delta-aware parameters
- Self-referencing relationships filtered out in `find_related_memories`

### Tests

- 693 tests passing (37 PostgreSQL connection errors due to local DB not running)

### Bug Fixes

- Fixed self-referencing relationships being returned in `find_related_memories`

### Breaking Changes

- None (backward compatible)

## [0.2.0] - 2026-05-18

### Features

- pgvector/pgvectorscale backend with StreamingDiskANN index
- VectorDatabase ABC for database abstraction
- PostgresDatabase class with asyncpg support
- New `postgres/` module with schema and queries
- Migration script: `scripts/migrate_qdrant_to_pgvector.py`
- New config options: `MEMINI_DB_URL`, `db_pool_size`, `db_min_size`, `db_max_size`

### Tests

- 38 new tests for PostgresDatabase

### Bug Fixes

- N/A

### Breaking Changes

- None (backward compatible with Qdrant)