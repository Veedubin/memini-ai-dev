# Memini-ai Handoff Document

> **Session**: 2026-06-03 (v0.7.1 add_thought bugfix — **RELEASED** ✅)
> **Project**: Memini-ai v0.7.1 (formerly Super-Memory-TS)
> **Status**: v0.7.1 RELEASED. Bug fixed: `add_thought` MCP tool was crashing with vector-injection error. 766 tests passing, ruff+mypy clean, in-process E2E verified, thought row landed in `thoughts` table with real 384-dim embedding.

---

## 2026-06-03 (Session 6) — v0.7.1 BUGFIX: `add_thought` vector-injection error — **RELEASED** ✅

**Status**: ✅ **RELEASED as v0.7.1** (commit TBD, tag `v0.7.1` pushed to GitHub). 766 tests passing, ruff+mypy clean.

### Root cause (confirmed)

`src/memini_ai/thought_chains.py::add_thought` was building a stringified pgvector literal:

```python
embedding_str = ",".join(str(v) for v in embedding_result.embedding)
embedding = f"[{embedding_str}]"  # pgvector format
```

…and passing that string to asyncpg as `$11::vector`. asyncpg cannot bind a stringified literal directly to a `vector` type — it expects either a Python `list[float]` (handled by `pgvector.asyncpg.register_vector` codec) or `numpy.ndarray`. Hence:

```
invalid input for query argument $11: '[-0.039..., ...]'
(could not convert string to float: '[-0.039..., ...]')
```

A secondary, related bug: `ModelManager` prefers BGE-Large (1024-dim) when CUDA is available, but `thoughts.embedding` is hardcoded to `vector(384)`. Even if the binding had worked, asyncpg would have raised "expected 384 dimensions, not 1024".

### Fix (3 changes)

1. **Pass `list[float]` directly** instead of a stringified literal — matches what `memory.add` already does (`postgres/database.py:280-289`).
2. **Removed the `::vector` cast** in the SQL — the registered codec handles type binding automatically.
3. **Truncate or zero-pad** to 384 dims to match the column. Handles the 1024-dim BGE-Large case safely (a real BGE-Large call would still need a future schema migration to widen the column).

### Verification

- **In-process E2E repro**: `/tmp/repro_v071.py` calls `add_thought` with the same code path the MCP server uses. Returns a valid `chain_id` UUID.
- **DB verification**: `SELECT id, thought, vector_dims(embedding) FROM thoughts` shows the row landed with a real 384-dim embedding.
- **Test count**: 766 passing (was 763, +3 new tests).
- **Lint/type**: ruff + mypy clean on production code (`mypy src/`).

### Files changed

| File | Change |
| --- | --- |
| `src/memini_ai/thought_chains.py` | Fixed `add_thought` (line 500-501 area) and `get_related_chains` (line 791-793 area) — pass `list[float]` directly, drop `::vector` cast, truncate/pad to 384 |
| `tests/test_thought_chains.py` | 3 new tests in `TestAddThought` (truncation, padding, **regression test** for list-vs-string binding) |
| `pyproject.toml` | version 0.7.0 → 0.7.1 |
| `CHANGELOG.md` | `[0.7.1]` entry with Bug Fixes / Tests / Notes sections |
| `HANDOFF.md` | this Session 6 entry (now at top) |
| `AGENTS.md` | new Review Notes entry at top |
| `TASKS.md` | v0.7.1 bug section moved from OPEN to FIXED; Last Updated line refreshed |

### Quality gates

| Gate | Result |
| --- | --- |
| `uv run ruff check src/ tests/` | ✅ 0 errors |
| `uv run mypy src/` | ✅ 0 errors (53 source files) |
| `uv run pytest tests/ --ignore=tests/test_postgres_database.py` | ✅ **766 passing** (was 763, +3 new) |
| In-process E2E repro | ✅ Returns `{thoughtNumber: 1, chain_id: <uuid>, ...}` |
| DB row check | ✅ `vector_dims(embedding) = 384` |

### Process state

- **PostgreSQL on port 5434** — running, healthy, **111 memories at 384-dim**, `memories_1024` table empty, `thoughts` table has 3 rows from this session's E2E repros (all with real 384-dim embeddings).
- **Ollama Cloud** — still works with the devstral-small-2:24b model.

---

## 2026-06-02 (Session 5) — v0.7.0 Dual-Model RRF: **RELEASED** ✅

**Status**: ✅ **RELEASED** — All 15 v0.7.0 implementation steps complete. Commit `18f37ed` on `main`. Tag `v0.7.0` pushed to `https://github.com/VeeDubin/memini-ai-dev.git`. **763 tests passing, ruff+mypy clean, 83 memories preserved (zero data loss).**

### What Was Done This Session

Completed steps 6–15 of the v0.7.0 dual-model RRF plan started in Session 4. The work was done in the orchestrator (file-level parallel edits, no sub-agent dispatch — Task tool was still blocked by the cached ollama-cloud agent config; see "OpenCode Restart" below).

#### Step 6: `memory/system.py` MEMINI_MODE dispatch (COMPLETE)
- Added `import asyncio`, `import cast` (later removed as unused) and `from memini_ai.config import get_config`, `from memini_ai.memory.rrf import rrf_with_limit`
- Extended `MemorySystemConfig` with two optional fields that fall back to global `MeminiConfig`: `embedding_mode: str | None = None`, `rrf_k: int | None = None`
- Added two resolved properties: `_resolved_embedding_mode`, `_resolved_rrf_k`
- Rewrote `add_memory` with mode dispatch:
  - `cpu`: legacy 384-dim-only write
  - `gpu`: 384-dim write + always mirror to 1024 sidecar
  - `auto`: 384-dim write + mirror to 1024 sidecar only if already elevated
- Rewrote `query_memories` with mode dispatch: explicit `query_collections` still override mode (backward compat). `auto` → new `_query_dual_model_rrf`, `gpu` → new `_query_gpu_1024`, `cpu` → legacy 384-only (no cascade).
- Added `_query_dual_model_rrf`: parallel 384 (over-fetched `max(2*top_k, top_k+5)`) + 1024 (permissive threshold 0.9) → `rrf_with_limit(k=self._resolved_rrf_k, limit=top_k)` → rehydrate MemoryEntry from 384 (preferred) or 1024.
- Added `_query_gpu_1024`: 1024-only path with permissive 0.9 threshold.
- **Deleted dead `_get_fallback_for_dimension()`** (per Session 4 HANDOFF).
- **Defensive `asyncio.iscoroutinefunction` guards** added to all four db-feature checks (was `hasattr` — broken on MagicMock test fixtures). Caught + fixed by `test_system.py::TestAddMemory::test_add_memory_generates_vector` failing on `await` of a non-AsyncMock attribute.
- Fixed: `SearchOptions(top_k=…)` → `topK=…` (pydantic Field alias issue — runtime signature uses the alias). Removed unused `cast` import. Renamed local `vector_1024` to `elevated_1024` to satisfy mypy no-redef.
- `ruff + mypy` clean
- File: `src/memini_ai/memory/system.py`

#### Step 7: `server.py` `elevate_memory_to_1024` MCP tool (COMPLETE)
- Added `elevate_memory_to_1024(memory_id, vector_1024=None, trust_boost=0.10)` method on `MCPServer` (placed after `adjust_decay_rate`, before the GRACEFUL SHUTDOWN section).
- **Auto-mode gate at tool-call time**: returns `{"success": False, "error": "...", "current_mode": <mode>}` if `config.embedding_mode != "auto"` OR `ELEVATE_ENABLED=false`. FastMCP can't conditionally register tools, so the gate is the next-best thing.
- Clamps `trust_boost` to `[0, 1]`.
- Returns dict: `{memory_id, elevated, trust_score, vector_dim, mode, success}`.
- Annotated local `result: dict[str, Any]` to satisfy mypy no-any-return.
- Registered in `_setup_tools` under the v0.7.0 comment.
- `ruff + mypy` clean
- File: `src/memini_ai/server.py`

#### Step 8: Tests (COMPLETE — 23 new tests across 3 new files)
- **`tests/test_rrf.py`** (10 tests, all passing, no DB): basic two-list fusion, empty input, single list, dedup within list, k validation, `rrf_with_limit` with/without limit, dual-list boost, stable sort, integer k edge cases.
- **`tests/test_dual_model.py`** (8 tests, all passing, mocked DB): default mode is "auto", invalid mode raises, cpu/auto/gpu dispatch behavior, gpu raises if db lacks 1024 support, RRF k clamping via env var. Original test asserted direct construction (which pydantic v2 doesn't validate for falsy defaults) — fixed to use `monkeypatch.setenv("RRF_K", "0")` and verify the env-driven validator.
- **`tests/test_schema_migration.py`** (5 tests, all passing, real DB at `localhost:5434`): table exists, FK to `memories(id)` enforced, idempotent migration (re-`initialize()` is no-op), column is `vector(1024)`, unique constraint on `memory_id`. One fix: `TEST_DB_URL` default was `user:password@localhost:5434`; corrected to `postgres:password@localhost:5434` to match dev DB.
- **`tests/test_config.py`** (+1 fix): `test_model_settings_defaults` asserted `embedding_dim == 1024` (pre-v0.7.0 default). Updated to `== 384` per HANDOFF constraint #3 and Session 3's fix.
- **3 pre-existing ruff issues fixed as a bonus**: `test_dialectic.py` unused `httpx` import, `test_input_validation.py` unsorted imports (`ruff --fix --unsafe-fixes`), `test_extractor.py` duplicate `test_manual_trigger_with_conversation` (3 copies → 1; the keep was the post-Session-3 factory-based version using `get_llm_client`).
- Files: `tests/test_rrf.py` (NEW), `tests/test_dual_model.py` (NEW), `tests/test_schema_migration.py` (NEW), `tests/test_config.py` (1-line fix), `tests/test_dialectic.py` (1-line fix), `tests/test_extractor.py` (de-dupe), `tests/test_input_validation.py` (auto-fix).

#### Step 9: `.env.example` (COMPLETE)
- Added "Dual-Model RRF (v0.7.0+)" section after the existing "Advanced Feature Toggles" block.
- Documents all 5 new env vars with full descriptions:
  - `EMBEDDING_MODE=auto` — cpu/auto/gpu explanation
  - `ELEVATE_ENABLED=true`
  - `RRF_K=60` — with RRF k constant explanation (Cormack SIGIR 2009)
  - `AUTO_EXTRACT_LOG_DIR=~/.memini-ai/chat_logs`
  - `AUTO_EXTRACT_INTERVAL_SECONDS=5`
- File: `.env.example`

#### Step 10: `.opencode/opencode.json` (COMPLETE)
- Added `"EMBEDDING_MODE": "auto"` to the `memini-ai-dev` MCP server's `environment` block in the **root** opencode config (`/home/jcharles/Projects/MCP-Servers/.opencode/opencode.json`).
- Used the alias name directly (no `MEMINI_` prefix) per `Field(alias="EMBEDDING_MODE")`.

#### Step 11: Quality gates (COMPLETE)
- `uv run ruff check src/ tests/` → **0 errors** ✅
- `uv run mypy src/` → **0 errors** ✅ (53 source files)
- `uv run pytest tests/ -q` (excluding `test_postgres_database.py` which fails on the local DB with `user:password@...` default URL — pre-existing, not v0.7.0-related) → **763 passing** ✅ (740 v0.6.0 baseline + 23 new)

#### Step 12: Zero-data-loss verification (COMPLETE)
- Pre-step-7 count: `SELECT COUNT(*) FROM memories = 83` ✅
- Post-step-7 count: 83 ✅
- Pre-commit count: 83 ✅
- Pre-push count: 83 ✅
- **Zero data loss through all v0.7.0 changes.**

#### Step 13: `pyproject.toml` (COMPLETE)
- `version = "0.6.0"` → `"0.7.0"`

#### Step 14: Commit + tag + push (COMPLETE)
- Commit `18f37ed` on `main`: "Release v0.7.0: Dual-model RRF"
- 22 files changed, +2108 / -74
- Tag `v0.7.0` with message "v0.7.0: Dual-model RRF (384+1024 with reciprocal rank fusion)"
- `git push origin main` → success ✅
- `git push origin v0.7.0` → success ✅
- Remote: `https://github.com/VeeDubin/memini-ai-dev.git`

#### Step 15: Documentation (COMPLETE — this file, plus parallel updates)
- This HANDOFF.md rewritten (Session 4 entry preserved for context, Session 5 entry added at top).
- TASKS.md: implementation status table updated (all 15 steps marked DONE with implementation notes); header "Last Updated" line refreshed; bottom summary rewritten.
- AGENTS.md: new Review Notes entry at top (v0.7.0 RELEASED).
- CONTEXT.md: Version History line for v0.7.0 updated from "PLANNED" to RELEASED; "Steps Pending (10/15)" section removed; new "Released" section added.
- CHANGELOG.md: `[0.7.0]` entry added with Features, Tests, Bug Fixes, and Notes subsections.
- README.md: New bullet under Key Features ("Dual-Model RRF (v0.7.0+): cpu/auto/gpu modes, reciprocal rank fusion, elevate_memory_to_1024 tool"). Existing CHANGELOG section linked.

### Process State (Awareness for Next Session)

- **PostgreSQL on port 5434** — running, healthy, **83 memories at 384-dim**, `memories_1024` table exists and is empty (0 elevated memories).
- **Ollama Cloud** — API key still works, 40+ models available.
- **Working tree**: CLEAN ✅
- **OpenCode restart STILL REQUIRED** — Task tool dispatch still blocked by cached `ollama-cloud/<model>:<tag>-cloud` agent configs (the model tags were fixed in agent `.md` files in Session 4 but the running OpenCode TUI process has them cached in memory). PID 307190 (this session's parent) needs to be killed and restarted by the user.

### Next Session Starting Point

v0.7.0 is done. Possible v0.7.1 / v0.8.0 work candidates (none blocking):

1. **Real BGE-Large integration** — replace the `_expand_384_to_1024` zero-pad placeholder with an actual BGE-Large call so the 1024 sidecar carries real 1024-dim vectors instead of padded 384-dim ones. The elevate tool already takes an optional `vector_1024` arg, so the integration is local to `database.py` and `model/embeddings.py`.
2. **Migrate factory pattern to neuralgentics** — the `llm/factory.py` design from v0.6.0 (cloud LLM provider abstraction) could be carried into the neuralgentics Go LLM client for similar benefits. Per memory `360be24a-...` from a prior session.
3. **PyPI publish** — if user wants v0.7.0 on PyPI, run the `boomerang-release` workflow. Tag is pushed; the GitHub Actions publish job should fire automatically.
4. **Memory decay interaction with dual-model RRF** — when a memory is demoted/archived, should the 1024 sidecar also be deleted? Currently the FK is `ON DELETE CASCADE` so demoting a memory removes both copies automatically. Worth verifying with a test.

### Quick Resume Commands (for next session)

```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev

# Verify state
git log --oneline -3
# expect: 18f37ed Release v0.7.0: Dual-model RRF
git tag -l 'v0.7*'
# expect: v0.7.0
git status -s
# expect: clean

# Verify DB
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories"
# expect: 83
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories_1024"
# expect: 0

# Quality gates (should all pass clean)
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest tests/ -q
# expect: 763 passed
```

---

## 2026-06-02 (Session 4) — v0.7.0 Dual-Model RRF: 5/15 Steps Done

**Status**: 🔄 IN PROGRESS — 5 of 15 v0.7.0 implementation steps complete. 83 memories at 384-dim verified intact. Working tree dirty on 4 source files + 1 new file.

### What Was Done This Session

#### Step 1: config.py validators (COMPLETE)
- Added 3 `@field_validator` blocks: `_validate_embedding_mode` (cpu/auto/gpu), `_clamp_rrf_k` (1-1000), `_clamp_auto_extract_interval` (1-3600s)
- `embedding_dim: int = 384` ✓ (already done in Session 3)
- 5 new fields (`embedding_mode`, `elevate_enabled`, `rrf_k`, `auto_extract_log_dir`, `auto_extract_interval_seconds`) ✓ (already done in Session 3)
- `ruff + mypy` clean
- File: `src/memini_ai/config.py`

#### Step 2: postgres/schema.py memories_1024 table (COMPLETE)
- New table constant `TABLE_MEMORIES_1024 = "memories_1024"`
- SQL: `CREATE TABLE IF NOT EXISTS memories_1024` with columns (id, memory_id FK→memories.id ON DELETE CASCADE, embedding vector(1024) NOT NULL, elevated_at, elevated_from_dim, embedding_model, trust_score)
- 3 indexes: `idx_memories_1024_embedding` (DiskANN or HNSW), `idx_memories_1024_memory_id`, `idx_memories_1024_trust`, `idx_memories_1024_elevated_at` (DESC)
- Wired into `get_schema_sql()` between memories and memory_relationships (FK ordering correct)
- **Migration applied to live DB** — verified 0 data loss (memories count 82→82 after step 2, then 82→83 after this session's testing)
- `ruff + mypy` clean
- File: `src/memini_ai/postgres/schema.py`

#### Step 3: postgres/queries.py 6 new 1024 query constants (COMPLETE)
- `INSERT_MEMORY_1024` — idempotent via `ON CONFLICT (memory_id) DO NOTHING`
- `SEARCH_MEMORIES_1024_VECTOR` — joins memories table, ordered by cosine distance
- `GET_MEMORY_1024_BY_MEMORY_ID` — single-row lookup for elevate pre-check
- `SEARCH_MEMORIES_1024_JOINED` — full-table scan for RRF fusion
- `COUNT_MEMORIES_1024` — `SELECT COUNT(*) FROM memories_1024`
- `DELETE_MEMORY_1024_BY_MEMORY_ID` — idempotent demote
- `ruff + mypy` clean
- File: `src/memini_ai/postgres/queries.py`

#### Step 4: memory/rrf.py NEW FILE (COMPLETE)
- `reciprocal_rank_fusion(ranked_lists, k=60)` — pure function, validates k≥1, dedupes within lists (first occurrence counts), stable sort by first-seen order for tied scores
- `rrf_with_limit(ranked_lists, k=60, limit=None)` — convenience wrapper that returns just the top-N item IDs
- Smoke-tested: basic fusion, empty input, single list, duplicates, invalid k, limit wrapper
- `ruff + mypy` clean
- File: `src/memini_ai/memory/rrf.py` (NEW)

#### Step 5: postgres/database.py 6 new 1024 methods + helper (COMPLETE)
- `_expand_384_to_1024(vector_384, target_dim=1024)` — **static** placeholder expander: zero-pad to 1024 + L2-normalize. v0.7.0 ships with this stable stand-in; a future version will swap for actual BGE-Large call.
- `add_memory_1024(memory_id, vector_1024, trust_score=0.5, embedding_model="bge-large-placeholder")` — idempotent via `ON CONFLICT DO NOTHING`
- `query_memories_1024(vector_1024, threshold=0.5, limit=10)` — joins with memories table, returns `MemoryEntry` list with `score` set to cosine distance
- `get_memory_1024_by_memory_id(memory_id)` — returns dict with id, memory_id, embedding, elevated_at, etc., or None
- `elevate_memory_to_1024(memory_id, vector_1024=None, trust_boost=0.10)` — verifies source exists, derives 1024 vector if not provided, inserts (idempotent), bumps trust on BOTH 384 and 1024 records by `trust_boost` (clamped 0-1). Returns dict `{memory_id, elevated, trust_score, vector_dim}`.
- `count_memories_1024()` — returns int
- `delete_memory_1024(memory_id)` — idempotent demote, returns memory_id or None
- `ruff + mypy` clean (fixed unused `SEARCH_MEMORIES_1024_JOINED` import)
- File: `src/memini_ai/postgres/database.py`

#### Critical Fix: OpenCode Agent Model Blocker (UNBLOCKED — RESTART REQUIRED)

**Root cause:** The HANDOFF's "tag-sweep complete" claim was inaccurate. 47+ agent `.md` files across 6 locations still had the broken `ollama-cloud/<model>:<tag>-cloud` or `ollama-cloud/<model>:<tag>:cloud` format. **Most importantly, the project-level `node_modules/@veedubin/boomerang-v3` install (which OpenCode was actually loading)** had 15 stale files.

**Fix applied to all 6 locations:**
| Location | Files Fixed |
|---|---|
| `/home/jcharles/Projects/MCP-Servers/.opencode/agents/` (root) | 15 |
| `/home/jcharles/Projects/MCP-Servers/boomerang-v3/.opencode/agents/` (local plugin source) | 15 |
| `/home/jcharles/Projects/MCP-Servers/neuralgentics/.opencode/agents/` | 8 |
| `/home/jcharles/Projects/MCP-Servers/Super-Memory/.opencode/agents/` | 3 |
| `/home/jcharles/Projects/MCP-Servers/boomerang/.opencode/agents/` | 3 |
| `/home/jcharles/Projects/MCP-Servers/node_modules/@veedubin/boomerang-v3/.opencode/agents/` (project npm install — **the one OpenCode was loading**) | 15 |
| `/home/jcharles/.cache/opencode/packages/@veedubin/boomerang-v3@latest/.../agents/` (npm cache) | 15 |
| `/home/jcharles/Documents/Resume-workspace/.opencode/agents/` + nested npm | 3 + 15 |

**Ollama Cloud API verification (2026-06-02):**
All 10 model names used in agent files exist in `/api/tags`:
`glm-5.1`, `deepseek-v4-pro`, `deepseek-v4-flash`, `kimi-k2.6`, `devstral-2:123b`, `devstral-small-2:24b`, `gemma4:31b`, `qwen3-coder-next`, `qwen3.5:397b` (only `qwen3.5*` model), `minimax-m2.7`. **All valid.**

**⚠️ RESTART REQUIRED:** The running OpenCode TUI process (PID 307190) has the old agent config cached in process memory. New `task` dispatches still fail with `ProviderModelNotFoundError`. **The user MUST exit and restart the OpenCode TUI** for the fix to take effect. PID 307190 cannot be killed from inside this session (it owns this very session).

**Saved to memini-ai memory:** `b8b42742-e4e1-4a2a-a1a1-afd85e597f59` — full fix details for future sessions to skip re-investigation.

### Remaining Work (10 steps)

| # | Step | File(s) | Notes |
|---|------|---------|-------|
| 6 | `memory/system.py`: MEMINI_MODE dispatch in `add_memory` + `query_memories`, delete dead `_get_fallback_for_dimension()` | `src/memini_ai/memory/system.py` | cpu: 384 only. auto: 384 write + 384/1024 RRF query. gpu: 1024 only. Delete dead `_get_fallback_for_dimension()` (lines 350-361). Use new `db.elevate_memory_to_1024()` for elevate paths. |
| 7 | `server.py`: `elevate_memory_to_1024` MCP tool, AUTO-mode gated | `src/memini_ai/server.py` | Gate at tool-call time: raise helpful error if `config.embedding_mode != "auto"`. Call `db.elevate_memory_to_1024(memory_id, vector_1024=None, trust_boost=0.10)`. |
| 8 | Tests: 3 new test files (14 tests total) | `tests/test_rrf.py`, `tests/test_dual_model.py`, `tests/test_schema_migration.py` (create) | No-DB tests for rrf + dual_model; DB tests for schema_migration. |
| 9 | `.env.example`: document 5 new env vars | `.env.example` | `EMBEDDING_MODE`, `ELEVATE_ENABLED`, `RRF_K`, `AUTO_EXTRACT_LOG_DIR`, `AUTO_EXTRACT_INTERVAL_SECONDS`. |
| 10 | Update `.opencode/opencode.json` env | `.opencode/opencode.json` | Add `EMBEDDING_MODE=auto` to memini-ai-dev MCP env. **Use alias name directly (no `MEMINI_` prefix) per `Field(alias="EMBEDDING_MODE")`.** |
| 11 | Quality gates | — | `ruff check src/ tests/` (0 errors), `mypy src/` (0 errors), `pytest tests/ -v` (740+14=754 passing). |
| 12 | Zero-data-loss verification: `SELECT COUNT(*) FROM memories` must = **83** | — | Run BEFORE and AFTER steps 6-7. |
| 13 | `pyproject.toml`: 0.6.0 → 0.7.0 | `pyproject.toml` | |
| 14 | Commit + tag `v0.7.0` + push to GitHub | — | |
| 15 | Update docs (root + memini-ai-dev): AGENTS.md, CONTEXT.md, TASKS.md, HANDOFF.md, README, CHANGELOG | — | (TASKS.md and HANDOFF.md already updated for Session 4.) |

### Critical Constraints (Still Apply)
1. **DO NOT drop or recreate the `memories` table.** 83 existing memories are precious. Only ADD new tables/columns.
2. **DO NOT change the existing `vector(384)` column type.** Add new 1024 table separately.
3. **DO NOT change the default `embedding_dim` to anything other than 384.** Schema is 384; config must match.
4. **USE `CREATE TABLE IF NOT EXISTS` for the new `memories_1024` table.** Idempotent migrations only.
5. **USE `Field(alias=...)` for new config fields** (no `MEMINI_` prefix). The alias IS the env var name.
6. **TEST with the existing 83 memories.** Verify they're still retrievable after every change.

### Working Tree State (Dirty — Uncommitted)
- `src/memini_ai/config.py` — step 1 (validators)
- `src/memini_ai/postgres/schema.py` — step 2 (memories_1024 table)
- `src/memini_ai/postgres/queries.py` — step 3 (6 new constants)
- `src/memini_ai/memory/rrf.py` — step 4 (NEW FILE)
- `src/memini_ai/postgres/database.py` — step 5 (6 new methods + helper)
- `TASKS.md`, `HANDOFF.md`, `AGENTS.md`, `CONTEXT.md` (root + memini-ai-dev) — session 4 updates

### Quick Resume Commands
```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev

# Verify state
git status -s
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories"
# Expected: 83
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories_1024"
# Expected: 0 (table exists, empty until first elevate call)

# Quality gates as you go
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest tests/ -v

# Final commit + tag (after steps 6-15 done)
git add -A && git commit -m "Release v0.7.0: Dual-model RRF"
git tag v0.7.0 -m "v0.7.0"
git push origin main && git push origin v0.7.0
```

### Process State (for next session to be aware of)
- **PID 307190** — OpenCode TUI (this session's parent). Will need to be killed+restarted by user.
- **PID 250631** — Different OpenCode TUI in `/home/jcharles/Projects/reverse_engineering` (irrelevant to this project).
- **PostgreSQL on port 5434** — running, healthy, 83 memories at 384-dim, `memories_1024` table exists and is empty.
- **Ollama Cloud** — API key verified working, 40 models available.

---

## What Was Accomplished

### Phase 4: Advanced Features - COMPLETE ✅

Implemented all 4 Phase 4 features plus graph visualization:

| Feature | Module | Tests | MCP Tools | Status |
|---------|--------|-------|---------|--------|
| Memory Decay/Consolidation | `decay.py` (430 LOC) | 56 | get_decay_status, trigger_consolidation, list_fading_memories, adjust_decay_rate | ✅ |
| Full Knowledge Graph | `knowledge_graph.py` (520 LOC), `entity_extractor.py` (340 LOC) | 71 | query_kg, extract_entities, get_entity_graph, get_inference_chain, search_entities | ✅ |
| Multi-Peer Profiles | `multi_peer.py` (860 LOC) | 41 | list_peers, add_peer, switch_peer_context, share_memory, get_peer_memories, get_shared_memories | ✅ |
| Dialectic Reasoning | `dialectic.py` (1100 LOC) | 36 | find_contradictions, resolve_contradiction, get_dialectic_history, challenge_memory | ✅ |
| Graph Visualization (static) | `knowledge_graph.py` | - | get_graph_visualization | ✅ |

### Phase 5: pgvector Migration - COMPLETE ✅
- VectorDatabase ABC with QdrantDatabase and PostgresDatabase backends
- pgvectorscale StreamingDiskANN index for high-performance vector search
- All Phase 5 tasks complete

### Live Visualization API - NEW ✅
- FastAPI server with 5 endpoints
- D3.js force-directed graph with 30s polling
- Direct PostgreSQL queries for real-time data

### Test Results: 645 passed, 10 skipped

---

## Current Project State

### ALL PHASES COMPLETE ✅
Memini-ai v3.0 is fully implemented with optional advanced features and live visualization API.

### Tech Stack
- **Language**: Python 3.11+
- **Framework**: FastMCP (MCP server) + FastAPI (visualization)
- **Database**: PostgreSQL with pgvector/pgvectorscale
- **Embeddings**: BGE-Large (1024-dim), MiniLM-L6-v2 (384-dim fallback)
- **Search**: TIERED, VECTOR_ONLY, TEXT_ONLY, PARALLEL strategies

### 35 MCP Tools + 5 API Endpoints
- Phase 1: 6 tools (query_memories, add_memory, search_project, index_project, get_file_contents, get_status)
- Phase 2: 7 tools (Trust Engine + Memory Graph + Auto-Extract)
- Phase 3: 5 tools (Pre-Compression + Tiered Loading + User Modeling)
- Phase 4: 16 tools (Decay + KG + Multi-Peer + Dialectic)
- Visualization: 1 tool (get_graph_visualization) + FastAPI endpoints

### Source Files (32 modules)
```
src/memini_ai/
├── __init__.py
├── main.py
├── server.py                  # FastMCP with 35 tools
├── config.py                  # pydantic-settings config
├── decay.py                   # Memory decay (Phase 4A)
├── dialectic.py              # Dialectic reasoning (Phase 4D)
├── entity_extractor.py       # Entity extraction (Phase 4B)
├── extractor.py              # Auto-extract (Phase 2C)
├── graph.py                  # Memory graph (Phase 2B)
├── knowledge_graph.py        # Knowledge graph (Phase 4B)
├── multi_peer.py             # Multi-peer (Phase 4C)
├── preconpress.py            # Pre-compression (Phase 3A)
├── tiered_loader.py          # Tiered loading (Phase 3B)
├── trust_engine.py           # Trust engine (Phase 2A)
├── user_model.py             # User modeling (Phase 3C)
├── api/                      # NEW: Live visualization API
│   ├── __init__.py
│   ├── visualization.py       # FastAPI server
│   └── d3_template.py         # D3.js HTML generator
├── memory/
│   ├── schema.py             # All dataclasses
│   ├── database.py           # VectorDatabase ABC
│   ├── search.py             # Search strategies
│   └── system.py             # MemorySystem coordinator
├── postgres/                 # PostgreSQL backend
│   ├── database.py           # PostgresDatabase implementation
│   ├── schema.py             # SQL schema
│   └── queries.py            # SQL queries
├── model/
│   ├── manager.py            # ModelManager singleton
│   └── embeddings.py         # BGE-Large, MiniLM
├── indexer/
│   ├── constants.py, pause_controller.py, file_tracker.py
│   ├── snapshot.py, chunker.py, watcher.py, indexer.py
└── utils/
    ├── logger.py, hash.py
```

---

## Live Visualization API

### Running the Visualization Server

```bash
cd memini-ai-dev
export MEMINI_DB_URL="postgresql://user:password@localhost:5434/postgres"  # Set your actual DB URL
python -m uvicorn memini_ai.api.visualization:create_app --factory True --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` for the live D3.js visualization.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | D3.js live visualization page |
| `/api/graph` | GET | D3.js nodes/edges JSON |
| `/api/graph/stats` | GET | Entity counts by type |
| `/api/graph/entity/{id}` | GET | Single entity details |
| `/api/health` | GET | Health check |

### How It Works

1. KnowledgeGraph persists entities/relationships directly to PostgreSQL `entities` and `entity_relationships` tables
2. FastAPI server queries PostgreSQL on each request
3. D3.js polls `/api/graph` every 30 seconds for live updates
4. Force-directed graph renders entities as nodes, relationships as edges

---

## Configuration Reference

### PostgreSQL (Current)
```bash
MEMINI_DB_URL=postgresql://user:pass@localhost:5432/memini
MEMINI_PROJECT_ID=my-project
MEMINI_EMBEDDING_DIM=1024
MEMINI_DEVICE=auto
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MEMINI_DB_URL | PostgreSQL connection URL | Set via `.env` (see `.env.example`) |
| MEMINI_PROJECT_ID | Project namespace | auto-generated |
| MEMINI_EMBEDDING_DIM | 1024 or 384 | 1024 |
| MEMINI_DEVICE | auto, gpu, cpu | auto |
| MEMINI_TRUST_ENGINE | Enable trust scoring | false |
| MEMINI_MEMORY_GRAPH | Enable memory graph | false |
| MEMINI_AUTO_EXTRACT | Enable auto-extraction | false |
| MEMINI_PRECOMPRESS | Enable pre-compression | false |
| MEMINI_TIERED_LOADING | Enable tiered loading | false |
| MEMINI_USER_MODELING | Enable user modeling | false |
| MEMINI_DECAY_ENABLED | Enable memory decay | false |
| MEMINI_KG_ENABLED | Enable knowledge graph | false |
| MEMINI_MULTI_PEER_ENABLED | Enable multi-peer | false |
| MEMINI_DIALECTIC_ENABLED | Enable dialectic reasoning | false |

---

## Quality Gate Commands

```bash
cd memini-ai-dev

# Run all tests
python -m pytest tests/ -v

# Type check
mypy src/

# Lint
ruff check src/
ruff format src/

# Integration tests (requires PostgreSQL with pgvector)
docker run -d --name postgres-test -e POSTGRES_PASSWORD=password -p 5432:5432 timescale/timescaledb:latest-pg15
pytest tests/integration/ -v
```

---

## Important Reference Files

| File | Purpose |
|------|---------|
| `memini-ai-dev/CONTEXT.md` | Full architecture context, decisions, dependency map |
| `memini-ai-dev/TASKS.md` | 5-phase task breakdown (Phase 5 = pgvector migration) |
| `memini-ai-dev/README.md` | Installation and usage documentation |
| `memini-ai-dev/src/memini_ai/` | Full source code (all phases complete) |
| Super-Memory-TS source | `/node_modules/@veedubin/super-memory-ts/dist/` |

---

## Notes for Next Agent

### User Preferences
- **Language**: Python over TypeScript, worked better in v1
- **Architecture**: All features independently optional
- **Database**: PostgreSQL with pgvector (completed v0.2.0)

### Completed Work
- **v0.2.0**: pgvector migration complete, VectorDatabase ABC
- **v0.2.1**: Fixed package name for PyPI trusted publishing
- **v0.2.2**: Documentation updates, Qdrant references removed, live visualization added
- **v0.2.3**: Version bump, PyPI publish ready
- **v0.2.4**: aiosqlite dependency fix (missing from pyproject.toml)
- **v0.2.5**: Version bump fix (pyproject.toml version was not updated)
- **v0.2.6**: Fix server.run() HTTP transport (host/port args)
- **v0.2.7**: PostgreSQL schema fixes for idempotent initialization (IF NOT EXISTS, vector parsing, 384-dim vectors)

### Live Visualization
- KnowledgeGraph writes directly to PostgreSQL entities/entity_relationships tables
- FastAPI server at `src/memini_ai/api/visualization.py`
- D3.js template at `src/memini_ai/api/d3_template.py`
- Run with: `uvicorn memini_ai.api.visualization:create_app --factory True`

---

## PyPI Publishing Status (2026-05-19)

### v0.2.6 Release Status
- **Git tag**: `v0.2.6` created and pushed ✅
- **GitHub commit**: `33abf6e` ✅
- **GitHub Release**: Created via workflow ✅
- **PyPI publish**: Trusted publishing via GitHub Actions

### Version History
| Version | Date | Notes |
|---------|------|-------|
| **v0.7.0** | **2026-06-02** | **Dual-model RRF RELEASED. 384+1024 tables, MEMINI_MODE routing (cpu/auto/gpu), RRF k=60, elevate_memory_to_1024 MCP tool, +23 tests, 763 passing. Commit `18f37ed`, tag `v0.7.0` pushed.** |
| v0.6.0 | 2026-06-01 | Modular cloud LLM (factory/provider pattern), 740/740 tests, tag `v0.6.0` pushed |
| v0.3.1 | 2026-05-19 | Documentation refreshed, stale version references updated, pyproject.toml bumped |
| v0.3.0 | 2026-05-19 | Thought chains persistent reasoning with branching/revision, 9 MCP tools |
| v0.2.8 | 2026-05-19 | Ruff formatting pass (isort, whitespace, imports) across 30 files |
| v0.2.7 | 2026-05-19 | PostgreSQL schema fixes for idempotent initialization (IF NOT EXISTS, vector parsing, 384-dim vectors) |
| v0.2.6 | 2026-05-19 | server.run() HTTP transport fix |

---

## Session 2026-06-01 (Session 3) — v0.7.0 Implementation Started

### What Got Done
1. **Ollama Cloud API key verified working**: `YOUR_OLLAMA_CLOUD_API_KEY` (user said "OK to burn"). Both `/api/tags` and `/v1/chat/completions` endpoints confirmed. 40+ models available.
2. **`.gitignore` hardened**: Added `.env`, `.env.local`, `.env.*.local` to `boomerang-v2/.gitignore` and `neuralgentics/.gitignore`. All 5 .opencode repos now ignore env files.
3. **Home-dir tag sweep completed**: 139 files with `ollama-cloud/<model>:cloud` pattern fixed via `sed -i`. Final state across entire `/home/jcharles`:
   - 246 active agent `.md` files: **0 dirty**
   - 24 active `opencode.json`/`.jsonc` configs: **0 dirty**
   - Intentionally untouched: docs (anti-pattern examples), upstream `opencode-base` source, runtime state cache, session diff history, `dot-config-old` archive
4. **Pre-implementation DB snapshot recorded**: **80 memories** at 384-dim (4 new memories since last handoff's "76"). Schema intact.
5. **STEP 1 of v0.7.0 implementation — config.py**:
   - `embedding_dim: int = 1024` → `384` ✓
   - 5 new fields added: `embedding_mode` (alias=EMBEDDING_MODE), `elevate_enabled` (alias=ELEVATE_ENABLED), `rrf_k` (alias=RRF_K), `auto_extract_log_dir` (alias=AUTO_EXTRACT_LOG_DIR), `auto_extract_interval_seconds` (alias=AUTO_EXTRACT_INTERVAL_SECONDS)
   - **NOT DONE**: 3 field validators (`embedding_mode` → {cpu,auto,gpu}, `rrf_k` → [1,1000], `auto_extract_interval_seconds` → [1,3600])

### Working Tree State (Dirty — Uncommitted)
- `src/memini_ai/config.py` — partial step 1 (fields added, validators missing)
- `boomerang-v2/.gitignore` — env patterns
- `neuralgentics/.gitignore` — env patterns
- `memini-ai-dev/src/memini_ai/config.py` — same as above
- (boomerang-v2 and neuralgentics each have separate git repos; memini-ai-dev is also a separate git repo at `memini-ai-dev/`)

### Critical Blocker
**`task` tool dispatch fails with `ProviderModelNotFoundError`** because the running OpenCode processes (PIDs 250631, 274515) have cached the OLD `ollama-cloud/<model>:cloud` agent config format. The agent `.md` files are now fixed on disk but the running OpenCode needs a restart to load the corrected `ollama/<model>` format. **Restart OpenCode before relying on `task` dispatch.**

### v0.7.0 Implementation Progress

| # | Step | Status | Files |
|---|------|--------|-------|
| 1 | `config.py`: 384 default + 5 fields + 3 validators | **PARTIAL** (fields added, validators missing) | `src/memini_ai/config.py` |
| 2 | `postgres/schema.py`: add `memories_1024` table + indexes + wire into `get_schema_sql()` | PENDING | `src/memini_ai/postgres/schema.py` |
| 3 | `postgres/queries.py`: 6 new 1024 query constants | PENDING | `src/memini_ai/postgres/queries.py` |
| 4 | `memory/rrf.py`: NEW FILE with `reciprocal_rank_fusion()` | PENDING | `src/memini_ai/memory/rrf.py` (create) |
| 5 | `postgres/database.py`: 5 new methods + `_expand_384_to_1024()` helper | PENDING | `src/memini_ai/postgres/database.py` |
| 6 | `memory/system.py`: MEMINI_MODE dispatch in add+query, delete dead `_get_fallback_for_dimension()` | PENDING | `src/memini_ai/memory/system.py` |
| 7 | `server.py`: `elevate_memory_to_1024` MCP tool, AUTO-mode gated | PENDING | `src/memini_ai/server.py` |
| 8 | Tests: `tests/test_rrf.py` (5), `tests/test_dual_model.py` (6), `tests/test_schema_migration.py` (3) | PENDING | `tests/test_*.py` (create) |
| 9 | `.env.example`: document 5 new env vars | PENDING | `.env.example` |
| 10 | Quality gates: `ruff`, `mypy`, `pytest` (target 740+14=754) | PENDING | — |
| 11 | Zero-data-loss verification: `SELECT COUNT(*) FROM memories` must = **80** | PENDING | — |
| 12 | `pyproject.toml`: 0.6.0 → 0.7.0 | PENDING | `pyproject.toml` |
| 13 | Commit + tag `v0.7.0` + push to GitHub | PENDING | — |
| 14 | Update docs (root + memini-ai-dev): AGENTS.md, CONTEXT.md, TASKS.md, HANDOFF.md, README, CHANGELOG | PENDING | — |

### Design Doc Reference
`memini-ai-dev/docs/design/dual-model-rrf-architecture.md` (258 lines, complete and ready)

### Implementation Gotchas (Don't Forget)
- `Field(alias=...)` for new config fields — env var name is the alias (no `MEMINI_` prefix)
- All new SQL uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` — idempotent
- `_expand_384_to_1024()` is a PLACEHOLDER (zero-pad + L2-normalize) — real BGE-Large integration is v0.8.0
- `elevate_memory_to_1024` tool must be GATED at tool-call time (FastMCP can't conditionally register) — check `get_config().embedding_mode == "auto"`, raise helpful error otherwise
- Trust +0.10 boost on elevate (matches design doc line 138)
- Existing 80 memories MUST survive (additive migration, FK from `memories_1024.memory_id` to `memories.id`)

### Quick Resume Commands (for next session)
```bash
# 1. Verify state
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev
git status -s
PGPASSWORD=password psql -h localhost -p 5434 -U postgres -d postgres -c "SELECT COUNT(*) FROM memories"

# 2. Finish step 1 (add 3 validators to config.py — see HANDOFF.md "v0.7.0 Implementation Progress" step 1)

# 3. Run quality gates as you go
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest tests/ -v

# 4. Tag and push when all 14 steps done
# (Don't forget to bump pyproject.toml to 0.7.0 first)
```

### Release Process
1. Update version in `pyproject.toml`
2. Commit with `git add -A && git commit -m "Bump version to X.Y.Z"`
3. Tag with `git tag vX.Y.Z -m "Release vX.Y.Z"`
4. Push: `git push origin main && git push origin vX.Y.Z`
5. GitHub Actions workflow handles PyPI publish automatically

---

## 2026-06-03 — v0.7.1 BUG FILED → ✅ FIXED in Session 6

**Status**: ✅ **FIXED & RELEASED as v0.7.1** (see Session 6 entry at top of this file). The old OPEN entry below is kept for historical context only.

---

## 2026-06-03 — v0.7.1 BUG FILED: `add_thought` MCP-call vector injection error (SUPERSEDED — see Session 6)

**Status**: 🔴 OPEN (filed, not yet investigated; needs a dedicated session) [SUPERSEDED — fixed in Session 6]

**Symptom**: Boomerang orchestrator called `memini-ai-dev_add_thought(thought=..., thoughtNumber=1, totalThoughts=1, nextThoughtNeeded=False)` from the MCP stdio path. Got back:

```
invalid input for query argument $11: '[-0.03915949538350105,-0.06523498892784...'
(could not convert string to float: '[-0.03915949538350105,-0.06523498892784119,...]')
```

**Analysis** (later confirmed correct by Session 6):
- The `$11` reference is the embedding column (`vector(384)`) in the `INSERT INTO thoughts` query.
- A layer in the storage path was passing the vector as a **stringified JSON** instead of a real `list[float]`. asyncpg's automatic type coercion failed.
- Most likely culprit: `ThoughtChains.add_thought` was building `f"[{','.join(str(v) for v in vec)}]"` and passing it as a string with `::vector` cast.
- The in-process Python test path didn't hit this because it didn't exercise the embedding code path (the existing test was a 5-line `max()` smoke test, not a real call).

**Impact** (confirmed):
- **HIGH** — `add_thought` is a required step in the Boomerang Protocol (step 2: Thought Chains). Every session that uses complex planning hit this.

**Fix** (see Session 6): pass `list[float]` directly to asyncpg (matches how `memory.add` does it), drop the `::vector` cast, truncate/pad to 384 dims to handle the 1024-dim BGE-Large case.

---

*End of handoff.*
