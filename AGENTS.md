# Memini-ai Agent Context

> **Latest version**: memini-ai-dev **v1.0.3** (released 2026-07-16). v1.0.3 is a docs + lockfile-sync patch: HANDOFF/AGENTS/TASKS/CONTEXT catch-up to v1.0.2 reality, `uv.lock` regenerated to match the v1.0.2 `pyproject.toml` (was still at 1.0.0), and a new CRITICAL section in `AGENTS.md` documents the v1.0.0 `MEMINI_VECTOR_BACKEND` requirement. No code changes, no new env vars. The actual schema/code changes from prior releases: v0.7.7 (BGE-M3 opt-in), v0.7.8 (audit + doc rewrite), v0.7.9 (data-leak rule), v0.8.0 (image-recall RRF), v0.8.1 (CI re-trigger), v0.8.2 (detect-secrets security), v1.0.0 (embedded pgembed backend — MAJOR), v1.0.1 + v1.0.2 (migrate command fixes). See `CHANGELOG.md` for per-release detail and `HANDOFF.md` for session-by-session history.

## ⚠️ CRITICAL: `MEMINI_VECTOR_BACKEND` required when `MEMINI_DB_URL` is set (v1.0.0+)

**v1.0.0 changed the default backend from `postgres-external` to `pgembed` (in-process embedded PostgreSQL).** This is a breaking change for v0.8.x users who have `MEMINI_DB_URL` set in `.env` or `opencode.json` environment blocks.

**Symptom if missed**: `RuntimeError: memini-ai v1.0.0: MEMINI_DB_URL is set but MEMINI_VECTOR_BACKEND is not.` on MCP server start.

**Fix (one line)**: Add `export MEMINI_VECTOR_BACKEND=postgres-external` to the shell, the `.env`, OR the `opencode.json` `mcp.memini-ai-dev.environment` block. This preserves v0.8.x behavior exactly (connects to the external Postgres server). NO data migration needed.

**Three config locations to check (all three must agree)**:
1. `memini-ai-dev/.env` → `MEMINI_VECTOR_BACKEND=postgres-external`
2. `~/.config/opencode/opencode.json` → `mcp.memini-ai-dev.environment.MEMINI_VECTOR_BACKEND`
3. `MCP-Servers/.opencode/opencode.json` → same

**To migrate to embedded mode instead**: `unset MEMINI_DB_URL` then run `memini-ai migrate --from='<your old MEMINI_DB_URL>'`. The `memini-ai migrate` CLI was fixed in v1.0.1 + v1.0.2 — 6 bugs that prevented v1.0.0's command from working end-to-end are resolved.

**New env vars in v1.0.0**: `MEMINI_VECTOR_BACKEND` (pgembed|postgres-external), `MEMINI_PGEMBED_DATA_DIR` (default `~/.local/share/memini-ai/pgembed/data`), `MEMINI_TEAM_DB_URL` (optional, for RRF fusion with a team server), `MEMINI_FUSION_MODE` (rrf when team set).

## ⚠️ CRITICAL: Never Commit Memory Data (MUST FOLLOW)

**The memini-ai-dev repo is PUBLIC** (`github.com/Veedubin/memini-ai-dev`). Every commit is visible to the world.

**BEFORE `git add` of any new directory or `.dump` / `.jsonl` / `.sql` / `.csv` / `.parquet` / `.tar*` / `.zip` file, run:**

```bash
# Inspect file types — text/json/JSON-L/SQL/archive are suspicious
find <new_dir> -type f | xargs file | grep -iE "text|json|sql|archive"
# Check sizes — anything > 1MB is suspicious for source code
du -sh <new_dir>/*
# If any are data, exclude via .gitignore and STOP
```

**The .gitignore MUST include these patterns (already in place as of v0.7.8):**

```
*.dump
*.jsonl
archives/memini-migration-backup.jsonl
archives/memini-migration-to-bge-large-backup.jsonl
archives/memini-postgres-pre-migration.dump
```

**Background (2026-07-10, Session 42):** v0.7.8 commit initially contained 19MB of memory text + a 3.2MB PostgreSQL dump. User caught it: "Why are my memories being commited to a public repo?" Resolved before push, but the pattern (boomerang-coder moving a directory without inspecting contents) must not recur. The 3 safe files in `archives/` are the 2 migration `.py` scripts — those are source code and may be committed. **Everything else in `archives/` is data and must stay out of git.**

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
This is memini-ai-dev — a Python-based semantic memory server with PostgreSQL/pgvector backend. Key facts (as of v1.0.2, 2026-07-16):
- Language: Python 3.12+ (was 3.11+; pgembed 0.2.0 requires 3.12+)
- Framework: FastMCP (MCP server with **52 tools** — 35+ core + image-recall RRF arm + elevate + healthcheck + tier0/tier1)
- Database: PostgreSQL with pgvector + pgvectorscale + pg_textsearch
- **Backends (v1.0.0+)**: `pgembed` (default, in-process Postgres 17, no Docker) OR `postgres-external` (legacy v0.8.x Docker/team server, set `MEMINI_VECTOR_BACKEND=postgres-external`)
- Embeddings: MiniLM-L6-v2 (384-dim, default) / BGE-M3 (1024-dim, opt-in for new deployments, recommended for GPU). BGE-Large support was removed in v0.7.6.
- **Multi-modal RRF (v0.8.0+)**: when `MEMINI_IMAGE_SEARCH_ENABLED=true`, adds a 3rd RRF fan-out arm using CLIP over the `memories_image` table. Best-effort (text RRF proceeds with 2 lists on CLIP failure).
- Dual-model RRF: cpu/auto/gpu modes via `EMBEDDING_MODE` env, 1024-dim sidecar in `memories_1024` table, RRF k=60 (Cormack SIGIR 2009)
- CLI: `memini-ai init | status | stop | migrate [--dry-run] [--from=<url>]` (v1.0.0+, fixed in v1.0.1+v1.0.2)
- All features are independently optional via environment variables
- **Live DB state** (2026-07-16): 986 memories + 519 thoughts in `memini-postgres` (port 5434), all 13 tables present, 100% healthy

## Quality Gate Commands (copy-pasteable)
```bash
cd /home/jcharles/Projects/MCP-Servers/memini-ai-dev
ruff check src/
mypy src/
python -m pytest tests/ -v      # 809 passing baseline (v0.7.8+); 4 pre-existing env-var pollution failures documented
```

## Environment Variables (key ones)
| Variable | Description | Default |
|-----------|-------------|---------|
| `MEMINI_VECTOR_BACKEND` | `pgembed` (default, in-process) or `postgres-external` (v0.8.x compatible) — **REQUIRED if `MEMINI_DB_URL` is set** | `pgembed` |
| `MEMINI_DB_URL` | PostgreSQL connection (external mode only) | (unset = embedded mode) |
| `MEMINI_PGEMBED_DATA_DIR` | Embedded Postgres data directory (XDG-compliant) | `~/.local/share/memini-ai/pgembed/data` |
| `MEMINI_TEAM_DB_URL` | Optional team server for RRF fusion (writes go to primary only) | (unset) |
| `MEMINI_FUSION_MODE` | `rrf` when `MEMINI_TEAM_DB_URL` is set | (unset) |
| `MEMINI_EMBEDDING_DIM` | 384 or 1024 | 384 |
| `MEMINI_MODEL_NAME` | HF model ID or short alias (`bge-m3`, `minilm`) | MiniLM-L6-v2 |
| `MEMINI_EMBEDDING_MODE` | cpu / auto / gpu dispatch (v0.7.0+) | auto |
| `MEMINI_ELEVATE_ENABLED` | Enable `elevate_memory_to_1024` MCP tool (v0.7.0+) | true |
| `MEMINI_ENABLE_RRF` | Enable RRF across MiniLM + BGE-M3 columns | false |
| `MEMINI_AUTO_DETECT_MODEL` | New deployments with 0 memories auto-upgrade to BGE-M3 (v0.7.7+) | true |
| `MEMINI_STRICT_EMBEDDING_DIM` | Crash on dim mismatch instead of degrading (v0.7.7+) | false |
| `RRF_K` | Reciprocal Rank Fusion k constant (v0.7.0+) | 60 |
| `MEMINI_IMAGE_SEARCH_ENABLED` | Master gate for image-recall RRF arm (v0.8.0+) | false |
| `MEMINI_IMAGE_CLIP_MODEL` | `clip-ViT-B-32` or `clip-ViT-L-14` (v0.8.0+) | `clip-ViT-B-32` |
| `MEMINI_IMAGE_DIR` | Filesystem dir for stored images (v0.8.0+) | `~/.memini-ai/images` |
| `MEMINI_TRUST_ENGINE` | Enable trust scoring | false |
| `MEMINI_MEMORY_GRAPH` | Enable memory graph | false |
| `MEMINI_AUTO_EXTRACT` | Enable auto-extraction | false |
| `MEMINI_TIERED_LOADING` | Enable tiered loading | false |
| `MEMINI_KG_ENABLED` | Enable knowledge graph | false |
| `MEMINI_MULTI_PEER_ENABLED` | Enable multi-peer | false |
| `MEMINI_DIALECTIC_ENABLED` | Enable dialectic reasoning | false |
| `THOUGHT_CHAINS` | Enable persistent thought chains | false |
## Review Notes

- **2026-07-16 (Session 52, v1.0.3 release)**: **v1.0.3 DOCS + LOCKFILE SYNC PATCH RELEASED** ✅ — Patch release over v1.0.2 with no code changes. Three commits: `b88dd47` (docs catchup: HANDOFF/AGENTS/TASKS/CONTEXT were 9 versions stale at v0.7.6 / Session 40), `1c7d8ba` (uv.lock v1.0.2 sync — the v1.0.2 release had bumped pyproject.toml to 1.0.2 but didn't regenerate the lockfile, leaving it pinned at 1.0.0), `ff90815` (v1.0.3 bump). **What changed**: `AGENTS.md` (134→195 lines) gained the v1.0.3 banner + new CRITICAL section documenting the v1.0.0 `MEMINI_VECTOR_BACKEND` requirement + Project-Specific Context refresh (Python 3.12+, 52 tools, pgembed default, image-recall RRF, CLI commands) + env-var table expanded 14→24 rows. `HANDOFF.md` (978→1221 lines) + `TASKS.md` (751→932 lines) + `CONTEXT.md` (410→511 lines) all got the same 9-version backfill with per-release sections. New `CHANGELOG.md` `[1.0.3]` entry at the top. No code changes, no new env vars, no new dependencies. **Pre-commit detect-secrets hook passed on all 3 commits.** v1.0.3 tag pushed to origin; PyPI publish in progress via CI. Working tree clean. **Process lesson**: `bumpversion --patch --apply` should be followed by `uv lock` (currently not automatic), and the CHANGELOG entry should be edited BEFORE the bump, in the same commit, not after. v0.7.4 backlog items (text_only_search BM25, pre-existing test env-var pollution, OpenCode TUI restart framing) are all resolved in v0.7.7+.

- **2026-07-16 (Session 52)**: **v1.0.2 DB SERVER VERIFIED WORKING** ✅ — User asked "make sure the DB server still works" after v1.0.2 release. In-process `MCPServer.healthcheck()` returned `status=pass, readbackMatch=True, writeLatencyMs=2.9s, readLatencyMs=0.45ms`. `get_status`: `memoryCount=982, thoughtsCount=519, queryLatencyMs=0.67`. Live DB on port 5434 (memini-postgres, up 45h): 986 memories, 519 thoughts, all 13 tables present, 100% healthy. **Bug caught and fixed**: `.env` was missing the new `MEMINI_VECTOR_BACKEND` env var introduced in v1.0.0. v1.0.0 changed the default from `postgres-external` → `pgembed` and refuses to start if `MEMINI_DB_URL` is set without `MEMINI_VECTOR_BACKEND`. The MCP server itself was unaffected (root `opencode.json` already sets `MEMINI_VECTOR_BACKEND=postgres-external` in the environment block — that's authoritative for the MCP process), but any standalone script that imports `MCPServer`/`MemorySystem`/`create_database` would have crashed. **Fix applied**: added `MEMINI_VECTOR_BACKEND=postgres-external` to `.env` with a 2-line comment. No data touched, no commits needed. Git working tree clean on source files; `uv.lock` dirty (pre-existing from v1.0.2 release). Saved to memini-ai memory `7e943c67-e8b9-4243-8759-a7f026ee4fc0`.

- **2026-07-16 (Session 51)**: **v1.0.2 RELEASED** ✅ — `memini-ai migrate` CLI command had the same 6 bugs that v1.0.1 fixed in the standalone script. CLI brought to parity (`src/memini_ai/cli.py::_migrate()` +385/-63 LOC): `pg_restore` from pgembed's pg17 (not system pg18), pre-install `vector`+`vectorscale` extensions before restore, exclude `timescaledb`+`timescaledb_toolkit` from dump, post-restore verification (per-table row counts + random memory spot-check `text`+`embedding` + diskann index existence), `pg_restore` `check=False` with stderr error-line filtering, `--dry-run` flag. Commits `b050806` (fix) + `ad30e2c` (release). 100% backward compatible with v1.0.1.

- **2026-07-16 (Session 50)**: **v1.0.1 RELEASED** ✅ — `scripts/migrate_external_to_embedded.py` had 6 bugs that prevented v1.0.0's `memini-ai migrate` from working end-to-end. All 6 fixed: system `pg_dump` (must be >= source pg18) + pgembed's `pg_restore` (matches target pg17), `?host=` Unix socket URI parsing, pre-install extensions on target, exclude timescaledb from dump, sync (not async) `request_explicit_shutdown()`, spot-check column `text` not `content`. Added `--dry-run` flag, post-restore verification, better error messages, PGPASSWORD via subprocess env. Commits `63cfb8a` (fix) + `9b4d456` (release). Ruff + ast.parse clean. Companion release fix: 6 `opencode.json` files across the workspace had `MEMINI_DB_URL` set without `MEMINI_VECTOR_BACKEND` (would have broken MCP server on next TUI restart). All 6 fixed.

- **2026-07-16 (Sessions 48-49)**: **v1.0.0 EMBEDDED PGEMBED BACKEND RELEASED** ✅ **(MAJOR)** — Embedded PostgreSQL is now the default backend. The new `pgembed` driver starts an in-process Postgres 17 server on first query (no Docker required). v0.8.2 users with `MEMINI_DB_URL` set get a `RuntimeError` on startup with clear remediation (add `MEMINI_VECTOR_BACKEND=postgres-external`). Python 3.12+ required. `PostgresDatabase.__init__` now takes a `driver` parameter (internal; users go through unchanged `create_database()`). Data dir XDG-compliant (`~/.local/share/memini-ai/pgembed/data`). Driver pattern: `DatabaseDriver` Protocol + `EmbeddedPGDriver` + `ExternalPGDriver`. Multi-process server sharing (1 embedded Postgres shared by all memini-ai processes on same machine; 1s ping / 2s timeout / 5s drain grace heartbeat). RRF fusion across embedded + team server via `RRFDatabase` wrapper. CLI: `memini-ai init | status | stop | migrate`. 4 new env vars: `MEMINI_VECTOR_BACKEND`, `MEMINI_PGEMBED_DATA_DIR`, `MEMINI_TEAM_DB_URL`, `MEMINI_FUSION_MODE`. Design doc: `docs/design/v1.0.0-embedded-pgembed-architecture.md` (76KB). Commits `74b81cf` (feature) + `8c7b9f7` (release) + `c795931` (merge). **100% backward compatible with v0.8.2 if `MEMINI_VECTOR_BACKEND=postgres-external` is set**.

- **2026-07-13 (Session 47)**: **SECURITY INCIDENT — Ollama Cloud API key leaked in public Git history** 🚨 — Key `b319088f...` was found in the public Git history of `boomerang-v3`, `neuralgentics`, and `memini-ai-dev` (in `.opencode/opencode.json`, `scripts/install-boomerang.js`, `.env.example`, `HANDOFF.md`). **Response**: (1) Key rotated immediately. (2) `git-filter-repo` rewrote all commits in all 3 repos, replacing the secret with `YOUR_OLLAMA_CLOUD_API_KEY` placeholder. (3) Force-pushed rewritten `main` and all tags (this is the security exception that permits force-pushing public tags per the "Never Retag a Public Release" rule — the fix is a true removal of leaked material). (4) Cut new releases: boomerang-v3 v0.6.4, neuralgentics v0.12.4, memini-ai-dev v0.8.2. (5) Added `detect-secrets` baseline + pre-commit hooks + CI workflows (Session 46 / v0.8.2). **Prevention**: never paste a real secret into source/docs/tests. Use placeholders like `YOUR_OLLAMA_CLOUD_API_KEY`, `{env:OLLAMA_API_KEY}`, `sk-xxxxxxxx`. `detect-secrets` now runs on every push via pre-commit hook + GitHub Actions.

- **2026-07-13 (Session 46)**: **v0.8.2 SECURITY: detect-secrets BASELINE + CI** ✅ — Adds `detect-secrets` baseline + CI scan to prevent API key/secret leaks in commit history. Pre-commit hook + GitHub Actions workflow run `detect-secrets` on every push. 812 + 13 tests pass, ruff/mypy clean. Background: see Session 47 above (this session's work was the prevention mechanism; the actual leak was discovered + remediated same day). Commit `ed7e3ba` + release `7eed224`.

- **2026-07-13 (Session 45)**: **v0.8.1 CI RE-TRIGGER** ✅ — Pure CI re-trigger for `memini-vision>=0.1.0` dependency that wasn't on PyPI when v0.8.0 published. **No code changes from v0.8.0.** Original v0.8.0 tag preserved on origin (failed publish attempt). Commits `241e471` (v0.8.1 CHANGELOG entry) + `705fc36` (test trigger) + `52e8350` (v0.8.1 release).

- **2026-07-13 (Session 44)**: **v0.8.0 IMAGE-RECALL RRF FAN-OUT ARM RELEASED** ✅ — When `MEMINI_IMAGE_SEARCH_ENABLED=true`, `query_memories` adds a 3rd RRF fan-out arm that calls `memini-vision.ImageQuery.search_by_text` (CLIP text tower over the `memories_image` table) and fuses with the existing 384-dim MiniLM + 1024-dim BGE-M3 via unchanged `reciprocal_rank_fusion()` (k=60). Image arm is best-effort: any CLIP failure (model download, DB error) is caught, logged, and text RRF proceeds with 2 lists. `_query_dual_model_rrf` renamed to `_query_multi_model_rrf` (handles 2 OR 3 models). New `memories_image` table (migration `000008_add_memories_image.sql`): 768-dim CLIP image embeddings, 1:1 FK to `memories.id` ON DELETE CASCADE, `vector(768)` accommodates both ViT-B/32 (zero-padded) and ViT-L/14 (native). Created at memini-ai startup REGARDLESS of whether image search enabled (so videre-mcp can write without coordination). `source_type='image'` added to CHECK constraint. 5 new env vars: `MEMINI_IMAGE_SEARCH_ENABLED` (default `false`), `MEMINI_IMAGE_CLIP_MODEL`, `MEMINI_IMAGE_CLIP_DEVICE`, `MEMINI_IMAGE_DIR`, `MEMINI_IMAGE_DB_URL`. `[vision]` optional dep: `vision = ["memini-vision>=0.1.0"]`. **Text-only users see ZERO behavior change.** 799 tests pass, 3 skipped. Design doc: `docs/design/vision-memory-architecture.md` (30KB). Commits `25eb3aa` (design) + `15ad805` (impl).

- **2026-07-11/12 (Session 43)**: **v0.7.9 DATA-LEAK RULE FOLLOWUP** ✅ — Adds critical "Never Commit Memory Data" rule to `AGENTS.md` as follow-up to the v0.7.8 near-miss (19MB memory text + 3.2MB pg_dump almost committed). Pre-commit inspection pattern: `find <dir> -type f | xargs file | grep -iE "text|json|sql|archive"` and `du -sh <dir>/*` before `git add`. `.gitignore` now includes `*.dump`, `*.jsonl`, `archives/memini-migration-backup.jsonl`, `archives/memini-migration-to-bge-large-backup.jsonl`, `archives/memini-postgres-pre-migration.dump`. `uv.lock` refresh to match v0.7.8 `pyproject.toml`. 809 tests still pass, 3 skipped. Commits `2c71c2a` (rule) + `cb6fe6b` (v0.7.9 release).

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
| `CHANGELOG.md` | **Canonical** per-release notes (v0.2.0 → v1.0.2) — read this first for "what shipped" |
| `TASKS.md` | 5-phase task breakdown + per-release implementation status + backlog |
| `HANDOFF.md` | Session-by-session handoff notes (Sessions 1-52) |
| `CONTEXT.md` | Architecture decisions + version history table |
| `README.md` | Installation and usage (rewritten in v0.7.8 to fix CRITICAL doc bug) |
| `docs/upgrading-embeddings.md` | MiniLM → BGE-M3 migration recipe (v0.7.7+) |
| `docs/design/dual-model-rrf-architecture.md` | v0.7.0 384+1024 RRF design |
| `docs/design/vision-memory-architecture.md` | v0.8.0 image-recall RRF design (30KB) |
| `docs/design/v1.0.0-embedded-pgembed-architecture.md` | v1.0.0 embedded pgembed design (76KB) |
| `docs/audits/v0.7.7-audit.md` | v0.7.7 8-area audit (225 lines, 13 findings) |
| `docs/audits/v0.7.7-validation.md` | v0.7.7 audit validation (135 lines, 70 probes) |
| `archives/memini-embedding-migration-2026-07-10/` | BGE-Large + MiniLM→BGE-M3 migration scripts (kept as reference) |
| `.env.example` | All env vars documented (added v0.7.7 section in v0.7.8 fix) |
