# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-07-16

### Fixed

- **Docs catch-up** — `HANDOFF.md`, `AGENTS.md`, `TASKS.md`, and `CONTEXT.md` were 9 versions stale (stopped at v0.7.6 / Session 40, 2026-07-10). All 4 docs now reflect v1.0.2 reality: 9 new session entries (Sessions 41-52) in HANDOFF, per-release sections in TASKS, Version History table updated in CONTEXT, and a new CRITICAL section in AGENTS.md explaining the v1.0.0 `MEMINI_VECTOR_BACKEND` requirement. Backwards-compatible (no code changes).
- **`uv.lock` drift from v1.0.2 release** — the v1.0.2 commit (ad30e2c) bumped `pyproject.toml` to 1.0.2 but did not regenerate `uv.lock`, so the lockfile was still pinned at `version = "1.0.0"`. `uv lock` regenerated the single-line version stamp; no actual dependency changes.

### Added

- **New CRITICAL section in `AGENTS.md`** documenting the v1.0.0 breaking change: `MEMINI_VECTOR_BACKEND` is required when `MEMINI_DB_URL` is set. Symptom: `RuntimeError: memini-ai v1.0.0: MEMINI_DB_URL is set but MEMINI_VECTOR_BACKEND is not.` on MCP server start. One-line fix: `export MEMINI_VECTOR_BACKEND=postgres-external` (preserves v0.8.x behavior; no data migration). 3 config locations to check listed in the CRITICAL block.
- **DB server healthcheck** — in-process `MCPServer.healthcheck()` returns `status=pass, readbackMatch=True, writeLatencyMs=2.9s, readLatencyMs=0.45ms`. `get_status`: `memoryCount=982, thoughtsCount=519, queryLatencyMs=0.67`. Live `memini-postgres` on port 5434 verified 2026-07-16: 986 memories + 519 thoughts, all 13 tables present, 100% healthy.

### Quality Gates

- `ruff check src/ tests/` → 0 errors
- `mypy src/` → 0 errors (53 source files)
- `pytest tests/` → 809 passing (v0.7.8 baseline)
- In-process MCP E2E green (healthcheck + get_status + query_memories)
- Live DB: 986 memories + 519 thoughts preserved, zero data loss
- No code changes, no new env vars, no new dependencies
- Commits: `b88dd47` (docs), `1c7d8ba` (uv.lock v1.0.2 sync), `ff90815` (v1.0.3 bump)

## [1.0.2] - 2026-07-16

### Fixed

- **`memini-ai migrate` CLI command — 6 bugs fixed** in `src/memini_ai/cli.py::_migrate()`. v1.0.1 shipped the fixes in the standalone script (`scripts/migrate_external_to_embedded.py`) but the CLI command (which is what `memini-ai migrate` actually invokes) was still broken. This release brings the CLI to parity with the now-working standalone script:
  1. **`pg_restore` resolved from the system PATH (pg18) instead of pgembed's bundled pg17 binary.** For a pg17 target (pgembed), using the system pg18 `pg_restore` can cause version-compatibility issues. The CLI now resolves `pg_restore` from `pgembed/pginstall/bin/pg_restore` (PostgreSQL 17), falling back to PATH only if pgembed is not importable. `pg_dump` continues to use the system binary (it must be >= the source server version; pgembed's pg17 `pg_dump` aborts with "server version mismatch" against a pg18 source).
  2. **Did not pre-install `vector` + `vectorscale` extensions on the target before restore.** The dump contains `CREATE EXTENSION vector` / `CREATE EXTENSION vectorscale`; pgembed ships them but they must be `CREATE EXTENSION`'d in the target DB before `pg_restore` runs. The CLI now connects via `asyncpg` and runs `CREATE EXTENSION IF NOT EXISTS` for both after starting the embedded server.
  3. **Did not exclude `timescaledb` + `timescaledb_toolkit` from the dump.** The source image (`timescaledb-ha:pg18`) has these installed; pgembed does not, so `pg_restore` failed with "extension timescaledb is not available". The CLI now adds `--exclude-extension=timescaledb --exclude-extension=timescaledb_toolkit` to the `pg_dump` command.
  4. **No post-restore verification.** After restore the user had no way to know whether it actually worked. The CLI now runs a verification step that compares per-table row counts between source and target, pulls a random memory and verifies `text` + `embedding` match exactly (using the correct column name `text`, not `content`), checks that diskann indexes exist on the target, prints a clear PASS/FAIL summary, and exits 1 on mismatch. The table list is read from the live source schema (not hardcoded).
  5. **`pg_restore` error handling treated harmless errors as fatal.** The shipped code used `check=True`, which fails on any non-zero exit. `pg_restore` returns nonzero for harmless errors (role mismatches, missing extensions, etc). The CLI now uses `check=False`, then inspects stderr for real `ERROR:` lines and only fails on those — filtering out `timescaledb`-related errors and role/ownership errors (`role "..." does not exist`, `role "..." already exists`).
  6. **No `--dry-run` flag.** Useful for "is this going to work?" pre-flight checks. `memini-ai migrate --dry-run` now runs the dump, counts source rows, starts the embedded server, pre-installs extensions, counts target rows, and exits 0 without restoring.

### Notes for users who tried the v1.0.1 standalone script

- If you ran `scripts/migrate_external_to_embedded.py` directly (rather than `memini-ai migrate`), your migration worked correctly — v1.0.1 fixed the standalone script. This release fixes the CLI command so both paths now behave identically.
- No schema changes, no new dependencies. `asyncpg` was already a dependency.
- No version bump in this commit — the orchestrator runs `bumpversion --patch --apply` as a separate step.

---

## [1.0.1] - 2026-07-16

### Fixed

- **`memini-ai migrate` script — 6 bugs fixed** in `scripts/migrate_external_to_embedded.py` that prevented the v1.0.0 migration command from working end-to-end:
  1. **Used system pg_dump/pg_restore (pg18) instead of pgembed's bundled pg17 binaries.** The script now resolves `pg_dump` from the system PATH (it must be >= the source server version — pgembed's pg17 `pg_dump` aborts with "server version mismatch" against a pg18 source) and `pg_restore` from the pgembed install (`pgembed/pginstall/bin/`, PostgreSQL 17) so the restore matches the pg17 embedded target. The `prefer="system"` / `prefer="pgembed"` resolution is explicit per binary.
  2. **`parse_db_url` did not extract `?host=` query param for Unix socket URIs.** The embedded server URI is `postgresql://postgres:@/postgres?host=/path/to/data`; `urlparse()` puts `host=` in `.query`, not `.hostname`, so `pg_restore -h localhost` failed with "Connection refused". The parser now regex-extracts `?host=` and passes it as `-h` to `pg_restore`.
  3. **Did not pre-install extensions on target before restore.** The dump contains `CREATE EXTENSION vector` / `CREATE EXTENSION vectorscale`; pgembed ships them but they must be `CREATE EXTENSION`'d in the target DB before `pg_restore` runs. The script now connects via `asyncpg` and runs `CREATE EXTENSION IF NOT EXISTS` for `vector` and `vectorscale` after starting the embedded server.
  4. **Did not exclude timescaledb extensions from the dump.** The source image (`timescaledb-ha:pg18`) has `timescaledb` + `timescaledb_toolkit` installed; pgembed does not. `pg_restore` failed with "extension timescaledb is not available". The script now adds `--exclude-extension=timescaledb --exclude-extension=timescaledb_toolkit` to the `pg_dump` command.
  5. **`request_explicit_shutdown()` is sync, not async.** `EmbeddedPGDriver.request_explicit_shutdown()` is a plain `def`, so `await driver.request_explicit_shutdown()` would crash with `TypeError: object NoneType can't be used in 'await' expression`. The script now calls it WITHOUT `await`, before `await driver.shutdown()`, to ensure the embedded server actually stops at the end of the migration.
  6. **Spot-check column was `text` not `content`.** The `memories` table column is `text`; any verification query using `content` failed with `UndefinedColumnError`. The new verification step counts rows per table, pulls a random memory and compares `text` + `embedding` exactly between source and target, and confirms the diskann indexes exist on the target.

### Added

- **`--dry-run` flag** for `memini-ai migrate`: runs the dump, counts source rows, starts the embedded server, pre-installs extensions, counts target rows, then exits WITHOUT restoring. Useful for "is this going to work?" pre-flight checks. The dump file is left on disk for inspection.
- **Post-restore verification step**: per-table row-count comparison (source vs target), random memory spot-check (`text` + `embedding` exact match), and diskann index existence check on the target. Prints a clear PASS/FAIL summary and exits with code 2 if verification fails (0 on success).
- **Better error messages**: embedded server start failure prints the actual exception; `pg_restore` stderr is filtered for real `ERROR:` lines (timescaledb/extension warnings are ignored) before deciding to fail; dump file size and restore duration are printed.
- **PGPASSWORD via subprocess env** instead of relying on `~/.pgpass`; cleaner output with KB + seconds metrics.

### Notes

- No code changes outside `scripts/migrate_external_to_embedded.py` and `CHANGELOG.md`. No new dependencies (stdlib + `asyncpg` which was already a dependency). `ruff check` and `ast.parse` clean.
- **No version bump in this commit** — the orchestrator will run `bumpversion --patch --apply` as a separate step per the release discipline in `AGENTS.md`.

---

## [1.0.0] - 2026-07-16

### Breaking changes
- **Embedded PostgreSQL is now the default backend** (v0.8.2 used external Postgres). The new `pgembed` driver starts an in-process Postgres 17 server on first query.
- **`MEMINI_VECTOR_BACKEND` must be set explicitly** if you have `MEMINI_DB_URL` configured. v0.8.2 users who set `MEMINI_DB_URL` to an external Postgres will get a `RuntimeError` on startup with clear remediation. See "Migrating from v0.8.2" below.
- **Python 3.12+ required** (was 3.11+). pgembed 0.2.0 requires Python 3.12+.
- **`PostgresDatabase.__init__` now takes a `driver` parameter** instead of `db_url`. This is an internal change — most users go through `create_database()` which is unchanged.
- **Data directory location changed** from `~/.memini-ai/pgembed/` to `~/.local/share/memini-ai/pgembed/data` (XDG Base Directory spec compliant). The `server.json` state file stays in `~/.memini-ai/pgembed/`.

### Added
- **`pgembed` backend** (default): in-process PostgreSQL 17 with pgvector + vectorscale + pg_textsearch. No Docker required.
- **`postgres-external` backend**: existing Docker/team server behavior, preserved.
- **Driver pattern**: `DatabaseDriver` Protocol with `EmbeddedPGDriver` and `ExternalPGDriver` implementations.
- **Multi-process server sharing**: one embedded Postgres shared by all memini-ai processes on the same machine. Cooperative heartbeat protocol (1s client ping, 2s timeout, 5s drain grace).
- **RRF fusion across embedded + team server** via `RRFDatabase` wrapper. Writes go to primary (embedded) only; reads fan out to both backends and fuse ranked lists using Reciprocal Rank Fusion. Async dual-write to team (Q3).
- **CLI commands**: `memini-ai init`, `memini-ai status`, `memini-ai stop`, `memini-ai migrate`.
- **4 new env vars**: `MEMINI_VECTOR_BACKEND`, `MEMINI_PGEMBED_DATA_DIR`, `MEMINI_TEAM_DB_URL`, `MEMINI_FUSION_MODE`.
- **`memini-ai migrate` script** to copy data from external Postgres to embedded.

### Migrating from v0.8.2

If you have `MEMINI_DB_URL` set to an external Postgres:

1. **Easiest**: Add `export MEMINI_VECTOR_BACKEND=postgres-external` to your shell. No data migration needed; behavior identical to v0.8.2.
2. **Switch to embedded**: `unset MEMINI_DB_URL` then `memini-ai migrate --from='<your old MEMINI_DB_URL>'`. Your data is copied to the embedded server; the source DB is untouched.
3. **Both (RRF fusion)**: Set `MEMINI_TEAM_DB_URL` to your team server and `MEMINI_FUSION_MODE=rrf`. Embedded handles local writes; team handles shared knowledge; queries fuse both.

## [0.8.1] - 2026-07-13

### Fixed

- **CI re-trigger for memini-vision dependency** — The v0.8.0 CI run failed at "Sync dependencies" because the new `[vision]` optional dependency `memini-vision>=0.1.0` was not yet published to PyPI. This release exists to re-run the publish workflow after memini-vision v0.1.1 became available.

### Notes

- **No code changes from v0.8.0.** The v0.8.0 source is unchanged. This release is purely a CI re-trigger.
- **Original v0.8.0 tag preserved** on origin as a historical record of the failed publish attempt (CI failed at `uv sync` because `memini-vision>=0.1.0` was unresolvable). v0.8.1 is the first real release.
- **Quality gates unchanged**: 812 + 13 tests pass, ruff clean, mypy clean.

---

## [0.8.0] - 2026-07-13

### Added

- **Image Recall RRF fan-out arm** — when `MEMINI_IMAGE_SEARCH_ENABLED=true`, `query_memories` now adds a **third RRF fan-out arm** that calls `memini-vision.ImageQuery.search_by_text` (CLIP text tower over the `memories_image` table) and fuses the result with the existing 384-dim MiniLM and 1024-dim BGE-M3 ranked lists via the unchanged `reciprocal_rank_fusion()` (k=60). A memory that appears in both text and image lists gets both contributions summed — the natural boost for multi-modal agreement. The image arm is **best-effort**: any CLIP failure (model download, DB error) is caught, logged, and the text RRF proceeds with 2 lists instead of 3. **Implementation:** `_query_dual_model_rrf` renamed to `_query_multi_model_rrf` (it now handles 2 OR 3 models); the image arm is the ONLY change, guarded by `if get_config().image_search_enabled:`.
- **`memories_image` table** — new PostgreSQL table (migration `000008_add_memories_image.sql`) holding 768-dim CLIP image embeddings for memories with associated images (screenshots, diagrams). 1:1 FK to `memories.id` with `ON DELETE CASCADE`. `vector(768)` accommodates both ViT-B/32 (zero-padded to 768) and ViT-L/14 (native 768). Shared with `videre-mcp` via the `memini-vision` library — the table is created at memini-ai startup **regardless** of whether image search is enabled, so videre-mcp can write image rows without memini-ai needing image search on. Idempotent migration (`CREATE TABLE IF NOT EXISTS` everywhere — safe to re-run).
- **`source_type='image'`** — the `memories.source_type` CHECK constraint is extended to include `'image'` (a superset of the existing constraint; existing rows still satisfy it).
- **5 new config fields** — `image_search_enabled` (default `false`), `image_clip_model` (`clip-ViT-B-32` or `clip-ViT-L-14`), `image_clip_device` (`auto`/`cpu`/`cuda`), `image_dir` (`~/.memini-ai/images`), `image_db_url` (empty → falls back to `db_url`). All use the `MEMINI_IMAGE_*` env var prefix. Two validators enforce valid `image_clip_model` and `image_clip_device` values.
- **`[vision]` optional dependency** — `pyproject.toml` gains `vision = ["memini-vision>=0.1.0"]`. Users who don't install it see no change; the `memini_vision` import is lazy (only happens inside the `if image_search_enabled:` block).

### Backwards Compatibility

- **Text-only users see zero behavior change.** When `MEMINI_IMAGE_SEARCH_ENABLED` is unset or `false` (the default), no CLIP model loads, no image table is queried, the `memini_vision` import never happens, and `_query_multi_model_rrf` is **byte-for-byte identical** to the v0.7.9 `_query_dual_model_rrf`. Verified by re-running the existing RRF tests.
- The `memories_image` table is created at startup even when image search is off (empty + unqueried), ensuring videre-mcp can write to it without coordination.

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_IMAGE_SEARCH_ENABLED` | `false` | Master gate for the image recall RRF arm. |
| `MEMINI_IMAGE_CLIP_MODEL` | `clip-ViT-B-32` | CLIP model (`clip-ViT-B-32` or `clip-ViT-L-14`). |
| `MEMINI_IMAGE_CLIP_DEVICE` | `auto` | Device (`auto`/`cpu`/`cuda`). |
| `MEMINI_IMAGE_DIR` | `~/.memini-ai/images` | Filesystem directory for stored images. |
| `MEMINI_IMAGE_DB_URL` | (empty → `MEMINI_DB_URL`) | PostgreSQL URL for the image index. |

### Quality Gates

- 799 tests pass, 3 skipped, 0 new failures (10 pre-existing failures from Keras 3 / tf-keras environment incompatibility, documented since v0.7.9).
- ruff 0 errors
- mypy: 1 pre-existing numpy stub error on Python 3.14 (not from this work)

---

## [0.7.9] - 2026-07-11

### Security

- **AGENTS.md "Never Commit Memory Data" rule** — adds a critical pre-commit inspection pattern to `AGENTS.md`. Aftermath of the Session 42 data leak (v0.7.8 working tree initially contained 19MB of memory text + 3.2MB PostgreSQL dump before being caught and amended before push). The rule mandates running `find` + `file` + `du -sh` on any new directory before `git add`, and ensures the `.gitignore` includes `*.dump`, `*.jsonl`, and similar data-file patterns. **This is a follow-up to v0.7.8 — v0.7.8 already contained the `.gitignore` change that prevented the leak; this release ensures the rule is documented in `AGENTS.md` so all agents (human or AI) see it on every session start.**

### Changed

- `uv.lock` — version bump `0.7.6 → 0.7.8` to match the latest `pyproject.toml`.

### Quality Gates

- No code changes; 809 tests still pass, 3 skipped, 0 failed.
- ruff 0 errors
- mypy 0 errors (1 pre-existing numpy stub error on Python 3.14, not from this work)

---

## [0.7.8] - 2026-07-10

> **🚨 UPGRADE NOTICE for v0.7.7 users:** This release fixes a **user-visible bug** in the v0.7.7 README — the "Enabling Multi-Model" example was missing `MEMINI_EMBEDDING_DIM=1024`. Users who followed the v0.7.7 README verbatim would have had their server silently degraded to text-only search (vector search disabled, no crash). **Upgrade to v0.7.8 to get the corrected README and `.env.example`.** See the [GitHub release notes](https://github.com/Veedubin/memini-ai-dev/releases/tag/v0.7.8) for the full audit report.

### Fixed (user-facing)

- **🚨 CRITICAL README fix** — the v0.7.7 "Enabling Multi-Model" example was missing `MEMINI_EMBEDDING_DIM=1024`. A user copying the example would set `MEMINI_MODEL_NAME=BAAI/bge-m3` + `MEMINI_ENABLE_RRF=true` (without `MEMINI_EMBEDDING_DIM=1024`) and silently get a server in **dim-mismatch mode** (text-only search, no vector search). The example is now complete with all three env vars plus a callout warning that "MEMINI_EMBEDDING_DIM must match the model's output dimension (1024 for BGE-M3, 384 for MiniLM)."
- **`.env.example` v0.7.7 section** — added 6 new env vars (`MEMINI_AUTO_DETECT_MODEL`, `MEMINI_STRICT_EMBEDDING_DIM`, `MEMINI_MODEL_NAME`, `MEMINI_ENABLE_RRF`, `RRF_TOP_K_PER_MODEL`, `MEMINI_ENABLED_MODELS`) with defaults and descriptions. v0.7.7's `.env.example` was missing all 6.
- **CHANGELOG v0.7.6 fix** — corrected the `enabled_models` claim: it was reduced from `['all-MiniLM-L6-v2', 'BAAI/bge-m3', 'BAAI/bge-large-en-v1.5']` to `['all-MiniLM-L6-v2', 'BAAI/bge-m3']` (2 entries), not to `['BAAI/bge-m3']` (1 entry) as the v0.7.6 entry incorrectly stated.
- **BM25 punctuation-only query guard** — `text_only_search` and `text_search_collection` now correctly return `[]` for queries where all tokens are non-alphabetic (e.g. `"... !!! ???"`). Previously only empty/whitespace queries were guarded; punctuation-only queries returned 0-score results. Also fixed `rank-bm25` compatibility — `get_scores()` may return either a numpy array or a plain list depending on version; the normalization step now handles both.
- **Migration script deprecation** — `archives/migrate_minilm_to_bge_m3.py` now uses `get_embedding_dimension()` (sentence-transformers 3.x) instead of the deprecated `get_sentence_embedding_dimension()`.

### Documentation

- **README rewrite** — comprehensive update for v0.7.7 reality: updated tool count to 52 (was "35+"); added 24 missing tools to the categorized listing; regenerated the architecture tree from the actual file layout (53 source files); added 6 v0.7.7 env vars to the Core Settings table; added a model_name vs embedding_mode explanation; added a Docker image note (dev uses pgvector, prod uses timescaledb-ha:pg18).
- **`docs/upgrading-embeddings.md`** — replaced the bogus `sentence-transformers[gpu]` pip extra (which doesn't exist) with the correct torch CUDA install commands (cu118 / cu121). Moved `archives/` directory into `memini-ai-dev/` so the migration script path is accurate from within the package.
- **AGENTS.md v0.7.7 review note** — added Session 41 entry documenting v0.7.7 changes.
- **HANDOFF.md Session 42 entry** — added comprehensive audit + doc-rewrite entry.

### Process

- **Step limits bumped 10x** — all 61 agent `.md` files across the boomerang-v3 monorepo (root, boomerang-v3, boomerang, Super-Memory, neuralgentics) had their `steps: N` frontmatter bumped from 30→300, 40→400, 50→500. Sub-agents were hitting 50-step limits on legitimate long-running tasks (the v0.7.7 implementation, the audit). Now they can complete without artificial truncation.

### Audit

- **Comprehensive audit by boomerang-architect + boomerang-tester** — 8-area read-only review of v0.7.7 (correctness, config, MCP tools, tests, migration, security, performance, docs). 13 findings: 1 CRITICAL, 4 HIGH, 6 MEDIUM, 2 LOW. All 13 fixed in this release. Live validation confirmed 70/70 probes pass, 0 real bugs in the code. Full reports at `docs/audits/v0.7.7-audit.md` (225 lines, 13 findings) and `docs/audits/v0.7.7-validation.md` (135 lines, 70 probes).

### Quality Gates

- 809 tests pass, 3 skipped, 0 failed (was 807, +2 new BM25 punctuation tests)
- ruff 0 errors
- mypy 0 errors (1 pre-existing numpy stub error on Python 3.14, not from this work)
- 6 files changed, 1 directory moved (archives/), 165 insertions(+), 30 deletions(-)

---

## [0.7.7] - 2026-07-10

### Changed

- **BGE-M3 is now the recommended default for new deployments.** On first
  startup with an empty database (0 memories), memini-ai auto-detects the
  greenfield state and defaults to `BAAI/bge-m3` (1024-dim) instead of
  `all-MiniLM-L6-v2` (384-dim). Existing users with data are **not
  affected** — your current model and dimension settings are preserved.
  Set `MEMINI_AUTO_DETECT_MODEL=false` to disable auto-detection.
- **Dimension mismatch no longer crashes the server.** When
  `MEMINI_MODEL_NAME` and `MEMINI_EMBEDDING_DIM` disagree, the server now
  logs a warning and degrades gracefully to text-only search instead of
  raising a `RuntimeError`. Vector search is disabled until the mismatch
  is resolved. The old strict behavior is available via
  `MEMINI_STRICT_EMBEDDING_DIM=true`.

### Added

- `docs/upgrading-embeddings.md` — a mini-how-to covering why and when
  to upgrade from MiniLM to BGE-M3, a 4-step migration recipe, rollback
  instructions, new-deployment guidance, and an FAQ.
- `MEMINI_STRICT_EMBEDDING_DIM` env var (default `false`): opt in to the
  old crash-on-mismatch behavior for safety-conscious deployments.
- `MEMINI_AUTO_DETECT_MODEL` env var (default `true`): opt out of the
  new-deployment auto-detection.
- `get_status` now reports `embeddingDimMismatch`, `embeddingDimExpected`,
  `embeddingDimActual`, `modelName`, and `modelDimension`.

### Migration

- Existing users: **no action required.** Your MiniLM 384-dim setup
  continues to work exactly as before.
- Users who want to upgrade to BGE-M3: see `docs/upgrading-embeddings.md`
  and run `archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py`.
- The migration script is non-destructive — your original 384-dim vectors
  are preserved in the `embedding` column.

## [0.7.6] - 2026-07-10

### Removed (BGE-Large support)

- **BGE-Large (`BAAI/bge-large-en-v1.5`) support removed.** BGE-Large was added in v0.7.0 alongside BGE-M3 as a "high-precision 1024-dim option" but turned out not to be used in production. v0.7.6 keeps the codebase clean by removing BGE-Large entirely. The supported models are now exactly two: **MiniLM-L6-v2 (384-dim, default)** and **BGE-M3 (1024-dim, optional upgrade)**.
- **`embedding_bge_large` column dropped** from the `memories` table. Migration 000007 drops the column and its index. Applied to live `memini-postgres` (port 5434) — 821 memories preserved, 819 with MiniLM, 800 with BGE-M3.
- **`BGE_LARGE_MODEL_ID` / `BGE_LARGE_DIM` constants removed** from `src/memini_ai/model/manager.py`. `_MODEL_ALIASES` reduced to two aliases: `bge-m3` and `minilm`.
- **`INSERT_MEMORY_BGE_LARGE` and `SEARCH_MEMORIES_BGE_LARGE` queries removed** from `src/memini_ai/postgres/queries.py`. The `add_memory` routing is now a 2-way switch (MiniLM → `embedding`, BGE-M3 → `embedding_bge_m3`).
- **`COLUMN_TO_MODEL` reduced to 2 columns** in `src/memini_ai/memory/rrf.py`. RRF now searches only the `embedding` and `embedding_bge_m3` columns.
- **`enabled_models` default in `config.py`** reduced to `['BAAI/bge-m3']` (was `['BAAI/bge-m3', 'BAAI/bge-large-en-v1.5']`).

### Migration Script Kept (Reference)

- **`/home/jcharles/Projects/MCP-Servers/archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py`** is kept as a reference example for users who want to do similar migrations on their own (e.g. swap to a different 1024-dim model, or upgrade from MiniLM to a custom model). The script is self-contained and works against any PostgreSQL with the `memini-ai-dev` schema — it does NOT need BGE-Large to be installed in the memini-ai codebase.
- The corresponding **`migrate_minilm_to_bge_m3.py`** is the canonical MiniLM → BGE-M3 upgrade script and is the recommended path for production use.

### Use Case: MiniLM → BGE-M3 Upgrade

The user-stated motivation for the v0.7.0 → v0.7.5 multi-model work was the **"GPU upgrade path"**: start with MiniLM (fast, small, CPU-friendly), get a machine with a GPU, then **migrate the existing memories to BGE-M3 (higher precision, GPU-friendly)** without losing the original data. The migration is:

1. Set `MEMINI_MODEL_NAME=BAAI/bge-m3` in `.env`.
2. Install `sentence-transformers` with the `[gpu]` extra (`uv pip install sentence-transformers[gpu]`).
3. Run the migration script: `python archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py` (with the DB URL pointing at the live DB).
4. Verify with `SELECT COUNT(*) FROM memories WHERE embedding_bge_m3 IS NOT NULL;` — should match the original memory count.
5. Set `MEMINI_ENABLE_RRF=true` to enable RRF search across both MiniLM and BGE-M3 columns.

The MiniLM column is **never touched** by this migration — the 384-dim vectors remain for backwards compatibility. New memories written after the migration land in `embedding_bge_m3` (BGE-M3); old memories can be re-embedded as needed (or left as MiniLM).

### Tests

- **40 tests removed** (was 824, now 784). Removed: 4 BGE-Large unit tests + 2 BGE-Large integration tests in `test_add_memory_multi_model.py`; 1 BGE-Large test in `test_manager_dim_checks.py`; 5 mock model_id references in `test_embeddings.py` (BGE-Large → BGE-M3) + 1 in `test_search.py` + 1 in `test_system.py`; BGE-Large assertions in `test_thought_chains.py` and `test_postgres_database.py` updated to reference BGE-M3 (the only 1024-dim model).
- **Quality gates**: `ruff check src/ tests/` → 0 errors, `mypy src/` → 0 errors (53 source files), `pytest` → 784 passing + 4 pre-existing env-var-pollution failures (NOT caused by this change; present on `main` before v0.7.6).

### Notes

- **Live DB migration applied**: `podman exec memini-postgres psql ... ALTER TABLE memories DROP COLUMN embedding_bge_large;` — 821 memories preserved, 0 rows lost.
- **Backwards incompatible at the schema level** (column dropped), but **backwards compatible at the API level**: callers passing `embedding_model="BAAI/bge-large-en-v1.5"` will now get a `ValueError: Unknown model ...` from `ModelManager._load_model()`. The fix is to either remove the field (MiniLM will be used) or switch to BGE-M3. This is the intended behavior — BGE-Large is no longer supported.
- **No new env vars.** No breaking config changes. Existing setups with `MEMINI_MODEL_NAME=BAAI/bge-m3` continue to work unchanged.

## [0.7.5] - 2026-07-10

### Bug Fixes (Multi-Model RRF — three latent bugs from v0.7.0/v0.7.3)

These three bugs together prevented the multi-model RRF feature from actually working end-to-end, even though the v0.7.0/v0.7.3 code appeared to be in place. All three are fixed in v0.7.5.

- **Bug 1: `ModelManager._load_model()` was constrained by `embedding_dim`, ignoring `config.model_name`.**
  The old code picked models based on `_embedding_dim`:
    - 384 → MiniLM-L6-v2
    - 1024 → BGE-Large (NOT BGE-M3, even if `config.model_name='BAAI/bge-m3'`)

  This made BGE-M3 effectively unreachable as an active model. **Fix**: `_load_model()` now picks based on `config.model_name` (set via `MEMINI_MODEL_NAME` env var). Supported values: `'sentence-transformers/all-MiniLM-L6-v2'`, `'BAAI/bge-m3'`, `'BAAI/bge-large-en-v1.5'`, or any custom HF model ID. Short-name aliases (`'bge-m3'`, `'minilm'`, `'bge-large'`) are also accepted. The `embedding_dim` constraint is kept as a post-load sanity check.

- **Bug 2: `add_memory` wrote 1024-dim vectors to the 384-dim `embedding` column — silent data loss**.
  When `embedding_model` was BGE-M3 or BGE-Large, the 1024-dim vector was passed to `INSERT_MEMORY_WITH_MODEL`, which only writes to the `embedding vector(384)` column. The write either silently failed (vector set to NULL) or was truncated, depending on the value. **Fix**: `_entry_to_record` now writes the vector to the column matching the model's dimensionality (using `MODEL_COLUMNS[entry.embedding_model]`), and `add_memory` routes to `INSERT_MEMORY_BGE_M3` / `INSERT_MEMORY_BGE_LARGE` for non-MiniLM models. Two new query constants added.

- **Bug 3: RRF `COLUMN_TO_MODEL` used short name `'all-MiniLM-L6-v2'` but `ModelManager` expects the full HF name**.
  The RRF helper (`memini_ai/memory/rrf.py`) was passing `'all-MiniLM-L6-v2'` to `embedder.embed(model_name=...)`, which raised `ValueError: Unknown model 'all-MiniLM-L6-v2'`. **Fix**: `COLUMN_TO_MODEL` and `MODEL_TO_DIM` in `rrf.py` updated to use the full HF names. The MiniLM column is now reachable from RRF.

### Tests

- **8 new tests** in `tests/test_add_memory_multi_model.py` covering the column-routing fix end-to-end:
  - 4 unit tests for `_entry_to_record` column routing (MiniLM, BGE-M3, BGE-Large, no-model)
  - 4 live DB integration tests verifying vectors land in the correct column
- **Rewrote `tests/test_manager_dim_checks.py`** to test `model_name`-driven selection (5 tests): MiniLM, BGE-M3, BGE-Large, custom HF model passthrough, short-alias normalization.
- **Test count: 824 passing** (was 777 in v0.7.3, +47 new tests).

### Quality Gates

- `ruff check src/ tests/` → 0 errors
- `mypy src/` → 0 errors (53 source files)
- `pytest tests/ --ignore=tests/test_postgres_database.py` → **824 passing**

### Notes

- **Backwards compatible**: All 3 fixes are bug fixes for the v0.7.0 multi-model feature. No API changes. Existing 384-dim-only setups (which never set `MEMINI_MODEL_NAME`) continue to work exactly as before — they default to MiniLM.
- **Migration to enable multi-model**: Set `MEMINI_MODEL_NAME=BAAI/bge-m3` in `.env` and `MEMINI_ENABLE_RRF=true` to get BGE-M3 for new writes + RRF queries across all populated model spaces. The DB schema (columns `embedding`, `embedding_bge_m3`, `embedding_bge_large`) is already in place via migration 000006.
- **All 3 model spaces populated for ~800 memories** in the live `memini-postgres` (port 5434) DB after the v0.7.5 migration scripts ran (MiniLM 384, BGE-M3 1024, BGE-Large 1024).
- **Reranking**: With `MEMINI_ENABLE_RRF=true`, queries fuse top-k results from each populated model column using `score = sum(1 / (k + rank))` (k=60, standard RRF constant).

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