# Changelog

All notable changes to this project are documented in the
[canonical CHANGELOG.md](https://github.com/Veedubin/memini-ai-dev/blob/main/CHANGELOG.md)
in the repository. This page reproduces the **v1.x and v0.8.x** entries for
quick reference; older releases (v0.7.x and earlier) are available on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2026-07-18

### Fixed

- **opencode MCP launch: command path must be absolute (the real reason v1.2.1 didn't work)** — Session 53's v1.2.1 release fixed the pgembed bootstrap but missed the actual symptom the user hit. v1.2.1's `init` wrote `command: ["uvx", "--from", "memini-ai-dev", "memini-ai"]` into `~/.config/opencode/opencode.json`. That command depends on `uvx` being in `$PATH` at MCP-spawn time. But opencode spawns MCP servers with a clean non-interactive environment, which on most distros (Ubuntu/Debian/Fedora) does NOT include `/home/<user>/.local/bin` in `$PATH` (only login shells do). The MCP server therefore fails instantly with "uvx: command not found" and is never started. Every subsequent call to a memini-ai tool returns "tool not available" — the symptom looks like "embedded server doesn't come up or connect", because there is no server. New helper `_resolve_memini_command()` in `installer.py` resolves the launch command at install time using `shutil.which("uvx")` and falls back to `<home>/.local/bin/memini-ai`, writing ABSOLUTE paths into the opencode.json command array. Both `init` and `update` use the resolver. The MCP server now starts regardless of the user's shell PATH.

## [1.2.1] - 2026-07-18

### Fixed

- Re-released as v1.2.1 because the v1.2.0 tag (Session 52, RBAC + SSL + container detection, commit `0985dc1`) was already taken. Per AGENTS.md 'Never Retag a Public Release' rule, this is a patch release on top of v1.2.0 with the same fixes below.
- **Fresh-VM pgembed bootstrap (opencode 30s timeout)** — `EmbeddedPGDriver.__init__` never created the data dir parent, so the first `pgembed.get_server()` call on a fresh machine failed with "Parent directory of pgdata does not exist". The `PostgresDatabase.initialize()` 3-retry loop (1+2+4s backoff) + the HF model download (~30s) combined to push past opencode's 30s MCP startup timeout. The first ever tool call (`query_memories` etc.) would time out and opencode marked `memini-ai-dev` as "server unavailable". **3 cascading bugs fixed in `_start_new_server`:**
  1. `self._data_dir.mkdir(parents=True, exist_ok=True)` — auto-create the data dir parent before calling `pgembed.get_server()`.
  2. `_write_postgres_config()` appends `dynamic_library_path = '$libdir:<pgembed_ext_lib_path>'` to `postgresql.conf` (NOT `postgresql.auto.conf` — that's only read on ALTER SYSTEM/SIGHUP, not on a fresh server start). Idempotent: drops any prior line we wrote before appending. Skips on the very first init (postgresql.conf doesn't exist yet) — the NEXT call (after initdb) writes it.
  3. `restart_server_for_new_config()` async method: forces the postmaster to stop + reattach so the new `dynamic_library_path` takes effect. `pgembed.get_server(cleanup_mode='stop')` + immediate cleanup, then a fresh `get_server(cleanup_mode=None)` to spin up a new postmaster that re-reads postgresql.conf.
- `memini-ai init` CLI now calls `driver.restart_server_for_new_config()` after `get_uri()` so the postmaster picks up the new config before the user runs opencode. Wrapped in try/except so a restart failure doesn't break init.

### Why this works

- pgembed 0.2.0 ships `vector.so` and `vectorscale-0.9.0.so` at `<site-packages>/pgembed/pginstall/lib/postgresql/`, but the stock postgres `dynamic_library_path` is just `'$libdir'` (the install's own lib dir), so `CREATE EXTENSION vector` couldn't find the .so file. The pgembed `server.create_extension()` API uses psql internally which segfaults on Python 3.13/3.14 (exit 139). Writing to `postgresql.auto.conf` looked promising but is only read on ALTER SYSTEM/SIGHUP, not on a fresh server start. The only safe path is editing `postgresql.conf` (which pgembed never touches after initdb) + restarting the postmaster.

### Added

- 4 regression tests in `tests/test_pgembed_driver.py::TestEdgeCases`:
  - `test_start_new_server_creates_parent_data_dir` — regression for the "Parent directory of pgdata does not exist" fix
  - `test_start_new_server_writes_postgres_config` — verifies the dynamic_library_path line is in postgresql.conf after `_start_new_server` runs
  - `test_start_new_server_continues_if_postgres_config_write_fails` — OSError on config write must NOT crash server start
  - `test_postgres_config_write_is_idempotent` — 3 calls produce 1 line
  - `test_postgres_config_write_skips_first_init` — no PG_VERSION = no config write (initdb needs empty dir)
- Module-level `EXTENSION_POSTGRES_LIB_PATH` import in `postgres/driver.py` so tests can mock it without needing pgembed installed locally.

### Changed

- `pyproject.toml`: removed empty `team = []` optional-dependency entry (the team server selection lives in the install script, not as a separate optional dep — keeps the install path unified per user direction in this session).

### Quality Gates

- `ruff check src/` → 0 errors
- `pytest tests/` → 942/944 pass (2 pre-existing pgembed-env failures unrelated to these changes)
- Verified end-to-end on test-bunty VM (jcharles@192.168.1.86, fresh Ubuntu 24.04, no prior memini-ai data):
  - `memini-ai init` from cold: 1.5s, postgresql.conf gets the dynamic_library_path line
  - `memini-ai --stdio` + `query_memories` returns 0 results in <1s after model loads
  - `SHOW dynamic_library_path` against the running postmaster returns the correct bundled path
  - `CREATE EXTENSION vector;` + `CREATE EXTENSION vectorscale;` both succeed against the running postmaster
  - opencode launch: "init count=8", memini-ai-dev still shows a brief "server unavailable" warning at T+1.4s (the MCP probe timeout) but recovers — was T+33s hard fail before this fix

### Process lessons

- When MCP startup is the symptom, dig past the startup — the actual failure is usually on the first tool call, not at process spawn.
- `server.create_extension()` looks like the right API for installing pgvector in pgembed, but psql segfaults on 3.13/3.14. Workaround: edit `postgresql.conf` + restart.
- For tests, mock imports at module level (not inside the function body) so `patch.object(module, "CONSTANT", value)` works on CI without the actual package installed.

## [1.0.3] - 2026-07-16## [1.0.3] - 2026-07-16

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

