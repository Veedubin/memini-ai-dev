# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.4] - 2026-08-01
- **Docs: `MEMINI_VECTOR_BACKEND` migration hint surfaced in three places** — Closes the most common v0.8.x→v1.0.0+ upgrade foot-gun. No code changes, no env var changes, no dependency changes.
  - **`.env.example`**: New `# ── Vector Backend Selection (v1.0.0+) ──` section at the top (right after Core Database Settings) documenting both `MEMINI_VECTOR_BACKEND` (values: `pgembed` | `postgres-external`, default `pgembed`, REQUIRED if `MEMINI_DB_URL` is set) and `MEMINI_PGEMBED_DATA_DIR` (default `~/.local/share/memini-ai/pgembed/data`). Includes a prominent warning block about the v1.0.0 breaking change.
  - **`README.md`**: Minimal MCP client config example now shows BOTH `pgembed` (new installs) and `postgres-external` + `MEMINI_DB_URL` (v0.8.x upgraders) as commented alternatives, with a 1-line note above explaining the upgrade path.
  - **`database.py` error message**: The v1.0.0 RuntimeError now includes an OpenCode-specific hint showing the exact `opencode.json` MCP environment block structure (`mcp.memini-ai-dev.environment.MEMINI_DB_URL` + `MEMINI_VECTOR_BACKEND`), since the most common cause for OpenCode users is forgetting to set the env var in the wrapper config.
  - **Tests**: 1 new test in `tests/test_config.py::TestV1BackendConfig` — `test_v100_error_message_mentions_opencode_wrapper` asserts the error message contains `"opencode.json"` or `"environment"` so future changes don't accidentally drop the wrapper hint. 1072 passing (4 pre-existing `memini-vision` `ModuleNotFoundError` env failures unchanged). `ruff check` 0 errors. `mypy` 0 new errors (14 pre-existing in 6 files unchanged).

## [1.5.3] - 2026-07-29
- **Docs: HANDOFF.md Session 60 entry covering the v1.5.1 KG add_memory timeout fix and the v1.5.2 SaaS doc removal; notes the file's prior staleness at Session 52. No code changes.**

## [1.5.2] - 2026-07-29
- **Docs: removed `docs/design/memini-cloud-thin-client-architecture.md`** — SaaS strategy docs do not belong in the public OSS repo; the `memini-ai-cloud` design now lives in a private repository. No code changes. NOTE: the v1.5.1 tag and git history still contain the original 794-line version of this document.

## [1.5.1] - 2026-07-29
- **Fix: `add_memory` MCP tool no longer hangs on entity-dense content when `KG_ENABLED=true`** — Three coordinated changes that together take `add_memory` on a 22-entity passage from ~20s (MCP `-32001` timeout at 60s) to 3.35s with one model load (was ~15+). All three were observed reloading the SentenceTransformer weights per KG entity (1-4s each); the underlying call chain was correct but synchronously re-entered the model acquire/release path on every entity save.
  - **Fire-and-forget KG extraction** — `server.py::MCPServer.add_memory` now schedules `_kg_extract_with_timeout(content, memory_id)` via `asyncio.create_task` and attaches a done-callback. The extraction runs in the background; `add_memory` returns its readback immediately. A 10-second `asyncio.wait_for` cap inside `_kg_extract_with_timeout` ensures even a pathological KG pass (hundreds of entities) cannot block the response beyond ~10s. TimeoutError and any other exception are logged with `kg_auto_extract_timeout` / `kg_auto_extract_failed` and swallowed — the `add_memory` response is never affected by KG errors. The done-callback consumes any unhandled exception so "Task exception was never retrieved" warnings do not leak.
  - **Model stays hot on `release()`** — `model/manager.py::ModelManager.release()` no longer auto-unloads the SentenceTransformer when `ref_count` reaches 0. The model is a process-scoped singleton; auto-unloading on ref_count=0 caused the per-entity save cycle (`save entity → acquire → generate_embedding → release → reload next entity`) to incur a full weight reload each time. With KG firing N entity saves per `add_memory`, that was N reloads × 1-4s = unbounded latency. Callers that want to free GPU memory should still call `unload()` directly; the singleton stays hot for the lifetime of the MCP server process, which is the desired long-running behaviour.
  - **`content_exists` short-circuit on entity save** — `knowledge_graph.py::_save_entity_to_storage` now hashes the `kg:entity:{json}` text and queries `Database.content_exists(hash)` **before** constructing a `MemoryEntry` or calling `add_memory`. Re-encountering a known entity becomes a cheap no-op (hash + one SQL query) instead of going through the full embed+insert path. This also avoids the `ValueError` raised inside `add_memory` when the same content is written twice (the redundancy check was previously a thrown exception, now a pre-checked early return).
  - **Test adjustment** — `tests/test_server.py::TestKGAutoExtractHook` updated for the fire-and-forget contract: the two assertions that previously checked `extract_and_register_entities.assert_called_once` immediately after `add_memory` now `await asyncio.sleep(0.05)` to let the event loop schedule the background task before asserting. The hook assertions still verify the call happened exactly once; the only behavioural change is the scheduling.
  - **Performance (measured)**: 20s → 3.35s on a 22-entity content sample; one model load (was ~15+ reloads). Users who disabled `KG_ENABLED` as a temporary mitigation (the only viable workaround before this fix) can re-enable it on v1.5.1.
- **Docs: `memini-cloud-thin-client-architecture.md` design doc** — New 794-line design document covering the proposed SaaS "thin client" for memini-ai-cloud. Documents the transport choices (HTTPS / WebSocket / gRPC), the local `CloudProxy` MCP server that brokers requests to the cloud, the v1 stateless → v2 session-token progression, configuration model, error handling, and a 31-card phased delivery roadmap (T-CLOUD-001 through T-CLOUD-031). Pure design (no code, no secrets, no API keys) — the actual SaaS backend lives in a separate private repo (`memini-ai-cloud`); memini-ai-dev's role is the open-source thin client that can talk to either a locally-hosted embedded pgembed backend or a hosted cloud backend through the same MCP surface. Captures the architecture decisions from the 2026-07-29 cloud-design brainstorm so future contributors don't re-litigate the transport choice.
- **Tests**: 1071 passing (4 pre-existing `memini-vision` `ModuleNotFoundError` env failures unrelated to this change — same env-stub issue noted in v1.5.0 / v1.4.2 entries). `ruff check` 0 errors. `mypy --strict` 0 new errors.

## [1.5.0] - 2026-07-28
- **Feature: `memini-ai init` TLS options for external/team servers** — Adds three new flags to the `init` CLI so operators can configure a secure SSL connection to an external PostgreSQL team server without hand-editing `.env`:
  - **`--tls`** — Convenience flag that appends `?sslmode=require` to the connection string (encrypts the link without certificate verification; appropriate for trusted-network team servers that don't ship a CA bundle).
  - **`--tls-ca <PATH>`** — Full certificate verification. Appends `?sslmode=verify-full&sslrootcert=<PATH>` and validates that the supplied file exists, is readable, and starts with the PEM `-----BEGIN` header. The path is also written to `.env` as `DB_SSLROOTCERT=<PATH>` so the pgembed launcher can re-export it on every start.
  - **`--no-tls`** — Explicit opt-out for the legacy plaintext mode (`?sslmode=disable`). Symmetric with `--tls` so users can flip back without opening the env file.
  - **Interactive `--team` flow upgrade** — When the user picks `verify-ca` or `verify-full` interactively, the installer now prompts for the CA bundle path with a no-TUI fallback (the path can also be passed via `--tls-ca` for scripted/headless init).
  - **Post-config 5-second non-fatal connection test** — After writing `.env` and `opencode.json`, `init` issues a short `psycopg.connect()` against the resolved DSN and prints `TLS test: PASS (sslmode=verify-full, rootcert=/etc/ssl/certs/team-ca.pem, DSN=postgresql://...)` or the exact `psycopg.OperationalError`. A failure is non-fatal by design — the configuration is already on disk and the user can iterate — but the message tells them whether the server actually accepted the cert.
  - **Backwards compatible** — All three flags are opt-in. The previous "no flag → no sslmode query parameter" behavior is preserved exactly, so existing v1.4.x `.env` files and `opencode.json` configs keep working unchanged. The embedded `pgembed` path is unaffected (pgembed listens on a local socket, no TLS negotiation).
- **Docs: doc-drift fix** — Root `AGENTS.md` had a hardcoded `Latest version: v1.0.3` line that drifted out of sync with the actual `pyproject.toml`. Replaced with a drift-proof pointer to `CHANGELOG.md` / `pyproject.toml` ("the actual released version is whatever `pyproject.toml` says"). This prevents the same drift from recurring silently across future releases.
- **Lockfile: `uv.lock` regenerated** to match v1.4.2's `pyproject.toml` state (the v1.4.2 release bumped `pyproject.toml` but didn't regen the lockfile in a few transitive entries; this commit syncs them).
- **Tests**: 1065 passing (+37 net new in `tests/test_installer.py` — 81 total in that file: 14 covering the three new flag paths through `_resolve_db_url` / `_write_env` / `_write_opencode_config`; 12 covering the interactive CA-prompt flow; 7 covering the post-config 5-second connection test pass/fail + sslmode/rootcert/DSN reporting; 4 covering the new `.env` `DB_SSLROOTCERT` write; 0 regressions). `ruff check` 0 errors. `mypy --strict` 0 new errors (10 pre-existing Keras/tf_keras env failures on HEAD unrelated to this change; the same pre-existing numpy 3.14 env stub issue persists).

## [1.4.2] - 2026-07-28
- **Fix: `get_status` distinguishes warm-up from misconfiguration** — During the ~5-15s window between server start and full DB-pool/model readiness, `get_status` previously returned `memoryReady=false, memoryCount=0`, which agents reliably misread as "backend misconfigured, fall back to a different server" (observed in the wild 2026-07-28 — session 38 traced the mis-dispatch to this signal). The response now carries an explicit lifecycle state:
  - **`status`** (string) — one of `"warming" | "ready" | "degraded" | "error"`. `"warming"` means "still initializing, retry shortly"; `"ready"` means fully operational; `"degraded"` means operational with a non-fatal component down (e.g. image arm unavailable when image search is enabled); `"error"` means an init failure was recorded.
  - **`warming`** (bool) — explicit `true` while components are still coming up, `false` once ready. Independent of `memoryReady` so callers can branch on warm-up without re-implementing the readiness check.
  - **`warmingMessage`** (string | `null`) — human-readable hint when warming (e.g. `"memory subsystem still initializing; retry get_status in a few seconds"`). `None` once ready.
  - **Counts as `null` during warming/error** — `memoryCount`, `thoughtsCount`, `kanbanCardCount` are reported as JSON `null` (not `0`) when `status` is `"warming"` or `"error"`, so an agent's `if not count:` no longer spuriously fires on a fresh start.
  - **Legacy fields retained** — `memoryReady`, `modelReady`, `indexerReady`, `initError` are still present with the same semantics as v1.4.1, so existing callers that branch on the booleans continue to work unchanged. The new `status` enum is the recommended signal; the booleans are kept for backward compatibility.
- **Tests**: 1028 passing (+8 net new: 5 server-state tests covering warming→ready, error, degraded, and the null-count contract; +3 regression tests on the legacy boolean fields to lock the backward-compat shape). `ruff check` 0 errors. `mypy --strict` 0 new errors (10 pre-existing Keras/transformers env failures on HEAD unrelated to this change).

## [1.4.1] - 2026-07-28
- **Fix: installer writes `--stdio` into generated opencode.json MCP command** — `_resolve_memini_command()` in `src/memini_ai/installer.py` returned command arrays without `--stdio`. Since v1.0.0 a bare `memini-ai` defaults to `streamable-http` transport, but OpenCode's local MCP config type speaks stdio JSON-RPC — every `memini-ai init`-generated `opencode.json` therefore produced a "server unavailable" condition in OpenCode. All 4 return paths (uvx-on-PATH, `~/.local/bin/memini-ai` uv-tool install, dev-checkout `-m memini_ai.cli`, and the bare-`uvx` warning fallback) now append `--stdio` so the spawned process speaks the transport OpenCode actually expects.
- **Fix: corrected `_resolve_memini_command` docstring** — The previous docstring falsely claimed `uvx --from memini-ai-dev memini-ai` "always pulls the latest from PyPI". This is incorrect: `uvx` reuses the cached tool environment for an unpinned `--from <pkg>` spec indefinitely (live-verified: cache pinned at 1.0.4 despite 1.4.0 on PyPI). The docstring now states the actual behavior and points operators to `uv tool upgrade memini-ai-dev` (or pinning `==X.Y.Z` in the `--from` spec) as the correct update path.
- **Tests**: 1020 passing (3 net new assertions: `--stdio` membership + positional-last check on each of the 3 testable return paths — uvx-on-PATH, local-install, bare-uvx fallback). `ruff check` 0 errors. `mypy` 0 new errors. `uv.lock` synced to the v1.4.0 lockfile state as part of this commit.

## [1.4.0] - 2026-07-28
- **Feature: feature-activation hooks** — Activates three previously-dormant features by wiring their config flags to the actual write/startup paths. Before this release, `KG_ENABLED`, `MEMINI_MULTI_PEER_ENABLED`, and the v1.0.0 memory-relationships infrastructure were pull-only — the underlying mechanisms existed but no write-path or startup hook invoked them, so live deployments showed zero entities, zero relationships, and zero peers despite the flags being on. v1.4.0 adds three opt-in hooks (all default OFF; flags OFF = zero behavior change, identical to v1.3.1):
  - **KG entity extraction on write (Layer A)** — `add_memory` now runs the regex `EntityExtractor` post-write when `KG_ENABLED=true`. Zero LLM cost (deterministic regex over the text body). Failure-isolated: any exception is logged and the `add_memory` response is unaffected.
  - **Near-duplicate auto-SUPERSEDES** — `add_memory` runs a single vector-similarity query for near-duplicates before the write; if the best match meets the threshold, a `SUPERSEDES` relationship from the new memory to the existing one is created with the similarity as `confidence`. Gated by two new env vars: `AUTO_RELATIONSHIP_DETECTION` (default `false`) and `AUTO_RELATIONSHIP_SIMILARITY_THRESHOLD` (default `0.95`, clamped to `[0.0, 1.0]`). Failure-isolated.
  - **Owner peer auto-registration (Layer C, startup)** — On MCP server start, when `MEMINI_MULTI_PEER_ENABLED=true` (and `MEMINI_USER_MODELING=true`, due to `MultiPeerManager.is_enabled` coupling), auto-register a default `owner` peer with `trust_level=1.0`. Idempotent: the `list_peers()` count guard ensures the registration is skipped if any peer already exists, so re-starts are safe. Failure-isolated: never blocks server startup.
- **Fix: peer hook gate now checks `is_enabled`** — The startup hook gate previously relied on `multi_peer_enabled` config flag alone, which bypassed the `MultiPeerManager.is_enabled` runtime check. If `multi_peer_enabled=true` but `MultiPeerManager.is_enabled=false` (e.g. `USER_MODELING` not set), the old code would attempt `list_peers()` on an uninitialized manager. The new code checks both `multi_peer_enabled` AND `self._multi_peer_manager.is_enabled` before entering the registration block. 1-line fix plus a regression test in `tests/test_server.py`.
- **Tests**: 1020 passing (+11 net new: 10 hook tests + 1 regression test for the peer-gate fix). `ruff check` 0 errors. `mypy` 0 new errors (13 pre-existing in tests/ unrelated to this change). Live-verified end-to-end against the running `memini-postgres` instance: KG hook fires and stores extracted entities as `kg:entity:` memories, near-dup hook fires with no false positives at the 0.95 threshold, peer hook fires when flags are aligned.

## [1.3.1] - 2026-07-24
- **Fix: pgvector 0.5.0 readback crash** — pgvector 0.5.0's asyncpg codec returns a `Vector` object that is not iterable, so every fresh install (pgvector `>=0.3.0` resolves to 0.5.0) crashed on `query_memories` / `get_memory` / `list_memories` with `'Vector' object is not iterable`. Writes were unaffected. Added `_to_float_list()` normalizer in `src/memini_ai/postgres/database.py` that handles `Vector` (`to_list()`), `numpy.ndarray`, pgvector text-format strings (`"[0.1,0.2,...]"`), and `list`/`tuple`/array, applied at every readback site. pgvector spec stays `>=0.3.0`; the new code path tolerates all versions (0.3.x, 0.4.x, 0.5.x). Live-verified on a VM against pgvector 0.5.0.
- **Fix: Visualization API works in pgembed mode** — `api/visualization.py` lifespan now uses `create_database()` (which respects `MEMINI_VECTOR_BACKEND`) instead of requiring `MEMINI_DB_URL`, so it boots correctly in pgembed (default) mode where `MEMINI_DB_URL` is intentionally unset.
- **Tests**: 1018 passing (+22). No dependency or env-var changes.

## [1.3.0] - 2026-07-22
- **Feature: Kanban** — New `kanban_cards` table (plain Postgres rows, no pgvector column) + 4 MCP tools: `kanban_add_card`, `kanban_move_card`, `kanban_list_cards`, `kanban_get_card`. `get_status` now reports `kanbanCardCount`. 7 valid statuses (`triage`, `todo`, `ready`, `running`, `blocked`, `done`, `archived`); unique constraint on `(repo, external_id)` makes the `add` operation idempotent. Backed by a dedicated `DatabaseKanbanMixin` (`add_kanban_card`, `move_kanban_card`, `list_kanban_cards`, `get_kanban_card`, `count_kanban_cards`).
- **Fix: `sourceType='github'` silently rejected** — The `memories_source_type_check` CHECK constraint previously did not list `github`, so any `add_memory` call from the GitHub triage poller failed with a constraint violation. `MemorySourceType` enum now includes both `github` and `image` (the latter was already accepted at the DB layer but missing from the enum). For new deployments, `_ensure_schema()` auto-extends the constraint at startup. **For existing deployments, run migration `000009_add_github_source_type.sql` once on the live database** (idempotent `DROP IF EXISTS` + `ADD`; the new constraint is a strict superset of the old one, so all existing rows still satisfy it).
- **Improvement: Fail-loud error propagation** — All 14 write-tool wrappers in `server.py` now return `{"success": False, "error": "<real str(e)>"}` on DB failure instead of silently swallowing exceptions. Kanban write tools use a dedicated `_kanban_db_error()` formatter for richer messages (e.g. invalid status, not found).
- **Tests**: 24 new tests in `tests/test_kanban.py` covering schema constants, source-type enum, kanban tool wrappers, and `get_status.kanbanCardCount` (11 DB-dependent tests are skipped on the default CI runner; they pass against a live `memini-postgres` instance).

## [1.2.4] - 2026-07-21
- **Docs**: ecosystem diagram in the README now shows memini-ai as the first-class MCP (registered in opencode.json) vs brokered servers.

## [1.2.3] - 2026-07-21
- **Docs**: mkdocs-material site + gh-pages deploy workflow (https://veedubin.github.io/memini-ai-dev/).
- **Docs**: memory-lifecycle mermaid diagram in docs/architecture.md; pruned stray memory_report*.md files; DESIGN-thought-chains.md moved into docs/design/.
- **Docs**: fixed changelog link case in docs/architecture.md.

## [1.2.2] - 2026-07-21
- **Docs**: README rewritten to focus on elevator pitch, quickstart, and features. Version history moved to CHANGELOG.md.
- **Docs**: CHANGELOG.md backfilled with v1.2.x, v1.1.x, and v1.0.4 sections.

## [1.2.1] - 2026-07-20
- **Fix**: Embedded `pgembed` now self-bootstraps on a fresh VM (previously hit opencode's 30s MCP startup timeout).
- **Fix**: `EmbeddedPGDriver` now `mkdir`s the data dir parent before first `pgembed.get_server()`.
- **Fix**: `CREATE EXTENSION vector` now succeeds — pgembed 0.2.0's bundled `vector.so` is placed on `dynamic_library_path` via `postgresql.conf` edit + postmaster restart (server.create_extension segfaults on 3.13/3.14).
- **Fix**: `memini-ai init` now calls `restart_server_for_new_config()` so the new config is applied before the user runs opencode.
- **Docs**: CHANGELOG v1.2.0 release notes added.

## [1.2.0] - 2026-07-19
- **RBAC**: Per-project PostgreSQL users — auto-generated admin role (`memini_admin`) + project roles per directory, passwords via `secrets.token_urlsafe(32)`, `opencode.json` uses `{env:...}` syntax so secrets never land in committed config.
- **RBAC**: Default OPEN (all users see all memories, backward compatible); opt-in lockdown via `MEMINI_PEER_ENFORCEMENT=true` + `MEMINI_PEER_ID=<project>`. `peer_id` written on all inserts (tagging), filtering only when enforcement on.
- **RBAC**: New CLI: `memini-ai user add/list/remove`, `memini-ai init --new-db`.
- **SSL/TLS**: Five modes (prefer/require/verify-ca/verify-full/disable), prompted during team server setup, appended to `MEMINI_DB_URL` as `?sslmode={mode}`. Fixed control-flow bug in `_build_ssl_context`.
- **Installer**: Container runtime detection — detects podman, docker, containerd via `shutil.which`; offers to install podman (recommended) or docker if none found; installs `podman-docker` so docker commands work with podman. Only prompted for team server mode.

## [1.1.0] - 2026-07-18
- **Installer**: Standalone two-init installer pattern (same as neuralgentics) so memini-ai-dev works standalone, not just as part of neuralgentics.
- **Installer**: New CLI: `memini-ai init --homedir` (writes `~/.config/opencode/opencode.json`), `memini-ai init --project` (writes `./.opencode/opencode.json`), `memini-ai update` (checks PyPI, backs up config, updates MCP entry, refreshes cache).
- **Installer**: pgembed (default) vs team server mode, CPU/Auto/GPU embedding mode selection, image search / trust engine / KG / thought chains all toggleable, Ollama Cloud API key (optional), SHA-256 idempotency on config writes, state file (`.memini-ai-state.json`).

## [1.0.4] - 2026-07-17
- **Docs**: Patch bump from v1.0.3 — adds the missing CHANGELOG entry and doc-header refreshes that v1.0.3 should have included (HANDOFF/AGENTS/TASKS/CONTEXT headers refreshed to v1.0.3, new Session 52 entry). No code changes, no new env vars, no new dependencies.

## [1.0.3] - 2026-07-16
- **Docs**: HANDOFF/AGENTS/TASKS/CONTEXT catch-up to v1.0.2 reality.
- **Lockfile**: Regenerated `uv.lock` to match v1.0.2 `pyproject.toml` (was still at 1.0.0).
- **Docs**: Added CRITICAL section in `AGENTS.md` documenting the v1.0.0 `MEMINI_VECTOR_BACKEND` requirement.

## [1.0.2] - 2026-07-16
- **CLI**: Fixed `memini-ai migrate` command to match standalone script parity (6 bugs fixed: pg_restore from pgembed's pg17, pre-install extensions, exclude timescaledb, post-restore verification, error handling, dry-run flag).

## [1.0.1] - 2026-07-16
- **Migration**: Fixed `scripts/migrate_external_to_embedded.py` (6 bugs: system pg_dump + pgembed pg_restore, Unix socket URI parsing, pre-install extensions, exclude timescaledb, sync shutdown, spot-check column).
- **Config**: Fixed 6 `opencode.json` files missing `MEMINI_VECTOR_BACKEND` (would have broken MCP server on next TUI restart).

## [1.0.0] - 2026-07-16
- **Backend**: Embedded PostgreSQL (`pgembed`) is now the default backend. No Docker required.
- **Backend**: New `MEMINI_VECTOR_BACKEND` env var (`pgembed` or `postgres-external`).
- **Backend**: Multi-process server sharing (1 embedded Postgres shared by all memini-ai processes on same machine).
- **Backend**: RRF fusion across embedded + team server via `RRFDatabase` wrapper.
- **CLI**: New `memini-ai init | status | stop | migrate` commands.
- **Docs**: Design doc `docs/design/v1.0.0-embedded-pgembed-architecture.md` (76KB).

## [0.8.2] - 2026-07-13
- **Security**: Added `detect-secrets` baseline + CI scan to prevent API key/secret leaks.
- **Security**: Pre-commit hook + GitHub Actions workflow for `detect-secrets`.

## [0.8.1] - 2026-07-13
- **CI**: Pure re-trigger for `memini-vision>=0.1.0` dependency (wasn't on PyPI when v0.8.0 published).

## [0.8.0] - 2026-07-13
- **Vision**: Image-recall RRF fan-out arm (3rd RRF arm using CLIP over `memories_image` table).
- **Vision**: New `memories_image` table (768-dim CLIP embeddings, 1:1 FK to `memories.id`).
- **Vision**: 5 new env vars (`MEMINI_IMAGE_SEARCH_ENABLED`, `MEMINI_IMAGE_CLIP_MODEL`, etc.).
- **Vision**: Best-effort image arm (text RRF proceeds with 2 lists on CLIP failure).
- **Docs**: Design doc `docs/design/vision-memory-architecture.md` (30KB).

## [0.7.9] - 2026-07-12
- **Security**: Added "Never Commit Memory Data" rule to `AGENTS.md`.
- **Security**: Pre-commit inspection pattern for data files before `git add`.
- **Security**: `.gitignore` hardened for `*.dump`, `*.jsonl`, and `archives/` data files.

## [0.7.8] - 2026-07-10
- **Docs**: Comprehensive audit and rewrite (13 doc problems fixed: 1 CRITICAL, 4 HIGH, 6 MEDIUM, 2 LOW).
- **Docs**: README rewritten (tool count 35+→52, added 24 missing tools, regenerated architecture tree).
- **Docs**: `.env.example` updated with v0.7.7 env vars.
- **Docs**: `upgrading-embeddings.md` Step 2 corrected (torch CUDA install).
- **Fix**: BM25 punctuation-only query guard (returns `[]` for queries with no alphabetic chars).
- **Fix**: `get_sentence_embedding_dimension` deprecation in migration script.
- **Process**: Bumped `steps: N` frontmatter 10x across all 61 agent `.md` files.

## [0.7.7] - 2026-07-10
- **Embeddings**: New `MEMINI_AUTO_DETECT_MODEL` env var (new deployments auto-upgrade to BGE-M3).
- **Embeddings**: New `MEMINI_STRICT_EMBEDDING_DIM` env var (dim mismatch degrades to text-only instead of crashing).
- **Fix**: BM25 `text_only_search` empty-corpus `ZeroDivisionError`.
- **Fix**: `get_sentence_embedding_dimension` deprecation warning.
- **Fix**: 4 pre-existing test failures via `autouse=True` `_isolate_env` fixture.
- **Observability**: `get_status` now reports `modelName`, `modelDimension`, `embeddingDimMismatch`.
- **Docs**: New `docs/upgrading-embeddings.md` (4-step migration recipe).

## [0.7.6] - 2026-07-10
- **Embeddings**: Removed BGE-Large support (kept BGE-M3 and MiniLM-L6-v2).
- **Embeddings**: Migration script for BGE-Large kept as reference example.
- **Embeddings**: MiniLM → BGE-M3 upgrade path is now the canonical migration story.

## [0.7.5] - 2026-07-10
- **Embeddings**: Fixed 3 latent bugs in multi-model RRF (model selection, column routing, RRF mapping).
- **Embeddings**: Model name-driven selection with alias support.
- **Embeddings**: Multi-model column routing (new `INSERT_MEMORY_BGE_M3` / `INSERT_MEMORY_BGE_LARGE` queries).
- **Embeddings**: Full-HF-name RRF column mapping.

## [0.7.3] - 2026-07-06
- **Search**: Fixed `query_memories` default `threshold=0.72` (unrealistically tight for MiniLM-L6-v2).
- **Search**: RRF now propagates `threshold` and `exact_search` to 384-side `SearchOptions`.
- **Observability**: `get_status` now returns `memoryCount`, `thoughtsCount`, `queryLatencyMs`.
- **Observability**: `add_memory` post-write read-back (returns `error="post_write_readback_failed"`).
- **Observability**: New `healthcheck` MCP tool (write+read round-trip with PASS/FAIL).

## [0.7.2] - 2026-06-04
- **Docs**: CHANGELOG entry documents Session 10 health-check verification.
- **Docs**: Corrected stale Session 9 "memory server broken" diagnosis.

## [0.7.1] - 2026-06-03
- **Thought Chains**: Fixed `add_thought` MCP-call vector-injection error.
- **Thought Chains**: Pass `list[float]` directly (matches `memory.add`).
- **Thought Chains**: Truncate/pad to 384 dims for `vector(384)` column.

## [0.7.0] - 2026-06-02
- **Embeddings**: Dual-model RRF (384-dim MiniLM + 1024-dim BGE-M3/BGE-Large).
- **Embeddings**: New `memories_1024` table (1024-dim sidecar).
- **Embeddings**: `EMBEDDING_MODE` env var (`cpu`, `auto`, `gpu`).
- **Embeddings**: `elevate_memory_to_1024` MCP tool (auto-mode gated).
- **Search**: Reciprocal Rank Fusion (RRF) with k=60.
- **Docs**: Design doc `docs/design/dual-model-rrf-architecture.md`.

## [0.3.1] - 2026-05-19
- **Docs**: Documentation refreshed, stale version references updated.

## [0.3.0] - 2026-05-19
- **Thought Chains**: Persistent reasoning with branching/revision.
- **Thought Chains**: 9 new MCP tools (`start_thought_chain`, `add_thought`, etc.).
- **Schema**: PostgreSQL schema for thought chains.
- **Search**: `exact_search` for DiskANN.

## [0.2.8] - 2026-05-19
- **Style**: Ruff formatting pass across 30 files.

## [0.2.7] - 2026-05-19
- **Schema**: PostgreSQL schema fixes (IF NOT EXISTS, vector parsing, 384-dim vectors).

## [0.2.0] - 2026-05-18
- **Initial Release**: Trust-weighted memory, knowledge graph, tiered loading, MCP-compatible.
