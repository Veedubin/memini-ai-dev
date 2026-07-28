# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
