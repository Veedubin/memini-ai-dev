# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
