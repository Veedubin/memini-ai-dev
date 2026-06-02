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
- Framework: FastMCP (MCP server with 35 tools)
- Database: PostgreSQL with pgvector + pgvectorscale
- Embeddings: BGE-Large (1024-dim) / MiniLM-L6-v2 (384-dim)
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
| MEMINI_TRUST_ENGINE | Enable trust scoring | false |
| MEMINI_MEMORY_GRAPH | Enable memory graph | false |
| MEMINI_AUTO_EXTRACT | Enable auto-extraction | false |
| MEMINI_TIERED_LOADING | Enable tiered loading | false |
| MEMINI_KG_ENABLED | Enable knowledge graph | false |
| MEMINI_MULTI_PEER_ENABLED | Enable multi-peer | false |
| MEMINI_DIALECTIC_ENABLED | Enable dialectic reasoning | false |
| THOUGHT_CHAINS | Enable persistent thought chains | false |

## Review Notes
- **2026-06-02**: **v0.7.0 DUAL-MODEL RRF — 5/15 STEPS DONE** — Session 4 continued v0.7.0 implementation. **Step 1 (config validators), Step 2 (memories_1024 table), Step 3 (6 new 1024 query constants), Step 4 (memory/rrf.py NEW), Step 5 (6 new database methods + _expand_384_to_1024 helper) — ALL COMPLETE**. Migration applied to live DB; **83 memories at 384-dim verified intact** (was 80 in Session 3, +3 from this session's testing). Working tree dirty on `config.py`, `schema.py`, `queries.py`, `database.py` + new file `memory/rrf.py`. `ruff + mypy` clean for all 5 modified/created files. **Agent-blocker fix:** All 47+ agent `.md` files across 6 locations (root, boomerang-v3, neuralgentics, Super-Memory, boomerang, plus the critical `node_modules/@veedubin/boomerang-v3` install and the npm cache) corrected from `ollama-cloud/<model>:<tag>-cloud` → `ollama/<model>:<tag>`. Ollama Cloud API confirmed all 10 model names exist. **OpenCode restart STILL REQUIRED** — running process (PID 307190) has old config cached. Saved to memini-ai memory `b8b42742-e4e1-4a2a-a1a1-afd85e597f59`. See `TASKS.md` v0.7.0 Implementation Status table for remaining 10 steps.
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
