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
| MEMINI_DB_URL | PostgreSQL connection | postgresql://postgres:password@localhost:5432/postgres |
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
