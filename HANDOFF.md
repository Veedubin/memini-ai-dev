# Memini-ai Handoff Document

> **Session**: 2026-05-19
> **Project**: Memini-ai v3.0 (formerly Super-Memory-TS)
> **Status**: ALL PHASES COMPLETE - pgvector migration DONE, live visualization added, v0.2.5 released

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
export MEMINI_DB_URL="postgresql://postgres:password@localhost:5434/postgres"
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
| MEMINI_DB_URL | PostgreSQL connection URL | postgresql://postgres:password@localhost:5432/postgres |
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

### Live Visualization
- KnowledgeGraph writes directly to PostgreSQL entities/entity_relationships tables
- FastAPI server at `src/memini_ai/api/visualization.py`
- D3.js template at `src/memini_ai/api/d3_template.py`
- Run with: `uvicorn memini_ai.api.visualization:create_app --factory True`

---

## PyPI Publishing Status (2026-05-19)

### v0.2.5 Release Status
- **Git tag**: `v0.2.5` created and pushed ✅
- **GitHub commit**: `b1077b7` ✅
- **GitHub Release**: Created via workflow ✅
- **PyPI publish**: Trusted publishing via GitHub Actions

### Version History
| Version | Date | Notes |
|---------|------|-------|
| v0.2.0 | 2026-05-18 | pgvector migration complete |
| v0.2.1 | 2026-05-18 | Package name fix |
| v0.2.2 | 2026-05-18 | Documentation updates |
| v0.2.3 | 2026-05-18 | Version bump |
| v0.2.4 | 2026-05-19 | aiosqlite dependency fix |
| v0.2.5 | 2026-05-19 | Version bump fix |

### Release Process
1. Update version in `pyproject.toml`
2. Commit with `git add -A && git commit -m "Bump version to X.Y.Z"`
3. Tag with `git tag vX.Y.Z -m "Release vX.Y.Z"`
4. Push: `git push origin main && git push origin vX.Y.Z`
5. GitHub Actions workflow handles PyPI publish automatically

---

*End of handoff.*
