# Memini-ai Development Tasks

> **Project**: Memini-ai v3.0 (formerly Super-Memory-TS)
> **Meaning**: "I remember" in Latin
> **Language**: Python (porting from TypeScript)
> **Framework**: FastMCP
> **Last Updated**: 2026-05-19 (v0.2.7: PostgreSQL schema fixes for idempotent initialization)

---

## Overview

Memini-ai is a local-first semantic memory server with vector search, designed to preserve context across AI agent sessions beyond simple RAG.

### Core Architecture (PostgreSQL/pgvector Backend)

```
memini-ai/
├── src/
│   ├── __init__.py
│   ├── server.py           # FastMCP server (35 tools)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── visualization.py  # FastAPI for live KG visualization
│   │   └── d3_template.py     # D3.js visualization HTML
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── system.py      # MemorySystem class
│   │   ├── database.py    # VectorDatabase ABC + implementations
│   │   ├── search.py       # TIERED/VECTOR_ONLY/TEXT_ONLY/PARALLEL strategies
│   │   └── schema.py      # MemoryEntry, SearchOptions, etc.
│   ├── postgres/          # PostgreSQL/pgvector backend
│   │   ├── database.py    # PostgresDatabase implementation
│   │   ├── schema.py      # SQL schema definitions
│   │   └── queries.py     # SQL query builders
│   ├── model/
│   │   ├── __init__.py
│   │   └── embeddings.py  # BGE-Large (1024-dim), MiniLM-L6-v2 (384-dim)
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── indexer.py     # Project file indexing
│   │   ├── chunker.py     # Semantic chunking
│   │   └── watcher.py     # File watching with watchdog
│   ├── config.py          # Environment + JSON config
│   └── utils/
│       ├── __init__.py
│       ├── hash.py        # SHA-256 for deduplication
│       └── logger.py
├── tests/
├── docs/
├── pyproject.toml
└── main.py
```

---

## Phase 1: Foundation (Memory Kernel) ✅ COMPLETE

### 1.1 Project Setup ✅
- [x] Initialize Python project structure
- [x] Set up pyproject.toml with dependencies
- [x] Configure uv for package management
- [x] Add ruff/mypy for linting/type checking

### 1.2 Core Memory System (Port from TypeScript) ✅
- [x] Port `memory/schema.py` - MemoryEntry, SearchOptions, SearchFilter types
- [x] Port `memory/database.py` - VectorDatabase ABC + QdrantDatabase + PostgresDatabase
- [x] Port `memory/search.py` - TIERED, VECTOR_ONLY, TEXT_ONLY, PARALLEL strategies
- [x] Port `memory/system.py` - MemorySystem class combining db + search
- [x] Port `model/embeddings.py` - BGE-Large (1024-dim), MiniLM fallback

### 1.3 MCP Server (FastMCP) ✅
- [x] Port `server.py` - 6 MCP tools (5 + get_status):
  - [x] `query_memories` - Semantic search
  - [x] `add_memory` - Store memory
  - [x] `search_project` - Search indexed files
  - [x] `index_project` - Trigger indexing
  - [x] `get_file_contents` - Reconstruct from chunks
  - [x] `get_status` - Component status
- [x] Configure stdio transport for MCP

### 1.4 Configuration Management ✅
- [x] Port `config.py` - Environment variable + JSON config
- [x] Support all MEMINI_* env vars (MEMINI_DB_URL, MEMINI_PROJECT_ID, etc.)
- [x] Implement config validation

### 1.5 Project Indexer ✅
- [x] Port `indexer/indexer.py` - File indexing with SHA-256
- [x] Port `indexer/chunker.py` - Semantic chunking at function/class boundaries
- [x] Port `indexer/watcher.py` - File watching (use watchdog instead of chokidar)

---

## Phase 2: Memory Kernel Features (3 Core Features) ✅ COMPLETE

### 2.1 Trust Engine ✅ COMPLETE
- [x] Add `trust_score` field to MemoryEntry (default 0.5)
- [x] Add `retrieval_count` field
- [x] Add `is_archived` field
- [x] Create `src/memini_ai/trust_engine.py` module (291 lines)
- [x] Implement feedback signals:
  - [x] Agent uses memory: +0.05
  - [x] Agent ignores memory: -0.02
  - [x] User corrects agent: -0.15
  - [x] User confirms explicitly: +0.10
- [x] Below 0.2: archive (not delete)
- [x] Above 0.8: promote (auto-inject on session start)
- [x] Expose tools: `get_trust_score`, `adjust_trust`, `list_archived`
- [x] Make it optional (can be disabled via MEMINI_TRUST_ENGINE=false)
- [x] 45 tests passing

### 2.2 Memory Graph ✅ COMPLETE
- [x] Add `relationships` JSON field to MemoryEntry schema
- [x] Create `Relationship` dataclass and `RelationshipType` enum
- [x] Create `src/memini_ai/graph.py` module (~400 lines)
- [x] Track relationships: SUPERSEDES, RELATED_TO, CONTRADICTS, DERIVED_FROM
- [x] Entity extraction (regex/heuristic-based)
- [x] Second-pass query for related memories
- [x] Expose tools: `find_related_memories`, `create_relationship`, `get_relationship_summary`
- [x] Make it optional (can be disabled via MEMINI_MEMORY_GRAPH=false)
- [x] 43 tests passing

### 2.3 Auto-Extract ✅ COMPLETE
- [x] Create `src/memini_ai/extractor.py` module (~350 lines)
- [x] Create `ConversationTurnTracker` class
- [x] After every N turns (configurable, default 5), fire LLM pass
- [x] Extract: facts, decisions, patterns, preferences
- [x] Store automatically via existing add_memory flow
- [x] Expose tool: `trigger_extraction`
- [x] Make it optional (can be disabled via MEMINI_AUTO_EXTRACT=false)
- [x] 35 tests passing

---

## Phase 3: Tier 2 Features (After Kernel Stable) ✅ COMPLETE

### 3.1 Pre-Compression Extraction ✅ COMPLETE
- [x] Hook into OpenCode context compaction event
- [x] Extract unsaved facts BEFORE context window squeezes
- [x] Requires Auto-Extract pipeline to exist first (Phase 2C ✅)
- [x] **New module**: `src/memini_ai/precompress.py` (~135 LOC)
- [x] **MCP tool**: `preconpress_extraction`
- [x] **Config**: `PRECOMPRESS`, `PRECOMPRESS_THRESHOLD`
- [x] 32 tests passing

### 3.2 Tiered Loading L0/L1/L2 ✅ COMPLETE
- [x] L0 (100 tokens): Project summary - auto-inject on session start
- [x] L1 (2K tokens): Key decisions and patterns - loaded when planning
- [x] L2 (full): Specific memories - retrieved on demand
- [x] Requires Trust Engine (Phase 2A ✅) and Memory Graph (Phase 2B ✅)
- [x] **New module**: `src/memini_ai/tiered_loader.py` (~555 LOC)
- [x] **Schema**: `SummaryTier`, `TieredSummary` in schema.py
- [x] **MCP tools**: `get_tier0_summary`, `get_tier1_summary`
- [x] **Config**: `TIERED_LOADING`, `TIER0_MAX_TOKENS`, `TIER1_MAX_TOKENS`
- [x] 42 tests passing

### 3.3 User Modeling (Honcho-style) ✅ COMPLETE
- [x] Build persistent user profile
- [x] Track preferences, communication style, expertise
- [x] Update dialectically via LLM reasoning
- [x] Requires 50-100 sessions of data (defer full validation)
- [x] **New module**: `src/memini_ai/user_model.py` (~530 LOC)
- [x] **Schema**: `UserProfile`, `UserPreference` in schema.py
- [x] **MCP tools**: `get_user_profile`, `update_user_profile`
- [x] 45 tests passing
- **Config**: `USER_MODELING`, `USER_MODEL_MIN_SESSIONS`

---

## Phase 4: Advanced Features (Optional) ✅ COMPLETE

### 4.1 Memory Decay/Consolidation ✅ COMPLETE
- [x] Add `decay_rate`, `last_accessed`, `access_count` fields to MemoryEntry
- [x] Create `src/memini_ai/decay.py` module (~430 lines)
- [x] Implement exponential decay formula with configurable half-life
- [x] Implement consolidation (find similar memories, merge)
- [x] Expose tools: `get_decay_status`, `trigger_consolidation`, `list_fading_memories`, `adjust_decay_rate`
- [x] Make it optional (can be disabled via MEMINI_DECAY_ENABLED=false)
- [x] 56 tests passing

### 4.2 Full Knowledge Graph ✅ COMPLETE
- [x] Create `src/memini_ai/entity_extractor.py` module (~340 lines)
- [x] Create `src/memini_ai/knowledge_graph.py` module (~520 lines)
- [x] Named entity extraction (PERSON, ORGANIZATION, CONCEPT, CODE, PROJECT)
- [x] Transitive inference engine with BFS path finding
- [x] SPARQL-lite query interface
- [x] Expose tools: `query_kg`, `extract_entities`, `get_entity_graph`, `get_inference_chain`, `search_entities`
- [x] Make it optional (can be disabled via MEMINI_KG_ENABLED=false)
- [x] 71 tests passing (40 KG + 31 extractor)

### 4.3 Multi-Peer Profiles ✅ COMPLETE
- [x] Add `peer_id` field to MemoryEntry schema
- [x] Create `src/memini_ai/multi_peer.py` module (~860 lines)
- [x] PeerProfile with PeerRole (OWNER, COLLABORATOR, READONLY, GUEST)
- [x] MemoryPermission (PRIVATE, SHARED, INHERITED)
- [x] Peer context switching and memory sharing
- [x] Expose tools: `list_peers`, `add_peer`, `switch_peer_context`, `share_memory`, `get_peer_memories`, `get_shared_memories`
- [x] Make it optional (can be disabled via MEMINI_MULTI_PEER_ENABLED=false)
- [x] 41 tests passing

### 4.4 Dialectic Reasoning ✅ COMPLETE
- [x] Create `src/memini_ai/dialectic.py` module (~1100 lines)
- [x] DialecticArgument, DialecticResolution, DialecticChallenge dataclasses
- [x] Contradiction detection via CONTRADICTS relationships
- [x] LLM-based pro/con argument generation
- [x] Resolution synthesis with confidence scoring
- [x] Expose tools: `find_contradictions`, `resolve_contradiction`, `get_dialectic_history`, `challenge_memory`
- [x] Make it optional (can be disabled via MEMINI_DIALECTIC_ENABLED=false)
- [x] 36 tests passing

### 4.5 Graph Visualization ✅ COMPLETE
- [x] Add `to_d3_json()` method to KnowledgeGraph for D3.js export
- [x] Add `generate_visualization_html()` function for self-contained HTML
- [x] D3.js v7 force-directed graph with interactive nodes/edges
- [x] Node colors by entity type (PERSON=blue, ORG=green, CONCEPT=purple, CODE=orange, PROJECT=yellow)
- [x] Edge colors by relationship type (SUPERSEDES=red, RELATED_TO=blue, CONTRADICTS=purple, DERIVED_FROM=green)
- [x] Interactive: zoom, pan, drag, hover tooltips
- [x] Dark/light mode support
- [x] New MCP tool: `get_graph_visualization(limit)` returns HTML page
- [x] 645 tests passing

---

## Phase 5: pgvector Migration (COMPLETE)

> **Completed**: Migrated from Qdrant to pgvector for unified database architecture.
> See `docs/pgvector_migration.md` for full context.

### 5.1 Why pgvector?
- **Single DB**: Consolidate memory + user data + relationships on Postgres
- **SQL joins**: Complex queries like "memories contradicting X by user Y" become trivial
- **Transactions**: Full ACID for trust engine, multi-peer permissions
- **Simpler ops**: One connection string, one backup strategy
- **Good enough**: pgvector is ~95% as fast as Qdrant for typical workloads

### 5.2 Scope
- [x] Add `asyncpg` dependency
- [x] Create `src/memini_ai/postgres/` module for pgvector operations
- [x] Create schema migration scripts
- [x] Implement vector similarity search via `pg_vector` extension
- [x] Migrate all Qdrant-specific code to abstraction layer
- [x] Add config: `MEMINI_DB_URL` (postgres://...)

### 5.3 Schema Design
```sql
-- Core memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1024),  -- pgvector
    source_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    trust_score FLOAT DEFAULT 0.5,
    retrieval_count INT DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE,
    peer_id UUID REFERENCES peers(id),
    metadata JSONB
);

-- Memory relationships (instead of JSON field)
CREATE TABLE memory_relationships (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES memories(id),
    target_id UUID REFERENCES memories(id),
    relationship_type VARCHAR(50),
    confidence FLOAT DEFAULT 1.0
);

-- Entities (from knowledge graph)
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name VARCHAR(500),
    entity_type VARCHAR(50),
    canonical_name VARCHAR(500),
    confidence FLOAT,
    peer_id UUID REFERENCES peers(id)
);

-- Entity relationships
CREATE TABLE entity_relationships (
    id UUID PRIMARY KEY,
    source_entity_id UUID REFERENCES entities(id),
    target_entity_id UUID REFERENCES entities(id),
    relationship_type VARCHAR(50)
);

-- Peers (for multi-peer)
CREATE TABLE peers (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    role VARCHAR(50),
    trust_level FLOAT,
    preferences JSONB
);
```

### 5.4 Implementation Order
1. [x] Add postgres dependency, create `postgres/` module
2. [x] Implement `PostgresDatabase` class with vector search
3. [x] Create migration script (Qdrant → Postgres, one-time)
4. [x] Update `MemoryDatabase` to use abstraction layer
5. [x] Update all features to use new DB layer
6. [x] Qdrant kept for backward compatibility (optional removal later)

### 5.5 PostgreSQL Configuration Issues (FIXED in v0.2.7) ✅
- **IF NOT EXISTS**: All CREATE TABLE and CREATE INDEX statements now idempotent
- **Vector parsing**: Fixed `_row_to_memory()` to use `json.loads()` instead of `list()` for vector strings
- **384-dim vectors**: Schema changed from 1024 to 384 to match MiniLM embedding model
- **Test fixtures**: Fixed invalid UUIDs and wrong-dimension vectors in tests

### 5.6 PyPI Publishing (CONFIGURED)
- [x] Trusted publishing configured on PyPI for `memini-ai-dev`
- [x] GitHub Actions workflow uses `environment: pypi`
- [x] Package name updated to `memini-ai-dev` in pyproject.toml

### 5.5 Abstraction Layer
```python
# src/memini_ai/database.py (abstract interface)
class VectorDatabase(ABC):
    @abstractmethod
    async def search_vectors(self, query: np.ndarray, limit: int) -> list[MemoryMatch]: ...
    
    @abstractmethod
    async def insert_memory(self, memory: MemoryEntry) -> str: ...
    
    @abstractmethod
    async def get_memory(self, memory_id: str) -> MemoryEntry | None: ...

# Implementations
class QdrantDatabase(VectorDatabase): ...  # Keep for backward compat
class PostgresDatabase(VectorDatabase): ...  # New implementation
```

### 5.6 Key Benefits After Migration
- Trust Engine: `UPDATE memories SET trust_score = trust_score - 0.02 WHERE last_accessed < NOW() - INTERVAL '7 days'`
- Memory Graph: SQL joins instead of JSON parsing
- Multi-Peer: Foreign keys + row-level security
- Dialectic: `SELECT * FROM memories m1 JOIN memory_relationships r ON m1.id = r.source_id WHERE r.relationship_type = 'CONTRADICTS' AND r.target_id = '...'`
- Decay: Cron-jobable SQL aggregation

---

## Quality Gates

Each phase requires:
1. All tests passing (`pytest`) - **Phase 4: 647 passed, 10 skipped** (Phase 1: 201 + Phase 2: 123 + Phase 3: 119 + Phase 4: 204)
2. Type checking passing (`mypy`)
3. Linting passing (`ruff`)
4. Integration tests with MCP client

---

## Dependencies (Phase 1 - CONFIRMED WORKING)

### Core Dependencies (Installed and Tested)

| TypeScript | Python | Status |
|------------|--------|--------|
| @modelcontextprotocol/sdk | fastmcp | ✅ Working |
| @qdrant/js-client-rest | asyncpg + pgvector | ✅ Working |
| @xenova/transformers | sentence-transformers + torch | ✅ Working |
| chokidar | watchdog | ✅ Working |
| fuse.js | rank-bm25 | ✅ Working |
| zod | pydantic | ✅ Working |

### Phase 2 Additional Dependencies

| Package | Purpose |
|---------|---------|
| ollama (or API) | Local LLM for auto-extraction |
| httpx | Async HTTP for LLM API calls |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MEMINI_DB_URL | PostgreSQL connection URL | postgresql://postgres:password@localhost:5432/postgres |
| MEMINI_PROJECT_ID | Project namespace | auto-generated |
| MEMINI_EMBEDDING_DIM | 1024 or 384 | 1024 |
| MEMINI_DEVICE | auto, gpu, cpu | auto |
| MEMINI_AUTO_EXTRACT | Enable auto-extraction | false |
| MEMINI_TRUST_ENGINE | Enable trust scoring | false |
| MEMINI_MEMORY_GRAPH | Enable memory graph | false |
| MEMINI_DECAY_ENABLED | Enable memory decay | false |
| MEMINI_DECAY_HALF_LIFE_DAYS | Decay half-life | 90 |
| MEMINI_KG_ENABLED | Enable knowledge graph | false |
| MEMINI_KG_INFERENCE_DEPTH | Max inference depth | 3 |
| MEMINI_MULTI_PEER_ENABLED | Enable multi-peer | false |
| MEMINI_DIALECTIC_ENABLED | Enable dialectic reasoning | false |

---

## Reference Documentation

- Super-Memory-TS v2.6.5: `/node_modules/@veedubin/super-memory-ts/dist/`
- Research docs: `memini-ai-dev/docs/research/`
- Memory Report: `memini-ai-dev/docs/memory_report.agent.final.md`
- Reality Check: `memini-ai-dev/docs/reality_check.md`
- **pgvector Migration**: `memini-ai-dev/docs/pgvector_migration.md`
