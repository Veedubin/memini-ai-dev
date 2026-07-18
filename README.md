# memini-ai-dev

[![PyPI version](https://img.shields.io/pypi/v/memini-ai-dev.svg)](https://pypi.org/project/memini-ai-dev/)
[![PyPI](https://img.shields.io/pypi/fm/memini-ai-dev.svg)](https://pypi.org/project/memini-ai-dev/)

> "I remember" in Latin (pronounced *meh-mee-nee*)

Local-first semantic memory server with vector search, trust scoring, and persistent reasoning, fully MCP-compatible.

## Overview

**memini-ai-dev** is a professional-grade semantic memory server designed to provide AI agents with long-term, structured, and trust-weighted memory. It uses PostgreSQL with `pgvector` for high-performance vector search and implements a tiered memory architecture for efficient context management.

### Key Features

- **MCP-Compatible**: Seamless integration with any MCP client (OpenCode, Claude Desktop, etc.)
- **Persistent Thought Chains**: Store and retrieve complex reasoning chains with support for branching and revisions (v0.3.0+)
- **Trust-Weighted Memory**: Dynamic trust scoring for memories based on agent usage and user feedback
- **Tiered Memory Architecture**: Efficient context loading via L0 (Summary), L1 (Key Decisions), and L2 (Full Context)
- **Knowledge Graph**: Entity extraction and relationship tracking with live D3.js visualization
- **Dialectic Reasoning**: Built-in contradiction detection and resolution logic
- **Multi-Peer Sharing**: Share memory subsets across different agent peers
- **Vector Search**: Default 384-dim MiniLM embeddings for speed, with optional BGE-M3 (1024-dim) for higher precision on GPU
- **Memory Decay**: Temporal trust decay to ensure memory relevance over time
- **Project Isolation**: Strict memory separation by project ID
## Multi-Model RRF (v0.7.0+, refined v0.7.6)

memini-ai-dev supports **two embedding models** — a fast CPU-friendly default and a higher-precision GPU upgrade:

| Model | Dim | Use Case | Env Var (`MEMINI_MODEL_NAME`) |
|-------|-----|----------|-------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, lightweight, CPU-friendly (default) | `'minilm'` (alias) or full HF name |
| `BAAI/bge-m3` | 1024 | Higher precision, multi-lingual, GPU-friendly | `'bge-m3'` (alias) or full HF name |

**Recommended migration path**: start with MiniLM (default), get a GPU, then upgrade to BGE-M3 using the migration script in `archives/memini-embedding-migration-2026-07-10/migrate_minilm_to_bge_m3.py`. The MiniLM column is preserved — both vectors coexist for RRF search.

### Embedding Mode Dispatch (`EMBEDDING_MODE`)

| Mode | Behavior | Env Var |
|------|----------|---------|
| `cpu` | 384-dim-only (MiniLM) | `EMBEDDING_MODE=cpu` |
| `auto` | 384-dim writes; queries fuse 384 + 1024 via RRF (k=60) | `EMBEDDING_MODE=auto` (default) |
| `gpu` | 1024-dim-only (BGE-M3) | `EMBEDDING_MODE=gpu` |

### Database Schema

The PostgreSQL schema includes **two vector columns** for multi-model support:

```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    embedding vector(384),           -- MiniLM-L6-v2 (384-dim)
    embedding_bge_m3 vector(1024)   -- BGE-M3 (1024-dim, optional GPU upgrade)
);
```

The `embedding_bge_large` column (BGE-Large, 1024-dim) was removed in v0.7.6. The BGE-Large migration script is kept as a reference example for users who want to do similar migrations on their own (see `archives/memini-embedding-migration-2026-07-10/migrate_to_bge_large.py`).

### v0.7.5 Bug Fixes (Critical for Multi-Model)

The v0.7.5 release fixes **three latent bugs** that prevented the multi-model RRF feature from working end-to-end:

1. **Model Selection**: `ModelManager._load_model()` was constrained by `embedding_dim` instead of `config.model_name`, making BGE-M3 unreachable.
2. **Column Routing**: `add_memory` wrote 1024-dim vectors to the 384-dim `embedding` column — silent data loss for BGE-M3 writes.
3. **RRF Mapping**: RRF `COLUMN_TO_MODEL` used short name `'all-MiniLM-L6-v2'` but `ModelManager` expects full HF name.

**Fixes**: Model name-driven selection with alias support, multi-model column routing (new `INSERT_MEMORY_BGE_M3` query), and full-HF-name RRF column mapping. See [CHANGELOG.md](CHANGELOG.md#075---2026-07-10) for details. v0.7.6 then removed BGE-Large support to keep the codebase clean — see [CHANGELOG.md](CHANGELOG.md#076---2026-07-10).

### Enabling Multi-Model

```bash
# Enable BGE-M3 for new writes + RRF queries (full config)
export MEMINI_MODEL_NAME=BAAI/bge-m3
# or: export MEMINI_MODEL_NAME=bge-m3  # short alias

export MEMINI_EMBEDDING_DIM=1024
export MEMINI_ENABLE_RRF=true
# Start the server
uvx --from memini-ai-dev memini-ai --stdio
```

**Important:** `MEMINI_EMBEDDING_DIM` must match the model's output dimension (1024 for BGE-M3, 384 for MiniLM). A mismatch logs a warning and degrades to text-only search — see [docs/upgrading-embeddings.md](docs/upgrading-embeddings.md).

With `MEMINI_ENABLE_RRF=true`, queries fuse top-k results from each populated model column using **Reciprocal Rank Fusion (RRF)** with k=60 (standard constant).

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ with `pgvector` extension

### Quick Start

```bash
# Install via pip
pip install memini-ai-dev

# Run the server using uvx (Recommended)
uvx --from memini-ai-dev memini-ai --stdio
```

## RBAC — Per-Project Users

### Overview
memini-ai-dev supports per-project PostgreSQL users with auto-generated passwords. By default, all users have read/write/edit access to all memories (open access). Per-project isolation is opt-in via `MEMINI_PEER_ENFORCEMENT=true`.

### New Team DB Setup (`memini-ai init --homedir --team --new-db`)
- Creates a `memini_admin` role with a generated password
- Creates a `memini` database with `pgvector` and `vectorscale` extensions
- Writes admin credentials to `~/.config/opencode/.env` (chmod 600)
- The password is printed once during install and never shown again

### Project User Setup (`memini-ai init --project --team`)
- Reads admin credentials from the homedir `.env`
- Derives a postgres-safe role name from the directory (e.g., `foo_bar_123/` → `foo-bar-123`)
- Creates a project role with CRUD permissions (no superuser)
- Writes project credentials to `./.env` (chmod 600)
- `opencode.json` uses `{env:...}` syntax — secrets are never committed to git

### Default: Open Access
- By default, `MEMINI_PEER_ENFORCEMENT=false` — all users see all memories
- `peer_id` is written on all inserts (for tagging) but does NOT filter reads
- This preserves backward compatibility — existing setups work unchanged

### Opt-in Lockdown
- Set `MEMINI_PEER_ENFORCEMENT=true` and `MEMINI_PEER_ID=<project-name>` in `.env`
- Queries filter to only show memories with matching `peer_id` (or `NULL` peer_id = visible to all)
- Lockdown is per-project, per-device — you choose which projects are isolated

### User Management
```bash
memini-ai user add --name <name> [--admin]    # Add a user
memini-ai user list                            # List all users
memini-ai user remove --name <name>            # Drop a user
```

### The .env File Pattern
- **Homedir `.env** (`~/.config/opencode/.env`): admin credentials
- **Project `.env** (`./.env`): project credentials
- `opencode.json` references via `{env:MEMINI_PROJECT_USER}` etc.
- Never put passwords in `opencode.json` (it may be committed to git)
- `.env` files are chmod 600

## SSL/TLS Database Connections

### Overview
memini-ai-dev supports SSL/TLS for PostgreSQL team server connections.

### SSL Modes (during `memini-ai init --team`)
- `prefer` (default) — use SSL if available, fall back to plain
- `require` — always use SSL, reject non-SSL connections
- `verify-ca` — verify server certificate (CA)
- `verify-full` — verify server certificate (CA + hostname)
- `disable` — never use SSL

### Configuration
The SSL mode is stored in `.env` as `MEMINI_DB_SSLMODE` and appended to the connection URL as `?sslmode={mode}`.

### For Production
Use `verify-full` with a proper CA certificate. The PostgreSQL server must be configured with `ssl = on` and a valid certificate.

## Container Runtime Detection

### Overview
When using team server mode, memini-ai-dev can detect and help install container runtimes.

### Supported Runtimes
- **Podman** (recommended) — rootless, daemonless, no security surface. `podman-docker` makes docker commands work transparently.
- **Docker** — industry standard, most tutorials assume it. Requires a background daemon.
- **Containerd** — lightweight, for CI/headless.

### Auto-Install
If no container runtime is found during team server setup, the installer offers to install one:
```
No container runtime found. Which would you like to install?

  1. Podman (recommended)
     Rootless and daemonless — no background service needed.
     More secure: containers run as your user, not root.
     Works with docker commands via podman-docker.

  2. Docker
     The industry standard — most tutorials assume it.
     Requires a background daemon (dockerd).

  3. Skip — I'll install it myself

  Enter 1, 2, or 3 [1]:
```

### podman-docker
When podman is installed, `podman-docker` is also installed so `docker` commands work with podman's rootless backend.

### pgembed Mode
Container runtimes are NOT needed for pgembed (embedded PostgreSQL). The detection only runs for team server mode.

## Backend Selection

v1.0.0 ships with two backends:

| Backend | Use case | Data location | Vectorscale? |
|---------|----------|---------------|--------------|
| `pgembed` (default) | Solo user, single machine | `~/.local/share/memini-ai/pgembed/data` | ✅ (DiskANN) |
| `postgres-external` | Team server, Docker, managed | Wherever your `MEMINI_DB_URL` points | ✅ (DiskANN) |

**Switch to external mode** (v0.8.2 behavior):
```bash
export MEMINI_VECTOR_BACKEND=postgres-external
export MEMINI_DB_URL=postgresql://user:pass@host:port/db
```

**Use both with RRF fusion**:
```bash
export MEMINI_VECTOR_BACKEND=pgembed
export MEMINI_TEAM_DB_URL=postgresql://team-server/db
export MEMINI_FUSION_MODE=rrf
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Veedubin/memini-ai-dev.git
cd memini-ai-dev

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Configuration

Configured via environment variables or a JSON config file.

### Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_DB_URL` | (empty) | PostgreSQL connection URL (set via `.env`, see `.env.example`). **Ignored if `MEMINI_VECTOR_BACKEND=pgembed`.** |
| `MEMINI_PROJECT_ID` | auto-generated | Project identifier for isolation |
| `MEMINI_EMBEDDING_DIM` | `384` | Embedding dimension (384 for MiniLM, 1024 for BGE-M3) |
| `MEMINI_CHUNK_SIZE` | `512` | Chunk size for file indexing |
| `MEMINI_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MEMINI_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `MEMINI_AUTO_DETECT_MODEL` | `true` | Auto-upgrade new deployments to BGE-M3 (1024-dim) |
| `MEMINI_STRICT_EMBEDDING_DIM` | `false` | Dim mismatch raises RuntimeError (opt-in crash) |
| `MEMINI_MODEL_NAME` | `all-MiniLM-L6-v2` | Active model (MiniLM, BGE-M3, or custom) |
| `MEMINI_ENABLE_RRF` | `true` | Enable RRF fusion across model spaces |
| `RRF_TOP_K_PER_MODEL` | `20` | Results per model before RRF fusion |
| `MEMINI_ENABLED_MODELS` | `["all-MiniLM-L6-v2", "BAAI/bge-m3"]` | Models for RRF dispatch |
| `MEMINI_VECTOR_BACKEND` | `pgembed` | Vector backend: `pgembed` (default) or `postgres-external` |
| `MEMINI_PGEMBED_DATA_DIR` | `~/.local/share/memini-ai/pgembed/data` | Data directory for embedded Postgres |
| `MEMINI_TEAM_DB_URL` | (empty) | Team server URL for RRF fusion |
| `MEMINI_FUSION_MODE` | (empty) | Fusion mode: `rrf` (Reciprocal Rank Fusion) |
| `MEMINI_WORKERS` | (cpu_count) | Number of worker threads |
| `MEMINI_LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `MEMINI_DEVICE` | `auto` | Device for embeddings (`auto`, `cpu`, `cuda`) |
| `MEMINI_CONFIG_PATH` | None | Path to JSON config file |

#### ⭐ Advanced Feature Toggles (Disabled by Default)

Set to `true` to enable these professional memory capabilities.

| Feature | Env Var | Description |
|---------|---------|-------------|
| **Thought Chains** | `THOUGHT_CHAINS` | Persistent reasoning with branching/revision |
| **Trust Engine** | `MEMINI_TRUST_ENGINE` | Trust scoring and archive/promotion logic |
| **Tiered Loading** | `MEMINI_TIERED_LOADING` | L0/L1/L2 summary generation |
| **Knowledge Graph** | `MEMINI_KG_ENABLED` | Entity extraction and KG queries |
| **Memory Graph** | `MEMINI_MEMORY_GRAPH` | Visual relationship mapping |
| **Dialectic** | `MEMINI_DIALECTIC_ENABLED` | Contradiction detection and resolution |
| **Multi-Peer** | `MEMINI_MULTI_PEER_ENABLED` | Peer-to-peer memory sharing |
| **User Modeling** | `MEMINI_USER_MODELING` | Persistent user profile and style tracking |
| **Memory Decay** | `MEMINI_DECAY_ENABLED` | Temporal trust decay engine |
| **Auto-Extract** | `MEMINI_AUTO_EXTRACT` | Automatic memory extraction from conversations |
| **Pre-Compression** | `MEMINI_PRECOMPRESS` | Context-aware pre-compression extraction |

#### Trust Engine Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_TRUST_THRESHOLD_ARCHIVE` | `0.2` | Archive memories below this trust |
| `MEMINI_TRUST_THRESHOLD_PROMOTE` | `0.8` | Promote to L1 above this trust |
| `MEMINI_TRUST_DELTA_USE` | `+0.05` | Trust delta for `agent_used` signal |
| `MEMINI_TRUST_DELTA_IGNORED` | `-0.05` | Trust delta for `agent_ignored` signal |
| `MEMINI_TRUST_DELTA_CORRECT` | `-0.15` | Trust delta for `user_corrected` signal |
| `MEMINI_TRUST_DELTA_CONFIRM` | `+0.10` | Trust delta for `user_confirmed` signal |

#### TLS/SSL Configuration

PostgreSQL connections support TLS encryption to prevent data exfiltration on the network.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_SSLMODE` | `prefer` | PostgreSQL SSL mode — see table below |
| `DB_SSLROOTCERT` | _(empty)_ | Path to CA certificate for server verification |

**SSL Mode Values** (from [libpq docs](https://www.postgresql.org/docs/current/libpq-ssl.html)):

| Value | Encryption | Server Cert Verified | Hostname Verified | Use Case |
|-------|-----------|---------------------|-------------------|----------|
| `disable` | No | No | No | Development only (no TLS) |
| `allow` | Optional | No | No | Rarely useful |
| `prefer` | Tried first | No | No | **Default** — fallback to plaintext |
| `require` | Yes | No | No | Encrypted but no identity check |
| `verify-ca` | Yes | Yes | No | CA verified, hostname not checked |
| `verify-full` | Yes | Yes | Yes | **Recommended for production** |

**Quick Start — Local Development with TLS:**

```bash
# 1. Generate self-signed certificates
cd memini-ai-dev
./scripts/generate-local-certs.sh

# 2. Configure PostgreSQL to use the generated certs
#    (see docker-compose.yml for commented SSL config)

# 3. Set environment variables
export DB_SSLMODE=require
export DB_SSLROOTCERT=/path/to/memini-ai-dev/certs/ca.crt

# 4. Start the server
uvx --from memini-ai-dev memini-ai --stdio
```

**Production Recommendation:** Use `DB_SSLMODE=verify-full` with certificates from a trusted CA, not self-signed certificates.

## Usage

### MCP Tools (52)

`memini-ai-dev` provides a comprehensive suite of tools categorized by capability:

#### 🧠 Basic Memory
- `query_memories`: Semantic search with tiered strategy.
- `add_memory`: Store new memories with source tracking.
- `delete_memory`: Remove specific memory entries.
- `get_memory`: Fetch a memory by ID.

#### 📁 Project Indexing
- `search_project`: Semantic search across indexed project files.
- `index_project`: Trigger recursive project indexing.
- `get_file_contents`: Reconstruct files from semantic chunks.
- `get_indexing_status`: Check progress of background indexing.

#### 📈 Trust & Tiering
- `get_trust_score`: Retrieve trust level for a memory.
- `adjust_trust`: Manually apply feedback signals.
- `get_tier0_summary`: Get high-trust L0 project summary.
- `get_tier1_summary`: Get L1 key decisions summary.

#### ⛓️ Thought Chains (v0.3.0)
- `start_thought_chain`: Initialize a new reasoning chain.
- `add_thought`: Add a step to a chain (supports revisions/branching).
- `get_thought_chain`: Retrieve a full reasoning tree.
- `abandon_thought_chain`: Mark a reasoning path as incorrect.

#### 🕸️ Knowledge Graph & Dialectic
- `query_kg`: Execute formal KG queries.
- `extract_entities`: Extract entities from a memory.
- `get_entity_graph`: Find all connections for an entity.
- `find_contradictions`: Detect conflicting memories.
- `resolve_contradiction`: Generate a dialectic resolution.

#### 👥 Multi-Peer & User Modeling
- `share_memory`: Share a memory with another peer.
- `get_peer_memories`: Search another peer's accessible memory.
- `get_user_profile`: Retrieve the learned user style profile.
- `update_user_profile`: Update profile from current conversation.

#### 🛠️ System & Maintenance
- `get_status`: Health check for all server components (v0.7.7: reports modelName, modelDimension, embeddingDimMismatch).
- `trigger_consolidation`: Manually merge similar memories.
- `get_decay_status`: View fading memories and decay stats.
- `log_audit_event`: Manually log an audit event.
- `get_audit_log`: Query audit log with filters.
- `get_security_summary`: Get aggregated security metrics.
- `healthcheck`: End-to-end write+read health probe.

#### 🔬 Advanced Memory Operations
- `elevate_memory_to_1024`: Promote a memory to 1024-dim space (v0.7.0+).
- `find_related_memories`: Find memories related to a given memory.
- `get_relationship_summary`: Get all relationships for a memory.
- `get_shared_memories`: Get all memories shared with the current peer.
- `list_archived`: List archived memories.
- `list_fading_memories`: List memories approaching archive threshold.
- `list_peers`: List all known peers.
- `adjust_decay_rate`: Adjust decay rate for a specific memory.
- `get_dialectic_history`: Get dialectic history for a memory.
- `get_graph_visualization`: Get an HTML visualization of the knowledge graph.
- `get_inference_chain`: Find inference paths between two entities.
- `search_entities`: Search for entities by name.

#### ⛓️ Thought Chain Management (v0.7.0+)
- `branch_thought`: Start a new branch from an existing thought.
- `pause_thought_chain`: Pause a thought chain.
- `resume_thought_chain`: Resume a paused thought chain.
- `revise_thought`: Create a revision of an existing thought.
- `get_related_chains`: Search for thought chains with similar reasoning.

### Python API

```python
from memini_ai.memory.system import MemorySystem
from memini_ai.memory.schema import MemoryEntry, MemorySourceType, SearchOptions, SearchStrategy

async def main():
    system = MemorySystem()
    await system.initialize()

    # Add a memory
    entry = MemoryEntry(
        text="Python list comprehension tutorial",
        source_type=MemorySourceType.session,
    )
    memory_id = await system.add_memory(entry)

    # Query memories using Tiered strategy
    options = SearchOptions(topK=10, strategy=SearchStrategy.TIERED)
    results = await system.query_memories("list comprehension", options)

asyncio.run(main())
```

## Docker Compose

For local development with PostgreSQL/pgvector:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  memini-ai:
    build: .
    depends_on:
      - postgres
    environment:
      - MEMINI_DB_URL=postgresql://user:password@postgres:5432/postgres  # Set via .env
      - THOUGHT_CHAINS=true
    volumes:
      - .:/app

volumes:
  postgres_data:
```

## Testing & Quality

```bash
# Run all tests
pytest tests/ -v

# Run integration tests (requires PostgreSQL with pgvector)
pytest tests/integration/ -v

# Quality Gate: Lint and Type Check
ruff check src/
mypy src/
```

## Architecture

```
memini_ai/
├── config.py           # Configuration & Env Var management (v0.7.7: auto-detect, strict-dim)
├── server.py          # FastMCP server (52 tools, v0.7.7: elevate_memory_to_1024, RRF)
├── api/
│   ├── visualization.py  # FastAPI server for live KG visualization
│   └── d3_template.py     # D3.js visualization template
├── audit/
│   └── logger.py      # Audit logging (v0.7.0+)
├── decay.py           # Temporal trust decay engine (v0.7.0+)
├── dialectic.py       # Contradiction detection and resolution (v0.7.0+)
├── entity_extractor.py # Entity extraction (v0.7.0+)
├── extractor.py       # Auto-extraction from conversations (v0.7.0+)
├── graph.py           # Knowledge Graph (v0.7.0+)
├── indexer/
│   ├── chunker.py     # Semantic chunking logic
│   ├── constants.py   # Indexer constants (v0.7.0+)
│   ├── file_tracker.py # SQLite persistence for index state
│   ├── indexer.py     # ProjectIndexer
│   ├── pause_controller.py # Indexer pause/resume (v0.7.0+)
│   └── watcher.py     # Inotify-based file watching
├── knowledge_graph.py # Knowledge Graph queries (v0.7.0+)
├── llm/
│   ├── base.py        # LLM base class (v0.7.0+)
│   ├── factory.py    # LLM factory (v0.7.0+)
│   ├── __init__.py   # LLM package
│   ├── ollama.py     # Ollama LLM (v0.7.0+)
│   └── openai_compat.py # OpenAI-compatible LLM (v0.7.0+)
├── memory/
│   ├── database.py    # VectorDatabase ABC
│   ├── merger.py      # Memory consolidation (v0.7.0+)
│   ├── rrf.py        # Reciprocal Rank Fusion (v0.7.0+)
│   ├── schema.py     # Pydantic models & MemoryEntry
│   ├── search.py     # Tiered, Vector, Text, and Parallel strategies (v0.7.7: BM25 punctuation guard)
│   └── system.py     # MemorySystem coordinator (v0.7.7: RRF, dim-mismatch fallback)
├── model/
│   ├── embeddings.py  # Embedding generation logic (v0.7.7: dim-mismatch handling)
│   ├── __init__.py   # Model package
│   └── manager.py    # ModelManager singleton (v0.7.7: auto-detect, model aliases)
├── multi_peer.py      # Peer-to-peer memory sharing (v0.7.0+)
├── postgres/
│   ├── database.py   # PostgresDatabase implementation (v0.7.7: memories_1024 table)
│   ├── __init__.py   # Postgres package
│   ├── queries.py    # SQL query builders (v0.7.7: 6 new 1024-dim queries)
│   └── schema.py     # SQL schema definitions (pgvector, v0.7.7: 1024-dim column)
├── precompress.py     # Context-aware pre-compression extraction (v0.7.0+)
├── rate_limiter.py    # Rate limiting (v0.7.0+)
├── thought_chains.py  # Persistent reasoning with branching/revision (v0.7.0+)
├── tiered_loader.py   # L0/L1/L2 summary generation (v0.7.0+)
├── trust_engine.py    # Trust scoring and archive/promotion logic (v0.7.0+)
├── user_model.py      # Persistent user profile and style tracking (v0.7.0+)
├── utils/
│   ├── hash.py       # Content hashing
│   ├── __init__.py   # Utils package
│   ├── logger.py     # Structured logging
│   └── sanitizer.py  # Input sanitization (v0.7.0+)
└── __init__.py        # Package init
    └── hash.py        # SHA-256 utilities
```

## License

MIT License - see LICENSE file for details.

## Links

- [PyPI Project](https://pypi.org/project/memini-ai-dev/)
- [GitHub Repository](https://github.com/Veedubin/memini-ai-dev)
- [pgvector](https://github.com/pgvector/pgvector)
- [FastMCP](https://github.com/jlowin/fastmcp)
