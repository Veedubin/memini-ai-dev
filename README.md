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
- **Vector Search**: Default 384-dim MiniLM embeddings for speed, with BGE-Large (1024-dim) support
- **Memory Decay**: Temporal trust decay to ensure memory relevance over time
- **Project Isolation**: Strict memory separation by project ID

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
| `MEMINI_DB_URL` | (empty) | PostgreSQL connection URL (e.g., `postgresql://postgres:password@localhost:5432/postgres`) |
| `MEMINI_PROJECT_ID` | auto-generated | Project identifier for isolation |
| `MEMINI_EMBEDDING_DIM` | `384` | Embedding dimension (384 for MiniLM, 1024 for BGE-Large) |
| `MEMINI_CHUNK_SIZE` | `512` | Chunk size for file indexing |
| `MEMINI_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MEMINI_BATCH_SIZE` | `32` | Batch size for embedding generation |
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

## Usage

### MCP Tools (35+)

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
- `get_status`: Health check for all server components.
- `trigger_consolidation`: Manually merge similar memories.
- `get_decay_status`: View fading memories and decay stats.

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
      - MEMINI_DB_URL=postgresql://postgres:password@postgres:5432/postgres
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
├── config.py           # Configuration & Env Var management
├── server.py          # FastMCP server (35+ tool implementations)
├── api/
│   ├── visualization.py  # FastAPI server for live KG visualization
│   └── d3_template.py     # D3.js visualization template
├── memory/
│   ├── schema.py      # Pydantic models & MemoryEntry
│   ├── database.py    # VectorDatabase ABC
│   ├── search.py      # Tiered, Vector, Text, and Parallel strategies
│   └── system.py      # MemorySystem coordinator
├── postgres/          # PostgreSQL/pgvector backend
│   ├── database.py    # PostgresDatabase implementation
│   ├── schema.py      # SQL schema definitions (pgvector)
│   └── queries.py     # SQL query builders
├── model/
│   ├── manager.py     # ModelManager singleton (MiniLM/BGE)
│   └── embeddings.py  # Embedding generation logic
├── indexer/
│   ├── indexer.py     # ProjectIndexer
│   ├── chunker.py     # Semantic chunking logic
│   ├── watcher.py     # Inotify-based file watching
│   └── file_tracker.py # SQLite persistence for index state
└── utils/
    ├── logger.py      # Structured logging
    └── hash.py        # SHA-256 utilities
```

## License

MIT License - see LICENSE file for details.

## Links

- [PyPI Project](https://pypi.org/project/memini-ai-dev/)
- [GitHub Repository](https://github.com/Veedubin/memini-ai-dev)
- [pgvector](https://github.com/pgvector/pgvector)
- [FastMCP](https://github.com/jlowin/fastmcp)
