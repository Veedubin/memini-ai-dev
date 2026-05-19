# Memini-ai

> "I remember" in Latin (pronounced *meh-mee-nee*)

Local-first semantic memory server with vector search, MCP-compatible.

## Overview

Memini-ai is a Python rewrite of [Super-Memory-TS](https://github.com/veedubin/super-memory-ts), designed as a local-first semantic memory server with vector search capabilities. It provides persistent memory storage and retrieval using PostgreSQL with pgvector as the backend.

### Key Features

- **MCP-Compatible**: Works with any MCP-compatible client (OpenCode, Claude Desktop, etc.)
- **Vector Search**: BGE-Large embeddings (1024-dim) with MiniLM fallback (384-dim)
- **Hybrid Search**: Combines vector similarity with BM25 text search
- **Project Isolation**: Memories are isolated by project ID
- **File Indexing**: Index and search project files with semantic chunking
- **Knowledge Graph**: Live D3.js visualization of entities and relationships
- **CPU-First**: Designed to run on CPU, optional GPU acceleration

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension

### Quick Start

```bash
# Install memini-ai
pip install memini-ai-dev

# Run the server
memini-ai --stdio
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

Memini-ai can be configured via environment variables or a JSON config file.

### Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_DB_URL` | (empty) | PostgreSQL connection URL (e.g., `postgresql://postgres:password@localhost:5434/postgres`) |
| `MEMINI_PROJECT_ID` | auto-generated | Project identifier for isolation |
| `MEMINI_EMBEDDING_DIM` | `1024` | Embedding dimension (1024 for BGE-Large or 384 for MiniLM) |
| `MEMINI_CHUNK_SIZE` | `512` | Chunk size for file indexing |
| `MEMINI_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MEMINI_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `MEMINI_WORKERS` | (cpu_count) | Number of worker threads |
| `MEMINI_LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `MEMINI_DEVICE` | `auto` | Device for embeddings (`auto`, `cpu`, `cuda`) |
| `MEMINI_CONFIG_PATH` | None | Path to JSON config file |

#### PostgreSQL Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_TABLE_NAME` | `memories` | Qdrant table name (legacy) |
| `MEMINI_DB_POOL_SIZE` | `10` | Connection pool size |
| `MEMINI_DB_MIN_SIZE` | `2` | Min pool connections |
| `MEMINI_DB_MAX_SIZE` | `20` | Max pool connections |

#### ⭐ Optional Features (all disabled by default)

All optional features are **disabled by default**. Set to `true` or appropriate value to enable.

##### Trust Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_TRUST_ENGINE` | `false` | **Enable trust engine** |
| `MEMINI_TRUST_THRESHOLD_ARCHIVE` | `0.2` | Archive memories below this trust |
| `MEMINI_TRUST_THRESHOLD_PROMOTE` | `0.8` | Promote to L1 above this trust |
| `MEMINI_TRUST_DELTA_USE` | `+0.05` | Trust delta for `agent_used` signal |
| `MEMINI_TRUST_DELTA_IGNORED` | `-0.02` | Trust delta for `agent_ignored` signal |
| `MEMINI_TRUST_DELTA_CORRECT` | `-0.15` | Trust delta for `user_corrected` signal |
| `MEMINI_TRUST_DELTA_CONFIRM` | `+0.10` | Trust delta for `user_confirmed` signal |

##### Memory Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_MEMORY_GRAPH` | `false` | **Enable memory graph** |
| `MEMINI_GRAPH_ENTITY_EXTRACTION` | `true` | Auto extract entities from memories |
| `MEMINI_GRAPH_RELATIONSHIP_SUGGESTIONS` | `true` | Suggest relationships between memories |

##### Knowledge Graph

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_KG_ENABLED` | `false` | **Enable knowledge graph** (required for live visualization) |
| `MEMINI_KG_ENTITY_EXTRACTION` | `true` | Auto extract entities from memories |
| `MEMINI_KG_INFERENCE_DEPTH` | `3` | Max inference path depth |
| `MEMINI_KG_MAX_RESULTS` | `100` | Max knowledge graph query results |

##### Tiered Loading

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_TIERED_LOADING` | `false` | **Enable tiered loading** (L0/L1/L2 summaries) |
| `MEMINI_TIER0_MAX_TOKENS` | `100` | L0 summary max tokens (~100 tokens) |
| `MEMINI_TIER1_MAX_TOKENS` | `2000` | L1 summary max tokens (~2K tokens) |
| `MEMINI_TIER0_CACHE_TTL` | `3600` | L0 cache TTL in seconds |
| `MEMINI_TIER1_CACHE_TTL` | `7200` | L1 cache TTL in seconds |

##### Auto-Extract

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_AUTO_EXTRACT` | `false` | **Enable automatic memory extraction** |
| `MEMINI_AUTO_EXTRACT_TURNS` | `5` | Conversation turns between auto-extractions |
| `MEMINI_LLM_URL` | `http://localhost:11434/api/generate` | LLM endpoint for extraction |
| `MEMINI_LLM_MODEL` | `llama3.2` | LLM model name |

##### Pre-Compression

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_PRECOMPRESS` | `false` | **Enable pre-compression extraction** |
| `MEMINI_PRECOMPRESS_THRESHOLD` | `0.8` | Threshold for pre-compression |

##### User Modeling

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_USER_MODELING` | `false` | **Enable user modeling** |
| `MEMINI_USER_MODEL_MIN_SESSIONS` | `50` | Min sessions to build user profile |

##### Memory Decay

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_DECAY_ENABLED` | `false` | **Enable memory decay** |
| `MEMINI_DECAY_HALF_LIFE_DAYS` | `90` | Memory half-life in days |
| `MEMINI_CONSOLIDATION_INTERVAL_HOURS` | `168` | Consolidation interval (hours) |
| `MEMINI_CONSOLIDATION_SIMILARITY_THRESHOLD` | `0.92` | Merge similar memories above this |

##### Multi-Peer

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_MULTI_PEER_ENABLED` | `false` | **Enable multi-peer sharing** |
| `MEMINI_MULTI_PEER_GUEST_SHARING` | `true` | Allow guest memory sharing |

##### Dialectic (Contradiction Detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_DIALECTIC_ENABLED` | `false` | **Enable dialectic reasoning** |
| `MEMINI_DIALECTIC_LLM_PROVIDER` | `ollama` | LLM provider for dialectic |
| `MEMINI_DIALECTIC_LLM_MODEL` | `llama3` | LLM model for dialectic |
| `MEMINI_DIALECTIC_AUTO_THRESHOLD` | `0.5` | Auto-trigger threshold |

#### Quick Start with All Features Enabled

```bash
export MEMINI_DB_URL="postgresql://postgres:password@localhost:5434/postgres"
export MEMINI_EMBEDDING_DIM=384
export MEMINI_TRUST_ENGINE=true
export MEMINI_MEMORY_GRAPH=true
export MEMINI_KG_ENABLED=true
export MEMINI_TIERED_LOADING=true
export MEMINI_AUTO_EXTRACT=true
export MEMINI_PRECOMPRESS=true
export MEMINI_USER_MODELING=true
export MEMINI_DECAY_ENABLED=true
export MEMINI_MULTI_PEER_ENABLED=true
export MEMINI_DIALECTIC_ENABLED=true

# Run the server
memini-ai --stdio
```

### JSON Config File

```json
{
  "database": {
    "url": "postgresql://postgres:password@localhost:5432/postgres"
  },
  "model": {
    "embedding_dim": 1024
  },
  "indexer": {
    "chunk_size": 512,
    "chunk_overlap": 64
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Usage

### MCP Tools

Memini-ai provides 6 MCP tools:

#### `query_memories`
Semantic search over memories.

```json
{
  "query": "What files were modified yesterday?",
  "limit": 10,
  "strategy": "tiered"
}
```

**Strategies:**
- `tiered` (default): MiniLM primary + BGE fallback
- `vector_only`: Pure semantic similarity
- `text_only`: BM25 keyword search
- `parallel`: Dual-tier with RRF fusion

#### `add_memory`
Store a new memory entry.

```json
{
  "content": "Remember to update the config file",
  "sourceType": "session",
  "sourcePath": "/path/to/file",
  "metadata": {"key": "value"}
}
```

#### `search_project`
Search indexed project files.

```json
{
  "query": "authentication middleware",
  "topK": 20,
  "fileTypes": [".py", ".ts"],
  "paths": ["src/"]
}
```

#### `index_project`
Trigger project indexing.

```json
{
  "path": ".",
  "force": false,
  "background": true
}
```

#### `get_file_contents`
Reconstruct a file from indexed chunks.

```json
{
  "filePath": "src/main.py",
  "triggerIndex": false
}
```

#### `get_status`
Get server component status.

```json
{}
```

### Python API

```python
from memini_ai.memory.system import MemorySystem
from memini_ai.memory.schema import MemoryEntry, MemorySourceType, SearchOptions, SearchStrategy

async def main():
    # Create and initialize
    system = MemorySystem()
    await system.initialize()

    # Add a memory
    entry = MemoryEntry(
        text="Python list comprehension tutorial",
        source_type=MemorySourceType.session,
    )
    memory_id = await system.add_memory(entry)

    # Query memories
    options = SearchOptions(topK=10, strategy=SearchStrategy.TIERED)
    results = await system.query_memories("list comprehension", options)

    # Delete memory
    await system.delete_memory(memory_id)

asyncio.run(main())
```

## Docker Compose

For local development with PostgreSQL/pgvector:

```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
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
    volumes:
      - .:/app

volumes:
  postgres_data:
```

```bash
docker-compose up -d
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only (skip integration)
pytest tests/ -v --ignore=tests/integration/

# Run integration tests (requires PostgreSQL with pgvector)
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src/memini_ai --cov-report=term-missing
```

### Integration Tests with Docker

```bash
# Start PostgreSQL with pgvector for integration tests
docker run -d --name postgres-test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=memini_test -p 5432:5432 pgvector/pgvector:pg16

# Run integration tests
pytest tests/integration/ -v

# Cleanup
docker stop postgres-test && docker rm postgres-test
```

## Quality Gates

Before submitting changes, ensure:

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/

# Tests
pytest tests/ -x
```

## Performance

Memini-ai is designed for sub-10ms query latency on cached queries.

Typical performance:
- **Query latency**: < 10ms (after warmup)
- **Indexing**: ~1000 files/second
- **Memory footprint**: ~500MB (without model)

### Performance Tuning

```bash
# Use faster embedding model (384-dim instead of 1024)
export MEMINI_EMBEDDING_DIM=384

# Increase workers for faster indexing
export MEMINI_WORKERS=8

# Larger batch size for embedding generation
export MEMINI_BATCH_SIZE=64
```

## Architecture

```
memini_ai/
├── config.py           # Configuration management
├── server.py          # FastMCP server (35 tools)
├── api/
│   ├── visualization.py  # FastAPI server for live KG visualization
│   └── d3_template.py     # D3.js visualization template
├── memory/
│   ├── schema.py      # Pydantic models
│   ├── database.py    # VectorDatabase ABC
│   ├── search.py      # 4 search strategies
│   └── system.py      # MemorySystem coordinator
├── postgres/          # PostgreSQL/pgvector backend
│   ├── database.py    # PostgresDatabase implementation
│   ├── schema.py      # SQL schema definitions
│   └── queries.py     # SQL query builders
├── model/
│   ├── manager.py     # ModelManager singleton
│   └── embeddings.py  # Embedding generation
├── indexer/
│   ├── indexer.py     # ProjectIndexer
│   ├── chunker.py     # Semantic chunking
│   ├── watcher.py     # File watching
│   └── file_tracker.py # SQLite persistence
└── utils/
    ├── logger.py      # Structured logging
    └── hash.py        # SHA-256 utilities
```

## License

MIT License - see LICENSE file for details.

## Links

- [Repository](https://github.com/Veedubin/memini-ai-dev)
- [Documentation](https://github.com/Veedubin/memini-ai-dev#readme)
- [pgvector](https://github.com/pgvector/pgvector)
- [FastMCP](https://github.com/jlowin/fastmcp)