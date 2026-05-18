# Memini-ai v3.0

> "I remember" in Latin (pronounced *meh-mee-nee*)

Local-first semantic memory server with vector search, MCP-compatible.

## Overview

Memini-ai is a Python rewrite of [Super-Memory-TS](https://github.com/veedubin/super-memory-ts), designed as a local-first semantic memory server with vector search capabilities. It provides persistent memory storage and retrieval using Qdrant as the backend vector database.

### Key Features

- **MCP-Compatible**: Works with any MCP-compatible client (OpenCode, Claude Desktop, etc.)
- **Vector Search**: BGE-Large embeddings (1024-dim) with MiniLM fallback (384-dim)
- **Hybrid Search**: Combines vector similarity with BM25 text search
- **Project Isolation**: Memories are isolated by project ID
- **File Indexing**: Index and search project files with semantic chunking
- **Graceful Degradation**: Works without Qdrant (returns errors for memory operations)
- **CPU-First**: Designed to run on CPU, optional GPU acceleration

## Installation

### Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/documentation/guides/installation/) running locally or remotely

### Quick Start

```bash
# Install memini-ai
pip install memini-ai

# Start Qdrant (if not running)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Run the server
memini-ai --stdio
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/veedubin/memini-ai.git
cd memini-ai

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

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMINI_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `MEMINI_PROJECT_ID` | auto-generated | Project identifier for isolation |
| `MEMINI_EMBEDDING_DIM` | `1024` | Embedding dimension (1024 or 384) |
| `MEMINI_CHUNK_SIZE` | `512` | Chunk size for file indexing |
| `MEMINI_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MEMINI_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `MEMINI_WORKERS` | `4` | Number of worker threads |
| `MEMINI_LOG_LEVEL` | `INFO` | Logging level |
| `MEMINI_CONFIG_PATH` | None | Path to JSON config file |

### JSON Config File

```json
{
  "qdrant": {
    "url": "http://localhost:6333"
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

For local development with Qdrant:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  memini-ai:
    build: .
    depends_on:
      - qdrant
    environment:
      - MEMINI_QDRANT_URL=http://qdrant:6333
    volumes:
      - .:/app

volumes:
  qdrant_data:
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

# Run integration tests (requires Qdrant)
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src/memini_ai --cov-report=term-missing
```

### Integration Tests with Docker

```bash
# Start Qdrant for integration tests
docker run -d --name qdrant-test -p 6333:6333 qdrant/qdrant

# Run integration tests
pytest tests/integration/ -v

# Cleanup
docker stop qdrant-test && docker rm qdrant-test
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
├── server.py          # FastMCP server (6 tools)
├── memory/
│   ├── schema.py      # Pydantic models
│   ├── database.py    # Qdrant CRUD operations
│   ├── search.py      # 4 search strategies
│   └── system.py      # MemorySystem coordinator
├── model/
│   ├── manager.py     # ModelManager singleton
│   └── embeddings.py # Embedding generation
├── indexer/
│   ├── indexer.py     # ProjectIndexer
│   ├── chunker.py     # Semantic chunking
│   ├── watcher.py     # File watching
│   └── file_tracker.py # SQLite persistence
└── utils/
    ├── logger.py     # Structured logging
    └── hash.py       # SHA-256 utilities
```

## License

MIT License - see LICENSE file for details.

## Links

- [Repository](https://github.com/veedubin/memini-ai)
- [Documentation](https://github.com/veedubin/memini-ai#readme)
- [Qdrant](https://qdrant.tech/)
- [FastMCP](https://github.com/jlowin/fastmcp)