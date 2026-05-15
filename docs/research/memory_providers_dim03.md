# Dimension 3: Super-Memory-TS Deep Analysis

## Overview
- **Repo**: https://github.com/Veedubin/Super-Memory-TS
- **NPM**: @veedubin/super-memory-ts
- **License**: MIT
- **Version**: v2.6.5 (actively developed, 74 commits)
- **By**: Veedubin / Frozen Maple Labs
- **Language**: TypeScript (96.4%)

## Core Architecture
- **Type**: MCP (Model Context Protocol) server
- **Database**: Qdrant (HNSW indexing, payload filtering)
- **Embeddings**: BGE-Large (GPU, 1024-dim) / MiniLM-L6-v2 (CPU, 384-dim)
- **Precision**: fp16 (~325MB), fp32 (~650MB), q8 (~162MB), q4 (~81MB)
- **Vector Search**: HNSW, sub-10ms query latency
- **Local-first**: All data stays on your machine

## Key Capabilities

### 1. Semantic Memory Search
Store and retrieve memories using natural language queries with embedding-based similarity search.

### 2. Project Indexing
Automatically index project files for code-aware search:
- Supports TypeScript, JavaScript, Python, Markdown, JSON, and more
- Semantic chunking at function/class boundaries
- Incremental updates via SHA-256 hash comparison
- File watching with 500ms debounce (chokidar)

### 3. Project Isolation
Multi-project memory isolation using `projectId` tagging in Qdrant payloads. Set via `BOOMERANG_PROJECT_ID` env var.

### 4. Tiered Search Strategies
| Mode | Strategy | Best For |
|------|----------|----------|
| **TIERED** (default) | MiniLM primary + BGE fallback | Interactive queries, real-time responses |
| **PARALLEL** | Dual-tier with RRF fusion | Comprehensive recall, research tasks |
| **vector_only** | Pure semantic similarity | Exact semantic matching |
| **text_only** | Keyword matching via Fuse.js | Exact keyword matching |

### 5. MCP Tool Interface (5 tools)
| Tool | Purpose |
|------|---------|
| `query_memories` | Semantic search over stored memories |
| `add_memory` | Store a new memory entry |
| `search_project` | Search indexed project files |
| `index_project` | Trigger project indexing |
| `get_file_contents` | Reconstruct file contents from indexed chunks |

### 6. Multi-Collection Search with RRF
Search across multiple Qdrant collections (e.g., during model migration) with Reciprocal Rank Fusion.

### 7. Dual Integration Modes
- **MCP Server (External)**: Standalone MCP server via HTTP/stdio
- **Built-in (Boomerang)**: Core modules imported directly into Boomerang plugin

## Memory Schema
```typescript
interface MemoryEntry {
  id: string;              // UUID
  text: string;            // Content
  vector: Float32Array;    // 1024-dim embedding
  sourceType: MemorySourceType;  // session, file, web, boomerang, project
  sourcePath?: string;
  timestamp: Date;
  contentHash: string;     // SHA-256 for deduplication
  metadataJson?: string;
}
```

## Performance Benchmarks
- Semantic query (HNSW): <10ms p50
- Embedding generation: <100ms (BGE-Large single text)
- Batch embedding: <50ms/text (batch of 8)
- Memory add: ~20/sec
- Memory query: ~100/sec
- Project file indexing: ~100 files/min

## Companion: Boomerang-v2
- **Package**: @veedubin/boomerang-v2
- **Type**: Multi-agent orchestration plugin for OpenCode
- **Features**: 14 specialized agents, 8-step protocol, built-in semantic memory

## What Super-Memory-TS LACKS (compared to Hermes providers)
1. **No automatic memory extraction** — memories must be explicitly added via `add_memory`
2. **No knowledge graph** — pure vector similarity, no entity relationships
3. **No trust scoring** — all memories treated equally
4. **No session-end extraction** — no automatic summarization
5. **No dialectic reasoning** — no LLM-synthesized insights about the user
6. **No tiered context loading** — doesn't do L0/L1/L2 abstraction
7. **No cross-session user modeling** — no persistent user profile
8. **No pre-compression extraction** — doesn't capture before context compression
9. **No context fencing** — no protection against recursive memory pollution
10. **No built-in memory types** — no distinction between session/user/organizational memory
11. **No contradiction detection** — doesn't detect conflicting memories
12. **No memory decay/consolidation** — old memories never fade or get compressed
