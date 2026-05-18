# Memini-ai Context Document

> **Project**: Memini-ai v3.0 (formerly Super-Memory-TS)
> **Meaning**: "I remember" in Latin (pronounced "meh-mee-nee")
> **Goal**: Local-first semantic memory server, MCP-compatible, Python-based
> **Last Updated**: 2026-05-19

---

## Project Origin

User is building Memini-ai v3.0 as a rewrite of Super-Memory-TS (TypeScript) to Python. Reasons:
1. User prefers Python over TypeScript
2. "Never seemed to work as well as it did in V1 in Python"
3. Wants to rename to avoid confusion with other "super-memory" products
4. Will use FastMCP if possible (instead of @modelcontextprotocol/sdk)

---

## Key Architectural Decisions

### Keep from Super-Memory-TS
- Qdrant backend (HNSW indexing)
- BGE-Large embeddings (1024-dim) - "the 1024 dim model path"
- MiniLM-L6-v2 (384-dim) as CPU fallback
- MCP server with same 5 tools
- Sub-10ms query latency
- Local-first, privacy-preserving

### Add: Memory Kernel v3.0
Based on Reality Check analysis, build these 3 features FIRST:

1. **Auto-Extract**: After every N turns (default 5), fire LLM pass to extract facts/decisions/preferences automatically
2. **Memory Graph**: Track relationships (SUPERSEDES, RELATED_TO, CONTRADICTS, DERIVED_FROM) via JSON field in Qdrant payload
3. **Trust Engine**: Every memory starts at trust=0.5, adjust based on agent behavior feedback

### Tier 2 (add later)
- Pre-compression extraction (capture before context squeeze)
- Tiered loading L0/L1/L2 (token-efficient context)
- User modeling (Honcho-style)

---

## The Three Problems We're Solving

1. **"I have to manually remember to save memories"** → Auto-Extract
2. **"My memories are just a pile of vectors with no structure"** → Memory Graph
3. **"Bad memories poison my context"** → Trust Engine

---

## Six Capability Layers (Long-term Vision)

| Layer | Description | Inspired By |
|-------|-------------|-------------|
| 1. Foundation | Qdrant/HNSW, BGE-Large, 5 MCP tools | Super-Memory-TS |
| 2. Extraction | Auto LLM-based extraction | Mem0, ByteRover |
| 3. Knowledge Graph | Lightweight relationships | Hindsight, Supermemory |
| 4. User Modeling | Dialectic peer profiles | Honcho |
| 5. Trust & Decay | Asymmetric scoring, contradiction detection | Holographic, RetainDB |
| 6. Tiered Loading | L0/L1/L2 abstraction levels | OpenViking |

**All layers optional and independently enableable.**

---

## Current Status: Phase 1 Complete ✅

**Implemented**: Foundation phase of Memini-ai v3.0 (Python rewrite of Super-Memory-TS)

### What's Implemented
- Full Python port of Super-Memory-TS v2.6.5 architecture
- 23 source modules in `src/memini_ai/`
- 201 passing tests, 10 skipped (integration tests)
- 6 MCP tools (including get_status bonus)
- Async-first with asyncio.to_thread()

### What's Next
**Phase 2: Memory Kernel Features** (Auto-Extract, Memory Graph, Trust Engine)

---

## Super-Memory-TS v2.6.5 Architecture (to port)

### Directory Structure
```
src/
├── memory/
│   ├── index.ts          # MemorySystem class
│   ├── schema.ts         # MemoryEntry, SearchOptions types
│   ├── database.ts       # Qdrant operations
│   └── search.ts         # TIERED/VECTOR_ONLY/TEXT_ONLY/PARALLEL
├── model/
│   ├── index.ts
│   ├── embeddings.ts     # generateEmbedding(), generateEmbeddings()
│   └── types.ts
├── project-index/
│   ├── index.ts
│   ├── indexer.ts        # File indexing with SHA-256
│   ├── chunker.ts        # Semantic chunking
│   ├── watcher.ts        # File watching (chokidar)
│   ├── file-tracker.ts
│   ├── snapshot.ts
│   ├── pause-controller.ts
│   └── constants.ts
├── server.ts             # MCP server
├── config.ts            # Environment + JSON config
└── utils/
    ├── logger.ts
    ├── hash.ts          # SHA-256
    └── errors.ts
```

### MemoryEntry Schema
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
  sessionId?: string;
  projectId?: string;
  score?: number;          // Qdrant similarity score
}
```

### 6 MCP Tools (Implemented)
1. `query_memories` - Semantic search over memories
2. `add_memory` - Store a new memory entry
3. `search_project` - Search indexed project files
4. `index_project` - Trigger project indexing
5. `get_file_contents` - Reconstruct file from indexed chunks
6. `get_status` - Component readiness status (bonus)

### Search Strategies
- **TIERED** (default): MiniLM primary + BGE fallback, threshold 0.72
- **PARALLEL**: Dual-tier with RRF fusion
- **VECTOR_ONLY**: Pure semantic similarity
- **TEXT_ONLY**: Fuse.js keyword matching

### Dependencies to Port
| TS | Python |
|----|--------|
| @modelcontextprotocol/sdk | fastmcp |
| @qdrant/js-client-rest | qdrant-client |
| @xenova/transformers | transformers + torch |
| chokidar | watchdog |
| fuse.js | rank_bm25 / Whoosh |
| xxhash-wasm | xxhash |
| zod | pydantic |

---

## Key Insights from Research

### From memory_providers_dim01.md (Hermes Comparison)
- 8 providers: Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory
- Hermes uses Python ABC protocol, NOT MCP-compatible directly
- Each provider isolation via $HERMES_HOME/ paths

### From memory_providers_dim02.md (Vendor Profiles)
- **Hindsight**: Best benchmarks (91.4-94.6% LongMemEval), TEMPR 4-strategy retrieval
- **Honcho**: Only provider built for user modeling, dialectic reasoning
- **OpenViking**: 80-90% token reduction via L0/L1/L2 tiered loading
- **Holographic**: Zero dependencies, HRR algebra, trust scoring (+0.05/-0.10)
- **Mem0**: Server-side LLM extraction, Apache 2.0, 51.4K stars

### From memory_providers_dim03.md (Super-Memory-TS Analysis)
Current gaps (12 total):
1. No automatic memory extraction
2. No knowledge graph
3. No user modeling
4. No memory synthesis
5. No tiered context loading
6. No trust scoring
7. No contradiction detection
8. No temporal awareness
9. No memory decay/consolidation
10. No context fencing
11. No pre-compression hooks
12. No multi-modal memory types

### From memory_providers_dim04.md (OpenCode Compatibility)
- Super-Memory-TS IS MCP-compatible (runs as stdio/HTTP MCP server)
- Hermes protocol is Python-only ABC, needs MCP bridge for OpenCode
- Mem0, Hindsight, Honcho all have MCP bridges

### From memory_providers_insight.md (Recommendations)
Recommended vNext architecture: 6-layer hybrid
- Keep MCP, local-first, Qdrant/HNSW
- Add auto-extraction (Mem0/ByteRover style)
- Add lightweight knowledge graph
- Add trust scoring + contradiction detection
- Add tiered loading L0/L1/L2
- All optional, backward compatible

### From reality_check.md (Build Priorities)
**DO NOT build all 6 layers at once!**

Build order:
1. **Memory Kernel** (4-6 weeks, ~2500 lines):
   - Auto-Extract
   - Memory Graph
   - Trust Engine
2. **Tier 2** (2-3 months each, after kernel stable):
   - Pre-Compression Extraction
   - Tiered Loading L0/L1/L2
3. **Tier 3** (maybe never):
   - Full KG, memory decay, multi-peer, dialectic

Every feature MUST be independently optional - can be disabled.

---

## MCP Command Reference

User mentioned "1024 dim model path" but couldn't remember the MCP command. The relevant commands for super-memory-ts were:
- `query_memories` - search memories
- `add_memory` - store memory
- `search_project` - search indexed code
- `index_project` - trigger indexing
- `get_file_contents` - reconstruct files

For Memini-ai, we may add:
- `get_trust_score` - get memory trust
- `adjust_trust` - update trust based on feedback
- `list_archived` - list archived low-trust memories

---

## Version History

| Version | Language | Notes |
|---------|----------|-------|
| v1.0 | Python | Original implementation (user prefers this) |
| v2.0 | TypeScript | Super-Memory-TS (current npm package) |
| v3.0 | Python | Memini-ai (rewriting v2.0 in Python) |

---

## Important Files to Reference

1. Super-Memory-TS source: `/node_modules/@veedubin/super-memory-ts/dist/`
2. Research docs: `memini-ai-dev/docs/research/`
3. Memory Report: `memini-ai-dev/docs/memory_report.agent.final.md`
4. Reality Check: `memini-ai-dev/docs/reality_check.md`
5. This file: `memini-ai-dev/CONTEXT.md`
6. Tasks: `memini-ai-dev/TASKS.md`
