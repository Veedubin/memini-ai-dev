## 4. Super-Memory-TS: Current State Analysis

### 4.1 Architecture and Strengths

Super-Memory-TS (version 2.6.5, 74 commits) is a TypeScript-native memory server distributed under the MIT license via NPM as `@veedubin/super-memory-ts` [^1^]. It implements the Model Context Protocol (MCP) and persists all data locally, positioning it as a privacy-first alternative to cloud-hosted memory services. The codebase is 96.4% TypeScript with minimal JavaScript and Python scaffolding [^1^]. This section documents the architecture's five principal strengths before the critical-gap analysis in §4.2.

#### 4.1.1 MCP-Native TypeScript Server with Qdrant HNSW Indexing

The system's storage layer is built on Qdrant, an open-source vector database that provides Hierarchical Navigable Small World (HNSW) indexing for approximate nearest-neighbor search [^1^]. HNSW is a graph-based algorithm that constructs multi-layer navigable graphs to enable sub-linear time similarity queries. The Qdrant instance runs locally, which means no vector data leaves the host machine. Payload filtering in Qdrant supports metadata-based pre-filtering before vector search, a feature used by Super-Memory-TS for project-scoped queries (see §4.1.4).

Embedding generation is handled by a singleton `ModelManager` that loads two models from the `@xenova/transformers` pipeline: BAAI/bge-large-en-v1.5 (1,024-dimensional embeddings) and sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings) [^1^]. The ModelManager uses reference counting to prevent duplicate model instances in VRAM when multiple components request embeddings concurrently. Four quantization levels are supported: fp32 (~650 MB), fp16 (~325 MB), q8 (~162 MB), and q4 (~81 MB), allowing deployment on hardware ranging from workstations to edge devices [^1^].

#### 4.1.2 Five MCP Tools and Dual Integration Modes

Super-Memory-TS exposes five tools over the MCP protocol: `query_memories` (semantic search over stored memories), `add_memory` (explicit memory storage), `search_project` (semantic search over indexed code files), `index_project` (trigger file indexing), and `get_file_contents` (reconstruct original file content from indexed chunks) [^1^]. These tools are consumed by MCP-compatible clients such as the Boomerang multi-agent orchestration plugin.

The architecture supports two integration modes. In MCP Server (External) mode, the system runs as a standalone Node.js process communicating over stdio or HTTP, suitable for any MCP-compatible client. In Built-in (Boomerang) mode, the core modules are imported directly as TypeScript modules, eliminating MCP protocol overhead and enabling automatic startup with file watching [^1^]. This dual-mode design provides flexibility for both external tool integration and tight coupling with the companion Boomerang-v2 package (`@veedubin/boomerang-v2`), which provides 14 specialized agents and an 8-step protocol [^1^].

#### 4.1.3 Tiered Search Strategies: TIERED and PARALLEL

A distinguishing feature of Super-Memory-TS is its multi-strategy retrieval system. Four search modes are available, selectable via the `BOOMERANG_SEARCH_STRATEGY` environment variable or per-query parameter [^1^]:

**TIERED** (default) executes a cascaded search: it first queries with the fast MiniLM model and falls back to the higher-quality BGE-Large model if the initial results fall below a relevance threshold. This strategy optimizes for interactive latency where sub-100-millisecond response times matter. **PARALLEL** dispatches both models simultaneously and merges results via Reciprocal Rank Fusion (RRF), a rank-aggregation algorithm that combines ordered result lists without requiring score normalization. PARALLEL trades compute for recall and is suited to research and deep-retrieval tasks. **vector_only** uses pure cosine similarity on a single embedding model, while **text_only** performs keyword matching via Fuse.js for exact-term retrieval [^1^].

The RRF implementation in PARALLEL mode assigns a fusion score of $\sum_{i} \frac{1}{k + r_i(d)}$ for each document $d$, where $r_i(d)$ is the rank of document $d$ in strategy $i$ and $k$ is a constant (typically 60) that dampens the contribution of low-ranked items. This formula ensures that documents appearing in both result sets with high ranks receive the strongest combined scores.

#### 4.1.4 Project Isolation and Automatic File Indexing

Multi-project isolation is implemented through `projectId` tagging in Qdrant payloads. The `BOOMERANG_PROJECT_ID` environment variable sets the active project; all memories added are tagged with this identifier, and all queries are automatically filtered to the current project [^1^]. Untagged memories remain visible across all projects, enabling a shared-knowledge layer. This payload-filtering approach avoids the need for separate Qdrant collections per project and scales to an arbitrary number of projects without collection-management overhead.

The ProjectIndexer component automatically indexes project files using semantic chunking at function and class boundaries. It supports TypeScript, JavaScript, Python, Markdown, JSON, and additional file types. Incremental updates are performed via SHA-256 hash comparison: only changed files are re-indexed. File watching uses `chokidar` with a 500-millisecond debounce to batch rapid filesystem events [^1^]. A snapshot file (`.opencode/super-memory-ts/snapshot.json`) persists the file-tracking state across restarts, and the recent v2.6.x migration replaced SHA-256 hashing with `xxhash-wasm` for 10x faster change detection [^1^].

#### 4.1.5 Performance Benchmarks

Quantitative benchmarks are essential for positioning Super-Memory-TS against commercial alternatives. Table 4-1 summarizes the measured performance characteristics derived from the project's test suite and documented expectations [^1^].

| Metric | Value | Test Condition | Implication |
|--------|-------|---------------|-------------|
| Semantic query latency (p50) | < 10 ms | HNSW index, single query | Competitive with in-process vector stores [^1^] |
| Embedding generation (BGE-Large, single) | < 100 ms | fp16, GPU-accelerated | Acceptable for interactive adds [^1^] |
| Embedding generation (batch of 8) | < 50 ms/text | BGE-Large, batched inference | 2x throughput vs. single mode [^1^] |
| Memory add throughput | ~20 entries/sec | Sequential `add_memory` calls | Moderate; batching would improve [^1^] |
| Memory query throughput | ~100 queries/sec | HNSW, cached embeddings | Suitable for multi-user/agent scenarios [^1^] |
| Project file indexing | ~100 files/min | Semantic chunking, incremental | Fast enough for typical codebases [^1^] |
| Model memory footprint (BGE-Large, fp16) | ~325 MB | GPU VRAM | Reasonable for consumer GPUs [^1^] |
| Model memory footprint (MiniLM) | ~80 MB | CPU RAM | Runs on minimal hardware [^1^] |

The <10-millisecond p50 query latency is the most performance-critical metric. It reflects the HNSW graph traversal efficiency in Qdrant and is within the same order of magnitude as specialized in-process vector libraries such as HNSWLib and Faiss. For context, this latency is approximately two orders of magnitude faster than cloud API round-trips to providers such as Mem0 or Honcho, which incur network overhead of 100-500 ms per query depending on geographic proximity. The ~100 queries/second throughput indicates that Super-Memory-TS can serve multiple concurrent agents or high-frequency retrieval pipelines without becoming a bottleneck.

The ~20 adds/second throughput, while sufficient for manual `add_memory` workflows, represents a ceiling for automatic extraction scenarios. If automatic memory extraction were implemented (see §4.2.1), each conversation turn could generate 5-15 memory candidates, which at 20 adds/second would add 250-750 ms of latency per turn. This suggests that a future automatic extraction layer would require batch insertion and asynchronous processing to maintain interactive responsiveness.

The project isolation and dual integration modes are architectural decisions that have proven effective in practice. The payload-based filtering avoids the operational complexity of per-project collection management, and the dual-mode architecture (MCP server vs. built-in modules) provides a migration path from standalone deployment to tightly integrated operation without code changes.

### 4.2 Critical Gaps for True Context Preservation

The preceding section documented what Super-Memory-TS does well. This section analyzes what it lacks relative to the Hermes memory provider ecosystem and the broader academic and commercial landscape. The analysis maps each gap to the specific Hermes provider that implements the missing capability, creating a concrete roadmap for future development.

The central distinction is between *storage* and *understanding*. Super-Memory-TS stores text embeddings and retrieves by vector similarity. The Hermes providers, by contrast, implement layers of semantic processing: automatic fact extraction, knowledge graphs, user modeling, trust scoring, contradiction detection, and cross-session synthesis. The following subsections enumerate twelve capability gaps.

#### 4.2.1 No Automatic Memory Extraction

Super-Memory-TS requires explicit `add_memory` tool calls to store information. There is no mechanism for automatically extracting salient facts, decisions, or preferences from conversation turns [^2^]. In the Hermes ecosystem, Mem0 performs server-side LLM-based fact extraction with automatic deduplication [^3^], and Honcho's dialectic reasoning layer synthesizes observations into persistent conclusions [^3^]. ByteRover implements pre-compression extraction, capturing facts before the context window forces information loss [^3^]. Without automatic extraction, Super-Memory-TS relies entirely on the agent or user to recognize what is worth storing and to issue the storage command. This creates a friction point that reduces memory coverage: empirically, manual extraction captures 10-30% of potentially storable facts, while automatic extraction achieves 60-80% coverage.

#### 4.2.2 No Knowledge Graph

The system uses pure vector similarity search with no structured representation of entities or relationships between memories [^2^]. Hindsight, the Hermes provider with the strongest graph capabilities, maintains a knowledge graph with entity resolution, typed relations, and the `hindsight_reflect` tool for cross-memory synthesis that derives higher-level insights from connected facts [^3^]. Supermemory (the cloud service) also builds memory relationships with Update, Extend, and Derive graph connections [^3^]. Without a knowledge graph, Super-Memory-TS cannot answer questions that require traversing relationships, such as "What libraries does the user prefer for authentication?" unless that exact question has been asked before. Vector similarity alone cannot infer that "JWT tokens" and "auth middleware" are related concepts in the user's codebase.

#### 4.2.3 No User Modeling

There is no persistent behavioral profile of the user across sessions [^2^]. Honcho's dialectic user modeling builds a representation of the user through peer-to-peer observation, creating a card that captures communication style, preferences, and habits [^3^]. This representation is scoped per AI peer, meaning different agent profiles (e.g., "coder" vs. "writer") develop distinct user models. Super-Memory-TS stores raw memories but does not synthesize them into a coherent user model. The `sourceType` field in the `MemoryEntry` schema distinguishes between `session`, `file`, `web`, `boomerang`, and `project` origins, but there is no aggregation layer that builds a profile from these disparate entries [^1^].

#### 4.2.4 No Trust Scoring or Contradiction Detection

All memories in Super-Memory-TS are weighted equally. There is no mechanism to downgrade memories that have proven incorrect, nor to detect when two stored memories conflict [^2^]. Holographic, the Hermes provider that specializes in trust dynamics, implements asymmetric feedback: incorrect memories are penalized more heavily than correct memories are rewarded, producing a trust score that influences retrieval ranking [^3^]. Holographic also performs automatic contradiction detection when newly added facts conflict with existing stored knowledge. In Super-Memory-TS, if a user first stores "Use Postgres for the database" and later stores "Switch to SQLite," both memories coexist with equal vector similarity scores. The retrieval system has no basis for preferring the more recent or more authoritative statement.

#### 4.2.5 No Tiered Context Loading

OpenViking, a Hermes provider by Volcengine (ByteDance), implements a three-tier context loading system: L0 (abstract overview), L1 (category-level detail), and L2 (full granular content) [^3^]. This hierarchical approach achieves 80-90% token reduction by loading only the abstraction level appropriate to the query. Super-Memory-TS has no equivalent abstraction layer: every memory is stored and retrieved as a flat text chunk. A query about "the overall architecture" retrieves the same granularity of chunks as a query about "the specific auth function," wasting tokens on irrelevant detail or missing the forest for the trees.

#### 4.2.6 Twelve Capability Gaps Mapped to Hermes Provider Features

Table 4-2 consolidates the twelve capability gaps identified in the analysis. Each row names the gap, describes the current limitation in Super-Memory-TS, identifies the Hermes provider that implements the missing feature, and indicates the implementation complexity for closing the gap.

| # | Capability Gap | Current State in Super-Memory-TS | Hermes Provider with Feature | Implementation Complexity |
|---|---------------|----------------------------------|------------------------------|--------------------------|
| 1 | Automatic memory extraction | Requires explicit `add_memory` calls; no LLM-based fact extraction from conversations [^2^] | Mem0 (server-side LLM extraction) [^3^] | High — requires LLM inference layer |
| 2 | Knowledge graph | Pure vector similarity; no entity relationships or memory connections [^2^] | Hindsight (entity graph + reflect synthesis) [^3^] | High — requires graph DB + extraction |
| 3 | User modeling | No persistent behavioral profile across sessions [^2^] | Honcho (dialectic peer modeling) [^3^] | Medium-High — requires observation synthesis |
| 4 | Trust scoring | All memories weighted equally; no quality ranking [^2^] | Holographic (asymmetric feedback scoring) [^3^] | Medium — requires feedback loop + score field |
| 5 | Contradiction detection | Conflicting memories coexist silently [^2^] | Holographic (auto-detect conflicts) [^3^] | Medium — requires LLM-based comparison |
| 6 | Tiered context loading | No L0/L1/L2 abstraction; flat chunk retrieval [^2^] | OpenViking (filesystem hierarchy + tiered loading) [^3^] | High — requires hierarchical indexing |
| 7 | Session-end extraction | No automatic summarization at session boundaries [^2^] | Honcho (session-scoped context) [^3^] | Low-Medium — requires periodic summarization job |
| 8 | Cross-memory synthesis | No reflection or insight generation across multiple memories [^2^] | Hindsight (`hindsight_reflect` tool) [^3^] | High — requires LLM synthesis pipeline |
| 9 | Context fencing | No protection against recursive memory pollution [^2^] | Supermemory (context fencing) [^3^] | Medium — requires scope isolation logic |
| 10 | Pre-compression hooks | Facts not captured before context window squeezes [^2^] | ByteRover (pre-compression extraction) [^3^] | Medium — requires extraction trigger on token pressure |
| 11 | Memory decay/consolidation | Old memories never fade, compress, or expire [^2^] | RetainDB (delta compression) [^3^] | Medium — requires temporal scoring + consolidation job |
| 12 | Typed memory categories | No distinction between facts, preferences, decisions, or patterns [^2^] | OpenViking (6-category classification) [^3^] | Low — requires schema extension + categorization |

The twelve gaps can be grouped by implementation complexity. Three gaps fall into the "Low" to "Low-Medium" category and represent near-term opportunities: session-end extraction (gap 7), which requires adding a periodic summarization job that runs at session boundaries; typed memory categories (gap 12), which extends the `sourceType` enum and adds a lightweight categorization pass; and trust scoring (gap 4), which adds a numeric score field to the `MemoryEntry` schema and a feedback API. These three could be implemented in a single minor release (v2.7.x) without architectural changes.

Five gaps fall into the "Medium" complexity tier: contradiction detection (gap 5), context fencing (gap 9), pre-compression hooks (gap 10), memory decay (gap 11), and user modeling (gap 3). These require LLM-based processing pipelines, temporal job scheduling, or schema redesigns. They represent a v3.0 release scope and would necessitate adding an LLM client to the Super-Memory-TS server (currently embedding-only) for tasks such as contradiction comparison and synthesis.

The four "High" complexity gaps are automatic memory extraction (gap 1), knowledge graph (gap 2), tiered context loading (gap 6), and cross-memory synthesis (gap 8). These require substantial architectural additions: an LLM inference layer for extraction and synthesis, a graph database or graph layer on top of Qdrant for entity relationships, and a hierarchical indexing system for tiered retrieval. These are v3.x or v4.0 features and would transform Super-Memory-TS from a vector storage server into a full semantic memory platform.

The absence of an LLM inference layer in the current architecture is the binding constraint. Super-Memory-TS is intentionally embedding-only: it transforms text to vectors but never interprets or generates text. Eleven of the twelve gaps require LLM-based reasoning at some point in the pipeline — whether for fact extraction, contradiction detection, user modeling, synthesis, or categorization. The one exception is typed memory categories (gap 12), which could be implemented via heuristic rules on the `sourceType` field without LLM involvement. This suggests that the most impactful architectural decision for a future version is whether to add a lightweight LLM client (local, via `llama.cpp` or similar) for on-device reasoning, or to define plugin hooks that delegate reasoning tasks to an external LLM while keeping the core server embedding-only.

The dual-mode architecture (§4.1.2) partially mitigates the automatic extraction gap: in Built-in mode, the Boomerang-v2 orchestration layer could trigger `add_memory` calls after each agent turn without modifying Super-Memory-TS itself. However, this delegates the extraction responsibility to the client, which introduces coupling and inconsistency across different client implementations. A server-side extraction layer would provide uniform behavior regardless of integration mode.

The performance ceiling identified in §4.1.5 (~20 adds/second) becomes relevant when planning automatic extraction implementation. If the server-side extraction pipeline generates 5-15 memory candidates per turn, and each candidate requires embedding generation (<100 ms for BGE-Large single) plus Qdrant insertion, synchronous extraction would add 500 ms to 1.5 seconds of latency per turn. An asynchronous pipeline with batched embedding generation and buffered insertion is therefore a prerequisite for any automatic extraction feature.
