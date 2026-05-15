# Memory Providers Landscape: Comprehensive Analysis for Super-Memory Evolution

## Executive Summary
### Key Findings
#### Hermes Agent integrates 8 external memory providers, each with unique architectural approaches to context preservation beyond RAG
#### Super-Memory-TS is MCP-native and OpenCode-compatible, while Hermes uses a Python-only plugin protocol
#### The gap between "memory as RAG" and "true context preservation" spans 12 capability dimensions
#### A hybrid architecture combining MCP compatibility with Hermes-style features offers the strongest next-generation memory system

## 1. The Memory Problem: Why RAG Is Not Enough (~2000 words, 1 table)
### 1.1 The Context Crisis in AI Agents
#### 1.1.1 Context dilution: agents forget decisions made hours earlier as the context window silently pushes them out
#### 1.1.2 Vendor lock-in: no way to carry knowledge across tools (Claude Code, Gemini CLI, OpenCode) or devices
#### 1.1.3 The re-explaining tax: 10+ minutes re-explaining architecture every new session
### 1.2 RAG's Fundamental Limitations
#### 1.2.1 RAG retrieves text chunks by similarity — it does not understand, synthesize, or model the user
#### 1.2.2 Raw retrieval lacks temporal awareness: contradicting facts from different times create confusion
#### 1.2.3 No trust mechanism: all retrieved chunks weighted equally regardless of verification or usefulness
### 1.3 What True Context Preservation Requires
#### 1.3.1 Six capability layers: extraction, synthesis, user modeling, contextual recall, temporal awareness, and trust (table)

## 2. The Hermes Agent Memory Ecosystem (~3000 words, 2 tables)
### 2.1 The Memory Provider Protocol
#### 2.1.1 The MemoryProvider ABC: Python abstract base class with initialize, sync_turn, prefetch, and extract hooks
#### 2.1.2 Six automatic operations: context injection, prefetch, sync, session-end extraction, mirroring, and tool registration
#### 2.1.3 Single-select architecture: only one external provider active at a time alongside built-in MEMORY.md/USER.md
### 2.2 The Eight Memory Providers at a Glance
#### 2.2.1 Full comparison table: Provider, Storage, Cost, Tools, Dependencies, Unique Feature (table)
#### 2.2.2 Categorization by approach: vector-RAG (Mem0, RetainDB), knowledge graph (Hindsight), user modeling (Honcho), structured hierarchy (OpenViking, ByteRover), algebraic (Holographic), semantic graph (Supermemory)
### 2.3 Profile Isolation Architecture
#### 2.3.1 Five isolation mechanisms: local storage, config file, cloud-derived, env var, and multi-peer workspace isolation
#### 2.3.2 Each profile maintains separate credentials and memory namespaces automatically

## 3. Deep Vendor Analysis: The Four Architectural Schools (~5000 words, 4 tables)
### 3.1 School 1: Structured Knowledge Extraction (Hindsight, ByteRover)
#### 3.1.1 Hindsight's TEMPR architecture: four parallel retrieval strategies (Temporal, Entity, Metadata, BM25)
#### 3.1.2 The reflect operation: unique cross-memory synthesis deriving higher-level insights from stored knowledge
#### 3.1.3 ByteRover's knowledge tree: human-readable Markdown hierarchy with ADD/UPDATE/UPSERT/MERGE/DELETE curation
#### 3.1.4 Pre-compression extraction: capturing insights before context compression discards them
#### 3.1.5 Benchmark comparison: Hindsight 91.4-94.6% LongMemEval, ByteRover 92.2% LoCoMo (table)
### 3.2 School 2: Dialectic User Modeling (Honcho, Mem0)
#### 3.2.1 Honcho's two-layer context injection: base layer (session summary + representation + peer card) plus dialectic supplement
#### 3.2.2 Three orthogonal config knobs: contextCadence, dialecticCadence, dialecticDepth controlling cost and depth independently
#### 3.2.3 Multi-peer workspace: separate AI peer profiles per agent persona against the same user
#### 3.2.4 Mem0's dual memory scope: session memories (short-term) and user memories (long-term) with server-side LLM extraction
#### 3.2.5 Mem0's hybrid triple-store: vector + key-value + knowledge graph layers (table)
### 3.3 School 3: Tiered Context Engineering (OpenViking, Holographic)
#### 3.3.1 OpenViking's L0/L1/L2 tiered loading: abstract (~100 tokens) → overview (~2K) → full detail, achieving 80-90% token reduction
#### 3.3.2 Directory recursive retrieval: vector similarity for directory identification, secondary search within, logged trajectories
#### 3.3.3 Holographic's HRR algebra: compositional queries (AND across entities) on local SQLite + FTS5
#### 3.3.4 Trust scoring: asymmetric feedback (+0.05 helpful / -0.10 unhelpful) with automated contradiction detection
#### 3.3.5 Zero-external-dependency design: SQLite-only, operational in seconds (table)
### 3.4 School 4: Semantic Memory Graphs (Supermemory, RetainDB)
#### 3.4.1 Supermemory's document vs memory distinction: raw input vs intelligent knowledge units with Update/Extend/Derive relationships
#### 3.4.2 Context fencing: stripping recalled memories from captured turns to prevent recursive memory pollution
#### 3.4.3 Session-end graph ingest: building knowledge graph from entire conversation sessions
#### 3.4.4 RetainDB's full chronological retrieval: complete memory timeline instead of lossy semantic search
#### 3.4.5 Delta compression and hybrid search (Vector + BM25 + reranking) for retrieval precision (table)

## 4. Super-Memory-TS: Current State Analysis (~2500 words, 2 tables)
### 4.1 Architecture and Strengths
#### 4.1.1 MCP-native TypeScript server with Qdrant HNSW indexing, BGE-Large/MiniLM embeddings
#### 4.1.2 Five MCP tools: query_memories, add_memory, search_project, index_project, get_file_contents
#### 4.1.3 Tiered search strategies: TIERED (MiniLM primary + BGE fallback) and PARALLEL (RRF fusion)
#### 4.1.4 Project isolation via projectId tagging and automatic file indexing with semantic chunking
#### 4.1.5 Performance: <10ms query latency, ~20 adds/sec, ~100 queries/sec (table)
### 4.2 Critical Gaps for True Context Preservation
#### 4.2.1 No automatic memory extraction: requires explicit add_memory calls vs automatic session-end extraction
#### 4.2.2 No knowledge graph: pure vector similarity without entity relationships or memory connections
#### 4.2.3 No user modeling: no persistent behavioral profile of the user across sessions
#### 4.2.4 No trust scoring or contradiction detection: all memories weighted equally, conflicts coexist silently
#### 4.2.5 No tiered context loading: no L0/L1/L2 abstraction for token-efficient retrieval
#### 4.2.6 Twelve capability gaps mapped to Hermes provider features (table)

## 5. Opencode Compatibility Assessment (~2000 words, 2 tables)
### 5.1 The Protocol Divide
#### 5.1.1 Hermes MemoryProvider ABC is Python-only, plugin-based, NOT MCP-compatible
#### 5.1.2 OpenCode uses MCP (Model Context Protocol): language-agnostic JSON-RPC over stdio/HTTP/SSE
#### 5.1.3 Core incompatibility: automatic hooks (prefetch, sync_turn) vs agent-initiated tool calls
#### 5.1.4 Single-select Hermes vs multi-MCP-server OpenCode architecture (table)
### 5.2 Bridge Paths: MCP Wrappers for Hermes Providers
#### 5.2.1 Mem0 MCP: cloud-hosted at mcp.mem0.ai, works with OpenCode
#### 5.2.2 Honcho MCP: opencode-honcho package with persistent memory
#### 5.2.3 Hindsight MCP: OAuth-secured at api.hindsight.vectorize.io/mcp
#### 5.2.4 Supermemory: deep native opencode-supermemory plugin with preemptive compaction
#### 5.2.5 Super-Memory-TS: fully MCP-native, powers Boomerang-v2 for OpenCode (table)
### 5.3 Strategic Implication
#### 5.3.1 Super-Memory-TS is architecturally aligned with OpenCode; Hermes providers require translation layers
#### 5.3.2 Best path forward: extend Super-Memory-TS with Hermes-inspired features while maintaining MCP compatibility

## 6. Recommendations: The Next-Generation Memory Architecture (~3500 words, 2 tables, 1 chart)
### 6.1 Best Ideas to Incorporate from Each Provider
#### 6.1.1 From Hindsight: knowledge graph with entity relationships + cross-memory reflect synthesis
#### 6.1.2 From Honcho: dialectic reasoning for LLM-synthesized user insights + multi-peer profile separation
#### 6.1.3 From OpenViking: L0/L1/L2 tiered context loading for 80-90% token reduction
#### 6.1.4 From Holographic: trust scoring with asymmetric feedback + automated contradiction detection
#### 6.1.5 From ByteRover: pre-compression extraction hooks + human-readable Markdown export format
#### 6.1.6 From Mem0: automatic LLM-based fact extraction + multi-modal memory types (session/user/organizational)
#### 6.1.7 From Supermemory (cloud): context fencing against recursive pollution + memory relationship graph (Update/Extend/Derive)
#### 6.1.8 From RetainDB: full chronological retrieval option + hybrid search (Vector + BM25 + reranking)
### 6.2 Proposed vNext Architecture: "Super-Memory 3.0"
#### 6.2.1 Core principle: MCP-native local-first foundation + optional extraction/synthesis layers
#### 6.2.2 Layer 1 - Keep: Qdrant HNSW, project indexing, tiered search, MCP protocol, project isolation
#### 6.2.3 Layer 2 - Add Automatic Extraction: session-end LLM pass extracting facts, decisions, patterns (inspired by Mem0/ByteRover)
#### 6.2.4 Layer 3 - Add Knowledge Graph: lightweight entity extraction with Update/Extend/Derive relationships (inspired by Hindsight/Supermemory)
#### 6.2.5 Layer 4 - Add User Modeling: dialectic reasoning building persistent user profile (inspired by Honcho)
#### 6.2.6 Layer 5 - Add Trust & Temporal: trust scoring, contradiction detection, memory decay/consolidation (inspired by Holographic)
#### 6.2.7 Layer 6 - Add Tiered Loading: L0/L1/L2 abstraction for token-efficient context injection (inspired by OpenViking)
#### 6.2.8 Architecture diagram showing six layers with data flow (table + description)
### 6.3 Implementation Priority Matrix
#### 6.3.1 Phase 1 (MVP): automatic extraction + trust scoring — highest impact, lowest complexity
#### 6.3.2 Phase 2: knowledge graph + contradiction detection — medium complexity, high differentiation
#### 6.3.3 Phase 3: user modeling + tiered loading — highest complexity, most differentiation
#### 6.3.4 Phase 4: memory decay/consolidation + human-readable export — polish and sustainability
#### 6.3.5 Priority matrix: Impact vs Complexity for all 12 proposed features (table)

# References
## memory_report_outline_references_raw.md
- **Type**: Citation collection
- **Description**: All sources gathered during research
- **Path**: /mnt/agents/output/memory_report_outline_references_raw.md

## Research Artifacts
- **Type**: Research dimension reports
- **Description**: Compiled research on all 8 memory providers, Super-Memory-TS, and Opencode compatibility
- **Path**: /mnt/agents/output/research/memory_providers_dim01.md through dim04.md, memory_providers_insight.md, memory_providers_cross_verification.md
