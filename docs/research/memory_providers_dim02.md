# Dimension 2: Deep Vendor Profiles

## 1. Honcho (honcho.dev)
- **Docs**: https://docs.honcho.dev
- **GitHub**: https://github.com/plastic-labs/honcho
- **Architecture**: Two-layer context injection — base layer (session summary + representation + peer card) plus dialectic supplement (LLM reasoning). Dialectic automatically selects cold-start prompts (general user facts) vs. warm prompts (session-scoped context).
- **Three orthogonal config knobs**: contextCadence, dialecticCadence, dialecticDepth (1-3 passes)
- **Multi-peer setup**: Workspace contains Peers (users + AI agents). Each AI peer builds independent representation/card from its own observations.
- **Reasoning engine ("deriver")**: Background reasoning processes messages to extract premises, draw conclusions, build representations.
- **Tools (5)**: honcho_profile, honcho_search, honcho_context, honcho_reasoning, honcho_conclude
- **Key insight**: The only provider built specifically for user modeling — not just storing facts but building a behavioral model of the user over time.
- **Cost**: Paid cloud / free self-hosted
- **License**: AGPL-3.0

## 2. OpenViking (github.com/volcengine/OpenViking)
- **Docs**: No dedicated docs site; README-driven
- **GitHub**: https://github.com/volcengine/OpenViking
- **By**: ByteDance (Volcengine)
- **Architecture**: Hierarchical virtual filesystem with `viking://` URI scheme. Tiered context loading:
  - **L0 (Abstract)**: ~100 tokens, one-sentence summary for quick identification
  - **L1 (Overview)**: ~2K tokens, core info and usage scenarios for planning
  - **L2 (Detail)**: Full content, loaded only on demand
- **Retrieval**: Directory recursive retrieval — vector similarity to identify directory, secondary search within directory, drilling down recursively. Every step logged as visible trajectory.
- **Auto-extraction**: On session commit, extracts into 6 categories (profile, preferences, entities, events, cases, patterns)
- **Tools (5)**: viking_search, viking_read, viking_browse, viking_remember, viking_add_resource
- **Key insight**: 80-90% token reduction — instead of loading all context, agent "skims" at L0, refines at L1, deep-dives at L2 only when needed.
- **Cost**: Free (AGPL-3.0)
- **Deployment**: Self-hosted only, requires Docker + LLM for extraction

## 3. Mem0 (mem0.ai)
- **Docs**: https://docs.mem0.ai
- **GitHub**: https://github.com/mem0ai/mem0
- **Architecture**: Hybrid triple-store — vector + key-value + knowledge graph. LLM pass extracts structured facts from conversations, stores across all three layers.
- **Dual memory scope**: Session memories (short-term) + User memories (long-term). Both searched and injected before each response.
- **Server-side LLM extraction**: Mem0's infrastructure decides what's worth keeping — no manual curation.
- **Circuit breaker**: If API fails 5 times, stops calling for 2 minutes. Agent keeps working.
- **MCP Server**: https://mcp.mem0.ai/mcp — cloud-hosted MCP server
- **Tools (3)**: mem0_profile, mem0_search, mem0_conclude
- **Memory types**: Conversation memory, Session memory, User memory, Organizational memory
- **Key insight**: Best SDK quality, largest community (51.4K stars), self-host option, Apache license. Fastest 30-second setup.
- **Benchmark**: LongMemEval-S 67.6%
- **Cost**: Freemium (10K adds/1K recalls free), $19/mo Starter, $249/mo Pro (Graph)
- **License**: Apache 2.0

## 4. Hindsight (hindsight.vectorize.io)
- **Docs**: https://hindsight.vectorize.io
- **GitHub**: Part of Vectorize
- **Architecture**: TEMPR — four parallel retrieval strategies: Temporal, Entity, Metadata, and BM25 for exact keyword matches. Three-stage pipeline: retain (ingest) → recall (retrieve) → reflect (synthesize).
- **Knowledge graph**: Stores structured knowledge — discrete facts, named entities, relationships. NOT raw text chunks.
- **Reflect operation**: Unique cross-memory synthesis — periodic pass that reads across all stored memories to derive higher-level insights and consolidate related facts. No other provider has this.
- **Local mode**: Embedded PostgreSQL, no cloud account required. Full features locally for free.
- **MCP Server**: https://api.hindsight.vectorize.io/mcp/default/
- **Tools (3)**: hindsight_retain, hindsight_recall, hindsight_reflect
- **Key insight**: Highest LongMemEval scores (91.4-94.6% with Gemini-3). Independently validated (Virginia Tech arXiv paper). Best for coding agents.
- **Benchmarks**: LongMemEval 91.4-94.6%, BEAM 64.1%, LoCoMo 89.6%
- **Cost**: Free (local) / Cloud: $15/M retain, $0.75/M recall, $3/M reflect
- **License**: MIT

## 5. Holographic (built into Hermes)
- **Docs**: Built into Hermes docs
- **Architecture**: Holographic Reduced Representations (HRR) algebra on local SQLite + FTS5. Zero external dependencies.
- **Trust scoring**: Asymmetric feedback — +0.05 helpful / -0.10 unhelpful. Facts have trust scores (0.0-1.0).
- **Fact store tool**: 9 actions — add, search, probe, related, reason, contradict, update, remove, list
- **Unique capabilities**:
  - `probe` — entity-specific algebraic recall (all facts about a person/thing)
  - `reason` — compositional AND queries across multiple entities
  - `contradict` — automated detection of conflicting facts
- **Tools (2)**: fact_store (9 actions), fact_feedback
- **Key insight**: Most private option — SQLite file on disk, zero network calls, zero dependencies. Operational in seconds with no accounts.
- **Cost**: Completely free
- **License**: MIT

## 6. RetainDB (retaindb.com)
- **Docs**: Limited public docs
- **GitHub**: https://github.com/RetainDB
- **Architecture**: Turn-by-turn extraction with 3-turn context. Atomic memories with eventDate, documentDate, confidence scores.
- **Retrieval**: Full chronological retrieval — provides complete memory chronology rather than lossy semantic search. The answering model sees the full timeline.
- **Hybrid search**: Vector + BM25 + reranking
- **Delta compression**: Memory diff storage
- **Tools (5)**: retaindb_profile, retaindb_search, retaindb_context, retaindb_remember, retaindb_forget
- **Key insight**: 0% hallucination rate claimed, SOTA on preference recall (88%). "No lossy retrieval" philosophy — guarantees coverage but gives model more context to sift through.
- **Cost**: $20/mo (no free tier)
- **License**: Proprietary

## 7. ByteRover (byterover.dev)
- **Docs**: https://docs.byterover.dev
- **Architecture**: Hierarchical knowledge tree stored as human-readable Markdown files in `.brv/context-tree/`. Navigable hierarchy.
- **Curation engine**: Five operations — ADD, UPDATE, UPSERT, MERGE, DELETE. Agent decides how to modify the context tree.
- **Pre-compression extraction**: Fires specifically before Hermes compresses a long conversation, ensuring in-flight knowledge gets captured before context is summarized away.
- **Tiered retrieval**: Fuzzy text → LLM-driven search
- **Connectors**: MCP, Skill, Hook, Rules types for 20+ agents
- **Tools (3)**: brv_query, brv_curate, brv_status
- **Key insight**: Full visibility into what the agent knows — human-readable, editable Markdown. Best LoCoMo score (92.2%). Local-first, portable.
- **Benchmark**: LoCoMo 92.2% (single-hop 95.4%, temporal 94.4%)
- **Cost**: Free (local) / $19/mo Pro (cloud sync)

## 8. Supermemory (supermemory.ai)
- **Docs**: https://supermemory.ai/docs
- **GitHub**: https://github.com/supermemoryai
- **Architecture**: Semantic understanding graph built on top of entities. Documents vs Memories distinction — documents are raw input, memories are intelligent knowledge units created by Supermemory.
- **Memory relationships**: Three types — Updates (information changes), Extends (information enriches), Derives (information infers)
- **Context fencing**: Strips recalled memories from captured turns to prevent recursive memory pollution
- **Session-end conversation ingest**: Richer graph-level knowledge building
- **Multi-container mode**: Read/write across named containers
- **Connectors**: Google Drive, Notion, OneDrive, Web Crawler, Gmail, GitHub, S3
- **Tools (4)**: supermemory_store, supermemory_search, supermemory_forget, supermemory_profile
- **Key insight**: All-in-one philosophy — memory + RAG + user profiles + connectors. Most generous free tier (1M tokens + 10K searches).
- **Benchmark**: LongMemEval 81.6% (with GPT-4o)
- **Cost**: Free (1M tokens/10K searches), $19/mo Pro, $399/mo Scale
