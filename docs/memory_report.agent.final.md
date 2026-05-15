# Memory Providers Landscape: Comprehensive Analysis for Super-Memory Evolution

> **Date:** May 15, 2026
> **Scope:** Analysis of 8 Hermes Agent memory providers vs. Super-Memory-TS
> **Focus:** Context preservation beyond RAG, OpenCode compatibility, and architectural recommendations

---

## Executive Summary

### Key Findings

AI agents forget. Not through malfunction, but through architectural design. Transformer-based Large Language Models (LLMs) exhibit a characteristic U-shaped attention curve — information at the center of a long context window is systematically under-weighted, with accuracy dropping more than 20 percentage points when critical facts move from the edges to the middle [^1^]. At production scales of 10 million tokens, Retrieval-Augmented Generation (RAG) — the conventional remedy — collapses from 65.2% to 26.1% accuracy on multi-session reasoning tasks [^12^]. For a five-person engineering team, the practical cost of this context loss — minutes spent re-establishing project state at the start of every new agent session — compounds to approximately $78,000 per year in lost productive time [^10^]. Memory is not a retrieval problem. It is a knowledge-management problem.

This report examines the Hermes Agent Memory ecosystem, which integrates eight external memory providers through a Python Abstract Base Class (ABC) protocol: Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, and Supermemory. These providers resolve into four distinct architectural schools: structured knowledge extraction (Hindsight, ByteRover), dialectic user modeling (Honcho, Mem0), tiered context engineering (OpenViking, Holographic), and semantic memory graphs (Supermemory, RetainDB). Each school represents a fundamentally different answer to how an agent should remember — from Hindsight's knowledge graph with cross-memory `reflect` synthesis (scoring 91.4–94.6% on LongMemEval, the highest of any provider) [^94^], to Honcho's dialectic reasoning engine that builds living behavioral models of the user through peer-to-peer observation [^103^], to OpenViking's L0/L1/L2 tiered loading that achieves 80–90% token reduction versus flat retrieval [^61^], to Holographic's zero-dependency trust scoring system that penalizes incorrect memories more heavily than it rewards correct ones (+0.05 helpful, −0.10 unhelpful), causing the memory store to self-correct over time [^1^].

Against this landscape, the report evaluates Super-Memory-TS — a Model Context Protocol (MCP)-native, TypeScript-based memory server built on Qdrant's Hierarchical Navigable Small World (HNSW) index, delivering sub-10-millisecond query latency and local-first privacy guarantees [^1^]. The architecture is MCP-native from inception, meaning it requires no translation layer to work with OpenCode, Claude Code, Cursor, or any other MCP-compatible host. This is a strategic differentiator: Hermes providers use a Python-only ABC protocol with automatic per-turn hooks (`prefetch`, `sync_turn`) that are structurally incompatible with MCP's agent-initiated tool-call model [^62^] [^121^]. Every major Hermes provider now offers an MCP bridge — Mem0 via a cloud-hosted endpoint, Hindsight via OAuth-secured MCP, Honcho via a native OpenCode plugin — but each bridge exposes only a subset of the provider's Hermes-native capabilities [^38^] [^117^] [^107^].

The gap analysis reveals twelve critical capability dimensions where Super-Memory-TS falls short of the Hermes provider state of the art: no automatic memory extraction (requiring manual `add_memory` calls versus Mem0's 60–80% automatic fact coverage) [^3^]; no knowledge graph (pure vector similarity versus Hindsight's entity-resolved TEMPR retrieval) [^94^]; no user modeling (raw memory storage versus Honcho's dialectic peer profiles) [^103^]; no trust scoring or contradiction detection (equal weighting versus Holographic's asymmetric feedback) [^1^]; no tiered context loading (flat chunks versus OpenViking's 80–90% token reduction) [^61^]; and seven additional gaps spanning session-end extraction, cross-memory synthesis, context fencing, pre-compression hooks, memory decay, and typed memory categories. These twelve gaps map to the six capability layers that define true context preservation: extraction, synthesis, user modeling, contextual recall, temporal awareness, and trust with decay.

The report proposes a hybrid architecture — "Super-Memory 3.0" — that preserves Super-Memory-TS's MCP-native, local-first foundation while selectively incorporating the highest-leverage innovations from the Hermes ecosystem. The architecture organizes capabilities into six compositional layers: Layer 1 (foundation) keeps the existing Qdrant HNSW backend, dual-model embeddings, and five MCP tools unchanged; Layer 2 adds automatic LLM-based extraction inspired by Mem0 and ByteRover; Layer 3 adds a lightweight knowledge graph with Update/Extend/Derive relationships drawn from Hindsight and Supermemory; Layer 4 adds dialectic user modeling from Honcho; Layer 5 adds trust scoring, contradiction detection, and memory decay from Holographic and RetainDB; and Layer 6 adds L0/L1/L2 tiered loading from OpenViking for token-efficient context injection. Every layer is optional and independently enableable, maintaining full backward compatibility with existing MCP clients.

Implementation is staged across four phases driven by impact-to-complexity ratio. Phase 1 (trust scoring plus automatic extraction) delivers the highest immediate return: trust scoring requires only a schema field and a new tool, while automatic extraction transforms the system from passive storage into active curation. Phase 2 adds the knowledge graph and contradiction detection. Phase 3 brings user modeling and tiered loading — the highest-differentiation, highest-complexity features. Phase 4 closes with memory consolidation, human-readable export, and context fencing. A two-person team can execute each phase in approximately two to three weeks. The critical-path dependency is the LLM client abstraction in Phase 1 — supporting local, remote, and passthrough modes — which unlocks every subsequent phase while preserving deployability across air-gapped workstations to cloud-connected developer machines.

The conclusion for technical decision-makers is straightforward: no single existing system covers all six capability layers of true context preservation. Hermes providers offer the deepest individual capabilities but are locked behind a Python-only protocol. Super-Memory-TS offers the right architectural foundation — MCP-native, local-first, sub-10-millisecond queries — but lacks the semantic processing layers that distinguish memory-as-RAG from memory-as-understanding. The path forward is compositional: build on the MCP foundation, port the best ideas from each Hermes provider, and maintain the protocol compatibility that guarantees interoperability across the entire agent-tooling ecosystem.

---

## 1. The Memory Problem: Why RAG Is Not Enough

### 1.1 The Context Crisis in AI Agents

Every developer who has worked with a coding assistant for more than a single session has experienced the same sequence of events. The agent begins with energy: it reads the codebase, proposes a sound architecture, and implements the first few files with confidence. An hour passes. Then two. The conversation drifts — a quick bug fix, a dependency upgrade, a detour into configuration. When the developer returns to the original architectural thread, the agent no longer remembers *why* a certain decision was made three hours ago. The reasoning has been pushed out of the context window, replaced by more recent but less important turns. This is not a failure of retrieval. It is a failure of *context architecture* — the way modern agent systems manage, preserve, and synthesize the knowledge that accumulates across a working relationship.

#### 1.1.1 Context Dilution: The Quiet Forgetting

The phenomenon has several names in the research literature — "Lost in the Middle" [^1^], "context rot" [^2^], "attention dilution" [^3^] — but they describe the same underlying mechanism. Transformer-based Large Language Models (LLMs) attend disproportionately to the beginning and end of their input context while under-weighting information in the middle, producing a characteristic U-shaped performance curve. A landmark 2023 study by Liu et al. at Stanford and Meta AI found that accuracy on multi-document question-answering tasks drops by more than 20 percentage points when relevant information moves from the edges of the context to the center [^1^]. In extreme cases, GPT-3.5-Turbo's accuracy fell *below its closed-book performance* — meaning the model would have been better off with no context at all than with the correct answer buried mid-document [^1^].

Follow-up research published at ACL Findings 2024 traced the root cause to an intrinsic positional attention bias: LLMs assign higher attention weights to beginning and end tokens regardless of their semantic relevance [^4^]. Independently, researchers at MIT and Meta AI identified "attention sinks" — initial tokens that receive disproportionately high attention scores because softmax normalization forces attention weights to sum to one, making early tokens default receptacles for excess attention mass [^3^]. The compounding effect is what Chroma's 2025 systematic study termed "context rot": every one of the 18 frontier models tested — including GPT-4.1, Claude Opus 4, and Gemini 2.5 — exhibited measurable performance degradation at *every* increment of input length, not merely near the stated context limit [^2^]. A model advertising a 200K-token window showed significant degradation at 50K tokens [^5^].

For agent builders, the implication is stark. An agent working on a multi-hour task accumulates context at every turn: file reads, grep results, tool outputs, reasoning traces, error messages. Anthropic's engineering team stated the principle explicitly in a 2025 engineering blog: context must be treated as a limited resource with decreasing returns [^6^]. The attention budget — the amount of context a model can focus on without losing the thread of prior reasoning — is a hard constraint. Every token added to the window eats into it.

#### 1.1.2 Vendor Lock-In: Knowledge Imprisoned in Silos

The context crisis deepens when the working session ends. A developer who spends hours establishing architectural context with Claude Code cannot transfer that understanding to Gemini CLI, to OpenCode, or even to another instance of the same tool. Each environment maintains its own isolated conversation history. The knowledge — the decisions, the trade-offs, the rationale — is locked inside a vendor-specific session log that cannot be exported in a machine-actionable form.

This pattern creates what industry analysts call "behavioral lock-in" [^7^]: the more an agent learns about a user's preferences, codebase, and working style, the higher the switching cost becomes. Agents that rely on proprietary session memory with no persistent state are the least portable, while those that accumulate structured long-term memory create the strongest adhesion to their host platform [^7^]. The critical question for builders is whether the agent stores its learned context in a proprietary format controlled by the vendor, or in an open, portable representation that the user owns. Most current tools fall into the former category. The Hermes Agent Memory ecosystem, examined in the next chapter, is one of the few frameworks that addresses this through a standardized provider protocol — but even that ecosystem remains fragmented across eight distinct implementations with varying storage models and portability guarantees [^8^].

#### 1.1.3 The Re-Explaining Tax

The practical cost of context isolation is measured in minutes lost at the start of every new session. A developer must re-explain the project structure, the architectural decisions, the constraints, the team conventions — information that was painstakingly established in prior conversations but exists only as unstructured chat logs if it exists at all. The phenomenon maps directly onto the well-documented cost of context switching in software engineering. Research by Dr. Gloria Mark at UC Irvine found that it takes an average of 23 minutes and 15 seconds to fully regain focus after an interruption [^9^]. The cognitive cost of rebuilding a mental model — whether from a Slack notification or from opening a fresh agent session with zero retained context — follows the same curve.

For a five-person engineering team, Atlassian's 2025 survey of 3,500 engineers estimated the annual cost of tool context-switching at approximately $78,000 in lost productive time [^10^]. Applied to AI agent workflows, the calculation is similar: if each developer starts two new agent sessions per day and spends 10 minutes re-establishing context per session, the weekly re-explaining tax exceeds 100 minutes per person. Across a quarter, that compounds to hours of redundant explanation that a properly architected memory system would eliminate entirely.

### 1.2 RAG's Fundamental Limitations

The conventional response to the context crisis has been Retrieval-Augmented Generation (RAG): chunk prior conversations, embed them into a vector space, and retrieve the most semantically similar chunks at query time. RAG works well for document-grounded question answering. It fails structurally when applied to agent memory.

#### 1.2.1 Similarity Is Not Understanding

RAG retrieves text chunks by measuring the cosine distance between embedding vectors. It does not understand what it retrieves, synthesize across multiple chunks, or build a model of the user. Two passages that share no semantic overlap — "my birthday is in March" and "I prefer chocolate cake" — will never be retrieved together even though they are causally related (the cake preference was stated *in the context of* planning a birthday). This limitation, described by XTrace.ai as "reactive rather than associative" retrieval [^11^], means RAG cannot perform the associative leaps that human memory performs naturally. A query for "current task" may surface a log from three days ago because the phrasing is semantically similar, a problem researchers call "context pollution" [^11^].

The BEAM benchmark, the only public evaluation that tests memory systems at production-relevant scales of 1M and 10M tokens, reveals the consequences. On multi-session reasoning tasks, accuracy collapses from 65.2% at 1M tokens to 26.1% at 10M tokens — a 60% relative drop [^12^]. Temporal reasoning degrades from 61.8% to 16.3% over the same scale [^12^]. These are not edge cases. They are the core failure modes of a retrieval architecture that was designed for static document corpora, not for evolving, temporally structured agent interactions.

#### 1.2.2 Temporal Blindness

Standard RAG has no concept of time. Vector indexes flatten history into a list of isolated chunks, discarding the sequence in which facts were learned and the temporal relationships between them. When a user tells an agent "I'm switching from Python to TypeScript," a standard RAG system appends a new chunk without invalidating or superseding the prior Python-related entries [^11^]. Later, when the agent queries for "coding style," it retrieves both old and new instructions, creating a state conflict with no mechanism for resolution.

The LongMemEval benchmark directly tests this failure mode through its Knowledge Update (KU) and Temporal Reasoning (TR) categories. On KU questions — which ask whether the system correctly tracks facts that change over time ("Where does Alice work?" with different answers in session 5 versus session 20) — full-context baselines without structured memory achieve only 60.3% accuracy [^13^]. On temporal reasoning — questions requiring understanding of *when* events occurred and their chronological relationships — the same baselines score just 31.6% [^13^]. These figures demonstrate that raw retrieval, even backed by frontier models, cannot compensate for the absence of temporal structure.

#### 1.2.3 No Trust Mechanism

In a standard RAG pipeline, every retrieved chunk receives equal weight in the prompt. There is no mechanism to distinguish a verified architectural decision from a speculative suggestion, a frequently referenced convention from a one-time exception, or a confirmed preference from a transient mood. All chunks are weighted equally regardless of verification status, usage frequency, or recency of confirmation.

This uniform weighting is particularly damaging for coding agents, where a single incorrect retrieved chunk can steer an implementation in the wrong direction. The Holographic memory system addresses this through asymmetric trust scoring that penalizes incorrect retrievals more heavily than it rewards correct ones, but this capability is absent from conventional RAG architectures [^14^]. Without trust calibration, agents cannot learn which memories are reliable and which should be treated with skepticism — a prerequisite for any system that accumulates knowledge over time.

### 1.3 What True Context Preservation Requires

Moving beyond RAG requires a reconceptualization of what agent memory must do. Memory is not a retrieval problem. It is a *knowledge management* problem that spans the full lifecycle of information: from the moment it enters the system, through synthesis and modeling, to its eventual recall under the right conditions at the right time.

#### 1.3.1 Six Capability Layers

Analysis of the current memory provider landscape — encompassing Hermes-compatible providers (Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, and Supermemory) as well as independent systems — reveals six distinct capability layers that any architecture aiming for true context preservation must address. No single existing system covers all six comprehensively. Each layer addresses a specific failure mode of RAG-based approaches and corresponds to a documented requirement from the research literature.

| Capability Layer | Core Function | What RAG Lacks | Representative Implementation |
|:---|:---|:---|:---|
| **Extraction** | Automatically identify salient facts, decisions, and preferences from conversation streams | Manual chunking; no semantic understanding of what matters | Mem0 server-side LLM extraction [^15^]; ByteRover pre-compression extraction [^16^] |
| **Synthesis** | Derive higher-level insights by reflecting across multiple memories; generate summaries and cross-memory connections | Retrieves isolated chunks with no cross-referencing or insight generation | Hindsight Reflect layer [^13^]; Honcho dialectic reasoning [^17^] |
| **User Modeling** | Build and maintain a persistent profile of the user's preferences, habits, communication style, and goals | No user representation; retrieves by query similarity to stored text, not by user model | Honcho multi-peer profiles [^17^]; Hindsight Entity/Observation Network [^13^] |
| **Contextual Recall** | Surface the *right* memory at the *right* time based on situational relevance, not just textual similarity | Retrieves by vector similarity regardless of situational appropriateness | OpenViking L0/L1/L2 tiered loading [^18^]; Hindsight multi-strategy retrieval [^13^] |
| **Temporal Awareness** | Track when facts were learned, detect contradictions over time, and resolve conflicts using chronology | Flattened history with no sequence or recency structure | Hindsight TEMPR temporal retrieval [^13^]; ByteRover Context Tree session ordering [^16^] |
| **Trust & Decay** | Score memories by reliability, penalize incorrect recollections, compress or fade stale information | All chunks weighted equally; no forgetting or consolidation mechanism | Holographic trust scoring [^14^]; Mem0 memory decay with category-specific rates [^15^] |

The table above frames the gap between what RAG provides and what agent memory demands. Extraction addresses the problem of manual curation — without automatic identification of what is worth remembering, memory systems remain write-only repositories that depend on users to manually flag important moments. Synthesis addresses the isolation problem: individual memories, even if accurately stored, are less valuable than the insights that emerge from their combination. A user who expressed frustration with a slow build pipeline in January and mentioned switching to Vite in March has communicated a preference that no single retrieved chunk captures. User modeling moves beyond individual facts to construct a coherent representation of the person the agent is serving — the communication style (concise versus detailed), the technical preferences (type safety over speed), the recurring workflows. Contextual recall, as distinct from raw retrieval, requires the system to understand *why* a memory is relevant to the current moment, not merely that its text is similar to the query. Temporal awareness provides the chronological backbone that makes contradiction detection and knowledge updating possible. Trust and decay provide the feedback loop that allows the system to improve its memory quality over time — reinforcing useful recollections and suppressing or compressing those that prove unreliable or outdated.

Each of these six layers is necessary but not sufficient in isolation. Extraction without synthesis produces a pile of unconnected facts. Synthesis without temporal awareness generates confident but potentially outdated conclusions. User modeling without trust calibration produces brittle profiles that cannot adapt to changing preferences. A complete agent memory architecture must integrate all six into a coherent pipeline where the output of each layer feeds into the next, creating a system that not only stores what was said but understands what it means, how it relates to other knowledge, whether it remains true, and when it should be brought to bear on the present moment. The next chapter examines how the Hermes Agent Memory ecosystem's eight providers each address subsets of these layers — and where significant architectural gaps remain.

---

## 2. The Hermes Agent Memory Ecosystem

Chapter 1 established that context loss is the fundamental bottleneck in long-running agent systems and that naive Retrieval-Augmented Generation (RAG) fails because it retrieves text chunks without understanding the user's evolving state. Hermes Agent, the open-source autonomous agent framework maintained by Nous Research, addresses this gap through a pluggable memory-provider architecture that treats cross-session knowledge as a first-class subsystem [^1^]. Rather than embedding memory logic directly into the agent core, Hermes defines an abstract protocol — the `MemoryProvider` ABC (Abstract Base Class) — and ships with eight distinct implementations, each representing a different philosophical and technical approach to persistent agent memory.

This chapter maps the Hermes memory ecosystem in three layers. Section 2.1 examines the protocol itself: the contract that every provider must satisfy, the six automatic operations Hermes performs on the provider's behalf, and the single-select constraint that governs provider activation. Section 2.2 presents the eight providers in a unified comparison framework, then reorganizes them by architectural approach to expose the design space's underlying structure. Section 2.3 explains how Hermes isolates memory namespaces across profiles, ensuring that credentials, stored knowledge, and retrieval surfaces never leak between distinct agent identities.

---

### 2.1 The Memory Provider Protocol

Hermes treats memory as a provider plugin type, one of two specialized plugin categories alongside context-engine plugins [^62^]. The design follows a consistent pattern: single-select, config-driven, and managed through `hermes plugins` and `hermes memory setup` commands. A provider is not a passive database connector; it is an active participant in the agent's conversation lifecycle, with hooks that fire before, during, and after every turn.

#### 2.1.1 The MemoryProvider ABC: Contract and Lifecycle

Every memory provider inherits from `MemoryProvider`, defined in `agent/memory_provider.py` and exposed through the `agent.memory_provider` module [^62^]. The contract separates required methods, which every provider must implement, from optional hooks that providers override only when their backend supports the corresponding capability.

The required methods form a four-phase lifecycle. First, discovery: the `name` property and `is_available()` method let Hermes enumerate and filter providers at agent initialization. The `is_available()` method must perform only local checks — typically verifying that an environment variable or configuration file exists — and must never issue network calls [^62^]. Second, activation: `initialize(session_id, **kwargs)` receives the active session identifier and the `hermes_home` path, which the provider must use for all storage operations to maintain profile isolation. Third, tool registration: `get_tool_schemas()` returns a list of tool definitions that Hermes injects into the agent's tool surface, and `handle_tool_call(name, args)` dispatches those calls at runtime. Fourth, configuration: `get_config_schema()` declares the fields prompted during `hermes memory setup`, while `save_config(values, hermes_home)` persists non-secret values to the provider's native configuration location [^62^].

The optional hooks extend this lifecycle into the conversation itself. `prefetch(query)` fires before each Large Language Model (LLM) API call, allowing the provider to retrieve relevant context and return it as a string that Hermes injects into the system prompt. `queue_prefetch(query)` fires after each turn to pre-warm caches for the next exchange. `sync_turn(user, assistant)` persists each completed conversation turn to the provider's backend. `on_session_end(messages)` triggers final extraction or flush operations when a conversation terminates. `on_pre_compress(messages)` fires before Hermes compresses a long conversation, giving the provider a chance to capture insights before they are summarized away. `on_memory_write(action, target, content)` mirrors built-in memory-file operations to the external backend. Finally, `shutdown()` cleans up connections and threads at process exit [^62^].

The threading contract is strict: `sync_turn()` must be non-blocking. Providers with latent backends — cloud APIs, LLM-driven extraction pipelines, or database writes — must delegate the actual work to a daemon thread with a bounded join timeout, typically five seconds [^62^]. This ensures that a slow memory sync never stalls the agent's response path.

#### 2.1.2 Six Automatic Operations

When an external memory provider is active, Hermes performs six operations automatically, without requiring tool calls or explicit agent coordination [^1^]:

1. **Context injection.** Before each LLM call, Hermes calls the provider's `prefetch()` hook and inserts any returned context into the system prompt. This happens in the background and is non-blocking — the turn proceeds even if prefetch is slow.

2. **Prefetch.** After each completed turn, Hermes calls `queue_prefetch()` to pre-warm the provider's retrieval caches for the upcoming exchange, reducing latency on the next turn's context injection.

3. **Sync.** After each assistant response, Hermes calls `sync_turn()` with the user message and assistant response, persisting the conversation turn to the provider's backend. This is the primary ingestion pipeline.

4. **Session-end extraction.** When a conversation ends, Hermes calls `on_session_end()` with the full message history. Providers that support automatic memory extraction — such as Honcho's dialectic reasoning pass or Supermemory's graph ingest — use this hook to perform final synthesis.

5. **Mirroring.** Whenever the agent writes to the built-in memory files (MEMORY.md or USER.md), Hermes calls `on_memory_write()` with the action (add, update, delete), target file, and content, allowing the external provider to mirror these operations in its own storage layer.

6. **Tool registration.** At startup, Hermes calls `get_tool_schemas()` and injects the provider's tools into the agent's tool surface. The agent can then invoke these tools explicitly to search, store, curate, or manage memories on demand [^1^].

These six operations form a complete lifecycle: passive recall through prefetch and injection, active persistence through sync and mirroring, deferred synthesis through session-end extraction, and agent-driven management through tool registration. The built-in memory system — MEMORY.md and USER.md — continues to operate unchanged; the external provider is purely additive [^1^].

#### 2.1.3 Single-Select Architecture

Hermes enforces a hard constraint: only one external memory provider can be active at any time [^62^]. If a user attempts to register a second provider, the `MemoryManager` rejects it with a warning. This "single-select" architecture prevents two categories of failure. First, it avoids tool-schema bloat: each provider exposes between two and five tools, and stacking all eight would overwhelm the agent's tool surface with dozens of memory-related functions, increasing both token consumption and mis-invocation risk. Second, it eliminates backend conflicts — two providers attempting to sync the same conversation turn, inject conflicting context blocks, or mirror writes to incompatible storage systems would produce unpredictable behavior.

The selection is controlled by the `memory.provider` key in `config.yaml` and can be changed interactively via `hermes memory setup` or `hermes config set memory.provider <name>` [^1^]. Switching providers is a profile-scoped operation: each Hermes profile maintains its own `memory.provider` setting, so a user can run Honcho on their "personal" profile and Holographic on their "coding" profile without interference.

---

### 2.2 The Eight Memory Providers at a Glance

Hermes ships with eight built-in memory providers. As of May 2026, the set is closed: Nous Research no longer accepts new in-tree providers, directing contributors to publish standalone plugin repositories that implement the same `MemoryProvider` ABC and register through the same discovery path [^78^]. The existing eight therefore represent the curated, officially supported spectrum of memory architectures available to Hermes builders.

#### 2.2.1 Full Comparison Table

| Provider | Storage | Cost | Tools | Dependencies | Unique Feature |
|---|---|---|---|---|---|
| Honcho | Cloud | Paid | 5 | `honcho-ai` | Dialectic user modeling + session-scoped context [^1^] |
| OpenViking | Self-hosted | Free | 5 | `openviking` + server | Filesystem hierarchy + tiered loading (L0/L1/L2) [^1^] |
| Mem0 | Cloud | Paid | 3 | `mem0ai` | Server-side LLM extraction [^1^] |
| Hindsight | Cloud / Local | Free / Paid | 3 | `hindsight-client` | Knowledge graph + reflect synthesis [^1^] |
| Holographic | Local | Free | 2 | None | HRR algebra + trust scoring [^1^] |
| RetainDB | Cloud | $20/mo | 5 | `requests` | Full chronological retrieval + delta compression [^1^] |
| ByteRover | Local / Cloud | Free / Paid | 3 | `brv` CLI | Pre-compression extraction + human-readable Markdown tree [^1^] |
| Supermemory | Cloud | Paid | 4 | `supermemory` | Context fencing + session graph ingest + multi-container [^1^] |

The table reveals significant diversity across multiple dimensions. Tool count varies from two (Holographic) to five (Honcho, OpenViking, RetainDB), directly impacting the granularity of agent-driven memory control. Cost models range from fully free with zero dependencies (Holographic) to subscription-only cloud services (RetainDB at $20/month). Storage location splits roughly evenly: four providers are cloud-first (Honcho, Mem0, RetainDB, Supermemory), two are local-first (Holographic, ByteRover), and two offer hybrid or self-hosted options (Hindsight, OpenViking).

The "Dependencies" column highlights a friction gradient. Holographic requires no pip packages at all — it runs on Python's built-in SQLite module with NumPy as an optional enhancement for Holographic Reduced Representation (HRR) algebra [^1^]. At the opposite extreme, OpenViking requires both the `openviking` Python package and a running server instance, typically deployed via Docker [^71^]. This dependency gradient is a primary selection criterion for builders: air-gapped or low-friction environments favor Holographic or ByteRover, while teams with existing DevOps infrastructure can accommodate OpenViking or self-hosted Hindsight.

The unique-feature column captures each provider's architectural differentiator. Honcho's dialectic reasoning builds a behavioral model of the user rather than merely storing facts [^36^]. OpenViking's tiered loading reduces token consumption by 80–90% compared to flat RAG retrieval [^71^]. Hindsight's reflect operation performs cross-memory synthesis, periodically reading across all stored memories to derive higher-level insights that no other provider generates [^102^]. Holographic's trust scoring assigns asymmetric feedback weights (+0.05 for helpful, −0.10 for unhelpful), enabling the memory store to self-correct over time [^35^]. RetainDB's chronological retrieval philosophy guarantees that the answering model sees the complete memory timeline rather than a lossy semantic subset [^102^]. ByteRover's pre-compression extraction fires before Hermes summarizes long conversations, capturing in-flight knowledge that would otherwise be discarded [^32^]. Supermemory's context fencing strips recalled memories from captured turns, preventing recursive memory pollution [^54^].

#### 2.2.2 Categorization by Architectural Approach

While the comparison table organizes providers by name and operational characteristics, a more analytically useful framework groups them by their underlying memory philosophy. The eight providers instantiate six distinct architectural approaches, some of which overlap when a provider combines multiple strategies.

| Architectural Approach | Representative Provider(s) | Core Abstraction | Retrieval Strategy | Key Trade-off |
|---|---|---|---|---|
| **Vector RAG** | Mem0, RetainDB | Vector embeddings + BM25 keyword search | Hybrid semantic + keyword fusion | Simple and fast, but retrieves raw chunks without synthesis |
| **Knowledge Graph** | Hindsight | Structured facts, entities, and relationships | TEMPR: Temporal + Entity + Metadata + BM25 in parallel | Highest precision and cross-memory reasoning, but requires structured extraction |
| **User Modeling** | Honcho | Peer representations + dialectic conclusions | Cold/warm prompt selection with LLM-synthesized reasoning | Deepest user understanding, but cloud-dependent and AGPL-licensed |
| **Structured Hierarchy** | OpenViking, ByteRover | Hierarchical directories (filesystem) or Markdown trees | Tiered traversal: abstract → overview → full detail | Human-navigable and token-efficient, but requires curation discipline |
| **Algebraic** | Holographic | Holographic Reduced Representations (HRR vectors) | Algebraic composition: binding, probing, AND reasoning across entities | Zero dependencies and extremely fast, but representation is lossy vs. full text |
| **Semantic Graph** | Supermemory | Entity-centric semantic graph with typed relationships | Graph traversal across Update / Extend / Derive edges | Rich relationship modeling, but cloud-only and most complex pricing |

Mem0 and RetainDB both fall under the vector-RAG umbrella, though they differ in execution. Mem0 operates a managed triple-store — vector embeddings, entity store, and SQL audit log — with server-side LLM extraction that runs on Mem0's infrastructure, not the user's machine [^77^]. RetainDB, by contrast, emphasizes full chronological retrieval: instead of returning the top-K most similar chunks, it presents the answering model with a complete, temporally ordered timeline of all relevant memories [^102^]. This philosophical difference — approximate semantic relevance versus exact temporal coverage — defines the trade-off within the vector-RAG category.

Hindsight stands alone as the knowledge-graph provider. Its TEMPR retrieval engine runs four strategies in parallel — Temporal, Entity, Metadata, and BM25 — then fuses results through a reranking stage [^102^]. The reflect operation, unique to Hindsight, periodically reads across the entire memory bank to derive higher-level insights and consolidate related facts, effectively performing automated cross-memory reasoning [^102^]. On LongMemEval, the standard benchmark for long-context agent memory, Hindsight achieves 91.4% with Gemini-3, 89.0% with a 120-billion-parameter open-source model, and 83.6% with a 20-billion-parameter model — the highest scores of any Hermes provider [^102^].

Honcho occupies the user-modeling category exclusively. While other providers store facts, Honcho's reasoning engine — termed the "Deriver" — continuously processes incoming messages to generate peer representations, session summaries, and dialectic conclusions [^106^]. The dialectic system automatically selects between cold-start prompts ("Who is this person? What are their preferences?") and warm-session prompts ("Given what's been discussed, what context is most relevant?") [^110^]. Three orthogonal configuration knobs — `contextCadence`, `dialecticCadence`, and `dialecticDepth` — let builders tune cost and reasoning depth independently [^110^]. This architecture makes Honcho the only provider whose primary goal is building a behavioral model of the user rather than merely indexing conversation text.

OpenViking and ByteRover share the structured-hierarchy approach but implement it differently. OpenViking uses a virtual filesystem accessible through the `viking://` URI scheme, organizing memories, resources, and skills into directories with automatic L0/L1/L2 summary generation [^71^]. ByteRover stores knowledge as human-readable Markdown files in `.brv/context-tree/`, organized into domains and topics, with a curation engine that supports five operations: ADD, UPDATE, UPSERT, MERGE, and DELETE [^32^]. Both prioritize human inspectability and navigability over opaque embedding spaces, making them attractive for debugging and audit scenarios.

Holographic is the sole algebraic provider. HRR vectors encode concepts into a compressed representational space where retrieval becomes algebraic rather than similarity-based [^35^]. The `probe` operation recalls all facts about a specific entity; the `reason` operation performs compositional AND queries across multiple entities; and the `contradict` operation detects conflicting facts automatically [^1^]. With zero external dependencies — only Python's built-in SQLite module — Holographic is the fastest provider to deploy and the most suitable for offline or air-gapped environments [^102^].

Supermemory represents the semantic-graph approach. It distinguishes between Documents (raw input) and Memories (intelligent knowledge units), then connects memories through three relationship types: Updates (information changes), Extends (information enriches), and Derives (information infers) [^56^]. Context fencing prevents recalled memories from being re-ingested into the graph, a form of recursive pollution that plagues naive memory systems [^54^]. Supermemory also offers the broadest integration surface, with connectors for Google Drive, Notion, OneDrive, Gmail, GitHub, and S3, positioning it as an all-in-one memory and RAG platform rather than a pure conversation-memory system [^54^].

---

### 2.3 Profile Isolation Architecture

Hermes supports multiple profiles — distinct agent identities, each with its own configuration, memory files, plugins, and credentials [^1^]. When a memory provider is active, its data and configuration must remain strictly isolated per profile to prevent cross-contamination. The Hermes memory subsystem implements this isolation through five mechanisms, each matched to the provider's storage and configuration model.

#### 2.3.1 Five Isolation Mechanisms

The isolation strategy depends on where and how a provider stores its data. Hermes distinguishes five patterns, each requiring a different isolation technique [^1^]:

**Local storage isolation.** Providers that store data on the local filesystem — Holographic and ByteRover — use the `hermes_home` path passed to `initialize()` as their root directory. Holographic defaults to `$HERMES_HOME/memory_store.db` for its SQLite database; ByteRover stores its context tree under `.brv/context-tree/` relative to the current working directory [^1^]. Because each Hermes profile has a distinct `$HERMES_HOME` value (for example, `~/.hermes` for the default profile and `~/.hermes.coder` for a profile named "coder"), the filesystem paths diverge automatically. The provider implementation must never hardcode `~/.hermes` and must always derive storage locations from the `hermes_home` argument or the `get_hermes_home()` utility [^62^].

**Config file isolation.** Providers that persist credentials and settings in configuration files — Honcho, Mem0, Hindsight, and Supermemory — store these files under `$HERMES_HOME/`. Honcho uses `honcho.json`, Hindsight uses `hindsight/config.json`, Supermemory uses `supermemory.json`, and Mem0 stores its configuration in the provider's own format within the Hermes home directory [^1^]. Because each profile has a separate `$HERMES_HOME`, each profile maintains its own API keys, container tags, and provider settings. A user can run Honcho's cloud service on their personal profile while running Hindsight in local mode on their coding profile, with no credential leakage between the two.

**Cloud-derived isolation.** Cloud-only providers whose backends support multi-tenancy use auto-derived, profile-scoped identifiers. RetainDB auto-generates profile-scoped project names, ensuring that memories from one Hermes profile map to a distinct namespace in the RetainDB cloud backend [^1^]. This removes the need for the user to manually configure project separation while still guaranteeing that retrieval from one profile never surfaces memories from another.

**Environment variable isolation.** Providers configured entirely through environment variables — primarily OpenViking — rely on each profile's `.env` file [^1^]. When Hermes activates a profile, it loads that profile's `.env` into the process environment. OpenViking's endpoint URL and API key are read from variables such as `OPENVIKING_ENDPOINT` and `OPENVIKING_API_KEY`, which can differ per profile. Switching profiles automatically switches the OpenViking target server, achieving isolation without explicit namespacing in the provider code.

**Multi-peer workspace isolation.** Honcho adds a fifth isolation layer through its multi-peer architecture. Within a single Honcho workspace, each Hermes profile receives its own AI peer identity [^110^]. When two profiles interact with the same user, Honcho maintains separate peer cards and representations for each profile's AI peer, preventing cross-contamination at the user-modeling level [^110^]. This is isolation *within* the provider, not just *of* the provider — a finer granularity than the other mechanisms support.

#### 2.3.2 Automatic Namespace Separation

The practical consequence of these five mechanisms is that a builder never needs to manually configure memory isolation. When a new profile is created via `hermes profiles create <name>`, Hermes generates a fresh `$HERMES_HOME` directory. All four isolation categories activate automatically: local-storage providers get a fresh directory tree, config-file providers start with empty configuration files, cloud providers receive auto-derived profile-scoped identifiers, and environment-variable providers inherit a fresh `.env` file.

The `hermes memory setup` command respects this isolation. When run inside a profile, it writes configuration only to that profile's `$HERMES_HOME` and validates credentials against that profile's `.env` [^1^]. The `hermes memory status` command reports the active provider for the current profile only. Switching profiles with `hermes profiles switch <name>` changes the active memory provider, its credentials, and its data namespace atomically.

This architecture enables sophisticated multi-agent workflows. A developer might maintain three profiles: "default" running Holographic for zero-dependency local tasks, "team" running Hindsight in local mode for structured coding knowledge shared across a codebase, and "personal" running Honcho for user-modeling across all personal conversations. Each profile's memory provider, credentials, stored knowledge, and retrieval behavior are fully independent, yet all operate through the identical `MemoryProvider` ABC contract that Hermes manages uniformly.

The isolation guarantees extend to the built-in memory system as well. MEMORY.md and USER.md live inside `$HERMES_HOME/`, so each profile maintains its own agent notes and user-profile files alongside the external provider's data [^1^]. The external provider's mirroring hook (`on_memory_write`) receives writes only from its own profile's built-in memory files, never from another profile's files. This end-to-end isolation — from filesystem paths through configuration files to cloud namespaces and peer identities — ensures that a memory provider activated in one profile behaves as if no other profile exists, giving builders the confidence to experiment with different memory strategies without risk of data leakage.

---

## 3. Deep Vendor Analysis: The Four Architectural Schools

The eight memory providers examined in Chapter 2 do not cluster randomly. When analyzed by their core design philosophy, they resolve into four distinct architectural schools, each offering a fundamentally different answer to the question of how an agent should remember. These schools are: (1) **Structured Knowledge Extraction**, which treats memory as a curated knowledge graph that must be actively synthesized rather than passively stored; (2) **Dialectic User Modeling**, which prioritizes building a living model of the user through ongoing reasoning rather than simple fact retrieval; (3) **Tiered Context Engineering**, which addresses memory at the level of context economy—reducing token consumption through hierarchical loading; and (4) **Semantic Memory Graphs**, which encode relationships between memories as first-class primitives in a persistent graph structure.

This chapter examines each school in depth, profiling both vendors within it and analyzing the specific architectural decisions that produce their respective benchmark outcomes. Every school offers at least one technique that a next-generation memory system should consider for adoption; the analysis that follows isolates precisely which ideas transfer across architectures and which are tightly coupled to their original design context.

---

### 3.1 School 1: Structured Knowledge Extraction (Hindsight, ByteRover)

The defining conviction of this school is that raw conversational text is an unsuitable substrate for long-term agent memory. Both Hindsight and ByteRover transform every incoming interaction through a structured extraction pipeline before it ever enters storage. Where a conventional Retrieval-Augmented Generation (RAG) system stores text chunks and searches them at query time, these providers insist that meaning must be captured, organized, and synthesized at ingestion time. The trade-off is additional latency during the write path; the payoff is retrieval precision that consistently exceeds 90% on standard benchmarks.

#### 3.1.1 Hindsight's TEMPR Architecture: Four Parallel Retrieval Strategies

Hindsight, developed by Vectorize and published in a December 2025 arXiv paper, organizes its retrieval layer around an acronym: TEMPR (Temporal, Entity, Metadata, and BM25 for exact keyword matching) [^94^]. The four strategies execute in parallel, and their results are aggregated through a reranking stage before being injected into the agent's context window.

The **Temporal** strategy addresses a specific failure mode of vector-only systems: they cannot answer "when" questions. By indexing memories with explicit temporal annotations—session boundaries, timestamps, and sequential ordering—Hindsight can answer queries such as "What did the user prefer *before* the migration?" that pure semantic similarity would miss entirely. The **Entity** strategy maintains a resolved entity index so that pronouns and aliases map to canonical identifiers; "Chris" and "the CTO" and "he" all resolve to the same entity node if the system has sufficient evidence. **Metadata** enables filtering by memory type (world knowledge, experience, opinion, observation), project scope, and confidence tier. **BM25** (Best Match 25), a probabilistic ranking function for keyword retrieval, handles exact terminology matches that embedding-based approaches often dilute [^106^].

The combined pipeline produces retrieval results that no single strategy could achieve alone. On LongMemEval, a benchmark of 500 questions across five core abilities—Information Extraction (IE), Multi-session Reasoning (MR), Temporal Reasoning (TR), Knowledge Update (KU), and Abstention (ABS)—Hindsight scores 91.4% overall with Gemini 3 Pro Preview, rising to 94.6% in the "S" setting with 115,000 tokens across approximately 50 sessions [^94^]. The system also achieves 64.1% on BEAM and 89.6% on LoCoMo [^94^]. These scores represent the highest published results of any provider in this analysis.

#### 3.1.2 The Reflect Operation: Cross-Memory Synthesis

The most architecturally distinctive feature of Hindsight is the `reflect` operation, a periodic pass that reads across all stored memories to derive higher-level insights and consolidate related facts [^94^]. This is not merely an optimization; it is a design primitive that no other provider in the eight-vendor set implements. The reflect pass functions as a background reasoning process, analogous to human memory consolidation during sleep. It detects patterns across disparate sessions, elevates frequently corroborated observations into higher-confidence beliefs, and surfaces emergent relationships that individual retrievals would miss.

The reflect operation closes a critical loop in the memory lifecycle. Hindsight's three-stage pipeline—**retain** (ingest) → **recall** (retrieve) → **reflect** (synthesize)—ensures that memory is not merely accumulated but continuously refined. The `hindsight_reflect` tool is exposed directly to the agent, meaning the agent itself can request a synthesis pass when it senses that accumulated observations may contain a latent insight [^106^]. This is a fundamentally different interaction model from providers where memory is write-and-forget.

#### 3.1.3 ByteRover's Knowledge Tree: Human-Readable Markdown Hierarchy

ByteRover takes a different approach to structured knowledge: rather than building a graph of entities and relationships, it organizes memory into a hierarchical knowledge tree of human-readable Markdown files stored in `.brv/context-tree/` [^93^]. The tree structure follows a semantic hierarchy—domain → topic → subtopic—that makes the agent's entire knowledge base navigable by a human operator without tooling.

This decision has practical implications for agent systems where observability and debuggability are requirements. Because every memory is a Markdown file, version control (via Git) applies natively. Diffs between memory states are human-readable. When a retrieval fails, an engineer can browse the context tree directly to identify whether the failure stems from missing extraction, incorrect categorization, or structural misplacement. The architecture trades the density of a graph representation for the inspectability of a filesystem.

#### 3.1.4 Pre-Compression Extraction: Capturing In-Flight Knowledge

ByteRover implements a mechanism it calls **pre-compression extraction**, which fires specifically before the host agent's context window is compressed [^97^]. When a long conversation exceeds the token threshold and must be summarized to fit within the model's context limit, valuable in-flight knowledge is at risk of being discarded. ByteRover intercepts this moment, extracts the most salient facts from the conversation before compression occurs, and curates them into the knowledge tree. This hook ensures that knowledge generated during a session survives the session's own cleanup process—a failure mode that affects any agent system relying on context-window management without explicit pre-compression capture.

#### 3.1.5 Curation Engine: Five Operations for Memory Maintenance

ByteRover exposes a curation engine with five discrete operations: **ADD**, **UPDATE**, **UPSERT**, **MERGE**, and **DELETE** [^93^]. Unlike systems where memories are appended passively, ByteRover requires the agent to decide how new information relates to existing knowledge. An UPSERT replaces an outdated fact; a MERGE combines two overlapping entries; a DELETE removes information the user has explicitly contradicted. This curation discipline produces a knowledge base that is actively maintained rather than monotonically growing, which directly contributes to ByteRover's temporal reasoning score of 94.4% on LoCoMo—best-in-class by a wide margin over Hindsight (83.8%) and Memobase (85.1%) [^99^].

On the LoCoMo benchmark, ByteRover 2.0 achieves 92.2% overall accuracy with the following category breakdown: single-hop 95.4%, temporal 94.4%, multi-hop 85.1%, and open-domain 77.2% [^99^]. A later academic evaluation using a harness with Gemini 3 Flash as judge reports even higher scores: 96.1% overall, with particularly strong gains on multi-hop questions where ByteRover outperforms the next-best system (Honcho at 89.9%) by 6.2 percentage points [^93^].

**Table 3.1: Structured Knowledge Extraction School — Feature and Benchmark Comparison**

| Dimension | Hindsight | ByteRover |
|:---|:---|:---|
| **Core abstraction** | Structured knowledge graph (facts, entities, relationships) | Hierarchical Markdown knowledge tree |
| **Extraction timing** | At retain-time (structured extraction) | Pre-compression + per-turn extraction |
| **Retrieval strategy** | TEMPR: Temporal + Entity + Metadata + BM25 (parallel) | Tiered: fuzzy text → LLM-driven search |
| **Synthesis mechanism** | `reflect` — cross-memory synthesis pass | Curation engine (ADD/UPDATE/UPSERT/MERGE/DELETE) |
| **Storage format** | Embedded PostgreSQL (local) or cloud | Human-readable Markdown files in `.brv/context-tree/` |
| **LongMemEval (overall)** | 91.4–94.6% [^94^] | 92.8% (LongMemEval-S) [^93^] |
| **LoCoMo (overall)** | 89.6% [^94^] | 92.2–96.1% [^93^] [^99^] |
| **Single-hop recall** | 86.2% [^99^] | 95.4% [^99^] |
| **Temporal reasoning** | 83.8% [^99^] | 94.4% [^99^] |
| **Multi-hop reasoning** | 70.8% [^99^] | 85.1% [^99^] |
| **Key differentiator** | `reflect` synthesis operation | Human-readable, Git-friendly Markdown tree |
| **License** | MIT | Proprietary (freemium) |
| **Dependencies** | Embedded PostgreSQL | None (local-first) |

The data in Table 3.1 reveals a revealing performance profile. Hindsight dominates on LongMemEval's full setting (94.6%) and demonstrates the strongest cross-session synthesis capability, attributable to its `reflect` operation and entity-aware retrieval. ByteRover, however, leads on LoCoMo and achieves dramatically better single-hop and temporal scores. The divergence is architectural: Hindsight's graph excels at connecting distant sessions through entity and temporal links, while ByteRover's curated tree with embedded timestamps provides more precise local grounding. A system architect choosing between these approaches should consider whether the dominant access pattern is cross-session synthesis (favoring Hindsight) or precise temporal and single-hop recall (favoring ByteRover). The most defensible design may incorporate both: a graph for relationship traversal and a curated tree for temporal precision.

---

### 3.2 School 2: Dialectic User Modeling (Honcho, Mem0)

Where School 1 treats memory as structured knowledge to be extracted and organized, School 2 treats memory as a model of the user to be continuously refined through reasoning. The central abstraction is not the fact but the *peer*—a representation of the human interlocutor that evolves over time. Both Honcho and Mem0 inject user-specific context into the agent's prompt before each response, but they differ sharply in the sophistication of that injection and in the philosophical question of whether memory should be agent-modeled or server-extracted.

#### 3.2.1 Honcho's Two-Layer Context Injection: Base Layer Plus Dialectic Supplement

Honcho, developed by Plastic Labs and the original external memory provider for the Hermes agent ecosystem, implements a two-layer context injection system [^103^]. The **base layer** assembles three components in a specific order: (1) a session summary generated automatically at the start of each turn, providing immediate conversational continuity; (2) a **user representation**, which is Honcho's accumulated model of the user's preferences, facts, and behavioral patterns; and (3) an **AI peer card**, which captures the identity and observed patterns of the agent itself [^101^].

The **dialectic supplement** is where Honcho diverges from all other providers. Instead of merely retrieving stored facts, Honcho's dialectic layer engages in on-demand reasoning about the user. When the agent asks a question about the user—"Does this user prefer concise or detailed responses?"—Honcho does not simply search for a stored preference. It synthesizes an answer from conversation history using an LLM reasoning pass. This produces insights that were never explicitly stored but can be inferred from observed behavior [^101^].

Honcho further implements an automatic **cold/warm prompt selection** strategy. When no prior session exists or the user representation is empty, Honcho selects a lightweight cold-start prompt that encourages the model to learn about the user actively. Once a representation has accumulated, it switches to warm-start mode with full base context injection [^103^]. This bifurcation avoids wasting tokens on rich context injection when the context does not yet exist.

#### 3.2.2 Three Orthogonal Config Knobs: Cost, Depth, and Frequency

Honcho exposes three configuration parameters that give operators direct control over the cost-reasoning trade-off: `contextCadence` (how often context is injected), `dialecticCadence` (how often dialectic reasoning is invoked), and `dialecticDepth` (the number of reasoning passes, from 1 to 3) [^103^]. These knobs are orthogonal, meaning an operator can increase context frequency without increasing reasoning depth, or deepen reasoning without increasing frequency. This separation is architecturally significant because it acknowledges that context injection and reasoning synthesis have different cost profiles and different optimal cadences. Most providers conflate these into a single retrieval call, removing the operator's ability to tune them independently.

#### 3.2.3 Multi-Peer Workspace: Separate AI Peer Profiles

Honcho models every conversation as an interaction between **peers** [^103^]. The user peer represents the human; the AI peer represents the agent instance. Critically, each AI profile gets its own independent AI peer, meaning different agent personas develop distinct representations of the same user based on their own observations. A coding assistant and a writing coach interacting with the same human will build different models of that user's preferences, because their interaction contexts differ. This multi-peer architecture prevents a monolithic user profile from collapsing contextually distinct observations into a single, incoherent representation.

The reasoning engine—internally called the **deriver**—runs as a background process that extracts premises from messages, draws conclusions, and updates representations [^103^]. Honcho exposes five tools to the agent: `honcho_profile` (fast peer card retrieval with no LLM overhead), `honcho_search` (semantic search over stored memories), `honcho_context` (dialectic Q&A powered by Honcho's LLM), `honcho_reasoning` (explicit reasoning invocation), and `honcho_conclude` (durable fact writeback when the user states preferences or corrections) [^101^].

#### 3.2.4 Mem0's Dual Memory Scope: Session Plus User Memories

Mem0, the most widely adopted provider in this analysis with 51,400 GitHub stars, takes a fundamentally different approach to the same problem [^96^]. Rather than building a dialectic model of the user, Mem0 operates as a **hybrid triple-store**: a vector database for semantic search, a key-value store for fast fact lookup, and a knowledge graph for relationship-aware retrieval [^96^]. Every incoming message passes through a server-side LLM extraction layer that decides what facts are worth storing and how they should be classified across the three storage layers.

Mem0's most consequential architectural decision is its **dual memory scope**. Memories are classified as either **session memories** (short-term, scoped to the current conversation) or **user memories** (long-term, persistent across all conversations) [^96^]. Both scopes are searched and injected before each response, but they decay at different rates. Session memories expire naturally when the session ends; user memories persist until explicitly updated or deleted. This scoping prevents transient conversational details from polluting the long-term user model while ensuring that recent context remains available within the current session.

#### 3.2.5 Mem0's Hybrid Triple-Store: Vector Plus Key-Value Plus Knowledge Graph

The triple-store architecture deserves detailed examination because it represents an attempt to unify three historically separate retrieval paradigms. The vector layer handles semantic similarity: "find memories about travel" retrieves both explicit travel mentions and semantically related content. The key-value layer handles precise fact lookup: "user's preferred language is Python" is stored as a directly addressable pair. The knowledge graph layer handles relational inference: if the system knows the user works at Company X and Company X is in Industry Y, it can infer industry-relevant context even if the user never explicitly stated their industry preference.

Mem0 also implements a **circuit breaker pattern** for resilience: if the memory API fails five consecutive times, the circuit opens and stops calling the API for two minutes, allowing the agent to continue operating without memory rather than failing entirely [^96^]. This graceful degradation pattern is notable because most providers do not isolate memory failures from agent operation.

On LongMemEval-S, Mem0 scores 67.6% with GPT-4o [^96^]. This score, while substantially below Hindsight and ByteRover, must be evaluated in context: Mem0 optimizes for setup speed and operational resilience rather than raw retrieval precision. Its Apache 2.0 license and self-host option make it the default choice for teams that need a commercially unencumbered memory layer without custom engineering.

**Table 3.2: Dialectic User Modeling School — Feature and Benchmark Comparison**

| Dimension | Honcho | Mem0 |
|:---|:---|:---|
| **Core abstraction** | Dialectic peer model (user + AI representations) | Hybrid triple-store (vector + key-value + knowledge graph) |
| **Context injection** | Two-layer: base (summary + representation + peer card) + dialectic supplement | Dual-scope: session memories + user memories, both injected per turn |
| **User modeling** | LLM-synthesized dialectic reasoning on-demand | Server-side LLM extraction at ingestion time |
| **Configurability** | Three orthogonal knobs: contextCadence, dialecticCadence, dialecticDepth (1–3) | Minimal configuration; automatic extraction decisions |
| **Multi-persona support** | Multi-peer workspace: separate AI peer per agent persona | Unified user profile across all agent instances |
| **Resilience pattern** | No explicit circuit breaker | Circuit breaker: 5 failures → 2-minute API suspension [^96^] |
| **LongMemEval-S** | — | 67.6% (GPT-4o) [^96^] |
| **Tools exposed** | 5: profile, search, context, reasoning, conclude | 3: profile, search, conclude |
| **License** | AGPL-3.0 | Apache 2.0 |
| **Setup characteristic** | Deeper integration, more config surface | 30-second setup; fastest time-to-first-memory |
| **Key differentiator** | Dialectic reasoning layer with orthogonal cost controls | Triple-store unification + circuit breaker resilience |

The comparison in Table 3.2 exposes a fundamental architectural fork in the user-modeling road. Honcho invests in depth: its dialectic layer produces insights that no extraction-only system could generate, but at the cost of additional LLM calls and a more complex configuration surface. Its AGPL-3.0 license is a strategic constraint for commercial deployments, as self-hosting triggers copyleft obligations [^96^]. Mem0 invests in breadth and accessibility: the triple-store handles more query types without reasoning overhead, the circuit breaker ensures operational continuity, and the Apache 2.0 license removes legal friction. A system architect should adopt Honcho's dialectic approach when the user model must support inferential questions that go beyond stored facts; Mem0's triple-store is the pragmatic default for teams that need memory coverage quickly without per-user reasoning costs.

---

### 3.3 School 3: Tiered Context Engineering (OpenViking, Holographic)

School 3 addresses memory from the opposite direction of Schools 1 and 2. Rather than asking "how do we extract and organize knowledge?" these providers ask "how do we deliver the right context at the right cost?" The core problem they solve is **context economy**: minimizing token consumption while preserving retrieval accuracy. Both employ tiered or compressed representations, but their mechanisms differ substantially—OpenViking through explicit hierarchical levels, Holographic through algebraic compression.

#### 3.3.1 OpenViking's L0/L1/L2 Tiered Loading: Abstract to Overview to Full Detail

OpenViking, developed by the Volcengine team at ByteDance, replaces flat vector storage with a hierarchical virtual filesystem accessed via a `viking://` URI scheme [^71^]. Every piece of context—memories, resources, and skills—is organized into directories, and every directory contains three automatically generated layers:

- **L0 (Abstract)**: A one-sentence summary of approximately 100 tokens, used for quick identification and cheap relevance checking.
- **L1 (Overview)**: Core information and usage scenarios of approximately 2,000 tokens, sufficient for planning and decision-making.
- **L2 (Detail)**: The full original content, loaded only on demand when deep reading is required [^61^].

The tiered loading model enables an agent to "skim" before it "reads." Consider an incident-response agent consulting 50 internal runbooks. With traditional RAG, loading retrieved chunks might consume over 50,000 tokens. With OpenViking, the agent scans 50 L0 abstracts (~5,000 tokens), narrows to three relevant runbooks via their L1 overviews (~6,000 tokens), and pulls full L2 detail only for the single runbook it actually needs. Red Hat's evaluation of this pattern reports **80–90% token reduction** compared to flat loading approaches [^61^].

The L0/L1/L2 decomposition is performed automatically at write time using the configured LLM, meaning the tiering is consistent across all stored content without manual curation. On session commit, OpenViking also extracts memories into six categories: profile, preferences, entities, events, cases, and patterns [^71^]. This six-category extraction provides a secondary organizational axis orthogonal to the directory hierarchy, enabling retrieval by content type as well as by location.

#### 3.3.2 Directory Recursive Retrieval with Logged Trajectories

OpenViking's retrieval mechanism mirrors the directory structure: it uses vector similarity to identify the correct directory, performs a secondary search within that directory, and drills down recursively into subdirectories [^71^]. Every step of this traversal is logged as a **visible trajectory**, which means retrieval is fully observable and debuggable. When an agent retrieves the wrong context, an engineer can trace the exact path the retrieval algorithm took through the filesystem hierarchy and identify where it diverged from the expected path [^61^].

This observability property contrasts sharply with vector-only retrieval, where the path from query to result is a single similarity computation across an embedding space—a process that offers little insight when it fails. Directory recursive retrieval also enables a form of "structural priors" where the agent can constrain searches to specific subtrees, effectively applying scope filters without explicit metadata indexing.

#### 3.3.3 Holographic's HRR Algebra: Compositional Queries on Local SQLite

Holographic, built directly into the Hermes agent framework by Nous Research, takes a radically different approach to context economy. Instead of tiering content by detail level, it compresses memory into a mathematical representation using **Holographic Reduced Representations (HRR)** [^35^]. HRR is a family of techniques where information is stored as superposed complex-valued vectors, and recall is performed algebraically rather than through similarity search. The practical implication is retrieval in the sub-millisecond range, running on pure SQLite with zero external dependencies beyond NumPy (which is optional) [^4^].

The HRR algebra enables compositional queries that vector similarity cannot express. The `probe` action retrieves all facts about a specific entity ("everything about Person X"). The `reason` action performs compositional AND queries across multiple entities ("find facts where both Person X and Project Y are involved") [^1^]. These operations are algebraic: they manipulate the compressed representations mathematically rather than searching through an index. The result is a memory system with nine discrete actions (`add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`) exposed through a single `fact_store` tool, plus a `fact_feedback` tool for trust scoring [^1^].

#### 3.3.4 Trust Scoring: Asymmetric Feedback with Automated Contradiction Detection

Holographic's most transferable innovation is its **trust scoring** mechanism. Every fact is assigned a trust score between 0.0 and 1.0, with a default of 0.5. When a recalled memory is rated as helpful by the agent or user, its trust score increases by +0.05. When rated as unhelpful, the score decreases by -0.10 [^1^]. This asymmetry—larger penalties than rewards—causes the memory store to self-correct over time. Facts that are repeatedly contradicted by newer information see their trust scores decay toward zero, effectively suppressing them from future retrieval without requiring explicit deletion.

The `contradict` action extends this mechanism by actively detecting conflicting facts. When a new fact is added, the system checks for contradictions with existing stored facts and flags them for resolution [^35^]. This is a significant departure from the accumulation model used by most providers, where conflicting memories simply coexist and the retrieval system must resolve them at query time. Holographic pushes contradiction detection to ingestion time, which is cheaper because it happens once per write rather than on every read.

#### 3.3.5 Zero-External-Dependency Design

Holographic's architectural constraint of zero external dependencies makes it unique among the eight providers. Storage is local SQLite. Retrieval uses SQLite's built-in FTS5 (Full-Text Search version 5) module for keyword search combined with HRR algebra for compositional queries [^1^]. No API keys are required. No cloud accounts. No network calls. The entire system is operational in seconds after running `hermes memory setup` and selecting "holographic" [^4^].

This constraint is simultaneously Holographic's greatest strength and its defining limitation. It is the optimal choice for air-gapped environments, single-user local workflows, and fast experimental iteration. However, it does not perform LLM-based fact extraction, meaning it stores and retrieves conversational content without transforming it into structured knowledge. It does not build entity relationships, perform cross-memory synthesis, or maintain temporal indices. For agents where the knowledge extraction overhead of Schools 1 and 2 is acceptable, Holographic's retrieval will be less semantically precise. For agents where minimal latency and zero dependency count are paramount, no other provider matches its operational simplicity.

**Table 3.3: Tiered Context Engineering School — Feature and Benchmark Comparison**

| Dimension | OpenViking | Holographic |
|:---|:---|:---|
| **Core abstraction** | Hierarchical virtual filesystem (`viking://` URI) | HRR algebraic compression on SQLite |
| **Context economy mechanism** | L0/L1/L2 tiered loading (~100 tokens → ~2K → full) | HRR vector superposition + FTS5 keyword search |
| **Token reduction claim** | 80–90% vs. flat loading [^61^] | Sub-millisecond retrieval via algebraic operations |
| **Retrieval method** | Directory recursive retrieval: vector similarity → directory search → recursive drill-down | Algebraic: `probe` (entity), `reason` (compositional AND), `search` (FTS5) |
| **Observability** | Full trajectory logging for every retrieval path | Local SQLite; directly queryable with standard tools |
| **Trust mechanism** | None | Asymmetric trust scoring (+0.05 helpful / -0.10 unhelpful) |
| **Contradiction handling** | Not explicit | Automated `contradict` action at ingestion time |
| **Extraction categories** | Six: profile, preferences, entities, events, cases, patterns | None (stores content without LLM extraction) |
| **External dependencies** | Docker + LLM for extraction | Zero (SQLite only; NumPy optional for HRR) |
| **License** | Apache 2.0 | MIT |
| **Key differentiator** | Tiered loading with logged trajectories | Trust scoring + compositional algebra + zero dependencies |

Table 3.3 presents two providers solving the same token-efficiency problem through incommensurable approaches. OpenViking reduces token consumption by loading less content; Holographic reduces retrieval latency by compressing content algebraically. OpenViking requires an LLM for automatic tiering at write time, producing structured summaries that improve with model capability. Holographic is entirely deterministic, requiring no LLM calls for storage or retrieval, but cannot generate the structured abstractions that make OpenViking's L0/L1 layers useful. The architect designing a next-generation memory system should consider OpenViking's tiered loading for the context injection path and Holographic's trust scoring for the memory quality path—these two ideas compose naturally without requiring either's full architecture.

---

### 3.4 School 4: Semantic Memory Graphs (Supermemory, RetainDB)

School 4 elevates relationships between memories to first-class architectural status. Rather than treating each memory as an isolated fact to be retrieved by similarity, these providers encode how memories relate to one another—whether one updates another, extends it, or derives from it—and leverage those relationships during retrieval. The graph is not merely a storage format; it is an active participant in the retrieval process.

#### 3.4.1 Supermemory's Document vs. Memory Distinction

Supermemory distinguishes between two fundamentally different entities: **documents** and **memories** [^95^]. Documents are raw input—conversation transcripts, PDFs, web pages, emails. Memories are intelligent knowledge units that Supermemory creates by extracting atomic facts from documents. A single document may yield multiple memories; a single memory may draw from multiple documents. This separation is architecturally significant because it allows the system to update a memory without reprocessing its source documents, and to remove a document from storage without losing the knowledge it contributed.

The distinction also enables a cleaner ingestion pipeline. Connectors for Google Drive, Notion, OneDrive, Gmail, and GitHub feed documents into the system; the extraction layer converts those documents into memories independently of the storage layer [^105^]. This means the connector infrastructure and the memory infrastructure can evolve on separate timelines.

#### 3.4.2 Memory Relationships: Update, Extend, and Derive

Supermemory encodes three relationship types between memories, each with distinct semantic implications [^95^]. An **Update** relationship indicates that a new memory supersedes an older one—the user's preference changed, their role changed, their address changed. The older memory is not deleted; it is marked as superseded, preserving a historical trace while ensuring retrieval returns the current value. An **Extend** relationship indicates that a new memory enriches an existing one without contradicting it—additional details about a project, supplementary preferences, expanded context. A **Derive** relationship indicates that a memory was inferred from a pattern across other memories rather than extracted directly from a document [^95^].

These three relationship types enable the retrieval system to perform inference that pure semantic search cannot. When asked "What does the user think about remote work?" a vector system retrieves the most semantically similar stored statements. Supermemory's graph system can trace Update relationships to find the *current* belief even if earlier statements were more voluminous, follow Extend relationships to gather comprehensive context, and traverse Derive relationships to surface inferences that the user never explicitly stated. On LongMemEval, Supermemory scores 81.6% with GPT-4o, placing it mid-tier among tested providers but with a retrieval architecture that is qualitatively different from vector-only approaches [^96^].

#### 3.4.3 Context Fencing: Preventing Recursive Memory Pollution

Supermemory implements a mechanism called **context fencing** that strips recalled memories from captured conversation turns [^96^]. Without this protection, an agent that recalls a memory during a conversation might then store that conversation turn back into memory, now containing the recalled memory as part of the new content. On subsequent retrievals, the same memory appears multiple times—once in its original form and once embedded in a later conversation—creating a feedback loop that progressively pollutes the memory store. Context fencing detects when a recalled memory is present in an outgoing agent response and removes it before the turn is stored, breaking the recursion. This is a subtle but critical design element that most memory systems lack, and it becomes increasingly important as sessions grow longer and recall frequency increases.

#### 3.4.4 Session-End Graph Ingest: Building Knowledge from Conversations

Supermemory performs its richest knowledge graph construction at session end, when it ingests the complete conversation and extracts facts, entities, and relationships in a single batch pass [^96^]. This deferred extraction strategy, shared with several other providers, has two advantages: it captures conversational context that turn-by-turn extraction might miss (because the session-end pass sees the full arc of the conversation), and it amortizes the LLM extraction cost across the entire session rather than incurring it on every turn. The trade-off is that memories are not available for retrieval within the session that generated them; they become available only in subsequent sessions.

Supermemory also supports **multi-container mode**, allowing read and write operations across named containers [^96^]. Containers function as isolation boundaries—one per user, project, or workspace—preventing retrieval from crossing contexts inappropriately. A query scoped to the "engineering" container will not retrieve memories from the "marketing" container even if they are semantically similar.

#### 3.4.5 RetainDB's Full Chronological Retrieval: Complete Memory Timeline

RetainDB, the second provider in this school, takes an almost opposite approach to retrieval architecture. Where Supermemory traverses a semantic graph, RetainDB provides **full chronological retrieval**—a complete timeline of all stored memories in the order they were recorded [^96^]. The answering model sees the full memory history rather than a filtered subset selected by semantic similarity. RetainDB's philosophy is explicitly "no lossy retrieval": it guarantees coverage by giving the model all available context and letting the model's own attention mechanism determine what is relevant.

This approach carries both costs and benefits. The cost is increased token consumption: delivering a complete memory timeline to the context window requires more tokens than delivering a filtered set of top-k results. The benefit is the elimination of retrieval failures caused by embedding mismatch—cases where the query's embedding does not align with the memory's embedding despite semantic relevance. RetainDB claims a **0% hallucination rate** (defined as the memory system inventing facts not present in the conversation history) and reports State-of-the-Art (SOTA) performance on preference recall at **88%** accuracy [^96^]. These claims reflect the conservative nature of full-timeline retrieval: by never filtering, the system never incorrectly excludes a relevant memory.

#### 3.4.6 Delta Compression and Hybrid Search

RetainDB complements its chronological retrieval with two technical mechanisms: **delta compression** and **hybrid search**. Delta compression stores only the differences between successive memory versions rather than full copies, reducing storage overhead for frequently updated facts [^96^]. The hybrid search layer combines vector similarity, BM25 keyword matching, and a reranking stage, giving the system three independent signals for relevance ranking [^96^]. This three-signal approach produces more robust retrieval than any single method, particularly for queries that combine specific terminology with conceptual breadth.

**Table 3.4: Semantic Memory Graphs School — Feature and Benchmark Comparison**

| Dimension | Supermemory | RetainDB |
|:---|:---|:---|
| **Core abstraction** | Fact-based semantic graph with relationship edges | Full chronological memory timeline |
| **Memory vs. source** | Explicit document/memory separation | Turn-by-turn extraction with 3-turn context |
| **Relationship types** | Three: Update, Extend, Derive [^95^] | None (chronological ordering only) |
| **Retrieval philosophy** | Graph traversal with relationship-aware inference | "No lossy retrieval" — complete timeline delivery |
| **Context protection** | Context fencing strips recalled memories before storage | Delta compression for memory versioning |
| **Search architecture** | Graph-traversal + semantic | Hybrid: vector + BM25 + reranking [^96^] |
| **LongMemEval** | 81.6% (GPT-4o) [^96^] | — |
| **Preference recall** | — | 88% (SOTA claimed) [^96^] |
| **Hallucination rate** | — | 0% claimed [^96^] |
| **Ingestion timing** | Session-end batch extraction | Per-turn atomic extraction |
| **Multi-context isolation** | Multi-container mode (named containers) | Scoped by user profile |
| **Key differentiator** | Relationship-typed graph with context fencing | Full chronological delivery + delta compression |
| **Cost** | Free tier (1M tokens/10K searches), $19/mo Pro | $20/mo (no free tier) |

The comparison in Table 3.4 illuminates a deep architectural tension in the memory-graph space. Supermemory optimizes for the quality of individual memories and the relationships between them; its graph enables inferential retrieval that surfaces knowledge no single memory contains verbatim. RetainDB optimizes for coverage and fidelity; by delivering the complete memory timeline, it eliminates the risk of retrieval filtering errors at the cost of increased token consumption. These approaches are not mutually exclusive—a system could maintain a Supermemory-style relationship graph for inferential queries while retaining the ability to fall back to full chronological delivery when coverage is more important than concision. Context fencing, Supermemory's most transferable innovation, should be considered a mandatory component of any memory system where the agent recalls memories during conversation and those conversations are themselves stored.

---

### Cross-School Implications for Memory System Design

The four schools present a design space rather than a set of mutually exclusive choices. A next-generation memory system can compose techniques from multiple schools without adopting any single school's full architecture. The most defensible composition, informed by the benchmark data and architectural analysis above, would include: (1) structured extraction from School 1, either through Hindsight's entity-aware pipeline or ByteRover's curated Markdown tree; (2) tiered context loading from School 3, modeled on OpenViking's L0/L1/L2 decomposition, to constrain token consumption; (3) trust scoring from School 3, modeled on Holographic's asymmetric feedback mechanism, to prevent memory pollution over time; (4) relationship encoding from School 4, modeled on Supermemory's Update/Extend/Derive types, to enable inferential retrieval; and (5) context fencing from School 4 to break recursive memory contamination loops. Honcho's dialectic reasoning layer from School 2 represents an optional enhancement for systems where user-model depth justifies the additional LLM inference cost; for most agent applications, Mem0's extraction-based approach provides sufficient user modeling at lower operational overhead.

The benchmark data supports this compositional strategy. No single provider leads on all metrics. Hindsight excels at cross-session synthesis (91.4–94.6% LongMemEval), ByteRover at temporal precision (94.4% LoCoMo temporal), and RetainDB at preference recall (88%). A system that combines Hindsight's TEMPR retrieval with ByteRover's pre-compression extraction, OpenViking's tiered loading, Holographic's trust scoring, and Supermemory's relationship graph would inherit the strengths of each while mitigating the weaknesses. The following chapter examines how closely Super-Memory-TS's current architecture aligns with this composite ideal, and identifies the specific gaps that must be closed to achieve it.

---

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

---

## 5. Opencode Compatibility Assessment

For builders evaluating whether to invest in the Hermes memory-provider ecosystem or the OpenCode toolchain, the first question is deceptively simple: will the memory system I choose work with the agent framework I prefer? The answer hinges on a single architectural variable — the Model Context Protocol (MCP). Hermes and OpenCode speak fundamentally different languages when it comes to memory integration. This chapter maps the protocol divide, catalogs every viable bridge path, and positions Super-Memory-TS within that landscape.

### 5.1 The Protocol Divide

#### 5.1.1 Hermes MemoryProvider ABC: Python-Only and Plugin-Based

Hermes, developed by Nous Research, defines memory integration through an abstract base class (ABC) called `MemoryProvider`, residing in `agent/memory_provider.py` [^62^]. A provider is a Python package that subclasses this ABC, implements lifecycle methods (`initialize`, `sync_turn`, `prefetch`), and registers itself via a `register()` entry point in `plugins/memory/<name>/__init__.py` [^70^]. The directory also contains a `plugin.yaml` metadata file and a README [^62^]. Discovery is hierarchical: Hermes scans bundled plugins in `<repo>/plugins/memory/`, user plugins at `~/.hermes/plugins/memory/`, project-specific plugins at `.hermes/plugins/memory/`, and pip-installed packages exposing the `hermes_agent.plugins` entry point [^119^]. Critically, memory providers are *single-select* — only one can be active at a time, chosen via the `memory.provider` key in `config.yaml` [^119^]. The entire mechanism is Python-native: it imports modules, executes ABC methods, and relies on in-process hooks such as `pre_llm_call` and `post_llm_call` to trigger memory operations automatically [^119^]. There is no JSON-RPC layer, no stdio transport, and no language-agnostic interface.

#### 5.1.2 OpenCode Uses MCP: Language-Agnostic JSON-RPC

OpenCode, the open-source AI coding agent from opencode.ai, adopts the Model Context Protocol (MCP) as its exclusive extension mechanism for external memory. MCP is an open specification originally published by Anthropic in November 2024; it standardizes communication between AI hosts and external tool servers using JSON-RPC 2.0 messages over two transports: stdio for local servers and HTTP with Server-Sent Events (SSE) for remote ones [^121^] [^125^]. An MCP server exposes *tools* (functions the LLM can invoke), *resources* (read-only data addressable by URI), and *prompts* (reusable templates) [^121^]. OpenCode supports both local MCP servers (started as subprocesses via a `command` array) and remote MCP servers (connected via a `url` with optional OAuth authentication) configured in `opencode.jsonc` [^22^]. Because MCP is language-agnostic, a server can be written in TypeScript, Python, Go, or any other language — the only requirement is that it speaks JSON-RPC 2.0 over one of the supported transports [^127^].

#### 5.1.3 Core Incompatibility: Automatic Hooks vs Agent-Initiated Tool Calls

The incompatibility between Hermes memory providers and OpenCode is structural, not cosmetic. Hermes providers rely on *automatic hooks*: the `prefetch()` method runs before each LLM turn to inject relevant memories into context, and `sync_turn()` fires after each response to persist the conversation [^62^]. The agent runtime itself drives these calls — the LLM does not decide when to remember or recall. In OpenCode's MCP model, by contrast, memory operations are *agent-initiated tool calls*: the LLM sees available tools (e.g., `memory_save`, `memory_search`) and decides whether, when, and how to invoke them [^20^]. There is no automatic prefetch hook; context injection happens at session start or when the agent explicitly calls a retrieval tool. The Hermes ABC has no concept of tool schemas, and the MCP protocol has no concept of per-turn prefetch hooks. Translating between the two requires more than a thin adapter — it requires rethinking who controls the memory lifecycle.

#### 5.1.4 Single-Select vs Multi-Server Architecture

Beyond the hook-vs-tool distinction, the two architectures differ in their provider cardinality and operational scope. Table 5-1 contrasts these dimensions directly.

**Table 5-1. Hermes Memory Protocol vs OpenCode MCP Architecture**

| Dimension | Hermes MemoryProvider ABC | OpenCode MCP |
|---|---|---|
| **Integration mechanism** | Python ABC class + `plugin.yaml` metadata [^62^] | MCP server via stdio or HTTP/SSE [^22^] |
| **Language binding** | Python only | Language-agnostic (JSON-RPC 2.0) [^121^] |
| **Tool exposure** | Provider-specific methods auto-registered at load time [^119^] | MCP tools discovered dynamically at runtime [^125^] |
| **Context injection** | Automatic — `prefetch()` before every turn, `sync_turn()` after [^62^] | Agent-initiated via explicit tool calls [^20^] |
| **Activation** | `memory.provider` key in `config.yaml` (single-select) [^119^] | `mcp` object in `opencode.jsonc` (multi-server) [^22^] |
| **Memory extraction** | Automatic session-end extraction via `on_session_end` hook [^119^] | Agent-driven via `add_memory` or compaction-triggered save [^16^] |
| **Provider concurrency** | One active at a time | Multiple MCP servers simultaneously [^22^] |
| **Multi-modality** | Text only via ABC interface | Tools, resources, and prompts [^121^] |

The most consequential difference is the last row: concurrency. A Hermes agent can run exactly one external memory backend at a time — Honcho XOR Mem0 XOR Hindsight. An OpenCode agent can load Mem0 MCP, Honcho MCP, Hindsight MCP, and a local community server all at once, letting the LLM choose which tool to call for which memory operation [^22^]. This multi-server capability means OpenCode users can compose memory systems (semantic search from one provider, knowledge-graph recall from another) in ways Hermes's single-select architecture does not permit.

### 5.2 Bridge Paths: MCP Wrappers for Hermes Providers

Despite the protocol divide, every major Hermes-compatible memory provider now offers an MCP-accessible path into OpenCode. The bridge strategies differ — some are cloud-hosted endpoints, others are native plugins — but all translate the provider's storage backend into MCP tool calls that OpenCode can consume. Table 5-2 catalogues the five primary bridge paths.

**Table 5-2. MCP Bridge Paths from Hermes Providers to OpenCode**

| Provider | Bridge Type | MCP Endpoint / Package | Tool Count | Key OpenCode Feature |
|---|---|---|---|---|
| Mem0 | Cloud-hosted MCP | `https://mcp.mem0.ai/mcp` [^38^] | 11 (add, search, get, update, delete, list_entities, etc.) | One-command setup via `npx mcp-add` [^73^] |
| Honcho | Native OpenCode plugin | `@honcho-ai/opencode-honcho` [^107^] | 7 (search, chat, create_conclusion, get/set config, etc.) | Persistent memory across context wipes and session restarts [^31^] |
| Hindsight | OAuth-secured MCP | `https://api.hindsight.vectorize.io/mcp/{bank_id}/` [^117^] | 27+ (retain, recall, reflect, mental models, directives, tags) | Multi-bank isolation; RFC 9728 OAuth with PKCE [^117^] |
| Supermemory | Deep native plugin | `opencode-supermemory` [^19^] | 5 (add, search, profile, list, forget) | Preemptive compaction at 80% context; `<private>` tag redaction [^16^] |
| Super-Memory-TS | MCP-native (stdio + HTTP) | `@veedubin/super-memory-ts` [^91^] | 5 (query_memories, add_memory, search_project, index_project) | Powers Boomerang-v2; local-first Qdrant; HNSW <10ms [^91^] |

#### 5.2.1 Mem0 MCP: Cloud-Hosted, Zero-Setup

Mem0's bridge is the simplest: a cloud-hosted MCP server at `mcp.mem0.ai/mcp` that requires no local installation [^38^]. Users add it to OpenCode with a single `npx mcp-add` command, supplying their Mem0 API key [^73^]. The server exposes eleven tools — the most comprehensive of any bridge path — including `add_memory`, `search_memories`, `get_memory`, `update_memory`, `delete_memory`, `list_entities`, `list_events`, and `get_event_status` [^38^]. Because the server is remote over HTTP+SSE, it works across all MCP-compatible clients (Claude Code, Cursor, Windsurf, VS Code, and OpenCode) from a single configuration [^73^]. The trade-off is cloud dependency: all memory content is stored on Mem0's servers, which may conflict with data-sovereignty requirements.

#### 5.2.2 Honcho MCP: The `opencode-honcho` Plugin

Honcho takes a different bridge strategy — instead of a raw MCP endpoint, it ships a dedicated OpenCode plugin package (`@honcho-ai/opencode-honcho`) that wraps its MCP server with OpenCode-specific lifecycle integration [^107^]. Installation is via `bunx @honcho-ai/opencode-honcho install`, followed by a `/honcho:setup` command inside OpenCode [^31^]. The plugin surfaces seven tools, including `honcho_search`, `honcho_chat` (reasoning-backed context queries), and `honcho_create_conclusion` (durable memory writes) [^107^]. What distinguishes Honcho's bridge is its *persistence guarantee*: memories survive not just session restarts but also context wipes, because the plugin hooks into OpenCode's `experimental.session.compacting` event to anchor snapshots before compaction [^112^]. Workspace mapping and peer modeling are also included, with OpenCode projects automatically mapping to Honcho workspaces [^107^].

#### 5.2.3 Hindsight MCP: OAuth-Secured Knowledge Graph

Hindsight exposes its temporal + semantic + entity memory architecture as an MCP server at `api.hindsight.vectorize.io/mcp` [^117^]. Two connection modes are supported: *single-bank* (recommended), where the bank ID is encoded in the URL path and the agent is scoped to one memory space; and *multi-bank*, where a header selects the target bank and the agent can operate across multiple banks in a single session [^117^]. Authentication uses OAuth with PKCE per RFC 9728 — no API key management is required for supported clients [^117^]. The tool surface is the broadest of any provider: 27+ tools covering not just `retain`, `recall`, and `reflect`, but also mental-model management, directive handling, document search, and tag operations [^117^]. This richness makes Hindsight's MCP bridge the most feature-complete for users who need structured knowledge-graph memory with temporal reasoning.

#### 5.2.4 Supermemory: Deep Native Integration with Preemptive Compaction

Supermemory's OpenCode plugin (`opencode-supermemory`) offers the deepest native integration of any bridge path [^19^]. Rather than merely exposing MCP tools, the plugin hooks into OpenCode's session lifecycle to implement *preemptive compaction*: when context usage hits 80% (configurable), the plugin triggers summarization while the model still has "breathing room," injects project memories into the compaction prompt to preserve critical constraints, saves the summary back to Supermemory, and auto-continues the session [^16^]. On session start, it injects three context layers: user profile (cross-project preferences), project memories (all knowledge scoped to the current directory), and semantically relevant memories from a search query [^20^]. Memory types are categorized as `project-config`, `architecture`, `error-solution`, `preference`, `learned-pattern`, or `conversation` [^20^]. Privacy is handled via `<private>` tags that redact content before it leaves the machine [^16^].

#### 5.2.5 Super-Memory-TS: Fully MCP-Native, Powers Boomerang-v2

Super-Memory-TS occupies a unique position in this landscape because it was built *as* an MCP server from the start — no bridge, no wrapper, no translation layer [^91^]. It runs as a standalone Node.js process speaking MCP over stdio or HTTP, with five tools: `query_memories`, `add_memory`, `search_project`, and `index_project` [^91^]. The backend is Qdrant with an HNSW (Hierarchical Navigable Small World) index, delivering sub-10ms query latency. Embeddings use BGE-Large (1024-dimensional, GPU) with a MiniLM-L6-v2 CPU fallback, stored in fp16 precision at ~325MB per model instance [^91^]. Project isolation is enforced through payload filtering by `projectId`, and file indexing uses xxhash-wasm for incremental change detection [^91^].

What makes Super-Memory-TS architecturally significant for OpenCode builders is its role as the memory engine for *Boomerang-v2*, a multi-agent orchestration plugin for OpenCode with 14 specialized agents and an 8-step protocol [^91^]. In "built-in" mode, Super-Memory-TS's core modules are imported directly into Boomerang, eliminating MCP protocol overhead while maintaining the same tool interface [^91^]. This dual-mode architecture — external MCP server for generic clients, direct module import for OpenCode plugins — demonstrates how an MCP-native memory system can serve both ecosystems without protocol translation.

### 5.3 Strategic Implication

#### 5.3.1 Architectural Alignment: Super-Memory-TS vs Hermes Providers

The analysis yields a clear alignment map. Super-Memory-TS is natively compatible with OpenCode because both speak MCP. Hermes memory providers are not — they require a translation layer (an MCP wrapper, a dedicated plugin, or a hosted MCP endpoint) to function within OpenCode. The five bridge paths in Table 5-2 represent different strategies for building that layer: Mem0 and Hindsight use cloud-hosted MCP servers; Honcho and Supermemory ship native OpenCode plugins; Super-Memory-TS needs no layer at all.

For builders choosing between ecosystems, this distinction matters. If the target deployment is Hermes, the eight providers analyzed in Chapter 3 are first-class citizens with automatic prefetch, sync-turn hooks, and profile isolation. If the target is OpenCode, those same providers become available only through their MCP bridges — each with a subset of their Hermes-native capabilities. Honcho's plugin preserves persistence across compactions but loses the `sync_turn` hook; Mem0's cloud endpoint offers semantic search but no automatic session-end extraction; Hindsight's MCP server exposes its full knowledge-graph API but requires the agent to initiate every call.

#### 5.3.2 Best Path Forward: Extend Super-Memory-TS with Hermes-Inspired Features

The optimal strategy for builders working in the OpenCode ecosystem is to treat Super-Memory-TS as the architectural foundation and selectively port the most valuable Hermes-provider concepts into it. This approach preserves MCP compatibility — which guarantees interoperability with OpenCode, Claude Code, Cursor, and any other MCP host — while closing the feature gaps identified in Chapter 4. Specifically, the Hermes-inspired enhancements with the highest leverage are: (1) automatic LLM-based memory extraction at session boundaries (inspired by Mem0's server-side extraction and Honcho's conclusion creation), (2) a lightweight knowledge graph for memory relationships (inspired by Hindsight's entity-temporal architecture), and (3) preemptive compaction hooks (inspired by Supermemory's 80%-threshold strategy, already proven in OpenCode). Because Super-Memory-TS is already MCP-native, each addition extends its tool surface in a way that any MCP client can discover and use automatically — no translation layer, no single-select constraint, no Python dependency.

---

## 6. Recommendations: The Next-Generation Memory Architecture

The preceding five chapters traced a path from the context crisis in modern agent systems (Chapter 1), through the Hermes memory-provider ecosystem and its four architectural schools (Chapters 2 and 3), to a detailed gap analysis of Super-Memory-TS (Chapter 4) and a protocol-compatibility assessment (Chapter 5). The evidence converges on one conclusion: Super-Memory-TS's MCP-native, local-first foundation is architecturally sound, but its embedding-only design leaves twelve capability gaps preventing true context preservation. This chapter synthesizes all prior findings into a concrete, phased roadmap for evolving Super-Memory-TS into a next-generation memory platform — one that incorporates the best ideas from the Hermes provider ecosystem without sacrificing the MCP compatibility that is its strategic differentiator.

### 6.1 Best Ideas to Incorporate from Each Provider

The eight Hermes memory providers do not contribute equally to the proposed architecture. Each offers one or two innovations that transfer cleanly, alongside capabilities tightly coupled to their original context. Table 6.1 isolates the transferable innovation from each provider, maps it to the capability layer it addresses (from Chapter 1's six-layer framework), and cites the supporting evidence.

**Table 6.1: Transferable Innovations from the Hermes Provider Ecosystem**

| Provider | Transferable Innovation | Capability Layer | Supporting Evidence |
|:---|:---|:---|:---|
| **Hindsight** | Knowledge graph with entity relationships + cross-memory `reflect` synthesis | Synthesis + Temporal Awareness | 91.4–94.6% LongMemEval (highest of any provider) [^94^]; TEMPR four-strategy retrieval [^106^]; reflect operation unique to Hindsight [^102^] |
| **Honcho** | Dialectic reasoning for LLM-synthesized user insights + multi-peer profile separation | User Modeling | Cold/warm prompt selection [^110^]; three orthogonal config knobs [^103^]; multi-peer workspace isolation [^110^] |
| **OpenViking** | L0/L1/L2 tiered context loading for 80–90% token reduction | Contextual Recall | 80–90% token reduction [^61^]; six-category auto-extraction [^71^]; logged retrieval trajectories [^61^] |
| **Holographic** | Trust scoring with asymmetric feedback + automated contradiction detection | Trust & Decay | +0.05 helpful / −0.10 unhelpful [^1^]; automated `contradict` action [^35^]; zero dependencies [^4^] |
| **ByteRover** | Pre-compression extraction hooks + human-readable Markdown export format | Extraction + Temporal Awareness | 92.2% LoCoMo overall; 94.4% temporal reasoning [^99^]; five-operation curation engine [^93^] |
| **Mem0** | Automatic LLM-based fact extraction + multi-modal memory types (session/user/organizational) | Extraction | 51,400 GitHub stars [^96^]; circuit breaker pattern [^96^]; Apache 2.0 license [^96^] |
| **Supermemory (cloud)** | Context fencing against recursive pollution + memory relationship graph (Update/Extend/Derive) | Trust & Decay + Synthesis | Context fencing [^54^]; three relationship types [^95^]; 81.6% LongMemEval [^96^] |
| **RetainDB** | Full chronological retrieval option + hybrid search (Vector + BM25 + reranking) | Contextual Recall | 0% hallucination claimed; 88% SOTA preference recall [^96^]; delta compression [^96^] |

An innovation was included only if it addresses one of Chapter 1's six capability layers, is supported by quantitative data from Chapter 3, and can be implemented without adopting the source provider's full stack. Hindsight's `reflect` operation transfers cleanly because its synthesis logic can run as a periodic LLM pass over any vector store. Honcho's dialectic reasoning is partially transferable: the peer-card abstraction ports cleanly, but the multi-peer workspace depends on Honcho's Deriver engine and AGPL-3.0 license [^96^], making full adoption costly. The table reveals coverage patterns across the six layers. Extraction is addressed by Mem0 and ByteRover; synthesis by Hindsight and Supermemory; user modeling exclusively by Honcho; contextual recall by OpenViking and RetainDB; trust and decay by Holographic and Supermemory. Temporal awareness receives contributions from both Hindsight (TEMPR retrieval) and ByteRover (curation engine with temporal precision of 94.4% [^99^]). No single provider covers all six layers, which is why a compositional approach — selecting the best technique for each layer — outperforms adopting any single architecture wholesale.

#### 6.1.1 From Hindsight: Knowledge Graph with Entity Relationships and Cross-Memory Reflect Synthesis

Hindsight's highest scores (91.4–94.6% LongMemEval) derive from two separable architectural decisions [^94^]: a knowledge graph storing structured facts with named entities and typed relationships, and the `reflect` operation performing periodic cross-memory synthesis [^102^]. The graph enables entity-resolved retrieval where "Chris," "the CTO," and "he" map to the same canonical node. The reflect pass derives higher-level insights no individual retrieval captures. Both can be layered on Qdrant by adding graph metadata to memory entries and a scheduled synthesis job.

#### 6.1.2 From Honcho: Dialectic Reasoning and Multi-Peer Profile Separation

Honcho's Deriver engine synthesizes answers about the user from conversation history rather than retrieving stored facts [^103^], producing inferential insights that extraction-only systems cannot capture. Multi-peer architecture gives each agent persona an independent profile of the same user, preventing context collapse [^110^]. The transferable component is a dialectic LLM pass scoped per `projectId`, synthesizing a user-profile document from observed preferences, decisions, and patterns across stored memories.

#### 6.1.3 From OpenViking: L0/L1/L2 Tiered Context Loading

OpenViking organizes every memory into three tiers: L0 (abstract, ~100 tokens), L1 (overview, ~2,000 tokens), and L2 (full detail) [^71^]. An agent scanning 50 runbooks consumes ~5,000 tokens at L0 to identify three candidates, ~6,000 at L1 to select one, and full L2 only for the chosen document — 80–90% reduction versus flat retrieval [^61^]. This applies directly: each `MemoryEntry` stores three summary levels, and `query_memories` accepts a `tier` parameter controlling granularity.

#### 6.1.4 From Holographic: Trust Scoring and Automated Contradiction Detection

Holographic assigns every fact a trust score between 0.0 and 1.0 with asymmetric feedback (+0.05 correct, −0.10 incorrect) [^1^], causing the store to self-correct over time. The `contradict` action detects conflicts at ingestion time [^35^]. Implementation requires only a `trustScore` field and a `feedback_memory` tool — the lowest-complexity, highest-impact addition in the entire roadmap.

#### 6.1.5 From ByteRover: Pre-Compression Extraction and Human-Readable Markdown Export

ByteRover's pre-compression hook fires before context window compression, capturing in-flight knowledge that would otherwise be discarded [^97^]. Its Markdown knowledge tree in `.brv/context-tree/` provides full inspectability and Git version control [^93^]. The curation engine (ADD, UPDATE, UPSERT, MERGE, DELETE) produces 94.4% temporal reasoning on LoCoMo [^99^]. For Super-Memory-TS, the hook becomes an MCP tool at a configurable token-pressure threshold, and Markdown export converts Qdrant entries to a navigable file tree.

#### 6.1.6 From Mem0: Automatic LLM-Based Fact Extraction and Multi-Modal Memory Types

Mem0's server-side extraction achieves 60–80% coverage of storable facts versus 10–30% for manual extraction [^3^]. Its memory types (session, user, organizational) with different decay rates prevent transient details from polluting long-term storage [^96^]. The circuit breaker pattern (five failures → two-minute suspension) ensures graceful degradation [^96^]. The extraction layer is a pluggable LLM client running at session boundaries; the schema adds a `scope` field to `MemoryEntry`.

#### 6.1.7 From Supermemory: Context Fencing and Memory Relationship Graph

Supermemory's context fencing strips recalled memories from conversation turns before storage, preventing a feedback loop where recalled content is re-ingested and retrieved at increasing frequency [^54^]. Without fencing, every retrieval increases the probability of future retrieval for the same content, creating rich-get-richer distortion that progressively corrupts the memory store. The relationship types — Update (supersedes), Extend (enriches), Derive (infers) — enable inferential retrieval that pure vector search cannot perform [^95^]. Implementation adds a `relationships` array to `MemoryEntry` and a fencing filter to the `add_memory` path.

#### 6.1.8 From RetainDB: Full Chronological Retrieval and Hybrid Search

RetainDB delivers a complete memory timeline rather than a filtered subset [^96^]. Hybrid search combines vector similarity, BM25 keyword matching, and reranking [^96^]. The 0% hallucination claim reflects the conservative nature of full-timeline retrieval [^96^]. For Super-Memory-TS, BM25 is added via SQLite FTS5 alongside HNSW, and chronological retrieval returns memories ordered by `timestamp`.

### 6.2 Proposed vNext Architecture: "Super-Memory 3.0"

The preceding analysis yields a six-layer architecture preserving Super-Memory-TS's MCP-native, local-first foundation while adding extraction, synthesis, user modeling, trust, and tiered loading. The core principle is compositional enhancement: each layer is optional and independently enableable, maintaining backward compatibility with existing MCP clients while exposing new capabilities to those that choose them.

#### 6.2.1 Core Principle: MCP-Native Local-First Foundation with Optional Enhancement Layers

Chapter 5 established that MCP compatibility is Super-Memory-TS's strategic differentiator. Hermes providers require Python-specific ABC implementations and automatic hooks incompatible with the language-agnostic MCP protocol [^62^] [^121^]. The vNext architecture maintains MCP as the sole interface and implements all capabilities as additional MCP tools — never as automatic background operations. Every new capability is explicitly controllable via agent-initiated tool calls, preserving compatibility with all MCP hosts (OpenCode, Claude Code, Cursor, Windsurf) while making the system predictable and debuggable.

The local-first constraint also remains. All vector data stays in local Qdrant. LLM-based operations can use local models (via `llama.cpp`) or external APIs, but storage never leaves the host machine, addressing data-sovereignty concerns that disqualify cloud-only providers for teams with confidentiality requirements.

#### 6.2.2 Layer 1 — Keep: Qdrant HNSW, Project Indexing, Tiered Search, MCP Protocol, Project Isolation

The foundation layer remains unchanged from v2.6.5. Qdrant's HNSW index provides sub-10ms query latency [^1^]. Dual-model embeddings (BGE-Large 1024-dim, MiniLM-L6-v2 384-dim) support four search strategies with RRF fusion [^1^]. Project isolation via `projectId` payload filtering scales without collection-management overhead [^1^]. The five existing MCP tools continue functioning exactly as before.

#### 6.2.3 Layer 2 — Add Automatic Extraction: Session-End LLM Pass Extracting Facts, Decisions, and Patterns

Layer 2 addresses the most impactful gap: the absence of automatic extraction. Inspired by Mem0 and ByteRover, this layer adds `extract_memories` (accepts a transcript, returns structured candidates using a lightweight local LLM) and `review_extractions` (lets the agent approve, edit, or reject candidates). The schema adds `extractionSource` (session, pre-compression, manual), `memoryType` (fact, preference, decision, pattern), and `confidence` (0.0–1.0). The `memoryType` field maps to OpenViking's six-category classification [^71^], adapted to Super-Memory-TS's domain.

#### 6.2.4 Layer 3 — Add Knowledge Graph: Lightweight Entity Extraction with Update/Extend/Derive Relationships

Layer 3 adds graph metadata without a separate graph database. When extracting, the tool identifies named entities and proposes relationships — Update (supersedes), Extend (enriches), Derive (infers) from Supermemory's taxonomy [^95^] — stored as JSON in a `relationships` array on `MemoryEntry`. The `query_memories` tool gains `traverse_relationships` to follow edges beyond direct similarity. A `reflect` tool (inspired by Hindsight) performs periodic synthesis, storing results as entries with `memoryType: pattern` [^102^].

#### 6.2.5 Layer 4 — Add User Modeling: Dialectic Reasoning Building Persistent User Profile

Layer 4 implements Honcho-inspired modeling via `build_user_profile`, reading memories with `memoryType` in {preference, pattern, decision} and synthesizing a profile containing communication style, technical preferences, workflows, and goals. Each `projectId` maintains an independent profile, preventing context collapse from conflating observations across distinct domains [^110^].

#### 6.2.6 Layer 5 — Add Trust and Temporal: Trust Scoring, Contradiction Detection, Memory Decay and Consolidation

Layer 5 implements Holographic's trust scoring via `feedback_memory` (+0.05/−0.10) [^1^], with `query_memories` weighting results by `trustScore * recency_boost`. The `detect_contradictions` tool scans new memories against existing entries [^35^]. Memory decay is `run_consolidation`, compressing low-trust, aged memories into summaries. Decay parameters are configurable via MCP configuration.

#### 6.2.7 Layer 6 — Add Tiered Loading: L0/L1/L2 Abstraction for Token-Efficient Context Injection

Layer 6 implements OpenViking-inspired tiered loading. The `extract_memories` tool generates three summary levels: L0 (~20 tokens), L1 (~200 tokens), L2 (full text), stored in `abstract`, `overview`, and `detail` fields. The `query_memories` tool accepts a `tier` parameter. In `adaptive` mode, it returns L0 summaries first, expands to L1 for top-K results, and provides L2 only via a new `get_memory_detail` tool — the "skim before reading" pattern producing 80–90% token reduction [^61^].

#### 6.2.8 Architecture Summary: Six Layers with Data Flow

Table 6.2 consolidates the six-layer architecture with tool inventory and operational impact for each layer.

**Table 6.2: Super-Memory 3.0 Six-Layer Architecture**

| Layer | Name | Source of Inspiration | New/Modified Tools | Operational Impact |
|:---|:---|:---|:---|:---|
| Layer 1 | Foundation (Keep) | Super-Memory-TS v2.6.5 | `query_memories`, `add_memory`, `search_project`, `index_project`, `get_file_contents` | Sub-10ms queries; project isolation; dual-model search |
| Layer 2 | Automatic Extraction | Mem0 + ByteRover | `extract_memories`, `review_extractions` | 60–80% fact coverage vs. 10–30% manual [^3^] |
| Layer 3 | Knowledge Graph | Hindsight + Supermemory | `reflect`; `traverse_relationships` parameter | Cross-memory synthesis; relationship traversal [^95^] |
| Layer 4 | User Modeling | Honcho | `build_user_profile` | Per-project dialectic user profile [^103^] |
| Layer 5 | Trust & Temporal | Holographic + RetainDB | `feedback_memory`, `detect_contradictions`, `run_consolidation` | Asymmetric trust scoring; automated decay [^1^] [^35^] |
| Layer 6 | Tiered Loading | OpenViking | `get_memory_detail`; `tier` parameter | 80–90% token reduction in adaptive mode [^61^] |

The data flow follows a consistent pattern across both write and read paths. On the write path, conversation turns pass through Layer 2 (extraction) to generate structured candidates, then Layer 3 (graph relationships) to identify entities and connections, then Layer 5 (trust and contradiction check) before persistence. On the read path, queries enter Layer 6 (tier selection) to determine granularity, Layer 1 (retrieval) for the initial result set, Layer 5 (trust ranking) to reorder by quality, optionally Layer 3 (relationship traversal) to enrich results with connected memories, and finally Layer 4 (profile injection) to contextualize results against the user's known preferences. Every layer is optional: a client using only Layer 1 gets the current Super-Memory-TS behavior; a client enabling all six layers gets a full context-preservation pipeline with agent-initiated control at every step. This compositional design means existing MCP clients require no changes to continue operating — new capabilities are exposed as new tools that clients opt into by calling them.

### 6.3 Implementation Priority Matrix

The six-layer architecture represents a multi-year trajectory. This section proposes a four-phase plan based on **impact** (gap closure from Chapter 4, scored 1–5) and **complexity** (engineering effort including dependencies, LLM integration, and schema migrations, scored 1–5).

#### 6.3.1 Phase 1 (MVP): Automatic Extraction and Trust Scoring

Phase 1 targets the highest impact-to-complexity ratio. Automatic extraction (Layer 2) transforms Super-Memory-TS from passive storage into active memory curation, addressing gap #1 from Chapter 4. Trust scoring (Layer 5) requires only a schema field and a new tool yet enables quality ranking, addressing gap #4.

The implementation order is: (1) add `trustScore` and `memoryType` fields to `MemoryEntry`; (2) implement `feedback_memory`; (3) add a pluggable LLM client (local `llama.cpp`, remote API, passthrough); (4) implement `extract_memories` and `review_extractions`. The LLM client is the binding architectural decision: it must support local, remote, and passthrough modes to maintain deployability across environments from air-gapped workstations to cloud-connected machines. This abstraction is what keeps the architecture MCP-native while enabling LLM-based reasoning — the server exposes an LLM interface, but the implementation is swappable based on deployment constraints.

#### 6.3.2 Phase 2: Knowledge Graph and Contradiction Detection

Phase 2 builds on Phase 1's extraction infrastructure. The `relationships` array and `reflect` tool (Layer 3) require the `memoryType` field and extraction pipeline. Contradiction detection (Layer 5) requires relationship metadata and an LLM comparison pass — medium complexity but high differentiation, as no other MCP-native memory system offers automated contradiction detection.

The implementation order is: (1) add `relationships` array; (2) extend `extract_memories` to propose relationships; (3) implement `reflect` as scheduled or on-demand synthesis; (4) implement `detect_contradictions` using the Phase 1 LLM client. Graph traversal should be conservative: one hop by default, with configurable depth.

#### 6.3.3 Phase 3: User Modeling and Tiered Loading

Phase 3 addresses the highest-complexity, highest-differentiation features. User modeling (Layer 4) requires a mature extraction pipeline and a stable dialectic prompt producing consistent profiles. Tiered loading (Layer 6) triples LLM inference cost per extraction, requiring batching, caching, and quantization tuning.

The implementation order is: (1) implement L0/L1/L2 summary generation in `extract_memories`; (2) add `tier` parameter and `get_memory_detail`; (3) implement `build_user_profile` with per-project isolation; (4) add `adaptive` tier mode. Phase 3 is the transition from vector memory server to full semantic memory platform.

#### 6.3.4 Phase 4: Memory Decay, Consolidation, and Human-Readable Export

Phase 4 focuses on sustainability and polish. The `run_consolidation` tool (Layer 5) prevents unbounded growth in long-running deployments — a critical concern once Phases 1–3 have populated the store with automatically extracted memories. The Markdown export format (inspired by ByteRover) provides a human-readable backup and audit trail, converting the full memory store or a project-scoped subset into a navigable file tree [^93^]. Context fencing (from Supermemory) closes the recursive-pollution gap by stripping recalled memories from stored conversation turns [^54^]. Full chronological retrieval, drawn from RetainDB, offers a conservative fallback when semantic retrieval fails, returning memories ordered by `timestamp` rather than by similarity.

The implementation order is: (1) consolidation job with configurable thresholds; (2) `export_memories` with Markdown and JSON formats; (3) context fencing filter in `add_memory`; (4) chronological retrieval mode in `query_memories`. Phase 4 completes the transition from MVP to production-ready platform.

#### 6.3.5 Priority Matrix: Impact versus Complexity for All Proposed Features

Table 6.3 maps each feature onto an impact-complexity grid. Impact is scored 1–5 (1 = minor convenience, 5 = critical Chapter 4 gap). Complexity is scored 1–5 (1 = schema addition, 5 = new database plus LLM pipeline). The ratio (impact / complexity) determines scheduling priority.

**Table 6.3: Implementation Priority Matrix — Impact vs. Complexity**

| Feature | Source Provider | Impact | Complexity | Ratio | Phase | Gap |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Trust scoring (`feedback_memory`) | Holographic [^1^] | 4 | 1 | 4.00 | Phase 1 | #4 |
| Typed memory categories | OpenViking [^71^] | 3 | 1 | 3.00 | Phase 1 | #12 |
| Pre-compression extraction | ByteRover [^97^] | 4 | 2 | 2.00 | Phase 1 | #10 |
| Automatic LLM extraction | Mem0 [^96^] | 5 | 3 | 1.67 | Phase 1 | #1 |
| Chronological retrieval | RetainDB [^96^] | 3 | 2 | 1.50 | Phase 4 | — |
| Memory relationship graph | Supermemory [^95^] | 4 | 3 | 1.33 | Phase 2 | #2 |
| Contradiction detection | Holographic [^35^] | 4 | 3 | 1.33 | Phase 2 | #5 |
| Context fencing | Supermemory [^54^] | 4 | 3 | 1.33 | Phase 4 | #9 |
| Cross-memory synthesis | Hindsight [^102^] | 5 | 4 | 1.25 | Phase 2 | #8 |
| L0/L1/L2 tiered loading | OpenViking [^61^] | 5 | 4 | 1.25 | Phase 3 | #6 |
| Dialectic user modeling | Honcho [^103^] | 5 | 4 | 1.25 | Phase 3 | #3 |
| Multi-peer profile separation | Honcho [^110^] | 3 | 3 | 1.00 | Phase 3 | #7 |
| Memory decay/consolidation | Holographic + RetainDB [^96^] | 3 | 3 | 1.00 | Phase 4 | #11 |
| Human-readable Markdown export | ByteRover [^93^] | 2 | 2 | 1.00 | Phase 4 | — |

The ratio column justifies the phase sequencing. Trust scoring (4.00) and typed categories (3.00) deliver high impact at minimal complexity and unlock subsequent phases: trust scoring provides Layer 5's ranking signal, typed categories provide Layer 2's classification signal. Automatic extraction (1.67) is the Phase 1 centerpiece — without it, the graph, modeling, and tiered loading layers have no material to operate on. Pre-compression hooks (2.00) share Phase 1 because they use the same LLM client infrastructure.

Phase 2 features cluster at 1.25–1.33, building directly on Phase 1. The `reflect` tool is scheduled here because it operates on relationship metadata — delaying it would leave the graph layer without synthesis. Phase 3 contains the highest-stakes features: tiered loading triples LLM inference cost, requiring batching and caching, while user modeling requires a stable dialectic prompt for consistent profiles across diverse conversation types. Phase 4 addresses sustainability: consolidation prevents unbounded growth, export provides auditability, fencing closes the recursive-pollution gap [^54^], and chronological retrieval offers a conservative fallback when semantic search fails to surface relevant memories.

The total scope spans four phases of approximately 4–6 weeks each for a solo developer, or 2–3 weeks each for a two-person team. The critical-path dependency is the LLM client abstraction in Phase 1 — every subsequent phase depends on it. Its design, supporting local, remote, and passthrough modes, determines deployability across environments from air-gapped workstations to cloud-connected developer machines. A builder following this roadmap achieves a functionally differentiated platform by Phase 2, an architecturally complete platform by Phase 3, and a production-hardened platform by Phase 4 — all while preserving the MCP compatibility and local-first principles that make Super-Memory-TS unique.

---

# Appendix: Vendor Documentation Links

| Vendor | Official Docs | GitHub |
|--------|--------------|--------|
| **Honcho** | https://docs.honcho.dev | https://github.com/plastic-labs/honcho |
| **OpenViking** | README-driven | https://github.com/volcengine/OpenViking |
| **Mem0** | https://docs.mem0.ai | https://github.com/mem0ai/mem0 |
| **Hindsight** | https://hindsight.vectorize.io | Part of Vectorize |
| **Holographic** | Built into Hermes docs | N/A (built-in) |
| **RetainDB** | https://retaindb.com | https://github.com/RetainDB |
| **ByteRover** | https://docs.byterover.dev | N/A |
| **Supermemory** | https://supermemory.ai/docs | https://github.com/supermemoryai |
| **Super-Memory-TS** | https://github.com/Veedubin/Super-Memory-TS/blob/main/README.md | https://github.com/Veedubin/Super-Memory-TS |
| **Hermes Agent** | https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers | https://github.com/NousResearch/hermes-agent |
| **OpenCode** | https://opencode.ai/docs | https://github.com/opencode |

---

*Report generated: May 15, 2026*
