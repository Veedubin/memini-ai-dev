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
