# Dimension 4: Opencode Compatibility Analysis

## Opencode's Memory Ecosystem

OpenCode (https://opencode.ai) is an AI coding agent platform that fully supports MCP (Model Context Protocol) integration. It provides structured tool calling, message history handling, and model orchestration.

### MCP Server Support
OpenCode supports both local and remote MCP servers. Once added, MCP tools are automatically available to the LLM alongside built-in tools.

### Memory Plugins Available for OpenCode

| Plugin | Provider | Type | Description |
|--------|----------|------|-------------|
| **opencode-supermemory** | Supermemory | Plugin | Persistent memory across sessions using Supermemory cloud |
| **honcho-opencode** | Honcho | Plugin | Persistent memory surviving context wipes and session restarts |
| **mcp-memory-service** | Community | MCP Server | Open-source persistent memory with knowledge graph |
| **Mem0 MCP** | Mem0 | MCP Server | Cloud-hosted MCP server at mcp.mem0.ai |
| **Hindsight MCP** | Hindsight | MCP Server | OAuth-secured MCP at api.hindsight.vectorize.io/mcp |
| **Super-Memory-TS** | Veedubin | MCP Server | Local-first semantic memory with Qdrant |

### How Memory Works in OpenCode
1. **MCP tools exposed** to the LLM (e.g., `memory_save`, `memory_search`, `memory_update`)
2. **Agent decides when to call** — the LLM uses tools to save/recall memories
3. **Context injection** — on session start, relevant memories fetched and injected
4. **Smart compaction** — at 80% context capacity, sessions summarized and saved
5. **Privacy** — `<private>` tags redact content before storage

### Supermemory's Deep OpenCode Integration
- `/supermemory-init` command — deep research session exploring project
- Three things injected on session start: user profile, project memories, semantic search results
- Preemptive compaction at 80% context usage
- Memory scopes: `user` (cross-project) and `project` (current directory)
- Memory types: project-config, architecture, error-solution, preference, learned-pattern, conversation

## Hermes Memory Provider Protocol vs OpenCode

### The Core Incompatibility
**Hermes memory provider protocol is NOT directly compatible with OpenCode.** They are fundamentally different architectures:

| Aspect | Hermes Protocol | OpenCode/MCP |
|--------|----------------|--------------|
| **Integration mechanism** | Python ABC class + plugin.yaml | MCP (Model Context Protocol) |
| **Language** | Python only | Language-agnostic (JSON-RPC) |
| **Tool exposure** | Provider-specific tools auto-registered | MCP tools exposed via stdio/HTTP/SSE |
| **Context injection** | Automatic — prefetch + sync_turn hooks | Via MCP tool calls |
| **Activation** | `memory.provider` in config.yaml | MCP server config |
| **Memory extraction** | Automatic session-end extraction | Agent-initiated via tools |
| **Multi-provider** | Single-select (one at a time) | Multiple MCP servers simultaneously |

### Bridge Path: MCP Wrappers
Each Hermes provider that offers an MCP server CAN work with OpenCode:
- **Mem0 MCP**: https://mcp.mem0.ai/mcp — works with OpenCode
- **Honcho MCP**: honcho-ai/opencode-honcho package — works with OpenCode
- **Hindsight MCP**: api.hindsight.vectorize.io/mcp — OAuth-secured, works with OpenCode
- **Supermemory**: opencode-supermemory plugin — deep native integration

### Super-Memory-TS Compatibility
**Super-Memory-TS IS fully compatible with OpenCode** because it uses MCP:
- Runs as MCP server (stdio or HTTP)
- 5 standard MCP tools exposed
- Powers Boomerang-v2 (OpenCode plugin)
- No translation layer needed

## Key Insight for the User
The user's Super-Memory-TS project is architecturally aligned with OpenCode (both MCP-based). To incorporate ideas from Hermes providers, the user would need to either:
1. **Add MCP wrappers** around Hermes provider concepts
2. **Port Hermes provider features** into Super-Memory-TS directly
3. **Build a new memory system** that combines MCP compatibility with Hermes-style automatic extraction
