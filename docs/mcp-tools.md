# MCP tools

memini-ai exposes 52 MCP tools grouped by capability. All tools are available
over stdio (`memini-ai --stdio`) once your MCP client connects.

## Foundation memory

| Tool | Description |
|------|-------------|
| `add_memory` | Store a new memory with optional source-type and metadata. |
| `query_memories` | Semantic search with `tiered` / `vector_only` / `text_only` / `parallel` strategies. |
| `get_memory` | Fetch a single memory by ID. |
| `delete_memory` | Remove a memory entry by ID. |

## Project indexing

| Tool | Description |
|------|-------------|
| `index_project` | Trigger recursive indexing of a project directory. |
| `search_project` | Semantic search across indexed project files. |
| `get_file_contents` | Reconstruct a file from its indexed chunks. |
| `get_indexing_status` | Check progress of a background indexing job. |

## Trust engine

| Tool | Description |
|------|-------------|
| `get_trust_score` | Retrieve the trust score and level for a memory. |
| `adjust_trust` | Apply a feedback signal (`agent_used`, `agent_ignored`, `user_confirmed`, `user_corrected`). |
| `adjust_decay_rate` | Change the decay rate for a specific memory (sticky = low rate). |
| `get_decay_status` | View the decay engine status and fading memories. |
| `list_fading_memories` | List memories approaching the archive threshold. |
| `list_archived` | List memories already archived (below archive threshold). |

## Tiered loading

| Tool | Description |
|------|-------------|
| `get_tier0_summary` | High-trust L0 project summary (~100 tokens). |
| `get_tier1_summary` | L1 key-decisions summary (~2K tokens, trust >= 0.8). |

## Knowledge graph

| Tool | Description |
|------|-------------|
| `query_kg` | Execute a formal knowledge-graph query. |
| `extract_entities` | Extract named entities from a memory entry. |
| `search_entities` | Find entities by name. |
| `get_entity_graph` | Get all connections to/from an entity. |
| `get_inference_chain` | Find inference paths between two entities. |
| `get_graph_visualization` | Get a self-contained HTML page with a D3.js force-directed graph. |

## Memory relationships

| Tool | Description |
|------|-------------|
| `find_related_memories` | Find memories linked to a reference memory (`SUPERSEDES`, `PARTIAL_UPDATE`, etc.). |
| `get_relationship_summary` | Get a summary of all relationships for a memory. |
| `create_relationship` | Create a typed relationship between two memories. |
| `trigger_consolidation` | Manually merge similar memory pairs. |

## Dialectic

| Tool | Description |
|------|-------------|
| `find_contradictions` | Detect memory pairs that contradict each other. |
| `resolve_contradiction` | Generate an LLM-synthesized dialectic resolution for two memories. |
| `challenge_memory` | Submit a counter-argument challenge to a memory. |
| `get_dialectic_history` | View the argument history for a memory. |

## Thought chains

| Tool | Description |
|------|-------------|
| `start_thought_chain` | Create a new reasoning chain (optionally parented to another chain). |
| `add_thought` | Add a thought to a chain (auto-creates the chain if no `chain_id` is given). |
| `branch_thought` | Start a new branch from an existing thought. |
| `revise_thought` | Create a revision of an existing thought. |
| `get_thought_chain` | Retrieve a full chain with all thoughts organized by branch. |
| `get_related_chains` | Search for thought chains with reasoning similar to a query. |
| `pause_thought_chain` | Pause a chain. |
| `resume_thought_chain` | Resume a paused chain (returns the last thought). |
| `abandon_thought_chain` | Mark a reasoning path as incorrect. |

## Multi-peer

| Tool | Description |
|------|-------------|
| `list_peers` | List all known peers. |
| `add_peer` | Register a new peer with a role (`owner`, `collaborator`, `readonly`, `guest`). |
| `switch_peer_context` | Switch the active peer context for memory operations. |
| `share_memory` | Share a memory with another peer (`shared` or `inherited` permission). |
| `get_peer_memories` | Query another peer's accessible memories. |
| `get_shared_memories` | Get all memories shared with the current peer context. |

## User modeling

| Tool | Description |
|------|-------------|
| `get_user_profile` | Retrieve the learned user profile and style. |
| `update_user_profile` | Update the profile from the current conversation. |

## Extraction

| Tool | Description |
|------|-------------|
| `trigger_extraction` | Manually trigger memory extraction from a conversation. |
| `preconpress_extraction` | Capture context and extract memories before compaction. |

## Dual-model RRF

| Tool | Description |
|------|-------------|
| `elevate_memory_to_1024` | Promote a 384-dim memory into 1024-dim space (auto-mode only). |

## System and maintenance

| Tool | Description |
|------|-------------|
| `get_status` | Health check for all server components (reports `modelName`, `modelDimension`, `embeddingDimMismatch`). |
| `healthcheck` | End-to-end write + read round-trip probe with latency metrics. |
| `log_audit_event` | Manually log an audit event. |
| `get_audit_log` | Query the audit log with filters. |
| `get_security_summary` | Aggregated security metrics for the last N hours. |