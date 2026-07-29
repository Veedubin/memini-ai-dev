# memini-ai-cloud: Thin-Client SaaS Architecture

**Status:** Design — NO implementation  
**Date:** 2026-07-29  
**Author:** boomerang-architect  
**Scope:** Design for `memini-ai-cloud` — a thin MCP proxy client that mirrors the 52-tool memini-ai surface and forwards JSON-RPC to a cloud backend running embedding + Postgres + background jobs. Full memini-ai (local) can also use the cloud as a third RRF fusion arm alongside embedded + team.

> **SUPERSEDES** the client-side portions of `docs/design/memini-saas-horizontal-scaling.md` (2026-07-28).  
> **KEEPS** its server-side tenancy/scaling model (tiered tenancy, API keys, K8s shapes, cost model, 11 required changes, phased rollout).  
> **PIVOTS** the client model: the existing SaaS doc assumed full memini-ai server pods behind a gateway. The user's model is simpler — the client is a thin MCP proxy, the cloud does all heavy lifting.

---

## Table of Contents

1. [Transport Decision Matrix](#1-transport-decision-matrix)
2. [memini-ai-cloud Thin Client](#2-memini-ai-cloud-thin-client)
3. [Config Schema](#3-config-schema)
4. [Cloud Server Side](#4-cloud-server-side)
5. [Fusion Semantics](#5-fusion-semantics)
6. [Phased Implementation Plan](#6-phased-implementation-plan)
7. [Risks & Open Questions](#7-risks--open-questions)

---

## 1. Transport Decision Matrix

### 1.1 Options

| Transport | Pros | Cons | MCP Client Compat? | K8s Ingress Friendly? | Session Resumption? | Binary Vector Efficiency? |
|-----------|------|------|--------------------|-----------------------|---------------------|---------------------------|
| **JSON-RPC over HTTPS (streamable-http)** | Already the memini-ai default transport (`server.py:3845-3847`). FastMCP native. Works with every MCP client. TLS via ingress. Simple to debug. | No server push. Request-response only. | ✅ Yes — every MCP client speaks this | ✅ Yes — standard HTTP ingress | ❌ No — each request is independent | ⚠️ Poor — vectors serialized as JSON float arrays (~1.5KB per 384-dim vector) |
| **Encrypted WebSocket (WSS)** | Long-lived sessions. Server push (notifications, streaming). Session resumption via session tokens. | Sticky sessions needed in K8s. More complex ingress (WebSocket upgrade). Not all MCP clients support it. | ⚠️ Partial — MCP spec supports WebSocket but most clients use stdio or streamable-http | ⚠️ Needs sticky sessions or session-affinity ingress | ✅ Yes — session token on reconnect | ⚠️ Same JSON serialization as HTTPS |
| **HTTP/2 Streams** | Multiplexed streams over single connection. Header compression. Server push via PUSH_PROMISE. | More complex client. Not all MCP clients support it. | ❌ No — no MCP client uses raw HTTP/2 streams | ✅ Yes — standard HTTP/2 ingress | ⚠️ Partial — stream IDs are ephemeral | ⚠️ Same JSON serialization |
| **gRPC** | Binary protobuf — best for vector data. Streaming RPCs. Strong typing. Built-in auth (TLS + token). | Not MCP-compatible. Would need a gRPC→MCP bridge or a custom protocol. Heavier client dep. | ❌ No — MCP is JSON-RPC, not gRPC | ✅ Yes — gRPC ingress via Envoy | ✅ Yes — gRPC streaming with deadlines | ✅ Best — protobuf `repeated float` is 4 bytes/float |

### 1.2 Recommendation

**v1 client→cloud transport: JSON-RPC over HTTPS (streamable-http) with TLS 1.3.**

**Justification:**
1. **Zero client changes to the MCP ecosystem.** Every MCP client (OpenCode, Claude Desktop, Continue, Cursor) already speaks `streamable-http`. The thin client is a drop-in replacement for local memini-ai — the user changes one URL and one API key.
2. **Already the memini-ai default.** `server.py:3845-3847` defaults to `transport="streamable-http"`. The cloud server runs the same FastMCP server behind an auth gateway.
3. **K8s ingress is trivial.** Standard HTTP ingress (Envoy/Kong/NGINX) with TLS termination. No sticky sessions, no WebSocket upgrades, no protocol negotiation.
4. **Vector serialization overhead is acceptable for v1.** A 384-dim vector is ~1.5KB in JSON. At 10 embeddings/sec, that's 15KB/sec — negligible. At 100 embeddings/sec, 150KB/sec — still fine. The embedding happens server-side anyway; vectors only cross the wire in query responses (ranked lists of memory IDs, not raw vectors).
5. **Session-count and rate-limit enforcement is trivial at the HTTP layer.** Standard `X-RateLimit-*` headers. Per-API-key counters in the gateway.

**v1 internal cloud services transport: gRPC (embedding service).**

**Justification:**
1. The embedding service is internal — no MCP client compatibility needed.
2. Binary protobuf for vector data is 4 bytes/float vs ~12 bytes/float in JSON (3× savings).
3. gRPC streaming for batch embedding (send N texts, receive N vectors in one stream).
4. Already recommended in the SaaS doc (§11.2, Q1): "gRPC is faster for binary vector data."
5. The SaaS doc's embedding service already assumes gRPC (§4.1 component diagram).

**Future (v2+):** Add WebSocket support for long-lived sessions with server push (real-time trust updates, decay notifications, streaming thought chains). The thin client can negotiate transport at connect time: try WSS, fall back to HTTPS.

### 1.3 Transport Negotiation (v2)

```
Client connects → GET /health → 200 OK
Client sends: POST /rpc with header X-Transport-Negotiate: wss,streamable-http
Server responds: X-Transport: streamable-http (or X-Transport: wss with Upgrade)
```

For v1, the client hardcodes `streamable-http` and the server always responds with `streamable-http`.

---

## 2. memini-ai-cloud Thin Client

### 2.1 Package Structure

```
memini-ai-cloud/                    # NEW npm package OR Python package
├── package.json / pyproject.toml
├── src/
│   ├── client.ts / client.py       # JSON-RPC over HTTPS client
│   ├── proxy.ts / proxy.py         # MCP tool surface mirror (52 tools)
│   ├── auth.ts / auth.py           # API key management, TLS pinning
│   ├── session.ts / session.py     # Session resumption, reconnect
│   ├── cache.ts / cache.py         # Optional local cache (tier0/tier1)
│   └── config.ts / config.py       # Minimal config (URL + key + transport)
├── tests/
└── README.md
```

**Language choice:** Python (same ecosystem as memini-ai-dev). The thin client is a FastMCP server that proxies to the cloud. Users run it as an MCP server in their OpenCode config:

```json
{
  "mcp": {
    "memini-ai-cloud": {
      "type": "local",
      "command": "uvx",
      "args": ["--from", "memini-ai-cloud", "memini-cloud"],
      "environment": {
        "MEMINI_CLOUD_URL": "https://cloud.memini.ai",
        "MEMINI_CLOUD_API_KEY": "YOUR_MEMINI_CLOUD_API_KEY"  # pragma: allowlist secret
      }
    }
  }
}
```

### 2.2 Tool Surface Mirroring

**Decision: Proxy, don't re-declare.**

The thin client does NOT re-declare all 52 MCP tools. Instead, it:

1. On startup, calls `tools/list` on the cloud server to discover the tool surface.
2. Registers a single catch-all handler that forwards any `tools/call` to the cloud.
3. Passes through the tool name, arguments, and returns the result unchanged.

**Why proxy over re-declare:**
- **Zero maintenance.** When memini-ai adds a 53rd tool, the thin client picks it up automatically.
- **No drift.** Re-declaring 52 tools means 52 opportunities for the signature to go stale.
- **Smaller package.** ~200 LOC vs ~2,000 LOC for re-declaration.
- **FastMCP supports this natively.** The thin client is a FastMCP server with one dynamic tool: `_proxy_tool`.

**Implementation sketch (Python):**

```python
# memini_cloud/proxy.py
import httpx
from fastmcp import FastMCP

class CloudProxy:
    def __init__(self, cloud_url: str, api_key: str):
        self._url = cloud_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        self._tool_names: list[str] = []

    async def discover_tools(self) -> list[str]:
        """Call tools/list on the cloud, cache the tool names."""
        resp = await self._client.post("/rpc", json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "tools/list",
        })
        result = resp.json()
        self._tool_names = [t["name"] for t in result["result"]["tools"]]
        return self._tool_names

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Forward a tools/call to the cloud."""
        resp = await self._client.post("/rpc", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        return resp.json()

# FastMCP server setup:
mcp = FastMCP("memini-ai-cloud")
proxy = CloudProxy(config.cloud_url, config.cloud_api_key)

@mcp.tool()
async def _proxy_tool(tool_name: str, arguments: dict) -> dict:
    """Proxy any memini-ai tool call to the cloud."""
    return await proxy.call_tool(tool_name, arguments)
```

**Alternative (v2):** If tool discovery latency is a concern, the thin client can cache the tool list locally and refresh on version mismatch. The cloud server includes a `X-Memini-Version` header in every response.

### 2.3 Auth

**API key format:** `mem_<tenant_id>_<random_32_hex>` (from SaaS doc §6.1).

The thin client sends the API key as an `Authorization: Bearer <key>` header on every request. The cloud gateway validates it, injects `X-Tenant-ID` and `X-Target-DSN` headers, and forwards to the MCP server pod.

**TLS pinning (optional, v2):** The thin client can pin the cloud's TLS certificate fingerprint to prevent MITM attacks. Configured via `MEMINI_CLOUD_TLS_PIN` (SHA-256 fingerprint).

### 2.4 Session Resumption & Reconnect

**v1: No session state.** Every request is independent. The cloud server is stateless (per the SaaS doc §4.1: "MCP Server Pod (stateless)"). If the connection drops, the next request creates a new HTTP connection. No session tokens, no reconnect logic.

**v2 (WebSocket):** Session tokens for long-lived connections. The thin client stores a session token and reconnects automatically on disconnect. Pending requests are queued and replayed.

### 2.5 Offline Behavior

**v1: Fail fast.** If the cloud is unreachable, the thin client returns an error immediately. No local queue, no offline mode. The user's agent sees the error and can retry or fall back to local memini-ai.

**Rationale:** The thin client is a proxy, not a local memory store. Offline queuing would require local embedding + local Postgres — which is what full memini-ai already does. If the user needs offline, they use full memini-ai with `MEMINI_CLOUD_FUSION_MODE=rrf` (cloud as a fusion arm, embedded as primary).

### 2.6 Local Caching

**v1: No caching.** Every `query_memories` call goes to the cloud. The thin client is a pure proxy.

**v2 (optional): Tier0/Tier1 cache.** The thin client can cache `get_tier0_summary` and `get_tier1_summary` responses locally (in-memory, TTL-based). This reduces latency for the most common read path (agent startup context injection). Configured via `MEMINI_CLOUD_CACHE_TIER0=true` and `MEMINI_CLOUD_CACHE_TTL=3600`.

**Why not cache query results:** Query results depend on the embedding vector, which changes with the query text. Caching would require embedding locally — which defeats the purpose of the thin client.

---

## 3. Config Schema

### 3.1 Thin Client Config (memini-ai-cloud)

The thin client needs minimal config — no DB, no model, no embedded dirs:

```python
# memini_cloud/config.py
class CloudClientConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMINI_CLOUD_")

    # Required
    cloud_url: str = Field(
        default="https://cloud.memini.ai",
        alias="URL",
        description="Cloud server URL (JSON-RPC over HTTPS endpoint).",
    )
    cloud_api_key: str = Field(
        default="",
        alias="API_KEY",
        description="API key in format mem_<tenant_id>_<random_32_hex>.",
    )

    # Transport
    cloud_transport: Literal["streamable-http"] = Field(
        default="streamable-http",
        alias="TRANSPORT",
        description="Transport protocol. v1: streamable-http only.",
    )

    # Timeouts
    cloud_timeout_connect: float = Field(
        default=10.0,
        alias="TIMEOUT_CONNECT",
        description="Connection timeout in seconds.",
    )
    cloud_timeout_request: float = Field(
        default=30.0,
        alias="TIMEOUT_REQUEST",
        description="Request timeout in seconds.",
    )

    # Retry
    cloud_max_retries: int = Field(
        default=3,
        alias="MAX_RETRIES",
        description="Max retries on transient failures.",
    )

    # TLS (v2)
    cloud_tls_pin: str | None = Field(
        default=None,
        alias="TLS_PIN",
        description="SHA-256 fingerprint of the cloud's TLS certificate (optional).",
    )

    # Cache (v2)
    cloud_cache_tier0: bool = Field(
        default=False,
        alias="CACHE_TIER0",
        description="Cache tier0 summaries locally.",
    )
    cloud_cache_ttl: int = Field(
        default=3600,
        alias="CACHE_TTL",
        description="Cache TTL in seconds.",
    )
```

**Environment variables for the thin client:**

```bash
MEMINI_CLOUD_URL=https://cloud.memini.ai
MEMINI_CLOUD_API_KEY=YOUR_MEMINI_CLOUD_API_KEY  # pragma: allowlist secret
# That's it. No DB, no model, no embedded dirs.
```

### 3.2 Full memini-ai Config (Cloud as Fusion Arm)

Full memini-ai (local) can use the cloud as a third RRF fusion arm. New fields in `MeminiConfig`:

```python
# In src/memini_ai/config.py, inside class MeminiConfig(BaseSettings):

# ── v1.x: Cloud SaaS backend (thin-client proxy OR fusion arm) ──
cloud_enabled: bool = Field(
    default=False,
    alias="MEMINI_CLOUD_ENABLED",
    description="Enable cloud SaaS backend as a fusion arm or standalone backend.",
)
cloud_url: str = Field(
    default="https://cloud.memini.ai",
    alias="MEMINI_CLOUD_URL",
    description="Cloud server URL (JSON-RPC over HTTPS endpoint).",
)
cloud_api_key: str = Field(
    default="",
    alias="MEMINI_CLOUD_API_KEY",
    description="API key in format mem_<tenant_id>_<random_32_hex>.",
)
cloud_transport: Literal["streamable-http"] = Field(
    default="streamable-http",
    alias="MEMINI_CLOUD_TRANSPORT",
    description="Transport protocol for cloud communication.",
)
cloud_fusion_mode: Literal["none", "rrf"] = Field(
    default="none",
    alias="MEMINI_CLOUD_FUSION_MODE",
    description=(
        "Fusion mode for cloud backend. 'none' = cloud is standalone "
        "(no local DB). 'rrf' = fuse cloud results with embedded + team."
    ),
)
cloud_timeout_connect: float = Field(
    default=10.0,
    alias="MEMINI_CLOUD_TIMEOUT_CONNECT",
)
cloud_timeout_request: float = Field(
    default=30.0,
    alias="MEMINI_CLOUD_TIMEOUT_REQUEST",
)
cloud_max_retries: int = Field(
    default=3,
    alias="MEMINI_CLOUD_MAX_RETRIES",
)
cloud_tls_pin: str | None = Field(
    default=None,
    alias="MEMINI_CLOUD_TLS_PIN",
)
```

### 3.3 Composition with Existing Config

The config composes with the existing backend selection. Here are the valid combinations:

| Scenario | `MEMINI_VECTOR_BACKEND` | `MEMINI_TEAM_DB_URL` | `MEMINI_CLOUD_ENABLED` | `MEMINI_FUSION_MODE` | `MEMINI_CLOUD_FUSION_MODE` | Behavior |
|----------|------------------------|---------------------|------------------------|---------------------|--------------------------|----------|
| **Thin client only** | N/A (separate package) | N/A | N/A | N/A | N/A | Pure proxy. No local DB. |
| **Cloud-only (full memini-ai)** | `pgembed` or `postgres-external` | (unset) | `true` | `none` | `rrf` | Local DB + cloud as secondary RRF arm. Writes go to local primary, async dual-write to cloud. |
| **Embedded + Team + Cloud** | `pgembed` | `postgresql://team/db` | `true` | `rrf` | `rrf` | Three-arm RRF: embedded (primary) + team (secondary) + cloud (tertiary). Writes go to embedded, async dual-write to team + cloud. |
| **Team + Cloud (no embedded)** | `postgres-external` | `postgresql://team/db` | `true` | `rrf` | `rrf` | Team (primary) + cloud (secondary). Writes go to team, async dual-write to cloud. |
| **Cloud as primary** | `pgembed` | (unset) | `true` | `none` | `none` (but `cloud_enabled=true`) | Cloud is the ONLY backend. Local DB is unused. This is the "full memini-ai but cloud-hosted" mode. |

**Decision: Cloud CAN be primary.** When `MEMINI_CLOUD_ENABLED=true` and `MEMINI_CLOUD_FUSION_MODE=none` and no team server is configured, the cloud is the sole backend. The local embedded DB is still started (for schema init) but all reads/writes go to the cloud. This allows a user to run full memini-ai locally but store all memories in the cloud — useful for multi-device sync.

**Decision: Cloud CANNOT be primary when embedded is also primary.** The `create_database()` factory always makes the local backend (embedded or external) the primary. Cloud is always a secondary/tertiary arm. This preserves the existing RRFDatabase semantics (primary is authoritative, secondary is best-effort).

### 3.4 `create_database()` Extension

```python
def create_database(config: MeminiConfig | None = None) -> VectorDatabase:
    # ... existing backend selection (embedded vs external) ...
    # ... existing team fusion (RRFDatabase) ...

    # ── Cloud fusion arm ──
    if config.cloud_enabled and config.cloud_api_key:
        from memini_ai.memory.cloud_database import CloudDatabase

        cloud_db = CloudDatabase(
            url=config.cloud_url,
            api_key=config.cloud_api_key,
            timeout=config.cloud_timeout_request,
            max_retries=config.cloud_max_retries,
        )

        if config.cloud_fusion_mode == "rrf":
            # Wrap existing db (which may already be RRFDatabase) with cloud
            if isinstance(db, RRFDatabase):
                # Three-arm: embedded + team + cloud
                db = RRFDatabase(
                    primary=db._primary,
                    secondary=db._secondary,
                    tertiary=cloud_db,  # NEW: tertiary arm
                    k=config.rrf_k,
                )
            else:
                # Two-arm: local + cloud
                db = RRFDatabase(
                    primary=db,
                    secondary=cloud_db,
                    k=config.rrf_k,
                )
        else:
            # Cloud as sole backend (cloud_fusion_mode="none" but cloud_enabled=true)
            db = cloud_db
            logger.info("cloud_sole_backend", url=config.cloud_url)

    return db
```

### 3.5 `CloudDatabase` — A `VectorDatabase` That Talks JSON-RPC

```python
# src/memini_ai/memory/cloud_database.py (NEW FILE, ~400 LOC)

class CloudDatabase(VectorDatabase):
    """A VectorDatabase implementation that proxies to memini-ai-cloud over JSON-RPC/HTTPS.

    This is the client-side half of the thin-client model. It implements the
    full VectorDatabase ABC by translating each method into a tools/call JSON-RPC
    request to the cloud server.

    Design properties:
    - Stateless. No local DB, no embedding model, no connection pool.
    - Best-effort. Network failures are caught and logged; callers should treat
      this as a secondary/tertiary arm in RRFDatabase.
    - Retry with exponential backoff on transient failures (5xx, connection errors).
    - No local caching in v1.
    """

    def __init__(self, url: str, api_key: str, timeout: float = 30.0, max_retries: int = 3):
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        # Preflight: call get_status to verify connectivity
        try:
            status = await self._call_tool("get_status", {})
            logger.info("cloud_preflight_ok", memory_count=status.get("memoryCount"))
        except Exception as e:
            logger.warning("cloud_preflight_failed", error=str(e)[:200])
        self._initialized = True

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a memini-ai tool on the cloud server with retry."""
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post("/rpc", json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                })
                resp.raise_for_status()
                result = resp.json()
                if "error" in result:
                    raise CloudRPCError(result["error"])
                # FastMCP returns tool result in result.content[0].text
                content = result.get("result", {}).get("content", [])
                if content and content[0].get("type") == "text":
                    return json.loads(content[0]["text"])
                return result.get("result", {})
            except (httpx.HTTPError, CloudRPCError) as e:
                last_exc = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt * 0.5  # 0.5, 1, 2 seconds
                    await asyncio.sleep(wait)
        raise last_exc

    # Each VectorDatabase method delegates to _call_tool:
    async def add_memory(self, entry: MemoryEntry) -> str:
        await self.initialize()
        result = await self._call_tool("add_memory", {
            "content": entry.content,
            "sourceType": entry.source_type,
            "metadata": entry.metadata,
        })
        return result["memory_id"]

    async def query_memories(self, vector, options, collection_name=None) -> list[MemoryEntry]:
        await self.initialize()
        result = await self._call_tool("query_memories", {
            "query": options.query_text,  # Send text, not vector — cloud embeds it
            "limit": options.top_k,
            "strategy": options.strategy,
        })
        return [MemoryEntry(**e) for e in result.get("memories", [])]

    # ... all other methods follow the same pattern ...
```

**Key design decision: Send query text, not vectors.** The thin client does not embed locally. It sends the raw query text to the cloud, and the cloud embeds it server-side. This is the whole point of the thin-client model — the client has no embedding model.

---

## 4. Cloud Server Side

### 4.1 What's Already Designed (from SaaS Doc)

The existing SaaS doc (`memini-saas-horizontal-scaling.md`) covers the server side in detail. The thin-client model does NOT change these server-side components:

| Component | Status | Notes |
|-----------|--------|-------|
| **Auth Gateway** (Envoy/Kong/NGINX) | Designed (§4.1) | API key validation, TLS termination, rate limiting, tenant context injection. |
| **MCP Server Pods** (stateless) | Designed (§4.1) | Run the same FastMCP server. No model loaded (embedding is a separate service). |
| **Embedding Service** (shared, gRPC) | Designed (§4.1) | MiniLM-L6-v2 (384-dim) default, BGE-M3 (1024-dim) premium. Auto-scaled. |
| **Tiered Tenancy** (pooled + dedicated DB) | Designed (§3.4) | Starter = pooled (row-level tenant_id), Pro = dedicated DB, Enterprise = dedicated server. |
| **API Key Management** | Designed (§6) | `mem_<tenant_id>_<random_32_hex>`, bcrypt hashed, control plane DB. |
| **Rate Limiting** | Designed (§6.5) | Starter: 10 rps, Pro: 50 rps, Enterprise: custom. |
| **K8s Deployment Shapes** | Designed (§4.2) | HPA on CPU/queue depth, PDB, health checks. |
| **Cost Model** | Designed (§9) | ~$0.50–$2.50/user/month infrastructure cost. |
| **11 Required Changes** | Designed (§7) | Config-per-request (#1 blocker), auth middleware, tenant context, multi-DB routing, etc. |

### 4.2 What Changes with the Thin-Client Model

The thin-client model simplifies the server side in one key way: **the MCP server pods don't need to run the full memini-ai tool surface as MCP tools.** They can run a simpler JSON-RPC handler that directly calls the `MemorySystem` API.

**However, for v1, we keep the existing FastMCP server.** The cloud MCP server pods run the same `server.py` as local memini-ai, just with `transport="streamable-http"` and behind the auth gateway. This minimizes code divergence.

**The critical path (unchanged from SaaS doc §7.1):**

```
1 (config-per-request) → 3 (tenant context) → 4 (multi-DB routing)
2 (auth middleware) → 3 (tenant context)
5 (API key management) → 2 (auth middleware)
```

### 4.3 Session-Count Enforcement

The thin-client model introduces a new requirement: **limit the number of concurrent sessions per tenant.** This is enforced at the gateway:

- **Starter:** 1 concurrent session
- **Pro:** 5 concurrent sessions
- **Enterprise:** Custom

The gateway tracks active sessions by API key. A session is "active" if it has made a request in the last 60 seconds. On session limit exceeded, the gateway returns `429 Too Many Requests` with `X-Session-Limit: <max>` and `X-Session-Count: <current>`.

### 4.4 Rate-Limit Enforcement Points

| Enforcement Point | What | How |
|-------------------|------|-----|
| **Gateway** | Requests per second per API key | Token bucket. `X-RateLimit-*` headers. |
| **Gateway** | Concurrent sessions per API key | Active session counter with 60s TTL. |
| **Embedding Service** | Embeddings per second per tenant | gRPC interceptor. Queues excess requests. |
| **Postgres** | Connections per tenant | PgBouncer connection pooling. Max connections per pool. |

---

## 5. Fusion Semantics

### 5.1 RRF Across Embedded + Team + Cloud

When all three arms are active, `query_memories` fans out to all three in parallel and RRF-fuses the results:

```
query_memories("what is the user's preferred language?")
  ├─ embedded (primary)    → [mem_1, mem_3, mem_5]  (ranked by cosine similarity)
  ├─ team (secondary)      → [mem_2, mem_1, mem_4]  (ranked by cosine similarity)
  └─ cloud (tertiary)     → [mem_1, mem_6, mem_3]  (ranked by cosine similarity)

RRF(k=60) fuse:
  mem_1: 1/(60+1) + 1/(60+2) + 1/(60+1) = 0.0164 + 0.0161 + 0.0164 = 0.0489
  mem_2: 0 + 1/(60+1) + 0 = 0.0164
  mem_3: 1/(60+2) + 0 + 1/(60+3) = 0.0161 + 0.0159 = 0.0320
  ...

Final ranking: [mem_1, mem_3, mem_2, mem_5, mem_4, mem_6]
```

### 5.2 Dedup

Same `memory_id` in multiple arms → RRF naturally boosts the score (sum of contributions). The `MemoryEntry` object from the **primary** (embedded) wins on content conflicts (same `setdefault` pattern as existing `RRFDatabase`).

### 5.3 Write Routing

| Operation | Primary (embedded) | Team (secondary) | Cloud (tertiary) |
|-----------|-------------------|-------------------|------------------|
| `add_memory` | **Awaited** — canonical id returned | Fire-and-forget (Q3) | Fire-and-forget |
| `delete_memory` | **Awaited** | Not propagated | Not propagated |
| `adjust_trust` | **Awaited** | Not propagated | Not propagated |
| `update_user_profile` | **Awaited** | Not propagated | Not propagated |

**Rationale:** The primary is authoritative. Team and cloud are read-replicas for recall broadening. Writes to team/cloud are best-effort — if they fail, the primary still has the data. This matches the existing Q3 decision (team write-through is fire-and-forget).

### 5.4 Trust-Score Authority

**The primary (embedded) is the trust-score authority.** Trust adjustments (`adjust_trust`, decay engine, consolidation) run on the primary only. Team and cloud may have stale trust scores — the RRF fusion uses the primary's `MemoryEntry` on conflicts, so the primary's trust score wins.

**Cloud-side trust engine:** When the cloud is the sole backend (thin-client mode), the cloud runs its own trust engine, decay engine, and consolidation. These are scoped per tenant (SaaS doc §7, change #7).

### 5.5 Degradation Behavior

| Failure | Behavior |
|---------|----------|
| Cloud unreachable | Degrade to embedded + team (or embedded-only). Log warning. |
| Team unreachable | Degrade to embedded + cloud (or embedded-only). Log warning. |
| Embedded crashed | Propagate to caller (embedded is authoritative). On retry, auto-restart via heartbeat. |
| Cloud + Team both unreachable | Embedded-only. Log warning. |
| All three unreachable | Error propagated to caller. |

---

## 6. Phased Implementation Plan

### Phase 0: End-to-End Proof (2–3 weeks)

**Goal:** Thin client → cloud gateway → single-tenant memini-ai over streamable-http+TLS with static API key. Smallest possible end-to-end.

| Card | Title | Depends On | Effort |
|------|-------|------------|--------|
| **T-CLOUD-001** | `CloudDatabase` — VectorDatabase impl over JSON-RPC/HTTPS | — | M |
| **T-CLOUD-002** | `CloudClientConfig` — thin client config schema (URL + key + transport) | — | S |
| **T-CLOUD-003** | `memini-ai-cloud` package scaffold (pyproject.toml, FastMCP proxy server) | T-CLOUD-001, T-CLOUD-002 | M |
| **T-CLOUD-004** | Cloud gateway — Envoy/Kong with static API key auth + TLS termination | — | M |
| **T-CLOUD-005** | Deploy memini-ai server pod with `streamable-http` behind gateway | T-CLOUD-004 | S |
| **T-CLOUD-006** | End-to-end smoke test: thin client → gateway → server → add_memory + query_memories | T-CLOUD-003, T-CLOUD-005 | S |
| **T-CLOUD-007** | `MEMINI_CLOUD_*` config fields in `MeminiConfig` + `create_database()` extension | T-CLOUD-001 | M |
| **T-CLOUD-008** | Full memini-ai uses cloud as RRF fusion arm (embedded + cloud) | T-CLOUD-007 | M |
| **T-CLOUD-009** | Phase 0 integration tests (thin client + fusion arm) | T-CLOUD-006, T-CLOUD-008 | M |

**Phase 0 exit criteria:**
- [ ] Thin client connects to cloud over streamable-http+TLS.
- [ ] `add_memory` → cloud → Postgres → returns memory_id.
- [ ] `query_memories` → cloud → embedding service → Postgres → returns results.
- [ ] Full memini-ai with `MEMINI_CLOUD_ENABLED=true` fuses embedded + cloud results via RRF.
- [ ] Invalid API key → 401.
- [ ] Latency <500ms p95 for `add_memory`, <200ms p95 for `query_memories` (same as SaaS doc Phase 0).

### Phase 1: Multi-Tenant Cloud (4–8 weeks)

**Goal:** Multiple tenants on shared infrastructure. API key management. Rate limiting. Session counting.

| Card | Title | Depends On | Effort |
|------|-------|------------|--------|
| **T-CLOUD-010** | Config-per-request refactor (break `get_config()` singleton) | — | L |
| **T-CLOUD-011** | Tenant context propagation (`tenant_id` on all queries) | T-CLOUD-010 | M |
| **T-CLOUD-012** | API key management service (control plane DB, key generation, validation) | — | L |
| **T-CLOUD-013** | Auth middleware for streamable-http (validate API key, inject tenant context) | T-CLOUD-012 | M |
| **T-CLOUD-014** | Multi-DB connection routing (API key → DSN → connection pool) | T-CLOUD-010, T-CLOUD-011 | L |
| **T-CLOUD-015** | Rate limiting + session counting at gateway | T-CLOUD-013 | M |
| **T-CLOUD-016** | Pooled multi-tenant Postgres (row-level `tenant_id`) | T-CLOUD-014 | M |
| **T-CLOUD-017** | Shared embedding service (gRPC, stateless, auto-scaled) | — | L |
| **T-CLOUD-018** | K8s deployment (MCP server pods + embedding service + Postgres) | T-CLOUD-016, T-CLOUD-017 | L |
| **T-CLOUD-019** | Thin client: API key config + preflight health check | T-CLOUD-013 | S |
| **T-CLOUD-020** | Phase 1 integration tests (multi-tenant isolation, rate limiting) | T-CLOUD-015, T-CLOUD-018 | M |

**Phase 1 exit criteria:**
- [ ] 10 tenants can use the service simultaneously without data leakage.
- [ ] Tenant A's `query_memories` returns only Tenant A's memories.
- [ ] API key rotation works (old key stops working within 60s).
- [ ] Rate limiting works (starter: 10 rps).
- [ ] Session limit enforced (starter: 1 concurrent session).
- [ ] Embedding service scales independently of MCP server.

### Phase 2: Dedicated DB Tier + Full Fusion (4–8 weeks)

**Goal:** Pro tier with dedicated databases. Three-arm RRF (embedded + team + cloud). Background job tenancy.

| Card | Title | Depends On | Effort |
|------|-------|------------|--------|
| **T-CLOUD-021** | Dedicated DB provisioning (API key → dedicated DSN) | T-CLOUD-014 | L |
| **T-CLOUD-022** | Tier promotion (pooled → dedicated migration) | T-CLOUD-021 | L |
| **T-CLOUD-023** | Three-arm RRFDatabase (embedded + team + cloud) | T-CLOUD-008 | M |
| **T-CLOUD-024** | Cloud as sole backend mode (full memini-ai, cloud-hosted) | T-CLOUD-007 | M |
| **T-CLOUD-025** | Background job tenancy (decay, consolidation per tenant) | T-CLOUD-010 | M |
| **T-CLOUD-026** | LLM credential handling per tenant | T-CLOUD-010 | M |
| **T-CLOUD-027** | Thin client: session resumption (v2, WebSocket) | T-CLOUD-003 | M |
| **T-CLOUD-028** | Thin client: tier0/tier1 local cache (v2) | T-CLOUD-003 | S |
| **T-CLOUD-029** | HPA on embedding queue depth + p99 latency | T-CLOUD-018 | M |
| **T-CLOUD-030** | Phase 2 integration tests (dedicated DB isolation, three-arm RRF) | T-CLOUD-023, T-CLOUD-024 | M |

### Phase 3: GA — Metering, Billing, SLA (4–8 weeks)

| Card | Title | Depends On | Effort |
|------|-------|------------|--------|
| **T-CLOUD-031** | Usage metering (memories stored, embeddings generated, LLM tokens) | T-CLOUD-010 | M |
| **T-CLOUD-032** | Billing integration (Stripe) | T-CLOUD-031 | L |
| **T-CLOUD-033** | Admin dashboard (tenant management, key management, usage reports) | T-CLOUD-012 | L |
| **T-CLOUD-034** | GDPR compliance (per-tenant data export, per-tenant data deletion) | T-CLOUD-021 | M |
| **T-CLOUD-035** | Model upgrade migration per tenant (re-embed on model change) | T-CLOUD-017 | L |
| **T-CLOUD-036** | Export/import CLI (`memini-ai export` / `memini-ai import`) | — | M |
| **T-CLOUD-037** | SLA monitoring (uptime, latency percentiles) | T-CLOUD-018 | M |
| **T-CLOUD-038** | GA integration tests + load tests | T-CLOUD-031..037 | L |

### Dependency Graph (Phases 0–3)

```
Phase 0 (E2E proof):
  T-CLOUD-001 ──► T-CLOUD-003 ──► T-CLOUD-006
  T-CLOUD-002 ──► T-CLOUD-003
  T-CLOUD-004 ──► T-CLOUD-005 ──► T-CLOUD-006
  T-CLOUD-001 ──► T-CLOUD-007 ──► T-CLOUD-008
  T-CLOUD-006 + T-CLOUD-008 ──► T-CLOUD-009

Phase 1 (Multi-tenant):
  T-CLOUD-010 ──► T-CLOUD-011 ──► T-CLOUD-014 ──► T-CLOUD-016
  T-CLOUD-012 ──► T-CLOUD-013 ──► T-CLOUD-015
  T-CLOUD-013 ──► T-CLOUD-019
  T-CLOUD-017 (parallel)
  T-CLOUD-016 + T-CLOUD-017 ──► T-CLOUD-018
  T-CLOUD-015 + T-CLOUD-018 ──► T-CLOUD-020

Phase 2 (Dedicated + Fusion):
  T-CLOUD-014 ──► T-CLOUD-021 ──► T-CLOUD-022
  T-CLOUD-008 ──► T-CLOUD-023
  T-CLOUD-007 ──► T-CLOUD-024
  T-CLOUD-010 ──► T-CLOUD-025, T-CLOUD-026
  T-CLOUD-003 ──► T-CLOUD-027, T-CLOUD-028
  T-CLOUD-018 ──► T-CLOUD-029
  T-CLOUD-023 + T-CLOUD-024 ──► T-CLOUD-030

Phase 3 (GA):
  T-CLOUD-010 ──► T-CLOUD-031 ──► T-CLOUD-032
  T-CLOUD-012 ──► T-CLOUD-033
  T-CLOUD-021 ──► T-CLOUD-034
  T-CLOUD-017 ──► T-CLOUD-035
  T-CLOUD-018 ──► T-CLOUD-037
  (T-CLOUD-036 parallel)
  T-CLOUD-031..037 ──► T-CLOUD-038
```

---

## 7. Risks & Open Questions

### 7.1 Top 3 Risks

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| **1** | **Latency of cloud embedding vs local MiniLM.** A local `add_memory` takes ~1.3s (DB write + embedding). A cloud `add_memory` adds network round-trip (~50–200ms depending on region). At p95, this could push `add_memory` over the 60s MCP timeout if the cloud is under load. | **HIGH** | **MEDIUM** | Phase 0 must benchmark cloud latency. If >2s p95, consider edge embedding (embed client-side, send vector to cloud) as a v2 option. The thin client could optionally load a tiny embedding model (MiniLM-L6-v2 is ~80MB) for latency-sensitive users. |
| **2** | **Privacy: user memories leave the machine.** The thin client sends raw memory text to the cloud. This is a fundamental privacy trade-off — the user's agent conversations, code snippets, and personal notes are transmitted over the network and stored on cloud infrastructure. | **HIGH** | **CERTAIN** | Encryption at rest (AES-256-GCM with per-tenant keys). Encryption in transit (TLS 1.3). Data residency options (US/EU/APAC regions). Clear privacy policy. SOC 2 compliance for Pro/Enterprise tiers. Option for BYOK (bring your own key) — tenant manages their own encryption key via KMS. |
| **3** | **`get_config()` singleton refactor risk.** 36 call sites. High chance of missing one and introducing subtle bugs (wrong DB, wrong LLM key, wrong tenant). This is the #1 blocker for multi-tenancy and is unchanged from the SaaS doc §11.1 risk #6. | **HIGH** | **MEDIUM** | Type-checked `RequestContext`, exhaustive grep, integration tests per tenant. This is the critical path for Phase 1. |

### 7.2 Other Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 4 | **Cloud vendor lock-in.** Users who store all memories in the cloud cannot easily migrate to self-hosted. | MEDIUM | `memini-ai export` / `memini-ai import` CLI (T-CLOUD-036). Standard `pg_dump` format. |
| 5 | **Trust engine divergence.** Cloud-side trust scores may differ from local trust scores if the user runs both simultaneously. | MEDIUM | Primary (embedded) is trust-score authority. Cloud trust scores are advisory only in fusion mode. |
| 6 | **Background job tenancy.** Decay engine, consolidation engine, auto-extraction currently run globally. Must be scoped per tenant. | MEDIUM | SaaS doc §7 change #7. Iterate over tenants, run scoped jobs. |
| 7 | **LLM cost per tenant.** Entity extraction, tiered summaries, dialectic all call LLM. At $0.01–0.10/call, costs scale with usage. | MEDIUM | SaaS doc §11.1 risk #2. LLM call budgeting per tier. Make LLM features opt-in for starter tier. |
| 8 | **Connection pool exhaustion.** 100 tenants × 10 connections = 1,000 connections per MCP server pod. | MEDIUM | SaaS doc §11.1 risk #7. PgBouncer in front of Postgres. Connection pool per DSN, not per tenant. |
| 9 | **pgembed not applicable to cloud.** The cloud uses external Postgres (Docker/K8s), not pgembed. The thin client has no pgembed. | LOW | Cloud server uses `MEMINI_VECTOR_BACKEND=postgres-external`. Thin client has no vector backend at all. |

### 7.3 Open Questions

| # | Question | Notes |
|---|----------|-------|
| **Q1** | **Should the thin client be Python or TypeScript?** | Python matches the memini-ai ecosystem and can reuse `httpx` + `FastMCP`. TypeScript would be more natural for the OpenCode plugin ecosystem. Decision: **Python for v1** (same language, same FastMCP, same tooling). TypeScript thin client can be a community contribution. |
| **Q2** | **What's the signup flow?** | Self-serve: user signs up on cloud.memini.ai → gets API key → pastes into OpenCode config. Or invite-only: admin creates tenant + key → shares with user. Both need a control plane API (SaaS doc §11.2 Q6). |
| **Q3** | **How do we handle the kanban board in multi-tenant?** | The kanban board (`kanban_add_card`, `kanban_list_cards`) stores cards in the same DB. Per-tenant scoping needed (SaaS doc §11.2 Q7). |
| **Q4** | **What about the project indexer in SaaS?** | The project indexer indexes files on disk. In SaaS, each tenant's files must be isolated (S3 per tenant, or a filesystem abstraction). SaaS doc §11.2 Q3. |
| **Q5** | **Should we offer a "bring your own DB" option?** | Enterprise tier could connect to the customer's own Postgres. The MCP server just needs a DSN. Already supported via `MEMINI_DB_URL` — expose as a config option (SaaS doc §11.2 Q9). |
| **Q6** | **What's the migration path local→cloud and cloud→local?** | `memini-ai export` (dump all memories as JSONL) + `memini-ai import` (load JSONL into target). Standard format. T-CLOUD-036. |
| **Q7** | **How do we meter billing?** | Emit usage events (memories stored, embeddings generated, LLM tokens consumed) to a metering pipeline. Per-tenant aggregation. SaaS doc §7 change #8. |
| **Q8** | **Should the thin client support multiple cloud regions?** | v1: single URL. v2: `MEMINI_CLOUD_URL` can be a comma-separated list; client picks the lowest-latency region on startup. |
| **Q9** | **What happens to `get_status` in thin-client mode?** | The thin client proxies `get_status` to the cloud. The response includes cloud-side metrics (memory count, model name, query latency). No local metrics (there is no local DB). |
| **Q10** | **Can the thin client work with a self-hosted cloud?** | Yes. `MEMINI_CLOUD_URL` can point to any memini-ai server running `streamable-http` with auth. The cloud server is the same `server.py` — just deployed behind a gateway. |

---

## Appendix A: Key File Reference

| File | LOC | Role |
|------|-----|------|
| `memini-ai-cloud/src/memini_cloud/proxy.py` | ~200 | FastMCP proxy server (tool discovery + catch-all handler) |
| `memini-ai-cloud/src/memini_cloud/client.py` | ~150 | JSON-RPC over HTTPS client with retry |
| `memini-ai-cloud/src/memini_cloud/config.py` | ~60 | `CloudClientConfig` (URL + key + transport) |
| `memini-ai-dev/src/memini_ai/memory/cloud_database.py` | ~400 | `CloudDatabase` — `VectorDatabase` impl over JSON-RPC |
| `memini-ai-dev/src/memini_ai/config.py` | +50 | New `MEMINI_CLOUD_*` fields |
| `memini-ai-dev/src/memini_ai/memory/database.py` | +40 | `create_database()` extension for cloud fusion |
| `memini-ai-dev/src/memini_ai/memory/rrf_database.py` | +60 | Three-arm RRF (embedded + team + cloud) |
| `memini-ai-dev/src/memini_ai/server.py` | +20 | Auth middleware for `streamable-http` (Phase 1) |

## Appendix B: What We Are NOT Building (v1)

- **NOT** a WebSocket transport. v1 is streamable-http only.
- **NOT** local caching in the thin client. v1 is a pure proxy.
- **NOT** offline mode. The thin client fails fast if the cloud is unreachable.
- **NOT** a TypeScript thin client. v1 is Python (FastMCP).
- **NOT** a custom protocol. v1 uses standard MCP JSON-RPC 2.0 over HTTPS.
- **NOT** client-side embedding. The thin client sends raw text; the cloud embeds it.
- **NOT** a dashboard/UI. v1 is API-only.
- **NOT** changing the MCP tool surface. All 52 tools work identically over the cloud.

## Appendix C: Cross-Reference to SaaS Doc

| SaaS Doc Section | Status |
|------------------|--------|
| §1 (Baseline) | Unchanged |
| §2 (Architecture Recon) | Unchanged |
| §3 (Options Analysis) | Unchanged |
| §4 (Recommended Architecture) | **KEPT** — server-side components unchanged. Client-side pivoted to thin proxy. |
| §5 (Envelope/Request Flow) | **KEPT** — same JSON-RPC 2.0 over streamable-http. |
| §6 (API Key → Tenancy) | Unchanged |
| §7 (What Must Change) | Unchanged — 11 changes still required for multi-tenancy. |
| §8 (Security & Isolation) | Unchanged |
| §9 (Capacity & Cost Model) | Unchanged |
| §10 (Phased Rollout) | **SUPERSEDED** by this doc's §6 (Phased Implementation Plan). |
| §11 (Risks & Open Questions) | **AUGMENTED** by this doc's §7. |
