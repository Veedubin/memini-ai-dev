# Dimension 1: Hermes Agent Memory Provider Ecosystem

## The Comparison Table (8 Providers)

| Provider | Storage | Cost | Tools | Dependencies | Unique Feature |
|----------|---------|------|-------|--------------|----------------|
| **Honcho** | Cloud | Paid | 5 | `honcho-ai` | Dialectic user modeling + session-scoped context |
| **OpenViking** | Self-hosted | Free | 5 | `openviking` + server | Filesystem hierarchy + tiered loading |
| **Mem0** | Cloud | Paid | 3 | `mem0ai` | Server-side LLM extraction |
| **Hindsight** | Cloud/Local | Free/Paid | 3 | `hindsight-client` | Knowledge graph + reflect synthesis |
| **Holographic** | Local | Free | 2 | None | HRR algebra + trust scoring |
| **RetainDB** | Cloud | $20/mo | 5 | `requests` | Delta compression |
| **ByteRover** | Local/Cloud | Free/Paid | 3 | `brv` CLI | Pre-compression extraction |
| **Supermemory** | Cloud | Paid | 4 | `supermemory` | Context fencing + session graph ingest + multi-container |

## How Hermes Memory Providers Work

When a memory provider is active, Hermes automatically:
1. **Injects provider context** into the system prompt (what the provider knows)
2. **Prefetches relevant memories** before each turn (background, non-blocking)
3. **Syncs conversation turns** to the provider after each response
4. **Extracts memories on session end** (for providers that support it)
5. **Mirrors built-in memory writes** to the external provider
6. **Adds provider-specific tools** so the agent can search, store, and manage memories

The built-in memory (MEMORY.md / USER.md) continues to work exactly as before. The external provider is additive. Only one external provider can be active at a time.

## The MemoryProvider Protocol (ABC)

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """Check if this provider can activate. NO network calls."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup."""
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    def sync_turn(self, user_message, assistant_response, **kwargs) -> None:
        """Sync conversation turn to provider"""
        ...

    def prefetch(self, query: str, **kwargs) -> str | None:
        """Prefetch relevant memories before turn"""
        ...

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

Directory structure:
```
plugins/memory/<name>/
  __init__.py      # MemoryProvider implementation + register() entry point
  plugin.yaml      # Metadata (name, description, hooks)
  README.md        # Setup instructions, config reference, tools
```

## Profile Isolation

Each provider's data is isolated per profile:
- **Local storage providers** (Holographic, ByteRover) use `$HERMES_HOME/` paths which differ per profile
- **Config file providers** (Honcho, Mem0, Hindsight, Supermemory) store config in `$HERMES_HOME/` so each profile has its own credentials
- **Cloud providers** (RetainDB) auto-derive profile-scoped project names
- **Env var providers** (OpenViking) are configured via each profile's `.env` file
