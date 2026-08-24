"""Configuration management using pydantic-settings."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MeminiConfig(BaseSettings):
    """Main configuration class for memini-ai.

    Configuration priority: env vars > JSON config file > defaults.
    JSON config path: .opencode/memini-ai/config.json (auto-created if missing).
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    precision: str = "fp16"
    device: str = "auto"
    use_gpu: bool = False
    embedding_dim: int = 384
    batch_size: int = 32
    eager_load: bool = False

    # Dual-model RRF (v0.7.0+)
    embedding_mode: str = Field(default="auto", alias="EMBEDDING_MODE")
    elevate_enabled: bool = Field(default=True, alias="ELEVATE_ENABLED")
    rrf_k: int = Field(default=60, alias="RRF_K")
    auto_extract_log_dir: str = Field(
        default="~/.memini-ai/chat_logs", alias="AUTO_EXTRACT_LOG_DIR"
    )
    auto_extract_interval_seconds: int = Field(
        default=5, alias="AUTO_EXTRACT_INTERVAL_SECONDS"
    )

    # Embedding policy (v0.7.7+)
    # When True, a model dim mismatch raises RuntimeError at load time
    # (old behavior). When False (default), a dim mismatch logs a warning
    # and degrades to text-only search.
    strict_embedding_dim: bool = Field(
        default=False, alias="MEMINI_STRICT_EMBEDDING_DIM"
    )
    # When True (default), new deployments (0 memories) with the default
    # model (all-MiniLM-L6-v2) are auto-upgraded to BGE-M3 (1024-dim).
    # Existing deployments (memory_count > 0) keep their configured model.
    auto_detect_model: bool = Field(default=True, alias="MEMINI_AUTO_DETECT_MODEL")

    # Multi-model embedding support (v0.12.0+)
    # The active model used for NEW writes. Must be one of ENABLED_MODELS.
    model_name: str = Field(default="all-MiniLM-L6-v2", alias="MEMINI_MODEL_NAME")
    # All models the system knows about (for RRF query dispatch)
    enabled_models: list[str] = Field(
        default_factory=lambda: [
            "all-MiniLM-L6-v2",
            "BAAI/bge-m3",
        ],
        alias="MEMINI_ENABLED_MODELS",
    )
    # When True, query_memories merges results from all model spaces via RRF
    enable_rrf: bool = Field(default=True, alias="MEMINI_ENABLE_RRF")
    # How many results to fetch from each model's vector space before fusion
    rrf_top_k_per_model: int = Field(default=20, alias="RRF_TOP_K_PER_MODEL")

    # Image search (v0.8.0+, shared with videre-mcp via memini-vision)
    # When True, query_memories adds a 3rd RRF fan-out arm that calls
    # memini-vision.ImageQuery.search_by_text to fuse CLIP image results
    # with the existing 384 + 1024 text results. When False (the default),
    # no CLIP model loads, no image table is queried, RRF stays 2-list,
    # and query_memories is byte-for-byte identical to v0.7.9.
    image_search_enabled: bool = Field(
        default=False, alias="MEMINI_IMAGE_SEARCH_ENABLED"
    )
    image_clip_model: str = Field(
        default="clip-ViT-B-32", alias="MEMINI_IMAGE_CLIP_MODEL"
    )
    image_clip_device: str = Field(default="auto", alias="MEMINI_IMAGE_CLIP_DEVICE")
    image_dir: str = Field(default="~/.memini-ai/images", alias="MEMINI_IMAGE_DIR")
    image_db_url: str = Field(
        default="", alias="MEMINI_IMAGE_DB_URL"
    )  # empty → falls back to db_url at runtime

    # Database settings
    table_name: str = "memories"
    project_id: str | None = None
    query_collections: list[str] | None = None

    # PostgreSQL / pgvector settings
    db_url: str = ""  # Set via MEMINI_DB_URL env var or .env file
    db_pool_size: int = 10
    db_min_size: int = 2
    db_max_size: int = 20

    # PostgreSQL TLS/SSL settings
    # Valid sslmode values: disable, allow, prefer, require, verify-ca, verify-full
    # See: https://www.postgresql.org/docs/current/libpq-ssl.html#LIBPQ-SSL-SSLMODE-STATEMENTS
    db_sslmode: str = Field(
        default="disable",
        alias="DB_SSLMODE",
        description="PostgreSQL SSL mode (disable, allow, prefer, require, verify-ca, verify-full)",
    )
    db_sslrootcert: str | None = Field(
        default=None,
        alias="DB_SSLROOTCERT",
        description="Path to CA certificate for SSL server verification",
    )

    # ── v1.0.0: Backend selection (pgembed embedded vs external Postgres) ──
    # When 'pgembed' (default), an embedded PostgreSQL 17 server is started
    # in-process and uses MEMINI_PGEMBED_DATA_DIR as its data directory.
    # When 'postgres-external', connects to an external Postgres server
    # via MEMINI_DB_URL (Docker, team server, etc.).
    vector_backend: Literal["pgembed", "postgres-external"] = Field(
        default="pgembed",
        alias="MEMINI_VECTOR_BACKEND",
        description=(
            "Vector database backend. 'pgembed' (default) starts an embedded "
            "PostgreSQL 17 server in-process. 'postgres-external' connects to "
            "an external PostgreSQL server via MEMINI_DB_URL (Docker, team server, etc.)."
        ),
    )
    # Data directory for the embedded pgembed PostgreSQL server. Persistent
    # across restarts. Default: ~/.local/share/memini-ai/pgembed/data
    pgembed_data_dir: str = Field(
        default="~/.local/share/memini-ai/pgembed/data",
        alias="MEMINI_PGEMBED_DATA_DIR",
        description=(
            "Data directory for the embedded pgembed PostgreSQL server. "
            "Persistent across restarts. Default: ~/.local/share/memini-ai/pgembed/data"
        ),
    )
    # Optional team/shared PostgreSQL server URL for RRF fusion with
    # embedded results. Empty string means no team backend (single-backend mode).
    team_db_url: str = Field(
        default="",
        alias="MEMINI_TEAM_DB_URL",
        description=(
            "Optional team/shared PostgreSQL server URL for RRF fusion with "
            "embedded results."
        ),
    )
    # Fusion mode for multi-backend queries. 'none' (default) queries
    # only the primary backend. 'rrf' fuses results from embedded + team.
    fusion_mode: Literal["none", "rrf"] = Field(
        default="none",
        alias="MEMINI_FUSION_MODE",
        description=(
            "Fusion mode for multi-backend queries. 'none' (default) queries "
            "only the primary backend. 'rrf' fuses results from embedded + team."
        ),
    )

    # Indexer settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size: int = Field(default_factory=lambda: 10 * 1024 * 1024)
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["node_modules", ".git", "dist"]
    )

    # Logging
    log_level: str = "info"

    # Performance
    workers: int = Field(default_factory=lambda: os.cpu_count() or 4)

    # Trust Engine settings
    trust_engine_enabled: bool = Field(default=False, alias="TRUST_ENGINE")
    trust_threshold_archive: float = Field(default=0.2, alias="TRUST_THRESHOLD_ARCHIVE")
    trust_threshold_promote: float = Field(default=0.8, alias="TRUST_THRESHOLD_PROMOTE")
    trust_delta_use: float = Field(default=0.05, alias="TRUST_DELTA_USE")
    trust_delta_ignore: float = Field(default=-0.02, alias="TRUST_DELTA_IGNORED")
    trust_delta_correct: float = Field(default=-0.15, alias="TRUST_DELTA_CORRECT")
    trust_delta_confirm: float = Field(default=0.10, alias="TRUST_DELTA_CONFIRM")

    # Memory Graph settings
    memory_graph_enabled: bool = Field(default=False, alias="MEMORY_GRAPH")
    graph_entity_extraction: bool = Field(default=True, alias="GRAPH_ENTITY_EXTRACTION")
    graph_relationship_suggestions: bool = Field(
        default=True, alias="GRAPH_RELATIONSHIP_SUGGESTIONS"
    )

    # Auto-Extract settings
    auto_extract_enabled: bool = Field(default=False, alias="AUTO_EXTRACT")
    auto_extract_turns: int = Field(default=5, alias="AUTO_EXTRACT_TURNS")

    # LLM settings
    llm_url: str = Field(default="http://localhost:11434/api/generate", alias="LLM_URL")
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")

    # Pre-Compression Extraction settings
    precompress_enabled: bool = Field(default=False, alias="PRECOMPRESS")
    precompress_threshold: float = Field(default=0.8, alias="PRECOMPRESS_THRESHOLD")

    # Tiered Loading settings
    tiered_loading_enabled: bool = Field(default=False, alias="TIERED_LOADING")
    tier0_max_tokens: int = Field(default=100, alias="TIER0_MAX_TOKENS")
    tier1_max_tokens: int = Field(default=2000, alias="TIER1_MAX_TOKENS")
    tier0_cache_ttl: int = Field(default=3600, alias="TIER0_CACHE_TTL")  # seconds
    tier1_cache_ttl: int = Field(default=7200, alias="TIER1_CACHE_TTL")  # seconds

    # User Modeling settings
    user_modeling_enabled: bool = Field(default=False, alias="USER_MODELING")
    user_model_min_sessions: int = Field(default=50, alias="USER_MODEL_MIN_SESSIONS")

    # Phase 4A: Memory Decay settings
    decay_enabled: bool = Field(default=False, alias="DECAY_ENABLED")
    decay_half_life_days: int = Field(default=90, alias="DECAY_HALF_LIFE_DAYS")
    consolidation_interval_hours: int = Field(
        default=168, alias="CONSOLIDATION_INTERVAL_HOURS"
    )
    consolidation_similarity_threshold: float = Field(
        default=0.92, alias="CONSOLIDATION_SIMILARITY_THRESHOLD"
    )

    # Phase 4B: Knowledge Graph settings
    knowledge_graph_enabled: bool = Field(default=False, alias="KG_ENABLED")
    kg_entity_extraction: bool = Field(default=True, alias="KG_ENTITY_EXTRACTION")
    kg_inference_depth: int = Field(default=3, alias="KG_INFERENCE_DEPTH")
    kg_max_results: int = Field(default=100, alias="KG_MAX_RESULTS")

    # Phase 4C: Multi-Peer settings
    multi_peer_enabled: bool = Field(default=False, alias="MULTI_PEER_ENABLED")
    multi_peer_allow_guest_sharing: bool = Field(
        default=True, alias="MULTI_PEER_GUEST_SHARING"
    )

    # RBAC / peer isolation (v1.2.0+)
    # When True AND peer_id is set, query_memories filters results to only
    # show memories belonging to the current peer (or with NULL peer_id).
    # When False (default), all memories are visible to all users — open by
    # default, lockdown is opt-in. peer_id is always written on inserts for
    # tagging when set, regardless of enforcement.
    peer_enforcement: bool = Field(default=False, alias="MEMINI_PEER_ENFORCEMENT")
    # The peer identifier for this instance/project. When enforcement is on,
    # only memories with this peer_id (or NULL peer_id) are returned.
    # When enforcement is off, this is used for tagging writes but does not
    # filter reads.
    peer_id: str | None = Field(default=None, alias="MEMINI_PEER_ID")

    # Phase 1 feature-activation: Memory Relationships auto-detection.
    # When True, add_memory runs a vector similarity search for
    # near-duplicates and auto-creates a SUPERSEDES relationship for any
    # match above auto_relationship_similarity_threshold. Default OFF to
    # preserve the v1.3.1 hard-reject-on-exact-hash behavior exactly.
    auto_relationship_detection: bool = Field(
        default=False, alias="AUTO_RELATIONSHIP_DETECTION"
    )
    auto_relationship_similarity_threshold: float = Field(
        default=0.95, alias="AUTO_RELATIONSHIP_SIMILARITY_THRESHOLD"
    )

    # Phase 4D: Dialectic settings
    dialectic_enabled: bool = Field(default=False, alias="DIALECTIC_ENABLED")
    dialectic_llm_provider: str = Field(
        default="ollama", alias="DIALECTIC_LLM_PROVIDER"
    )
    dialectic_llm_model: str = Field(default="llama3", alias="DIALECTIC_LLM_MODEL")
    dialectic_auto_threshold: float = Field(
        default=0.5, alias="DIALECTIC_AUTO_THRESHOLD"
    )

    # Thought Chains settings (Phase 5)
    thought_chains_enabled: bool = Field(default=False, alias="THOUGHT_CHAINS")

    # Tool-surface gating (v1.6.0): which MCP tool families are registered.
    # Comma-separated subset of: core,trust,kanban,session,chains,kg,
    # dialectic,peers,memory_ops,audit,ops. Unknown names log a warning and
    # are ignored; "core" is always registered regardless of this value.
    # Default ships ~16 tools instead of ~56 (~5-7K tokens saved/request).
    tool_groups: str = Field(
        default="core,trust,kanban,session",
        alias="MEMINI_TOOL_GROUPS",
        description=(
            "Comma-separated MCP tool groups to expose: core,trust,kanban,"
            "session,chains,kg,dialectic,peers,memory_ops,audit,ops"
        ),
    )

    # Phase 2.1: Input Validation settings
    max_memory_content_size: int = Field(
        default=102400,  # 100KB
        alias="MAX_MEMORY_CONTENT_SIZE",
        description="Maximum memory content size in bytes (default 100KB)",
    )
    rate_limit_per_minute: int = Field(
        default=100,
        alias="RATE_LIMIT_PER_MINUTE",
        description="Maximum add_memory calls per peer per minute (default 100)",
    )
    sanitize_content: bool = Field(
        default=True,
        alias="SANITIZE_CONTENT",
        description="Enable content sanitization on add_memory (default True)",
    )

    # Operation timeout (v1.5.6+)
    # Per-tool-call ceiling in milliseconds, applied by server.py via
    # asyncio.wait_for. Previously hard-coded at 30s; large-memory writes
    # on slow CPU embedding backends (e.g. BGE-M3 on long text) could
    # exceed it and surface as MCP -32001 timeouts. Raise for slow
    # hardware; the server clamps to [1000, 600000].
    operation_timeout_ms: int = Field(
        default=30000,
        alias="MEMINI_OPERATION_TIMEOUT_MS",
        description=(
            "Per-operation timeout in milliseconds (default 30000 = 30s). "
            "Clamped to [1000, 600000]."
        ),
    )

    # Env diagnostic (v1.5.5+)
    # When true, log resolved MEMINI_* env vars on startup. Off by default;
    # only enable to diagnose MCP env-injection issues (e.g. the opencode
    # 1.18.11 schema quirk that silently drops the `env` key inside
    # mcp.<server> blocks in favor of `environment`).
    debug_env: bool = Field(
        default=False,
        alias="MEMINI_DEBUG_ENV",
        description=(
            "When true, log resolved MEMINI_* env vars on startup. "
            "Off by default; only enable to diagnose MCP env-injection issues."
        ),
    )

    _json_config_loaded: bool = False

    @field_validator("workers", mode="before")
    @classmethod
    def _clamp_workers(cls, v: int | str) -> int:
        """Clamp workers to reasonable range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 64:
            return 64
        return val

    @field_validator("chunk_size", mode="before")
    @classmethod
    def _clamp_chunk_size(cls, v: int | str) -> int:
        """Clamp chunk size to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 64:
            return 64
        if val > 8192:
            return 8192
        return val

    @field_validator("chunk_overlap", mode="before")
    @classmethod
    def _clamp_chunk_overlap(cls, v: int | str) -> int:
        """Clamp chunk overlap to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 0:
            return 0
        return val

    @field_validator("batch_size", mode="before")
    @classmethod
    def _clamp_batch_size(cls, v: int | str) -> int:
        """Clamp batch size to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 256:
            return 256
        return val

    @field_validator("max_file_size", mode="before")
    @classmethod
    def _clamp_max_file_size(cls, v: int | str) -> int:
        """Clamp max file size to max 100MB."""
        val = int(v) if isinstance(v, str) else v
        max_allowed = 100 * 1024 * 1024
        if val > max_allowed:
            return max_allowed
        return val

    @field_validator(
        "trust_threshold_archive", "trust_threshold_promote", mode="before"
    )
    @classmethod
    def _clamp_trust_threshold(cls, v: float | str) -> float:
        """Clamp trust threshold to valid range."""
        val = float(v) if isinstance(v, str) else v
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator(
        "trust_delta_use",
        "trust_delta_ignore",
        "trust_delta_correct",
        "trust_delta_confirm",
        mode="before",
    )
    @classmethod
    def _clamp_trust_delta(cls, v: float | str) -> float:
        """Clamp trust delta to valid range."""
        val = float(v) if isinstance(v, str) else v
        if val < -1.0:
            return -1.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator("precompress_threshold", mode="before")
    @classmethod
    def _clamp_precompress_threshold(cls, v: float | str) -> float:
        """Clamp precompress threshold to valid range."""
        val = float(v) if isinstance(v, str) else v
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator("user_model_min_sessions", mode="before")
    @classmethod
    def _clamp_user_model_min_sessions(cls, v: int | str) -> int:
        """Clamp user model min sessions to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 500:
            return 500
        return val

    @field_validator("decay_half_life_days", mode="before")
    @classmethod
    def _clamp_decay_half_life(cls, v: int | str) -> int:
        """Clamp decay half-life to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 365:
            return 365
        return val

    @field_validator("consolidation_interval_hours", mode="before")
    @classmethod
    def _clamp_consolidation_interval(cls, v: int | str) -> int:
        """Clamp consolidation interval to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 8760:  # Max 1 year
            return 8760
        return val

    @field_validator("consolidation_similarity_threshold", mode="before")
    @classmethod
    def _clamp_consolidation_threshold(cls, v: float | str) -> float:
        """Clamp consolidation similarity threshold to valid range."""
        val = float(v) if isinstance(v, str) else v
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator("kg_inference_depth", mode="before")
    @classmethod
    def _clamp_kg_inference_depth(cls, v: int | str) -> int:
        """Clamp KG inference depth to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 10:
            return 10
        return val

    @field_validator("kg_max_results", mode="before")
    @classmethod
    def _clamp_kg_max_results(cls, v: int | str) -> int:
        """Clamp KG max results to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 1000:
            return 1000
        return val

    @field_validator("dialectic_auto_threshold", mode="before")
    @classmethod
    def _clamp_dialectic_auto_threshold(cls, v: float | str) -> float:
        """Clamp dialectic auto threshold to valid range."""
        val = float(v) if isinstance(v, str) else v
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator("auto_relationship_similarity_threshold", mode="before")
    @classmethod
    def _clamp_auto_relationship_similarity_threshold(cls, v: float | str) -> float:
        """Clamp auto-relationship similarity threshold to [0.0, 1.0]."""
        val = float(v) if isinstance(v, str) else v
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val

    @field_validator("db_sslmode", mode="before")
    @classmethod
    def _validate_db_sslmode(cls, v: str) -> str:
        """Validate PostgreSQL SSL mode against supported values."""
        val = v.lower().strip() if isinstance(v, str) else str(v).lower().strip()
        valid_modes = {
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        }
        if val not in valid_modes:
            raise ValueError(
                f"Invalid db_sslmode '{val}'. Must be one of: {', '.join(sorted(valid_modes))}"
            )
        return val

    @field_validator("max_memory_content_size", mode="before")
    @classmethod
    def _clamp_max_memory_content_size(cls, v: int | str) -> int:
        """Clamp max memory content size to valid range (1KB - 10MB)."""
        val = int(v) if isinstance(v, str) else v
        if val < 1024:
            return 1024  # Minimum 1KB
        if val > 10 * 1024 * 1024:
            return 10 * 1024 * 1024  # Maximum 10MB
        return val

    @field_validator("rate_limit_per_minute", mode="before")
    @classmethod
    def _clamp_rate_limit_per_minute(cls, v: int | str) -> int:
        """Clamp rate limit to valid range."""
        val = int(v) if isinstance(v, str) else v
        if val < 1:
            return 1
        if val > 10000:
            return 10000
        return val

    @field_validator("operation_timeout_ms", mode="before")
    @classmethod
    def _clamp_operation_timeout_ms(cls, v: int | str) -> int:
        """Clamp per-operation timeout to [1000, 600000] ms (v1.5.6)."""
        val = int(v) if isinstance(v, str) else v
        if val < 1000:
            return 1000
        if val > 600000:
            return 600000
        return val

    # =============================================================================
    # Dual-model RRF validators (v0.7.0)
    # =============================================================================

    @field_validator("embedding_mode", mode="before")
    @classmethod
    def _validate_embedding_mode(cls, v: str) -> str:
        """Validate embedding_mode is one of: cpu, auto, gpu."""
        val = str(v).lower().strip()
        if val not in {"cpu", "auto", "gpu"}:
            raise ValueError(
                f"Invalid embedding_mode '{val}'. Must be one of: cpu, auto, gpu"
            )
        return val

    @field_validator("rrf_k", mode="before")
    @classmethod
    def _clamp_rrf_k(cls, v: int | str) -> int:
        """Clamp RRF k constant to valid range [1, 1000]."""
        val = int(v) if isinstance(v, str) else v
        return max(1, min(1000, val))

    @field_validator("auto_extract_interval_seconds", mode="before")
    @classmethod
    def _clamp_auto_extract_interval(cls, v: int | str) -> int:
        """Clamp auto-extract interval to valid range [1, 3600] seconds."""
        val = int(v) if isinstance(v, str) else v
        return max(1, min(3600, val))

    # =============================================================================
    # v1.0.0: Backend selection validators (pgembed vs external Postgres)
    # =============================================================================

    @field_validator("vector_backend", mode="before")
    @classmethod
    def _validate_vector_backend(cls, v: str) -> str:
        """Validate vector_backend is one of: pgembed, postgres-external."""
        val = str(v).lower().strip()
        allowed = {"pgembed", "postgres-external"}
        if val not in allowed:
            raise ValueError(
                f"Invalid vector_backend '{val}'. Must be one of: pgembed, postgres-external"
            )
        return val

    @field_validator("fusion_mode", mode="before")
    @classmethod
    def _validate_fusion_mode(cls, v: str) -> str:
        """Validate fusion_mode is one of: none, rrf."""
        val = str(v).lower().strip()
        allowed = {"none", "rrf"}
        if val not in allowed:
            raise ValueError(f"Invalid fusion_mode '{val}'. Must be one of: none, rrf")
        return val

    # =============================================================================
    # Image search validators (v0.8.0)
    # =============================================================================

    @field_validator("image_clip_model", mode="before")
    @classmethod
    def _validate_image_clip_model(cls, v: str) -> str:
        """Validate CLIP model is one of the two supported models."""
        val = str(v).strip()
        if val not in {"clip-ViT-B-32", "clip-ViT-L-14"}:
            raise ValueError(
                f"Invalid image_clip_model '{val}'. "
                "Must be one of: clip-ViT-B-32, clip-ViT-L-14"
            )
        return val

    @field_validator("image_clip_device", mode="before")
    @classmethod
    def _validate_image_clip_device(cls, v: str) -> str:
        """Validate CLIP device is one of: auto, cpu, cuda."""
        val = str(v).lower().strip()
        if val not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"Invalid image_clip_device '{val}'. Must be one of: auto, cpu, cuda"
            )
        return val

    def model_post_init(self, _context: object) -> None:
        """Apply JSON config loading after initialization."""
        # Only load JSON config once per instance
        if not self._json_config_loaded:
            self._json_config_loaded = True
            self._load_json_config()
            self._finalize_validation()

    def _load_json_config(self) -> None:
        """Load configuration from JSON file if it exists.

        JSON config is at .opencode/memini-ai/config.json and is only loaded
        if not already set via environment variables.
        """
        config_path = self._find_json_config_path()
        if config_path is None or not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                json_config = json.load(f)
            # Apply JSON config values that aren't set by environment variables
            for key, value in json_config.items():
                if key not in self.model_fields_set and key in self.model_fields:
                    object.__setattr__(self, key, value)
        except (json.JSONDecodeError, OSError):
            # Silently skip invalid JSON config - defaults are sufficient
            pass

    def _find_json_config_path(self) -> Path | None:
        """Find JSON config file path by traversing up from current directory."""
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            config_path = parent / ".opencode" / "memini-ai" / "config.json"
            if config_path.exists():
                return config_path
        return None

    def _finalize_validation(self) -> None:
        """Final validation and clamping that depends on multiple fields."""
        # Clamp chunk_overlap based on chunk_size
        if self.chunk_overlap > self.chunk_size:
            object.__setattr__(self, "chunk_overlap", self.chunk_size // 2)

    @property
    def effective_project_id(self) -> str:
        """Get effective project ID, generating from directory name if not set."""
        if self.project_id:
            return self.project_id
        # Generate from directory name, sanitized
        cwd = Path.cwd()
        return _sanitize_project_id(cwd.name)


def _sanitize_project_id(name: str) -> str:
    """Sanitize a directory name into a valid project ID."""
    # Remove non-alphanumeric characters except hyphens/underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    # Collapse multiple hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    # Default if empty
    if not sanitized:
        return "default-project"
    return sanitized


# Module-level singleton config instance
_config: MeminiConfig | None = None


def get_config() -> MeminiConfig:
    """Get the global config instance, creating if necessary."""
    global _config
    if _config is None:
        _config = MeminiConfig()
    return _config
