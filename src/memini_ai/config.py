"""Configuration management using pydantic-settings."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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
    embedding_dim: int = 1024
    batch_size: int = 32
    eager_load: bool = False

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
        default="prefer",
        alias="DB_SSLMODE",
        description="PostgreSQL SSL mode (disable, allow, prefer, require, verify-ca, verify-full)",
    )
    db_sslrootcert: str | None = Field(
        default=None,
        alias="DB_SSLROOTCERT",
        description="Path to CA certificate for SSL server verification",
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
    llm_url: str = Field(default="http://localhost:11434/api/generate", alias="LLM_URL")
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")

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
