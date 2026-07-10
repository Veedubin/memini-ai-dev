"""Tests for configuration management."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from memini_ai.config import MeminiConfig, _sanitize_project_id, get_config

# ---------------------------------------------------------------------------
# Env isolation fixture
# ---------------------------------------------------------------------------
# Several tests in this file assert that MeminiConfig defaults are None/False.
# When the shell (or the project .env file) has MEMINI_PROJECT_ID,
# THOUGHT_CHAINS, or other feature toggles set, those leak into the test
# process via pydantic-settings' env_file + env_prefix mechanism and break
# the default-value assertions.
#
# The fixture chdir's to a fresh temp directory (no .env, no
# .opencode/memini-ai/config.json) and deletes all MEMINI_/THOUGHT_CHAINS/
# related env vars so each test starts from a clean config baseline.
# Tests that explicitly set env vars (e.g. via monkeypatch.setenv) still
# work because monkeypatch is session-scoped per-test.
_ENV_VARS_TO_ISOLATE = [
    "MEMINI_PROJECT_ID",
    "THOUGHT_CHAINS",
    "MEMINI_TRUST_ENGINE",
    "MEMINI_KG_ENABLED",
    "MEMINI_TIERED_LOADING",
    "MEMINI_DIALECTIC_ENABLED",
    "MEMINI_MEMORY_GRAPH",
    "MEMINI_MULTI_PEER_ENABLED",
    "MEMINI_USER_MODELING",
    "MEMINI_DECAY_ENABLED",
    "MEMINI_AUTO_EXTRACT",
    "MEMINI_PRECOMPRESS",
    "MEMINI_DB_URL",
    "MEMINI_EMBEDDING_DIM",
    "MEMINI_EMBEDDING_MODE",
    "MEMINI_ELEVATE_ENABLED",
    "MEMINI_RRF_K",
    "DB_SSLMODE",
    "LLM_URL",
]


@pytest.fixture(autouse=True)
def _isolate_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Isolate config tests from shell/env leaks.

    Chdir to a temp dir (no .env, no JSON config) and delete all
    memini-related env vars so default-value assertions are reliable.
    """
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    # Also clear any other MEMINI_ vars that might leak from the shell
    for key in list(os.environ):
        if key.startswith("MEMINI_") or key == "THOUGHT_CHAINS":
            monkeypatch.delenv(key, raising=False)


class TestMeminiConfigDefaults:
    """Tests for default configuration values."""

    def test_model_settings_defaults(self) -> None:
        """Should have correct defaults for model settings."""
        config = MeminiConfig()
        assert config.precision == "fp16"
        assert config.device == "auto"
        assert config.use_gpu is False
        # v0.7.0: embedding_dim default is 384 (was 1024 in v0.6.x).
        # The default now matches the schema, which is vector(384).
        # The dual-model RRF path uses 1024-dim via the memories_1024
        # sidecar table.
        assert config.embedding_dim == 384
        assert config.batch_size == 32
        assert config.eager_load is False

    def test_database_settings_defaults(self) -> None:
        """Should have correct defaults for database settings."""
        config = MeminiConfig()
        assert config.table_name == "memories"
        assert config.project_id is None
        assert config.query_collections is None

    def test_indexer_settings_defaults(self) -> None:
        """Should have correct defaults for indexer settings."""
        config = MeminiConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.exclude_patterns == ["node_modules", ".git", "dist"]

    def test_logging_default(self) -> None:
        """Default log level should be 'info'."""
        config = MeminiConfig()
        assert config.log_level == "info"

    def test_performance_defaults(self) -> None:
        """Should have correct defaults for performance settings."""
        config = MeminiConfig()
        # workers depends on CPU count, so just check it's positive
        assert config.workers >= 1


class TestMeminiConfigEnvVars:
    """Tests for environment variable override."""

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables should override defaults."""
        monkeypatch.setenv("MEMINI_PRECISION", "fp32")
        monkeypatch.setenv("MEMINI_EMBEDDING_DIM", "384")
        monkeypatch.setenv("MEMINI_LOG_LEVEL", "debug")

        config = MeminiConfig()
        assert config.precision == "fp32"
        assert config.embedding_dim == 384
        assert config.log_level == "debug"

    def test_env_var_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use MEMINI_ prefix for all environment variables."""
        monkeypatch.setenv("MEMINI_TABLE_NAME", "custom-table")
        config = MeminiConfig()
        assert config.table_name == "custom-table"

    def test_env_var_int_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Integer env vars should be properly converted."""
        monkeypatch.setenv("MEMINI_CHUNK_SIZE", "1024")
        config = MeminiConfig()
        assert config.chunk_size == 1024

    def test_env_var_list_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """List env vars should be properly converted via JSON."""
        monkeypatch.setenv(
            "MEMINI_EXCLUDE_PATTERNS", '["venv", "__pycache__", "build"]'
        )
        config = MeminiConfig()
        assert config.exclude_patterns == ["venv", "__pycache__", "build"]


class TestMeminiConfigJsonConfig:
    """Tests for JSON config file loading."""

    def test_json_config_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should load values from JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".opencode" / "memini-ai" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "embedding_dim": 512,
                        "chunk_size": 256,
                    }
                )
            )

            # Change to temp directory so config is found
            monkeypatch.chdir(tmpdir)
            config = MeminiConfig()
            assert config.embedding_dim == 512
            assert config.chunk_size == 256

    def test_json_config_not_found_skipped(self) -> None:
        """Should skip if JSON config file doesn't exist."""
        config = MeminiConfig()
        # Should use defaults if no JSON config
        assert config.table_name == "memories"

    def test_json_config_invalid_json_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should skip invalid JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".opencode" / "memini-ai" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("invalid json {")

            monkeypatch.chdir(tmpdir)
            config = MeminiConfig()
            # Should use defaults
            assert config.chunk_size == 512

    def test_env_var_overrides_json_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables should take priority over JSON config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".opencode" / "memini-ai" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"embedding_dim": 512}))

            monkeypatch.chdir(tmpdir)
            monkeypatch.setenv("MEMINI_EMBEDDING_DIM", "384")
            config = MeminiConfig()
            assert config.embedding_dim == 384


class TestMeminiConfigValidation:
    """Tests for configuration validation and clamping."""

    def test_workers_clamped_to_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Workers should be clamped between 1 and 64."""
        monkeypatch.setenv("MEMINI_WORKERS", "0")
        config = MeminiConfig()
        assert config.workers == 1

        monkeypatch.setenv("MEMINI_WORKERS", "100")
        config = MeminiConfig()
        assert config.workers == 64

        monkeypatch.setenv("MEMINI_WORKERS", "16")
        config = MeminiConfig()
        assert config.workers == 16

    def test_chunk_size_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chunk_size should be clamped between 64 and 8192."""
        monkeypatch.setenv("MEMINI_CHUNK_SIZE", "32")
        config = MeminiConfig()
        assert config.chunk_size == 64

        monkeypatch.setenv("MEMINI_CHUNK_SIZE", "16384")
        config = MeminiConfig()
        assert config.chunk_size == 8192

    def test_chunk_overlap_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chunk_overlap should be clamped to valid range."""
        monkeypatch.setenv("MEMINI_CHUNK_OVERLAP", "-10")
        config = MeminiConfig()
        assert config.chunk_overlap == 0

        monkeypatch.setenv("MEMINI_CHUNK_OVERLAP", "1000")
        config = MeminiConfig()
        # Should be clamped to chunk_size // 2
        assert config.chunk_overlap == 256

    def test_batch_size_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_size should be clamped between 1 and 256."""
        monkeypatch.setenv("MEMINI_BATCH_SIZE", "0")
        config = MeminiConfig()
        assert config.batch_size == 1

        monkeypatch.setenv("MEMINI_BATCH_SIZE", "500")
        config = MeminiConfig()
        assert config.batch_size == 256

    def test_max_file_size_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_file_size should be clamped to max 100MB."""
        monkeypatch.setenv("MEMINI_MAX_FILE_SIZE", str(200 * 1024 * 1024))
        config = MeminiConfig()
        assert config.max_file_size == 100 * 1024 * 1024


class TestEffectiveProjectId:
    """Tests for effective_project_id property."""

    def test_returns_config_project_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return configured project_id if set."""
        monkeypatch.setenv("MEMINI_PROJECT_ID", "my-project")
        config = MeminiConfig()
        assert config.effective_project_id == "my-project"

    def test_generates_from_directory_name(self) -> None:
        """Should generate from directory name if project_id not set."""
        config = MeminiConfig()
        # Should use current directory name
        cwd_name = os.getcwd().split("/")[-1]
        assert config.effective_project_id == cwd_name


class TestSanitizeProjectId:
    """Tests for project ID sanitization function."""

    def test_keeps_valid_characters(self) -> None:
        """Should keep alphanumeric, hyphens, underscores."""
        assert _sanitize_project_id("my-project_123") == "my-project_123"

    def test_replaces_invalid_characters(self) -> None:
        """Should replace invalid characters with hyphens."""
        assert _sanitize_project_id("my project!") == "my-project"
        assert _sanitize_project_id("my@project") == "my-project"

    def test_collapses_multiple_hyphens(self) -> None:
        """Should collapse multiple hyphens into one."""
        assert _sanitize_project_id("my---project") == "my-project"

    def test_removes_leading_trailing_hyphens(self) -> None:
        """Should remove leading and trailing hyphens."""
        assert _sanitize_project_id("-my-project-") == "my-project"

    def test_returns_default_for_empty(self) -> None:
        """Should return 'default-project' for empty/invalid names."""
        assert _sanitize_project_id("---") == "default-project"
        assert _sanitize_project_id("") == "default-project"


class TestGetConfig:
    """Tests for the get_config singleton function."""

    def test_returns_memini_config_instance(self) -> None:
        """Should return a MeminiConfig instance."""
        config = get_config()
        assert isinstance(config, MeminiConfig)

    def test_returns_same_instance(self) -> None:
        """Should return the same instance on subsequent calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


# ---------------------------------------------------------------------------
# v0.7.7: Embedding policy config fields
# ---------------------------------------------------------------------------


class TestEmbeddingPolicyConfig:
    """Tests for strict_embedding_dim and auto_detect_model config fields."""

    def test_strict_embedding_dim_default_false(self) -> None:
        """strict_embedding_dim should default to False (lenient mode)."""
        config = MeminiConfig()
        assert config.strict_embedding_dim is False

    def test_auto_detect_model_default_true(self) -> None:
        """auto_detect_model should default to True (auto-detect new deployments)."""
        config = MeminiConfig()
        assert config.auto_detect_model is True

    def test_strict_embedding_dim_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEMINI_STRICT_EMBEDDING_DIM=true should set strict_embedding_dim=True."""
        monkeypatch.setenv("MEMINI_STRICT_EMBEDDING_DIM", "true")
        config = MeminiConfig()
        assert config.strict_embedding_dim is True

    def test_auto_detect_model_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEMINI_AUTO_DETECT_MODEL=false should set auto_detect_model=False."""
        monkeypatch.setenv("MEMINI_AUTO_DETECT_MODEL", "false")
        config = MeminiConfig()
        assert config.auto_detect_model is False
