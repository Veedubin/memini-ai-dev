"""Tests for configuration management."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from memini_ai.config import MeminiConfig, _sanitize_project_id, get_config


class TestMeminiConfigDefaults:
    """Tests for default configuration values."""

    def test_model_settings_defaults(self) -> None:
        """Should have correct defaults for model settings."""
        config = MeminiConfig()
        assert config.precision == "fp16"
        assert config.device == "auto"
        assert config.use_gpu is False
        assert config.embedding_dim == 1024
        assert config.batch_size == 32
        assert config.eager_load is False

    def test_database_settings_defaults(self) -> None:
        """Should have correct defaults for database settings."""
        config = MeminiConfig()
        assert config.qdrant_url == "http://localhost:6333"
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
        assert config.qdrant_max_retries == 3
        assert config.qdrant_retry_delay_ms == 1000
        # workers depends on CPU count, so just check it's positive
        assert config.workers >= 1


class TestMeminiConfigEnvVars:
    """Tests for environment variable override."""

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables should override defaults."""
        monkeypatch.setenv("MEMINI_QDRANT_URL", "http://custom:6333")
        monkeypatch.setenv("MEMINI_EMBEDDING_DIM", "384")
        monkeypatch.setenv("MEMINI_LOG_LEVEL", "debug")

        config = MeminiConfig()
        assert config.qdrant_url == "http://custom:6333"
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
        monkeypatch.setenv("MEMINI_EXCLUDE_PATTERNS", '["venv", "__pycache__", "build"]')
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
                        "qdrant_url": "http://json-config:6333",
                        "embedding_dim": 512,
                        "chunk_size": 256,
                    }
                )
            )

            # Change to temp directory so config is found
            monkeypatch.chdir(tmpdir)
            config = MeminiConfig()
            assert config.qdrant_url == "http://json-config:6333"
            assert config.embedding_dim == 512
            assert config.chunk_size == 256

    def test_json_config_not_found_skipped(self) -> None:
        """Should skip if JSON config file doesn't exist."""
        config = MeminiConfig()
        # Should use defaults if no JSON config
        assert config.qdrant_url == "http://localhost:6333"

    def test_json_config_invalid_json_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            config_path.write_text(json.dumps({"qdrant_url": "http://json-only"}))

            monkeypatch.chdir(tmpdir)
            monkeypatch.setenv("MEMINI_QDRANT_URL", "http://env-override")
            config = MeminiConfig()
            assert config.qdrant_url == "http://env-override"


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

    def test_qdrant_max_retries_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """qdrant_max_retries should be clamped between 1 and 10."""
        monkeypatch.setenv("MEMINI_QDRANT_MAX_RETRIES", "0")
        config = MeminiConfig()
        assert config.qdrant_max_retries == 1

        monkeypatch.setenv("MEMINI_QDRANT_MAX_RETRIES", "20")
        config = MeminiConfig()
        assert config.qdrant_max_retries == 10

    def test_qdrant_retry_delay_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """qdrant_retry_delay_ms should be clamped between 100 and 30000."""
        monkeypatch.setenv("MEMINI_QDRANT_RETRY_DELAY_MS", "50")
        config = MeminiConfig()
        assert config.qdrant_retry_delay_ms == 100

        monkeypatch.setenv("MEMINI_QDRANT_RETRY_DELAY_MS", "50000")
        config = MeminiConfig()
        assert config.qdrant_retry_delay_ms == 30000

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
