"""Tests for the memini-ai-dev installer module.

Covers: system deps, config paths, opencode.json builders, file write with
SHA-256 idempotency, state file, prompt flows, and CLI argument parsing.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import pytest

from memini_ai.installer import (
    BackupResult,
    InstallConfig,
    SysDepsResult,
    build_homedir_opencode_json,
    build_project_opencode_json,
    check_system_deps,
    compute_file_sha256,
    get_homedir_config_path,
    get_project_config_path,
    run_prompts,
    write_config_with_backup,
    write_state_file,
)

# ── check_system_deps ─────────────────────────────────────────────────────────


def test_check_system_deps_uv_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """uv on PATH → all_present=True."""
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    result = check_system_deps()
    assert isinstance(result, SysDepsResult)
    assert result.all_present is True
    uv_dep = [d for d in result.deps if d.name == "uv"]
    assert len(uv_dep) == 1
    assert uv_dep[0].present is True


def test_check_system_deps_uv_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """uv not on PATH → all_present=False."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = check_system_deps()
    assert isinstance(result, SysDepsResult)
    assert result.all_present is False
    uv_dep = [d for d in result.deps if d.name == "uv"]
    assert len(uv_dep) == 1
    assert uv_dep[0].present is False
    assert uv_dep[0].install_command is not None


# ── get_homedir_config_path ───────────────────────────────────────────────────


def test_get_homedir_config_path_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux → path ends with .config/opencode."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "home", lambda: Path("/home/user"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    path = get_homedir_config_path()
    assert str(path).endswith(".config/opencode")


def test_get_homedir_config_path_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS → path contains Library/Application Support/opencode."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(Path, "home", lambda: Path("/Users/user"))
    path = get_homedir_config_path()
    assert "Library" in str(path)
    assert "Application Support" in str(path)
    assert "opencode" in str(path)


# ── get_project_config_path ───────────────────────────────────────────────────


def test_get_project_config_path(tmp_path: Path) -> None:
    """Pass a tmp_path as cwd → returns {cwd}/.opencode."""
    path = get_project_config_path(cwd=tmp_path)
    assert path == tmp_path / ".opencode"


# ── build_homedir_opencode_json ───────────────────────────────────────────────


def _make_config(**overrides: object) -> InstallConfig:
    """Helper: create an InstallConfig with overrides."""
    defaults: dict[str, object] = {
        "backend": "pgembed",
        "team_host": None,
        "team_port": None,
        "team_database": None,
        "team_user": None,
        "team_password": None,
        "embedding": "auto",
        "image_search": True,
        "trust_engine": True,
        "knowledge_graph": True,
        "tiered_loading": True,
        "auto_extract": True,
        "precompress": True,
        "decay": True,
        "dialectic": True,
        "thought_chains": True,
        "ollama_api_key": None,
    }
    defaults.update(overrides)
    return InstallConfig(**defaults)  # type: ignore[arg-type]


def test_build_homedir_opencode_json_pgembed() -> None:
    """pgembed defaults → enabled, vector_backend=pgembed, dim=384, provider block."""
    config = _make_config()
    result = build_homedir_opencode_json(config)

    # Top-level keys
    assert "$schema" in result
    assert "autoupdate" in result
    assert "provider" in result
    assert "mcp" in result

    # MCP block
    mcp = result["mcp"]
    assert isinstance(mcp, dict)
    assert "memini-ai-dev" in mcp
    dev = mcp["memini-ai-dev"]
    assert isinstance(dev, dict)
    assert dev["enabled"] is True
    assert dev["type"] == "local"

    # Env vars
    env = dev["env"]
    assert isinstance(env, dict)
    assert env["MEMINI_VECTOR_BACKEND"] == "pgembed"
    assert env["MEMINI_EMBEDDING_DIM"] == "384"

    # Provider block
    provider = result["provider"]
    assert isinstance(provider, dict)
    assert "ollama" in provider
    ollama = provider["ollama"]
    assert isinstance(ollama, dict)
    assert ollama["name"] == "ollama"
    assert ollama["api"] == "https://ollama.com/v1"
    assert "models" in ollama


def test_build_homedir_opencode_json_team() -> None:
    """team backend → vector_backend=postgres-external, MEMINI_DB_URL starts with postgresql://."""
    config = _make_config(
        backend="team",
        team_host="pg.example.com",
        team_port="5432",
        team_database="memini",
        team_user="alice",
        team_password="s3cret",
    )
    result = build_homedir_opencode_json(config)
    mcp = result["mcp"]
    assert isinstance(mcp, dict)
    dev = mcp["memini-ai-dev"]
    assert isinstance(dev, dict)
    env = dev["env"]
    assert isinstance(env, dict)
    assert env["MEMINI_VECTOR_BACKEND"] == "postgres-external"
    assert env["MEMINI_DB_URL"].startswith("postgresql://")
    assert "alice" in env["MEMINI_DB_URL"]
    assert "s3cret" in env["MEMINI_DB_URL"]
    assert "pg.example.com" in env["MEMINI_DB_URL"]


def test_build_homedir_opencode_json_gpu() -> None:
    """GPU embedding → dim=1024, model_name=BAAI/bge-m3."""
    config = _make_config(embedding="gpu")
    result = build_homedir_opencode_json(config)
    mcp = result["mcp"]
    assert isinstance(mcp, dict)
    dev = mcp["memini-ai-dev"]
    assert isinstance(dev, dict)
    env = dev["env"]
    assert isinstance(env, dict)
    assert env["MEMINI_EMBEDDING_DIM"] == "1024"
    assert env["MEMINI_MODEL_NAME"] == "BAAI/bge-m3"


# ── build_project_opencode_json ───────────────────────────────────────────────


def test_build_project_opencode_json() -> None:
    """Project config: NO provider, NO $schema, NO autoupdate — just mcp block."""
    config = _make_config()
    result = build_project_opencode_json(config)

    assert "mcp" in result
    assert "provider" not in result
    assert "$schema" not in result
    assert "autoupdate" not in result

    mcp = result["mcp"]
    assert isinstance(mcp, dict)
    assert "memini-ai-dev" in mcp


# ── build_homedir_opencode_json — image_search disabled ────────────────────────


def test_build_opencode_json_image_search_disabled() -> None:
    """image_search=False → MEMINI_IMAGE_SEARCH_ENABLED == 'false'."""
    config = _make_config(image_search=False)
    result = build_homedir_opencode_json(config)
    mcp = result["mcp"]
    assert isinstance(mcp, dict)
    dev = mcp["memini-ai-dev"]
    assert isinstance(dev, dict)
    env = dev["env"]
    assert isinstance(env, dict)
    assert env["MEMINI_IMAGE_SEARCH_ENABLED"] == "false"


# ── write_config_with_backup ──────────────────────────────────────────────────


def test_write_config_with_backup_new_file(tmp_path: Path) -> None:
    """Write to empty tmp dir → file created, no backup."""
    content = json.dumps({"hello": "world"}, indent=2)
    result = write_config_with_backup(
        config_dir=tmp_path,
        filename="opencode.json",
        content=content,
        force=False,
        dry_run=False,
    )
    assert isinstance(result, BackupResult)
    assert result.backed_up is False
    assert result.backup_path is None
    assert (tmp_path / "opencode.json").exists()
    assert (tmp_path / "opencode.json").read_text(encoding="utf-8") == content


def test_write_config_with_backup_idempotent(tmp_path: Path) -> None:
    """Write same content twice → second call is a no-op (backup_path=None)."""
    content = json.dumps({"hello": "world"}, indent=2)
    # First write
    write_config_with_backup(
        config_dir=tmp_path,
        filename="opencode.json",
        content=content,
        force=False,
        dry_run=False,
    )
    # Second write — same content
    result = write_config_with_backup(
        config_dir=tmp_path,
        filename="opencode.json",
        content=content,
        force=False,
        dry_run=False,
    )
    assert result.backed_up is False
    assert result.backup_path is None


def test_write_config_with_backup_different(tmp_path: Path) -> None:
    """Write different content → backup created with timestamp."""
    content_v1 = json.dumps({"version": 1}, indent=2)
    content_v2 = json.dumps({"version": 2}, indent=2)

    # First write
    write_config_with_backup(
        config_dir=tmp_path,
        filename="opencode.json",
        content=content_v1,
        force=False,
        dry_run=False,
    )

    # Second write — different content, force=True
    result = write_config_with_backup(
        config_dir=tmp_path,
        filename="opencode.json",
        content=content_v2,
        force=True,
        dry_run=False,
    )
    assert result.backed_up is True
    assert result.backup_path is not None
    # Backup file should exist
    backup = Path(result.backup_path)
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == content_v1
    # Original should now have v2
    assert (tmp_path / "opencode.json").read_text(encoding="utf-8") == content_v2


# ── write_state_file ─────────────────────────────────────────────────────────


def test_write_state_file(tmp_path: Path) -> None:
    """Write state file → read back and check JSON structure."""
    write_state_file(
        config_dir=tmp_path,
        version="1.0.0",
        mode="homedir",
        files={"opencode.json": "abc123"},
    )
    state_path = tmp_path / ".memini-ai-state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["installed_version"] == "1.0.0"
    assert data["mode"] == "homedir"
    assert data["files"] == {"opencode.json": "abc123"}
    assert "installed_at" in data


# ── run_prompts ───────────────────────────────────────────────────────────────


def _make_namespace(**kwargs: object) -> object:
    """Create a simple argparse.Namespace-like object."""
    from types import SimpleNamespace

    defaults: dict[str, object] = {
        "yes": False,
        "embedded": False,
        "team": False,
        "cpu_embed": False,
        "auto_embed": False,
        "gpu_embed": False,
        "no_image_search": False,
        "no_features": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_run_prompts_yes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--yes flag → all defaults (pgembed, auto, all features on)."""
    # input() should NOT be called — if it is, the test will fail
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (_ for _ in ()).throw(
            AssertionError("input() should not be called with --yes")
        ),
    )
    flags = _make_namespace(yes=True)
    config = run_prompts(flags)
    assert config.backend == "pgembed"
    assert config.embedding == "auto"
    assert config.image_search is True
    assert config.trust_engine is True
    assert config.knowledge_graph is True
    assert config.tiered_loading is True
    assert config.auto_extract is True
    assert config.precompress is True
    assert config.decay is True
    assert config.dialectic is True
    assert config.thought_chains is True


def test_run_prompts_embedded_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--embedded flag -> backend is pgembed without prompting for backend."""
    calls: list[str] = []

    def _mock_input(prompt: str = "") -> str:
        calls.append(prompt)
        # Embedding choice prompt: return "2" (auto)
        if "Enter 1, 2, or 3" in prompt:
            return "2"
        # Yes/no prompts: return "" (accept default = True)
        if "[Y/n]" in prompt or "[y/N]" in prompt:
            return ""
        # API key prompt: return "" (skip)
        return ""

    monkeypatch.setattr("builtins.input", _mock_input)
    flags = _make_namespace(embedded=True)
    config = run_prompts(flags)
    assert config.backend == "pgembed"
    # Backend prompt should NOT have been called
    assert not any("store memories" in c for c in calls), (
        f"Backend prompt was called: {calls}"
    )


def test_run_prompts_no_features_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-features → all features False."""
    # We need to mock input() for the embedding prompt (which is still asked)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    flags = _make_namespace(no_features=True)
    config = run_prompts(flags)
    assert config.image_search is False
    assert config.trust_engine is False
    assert config.knowledge_graph is False
    assert config.tiered_loading is False
    assert config.auto_extract is False
    assert config.precompress is False
    assert config.decay is False
    assert config.dialectic is False
    assert config.thought_chains is False


# ── CLI argument parsing ──────────────────────────────────────────────────────


def test_cli_init_no_flags_backward_compat() -> None:
    """Parse ['init'] → homedir=False, project=False (backward compat)."""
    from memini_ai.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["init"])
    assert args.command == "init"
    assert args.homedir is False
    assert args.project is False


def test_cli_init_homedir_flag() -> None:
    """Parse ['init', '--homedir'] → args.homedir == True."""
    from memini_ai.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["init", "--homedir"])
    assert args.command == "init"
    assert args.homedir is True


def test_cli_update_subcommand_exists() -> None:
    """Parse ['update'] → args.command == 'update'."""
    from memini_ai.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["update"])
    assert args.command == "update"


# ── compute_file_sha256 ────────────────────────────────────────────────────────


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 computation on a known file."""
    import hashlib

    content = b"hello world"
    path = tmp_path / "test.txt"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(path) == expected
