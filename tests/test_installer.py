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
    ContainerRuntime,
    ContainerRuntimeResult,
    InstallConfig,
    SysDepsResult,
    build_admin_dsn,
    build_homedir_opencode_json,
    build_project_opencode_json,
    check_container_runtime,
    check_system_deps,
    compute_file_sha256,
    generate_password,
    get_homedir_config_path,
    get_project_config_path,
    read_admin_credentials,
    run_prompts,
    sanitize_project_name,
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
        "team_sslmode": "prefer",
        "team_new_db": False,
        "team_admin_user": None,
        "team_admin_password": None,
        "project_user": None,
        "project_password": None,
        "project_name": None,
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
        team_sslmode="require",
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
    # SSL mode appended (unless "disable").
    assert "?sslmode=require" in env["MEMINI_DB_URL"]


def test_build_homedir_opencode_json_team_ssl_disable() -> None:
    """team backend with sslmode=disable → no ?sslmode= in DSN."""
    config = _make_config(
        backend="team",
        team_host="pg.example.com",
        team_port="5432",
        team_database="memini",
        team_user="alice",
        team_password="s3cret",
        team_sslmode="disable",
    )
    result = build_homedir_opencode_json(config)
    env = result["mcp"]["memini-ai-dev"]["env"]
    assert isinstance(env, dict)
    assert "?sslmode=" not in env["MEMINI_DB_URL"]


def test_build_homedir_opencode_json_team_rbac() -> None:
    """team backend with project_user set → {env:...} references (no inline creds)."""
    config = _make_config(
        backend="team",
        team_host="localhost",
        team_port="5432",
        team_database="memini",
        project_user="foo-bar-123",
        project_password="generated-pw",
        project_name="foo-bar-123",
    )
    result = build_homedir_opencode_json(config)
    env = result["mcp"]["memini-ai-dev"]["env"]
    assert isinstance(env, dict)
    # No inline credentials in opencode.json — secrets stay in .env.
    assert "generated-pw" not in env["MEMINI_DB_URL"]
    assert env["MEMINI_DB_URL"] == (
        "postgresql://{env:MEMINI_PROJECT_USER}:{env:MEMINI_PROJECT_PASSWORD}"
        "@{env:MEMINI_DB_HOST}:{env:MEMINI_DB_PORT}/{env:MEMINI_DB_NAME}"
        "?sslmode={env:MEMINI_DB_SSLMODE}"
    )
    assert env["MEMINI_PEER_ID"] == "foo-bar-123"


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
    """--no-features → all features False (pgembed backend, no team prompts)."""

    # Mock input: "1" for backend (pgembed), "2" for embedding (auto), "" for yes/no.
    def _mock_input(prompt: str = "") -> str:
        if "Enter 1 or 2" in prompt:
            return "1"  # pgembed backend — avoid team flow entirely
        if "Enter 1, 2, or 3" in prompt:
            return "2"  # auto embedding
        if "[Y/n]" in prompt or "[y/N]" in prompt:
            return ""  # accept default
        return ""

    monkeypatch.setattr("builtins.input", _mock_input)
    flags = _make_namespace(no_features=True)
    config = run_prompts(flags)
    assert config.backend == "pgembed"
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


# ── sanitize_project_name ─────────────────────────────────────────────────────


def test_sanitize_project_name_basic() -> None:
    """Underscores → hyphens, non-alnum stripped, collapses + trims."""
    assert sanitize_project_name("foo_bar_123") == "foo-bar-123"


def test_sanitize_project_name_special_chars() -> None:
    """Special chars stripped, multiple hyphens collapsed."""
    # "My Project!" → no underscores, space + ! stripped → "MyProject"
    assert sanitize_project_name("My Project!") == "MyProject"
    assert sanitize_project_name("a__b---c") == "a-b-c"
    # Explicit hyphens in input are preserved.
    assert sanitize_project_name("My-Project!") == "My-Project"


def test_sanitize_project_name_empty() -> None:
    """Empty after sanitization → default 'memini-project'."""
    assert sanitize_project_name("___") == "memini-project"
    assert sanitize_project_name("!!!") == "memini-project"
    assert sanitize_project_name("") == "memini-project"


def test_sanitize_project_name_truncates_63() -> None:
    """Names longer than 63 chars are truncated."""
    long_name = "a" * 100
    result = sanitize_project_name(long_name)
    assert len(result) == 63
    assert result == "a" * 63


def test_sanitize_project_name_strips_hyphens() -> None:
    """Leading/trailing hyphens stripped."""
    assert sanitize_project_name("-foo-bar-") == "foo-bar"


# ── generate_password ─────────────────────────────────────────────────────────


def test_generate_password_length_and_charset() -> None:
    """Generated password is 43 chars, URL-safe base64 charset."""
    pw = generate_password()
    assert len(pw) == 43
    # URL-safe base64 charset: A-Z a-z 0-9 - _
    import re

    assert re.fullmatch(r"[A-Za-z0-9_-]+", pw) is not None


def test_generate_password_unique() -> None:
    """Two calls produce different passwords (extremely high entropy)."""
    pw1 = generate_password()
    pw2 = generate_password()
    assert pw1 != pw2


# ── check_container_runtime ───────────────────────────────────────────────────


def test_check_container_runtime_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No runtimes on PATH → has_any=False, all present=False."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def _run_quiet_none(cmd: list[str], timeout: float = 5.0) -> bool:
        return False

    import memini_ai.installer as inst

    monkeypatch.setattr(inst, "_run_quiet", _run_quiet_none)
    result = check_container_runtime()
    assert isinstance(result, ContainerRuntimeResult)
    assert result.has_any is False
    assert len(result.runtimes) == 3
    for rt in result.runtimes:
        assert isinstance(rt, ContainerRuntime)
        assert rt.present is False
        assert rt.compose_command == ""


def test_check_container_runtime_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """docker on PATH + `docker compose version` works -> present, compose_command set."""
    import memini_ai.installer as inst

    def _which(name: str) -> str | None:
        return "/usr/bin/docker" if name == "docker" else None

    def _run_quiet(cmd: list[str], timeout: float = 5.0) -> bool:
        return cmd == ["docker", "compose", "version"]

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(inst, "_run_quiet", _run_quiet)
    result = check_container_runtime()
    assert isinstance(result, ContainerRuntimeResult)
    assert result.has_any is True
    docker = [rt for rt in result.runtimes if rt.name == "docker"]
    assert len(docker) == 1
    assert docker[0].present is True
    assert docker[0].compose_command == "docker compose"
    podman = [rt for rt in result.runtimes if rt.name == "podman"]
    assert podman[0].present is False


def test_check_container_runtime_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    """podman on PATH + `podman compose version` works → present, compose_command set."""
    import memini_ai.installer as inst

    def _which(name: str) -> str | None:
        return "/usr/bin/podman" if name == "podman" else None

    def _run_quiet(cmd: list[str], timeout: float = 5.0) -> bool:
        return cmd == ["podman", "compose", "version"]

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(inst, "_run_quiet", _run_quiet)
    result = check_container_runtime()
    assert isinstance(result, ContainerRuntimeResult)
    assert result.has_any is True
    podman = [rt for rt in result.runtimes if rt.name == "podman"]
    assert len(podman) == 1
    assert podman[0].present is True
    assert podman[0].compose_command == "podman compose"
    docker = [rt for rt in result.runtimes if rt.name == "docker"]
    assert docker[0].present is False


# ── check_system_deps (team backend) ─────────────────────────────────────────


def test_check_system_deps_team_no_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """team backend + no container runtime → all_present=False (soft fail)."""
    import memini_ai.installer as inst

    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )

    def _run_quiet_none(cmd: list[str], timeout: float = 5.0) -> bool:
        return False

    monkeypatch.setattr(inst, "_run_quiet", _run_quiet_none)
    result = check_system_deps(backend="team")
    assert isinstance(result, SysDepsResult)
    # uv + python present, but no runtime → all_present=False
    assert result.all_present is False
    uv_dep = [d for d in result.deps if d.name == "uv"]
    assert uv_dep[0].present is True
    podman_dep = [d for d in result.deps if d.name == "podman"]
    assert len(podman_dep) == 1
    assert podman_dep[0].present is False


def test_check_system_deps_team_with_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    """team backend + podman on PATH → all_present=True."""
    import memini_ai.installer as inst

    def _which(name: str) -> str | None:
        if name == "uv":
            return "/usr/bin/uv"
        if name == "podman":
            return "/usr/bin/podman"
        return None

    def _run_quiet(cmd: list[str], timeout: float = 5.0) -> bool:
        return cmd == ["podman", "compose", "version"]

    monkeypatch.setattr(shutil, "which", _which)
    monkeypatch.setattr(inst, "_run_quiet", _run_quiet)
    result = check_system_deps(backend="team")
    assert isinstance(result, SysDepsResult)
    assert result.all_present is True


# ── read_admin_credentials + build_admin_dsn ──────────────────────────────────


def test_read_admin_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No .env in homedir → None."""
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent-home-12345"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert read_admin_credentials() is None


def test_read_admin_credentials_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid .env with admin creds → dict parsed."""
    # get_homedir_config_path() returns tmp_path/.config/opencode on Linux.
    config_dir = tmp_path / ".config" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / ".env").write_text(
        "# comment\n"
        "MEMINI_ADMIN_USER=memini_admin\n"
        "MEMINI_ADMIN_PASSWORD=s3cret-pw\n"
        "MEMINI_DB_HOST=localhost\n"
        "MEMINI_DB_PORT=5432\n"
        "MEMINI_DB_NAME=memini\n"
        "MEMINI_DB_SSLMODE=require\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    creds = read_admin_credentials()
    assert creds is not None
    assert creds["MEMINI_ADMIN_USER"] == "memini_admin"
    assert creds["MEMINI_ADMIN_PASSWORD"] == "s3cret-pw"
    assert creds["MEMINI_DB_SSLMODE"] == "require"


def test_build_admin_dsn() -> None:
    """Build admin DSN from creds dict, includes sslmode when not disable."""
    creds = {
        "MEMINI_ADMIN_USER": "memini_admin",
        "MEMINI_ADMIN_PASSWORD": "pw",
        "MEMINI_DB_HOST": "db.host",
        "MEMINI_DB_PORT": "5432",
        "MEMINI_DB_NAME": "memini",
        "MEMINI_DB_SSLMODE": "require",
    }
    dsn = build_admin_dsn(creds)
    assert dsn == "postgresql://memini_admin:pw@db.host:5432/memini?sslmode=require"


def test_build_admin_dsn_disable() -> None:
    """sslmode=disable → no ?sslmode= in DSN."""
    creds = {
        "MEMINI_ADMIN_USER": "memini_admin",
        "MEMINI_ADMIN_PASSWORD": "pw",
        "MEMINI_DB_HOST": "db.host",
        "MEMINI_DB_PORT": "5432",
        "MEMINI_DB_NAME": "memini",
        "MEMINI_DB_SSLMODE": "disable",
    }
    dsn = build_admin_dsn(creds)
    assert "?sslmode=" not in dsn


# ── _resolve_memini_command (Session 53/54 bugfix) ───────────────────────────
#
# Bug: opencode spawns MCP servers with a non-interactive env that does NOT
# include ``/home/<user>/.local/bin`` on most distros.  Bare
# ``["uvx", ...]`` in the opencode.json command array therefore fails with
# "uvx: command not found" and the MCP server never starts.  Fix: resolve
# uvx / memini-ai to absolute paths at install time.


def test_resolve_memini_command_uses_uvx_when_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """shutil.which('uvx') returns a path → that absolute path is used."""
    fake_uvx = tmp_path / "uvx"
    fake_uvx.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_uvx.chmod(0o755)
    monkeypatch.setattr(
        shutil, "which", lambda name: str(fake_uvx) if name == "uvx" else None
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    from memini_ai.installer import _resolve_memini_command

    cmd = _resolve_memini_command()
    assert cmd[0] == str(fake_uvx)
    assert "--from" in cmd
    assert "memini-ai-dev" in cmd
    assert "memini-ai" in cmd
    # Must be absolute, not bare "uvx"
    assert Path(cmd[0]).is_absolute()


def test_resolve_memini_command_falls_back_to_local_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No uvx on PATH + ~/.local/bin/memini-ai exists → use the local install.

    This is the test-bunty case: opencode spawns with PATH=/usr/bin:/bin
    (no .local/bin), and the only entry point is the local uv-tool install.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    local_memini = local_bin / "memini-ai"
    local_memini.write_text("#!/bin/sh\n", encoding="utf-8")
    local_memini.chmod(0o755)

    # shutil.which returns None for everything (simulates PATH without uvx)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: home)

    from memini_ai.installer import _resolve_memini_command

    cmd = _resolve_memini_command()
    assert cmd == [str(local_memini)]
    # Absolute path (no PATH lookup needed at MCP spawn time)
    assert Path(cmd[0]).is_absolute()


def test_resolve_memini_command_falls_back_to_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No uvx + no local install + no memini_ai.cli module → print warning,
    fall back to bare 'uvx' command.

    We simulate "no memini_ai.cli" by hiding the import via ``sys.modules``
    so the dev-checkout fallback path raises ImportError.
    """
    home = tmp_path / "empty-home"
    home.mkdir()

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: home)

    # Hide memini_ai.cli so the import in the fallback branch raises.
    import sys as _sys

    saved = _sys.modules.pop("memini_ai.cli", None)
    # Make ``import memini_ai.cli`` raise even if re-imported
    import builtins as _bi

    real_import = _bi.__import__

    def fake_import(name, *args, **kwargs):
        if name == "memini_ai.cli" or name.startswith("memini_ai.cli"):
            raise ImportError(f"hidden by test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_bi, "__import__", fake_import)

    from memini_ai.installer import _resolve_memini_command

    cmd = _resolve_memini_command()
    assert cmd == ["uvx", "--from", "memini-ai-dev", "memini-ai"]
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "uvx" in captured.out

    # restore
    if saved is not None:
        _sys.modules["memini_ai.cli"] = saved


def test_build_mcp_block_uses_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: opencode.json mcp.memini-ai-dev.command must be absolute.

    The whole point of the Session 53/54 fix: bare 'uvx' in the command
    array fails at MCP spawn time because opencode uses a non-interactive
    PATH that excludes ~/.local/bin.
    """
    from memini_ai.installer import _build_mcp_block, _default_config

    fake_uvx = tmp_path / "uvx"
    fake_uvx.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_uvx.chmod(0o755)
    monkeypatch.setattr(
        shutil, "which", lambda name: str(fake_uvx) if name == "uvx" else None
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    block = _build_mcp_block(_default_config())
    cmd = block["memini-ai-dev"]["command"]
    assert Path(cmd[0]).is_absolute(), f"command[0] must be absolute, got {cmd[0]!r}"
    assert cmd[0] != "uvx", "bare 'uvx' would fail at MCP spawn time"


def test_build_homedir_opencode_json_mcp_command_is_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: the opencode.json written by `init` has absolute paths."""
    from memini_ai.installer import _default_config, build_homedir_opencode_json

    fake_uvx = tmp_path / "uvx"
    fake_uvx.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_uvx.chmod(0o755)
    monkeypatch.setattr(
        shutil, "which", lambda name: str(fake_uvx) if name == "uvx" else None
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    cfg = _default_config()
    data = build_homedir_opencode_json(cfg)
    cmd = data["mcp"]["memini-ai-dev"]["command"]
    assert Path(cmd[0]).is_absolute(), f"command[0] must be absolute, got {cmd[0]!r}"
