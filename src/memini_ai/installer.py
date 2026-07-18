"""Standalone installer for memini-ai-dev.

Mirrors the neuralgentics two-init installer pattern. Provides two entry
points dispatched from ``cli.py``:

- ``run_install(mode, args)`` — installs config to homedir or project
- ``run_update(args)`` — updates existing config + refreshes package

Pure stdlib: ``argparse``, ``json``, ``pathlib``, ``shutil``, ``subprocess``.
No new dependencies. See ``docs/design/`` for the design document.

Backward-compatibility note: ``memini-ai init`` with NO flags still calls the
existing ``_init()`` logic (start embedded DB + print URI). This module is
only invoked when ``--homedir`` or ``--project`` flags are passed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from memini_ai import __version__

# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class SysDep:
    """A single system dependency check result."""

    name: str
    present: bool
    install_command: str | None


@dataclass
class SysDepsResult:
    """Aggregate result of system-dependency checks."""

    all_present: bool
    deps: list[SysDep]


@dataclass
class InstallConfig:
    """User-selected install options, captured from prompts or flags."""

    backend: str  # "pgembed" or "team"
    team_host: str | None
    team_port: str | None
    team_database: str | None
    team_user: str | None
    team_password: str | None
    embedding: str  # "cpu", "auto", "gpu"
    image_search: bool
    trust_engine: bool
    knowledge_graph: bool
    tiered_loading: bool
    auto_extract: bool
    precompress: bool
    decay: bool
    dialectic: bool
    thought_chains: bool
    ollama_api_key: str | None


@dataclass
class BackupResult:
    """Result of the SHA-256 idempotency-aware config write."""

    backed_up: bool
    backup_path: str | None


@dataclass
class PreDownloadResult:
    """Result of the ``uvx --from memini-ai-dev memini-ai --help`` warm-up."""

    success: bool
    output: str
    error: str | None


# ── Ollama Cloud model list (mirrors neuralgentics provider block) ───────────

_OLLAMA_CLOUD_MODELS: list[str] = [
    "kimi-k2.6",
    "glm-5.2",
    "deepseek-v4-pro",
    "devstral-2:123b",
    "deepseek-v4-flash",
    "qwen3-coder-next",
    "minimax-m3",
    "mistral-large-3:675b",
    "qwen3.5",
    "devstral-small-2:24b",
]


# ── System deps ─────────────────────────────────────────────────────────────


def check_system_deps() -> SysDepsResult:
    """Verify ``uv`` on PATH and Python 3.12+.

    Returns a structured result so callers can print install instructions.
    """
    deps: list[SysDep] = []

    uv_path = shutil.which("uv")
    deps.append(
        SysDep(
            name="uv",
            present=uv_path is not None,
            install_command="curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
    )

    py_ok = sys.version_info >= (3, 12)
    deps.append(
        SysDep(
            name="python>=3.12",
            present=py_ok,
            install_command=None,  # Python is already running; nothing to install
        )
    )

    return SysDepsResult(all_present=all(d.present for d in deps), deps=deps)


# ── Config paths ────────────────────────────────────────────────────────────


def get_homedir_config_path() -> Path:
    """Cross-platform homedir config path.

    Linux/WSL → ``~/.config/opencode/``
    macOS     → ``~/Library/Application Support/opencode/``
    """
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "opencode"
    # Linux, WSL, and any other Unix-like — follow XDG default.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg) / "opencode" if xdg else Path.home() / ".config" / "opencode"


def get_project_config_path(cwd: Path | None = None) -> Path:
    """Return ``{cwd}/.opencode/`` (default: current working directory)."""
    base = cwd if cwd is not None else Path.cwd()
    return base / ".opencode"


# ── Prompt helpers ───────────────────────────────────────────────────────────


def _prompt_choice(prompt: str, default: str, choices: set[str]) -> str:
    """Prompt for a single value, validate against ``choices``.

    Empty input returns ``default``. Re-prompts on invalid input.
    """
    while True:
        raw = input(prompt).strip()
        value = raw if raw else default
        if value in choices:
            return value
        print(f"  Invalid choice: {raw!r}. Expected one of {sorted(choices)}.")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    """Prompt for a yes/no answer. Empty input returns ``default``."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please enter 'y' or 'n'.")


def _prompt_optional(prompt: str) -> str | None:
    """Prompt for an optional string value. Empty → ``None``."""
    raw = input(prompt).strip()
    return raw if raw else None


# ── Run prompts ─────────────────────────────────────────────────────────────


def _default_config() -> InstallConfig:
    return InstallConfig(
        backend="pgembed",
        team_host=None,
        team_port=None,
        team_database=None,
        team_user=None,
        team_password=None,
        embedding="auto",
        image_search=True,
        trust_engine=True,
        knowledge_graph=True,
        tiered_loading=True,
        auto_extract=True,
        precompress=True,
        decay=True,
        dialectic=True,
        thought_chains=True,
        ollama_api_key=None,
    )


def _prompt_team_connection() -> tuple[
    str | None, str | None, str | None, str | None, str | None
]:
    """Prompt for team PostgreSQL connection details."""
    print("\n  Team PostgreSQL server connection:")
    host = _prompt_optional("    Host [localhost]: ") or "localhost"
    port = _prompt_optional("    Port [5432]: ") or "5432"
    database = _prompt_optional("    Database [postgres]: ") or "postgres"
    user = _prompt_optional("    User [postgres]: ") or "postgres"
    password = _prompt_optional("    Password (input visible): ")
    return host, port, database, user, password


def _prompt_embedding() -> str:
    """Prompt for embedding mode (cpu/auto/gpu)."""
    print(
        "\n  ? What embedding model should memini-ai use?\n"
        "    Embeddings convert text into vectors for semantic search.\n"
        "\n"
        "    1. CPU — Fast and lightweight (384-dim MiniLM)\n"
        "       Good search quality, low memory usage.\n"
        "       Best for laptops or machines without a GPU.\n"
        "\n"
        "    2. Auto (recommended) — CPU by default, auto-upgrades if you\n"
        "       add a GPU later. Best if you're not sure.\n"
        "\n"
        "    3. GPU — Highest quality (1024-dim BGE-M3)\n"
        "       Requires NVIDIA GPU or Apple Silicon.\n"
        "       Best for machines with a GPU.\n"
    )
    choice = _prompt_choice(
        "    Enter 1, 2, or 3 [2]: ", default="2", choices={"1", "2", "3"}
    )
    return {"1": "cpu", "2": "auto", "3": "gpu"}[choice]


def _prompt_backend() -> str:
    """Prompt for backend mode (pgembed/team)."""
    print(
        "\n  ? How should memini-ai store memories?\n"
        "\n"
        "    1. Built-in database (recommended)\n"
        "       No setup needed — everything runs locally.\n"
        "       Your memories are stored on your machine.\n"
        "       Best for getting started or solo use.\n"
        "\n"
        "    2. Team server\n"
        "       Connect to a shared PostgreSQL database.\n"
        "       Best for teams who want shared memory.\n"
        "       You'll need a PostgreSQL server running.\n"
    )
    choice = _prompt_choice("    Enter 1 or 2 [1]: ", default="1", choices={"1", "2"})
    return "pgembed" if choice == "1" else "team"


def _prompt_features(config: InstallConfig) -> InstallConfig:
    """Prompt for the batch of optional feature toggles."""
    print("\n  ? Enable optional features?\n")
    config.image_search = _prompt_yes_no("    Image search (CLIP):", default=True)
    config.trust_engine = _prompt_yes_no("    Trust engine:", default=True)
    config.knowledge_graph = _prompt_yes_no("    Knowledge graph:", default=True)
    config.tiered_loading = _prompt_yes_no("    Tiered loading:", default=True)
    config.auto_extract = _prompt_yes_no("    Auto-extract:", default=True)
    config.precompress = _prompt_yes_no("    Pre-compress extraction:", default=True)
    config.decay = _prompt_yes_no("    Memory decay:", default=True)
    config.dialectic = _prompt_yes_no("    Dialectic reasoning:", default=True)
    config.thought_chains = _prompt_yes_no("    Thought chains:", default=True)
    print("\n    All features can be toggled later in opencode.json under")
    print("    mcp.memini-ai-dev.env\n")
    return config


def _prompt_api_key() -> str | None:
    """Prompt for Ollama Cloud API key. Auto-skips if env var already set."""
    existing = os.environ.get("OLLAMA_API_KEY")
    if existing:
        print(
            "\n  Ollama Cloud API key: detected via $OLLAMA_API_KEY env var — skipping prompt.\n"
        )
        return None
    print("\n  ? Want to add your Ollama Cloud API key now?")
    if not _prompt_yes_no("    Add key now", default=False):
        return None
    return _prompt_optional("    Ollama Cloud API key: ")


def run_prompts(flags: argparse.Namespace) -> InstallConfig:
    """Interactive prompt session. Respects all skip flags.

    If ``flags.yes`` is True: skip all prompts, use defaults.
    If ``--embedded``/``--team`` flags are set: skip backend prompt.
    If ``--cpu-embed``/``--auto-embed``/``--gpu-embed`` flags are set: skip embedding prompt.
    If ``--no-image-search`` flag: disable image search.
    If ``--no-features`` flag: disable ALL optional features.
    """
    config = _default_config()

    if flags.yes:
        # Skip all prompts. Apply --no-* overrides to the defaults.
        if getattr(flags, "no_image_search", False):
            config.image_search = False
        if getattr(flags, "no_features", False):
            _disable_all_features(config)
        # If --team flag passed with --yes, still need team connection from env or prompt?
        # With --yes we use pgembed by default; --team with --yes → user must set env vars later.
        if getattr(flags, "team", False) and not getattr(flags, "embedded", False):
            config.backend = "team"
        return config

    # Backend selection
    if getattr(flags, "embedded", False):
        config.backend = "pgembed"
    elif getattr(flags, "team", False):
        config.backend = "team"
    else:
        config.backend = _prompt_backend()

    # Team connection details
    if config.backend == "team":
        host, port, database, user, password = _prompt_team_connection()
        config.team_host = host
        config.team_port = port
        config.team_database = database
        config.team_user = user
        config.team_password = password

    # Embedding mode
    if getattr(flags, "cpu_embed", False):
        config.embedding = "cpu"
    elif getattr(flags, "auto_embed", False):
        config.embedding = "auto"
    elif getattr(flags, "gpu_embed", False):
        config.embedding = "gpu"
    else:
        config.embedding = _prompt_embedding()

    # Feature toggles
    if getattr(flags, "no_features", False):
        _disable_all_features(config)
    elif getattr(flags, "no_image_search", False):
        config.image_search = False
        config = _prompt_features(config)
        config.image_search = False  # force-off even if user said yes
    else:
        config = _prompt_features(config)

    # API key
    config.ollama_api_key = _prompt_api_key()

    return config


def _disable_all_features(config: InstallConfig) -> None:
    """Set all optional feature toggles to False."""
    config.image_search = False
    config.trust_engine = False
    config.knowledge_graph = False
    config.tiered_loading = False
    config.auto_extract = False
    config.precompress = False
    config.decay = False
    config.dialectic = False
    config.thought_chains = False


# ── opencode.json builders ───────────────────────────────────────────────────


def _build_mcp_env(config: InstallConfig) -> dict[str, str]:
    """Build the ``mcp.memini-ai-dev.env`` block from an InstallConfig."""
    env: dict[str, str] = {
        "MEMINI_VECTOR_BACKEND": "pgembed"
        if config.backend == "pgembed"
        else "postgres-external",
    }

    if config.backend == "pgembed":
        env["MEMINI_DB_URL"] = "pgembed"
    else:
        # Team mode: build postgresql://user:pass@host:port/db
        user = config.team_user or "postgres"
        password = config.team_password or ""
        host = config.team_host or "localhost"
        port = config.team_port or "5432"
        database = config.team_database or "postgres"
        env["MEMINI_DB_URL"] = (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )

    env["MEMINI_EMBEDDING_DIM"] = "1024" if config.embedding == "gpu" else "384"
    env["MEMINI_EMBEDDING_MODE"] = config.embedding
    if config.embedding == "gpu":
        env["MEMINI_MODEL_NAME"] = "BAAI/bge-m3"

    env["MEMINI_IMAGE_SEARCH_ENABLED"] = "true" if config.image_search else "false"
    env["MEMINI_IMAGE_DIR"] = "~/.memini-ai/images"

    env["TRUST_ENGINE"] = "true" if config.trust_engine else "false"
    env["MEMORY_GRAPH"] = "true" if config.knowledge_graph else "false"
    env["KG_ENABLED"] = "true" if config.knowledge_graph else "false"
    env["TIERED_LOADING"] = "true" if config.tiered_loading else "false"
    env["AUTO_EXTRACT"] = "true" if config.auto_extract else "false"
    env["PRECOMPRESS"] = "true" if config.precompress else "false"
    env["DECAY_ENABLED"] = "true" if config.decay else "false"
    env["DIALECTIC_ENABLED"] = "true" if config.dialectic else "false"
    env["THOUGHT_CHAINS"] = "true" if config.thought_chains else "false"

    return env


def _build_mcp_block(config: InstallConfig) -> dict[str, object]:
    """Build the ``mcp.memini-ai-dev`` entry (shared by homedir + project)."""
    return {
        "memini-ai-dev": {
            "type": "local",
            "enabled": True,
            "command": ["uvx", "--from", "memini-ai-dev", "memini-ai"],
            "env": _build_mcp_env(config),
        }
    }


def build_homedir_opencode_json(config: InstallConfig) -> dict[str, object]:
    """Build the homedir opencode.json dict.

    Includes: ``$schema``, ``autoupdate``, provider block (Ollama Cloud models),
    MCP block (memini-ai-dev), ``tool_output``, LSP, formatter.
    """
    models_block: dict[str, dict[str, str]] = {
        m: {"name": m} for m in _OLLAMA_CLOUD_MODELS
    }

    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": True,
        "provider": {
            "ollama": {
                "name": "ollama",
                "api": "https://ollama.com/v1",
                "options": {"apiKey": "{env:OLLAMA_API_KEY}"},
                "models": models_block,
            }
        },
        "mcp": _build_mcp_block(config),
        "tool_output": {"max_lines": 10000, "max_bytes": 512000},
        "lsp": {
            "typescript": {
                "disabled": False,
                "command": ["npx", "typescript-language-server", "--stdio"],
            }
        },
        "formatter": {"prettier": {"disabled": False}},
    }


def build_project_opencode_json(config: InstallConfig) -> dict[str, object]:
    """Build the project opencode.json dict.

    Only the MCP block — no provider, no LSP, no formatter, no $schema.
    Provider is inherited from homedir (OpenCode's config-merging behavior).
    """
    return {"mcp": _build_mcp_block(config)}


# ── File write with SHA-256 idempotency ──────────────────────────────────────


def compute_file_sha256(path: Path) -> str:
    """Read ``path`` and return the SHA-256 hex digest of its contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_config_with_backup(
    config_dir: Path,
    filename: str,
    content: str,
    force: bool,
    dry_run: bool,
) -> BackupResult:
    """Write ``config_dir/filename`` with SHA-256 idempotency + backup.

    - If file doesn't exist → write.
    - If file exists and SHA-256 matches → skip (idempotent).
    - If file exists and differs → backup to ``opencode-bak-{timestamp}.json``
      then write (unless ``force`` is False, in which case skip with no backup).
    - If ``dry_run`` → print what would be done, write nothing.
    """
    target = config_dir / filename

    if dry_run:
        if target.exists():
            existing_sha = compute_file_sha256(target)
            new_sha = hashlib.sha256(content.encode()).hexdigest()
            if existing_sha == new_sha:
                print(f"  [dry-run] {target} — unchanged (SHA-256 match), would skip")
            else:
                print(
                    f"  [dry-run] {target} — would back up + overwrite (SHA-256 differs)"
                )
        else:
            print(f"  [dry-run] {target} — would create new file")
        return BackupResult(backed_up=False, backup_path=None)

    if target.exists():
        existing_sha = compute_file_sha256(target)
        new_sha = hashlib.sha256(content.encode()).hexdigest()
        if existing_sha == new_sha:
            print(f"  ✓ {target} — unchanged (SHA-256 match), skipped")
            return BackupResult(backed_up=False, backup_path=None)
        if not force:
            print(f"  ⚠ {target} — exists and differs (use --force to overwrite)")
            return BackupResult(backed_up=False, backup_path=None)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        backup_name = f"opencode-bak-{timestamp}.json"
        backup_path = config_dir / backup_name
        shutil.copy2(target, backup_path)
        print(f"  ↳ Backed up existing {filename} → {backup_name}")
        target.write_text(content, encoding="utf-8")
        print(f"  ✓ Wrote {target}")
        return BackupResult(backed_up=True, backup_path=str(backup_path))

    target.write_text(content, encoding="utf-8")
    print(f"  ✓ Wrote {target}")
    return BackupResult(backed_up=False, backup_path=None)


# ── State file ───────────────────────────────────────────────────────────────


def write_state_file(
    config_dir: Path,
    version: str,
    mode: str,
    files: dict[str, str],
) -> None:
    """Write ``.memini-ai-state.json`` tracking install metadata."""
    state = {
        "installed_version": version,
        "mode": mode,
        "files": files,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    state_path = config_dir / ".memini-ai-state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"  ✓ Wrote state file {state_path}")


# ── Package pre-download ────────────────────────────────────────────────────


def pre_download_package(dry_run: bool) -> PreDownloadResult:
    """Run ``uvx --from memini-ai-dev memini-ai --help`` to warm the uvx cache.

    The first MCP server start will then be fast (no download).
    """
    cmd = ["uvx", "--from", "memini-ai-dev", "memini-ai", "--help"]
    if dry_run:
        print(f"  [dry-run] Would run: {' '.join(cmd)}")
        return PreDownloadResult(success=True, output="[dry-run skipped]", error=None)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        success = result.returncode == 0
        return PreDownloadResult(
            success=success,
            output=result.stdout[-2000:] if result.stdout else "",
            error=result.stderr if result.returncode != 0 else None,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return PreDownloadResult(success=False, output="", error=str(exc))


# ── Embedded DB start ────────────────────────────────────────────────────────


def start_embedded_db() -> None:
    """Call the existing ``_init()`` logic from ``cli.py``.

    Imports lazily so the installer module remains importable without the
    full ``memini_ai.postgres.driver`` dependency chain at module load time.
    """
    from memini_ai.cli import _init

    asyncio.run(_init())


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary(
    config: InstallConfig,
    mode: str,
    config_dir: Path,
    deps: SysDepsResult,
    pre_download: PreDownloadResult,
) -> None:
    """Print a formatted summary of what was installed."""
    print("\n" + "=" * 60)
    print("  memini-ai-dev install complete")
    print("=" * 60)
    print(f"  Mode:        {mode}")
    print(f"  Config dir:  {config_dir}")
    print(f"  Version:     {__version__}")
    print(f"  Backend:     {config.backend}")
    print(f"  Embedding:   {config.embedding}")
    print()
    print("  Features enabled:")
    print(f"    Image search:    {config.image_search}")
    print(f"    Trust engine:    {config.trust_engine}")
    print(f"    Knowledge graph: {config.knowledge_graph}")
    print(f"    Tiered loading:  {config.tiered_loading}")
    print(f"    Auto-extract:    {config.auto_extract}")
    print(f"    Pre-compress:    {config.precompress}")
    print(f"    Decay:           {config.decay}")
    print(f"    Dialectic:       {config.dialectic}")
    print(f"    Thought chains:  {config.thought_chains}")
    print()
    if pre_download.success:
        print("  Package pre-download: ✓ (uvx cache warmed)")
    else:
        print(f"  Package pre-download: ✗ ({pre_download.error or 'failed'})")
    print()
    if config.backend == "pgembed":
        print("  Next steps:")
        print("    1. Set OLLAMA_API_KEY in your environment (or .env)")
        print("    2. Run `opencode` to start using memini-ai-dev")
    else:
        print("  Next steps:")
        print("    1. Ensure your team PostgreSQL server is running")
        print("    2. Set OLLAMA_API_KEY in your environment (or .env)")
        print("    3. Run `opencode` to start using memini-ai-dev")
    print("=" * 60)


# ── .env writing (optional) ─────────────────────────────────────────────────


def maybe_write_env_file(
    config_dir: Path, config: InstallConfig, dry_run: bool
) -> None:
    """Optionally write a ``.env`` file with team credentials + API key.

    Mirrors neuralgentics' pattern: secrets go in .env, not opencode.json.
    """
    lines: list[str] = []
    if config.ollama_api_key:
        lines.append(f"OLLAMA_API_KEY={config.ollama_api_key}")
    if config.backend == "team" and config.team_password:
        lines.append("# Team PostgreSQL credentials (saved from installer)")
        lines.append(
            f"MEMINI_DB_URL=postgresql://{config.team_user or 'postgres'}:"
            f"{config.team_password}@{config.team_host or 'localhost'}:"
            f"{config.team_port or '5432'}/{config.team_database or 'postgres'}"
        )

    if not lines:
        return

    env_path = config_dir / ".env"
    if dry_run:
        print(f"  [dry-run] Would write {env_path} ({len(lines)} lines)")
        return

    # Append rather than overwrite if .env exists
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    new_content = "\n".join(lines) + "\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    env_path.write_text(existing + new_content, encoding="utf-8")
    print(f"  ✓ Wrote {env_path} (credentials)")


# ── Main install orchestrator ───────────────────────────────────────────────


def run_install(mode: str, args: argparse.Namespace) -> int:
    """Main installer orchestrator.

    ``mode`` is ``"homedir"`` or ``"project"``. Dispatched from ``cli.py``
    only when ``--homedir`` or ``--project`` flags are passed.
    """
    # 1. Check system deps
    deps = check_system_deps()
    if not deps.all_present:
        print("\n  ✗ Missing system dependencies:\n")
        for dep in deps.deps:
            status = "✓" if dep.present else "✗"
            print(f"    {status} {dep.name}")
            if not dep.present and dep.install_command:
                print(f"       Install:  {dep.install_command}")
        print("\n  Install the missing dependencies and re-run.")
        return 1

    # 2. Run prompts (or use defaults/flags)
    config = run_prompts(args)

    # 3. Resolve config dir
    if mode == "homedir":
        config_dir = get_homedir_config_path()
    else:
        target = getattr(args, "target", None)
        config_dir = get_project_config_path(Path(target) if target else None)

    # 4. Create config dir
    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build opencode.json
    if mode == "homedir":
        opencode_json = build_homedir_opencode_json(config)
    else:
        opencode_json = build_project_opencode_json(config)
    content = json.dumps(opencode_json, indent=2) + "\n"

    # 6. Write with backup
    write_config_with_backup(
        config_dir, "opencode.json", content, args.force, args.dry_run
    )

    # 6b. Write .env if credentials/key provided
    maybe_write_env_file(config_dir, config, args.dry_run)

    # 7. Write state file
    if not args.dry_run:
        files_manifest = {
            "opencode.json": compute_file_sha256(config_dir / "opencode.json")
            if (config_dir / "opencode.json").exists()
            else "",
        }
        write_state_file(config_dir, __version__, mode, files_manifest)

    # 8. Pre-download package
    print("\n  Pre-downloading memini-ai-dev package (warming uvx cache)...")
    pre_download = pre_download_package(args.dry_run)

    # 9. Start embedded DB (homedir + pgembed only)
    if mode == "homedir" and config.backend == "pgembed" and not args.dry_run:
        print("\n  Starting embedded PostgreSQL (pgembed)...")
        start_embedded_db()

    # 10. Print summary
    print_summary(config, mode, config_dir, deps, pre_download)

    # 11/12. Offer next step
    if not args.dry_run:
        if mode == "homedir":
            if _prompt_yes_no(
                "\n  Initialize project config in the current directory?", default=True
            ):
                from memini_ai.installer import run_install as _install

                project_args = argparse.Namespace(
                    **{**vars(args), "homedir": False, "project": True}
                )
                return _install("project", project_args)
        else:
            if _prompt_yes_no("\n  Launch opencode now?", default=True):
                subprocess.run(["opencode"], check=False)

    return 0


# ── Update flow ──────────────────────────────────────────────────────────────


def _find_install_configs() -> list[tuple[Path, str]]:
    """Find all config dirs with a ``.memini-ai-state.json``.

    Returns list of ``(config_dir, installed_version)`` tuples.
    """
    results: list[tuple[Path, str]] = []

    # Homedir
    homedir = get_homedir_config_path()
    homedir_state = homedir / ".memini-ai-state.json"
    if homedir_state.exists():
        try:
            state = json.loads(homedir_state.read_text(encoding="utf-8"))
            results.append((homedir, state.get("installed_version", "unknown")))
        except (json.JSONDecodeError, OSError):
            results.append((homedir, "unknown"))

    # All project .opencode/ dirs we can find (CWD + one level up)
    cwd = Path.cwd()
    candidates = [cwd / ".opencode"]
    parent = cwd.parent
    if parent != Path(cwd.root):
        candidates.append(parent / ".opencode")
    for candidate in candidates:
        state_file = candidate / ".memini-ai-state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                if (
                    candidate,
                    state.get("installed_version", "unknown"),
                ) not in results:
                    results.append(
                        (candidate, state.get("installed_version", "unknown"))
                    )
            except (json.JSONDecodeError, OSError):
                continue

    return results


def _query_pypi_latest() -> str | None:
    """Query PyPI for the latest version of ``memini-ai-dev``.

    Uses ``pip index versions`` (pip 21+). Returns the version string or
    ``None`` if the query fails.
    """
    try:
        result = subprocess.run(
            ["pip", "index", "versions", "memini-ai-dev"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return None
        # Output looks like: "memini-ai-dev (1.0.4)\nAvailable versions: 1.0.4, ..."
        for line in result.stdout.splitlines():
            if line.startswith("memini-ai-dev ("):
                # Extract version from "memini-ai-dev (X.Y.Z)"
                start = line.index("(") + 1
                end = line.index(")")
                return line[start:end]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def _update_mcp_entry(
    existing_json: dict[str, object],
    config: InstallConfig,
) -> dict[str, object]:
    """Update ONLY the ``memini-ai-dev`` MCP entry, preserve everything else."""
    mcp = existing_json.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    mcp["memini-ai-dev"] = _build_mcp_block(config)["memini-ai-dev"]
    existing_json["mcp"] = mcp
    return existing_json


def _default_update_config_from_state(state: dict[str, object]) -> InstallConfig:
    """Reconstruct a default InstallConfig from a state file.

    For updates we use sensible defaults — the update is about refreshing the
    MCP entry, not re-prompting the user for every option. The user's existing
    env vars are preserved by the merge in ``_update_mcp_entry`` reading the
    existing opencode.json.
    """
    return _default_config()


def run_update(args: argparse.Namespace) -> int:
    """Update flow: find installs, check PyPI, backup, refresh, update state."""
    print("\n  Checking for memini-ai-dev updates...\n")

    installs = _find_install_configs()
    if not installs:
        print(
            "  No memini-ai-dev installs found. Run `memini-ai init --homedir` first."
        )
        return 1

    latest = _query_pypi_latest()
    if latest is None:
        print("  ✗ Could not query PyPI for the latest version.")
        print("    Check your network connection and try again.")
        return 1

    print(f"  Latest version on PyPI: {latest}")
    print(f"  Found {len(installs)} install(s):\n")
    for config_dir, installed_version in installs:
        print(f"    {config_dir} (installed: {installed_version})")

    any_updated = False
    for config_dir, installed_version in installs:
        print(f"\n  --- Updating {config_dir} ---")

        if installed_version == latest and not args.force:
            print(f"  Already up to date (v{installed_version}).")
            if args.check:
                continue

        if args.check:
            print(f"  Update available: {installed_version} → {latest}")
            any_updated = True
            continue

        # Backup
        opencode_path = config_dir / "opencode.json"
        if opencode_path.exists():
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            backup_name = f"opencode-bak-{timestamp}.json"
            backup_path = config_dir / backup_name
            shutil.copy2(opencode_path, backup_path)
            print(f"  ↳ Backed up → {backup_name}")

        # Read existing opencode.json
        existing_json: dict[str, object] = {}
        if opencode_path.exists():
            try:
                existing_json = json.loads(opencode_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print("  ⚠ Could not parse existing opencode.json — will overwrite")

        # Update the memini-ai-dev MCP entry, preserving everything else
        config = _default_update_config_from_state(
            json.loads(
                (config_dir / ".memini-ai-state.json").read_text(encoding="utf-8")
            )
            if (config_dir / ".memini-ai-state.json").exists()
            else {}
        )
        updated_json = _update_mcp_entry(existing_json, config)

        # Write
        if not args.dry_run:
            opencode_path.write_text(
                json.dumps(updated_json, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  ✓ Updated {opencode_path}")

            # Refresh package cache
            refresh = pre_download_package(dry_run=False)
            if refresh.success:
                print("  ✓ Package cache refreshed")
            else:
                print(f"  ⚠ Package cache refresh failed: {refresh.error}")

            # Update state file
            files_manifest = {
                "opencode.json": compute_file_sha256(opencode_path)
                if opencode_path.exists()
                else ""
            }
            write_state_file(config_dir, latest, "update", files_manifest)
        else:
            print(f"  [dry-run] Would update {opencode_path}")

        any_updated = True

    if args.check:
        print("\n  --check mode: no files modified.\n")
        return 0 if not any_updated else 0

    if any_updated:
        print(f"\n  ✓ Updated to v{latest}\n")
    else:
        print("\n  All installs are up to date.\n")

    return 0
