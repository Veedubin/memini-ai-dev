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
import contextlib
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
    team_sslmode: str  # "prefer", "require", "verify-ca", "verify-full", "disable"
    team_sslrootcert: (
        str | None
    )  # path to PEM CA cert (only when verify-ca/verify-full)
    team_new_db: bool  # True → admin/bootstrap flow, False → existing DB
    team_admin_user: str | None  # admin role (RBAC bootstrap)
    team_admin_password: str | None  # admin password (RBAC bootstrap)
    project_user: str | None  # generated project role (RBAC)
    project_password: str | None  # generated project password (RBAC)
    project_name: str | None  # sanitized project role name (RBAC)
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


@dataclass
class ContainerRuntime:
    """A single detected container runtime (podman, docker, containerd, …)."""

    name: str  # "podman", "docker", "containerd", or ""
    compose_command: str  # "podman compose", "docker compose", "docker-compose", or ""
    present: bool


@dataclass
class ContainerRuntimeResult:
    """Aggregate result of container-runtime checks."""

    runtimes: list[ContainerRuntime]
    has_any: bool


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


def check_system_deps(backend: str | None = None) -> SysDepsResult:
    """Verify ``uv`` on PATH and Python 3.12+.

    When ``backend == "team"``, also checks for a container runtime
    (podman/docker/containerd) and appends the results to ``deps``.
    In that case ``all_present`` requires at least one runtime to be
    present (pgembed does not need containers).

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

    if backend == "team":
        runtime_result = check_container_runtime()
        for rt in runtime_result.runtimes:
            deps.append(
                SysDep(
                    name=rt.name,
                    present=rt.present,
                    install_command=_container_install_command(rt.name),
                )
            )
        # uv + python must be present; for team mode at least one
        # container runtime must also be present (unless the user
        # will install one separately — handled by offer_container_install).
        base_ok = all(d.present for d in deps if d.name in {"uv", "python>=3.12"})
        return SysDepsResult(all_present=base_ok and runtime_result.has_any, deps=deps)

    return SysDepsResult(all_present=all(d.present for d in deps), deps=deps)


# ── Container runtime detection ──────────────────────────────────────────────


def _run_quiet(cmd: list[str], timeout: float = 5.0) -> bool:
    """Run ``cmd`` quietly; return True on exit code 0."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_compose_command(runtime: str) -> str:
    """Return the compose command string for ``runtime`` or "" if absent."""
    if runtime == "podman":
        if shutil.which("podman") is None:
            return ""
        # Built-in `podman compose` (v4+) takes priority over the pip pkg.
        if _run_quiet(["podman", "compose", "version"]):
            return "podman compose"
        if shutil.which("podman-compose") is not None:
            return "podman-compose"
        return ""
    if runtime == "docker":
        if shutil.which("docker") is None:
            return ""
        if _run_quiet(["docker", "compose", "version"]):
            return "docker compose"
        if shutil.which("docker-compose") is not None:
            return "docker-compose"
        return ""
    if runtime == "containerd":
        # containerd itself has no compose; nerdctl provides it.
        if shutil.which("nerdctl") is None:
            return ""
        if _run_quiet(["nerdctl", "compose", "version"]):
            return "nerdctl compose"
        return ""
    return ""


def check_container_runtime() -> ContainerRuntimeResult:
    """Detect available container runtimes and their compose variants.

    Returns one ``ContainerRuntime`` entry per known runtime (podman, docker,
    containerd), with ``present=True`` only when the runtime binary is on
    PATH. ``compose_command`` is populated when a compose variant is usable.
    """
    runtimes: list[ContainerRuntime] = []

    podman_present = shutil.which("podman") is not None
    runtimes.append(
        ContainerRuntime(
            name="podman",
            compose_command=_detect_compose_command("podman") if podman_present else "",
            present=podman_present,
        )
    )

    docker_present = shutil.which("docker") is not None
    runtimes.append(
        ContainerRuntime(
            name="docker",
            compose_command=_detect_compose_command("docker") if docker_present else "",
            present=docker_present,
        )
    )

    containerd_present = shutil.which("containerd") is not None
    runtimes.append(
        ContainerRuntime(
            name="containerd",
            compose_command=(
                _detect_compose_command("containerd") if containerd_present else ""
            ),
            present=containerd_present,
        )
    )

    return ContainerRuntimeResult(
        runtimes=runtimes, has_any=any(rt.present for rt in runtimes)
    )


def _container_install_command(runtime: str) -> str | None:
    """Return the apt install command for ``runtime`` (Linux default)."""
    if runtime == "podman":
        return "sudo apt-get install -y podman podman-docker podman-compose"
    if runtime == "docker":
        return "sudo apt-get install -y docker.io docker-compose-v2"
    if runtime == "containerd":
        return "sudo apt-get install -y containerd"
    return None


def _detect_distro_family() -> str:
    """Best-effort detection of Linux distro family: apt/dnf/pacman/mac."""
    system = platform.system()
    if system == "Darwin":
        return "mac"
    # /etc/os-release is the standard on most Linux distros.
    os_release = Path("/etc/os-release")
    if os_release.exists():
        try:
            text = os_release.read_text(encoding="utf-8")
        except OSError:
            return "apt"
        lower = text.lower()
        if any(x in lower for x in ("ubuntu", "debian", "linuxmint", "pop!_os")):
            return "apt"
        if any(x in lower for x in ("fedora", "rhel", "centos", "rocky", "alma")):
            return "dnf"
        if "arch" in lower:
            return "pacman"
    return "apt"


def _install_command_for(runtime: str, distro: str) -> str:
    """Build the install command for ``runtime`` on ``distro``."""
    pkgs = {
        "podman": {
            "apt": "podman podman-docker podman-compose",
            "dnf": "podman podman-docker podman-compose",
            "pacman": "podman podman-docker podman-compose",
            "mac": "podman podman-compose",
        },
        "docker": {
            "apt": "docker.io docker-compose-v2",
            "dnf": "docker docker-compose",
            "pacman": "docker docker-compose",
            "mac": "docker docker-compose",
        },
        "containerd": {
            "apt": "containerd",
            "dnf": "containerd",
            "pacman": "containerd",
            "mac": "containerd",
        },
    }
    pkg_str = pkgs.get(runtime, {}).get(distro, pkgs[runtime]["apt"])
    if distro == "mac":
        return f"brew install {pkg_str}"
    if distro == "dnf":
        return f"sudo dnf install -y {pkg_str}"
    if distro == "pacman":
        return f"sudo pacman -S --noconfirm {pkg_str}"
    return f"sudo apt-get install -y {pkg_str}"


def offer_container_install() -> bool:
    """Prompt the user to install a container runtime. Returns True on success.

    Only call when ``check_container_runtime().has_any`` is False. Prints
    the install commands and runs them via ``subprocess.run`` (apt only).
    Docker install is instructional only (the daemon setup is complex).
    """
    print(
        "\n  No container runtime found. Which would you like to install?\n"
        "\n"
        "  1. Podman (recommended)\n"
        "     Rootless and daemonless — no background service needed.\n"
        "     More secure: containers run as your user, not root.\n"
        "     Works with docker commands via podman-docker.\n"
        "\n"
        "  2. Docker\n"
        "     The industry standard — most tutorials assume it.\n"
        "     Requires a background daemon (dockerd).\n"
        "\n"
        "  3. Skip — I'll install it myself\n"
    )
    choice = _prompt_choice(
        "  Enter 1, 2, or 3 [1]: ", default="1", choices={"1", "2", "3"}
    )

    distro = _detect_distro_family()

    if choice == "1":
        cmd = _install_command_for("podman", distro)
        print(f"\n  Running: {cmd}")
        if distro == "mac":
            print("  (macOS: install Homebrew first if you don't have it.)")
            return False
        result = subprocess.run(cmd.split(), check=False)
        if result.returncode == 0:
            print("  Installed podman-docker so docker commands also work with podman.")
            return True
        print(f"  ✗ Install failed (exit {result.returncode}). See output above.")
        return False

    if choice == "2":
        cmd = _install_command_for("docker", distro)
        if distro == "mac":
            print(f"\n  Run: {cmd}")
            print(
                "  Or install Docker Desktop from https://docker.com/products/docker-desktop"
            )
            print(
                "  Then enable the daemon and re-run `memini-ai init --homedir --team`."
            )
            return False
        print(f"\n  Running: {cmd}")
        result = subprocess.run(cmd.split(), check=False)
        if result.returncode != 0:
            print(f"  ✗ Install failed (exit {result.returncode}). See output above.")
            return False
        # Enable + start the daemon (Linux systemd).
        print("  Enabling docker daemon…")
        subprocess.run(["sudo", "systemctl", "enable", "--now", "docker"], check=False)
        print(
            "  Docker installed. Add yourself to the docker group with:\n"
            "    sudo usermod -aG docker $USER\n"
            "  then log out + back in (or `newgrp docker`)."
        )
        return True

    # Skip
    print(
        "\n  Skipping container runtime install. The team observability stack\n"
        "  (memini-ai dashboard --team) will need a runtime — install one of:\n"
        "    podman: sudo apt-get install -y podman podman-docker podman-compose\n"
        "    docker: sudo apt-get install -y docker.io docker-compose-v2\n"
    )
    return False


def verify_runtime_installed(runtime: str) -> bool:
    """Verify the runtime was installed and is functional."""
    checks: dict[str, list[str]] = {
        "podman": ["podman", "info"],
        "docker": ["docker", "info"],
        "containerd": ["ctr", "version"],
    }
    cmd = checks.get(runtime)
    if not cmd:
        return False
    return _run_quiet(cmd, timeout=10.0)


# ── RBAC: password + role-name helpers ───────────────────────────────────────


def generate_password() -> str:
    """Generate a 43-char URL-safe random password (~256 bits of entropy).

    ``secrets.token_urlsafe(32)`` produces 32 random bytes encoded as base64
    url-safe, yielding a 43-character string. Suitable for PostgreSQL
    passwords.
    """
    return secrets.token_urlsafe(32)


def sanitize_project_name(dir_name: str) -> str:
    """Convert a directory name to a valid PostgreSQL role name.

    Rules (see RBAC design §4):
    - Underscores → hyphens (Postgres convention)
    - Strip non-alphanumeric chars except hyphens
    - Collapse multiple hyphens
    - Strip leading/trailing hyphens
    - Truncate to 63 chars
    - Default to ``"memini-project"`` if empty after sanitization
    """
    name = dir_name.replace("_", "-")
    name = re.sub(r"[^a-zA-Z0-9-]", "", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    name = name[:63]
    return name if name else "memini-project"


# ── RBAC: admin / DB bootstrap + project user creation ───────────────────────


_ADMIN_ROLE = "memini_admin"
_DEFAULT_DB_NAME = "memini"


async def create_admin_and_db(
    host: str,
    port: int,
    superuser_password: str,
    admin_password: str,
    *,
    db_name: str = _DEFAULT_DB_NAME,
    sslmode: str = "prefer",
    sslrootcert: str | None = None,
) -> str:
    """Create the ``memini_admin`` role + DB + extensions + schema.

    Connects as the ``postgres`` superuser, creates the admin role and
    database, then connects as admin to install the ``vector`` and
    ``vectorscale`` extensions and run the memini-ai schema migrations.

    Returns the admin DSN (``postgresql://memini_admin:...@host:port/db``).
    """
    import asyncpg

    super_dsn = build_team_dsn(
        host,
        str(port),
        "postgres",
        "postgres",
        superuser_password,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )
    conn = await asyncpg.connect(super_dsn)
    try:
        # Idempotent role creation (in case of re-run after partial failure).
        await conn.execute(
            f"DO $$ BEGIN "
            f"CREATE ROLE {_ADMIN_ROLE} WITH LOGIN PASSWORD $1 SUPERUSER; "
            f"EXCEPTION WHEN duplicate_object THEN NULL; END $$",
            admin_password,
        )
        # Idempotent DB creation.
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db_name}" OWNER {_ADMIN_ROLE}')
    finally:
        await conn.close()

    admin_dsn = build_team_dsn(
        host,
        str(port),
        db_name,
        _ADMIN_ROLE,
        admin_password,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # vectorscale may not be available in all environments; tolerate absence.
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"  ⚠ Could not install vectorscale extension: {exc}")
        # Run the memini-ai schema migrations.
        try:
            from memini_ai.postgres.schema import get_schema_sql

            schema_sql = get_schema_sql(use_vectorscale=True)
            await conn.execute(schema_sql)
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"  ⚠ Could not apply schema migrations: {exc}")
    finally:
        await conn.close()

    return admin_dsn


async def create_project_user(
    admin_dsn: str,
    project_name: str,
    *,
    db_name: str = _DEFAULT_DB_NAME,
) -> tuple[str, str]:
    """Create a project-scoped PostgreSQL role with CRUD grants.

    Returns ``(role_name, password)``. The role name is the sanitized
    project directory name (double-quoted in SQL for safety with hyphens).
    Grants: CONNECT on DB, USAGE on public schema, SELECT/INSERT/UPDATE/
    DELETE on all tables + sequences, plus ALTER DEFAULT PRIVILEGES so future
    migrations don't break existing project users.
    """
    import asyncpg

    role_name = sanitize_project_name(project_name)
    password = generate_password()

    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(
            f"DO $$ BEGIN "
            f'CREATE ROLE "{role_name}" WITH LOGIN PASSWORD $1; '
            f"EXCEPTION WHEN duplicate_object THEN "
            f'ALTER ROLE "{role_name}" WITH LOGIN PASSWORD $1; '
            f"END $$",
            password,
        )
        await conn.execute(f'GRANT CONNECT ON DATABASE "{db_name}" TO "{role_name}"')
        await conn.execute(f'GRANT USAGE ON SCHEMA public TO "{role_name}"')
        await conn.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE "
            f'ON ALL TABLES IN SCHEMA public TO "{role_name}"'
        )
        await conn.execute(
            f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{role_name}"'
        )
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {_ADMIN_ROLE} IN SCHEMA public "
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{role_name}"'
        )
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES FOR ROLE {_ADMIN_ROLE} IN SCHEMA public "
            f'GRANT USAGE, SELECT ON SEQUENCES TO "{role_name}"'
        )
    finally:
        await conn.close()

    return role_name, password


async def create_device_user(
    admin_dsn: str,
    hostname: str,
    *,
    db_name: str = _DEFAULT_DB_NAME,
) -> tuple[str, str]:
    """Create a device-scoped PostgreSQL role. Returns ``(role_name, password)``.

    A device role is for a machine that joins an existing team DB without a
    project directory (e.g. a CI runner or a second laptop). Same grants as
    a project user; role name is ``memini-device-<sanitized-hostname>``.
    """
    safe_host = sanitize_project_name(hostname)
    return await create_project_user(
        admin_dsn, f"memini-device-{safe_host}", db_name=db_name
    )


# ── RBAC: .env file writers ──────────────────────────────────────────────────


def _write_env_file(path: Path, content: str) -> None:
    """Write content to ``path`` with chmod 600 (owner read/write only)."""
    path.write_text(content, encoding="utf-8")
    # chmod can fail on some filesystems (FAT32, WSL drift). Don't abort the
    # install over it — the content is still on disk.
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def write_admin_env(
    config_dir: Path,
    host: str,
    port: int,
    password: str,
    *,
    db_name: str = _DEFAULT_DB_NAME,
    sslmode: str = "prefer",
    sslrootcert: str | None = None,
) -> None:
    """Write admin credentials to ``config_dir/.env`` with chmod 600.

    Mirrors the homedir .env layout from RBAC design §5. The MCP server
    reads these via the ``{env:...}`` references in opencode.json.
    """
    env_path = config_dir / ".env"
    content = (
        "# memini-ai admin credentials (generated by memini-ai init --homedir --team --new-db)\n"
        "# DO NOT commit this file to version control\n"
        f"MEMINI_ADMIN_USER={_ADMIN_ROLE}\n"
        f"MEMINI_ADMIN_PASSWORD={password}\n"
        f"MEMINI_DB_HOST={host}\n"
        f"MEMINI_DB_PORT={port}\n"
        f"MEMINI_DB_NAME={db_name}\n"
        f"MEMINI_DB_SSLMODE={sslmode}\n"
    )
    if sslrootcert:
        content += f"DB_SSLROOTCERT={sslrootcert}\n"
    _write_env_file(env_path, content)
    print(f"  ✓ Wrote {env_path} (admin credentials, chmod 600)")


def write_project_env(
    config_dir: Path,
    role_name: str,
    password: str,
    host: str,
    port: int,
    db_name: str,
    *,
    sslmode: str = "prefer",
    sslrootcert: str | None = None,
) -> None:
    """Write project credentials to ``config_dir/.env`` with chmod 600.

    Includes ``MEMINI_PEER_ID`` so the MCP server can scope queries to this
    project (when ``MEMINI_PEER_ENFORCEMENT=true`` is enabled separately).
    """
    env_path = config_dir / ".env"
    content = (
        "# memini-ai project credentials (generated by memini-ai init --project --team)\n"
        "# DO NOT commit this file to version control\n"
        f"MEMINI_PROJECT_USER={role_name}\n"
        f"MEMINI_PROJECT_PASSWORD={password}\n"
        f"MEMINI_DB_HOST={host}\n"
        f"MEMINI_DB_PORT={port}\n"
        f"MEMINI_DB_NAME={db_name}\n"
        f"MEMINI_DB_SSLMODE={sslmode}\n"
        f"MEMINI_PEER_ID={role_name}\n"
    )
    if sslrootcert:
        content += f"DB_SSLROOTCERT={sslrootcert}\n"
    _write_env_file(env_path, content)
    print(f"  ✓ Wrote {env_path} (project credentials, chmod 600)")


def read_admin_credentials() -> dict[str, str] | None:
    """Read admin credentials from ``~/.config/opencode/.env``.

    Returns a dict with keys ``MEMINI_ADMIN_USER``, ``MEMINI_ADMIN_PASSWORD``,
    ``MEMINI_DB_HOST``, ``MEMINI_DB_PORT``, ``MEMINI_DB_NAME``,
    ``MEMINI_DB_SSLMODE`` (if present), ``DB_SSLROOTCERT`` (if present), or
    ``None`` if the file is missing or doesn't contain admin credentials.
    """
    homedir = get_homedir_config_path()
    env_path = homedir / ".env"
    if not env_path.exists():
        return None
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    creds: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # MEMINI_* keys + DB_SSLROOTCERT (the config.py alias for the CA path,
        # written by write_admin_env when sslrootcert is provided).
        if key.startswith("MEMINI_") or key == "DB_SSLROOTCERT":
            creds[key] = value
    if "MEMINI_ADMIN_USER" not in creds or "MEMINI_ADMIN_PASSWORD" not in creds:
        return None
    return creds


def build_admin_dsn(creds: dict[str, str]) -> str:
    """Build an admin DSN from a credentials dict (from read_admin_credentials).

    Honors ``MEMINI_DB_SSLMODE`` and ``DB_SSLROOTCERT`` when present, matching
    what ``write_admin_env`` writes and what ``config.py`` reads.
    """
    user = creds.get("MEMINI_ADMIN_USER", _ADMIN_ROLE)
    password = creds.get("MEMINI_ADMIN_PASSWORD", "")
    host = creds.get("MEMINI_DB_HOST", "localhost")
    port = creds.get("MEMINI_DB_PORT", "5432")
    db_name = creds.get("MEMINI_DB_NAME", _DEFAULT_DB_NAME)
    sslmode = creds.get("MEMINI_DB_SSLMODE", "prefer")
    sslrootcert = creds.get("DB_SSLROOTCERT") or None
    return build_team_dsn(
        host,
        port,
        db_name,
        user,
        password,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )


async def check_db_exists(admin_dsn: str) -> bool:
    """Check whether the memini database already exists and has the schema.

    Connects via ``admin_dsn`` and probes for the ``memories`` table.
    """
    import asyncpg

    try:
        conn = await asyncpg.connect(admin_dsn)
    except Exception:  # pragma: no cover - environment-dependent
        return False
    try:
        exists = await conn.fetchval(
            "SELECT to_regclass('public.memories') IS NOT NULL"
        )
        return bool(exists)
    except Exception:  # pragma: no cover - environment-dependent
        return False
    finally:
        await conn.close()


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
        team_sslmode="prefer",
        team_sslrootcert=None,
        team_new_db=False,
        team_admin_user=None,
        team_admin_password=None,
        project_user=None,
        project_password=None,
        project_name=None,
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
    str | None, str | None, str | None, str | None, str | None, bool, str, str | None
]:
    """Prompt for team PostgreSQL connection details + SSL mode + CA cert.

    Returns ``(host, port, database, user, password, new_db, sslmode, sslrootcert)``.
    ``new_db`` is True when the user said the database doesn't exist yet
    (the admin/bootstrap flow). ``sslmode`` is one of prefer/require/verify-ca/
    verify-full/disable. ``sslrootcert`` is a validated PEM path when the
    user chose verify-ca or verify-full AND provided a CA path, otherwise None.
    """
    print("\n  Team PostgreSQL server connection:")
    host = _prompt_optional("    Host [localhost]: ") or "localhost"
    port = _prompt_optional("    Port [5432]: ") or "5432"
    database = _prompt_optional("    Database [memini]: ") or "memini"
    user = _prompt_optional("    User [postgres]: ") or "postgres"
    password = _prompt_optional("    Password (input visible): ")
    new_db = _prompt_yes_no(
        "    Is this a new database? (admin/bootstrap flow)", default=False
    )
    sslmode = _prompt_ssl_mode()
    sslrootcert = _prompt_ca_cert(sslmode)
    return host, port, database, user, password, new_db, sslmode, sslrootcert


def _prompt_ca_cert(sslmode: str) -> str | None:
    """Prompt for an optional CA certificate path when verify-ca/verify-full.

    Returns a validated PEM path, or ``None`` when the user skips or when the
    sslmode does not require a CA (prefer/require/disable). Loops on invalid
    input so the user can re-type the path without restarting the prompt flow.
    """
    if sslmode not in {"verify-ca", "verify-full"}:
        return None
    print(
        "\n  A CA certificate is recommended for "
        f"{sslmode!r} to verify the server's identity."
    )
    if not _prompt_yes_no("    Provide a CA cert path now", default=True):
        return None
    while True:
        raw = _prompt_optional("    CA cert path (PEM): ")
        if not raw:
            return None
        try:
            return validate_ca_cert(raw)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"    ✗ {exc}")
            print("    Try again, or press Enter to skip (no sslrootcert).")


def _prompt_ssl_mode() -> str:
    """Prompt for the database SSL mode. Default: ``prefer``."""
    print(
        "\n  SSL mode for the database connection:\n"
        "    1. prefer (default) — use SSL if available, fall back to plain\n"
        "    2. require — always use SSL, reject non-SSL connections\n"
        "    3. verify-ca — verify server certificate (CA)\n"
        "    4. verify-full — verify server certificate (CA + hostname)\n"
        "    5. disable — never use SSL\n"
    )
    choice = _prompt_choice(
        "    Enter 1-5 [1]: ", default="1", choices={"1", "2", "3", "4", "5"}
    )
    return {
        "1": "prefer",
        "2": "require",
        "3": "verify-ca",
        "4": "verify-full",
        "5": "disable",
    }[choice]


# ── TLS: CA cert validation + connection test (T-TLS-001) ────────────────────


_PEM_CERT_HEADER = "-----BEGIN CERTIFICATE-----"


def validate_ca_cert(path: str) -> str:
    """Validate that ``path`` exists and is a PEM-encoded certificate file.

    Returns the absolute path on success. Raises ``ValueError`` with a clear
    message when the path is missing, not a file, or not PEM-encoded.

    We only check for the PEM header (the most common format for CA bundles).
    We do NOT parse the cert with ``ssl``/``cryptography`` — that would add a
    hard dependency and reject valid-but-unusual encodings. The connection
    test (``test_team_connection``) is the real validation.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"CA cert path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"CA cert path is not a file: {path}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise OSError(f"Could not read CA cert file {path}: {exc}") from exc
    if _PEM_CERT_HEADER not in text:
        raise ValueError(
            f"CA cert file {path} does not appear to be PEM-encoded "
            f"(no '{_PEM_CERT_HEADER}' header found). "
            "PostgreSQL sslrootcert requires a PEM file."
        )
    return str(p.resolve())


@dataclass
class ConnectionTestResult:
    """Result of ``test_team_connection`` — structured success/failure."""

    success: bool
    sslmode: str
    sslrootcert: str | None
    dsn: str
    error: str | None


def build_team_dsn(
    host: str,
    port: str,
    database: str,
    user: str,
    password: str | None,
    *,
    sslmode: str = "prefer",
    sslrootcert: str | None = None,
) -> str:
    """Build a PostgreSQL DSN with optional ``sslmode`` + ``sslrootcert``.

    ``sslrootcert`` is appended to the query string when provided and the
    sslmode is not ``disable``. This mirrors what ``config.py`` reads
    (``DB_SSLMODE`` + ``DB_SSLROOTCERT``) and what ``database.py``'s
    ``_build_ssl_context`` uses to construct the SSL context.
    """
    dsn = f"postgresql://{user}:{password or ''}@{host}:{port}/{database}"
    if sslmode == "disable":
        return dsn
    params: list[str] = [f"sslmode={sslmode}"]
    if sslrootcert:
        params.append(f"sslrootcert={sslrootcert}")
    return f"{dsn}?{'&'.join(params)}"


async def test_team_connection(
    host: str,
    port: str,
    database: str,
    user: str,
    password: str | None,
    *,
    sslmode: str = "prefer",
    sslrootcert: str | None = None,
    timeout_seconds: float = 5.0,
) -> ConnectionTestResult:
    """Attempt a short connection to the team Postgres with the chosen TLS settings.

    Returns a structured ``ConnectionTestResult``. Never raises — failures
    (unreachable host, bad credentials, TLS handshake error) are captured in
    ``error`` so the caller can print a clear message without aborting the
    whole install. The connection is closed immediately after a successful
    ``SELECT 1``; the function never holds a connection open.
    """
    dsn = build_team_dsn(
        host,
        port,
        database,
        user,
        password,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )
    try:
        import asyncpg
    except ImportError:
        return ConnectionTestResult(
            success=False,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
            dsn=dsn,
            error="asyncpg not installed",
        )
    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 — timeout, connection, auth errors
        return ConnectionTestResult(
            success=False,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
            dsn=dsn,
            error=str(exc),
        )
    try:
        await conn.fetchval("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        return ConnectionTestResult(
            success=False,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
            dsn=dsn,
            error=str(exc),
        )
    finally:
        with contextlib.suppress(Exception):
            await conn.close()
    return ConnectionTestResult(
        success=True,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
        dsn=dsn,
        error=None,
    )


def _resolve_tls_from_flags(flags: argparse.Namespace) -> tuple[str, str | None]:
    """Map ``--tls`` / ``--tls-ca <path>`` / ``--no-tls`` flags to (sslmode, rootcert).

    Returns ``(sslmode, sslrootcert_or_None)`` when a TLS flag is set, or the
    sentinel ``("", None)`` when no TLS flag was passed (caller should prompt
    interactively). Validates the CA cert path when ``--tls-ca`` is given —
    raises on a missing/non-PEM file.
    """
    tls_ca = getattr(flags, "tls_ca", None)
    no_tls = getattr(flags, "no_tls", False)
    tls = getattr(flags, "tls", False)
    if no_tls:
        return ("disable", None)
    if tls_ca:
        resolved = validate_ca_cert(tls_ca)
        return ("verify-full", resolved)
    if tls:
        return ("require", None)
    return ("", None)  # no flag → caller prompts


def _print_connection_test(result: ConnectionTestResult) -> None:
    """Print a human-readable connection-test result. Non-fatal on failure."""
    if result.success:
        print("  ✓ Connection test passed (SELECT 1)")
        return
    print("  ⚠ Connection test FAILED — config will still be written.")
    print(f"    sslmode:    {result.sslmode}")
    if result.sslrootcert:
        print(f"    sslrootcert: {result.sslrootcert}")
    print(f"    DSN:        {result.dsn}")
    print(f"    error:      {result.error}")
    print(
        "    Hint: verify the host/port, credentials, and (for verify-* modes) "
        "that the server's certificate is signed by the CA you provided."
    )


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
            # Honor --tls / --tls-ca / --no-tls on --yes even without prompts.
            tls_mode, tls_ca = _resolve_tls_from_flags(flags)
            if tls_mode:
                config.team_sslmode = tls_mode
                config.team_sslrootcert = tls_ca
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
        # --tls / --tls-ca / --no-tls are shortcuts for the interactive SSL
        # answers, matching the --embedded / --team flag pattern. When none
        # of them is set, fall through to the full interactive prompt.
        flag_mode, flag_ca = _resolve_tls_from_flags(flags)
        if flag_mode:
            # Flag-driven TLS: still prompt for the non-TLS connection bits
            # (host/port/db/user/password/new-db) unless --yes was passed.
            print("\n  Team PostgreSQL server connection:")
            # Annotated as unions because the else-branch below rebinds these
            # from _prompt_team_connection(), which returns Optional strings.
            host: str | None = _prompt_optional("    Host [localhost]: ") or "localhost"
            port: str | None = _prompt_optional("    Port [5432]: ") or "5432"
            database: str | None = (
                _prompt_optional("    Database [memini]: ") or "memini"
            )
            user: str | None = _prompt_optional("    User [postgres]: ") or "postgres"
            password: str | None = _prompt_optional("    Password (input visible): ")
            new_db = _prompt_yes_no(
                "    Is this a new database? (admin/bootstrap flow)", default=False
            )
            sslmode = flag_mode
            sslrootcert = flag_ca
        else:
            host, port, database, user, password, new_db, sslmode, sslrootcert = (
                _prompt_team_connection()
            )
        config.team_host = cast("str", host)
        config.team_port = port
        config.team_database = database
        config.team_user = user
        config.team_password = password
        config.team_new_db = new_db
        config.team_sslmode = sslmode
        config.team_sslrootcert = sslrootcert

        # Container runtime detection (only relevant for team mode).
        runtime_result = check_container_runtime()
        if not runtime_result.has_any and not flags.yes:
            print("\n  ⚠ No container runtime detected (needed for the team")
            print("    observability stack — `memini-ai dashboard --team`).")
            if _prompt_yes_no("    Offer to install one now", default=True):
                offer_container_install()

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
    """Build the ``mcp.memini-ai-dev.env`` block from an InstallConfig.

    For team mode with RBAC (``config.project_user`` is set), the
    ``MEMINI_DB_URL`` uses ``{env:...}`` references so secrets stay in the
    ``.env`` file (never in opencode.json). For the legacy inline-credentials
    flow (existing DB, user typed a password at the prompt), the DSN is
    built directly and the SSL mode is appended unless ``disable``.

    For pgembed mode, ``MEMINI_DB_URL`` is the literal ``"pgembed"`` — no
    SSL (Unix socket only).
    """
    env: dict[str, str] = {
        "MEMINI_VECTOR_BACKEND": "pgembed"
        if config.backend == "pgembed"
        else "postgres-external",
    }

    if config.backend == "pgembed":
        env["MEMINI_DB_URL"] = "pgembed"
    elif config.project_user is not None:
        # RBAC flow: credentials live in .env, referenced via {env:...}.
        # opencode.json stays commit-safe.
        env["MEMINI_DB_URL"] = (
            "postgresql://{env:MEMINI_PROJECT_USER}:{env:MEMINI_PROJECT_PASSWORD}"
            "@{env:MEMINI_DB_HOST}:{env:MEMINI_DB_PORT}/{env:MEMINI_DB_NAME}"
            "?sslmode={env:MEMINI_DB_SSLMODE}"
        )
        if config.team_sslrootcert:
            # sslrootcert referenced via env so the path stays in .env (not
            # committed in opencode.json). Matches DB_SSLROOTCERT in config.py.
            env["MEMINI_DB_URL"] += "&sslrootcert={env:DB_SSLROOTCERT}"
        if config.project_name:
            env["MEMINI_PEER_ID"] = config.project_name
    else:
        # Legacy inline-credentials flow (existing DB, password at prompt).
        user = config.team_user or "postgres"
        password = config.team_password or ""
        host = config.team_host or "localhost"
        port = config.team_port or "5432"
        database = config.team_database or "postgres"
        sslmode = config.team_sslmode or "prefer"
        sslrootcert = config.team_sslrootcert
        env["MEMINI_DB_URL"] = build_team_dsn(
            host,
            port,
            database,
            user,
            password,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
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
    """Build the ``mcp.memini-ai-dev`` entry (shared by homedir + project).

    The ``command`` array uses ABSOLUTE paths (resolved at install time)
    so opencode can launch the MCP server without depending on the user's
    shell PATH.  See ``_resolve_memini_command()`` for the resolution order
    and the Session 53/54 bug this works around.
    """
    return {
        "memini-ai-dev": {
            "type": "local",
            "enabled": True,
            "command": _resolve_memini_command(),
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


# ── Absolute-path resolution for the MCP launch command ────────────────────
#
# Background (Session 53/54 bugfix): opencode spawns MCP servers with a clean
# non-interactive environment, which on most distros (Ubuntu/Debian/Fedora)
# does NOT include ``/home/<user>/.local/bin`` in ``$PATH``.  The bare
# ``["uvx", ...]`` command therefore fails instantly with
# "uvx: command not found" and the MCP server is never even started — every
# subsequent call to a memini-ai tool returns "tool not available".
#
# Fix: resolve ``uvx`` and ``memini-ai`` to their absolute paths at install
# time (when the shell PATH is full and reliable) and write those absolute
# paths into the ``opencode.json`` ``mcp.memini-ai-dev.command`` array.
# Falls back gracefully when neither is on PATH.


def _resolve_memini_command() -> list[str]:
    """Return the absolute-path command array to launch memini-ai over stdio.

    Every returned command array ends with ``--stdio`` so the spawned
    process speaks stdio JSON-RPC — the transport OpenCode's local MCP
    config type expects.  Since v1.0.0 (commit 74b81cf) a bare
    ``memini-ai`` invocation defaults to ``streamable-http``, which
    OpenCode cannot talk to and manifests as "server unavailable".

    Resolution order:
    1. ``shutil.which("uvx")`` — use that with
       ``--from memini-ai-dev memini-ai --stdio``.  Note: ``uvx`` reuses
       the cached tool environment for an unpinned ``--from memini-ai-dev``
       spec indefinitely — it does NOT re-resolve PyPI on each run.  To
       pick up a newly published release run
       ``uv tool upgrade memini-ai-dev`` (or pin an explicit ``==1.4.0``
       in the ``--from`` spec).  Confirmed live: cache pinned at 1.0.4
       despite 1.4.0 on PyPI.
    2. ``<home>/.local/bin/memini-ai`` — the local uv-tool install (what
       ``uv tool install memini-ai-dev`` or ``uvx --from memini-ai-dev
       memini-ai`` leaves behind). Fast, no re-download per opencode
       restart.
    3. The entry point of the currently-running Python — useful when the
       installer itself is running out of a dev checkout (``pip install -e``).
    4. Bare ``["uvx", ...]`` as a last resort, plus a printed warning.

    The home directory is evaluated on every call (not at module import)
    so tests that monkeypatch ``Path.home()`` work correctly.
    """
    uvx_path = shutil.which("uvx")
    if uvx_path:
        return [uvx_path, "--from", "memini-ai-dev", "memini-ai", "--stdio"]

    local_memini = Path.home() / ".local" / "bin" / "memini-ai"
    if local_memini.exists() and os.access(local_memini, os.X_OK):
        return [str(local_memini), "--stdio"]

    # Fall back to the running interpreter's memini-ai entry point.
    try:
        # Side-effect import: just need to know the module is importable, so
        # we can run ``python -m memini_ai.cli``.  noqa because we don't
        # reference any symbol — we just need the import to not crash.
        import memini_ai.cli  # noqa: F401, PLC0415  (intentional late import)

        # /.../site-packages/memini_ai/cli.py  →  invoke via the same Python
        runner = sys.executable
        return [runner, "-m", "memini_ai.cli", "--stdio"]
    except Exception:  # pragma: no cover — only hit if even the dev checkout is broken
        pass

    print(
        "  ⚠ WARNING: could not find 'uvx' on PATH or 'memini-ai' in "
        f"{Path.home() / '.local' / 'bin'}. The MCP server will use bare "
        "'uvx' which may fail if /home/<user>/.local/bin is not in the "
        "opencode spawn PATH."
    )
    return ["uvx", "--from", "memini-ai-dev", "memini-ai", "--stdio"]


# ── Package pre-download ────────────────────────────────────────────────────


def pre_download_package(dry_run: bool) -> PreDownloadResult:
    """Run the resolved memini-ai launch command with ``--help`` to warm any
    download/cache.  The first MCP server start will then be fast.

    Uses ``_resolve_memini_command()`` so we exercise the SAME command array
    opencode will use (not a bare ``uvx`` that may not be on opencode's PATH).
    """
    base_cmd = _resolve_memini_command()
    cmd = [*base_cmd, "--help"]
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
    """Optionally append a ``.env`` file with team credentials + API key.

    Mirrors neuralgentics' pattern: secrets go in .env, not opencode.json.

    For the RBAC flow (``config.project_user`` set, or ``config.team_new_db``
    with admin creds), the DB credentials are already written by
    ``write_admin_env`` / ``write_project_env`` (with chmod 600) — this
    function only appends the Ollama API key in that case.

    For the legacy inline-credentials flow (existing DB, password typed at
    the prompt, no RBAC), this appends individual ``MEMINI_DB_*`` vars plus
    the assembled ``MEMINI_DB_URL`` with SSL mode.
    """
    lines: list[str] = []
    if config.ollama_api_key:
        lines.append(f"OLLAMA_API_KEY={config.ollama_api_key}")

    # Only append DB credentials here for the legacy inline-credentials
    # flow. The RBAC flow (project_user set, or new-db admin bootstrap) has
    # already written a chmod-600 .env via write_admin_env / write_project_env.
    rbac_flow = config.project_user is not None or (
        config.backend == "team" and config.team_new_db
    )
    if config.backend == "team" and config.team_password and not rbac_flow:
        host = config.team_host or "localhost"
        port = config.team_port or "5432"
        database = config.team_database or "postgres"
        user = config.team_user or "postgres"
        sslmode = config.team_sslmode or "prefer"
        sslrootcert = config.team_sslrootcert
        dsn = build_team_dsn(
            host,
            port,
            database,
            user,
            config.team_password,
            sslmode=sslmode,
            sslrootcert=sslrootcert,
        )
        lines.append("# Team PostgreSQL credentials (saved from installer)")
        lines.append(f"MEMINI_DB_HOST={host}")
        lines.append(f"MEMINI_DB_PORT={port}")
        lines.append(f"MEMINI_DB_NAME={database}")
        lines.append(f"MEMINI_DB_SSLMODE={sslmode}")
        if sslrootcert:
            lines.append(f"DB_SSLROOTCERT={sslrootcert}")
        lines.append(f"MEMINI_DB_URL={dsn}")

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
    # 1. Run prompts FIRST so we know the backend before dep checks
    #    (container runtime only matters for team mode). We re-run the
    #    deps check after prompts so the user isn't warned about missing
    #    containers when they chose pgembed.
    config = run_prompts(args)

    # 2. Check system deps. For team backend, also check container runtime
    #    (soft-fail: missing container runtime is a warning, not a hard
    #    blocker — pgembed doesn't need containers, and the user may install
    #    one later via `memini-ai dashboard --team`).
    deps = check_system_deps(
        backend=config.backend if config.backend == "team" else None
    )
    hard_deps_ok = all(d.present for d in deps.deps if d.name in {"uv", "python>=3.12"})
    if not hard_deps_ok:
        print("\n  ✗ Missing system dependencies:\n")
        for dep in deps.deps:
            if dep.name in {"uv", "python>=3.12"} and not dep.present:
                print(f"    ✗ {dep.name}")
                if dep.install_command:
                    print(f"       Install:  {dep.install_command}")
        print("\n  Install the missing dependencies and re-run.")
        return 1

    # Soft-warn for missing container runtime (team mode only).
    if config.backend == "team":
        runtime_result = check_container_runtime()
        if not runtime_result.has_any:
            print(
                "\n  ⚠ No container runtime detected (needed for the team"
                " observability stack — `memini-ai dashboard --team`)."
            )
            print("    Install one of: podman (recommended), docker, containerd.")
            print("    The install will continue; you can add a runtime later.\n")
        else:
            present = [rt.name for rt in runtime_result.runtimes if rt.present]
            print(f"\n  Container runtime detected: {', '.join(present)}")

    # 3. Resolve config dir
    if mode == "homedir":
        config_dir = get_homedir_config_path()
    else:
        target = getattr(args, "target", None)
        config_dir = get_project_config_path(Path(target) if target else None)

    # 4. Create config dir
    if not args.dry_run:
        config_dir.mkdir(parents=True, exist_ok=True)

    # 5. RBAC: create admin/DB (new-db flow) or project/device user.
    if config.backend == "team" and not args.dry_run:
        _run_rbac_flow(config, mode, config_dir)

    # 6. Build opencode.json
    if mode == "homedir":
        opencode_json = build_homedir_opencode_json(config)
    else:
        opencode_json = build_project_opencode_json(config)
    content = json.dumps(opencode_json, indent=2) + "\n"

    # 7. Write with backup
    write_config_with_backup(
        config_dir, "opencode.json", content, args.force, args.dry_run
    )

    # 7b. Write .env if credentials/key provided (legacy flow + API key only;
    #     RBAC flow already wrote .env via write_admin_env/write_project_env).
    maybe_write_env_file(config_dir, config, args.dry_run)

    # 8. Write state file
    if not args.dry_run:
        files_manifest = {
            "opencode.json": compute_file_sha256(config_dir / "opencode.json")
            if (config_dir / "opencode.json").exists()
            else "",
        }
        write_state_file(config_dir, __version__, mode, files_manifest)

    # 9. Pre-download package
    print("\n  Pre-downloading memini-ai-dev package (warming uvx cache)...")
    pre_download = pre_download_package(args.dry_run)

    # 9b. Connection test (team mode only) — non-fatal.
    if config.backend == "team" and not args.dry_run:
        print("\n  Testing team PostgreSQL connection (5s timeout)...")
        result = asyncio.run(
            test_team_connection(
                host=config.team_host or "localhost",
                port=config.team_port or "5432",
                database=config.team_database or "postgres",
                user=config.team_user or "postgres",
                password=config.team_password,
                sslmode=config.team_sslmode or "prefer",
                sslrootcert=config.team_sslrootcert,
            )
        )
        _print_connection_test(result)

    # 10. Start embedded DB (homedir + pgembed only)
    if mode == "homedir" and config.backend == "pgembed" and not args.dry_run:
        print("\n  Starting embedded PostgreSQL (pgembed)...")
        start_embedded_db()

    # 11. Print summary
    print_summary(config, mode, config_dir, deps, pre_download)

    # 12. Offer next step
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


def _run_rbac_flow(config: InstallConfig, mode: str, config_dir: Path) -> None:
    """Execute the RBAC bootstrap / project-user creation flow (team mode).

    - New DB (homedir + ``team_new_db``): generate admin password, create
      admin role + DB + extensions + schema, write admin .env.
    - Existing DB, project mode: read admin creds from homedir .env, create
      a project-scoped role, write project .env.
    - Existing DB, homedir mode (device join): create a device role, write
      admin/device .env.

    Failures are caught and printed but do not abort the install — the
    opencode.json is still written so the user can fix the DB side and
    re-run. The RBAC flow is best-effort: if asyncpg isn't installed or the
    DB is unreachable, we print a clear message and continue.
    """
    try:
        if config.team_new_db:
            # Flow A: new team DB bootstrap.
            admin_password = config.team_admin_password or generate_password()
            config.team_admin_user = _ADMIN_ROLE
            config.team_admin_password = admin_password
            print(f"\n  Generated admin password: {admin_password}")
            print("  (Save this — it's printed only once.)")
            host = config.team_host or "localhost"
            port = int(config.team_port or "5432")
            db_name = config.team_database or _DEFAULT_DB_NAME
            sslmode = config.team_sslmode or "prefer"
            sslrootcert = config.team_sslrootcert
            # We need the postgres superuser password to bootstrap.
            superuser_password = (
                _prompt_optional("    postgres superuser password (for bootstrap): ")
                or ""
            )
            print(f"  Creating admin role + database '{db_name}'…")
            asyncio.run(
                create_admin_and_db(
                    host=host,
                    port=port,
                    superuser_password=superuser_password,
                    admin_password=admin_password,
                    db_name=db_name,
                    sslmode=sslmode,
                    sslrootcert=sslrootcert,
                )
            )
            print(f"  ✓ Created {_ADMIN_ROLE} role + '{db_name}' database")
            write_admin_env(
                config_dir,
                host,
                port,
                admin_password,
                db_name=db_name,
                sslmode=sslmode,
                sslrootcert=sslrootcert,
            )
            # The homedir also gets a default project user for itself.
            if mode == "homedir":
                admin_dsn = build_team_dsn(
                    host,
                    str(port),
                    db_name,
                    _ADMIN_ROLE,
                    admin_password,
                    sslmode=sslmode,
                    sslrootcert=sslrootcert,
                )
                project_name = sanitize_project_name(Path.home().name)
                role_name, project_pw = asyncio.run(
                    create_project_user(admin_dsn, project_name, db_name=db_name)
                )
                config.project_user = role_name
                config.project_password = project_pw
                config.project_name = role_name
                # Overwrite the admin .env with project creds appended so
                # the homedir MCP server uses the project role (least privilege).
                # Admin creds stay in the same file for re-bootstrap.
                _append_project_to_admin_env(config_dir, role_name, project_pw)
                print(f"  ✓ Created default project role '{role_name}' for homedir")

        elif mode == "project":
            # Flow B: existing DB, create a project-scoped role.
            admin_creds = read_admin_credentials()
            if not admin_creds:
                print(
                    "\n  ⚠ No admin credentials found in homedir .env.\n"
                    "    Run `memini-ai init --homedir --team --new-db` first,\n"
                    "    or copy the admin .env from the machine that bootstrapped\n"
                    "    the team DB. Skipping RBAC project-user creation.\n"
                )
                return
            admin_dsn = build_admin_dsn(admin_creds)
            project_name = sanitize_project_name(Path.cwd().name)
            print(f"  Creating project role '{project_name}'…")
            role_name, password = asyncio.run(
                create_project_user(admin_dsn, project_name)
            )
            config.project_user = role_name
            config.project_password = password
            config.project_name = role_name
            write_project_env(
                config_dir,
                role_name,
                password,
                host=admin_creds.get("MEMINI_DB_HOST", "localhost"),
                port=int(admin_creds.get("MEMINI_DB_PORT", "5432")),
                db_name=admin_creds.get("MEMINI_DB_NAME", _DEFAULT_DB_NAME),
                sslmode=admin_creds.get("MEMINI_DB_SSLMODE", "prefer"),
                sslrootcert=admin_creds.get("DB_SSLROOTCERT") or None,
            )
            print(f"  ✓ Created project role '{role_name}'")

        else:
            # Flow C: existing DB, homedir device join.
            admin_creds = read_admin_credentials()
            if not admin_creds:
                # Homedir already has admin creds from a --new-db run; nothing
                # to do. If neither admin nor project creds exist, the user is
                # probably on the legacy inline-credentials flow — skip RBAC.
                return
            admin_dsn = build_admin_dsn(admin_creds)
            hostname = platform.node() or "device"
            print(f"  Creating device role for hostname '{hostname}'…")
            role_name, password = asyncio.run(create_device_user(admin_dsn, hostname))
            config.project_user = role_name
            config.project_password = password
            config.project_name = role_name
            # Append device creds to the existing admin .env (homedir).
            _append_project_to_admin_env(config_dir, role_name, password)
            print(f"  ✓ Created device role '{role_name}'")

    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"\n  ⚠ RBAC flow failed: {exc}")
        print("    The opencode.json will still be written. Fix the DB issue")
        print("    and re-run `memini-ai init` to create the roles.")


def _append_project_to_admin_env(
    config_dir: Path, role_name: str, password: str
) -> None:
    """Append project/device credentials to an existing admin .env (homedir)."""
    env_path = config_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if "MEMINI_PROJECT_USER=" in existing:
        # Already has project creds — replace them.
        lines: list[str] = []
        for line in existing.splitlines():
            if line.startswith(("MEMINI_PROJECT_USER=", "MEMINI_PROJECT_PASSWORD=")):
                continue
            lines.append(line)
        existing = "\n".join(lines)
        if not existing.endswith("\n"):
            existing += "\n"
    project_block = (
        f"\n# memini-ai project/device credentials\n"
        f"MEMINI_PROJECT_USER={role_name}\n"
        f"MEMINI_PROJECT_PASSWORD={password}\n"
        f"MEMINI_PEER_ID={role_name}\n"
    )
    _write_env_file(env_path, existing + project_block)


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
