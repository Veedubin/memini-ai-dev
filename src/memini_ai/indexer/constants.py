"""Constants for project indexing - file filters and exclusion patterns."""

from __future__ import annotations

# Directories to skip during indexing
SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        ".eggs",
        "*.egg-info",
        "dist",
        "build",
        ".coverage",
        ".htmlcov",
        ".hypothesis",
        ".nox",
    }
)

# File extensions allowed for indexing (80+ extensions)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Python
        ".py",
        # JavaScript/TypeScript
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".mjs",
        ".cjs",
        ".mts",
        ".cts",
        # Java
        ".java",
        # C/C++
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".hh",
        # C#
        ".cs",
        # Go
        ".go",
        # Rust
        ".rs",
        # Ruby
        ".rb",
        # PHP
        ".php",
        # Swift
        ".swift",
        # Kotlin
        ".kt",
        ".kts",
        # Scala
        ".scala",
        ".sc",
        # Lua
        ".lua",
        # Perl
        ".pl",
        ".pm",
        # Shell
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".psm1",
        # Config formats
        ".json",
        ".jsonc",
        ".json5",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        # Markup
        ".xml",
        ".html",
        ".htm",
        ".xhtml",
        # Styles
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".styl",
        # Documentation
        ".md",
        ".markdown",
        ".rst",
        ".adoc",
        ".asciidoc",
        ".tex",
        ".bib",
        # Database
        ".sql",
        # GraphQL
        ".graphql",
        ".gql",
        # Protocol buffers
        ".proto",
        # Docker
        "dockerfile",
        ".dockerignore",
        # Git
        ".gitignore",
        ".gitattributes",
        ".gitkeep",
        # Environment
        ".env",
        ".env.example",
        ".env.local",
        ".env.development",
        ".env.production",
        # Build files
        "Makefile",
        "CMakeLists.txt",
        "setup.py",
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "poetry.lock",
        # Editor config
        ".editorconfig",
        ".prettierrc",
        ".prettierrc.json",
        ".prettierrc.yaml",
        ".eslintrc",
        ".eslintrc.json",
        ".eslintrc.yaml",
        ".babelrc",
        ".babelrc.json",
        ".browserslistrc",
        # Other
        ".npmrc",
        ".nvmrc",
        "tox.ini",
        "noxfile.py",
        ".envrc",
    }
)

# Files always excluded regardless of extension
ALWAYS_EXCLUDED: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        "~$*",
        "*.swp",
        "*.swo",
        "*.tmp",
        "*.temp",
    }
)

# Maximum file size for indexing (10MB default)
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Default chunk size (512 tokens)
DEFAULT_CHUNK_SIZE = 512

# Default chunk overlap (50 tokens)
DEFAULT_CHUNK_OVERLAP = 50
