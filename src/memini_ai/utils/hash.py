"""SHA-256 hashing utilities for content deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_content(content: str) -> str:
    """
    Generate SHA-256 hash of string content.

    Args:
        content: The string content to hash.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_file(file_path: str | Path) -> str:
    """
    Generate SHA-256 hash of file contents.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected file, got directory: {file_path}")

    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()
