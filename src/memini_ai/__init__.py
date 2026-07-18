"""Memini-ai - Local-first semantic memory server."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("memini-ai-dev")
except PackageNotFoundError:  # pragma: no cover — only when running from source tree
    __version__ = "1.2.2+local"

del _pkg_version
del PackageNotFoundError
