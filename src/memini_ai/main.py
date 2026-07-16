"""Memini-ai v1.0.0 - Local-first semantic memory server.

The real CLI lives in :mod:`memini_ai.cli`; this module re-exports its
``main`` so existing ``python -m memini_ai.main`` invocations keep working
after the ``[project.scripts]`` entry point moved to ``memini_ai.cli:main``.
"""

from memini_ai.cli import main

if __name__ == "__main__":
    main()
