"""Knowledge Graph Visualization API.

This package provides a FastAPI server for live knowledge graph visualization.
It serves D3.js force-directed graph data directly from PostgreSQL.
"""

from memini_ai.api.visualization import create_app

__all__ = ["create_app"]
