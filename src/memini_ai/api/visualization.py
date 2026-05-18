"""FastAPI server for knowledge graph visualization.

Provides REST endpoints for:
- GET /api/graph - Full graph data for D3.js
- GET /api/graph/stats - Entity statistics
- GET /api/graph/entity/{entity_id} - Single entity details
- GET / - HTML page with live-updating D3.js visualization
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from memini_ai.api.d3_template import generate_live_html
from memini_ai.config import get_config
from memini_ai.postgres.database import PostgresDatabase


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - init and cleanup."""
    # Startup: Initialize PostgreSQL connection
    config = get_config()
    if config.db_url:
        app.state.db = PostgresDatabase(config.db_url)
        await app.state.db.initialize()
    else:
        app.state.db = None

    yield

    # Shutdown: Close PostgreSQL connection
    if app.state.db:
        await app.state.db.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Memini-ai Knowledge Graph API",
        description="Live knowledge graph visualization API backed by PostgreSQL",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def get_visualization() -> str:
        """Serve the live D3.js visualization page."""
        return generate_live_html()

    @app.get("/api/graph")
    async def get_graph(limit: int = 1000) -> dict[str, Any]:
        """Get full graph data for D3.js visualization.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            Dict with nodes and edges arrays.
        """
        if not app.state.db:
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL not configured. Set MEMINI_DB_URL environment variable.",
            )

        try:
            nodes, edges = await app.state.db.get_entities_with_relationships(limit=limit)
            return {"nodes": nodes, "edges": edges, "count": len(nodes)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/graph/stats")
    async def get_graph_stats() -> dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Entity counts by type and total relationships.
        """
        if not app.state.db:
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL not configured. Set MEMINI_DB_URL environment variable.",
            )

        try:
            stats = await app.state.db.get_entity_stats()
            # Get relationship count
            _nodes, edges = await app.state.db.get_entities_with_relationships(limit=10000)
            stats["total_relationships"] = len(edges)
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/graph/entity/{entity_id}")
    async def get_entity(entity_id: str) -> dict[str, Any]:
        """Get a single entity by ID.

        Args:
            entity_id: The entity UUID.

        Returns:
            Entity details with relationships.
        """
        if not app.state.db:
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL not configured. Set MEMINI_DB_URL environment variable.",
            )

        try:
            entity = await app.state.db.get_entity(entity_id)
            if not entity:
                raise HTTPException(status_code=404, detail="Entity not found")

            # Get relationships for this entity
            _nodes, edges = await app.state.db.get_entities_with_relationships(limit=10000)
            entity_rels = [
                e for e in edges
                if e["source"] == entity_id or e["target"] == entity_id
            ]

            return {
                "entity": entity,
                "relationships": entity_rels,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "postgres": "connected" if app.state.db else "not configured",
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server using uvicorn.

    Args:
        host: Host to bind to.
        port: Port to listen on.
    """
    import uvicorn

    uvicorn.run("memini_ai.api.visualization:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    run_server()
