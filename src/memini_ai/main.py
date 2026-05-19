"""CLI entry point for memini-ai MCP server."""

from __future__ import annotations

import argparse
import asyncio
import sys

from memini_ai.server import server
from memini_ai.utils.logger import logger


def main() -> None:
    """Main entry point for the memini-ai CLI."""
    parser = argparse.ArgumentParser(
        description="memini-ai v3.0 - Local-first semantic memory server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as stdio MCP server instead of HTTP",
    )

    args = parser.parse_args()

    if args.stdio:
        # Run as stdio MCP server
        logger.info("starting_mcp_stdio")
        server.run(transport="stdio")
    else:
        # Run as HTTP server
        logger.info("starting_mcp_http", host=args.host, port=args.port)
        try:
            asyncio.run(server.run(transport="streamable-http", host=args.host, port=args.port))
        except KeyboardInterrupt:
            logger.info("server_interrupted")
            sys.exit(0)


if __name__ == "__main__":
    main()
