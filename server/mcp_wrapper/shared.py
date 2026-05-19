"""
mem0 MCP Wrapper — shared utilities
═══════════════════════════════════════════════════

Stateless shared module providing config constants, HTTP forwarding,
scope validation, and a lifespan factory used by both
standard_mcp_server.py and process_mcp_server.py.

This module has NO module-level mutable state. Each server file
manages its own httpx AsyncClient via its own lifespan.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mem0.mcp_wrapper")

# ---------------------------------------------------------------------------
# Config constants — environment variable injection with defaults
# ---------------------------------------------------------------------------
MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:8888")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "my_very_long_custom_key_123456")
MCP_STANDARD_PORT = int(os.getenv("MEM0_MCP_PORT", "8765"))
MCP_PROCESS_PORT = int(os.getenv("MEM0_PROCESS_MCP_PORT", "8766"))
DEFAULT_HEADERS = {"X-API-Key": MEM0_API_KEY}

if MEM0_API_KEY == "my_very_long_custom_key_123456":
    logger.warning(
        "MEM0_API_KEY is using the default value. "
        "Set MEM0_API_KEY for production use."
    )


# ---------------------------------------------------------------------------
# HTTP forwarding — stateless: takes httpx client as explicit parameter
# ---------------------------------------------------------------------------

async def _request(http: httpx.AsyncClient, method: str, path: str, **kw: Any) -> dict:
    """Unified HTTP request forwarder.

    Translates httpx errors into structured dicts so the LLM can read
    error details, rather than receiving opaque JSON-RPC errors.

    Args:
        http: An active httpx.AsyncClient (from the caller's lifespan).
        method: "GET" | "POST" | "PUT" | "DELETE"
        path: Relative path, e.g. "/memories" or "/memories/abc123"
        **kw: Forwarded to httpx.AsyncClient.request() (json=..., params=...)

    Returns:
        On success: the backend's JSON response dict.
        On HTTP error: {"error": True, "status": <int>, "detail": "<body>"}
        On network error: {"error": True, "status": None, "detail": "<message>"}
    """
    try:
        r = await http.request(method, path, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {
            "error": True,
            "status": e.response.status_code,
            "detail": e.response.text,
        }
    except httpx.RequestError as e:
        return {
            "error": True,
            "status": None,
            "detail": str(e),
        }


# ---------------------------------------------------------------------------
# Scope validation — raises ValueError so the LLM knows it's a parameter error
# ---------------------------------------------------------------------------

def _require_scope(user_id: Any, agent_id: Any, run_id: Any) -> None:
    """Validate that at least one session-scope identifier is provided.

    Raises ValueError immediately (before any HTTP call) when all three
    are falsy. FastMCP converts ValueError into a JSON-RPC error the LLM
    can parse and retry with corrected parameters.
    """
    if not any([user_id, agent_id, run_id]):
        raise ValueError(
            "At least one of user_id, agent_id, or run_id is required."
        )


# ---------------------------------------------------------------------------
# Dict filter — removes None-valued keys before sending to the backend
# ---------------------------------------------------------------------------

def _drop_none(d: dict) -> dict:
    """Return a new dict with all None-valued keys removed."""
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Lifespan factory — each server gets its own httpx client lifecycle
# ---------------------------------------------------------------------------

def create_mcp_lifespan(name: str, http_container: dict):
    """Create a lifespan async context manager for FastMCP.

    The lifespan manages the httpx.AsyncClient connection pool:
      - On startup: creates the client, stores it in http_container.
      - On shutdown: clears the reference (client auto-closes).

    Args:
        name: Human-readable label for log messages ("standard" / "process").
        http_container: A mutable dict that will hold {"client": AsyncClient}.
            The caller keeps a reference to this dict and passes
            http_container["client"] to _request() in each tool.

    Returns:
        An async context manager suitable for FastMCP(lifespan=...).
    """

    @contextlib.asynccontextmanager
    async def lifespan(server: FastMCP) -> None:  # noqa: ARG001
        async with httpx.AsyncClient(
            base_url=MEM0_BASE_URL,
            headers=DEFAULT_HEADERS,
            timeout=httpx.Timeout(30.0),
        ) as http:
            http_container["client"] = http
            logger.info("[%s] httpx client connected to %s", name, MEM0_BASE_URL)
            yield
        http_container["client"] = None

    return lifespan
