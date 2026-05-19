"""
mem0 Standard Memory MCP Server
═══════════════════════════════════════════════════

MCP tools for conversational AI agents. Exposes 12 tools that wrap
mem0's standard memory operations — creating, searching, updating,
and deleting memories about users, their preferences, and facts.

This server is designed for agents that interact with human users:
chatbots, personal assistants, customer-support agents.

Architecture
  Agent → MCP tools/call (port 8765) → httpx → FastAPI (:8888) → Memory SDK

Start
  python -m server.mcp_wrapper.standard_mcp_server
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from server.mcp_wrapper.shared import (
    MCP_STANDARD_PORT,
    _drop_none,
    _request,
    _require_scope,
    create_mcp_lifespan,
    logger,
)

# ═══════════════════════════════════════════════════════════════════════════
# Per-server httpx client container (set/cleared by lifespan)
# ═══════════════════════════════════════════════════════════════════════════
_http_container: dict = {"client": None}

# ═══════════════════════════════════════════════════════════════════════════
# FastMCP instance
# ═══════════════════════════════════════════════════════════════════════════
mcp = FastMCP(
    "mem0-standard-mcp",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
    port=MCP_STANDARD_PORT,
    lifespan=create_mcp_lifespan("standard", _http_container),
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper — each tool uses this to forward calls to the backend
# ═══════════════════════════════════════════════════════════════════════════


def _http():
    """Return the active httpx client (set by lifespan)."""
    return _http_container["client"]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — 1 tool
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def configure(config: Dict[str, Any]) -> dict:
    """Change the backend memory engine configuration at runtime.

    When to use: Only when the user explicitly asks to change the AI
    model, embedding provider, or vector database. This affects ALL
    subsequent requests for all users — use with caution.

    The config dict matches mem0's MemoryConfig schema:
      {
        "vector_store": {"provider": "qdrant", "config": {...}},
        "llm": {"provider": "openai", "config": {"model": "gpt-4o", ...}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", ...}}
      }
    """
    return await _request(_http(), "POST", "/configure", json=config)


# ═══════════════════════════════════════════════════════════════════════════
# Memories — Create
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def add_memory(
    messages: List[Dict[str, str]],
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    infer: bool = True,
    memory_type: str | None = None,
    prompt: str | None = None,
    vector_return_number: int = 2,
    graph_return_depth: int = 2,
) -> dict:
    """Store new memories about a user.

    When to use: Each time the user shares new information about
    themselves — preferences, facts, experiences, plans, opinions.
    Also use when the user explicitly asks you to remember something.
    Do NOT wait for the user to say "remember that."

    infer=True (default) — the full pipeline:
      1. LLM extracts facts from messages
      2. Searches existing memories for deduplication
      3. Decides ADD / UPDATE / DELETE / NONE for each fact
      4. Executes the decisions (vector store + optional graph store)

    infer=False — direct raw store, no LLM inference.

    Parameters:
      messages   — list of {"role": "user|assistant", "content": "..."}
      user_id    — scope identifier (at least one of user_id/agent_id/run_id required)
      agent_id   — scope identifier
      run_id     — scope identifier
      metadata   — optional custom metadata dict
      infer      — True: LLM extract + decide. False: direct store
      memory_type — "procedural_memory" for agent procedural, None for standard
      prompt     — optional custom LLM prompt for fact extraction
      vector_return_number — top-k candidates for vector recall (default 2)
      graph_return_depth   — graph search hops (default 2)

    Example:
      messages=[{"role": "user", "content": "I prefer dark roast coffee and short morning workouts."}]
      user_id="alice-001"
    """
    _require_scope(user_id, agent_id, run_id)

    body = _drop_none({
        "messages": messages,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": metadata,
        "infer": infer,
        "memory_type": memory_type,
        "prompt": prompt,
        "vector_return_number": vector_return_number,
        "graph_return_depth": graph_return_depth,
    })

    return await _request(_http(), "POST", "/memories", json=body)


# ═══════════════════════════════════════════════════════════════════════════
# Memories — Read
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def list_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Get all memories within a scope.

    When to use: When you need a full overview of everything remembered
    about a user, agent, or run before making decisions. Use this sparingly
    — for targeted lookups, prefer search_memories.

    Example:
      user_id="alice-001"
    """
    _require_scope(user_id, agent_id, run_id)
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request(_http(), "GET", "/memories", params=params)


@mcp.tool()
async def get_memory(memory_id: str) -> dict:
    """Get a single memory by its ID.

    When to use: You already have a memory ID from a previous search
    or list result, and need its full content.

    Example:
      memory_id="abc123-def456"
    """
    return await _request(_http(), "GET", f"/memories/{memory_id}")


@mcp.tool()
async def memory_history(memory_id: str) -> dict:
    """Get the change history of a memory.

    When to use: When you need to trace how a memory has evolved over
    time — all ADD, UPDATE, DELETE events for that memory.

    Example:
      memory_id="abc123-def456"
    """
    return await _request(_http(), "GET", f"/memories/{memory_id}/history")


# ═══════════════════════════════════════════════════════════════════════════
# Memories — Update
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def update_memory(memory_id: str, updated: Dict[str, Any]) -> dict:
    """Update the content of an existing memory.

    When to use: When the user corrects or changes information they
    previously shared. For example: "I don't like dark roast anymore,
    I prefer light roast now."

    Example:
      memory_id="abc123-def456"
      updated={"memory": "I prefer light roast coffee now."}
    """
    return await _request(_http(), "PUT", f"/memories/{memory_id}", json=updated)


# ═══════════════════════════════════════════════════════════════════════════
# Memories — Delete
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def delete_memory(memory_id: str) -> dict:
    """Delete a single memory by ID.

    When to use: When the user explicitly asks to forget a specific
    piece of information, or when you discover a stored memory is
    incorrect and needs removal.

    Example:
      memory_id="abc123-def456"
    """
    return await _request(_http(), "DELETE", f"/memories/{memory_id}")


@mcp.tool()
async def delete_all_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Delete ALL memories in a scope.

    When to use: When the user asks to "clear my memory" or "forget
    everything about me." This is a bulk destructive operation.

    A scope identifier is REQUIRED to prevent accidental global deletion.
    To wipe the entire database for all users, use reset_all instead.

    Example:
      user_id="alice-001"
    """
    _require_scope(user_id, agent_id, run_id)
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request(_http(), "DELETE", "/memories", params=params)


# ═══════════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search_memories(
    query: str,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 100,
    filters: Dict[str, Any] | None = None,
) -> dict:
    """Semantic search through stored memories.

    When to use: BEFORE answering any question that might relate to
    previously stored information. Always search first — don't rely
    on your own memory. This is the primary retrieval tool.

    The backend uses vector similarity search with optional graph
    traversal and reranking. Set filters to narrow the search scope.

    Parameters:
      query     — natural-language search query (e.g. "coffee preferences")
      user_id   — scope identifier
      limit     — max results (default 100; use 5-10 for focused queries)
      filters   — optional metadata filters, e.g. {"actor_id": "John"}

    Example:
      query="coffee preferences"
      user_id="alice-001"
      limit=5
    """
    _require_scope(user_id, agent_id, run_id)
    body = _drop_none({
        "query": query,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "limit": limit,
        "filters": filters,
    })
    return await _request(_http(), "POST", "/search", json=body)


# ═══════════════════════════════════════════════════════════════════════════
# Summaries
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def start_summary(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 200,
    trigger: str = "manual",
) -> dict:
    """Trigger background memory summary generation.

    When to use: After a long conversation ends, or when the user asks
    to "summarize what you remember about me." This only triggers the
    background job — use get_summary to retrieve the result.

    Example:
      user_id="alice-001"
      limit=200
    """
    _require_scope(user_id, agent_id, run_id)
    body = _drop_none({
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "limit": limit,
        "trigger": trigger,
    })
    return await _request(_http(), "POST", "/start_mem_summary", json=body)


@mcp.tool()
async def get_summary(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Get the latest generated memory summary.

    When to use: After calling start_summary, or at the start of a
    conversation to check if a summary already exists. Returns an
    empty structure if no summary has been generated yet.

    Example:
      user_id="alice-001"
    """
    _require_scope(user_id, agent_id, run_id)
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request(_http(), "GET", "/get_summary", params=params)


# ═══════════════════════════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def reset_all() -> dict:
    """Completely reset the entire memory store.

    When to use: ONLY when the user explicitly asks to wipe all
    stored memories across all users. This is an irreversible
    destructive operation. For scoped deletion, use delete_all_memories.
    """
    return await _request(_http(), "POST", "/reset")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
