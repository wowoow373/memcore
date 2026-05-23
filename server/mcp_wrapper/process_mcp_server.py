"""
mem0 Process Memory MCP Server
═══════════════════════════════════════════════════

MCP tools for code-execution AI agents. Exposes 2 tools that wrap
mem0's process memory operations — storing completed task step
summaries (Flow 1) and searching past task experience (Flow 2).

This server is designed for agents that execute multi-step coding
tasks: code generators, autonomous dev agents, task planners.
It is NOT for conversational / user-facing agents.

Three-layer process memory (see memory-api-spec.md):
  Graph   — Neo4j (:Step)-[:DEPENDS_ON]->(:Step)  fine-grained step nodes
  Chunk   — vector store (memory_type="process_chunk")  sub-goal groups
  Summary — vector store (memory_type="process_summary")  full task chains

Flow 1 (write):  After task completion → store summaries for future recall
Flow 2 (search): During task execution  → look up past experience (read-only)

Architecture
  Code Agent → MCP tools/call (port 8766) → httpx → FastAPI (:8888) → Memory SDK

Start
  python -m server.mcp_wrapper.process_mcp_server
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from server.mcp_wrapper.shared import (
    MCP_PROCESS_PORT,
    _drop_none,
    _request,
    _require_scope,
    create_mcp_lifespan,
)

# ═══════════════════════════════════════════════════════════════════════════
# Per-server httpx client container (set/cleared by lifespan)
# ═══════════════════════════════════════════════════════════════════════════
_http_container: dict = {"client": None}


def _http():
    """Return the active httpx client (set by lifespan)."""
    return _http_container["client"]


# ═══════════════════════════════════════════════════════════════════════════
# FastMCP instance
# ═══════════════════════════════════════════════════════════════════════════
mcp = FastMCP(
    "mem0-process-mcp",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
    port=MCP_PROCESS_PORT,
    lifespan=create_mcp_lifespan("process", _http_container),
)


# ═══════════════════════════════════════════════════════════════════════════
# Flow 1 — Write completed task summaries
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def write_process_memory(
    messages: List[Dict[str, Any]],
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> dict:
    """Store completed task step summaries for future recall.

    When to use: AFTER finishing an entire task (all steps complete),
    call this ONCE with the full array of step summaries. This triggers
    a pipeline that searches existing process memory for deduplication,
    calls an LLM to decide ADD/UPDATE/MERGE/NONE across three layers,
    and writes the results.

    Do NOT call this mid-task or for individual steps — use
    search_process_memory for that.

    --- messages format ---
    Each element is a step summary produced during task execution:

      {
        "Goal": "Add user authentication",
        "Step": "03 - Create auth.py",
        "Action": "create_file(path='auth.py')",
        "Dependency": [
          {"step_id": "01 - Read main.py", "description": "Parse entry logic"}
        ],
        "Brief": "Create auth.py and implement login/logout functions"
      }

    Field descriptions:
      - Goal  (required) — sub-goal. Steps with the same Goal are grouped
        into one Chunk in the vector store.
      - Step  (required) — unique step name. Becomes the :Step node name
        in the Neo4j graph.
      - Action (required) — the tool call string for reproducibility.
      - Dependency (optional, default []) — upstream steps this step
        depends on. Each entry becomes a DEPENDS_ON edge in the graph.
        Format: {"step_id": "...", "description": "..."}
      - Brief (required) — short semantic description. Used for
        embedding-based graph node matching during search.

    --- What happens internally (5-step LangGraph pipeline) ---
      1. preprocess — parses summaries into goals, steps, dependencies
      2. search     — three-layer dedup via ProcessMemorySearchEngine
      3. decide     — one LLM call decides ADD/UPDATE/MERGE/NONE for all
                      three layers
      4. execute    — writes Graph (Step nodes + DEPENDS_ON edges),
                      Chunks (goal vector store), and Summary (full chain)
      5. assemble   — packages results + recalled for the response

    --- Return format ---
      {
        "results": {
          "graph": {
            "deleted_entities": [...],
            "added_entities": [
              {"source": "01 - Read main.py", "relationship": "DEPENDS_ON",
               "destination": "03 - Create auth.py"}
            ]
          },
          "chunks": [
            {"id": "uuid", "goal": "Add user authentication", "event": "ADD"},
            {"id": "uuid", "goal": "Setup database", "event": "MERGE"}
          ],
          "summary": {
            "id": "uuid", "event": "ADD",
            "task_description": "Implement complete user auth system"
          }
        },
        "recalled": {
          "graph": {"chains": [...]},
          "chunks": [...],
          "summaries": [...]
        }
      }

      - results: what was actually written (event = ADD|UPDATE|MERGE|DELETE|NONE)
      - recalled: what already existed before this write (for agent awareness)

    --- Decision validation (internal) ---
      - ADD: id is set to null (new record created)
      - UPDATE: id must exist in recalled, otherwise downgraded to ADD
      - MERGE (chunks only): merge_with must exist in recalled chunk ids,
        otherwise downgraded to ADD
      - If recalled is empty → all events become ADD

    --- Error responses ---
      Missing scope:
        Tool execution error: "At least one of user_id, agent_id, or run_id
        is required." (raised before any HTTP call)

      Process memory not configured (backend):
        {"error": true, "status": 500,
         "detail": "Process memory is not configured..."}

      Invalid messages format (Pydantic validation):
        {"error": true, "status": 422,
         "detail": "[{\"type\": \"missing\", \"loc\": [\"body\", \"messages\"],
                     \"msg\": \"Field required\", ...}]"}

      Backend unreachable:
        {"error": true, "status": null,
         "detail": "All connection attempts failed"}

    --- Boundary cases ---
      - messages=[] → returns empty results, no writes
      - All Dependencies empty → Step nodes created, no edges
      - All summaries duplicate existing ones → all events are NONE
      - LLM returns invalid JSON → empty decisions, no writes
      - Chunk write fails → that layer is empty, other layers still succeed
      - Submitting identical summaries twice → second call produces
        MERGE/UPDATE events, no duplicate data

    Example call:
      messages=[
        {
          "Goal": "Add user authentication",
          "Step": "01 - Read main.py",
          "Action": "read_file(path='main.py')",
          "Dependency": [],
          "Brief": "Read main.py to understand entry point logic"
        },
        {
          "Goal": "Add user authentication",
          "Step": "03 - Create auth.py",
          "Action": "create_file(path='auth.py')",
          "Dependency": [
            {"step_id": "01 - Read main.py", "description": "Parse entry"}
          ],
          "Brief": "Create auth.py with login/logout implementation"
        }
      ],
      user_id="dev-001",
      run_id="task-run-42"
    """
    _require_scope(user_id, agent_id, run_id)

    body = _drop_none({
        "messages": messages,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": metadata,
    })

    return await _request(_http(), "POST", "/process-memories", json=body)


# ═══════════════════════════════════════════════════════════════════════════
# Flow 2 — Search process memory during task execution
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
async def search_process_memory(
    current_step: Dict[str, str],
    previous_step: Dict[str, str] | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    task_estimate: str | None = None,
    graph_hop: int = 1,
    chunk_top_k: int = 5,
    summary_top_k: int = 3,
    semantic_threshold: float = 0.6,
) -> dict:
    """Search process memory for relevant past task experience.

    When to use: BEFORE executing each step in a task. Call this to
    look up how similar steps were handled in past tasks — what
    dependencies they had, what sub-goals were involved, what the
    full task execution chain looked like.

    This is READ-ONLY. No data is written. Safe to call at any point
    during task execution without side effects.

    --- Three-layer parallel search ---

    1. Graph (Step node matching):
       - Your current_step.Brief is embedded and compared against
         Brief embeddings of all past Step nodes (cosine similarity).
       - Matched nodes are expanded 1-hop via DEPENDS_ON edges to
         discover dependency chains.
       - If previous_step is provided, results are further filtered
         by semantic similarity to the previous step's Brief.

    2. Chunk (Goal group recall):
       - Your current_step.Goal is embedded and searched against
         all past process_chunk vectors (groups of steps sharing a goal).

    3. Summary (Full task recall):
       - If task_estimate is provided, it is embedded; otherwise falls
         back to concatenating current_step.Goal + current_step.Brief.
       - Searched against past process_summary vectors (full task chains).

    --- current_step format ---
      {
        "Goal": "Add user authentication",
        "Step": "03 - Create auth.py",
        "Action": "create_file(path='auth.py')",
        "Brief": "Create auth.py and implement login/logout functions"
      }

      - Goal  (required) — sub-goal, used for chunk recall
      - Step  (required) — step name
      - Action (optional, default "") — tool call string
      - Brief (required) — semantic description, used for graph node matching

    --- previous_step format (optional) ---
      Same structure as current_step. When provided, candidate graph
      nodes are filtered: only nodes whose Brief embedding has cosine
      similarity >= semantic_threshold to previous_step.Brief are kept.

    --- Parameters ---
      current_step   — the step you are about to execute
      previous_step  — the step you just completed (enables semantic filter)
      user_id        — scope identifier (at least one required)
      agent_id       — scope identifier
      run_id         — scope identifier
      task_estimate  — optional task type description for summary search.
                       Falls back to Goal + Brief concatenation if None.
      graph_hop      — neighbor expansion hops from matched Step nodes
                       (default 1, range 0-5)
      chunk_top_k    — number of chunk results to retrieve (default 5, range 1-20)
      summary_top_k  — number of summary results (default 3, range 1-20)
      semantic_threshold — minimum cosine similarity for previous_step
                           filtering (default 0.6, range 0.0-1.0)

    --- Return format ---
      {
        "graph": {
          "matched_nodes": [
            {
              "name": "03 - Create auth.py",
              "brief": "Create auth.py with login/logout",
              "goal": "Add user authentication",
              "step": "03 - Create auth.py",
              "action": "create_file(path='auth.py')",
              "score": 0.92
            }
          ],
          "expanded_nodes": [
            {"source": "01 - Read main.py",
             "relationship": "DEPENDS_ON",
             "destination": "03 - Create auth.py"}
          ],
          "filtered_nodes": [...]
        },
        "chunks": [
          {
            "goal": "Add user authentication",
            "score": 0.85,
            "steps": [
              {"step": "01 - Read main.py", "brief": "..."},
              {"step": "03 - Create auth.py", "brief": "..."}
            ],
            "id": "chunk-uuid",
            "metadata": {...}
          }
        ],
        "summaries": [
          {
            "task_description": "Implement complete user auth system",
            "score": 0.78,
            "full_chain": [
              {"step": "01 - Read main.py", "brief": "..."},
              {"step": "02 - Create config.py", "brief": "..."},
              {"step": "03 - Create auth.py", "brief": "..."}
            ],
            "id": "summary-uuid",
            "metadata": {...}
          }
        ]
      }

    --- Error responses ---
      Missing scope:
        Tool execution error: "At least one of user_id, agent_id, or run_id
        is required."

      Process memory not configured:
        {"error": true, "status": 500,
         "detail": "Process memory is not configured..."}

      Invalid current_step (missing required fields):
        {"error": true, "status": 422,
         "detail": "[{\"type\": \"missing\", \"loc\": [\"body\", \"current_step\",
                     \"Goal\"], \"msg\": \"Field required\", ...}]"}

      Backend unreachable / timeout:
        {"error": true, "status": null, "detail": "<network error message>"}

    --- Boundary cases ---
      - No matching past steps → all three layers return empty arrays
      - graph_hop=0 → no neighbor expansion, expanded_nodes is empty
      - previous_step=None → semantic filter is skipped, filtered_nodes
        equals matched_nodes
      - task_estimate=None → falls back to Goal + Brief for summary search
      - Brief is empty string → graph search returns empty (no embedding)
      - semantic_threshold=1.0 → only exact cosine matches pass filter,
        filtered_nodes will be empty in almost all cases
      - Graph DB offline → graph layer returns empty, chunks/summaries
        still work if vector store is available
      - Vector store offline → chunks and summaries return empty,
        graph layer still works if Neo4j is available

    Example call:
      current_step={
        "Goal": "Add user authentication",
        "Step": "03 - Create auth.py",
        "Action": "create_file(path='auth.py')",
        "Brief": "Create auth.py and implement login/logout functions"
      },
      previous_step={
        "Goal": "Add user authentication",
        "Step": "01 - Read main.py",
        "Action": "read_file(path='main.py')",
        "Brief": "Read main.py to understand the entry point logic"
      },
      user_id="dev-001",
      run_id="task-run-42",
      task_estimate="Implement user authentication system",
      graph_hop=1
    """
    _require_scope(user_id, agent_id, run_id)

    body = _drop_none({
        "current_step": current_step,
        "previous_step": previous_step,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "task_estimate": task_estimate,
        "graph_hop": graph_hop,
        "chunk_top_k": chunk_top_k,
        "summary_top_k": summary_top_k,
        "semantic_threshold": semantic_threshold,
    })

    return await _request(_http(), "POST", "/process-memories/search", json=body)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
