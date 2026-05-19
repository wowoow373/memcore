import logging
import os
import secrets
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

# ── Auth ────────────────────────────────────────────────────────────────────

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
MIN_KEY_LENGTH = 16

if not ADMIN_API_KEY:
    logging.warning(
        "ADMIN_API_KEY not set - API endpoints are UNSECURED! "
        "Set ADMIN_API_KEY environment variable for production use."
    )
else:
    if len(ADMIN_API_KEY) < MIN_KEY_LENGTH:
        logging.warning(
            "ADMIN_API_KEY is shorter than %d characters - consider using a longer key for production.",
            MIN_KEY_LENGTH,
        )
    logging.info("API key authentication enabled")

# ── Env vars ─────────────────────────────────────────────────────────────────

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")
POSTGRES_PROCESS_COLLECTION = os.environ.get("POSTGRES_PROCESS_COLLECTION", "process_memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_LLM_API_KEY = os.environ.get("OPENAI_llm_API_KEY") or OPENAI_API_KEY
OPENAI_LLM_MODEL = os.environ.get("OPENAI_llm_Model", "gpt-4.1-nano-2025-04-14")
OPENAI_LLM_URL = os.environ.get("OPENAI_llm_URL")
OPENAI_EMBEDDER_API_KEY = os.environ.get("OPENAI_EMBEDDER_API_KEY") or OPENAI_API_KEY
OPENAI_EMBEDDER_MODEL = os.environ.get("OPENAI_EMBEDDER_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDER_URL = os.environ.get("OPENAI_EMBEDDER_URL")

HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# ── Default config ───────────────────────────────────────────────────────────

_llm_config: dict = {
    "provider": "openai",
    "config": {
        "api_key": OPENAI_LLM_API_KEY,
        "model": OPENAI_LLM_MODEL,
        "temperature": float(os.environ.get("OPENAI_llm_temperature", "0.2")),
    },
}
if OPENAI_LLM_URL:
    _llm_config["config"]["openai_base_url"] = OPENAI_LLM_URL

_embedder_config: dict = {
    "provider": "openai",
    "config": {
        "api_key": OPENAI_EMBEDDER_API_KEY,
        "model": OPENAI_EMBEDDER_MODEL,
        "embedding_dims": 1536,
    },
}
if OPENAI_EMBEDDER_URL:
    _embedder_config["config"]["openai_base_url"] = OPENAI_EMBEDDER_URL

_process_memory_config = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_PROCESS_COLLECTION,
            "embedding_model_dims": 1536,
            "hnsw": True,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "graph_search_depth": 10,
    "chunk_top_k": 5,
    "summary_top_k": 3,
    "semantic_filter_threshold": 0.6,
}

DEFAULT_CONFIG: dict = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
            "embedding_model_dims": 1536,
            "hnsw": True,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "llm": _llm_config,
    "embedder": _embedder_config,
    "history_db_path": HISTORY_DB_PATH,
    "process_memory": _process_memory_config,
}

MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mem0 REST APIs",
    description=(
        "A REST API for managing and searching memories for your AI Agents and Apps.\n\n"
        "## Authentication\n"
        "When the ADMIN_API_KEY environment variable is set, all endpoints require "
        "the `X-API-Key` header for authentication.\n\n"
        "## New: Process Memory\n"
        "Process memory captures step-level task execution experience. "
        "Use `POST /process-memories` to store task summaries, "
        "and `POST /process-memories/search` to recall relevant past experience during task execution."
    ),
    version="2.0.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if ADMIN_API_KEY:
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="X-API-Key header is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not secrets.compare_digest(api_key, ADMIN_API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return api_key


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models  —  every request model carries a json_schema_extra example
# so the Swagger "Try it out" button auto-populates usable test data.
# ══════════════════════════════════════════════════════════════════════════════


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")

    model_config = {
        "json_schema_extra": {
            "example": {"role": "user", "content": "My name is John and I live in New York."}
        }
    }


# ── Standard memory models ───────────────────────────────────────────────────

_standard_add_example = {
    "messages": [
        {"role": "user", "content": "My name is John and I live in New York."},
        {"role": "assistant", "content": "Nice to meet you John! How can I help you today?"},
    ],
    "user_id": "test-user-001",
    "agent_id": None,
    "run_id": None,
    "metadata": None,
    "infer": True,
    "memory_type": None,
}

_standard_add_dedup_example = {
    "messages": [
        {"role": "user", "content": "I also like to play tennis on weekends."},
    ],
    "user_id": "test-user-001",
    "agent_id": None,
    "run_id": None,
    "metadata": None,
    "infer": True,
    "memory_type": None,
}

_standard_search_example = {
    "query": "What does John like?",
    "user_id": "test-user-001",
    "run_id": None,
    "agent_id": None,
    "filters": None,
    "limit": 5,
    "threshold": None,
    "rerank": True,
}

_standard_search_filter_example = {
    "query": "What do you know about John?",
    "user_id": "test-user-001",
    "filters": {"actor_id": "John"},
    "limit": 10,
    "rerank": True,
}


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="List of messages to store. infer=True → LLM extracts facts. infer=False → raw store.",
    )
    user_id: Optional[str] = Field(None, description="User identifier for session scoping.", examples=["test-user-001"])
    agent_id: Optional[str] = Field(None, description="Agent identifier for session scoping.")
    run_id: Optional[str] = Field(None, description="Run identifier for session scoping.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store with the memories.")
    infer: bool = Field(
        True,
        description="True (default) → LLM extracts facts + decides ADD/UPDATE/DELETE/NONE. False → direct raw store.",
    )
    memory_type: Optional[str] = Field(
        None,
        description="'procedural_memory' for agent procedural, 'process_memory' for task summaries. None for standard.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                _standard_add_example,
                _standard_add_dedup_example,
            ]
        }
    }


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        description="Search query. SearchEngine does vector search + optional graph traversal + reranking.",
    )
    user_id: Optional[str] = Field(None, description="User identifier to scope the search.", examples=["test-user-001"])
    run_id: Optional[str] = Field(None, description="Run identifier to scope the search.")
    agent_id: Optional[str] = Field(None, description="Agent identifier to scope the search.")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Advanced metadata filters: eq/ne/gt/gte/lt/lte/in/nin/contains/icontains, AND/OR/NOT.",
    )
    limit: int = Field(100, description="Maximum number of results to return.")
    threshold: Optional[float] = Field(None, description="Minimum similarity score (0.0-1.0).", ge=0.0, le=1.0)
    rerank: bool = Field(True, description="Enable reranker to re-score vector results.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                _standard_search_example,
                _standard_search_filter_example,
            ]
        }
    }


class MemoryUpdate(BaseModel):
    data: str = Field(
        ...,
        description="New memory content to replace the existing memory.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"data": "Likes to play tennis on weekends with friends"}
        }
    }


# ── Process memory models ────────────────────────────────────────────────────

_process_add_example = {
    "messages": [
        {
            "Goal": "Add user authentication",
            "Step": "01 - Read main.py",
            "Action": "read_file(path='main.py')",
            "Dependency": [],
            "Brief": "Read main.py to understand the entry point logic",
        },
        {
            "Goal": "Add user authentication",
            "Step": "03 - Create auth.py",
            "Action": "create_file(path='auth.py')",
            "Dependency": [{"step_id": "01 - Read main.py", "description": "Parse entry logic"}],
            "Brief": "Create auth.py and implement login/logout functions",
        },
        {
            "Goal": "Setup database connection",
            "Step": "02 - Create config.py",
            "Action": "create_file(path='config.py')",
            "Dependency": [],
            "Brief": "Create config.py with database connection settings",
        },
    ],
    "user_id": "test-user-001",
    "agent_id": None,
    "run_id": None,
    "metadata": None,
}

_process_add_dedup_example = {
    "messages": [
        {
            "Goal": "Add user authentication",
            "Step": "01 - Read main.py",
            "Action": "read_file(path='main.py')",
            "Dependency": [],
            "Brief": "Read main.py to understand the entry point logic",
        },
        {
            "Goal": "Setup database connection",
            "Step": "02 - Create config.py",
            "Action": "create_file(path='config.py')",
            "Dependency": [],
            "Brief": "Create config.py with database connection settings",
        },
    ],
    "user_id": "test-user-001",
}

_process_search_basic_example = {
    "current_step": {
        "Goal": "Add user authentication",
        "Step": "03 - Create auth.py",
        "Action": "create_file(path='auth.py')",
        "Brief": "Create auth.py and implement login/logout functions",
    },
    "previous_step": None,
    "user_id": "test-user-001",
    "agent_id": None,
    "run_id": None,
    "task_estimate": "Implement a complete user authentication system",
    "graph_hop": 1,
    "chunk_top_k": 5,
    "summary_top_k": 3,
    "semantic_threshold": 0.6,
}

_process_search_filter_example = {
    "current_step": {
        "Goal": "Add user authentication",
        "Step": "03 - Create auth.py",
        "Action": "create_file(path='auth.py')",
        "Brief": "Create auth.py and implement login/logout functions",
    },
    "previous_step": {
        "Goal": "Add user authentication",
        "Step": "01 - Read main.py",
        "Action": "read_file(path='main.py')",
        "Brief": "Read main.py to understand the entry point logic",
    },
    "user_id": "test-user-001",
    "task_estimate": None,
    "graph_hop": 1,
    "chunk_top_k": 5,
    "summary_top_k": 3,
    "semantic_threshold": 0.6,
}


class ProcessStepSummary(BaseModel):
    """A single step summary produced by a Code Agent during task execution."""

    Goal: str = Field(..., description="Sub-goal; steps with the same Goal are grouped into one Chunk.")
    Step: str = Field(..., description="Unique step name. Becomes the Step node name in the graph.")
    Action: str = Field(..., description="Tool call / action string for reproducibility.")
    Dependency: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Upstream step references. Each forms a DEPENDS_ON edge.",
    )
    Brief: str = Field(..., description="Short semantic description used for embedding-based search.")


class ProcessMemoryCreate(BaseModel):
    """Flow 1: Write completed task summaries into three-layer process memory."""

    messages: List[ProcessStepSummary] = Field(
        ...,
        description="Array of step summaries from a completed task, in execution order.",
    )
    user_id: Optional[str] = Field(None, description="User identifier for session scoping.", examples=["test-user-001"])
    agent_id: Optional[str] = Field(None, description="Agent identifier for session scoping.")
    run_id: Optional[str] = Field(None, description="Run identifier for session scoping.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata to store with the memories.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                _process_add_example,
                _process_add_dedup_example,
            ]
        }
    }


class ProcessSearchStep(BaseModel):
    """A step summary used as search input during task execution."""

    Goal: str = Field(..., description="The sub-goal of the current step.")
    Step: str = Field(..., description="Step name.")
    Action: str = Field(default="", description="Tool call string.")
    Brief: str = Field(..., description="Short semantic description. Used for embedding-based graph node matching.")


class ProcessSearchRequest(BaseModel):
    """Flow 2: Search process memory during task execution for decision support."""

    current_step: ProcessSearchStep = Field(
        ...,
        description="The step being executed now. Brief → graph match; Goal → chunk recall.",
    )
    previous_step: Optional[ProcessSearchStep] = Field(
        None,
        description="Previously completed step. When set, its Brief is used for cosine-similarity semantic filtering.",
    )
    user_id: Optional[str] = Field(None, description="User identifier to scope the search.", examples=["test-user-001"])
    agent_id: Optional[str] = Field(None, description="Agent identifier to scope the search.")
    run_id: Optional[str] = Field(None, description="Run identifier to scope the search.")
    task_estimate: Optional[str] = Field(
        None,
        description="Optional task type estimate for summary search. Falls back to Goal+Brief concatenation if None.",
    )
    graph_hop: int = Field(1, description="Graph neighbor expansion hops from matched nodes.", ge=0, le=5)
    chunk_top_k: int = Field(5, description="Number of chunk results to retrieve.", ge=1, le=20)
    summary_top_k: int = Field(3, description="Number of summary results to retrieve.", ge=1, le=20)
    semantic_threshold: float = Field(
        0.6,
        description="Minimum cosine similarity for previous-step semantic filtering.",
        ge=0.0,
        le=1.0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                _process_search_basic_example,
                _process_search_filter_example,
            ]
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# Standard Memory Endpoints
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any], _api_key: Optional[str] = Depends(verify_api_key)):
    """Set memory configuration. Supports hot-reloading the entire Memory instance.

    Example body (minimal):
    ```json
    {
        "vector_store": {"provider": "pgvector", "config": {...}},
        "llm": {"provider": "openai", "config": {"api_key": "sk-...", "model": "gpt-4.1-nano-2025-04-14"}},
        "embedder": {"provider": "openai", "config": {"api_key": "sk-...", "model": "text-embedding-3-small"}}
    }
    ```

    To enable process memory, include the `process_memory` key (see ProcessMemoryConfig schema).
    """
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories (add-with-search-back)")
def add_memory(memory_create: MemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store new memories. The internal AddEngine first recalls existing memories via
    SearchEngine ("search"), then an LLM decides ADD/UPDATE/DELETE/NONE per fact ("back").

    **Returns**:
    - `results`: list of actual write operations (ADD/UPDATE/DELETE).
    - `recalled_memories`: memories recalled by SearchEngine before deciding.
    - `relations`: graph store results (when graph is enabled).

    **Test flow**:
    1. `POST /memories` with a conversation → facts are extracted and stored.
    2. `POST /memories` again with the same conversation → recalled_memories is non-empty, decisions are mostly NONE (dedup works).
    3. `POST /memories` with updated info → previous memories are UPDATE'd or DELETE'd.
    """
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params: dict = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    messages = [m.model_dump() for m in memory_create.messages]
    try:
        response = MEMORY_INSTANCE.add(messages=messages, **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="List all memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Retrieve all stored memories scoped to at least one identifier.

    Returns `{"results": [...], "relations": [...]}` when graph is enabled.
    """
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory by ID")
def get_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve a single memory by its UUID."""
    try:
        result = MEMORY_INSTANCE.get(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories (unified recall)")
def search_memories(search_req: SearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Unified search: vector similarity + optional graph traversal + optional reranking.

    **Returns**:
    - `results`: list of MemoryItem dicts with id, memory, score, metadata.
    - `relations`: graph traversal results (source, relationship, destination). Only present when graph is enabled.

    **Test flow**:
    1. `POST /memories` to add some facts first.
    2. `POST /search` with a query → returns matching results with scores.
    3. Try with `rerank=true` vs `rerank=false` to compare ordering.
    """
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: MemoryUpdate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Update an existing memory's content by ID. Re-embeds the new text and preserves session identifiers."""
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory.data)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Retrieve the ADD/UPDATE/DELETE history for a given memory."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Delete a single memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories for a scope")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Delete all memories scoped to the given identifiers. At least one required."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None}
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory(_api_key: Optional[str] = Depends(verify_api_key)):
    """Completely reset the vector store collection and history database. Irreversible."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Process Memory Endpoints (NEW)
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/process-memories", summary="[Process] Flow 1 — Write task summaries")
def add_process_memory(body: ProcessMemoryCreate, _api_key: Optional[str] = Depends(verify_api_key)):
    """Store completed task summaries into three-layer process memory.

    **Internal pipeline** (LangGraph):
    1. `preprocess` — parse summaries into goals, steps, dependencies, entity_type_map
    2. `search` — three-layer dedup search via ProcessMemorySearchEngine
    3. `decide` — one LLM call decides ADD/UPDATE/MERGE/NONE for all three layers
    4. `execute` — writes Graph (Step nodes + DEPENDS_ON edges), Chunks (Goal vectors), Summary (full chain vector)
    5. `assemble` — returns results + recalled

    **Returns**:
    ```json
    {
      "results": {
        "graph": {"deleted_entities": [], "added_entities": [...]},
        "chunks": [{"id": "uuid", "goal": "Add user authentication", "event": "ADD"}, ...],
        "summary": {"id": "uuid", "event": "ADD", "task_description": "..."}
      },
      "recalled": {
        "graph": {"chains": [...]},
        "chunks": [...],
        "summaries": [...]
      }
    }
    ```

    **Test flow**:
    1. POST with a set of summaries → chunks and summary are ADD'd, graph edges created.
    2. POST the same summaries again → MERGE/UPDATE events, no duplicates.
    3. POST /process-memories/search to verify recall works.
    """
    if not any([body.user_id, body.agent_id, body.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    summaries = [m.model_dump() for m in body.messages]
    params: dict = {k: v for k, v in body.model_dump().items() if v is not None and k != "messages"}

    try:
        response = MEMORY_INSTANCE.add(
            messages=summaries,
            memory_type="process_memory",
            **params,
        )
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_process_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-memories/search", summary="[Process] Flow 2 — Search during task execution")
def search_process_memory(body: ProcessSearchRequest, _api_key: Optional[str] = Depends(verify_api_key)):
    """Search process memory to recall relevant past experience during task execution.

    **Read-only** — no writes. Three-layer parallel search:

    1. **Graph**: Brief embedding → semantic Step node match → 1-hop neighbor expansion → optional previous-step semantic filtering.
    2. **Chunk**: Goal vector search → matching goal+steps groups.
    3. **Summary**: Task estimate vector search → full task execution chains.

    **Returns**:
    ```json
    {
      "graph": {
        "matched_nodes": [{"name": "03 - Create auth.py", "score": 0.92, "brief": "...", "goal": "..."}],
        "expanded_nodes": [{"source": "01", "relationship": "DEPENDS_ON", "destination": "03"}],
        "filtered_nodes": [...]
      },
      "chunks": [{"goal": "Add user authentication", "score": 0.85, "steps": [...], "id": "uuid"}],
      "summaries": [{"task_description": "...", "score": 0.78, "full_chain": [...], "id": "uuid"}]
    }
    ```

    **Test flow**:
    1. First POST /process-memories to write some summaries.
    2. POST /process-memories/search with a current_step → see matched graph nodes, chunks, and summaries.
    3. Add a previous_step → filtered_nodes is a subset of matched_nodes (semantic filter applied).
    4. Try with graph_hop=0 → no neighbor expansion.
    """
    if not any([body.user_id, body.agent_id, body.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    try:
        current_step = body.current_step.model_dump()
        previous_step = body.previous_step.model_dump() if body.previous_step else None

        return MEMORY_INSTANCE.search_process(
            current_step=current_step,
            previous_step=previous_step,
            user_id=body.user_id,
            agent_id=body.agent_id,
            run_id=body.run_id,
            task_estimate=body.task_estimate,
            graph_hop=body.graph_hop,
            chunk_top_k=body.chunk_top_k,
            summary_top_k=body.summary_top_k,
            semantic_threshold=body.semantic_threshold,
        )
    except Exception as e:
        logging.exception("Error in search_process_memory:")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
