# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **mem0** — a long-term memory layer for AI agents and assistants. It is a multi-package repository containing:

- **`mem0/`** — Python SDK (`mem0ai` on PyPI)
- **`mem0-ts/`** — TypeScript SDK (`mem0ai` on npm)
- **`server/`** — FastAPI-based self-hosted REST API server
- **`embedchain/`**, **`openmemory/`** — Legacy/experimental packages (excluded from linting)

## Development Commands

### Python SDK

The Python package uses **hatch** for environment management and **ruff** for linting/formatting.

```bash
# Create / enter a dev environment for a specific Python version
hatch shell dev_py_3_11

# Run all tests (uses pytest)
make test
hatch run test

# Run tests for a specific Python version
make test-py-3.9    # or 3.10, 3.11, 3.12

# Run a single test file or test case
pytest tests/test_memory.py
pytest tests/test_memory.py -k test_create_memory

# Format and lint
make format         # ruff format
make lint           # ruff check
make sort           # isort --profile black mem0/
make all            # format + sort + lint

# Build / publish
make build          # hatch build
make publish        # hatch publish

# Pre-commit hooks
pre-commit install
```

### TypeScript SDK (`mem0-ts/`)

```bash
cd mem0-ts
pnpm install

# Run tests
pnpm test
pnpm run test:unit
pnpm run test:integration

# Build / format
pnpm run build
pnpm run format
```

### REST API Server (`server/`)

```bash
cd server
# FastAPI server, typically run with:
uvicorn main:app --reload
# Or via Docker Compose:
docker-compose up
```

## High-Level Architecture

### Core Memory Classes

The primary public API lives in **`mem0/memory/main.py`**:

- **`Memory`** — Synchronous memory interface.
- **`AsyncMemory`** — Asynchronous memory interface.

Both expose `add()`, `search()`, `get()`, `get_all()`, `update()`, `delete()`, `history()`, and `reset()`.

`Memory` is also importable directly from the package root (`from mem0 import Memory`).

### LangGraph-Based `add()` Flow

The `Memory.add()` method was refactored to use **LangGraph**. It defines a state machine (`AddMemoryState`) with nodes for:

1. `preprocess` — validates metadata/filters, detects procedural memory type
2. `procedural_handler` — handles procedural memory directly
3. `direct_add` — non-inferred/direct vector insertion path
4. `extract_facts` — LLM-based fact extraction from messages
5. `retrieve_memories` — fetches existing memories from the vector store
6. `decide_actions` — determines whether to ADD, UPDATE, DELETE, or NONE for each fact
7. `dispatch_actions` / `execute_*` — applies the actions to the vector store
8. `graph_store` — optionally writes to the graph memory backend
9. `telemetry_and_wrapup` — captures telemetry and returns results

When modifying `add()` or its node methods, be aware of the concurrent branches and the shared `AddMemoryState` schema.

### Pluggable Backends (Factory Pattern)

Component creation is centralized in **`mem0/utils/factory.py`**:

- **`LlmFactory`** — creates LLM clients (openai, anthropic, groq, azure_openai, gemini, ollama, etc.)
- **`EmbedderFactory`** — creates embedding clients
- **`VectorStoreFactory`** — creates vector DB clients (qdrant, pgvector, chroma, pinecone, redis, milvus, etc.)
- **`GraphStoreFactory`** — creates graph DB clients (neo4j, memgraph, kuzu, apache_age, neptune)
- **`RerankerFactory`** — creates reranker instances (cohere, sentence_transformer, etc.)

Each backend implements a base ABC (e.g., `VectorStoreBase` in `mem0/vector_stores/base.py`).

### Configuration

Configuration uses **Pydantic v2** models defined in **`mem0/configs/base.py`** (`MemoryConfig`).

Key config sections:

- `vector_store` — vector DB provider and connection settings
- `llm` — LLM provider and model settings
- `embedder` — embedding provider and model settings
- `graph_store` — optional graph DB provider (if omitted, graph memory is disabled)
- `reranker` — optional reranker configuration
- `history_db_path` — local SQLite path for memory history (default: `~/.mem0/history.db`)
- `version` — API version string (default: `"v1.1"`)

`Memory.from_config(config_dict)` is the standard entry point for constructing a `Memory` instance from a dictionary.

### Graph Memory

Graph memory is implemented in **`mem0/memory/graph_memory.py`**, with provider-specific subclasses in:

- `mem0/memory/memgraph_memory.py`
- `mem0/memory/kuzu_memory.py`
- `mem0/memory/apache_age_memory.py`
- `mem0/graphs/neptune/`

Graph memory is only initialized when `config.graph_store.config` is provided.

### History & Telemetry

- **`mem0/memory/storage.py`** — `SQLiteManager` handles history tracking (create, get, update operations on memory history).
- **`mem0/memory/telemetry.py`** — Anonymous usage telemetry. Telemetry data is stored in a *separate* vector store collection (`mem0migrations`) so it never pollutes user data.

### REST API Server (`server/`)

- **`server/main.py`** — FastAPI app that wraps a global `Memory` instance (`MEMORY_INSTANCE`).
- Endpoints map 1:1 to `Memory` methods (`POST /memories`, `GET /memories`, `POST /search`, `PUT /memories/{id}`, `DELETE /memories/{id}`, etc.).
- Authentication is optional via `ADMIN_API_KEY` env var (`X-API-Key` header).
- Default stack uses **pgvector** + **neo4j** if environment variables are set.

### Important Code Patterns

- **Sensitive field redaction**: `mem0/memory/main.py` contains `_is_sensitive_field()` and `_safe_deepcopy_config()` used to scrub secrets (api_key, password, token, etc.) from telemetry payloads without breaking non-serializable runtime objects like `http_auth`.
- **Filter/metadata builder**: `_build_filters_and_metadata()` enforces that at least one of `user_id`, `agent_id`, or `run_id` is provided and constructs query filters consistently.
- **Ruff exclusions**: `embedchain/` and `openmemory/` are excluded from linting (`pyproject.toml` -> `[tool.ruff]`).

## Development Environment Quick Reference

### API Authentication (Local Development)
- **Admin API Key**: `my_very_long_custom_key_123456`
- **Header**: `X-API-Key: my_very_long_custom_key_123456`
- **Location**: Defined in `server/.env` as `ADMIN_API_KEY`

### Test Commands
```bash
# Quick API test for Memory.add()
curl -X POST http://localhost:8888/memories \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my_very_long_custom_key_123456" \
  -d '{"messages":[{"role":"user","content":"test"}], "user_id":"u1"}'

# Watch Docker logs for print output
docker compose logs -f mem0 --tail 10
```

### Plan File Location
- **Path**: `C:/Users/26523/.claude/plans/shimmying-wiggling-melody.md`
- **Purpose**: Tracks current development plan and verification steps

## Lessons from Past Mistakes

### 1. Always Check Plan Files First
**Problem**: Started implementation without reading the plan file, leading to wrong approach.
**Fix**: Always check `.claude/plans/*.md` before starting work.

### 2. File Path Handling
**Problem**: Used incorrect absolute path format (`/wsl.localhost/...`) when relative paths work better.
**Fix**: Use relative paths from current working directory (`mem0/memory/main.py`).


### 4. Docker Environment Assumptions
**Problem**: Did not verify container status, API key, or port mappings before testing.
**Fix**: Always run `docker compose ps` first, check `.env` for credentials.

### 5. Print Output Verification
**Problem**: Difficult to find print output in scrolling Docker logs.
**Fix**: Use `docker compose logs -f mem0 --tail N` combined with test command to capture output.
