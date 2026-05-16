# Mem0 REST API Server

Mem0 provides a REST API server (written using FastAPI). Users can perform all operations through REST endpoints. The API also includes OpenAPI documentation, accessible at `/docs` when the server is running.

## Features

- **Create memories:** Create memories based on messages for a user, agent, or run.
- **Retrieve memories:** Get all memories for a given user, agent, or run.
- **Search memories:** Search stored memories based on a query.
- **Update memories:** Update an existing memory.
- **Delete memories:** Delete a specific memory or all memories for a user, agent, or run.
- **Reset memories:** Reset all memories for a user, agent, or run.
- **OpenAPI Documentation:** Accessible via `/docs` endpoint.

## Running the server (Conda + Docker Infra)

This repository now uses a hybrid runtime:

- `mem0` REST API runs locally in a Conda environment.
- `postgres` (pgvector) and `neo4j` continue to run in Docker.

### 1) Create and activate Conda environment

Run these commands from the repository root:

```bash
conda env create -f server/conda-environment.yml
conda activate mem0
pip install -e .[graph]
```

The environment installs dependencies from `server/requirements.txt`. The extra `pip install -e .[graph]` installs local `mem0` in editable mode with graph extras, matching the previous dev Docker image behavior.

### 2) Start infra containers (postgres + neo4j)

```bash
cd server
docker compose up -d postgres neo4j
```

### 3) Verify `.env` points to Docker-exposed host ports

`server/.env` should contain host-style endpoints for local API runtime:

- `POSTGRES_HOST=localhost`
- `POSTGRES_PORT=8432`
- `NEO4J_URI=bolt://localhost:8687`
- `HISTORY_DB_PATH=~/.mem0/history.db`

### 4) Run mem0 REST API locally

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

API docs: `http://localhost:8888/docs`

### Optional Makefile shortcuts

From `server/`:

```bash
make infra_up
make run_local
```
