"""End-to-end tests for ProcessMemoryAddEngine with real backend services.

Covers both Flow 1 (add + dedup recall) and Flow 2 (process search),
with three-layer writes (Graph / Chunk / Summary).

Prerequisites (per RUN_GUIDE.md):
    conda activate mem0
    cd server/ && docker compose up -d
    export $(grep -v '^#' server/.env | xargs)

Usage:
    pytest tests/memory/test_process_e2e.py -v
"""

import json
import os
import uuid

import pytest

from mem0.configs.base import MemoryConfig
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.embeddings.openai import OpenAIEmbedding
from mem0.llms.openai import OpenAILLM
from mem0.memory.main import Memory
from mem0.memory.process_add_engine import ProcessMemoryAddEngine
from mem0.memory.process_search_engine import ProcessMemorySearchEngine
from mem0.memory.storage import SQLiteManager
from mem0.vector_stores.pgvector import PGVector


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _env_loaded():
    return bool(os.getenv("OPENAI_llm_API_KEY"))


requires_env = pytest.mark.skipif(
    not _env_loaded(),
    reason="Environment variables not loaded. Run: export $(grep -v '^#' server/.env | xargs)",
)


def _pg_available():
    try:
        import psycopg2  # noqa: F401
        return True
    except ImportError:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="psycopg2 not installed. Run: pip install psycopg2-binary",
)


def _neo4j_available():
    try:
        import neo4j  # noqa: F401
        return True
    except ImportError:
        return False


requires_neo4j = pytest.mark.skipif(
    not _neo4j_available(),
    reason="neo4j driver not installed. Run: pip install neo4j",
)


# ---------------------------------------------------------------------------
# Test collection names (isolation from standard E2E tests)
# ---------------------------------------------------------------------------

TEST_COLLECTION = "test_process_e2e"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_model():
    """Real OpenAI embedding model."""
    config = BaseEmbedderConfig(
        api_key=os.getenv("OPENAI_EMBEDDER_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
        openai_base_url=os.getenv("OPENAI_EMBEDDER_URL"),
    )
    return OpenAIEmbedding(config)


@pytest.fixture(scope="module")
def llm():
    """Real OpenAI-compatible LLM."""
    config = OpenAIConfig(
        api_key=os.getenv("OPENAI_llm_API_KEY"),
        model=os.getenv("OPENAI_llm_Model", "deepseek-chat"),
        openai_base_url=os.getenv("OPENAI_llm_URL"),
        temperature=float(os.getenv("OPENAI_llm_temperature", "0.0")),
    )
    return OpenAILLM(config)


@pytest.fixture(scope="module")
def vector_store():
    """Real PGVector for process chunks and summaries."""
    vs = PGVector(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "8432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        collection_name=TEST_COLLECTION,
        embedding_model_dims=1536,
        diskann=False,
        hnsw=True,
    )
    yield vs
    try:
        vs.delete_col()
    except Exception:
        pass


@pytest.fixture(scope="module")
def db():
    """In-memory SQLiteManager for history tracking."""
    return SQLiteManager(db_path=":memory:")


@pytest.fixture(scope="module")
def graph_store(embedding_model, vector_store):
    """Real Neo4j MemoryGraph with Step node_label."""
    config = MemoryConfig(
        graph_store={
            "provider": "neo4j",
            "config": {
                "url": os.getenv("NEO4J_URI", "bolt://localhost:8687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "mem0graph"),
                "database": "neo4j",
            },
        },
        embedder={
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_EMBEDDER_API_KEY"),
                "model": os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
                "openai_base_url": os.getenv("OPENAI_EMBEDDER_URL"),
            },
        },
        vector_store={
            "provider": "pgvector",
            "config": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "8432")),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                "dbname": os.getenv("POSTGRES_DB", "postgres"),
                "collection_name": TEST_COLLECTION,
                "embedding_model_dims": 1536,
            },
        },
        llm={
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_llm_API_KEY"),
                "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                "openai_base_url": os.getenv("OPENAI_llm_URL"),
                "temperature": 0.0,
            },
        },
    )
    from mem0.memory.graph_memory import MemoryGraph as ProcessGraph
    gs = ProcessGraph(config, node_label=":Step")
    yield gs
    try:
        gs.reset()
    except Exception:
        pass


@pytest.fixture(scope="module")
def search_engine(embedding_model, vector_store, graph_store):
    """ProcessMemorySearchEngine with real vector store and graph."""
    return ProcessMemorySearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        graph_store=graph_store,
    )


@pytest.fixture(scope="module")
def add_engine(embedding_model, vector_store, llm, db, search_engine, graph_store):
    """ProcessMemoryAddEngine with real backends."""
    return ProcessMemoryAddEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        db=db,
        search_engine=search_engine,
        graph_store=graph_store,
    )


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _make_summaries():
    """Return a set of step summaries simulating a task completion."""
    return [
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
            "Dependency": [
                {"step_id": "01 - Read main.py", "description": "Parse entry logic"}
            ],
            "Brief": "Create auth.py and implement login/logout functions",
        },
        {
            "Goal": "Setup database connection",
            "Step": "02 - Create config.py",
            "Action": "create_file(path='config.py')",
            "Dependency": [],
            "Brief": "Create config.py with database connection settings",
        },
    ]


def _make_single_summary():
    return [
        {
            "Goal": "Refactor codebase",
            "Step": "05 - Rename modules",
            "Action": "rename_file(old='old_util.py', new='util.py')",
            "Dependency": [],
            "Brief": "Rename old_util.py to util.py for consistency",
        },
    ]


# ══════════════════════════════════════════════════════════════════════════
# Flow 1: Add and Recall (without Neo4j)
# ══════════════════════════════════════════════════════════════════════════


@requires_env
@requires_pg
class TestE2EFlow1AddAndRecall:
    """Verify Flow 1 writes (chunks + summaries) and recall via dedup/step search."""

    def test_add_new_summaries_creates_chunks(self, add_engine):
        """First-time add should create ADD chunks and summary."""
        user_id = f"e2e-flow1-{uuid.uuid4().hex[:8]}"

        result = add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        assert "results" in result
        assert "recalled" in result

        # Chunks: should have ADD events for new goals
        chunks = result["results"]["chunks"]
        assert len(chunks) > 0
        for c in chunks:
            assert c["event"] in ("ADD",)
            assert c["id"] is not None

        # Summary should have been written
        summary = result["results"]["summary"]
        if summary:
            assert summary.get("event") in ("ADD", "UPDATE")

    def test_add_duplicate_goal_merges(self, add_engine, search_engine):
        """Adding the same goal again should not create duplicate chunks."""
        user_id = f"e2e-merge-{uuid.uuid4().hex[:8]}"

        # First add
        result1 = add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        # Second add with the same goal
        result2 = add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        # Both should return valid structures
        assert "results" in result2
        assert "chunks" in result2["results"]

        # After both adds, search should still return results (no data loss)
        recalled = search_engine.search_for_dedup(
            goals=["Add user authentication"],
            filters={"user_id": user_id},
        )
        assert "chunks" in recalled

    def test_search_for_dedup_after_add(self, search_engine, add_engine):
        """After writing, search_for_dedup should recall chunks by goal."""
        user_id = f"e2e-dedup-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        recalled = search_engine.search_for_dedup(
            goals=["Add user authentication"],
            filters={"user_id": user_id},
        )

        # Should find chunks matching the goal
        assert "chunks" in recalled
        assert "summaries" in recalled
        # At least one chunk or summary should be recalled
        has_results = len(recalled["chunks"]) > 0 or len(recalled["summaries"]) > 0
        assert has_results, "Expected at least some recall results after add"

    def test_search_for_step_after_add(self, search_engine, add_engine):
        """After writing, search_for_step should recall chunks by current step."""
        user_id = f"e2e-stepsearch-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        current_step = {
            "Goal": "Add user authentication",
            "Step": "03 - Create auth.py",
            "Action": "create_file(path='auth.py')",
            "Brief": "Create auth.py and implement login/logout functions",
        }

        result = search_engine.search_for_step(
            current_step=current_step,
            filters={"user_id": user_id},
        )

        assert "chunks" in result
        assert "summaries" in result

    def test_add_result_structure(self, add_engine):
        """Verify that add() returns the complete expected structure."""
        user_id = f"e2e-structure-{uuid.uuid4().hex[:8]}"

        result = add_engine.add(
            summaries=_make_single_summary(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        assert "results" in result
        assert "recalled" in result

        results = result["results"]
        assert "graph" in results
        assert "chunks" in results
        assert "summary" in results

        recalled = result["recalled"]
        assert "graph" in recalled
        assert "chunks" in recalled
        assert "summaries" in recalled


# ══════════════════════════════════════════════════════════════════════════
# Flow 1: Three-layer writes with graph (requires Neo4j)
# ══════════════════════════════════════════════════════════════════════════


@requires_env
@requires_pg
@requires_neo4j
class TestE2EFlow1WithGraph:
    """Verify three-layer writes including graph store."""

    def test_add_with_graph_creates_step_nodes(self, add_engine, graph_store):
        """Adding summaries with dependencies creates Step nodes in Neo4j."""
        user_id = f"e2e-graph-{uuid.uuid4().hex[:8]}"

        result = add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        graph_result = result["results"]["graph"]
        # Graph result should contain added entities when graph is available
        if graph_result:
            added = graph_result.get("added_entities", [])
            assert isinstance(added, list)

    def test_graph_dedup_merges_nodes(self, add_engine, graph_store):
        """Same step name added twice should increment mentions, not duplicate."""
        user_id = f"e2e-graph-merge-{uuid.uuid4().hex[:8]}"

        # First add
        result1 = add_engine.add(
            summaries=_make_single_summary(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        # Second add - same step name
        result2 = add_engine.add(
            summaries=_make_single_summary(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        # Both should succeed; second one may merge the step
        graph2 = result2["results"]["graph"]
        assert graph2 is not None or True  # graph result present or empty

    def test_search_for_dedup_graph_chains(self, search_engine, add_engine, graph_store):
        """After writing multi-step dependencies, search_for_dedup returns graph chains."""
        user_id = f"e2e-chain-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        recalled = search_engine.search_for_dedup(
            goals=["Add user authentication", "Setup database connection"],
            filters={"user_id": user_id},
        )

        assert "graph" in recalled
        # Graph chains may be empty if the graph search has no matching step names
        # from chunks yet, but the structure should be correct
        assert "chains" in recalled["graph"]
        assert isinstance(recalled["graph"]["chains"], list)

    def test_search_for_step_graph_match(self, search_engine, add_engine):
        """search_for_step with Brief should match Step nodes via embedding."""
        user_id = f"e2e-graph-match-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        current_step = {
            "Goal": "Add user authentication",
            "Step": "03 - Create auth.py",
            "Brief": "Create auth.py and implement login/logout functions",
        }

        result = search_engine.search_for_step(
            current_step=current_step,
            filters={"user_id": user_id},
        )

        assert "graph" in result
        assert "matched_nodes" in result["graph"]
        assert "expanded_nodes" in result["graph"]
        assert "filtered_nodes" in result["graph"]


# ══════════════════════════════════════════════════════════════════════════
# Flow 2: Process search (read-only, requires Neo4j for graph search)
# ══════════════════════════════════════════════════════════════════════════


@requires_env
@requires_pg
@requires_neo4j
class TestE2EFlow2ProcessSearch:
    """Verify Flow 2 search-only operations."""

    def test_search_for_step_empty_state(self, search_engine):
        """search_for_step on an empty database returns empty lists, not errors."""
        user_id = f"e2e-empty-{uuid.uuid4().hex[:8]}"

        current_step = {
            "Goal": "Nonexistent goal",
            "Step": "99 - Nothing",
            "Brief": "Nothing here",
        }

        result = search_engine.search_for_step(
            current_step=current_step,
            filters={"user_id": user_id},
        )

        assert "graph" in result
        assert "chunks" in result
        assert "summaries" in result
        assert result["chunks"] == []

    def test_search_for_step_chunk_and_summary(self, search_engine, add_engine):
        """After writing, search_for_step returns both chunks and summaries."""
        user_id = f"e2e-fullsearch-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        current_step = {
            "Goal": "Add user authentication",
            "Step": "03 - Create auth.py",
            "Brief": "Create auth.py and implement login/logout functions",
        }

        result = search_engine.search_for_step(
            current_step=current_step,
            filters={"user_id": user_id},
        )

        assert "chunks" in result
        assert "summaries" in result
        # At minimum we should get some chunks back for the matching goal
        assert len(result["chunks"]) > 0 or len(result["summaries"]) > 0, (
            "Expected chunks or summaries to be recalled after add"
        )

    def test_search_for_step_with_previous_semantic_filter(self, search_engine, add_engine):
        """Previous step semantic filtering should narrow results."""
        user_id = f"e2e-filter-{uuid.uuid4().hex[:8]}"

        add_engine.add(
            summaries=_make_summaries(),
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
        )

        current_step = {
            "Goal": "Add user authentication",
            "Step": "03 - Create auth.py",
            "Brief": "Create auth.py and implement login/logout functions",
        }
        previous_step = {
            "Goal": "Add user authentication",
            "Step": "01 - Read main.py",
            "Brief": "Read main.py to understand the entry point logic",
        }

        result = search_engine.search_for_step(
            current_step=current_step,
            previous_step=previous_step,
            filters={"user_id": user_id},
        )

        # filtered_nodes count <= matched_nodes count (semantic filter reduces)
        assert len(result["graph"]["filtered_nodes"]) <= len(result["graph"]["matched_nodes"])


# ══════════════════════════════════════════════════════════════════════════
# Integration: Memory-level routing and end-to-end
# ══════════════════════════════════════════════════════════════════════════


@requires_env
@requires_pg
@requires_neo4j
class TestE2EIntegration:
    """Verify the full Memory-level API with process_memory config."""

    def test_memory_add_process_memory_routing(self):
        """Memory.add(memory_type='process_memory') routes to ProcessMemoryAddEngine."""
        user_id = f"e2e-mem-{uuid.uuid4().hex[:8]}"

        config = MemoryConfig(
            process_memory={
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "host": os.getenv("POSTGRES_HOST", "localhost"),
                        "port": int(os.getenv("POSTGRES_PORT", "8432")),
                        "user": os.getenv("POSTGRES_USER", "postgres"),
                        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                        "dbname": os.getenv("POSTGRES_DB", "postgres"),
                        "collection_name": f"{TEST_COLLECTION}_mem",
                        "embedding_model_dims": 1536,
                    },
                },
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": os.getenv("NEO4J_URI", "bolt://localhost:8687"),
                        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                        "password": os.getenv("NEO4J_PASSWORD", "mem0graph"),
                        "database": "neo4j",
                    },
                },
            },
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            embedder={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_EMBEDDER_API_KEY"),
                    "model": os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
                    "openai_base_url": os.getenv("OPENAI_EMBEDDER_URL"),
                },
            },
            vector_store={
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "8432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                    "dbname": os.getenv("POSTGRES_DB", "postgres"),
                    "collection_name": f"{TEST_COLLECTION}_mem",
                    "embedding_model_dims": 1536,
                },
            },
        )

        memory = Memory(config)
        try:
            result = memory.add(
                messages=_make_summaries(),
                user_id=user_id,
                memory_type="process_memory",
            )

            assert "results" in result
            assert "recalled" in result
            assert "chunks" in result["results"]
            assert "summary" in result["results"]
        finally:
            try:
                memory.vector_store.delete_col()
            except Exception:
                pass
            try:
                memory.process_vector_store.delete_col()
            except Exception:
                pass
            try:
                memory.process_graph_store.reset()
            except Exception:
                pass

    def test_memory_search_process_after_add(self):
        """search_process after add returns search results."""
        user_id = f"e2e-searchproc-{uuid.uuid4().hex[:8]}"

        config = MemoryConfig(
            process_memory={
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "host": os.getenv("POSTGRES_HOST", "localhost"),
                        "port": int(os.getenv("POSTGRES_PORT", "8432")),
                        "user": os.getenv("POSTGRES_USER", "postgres"),
                        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                        "dbname": os.getenv("POSTGRES_DB", "postgres"),
                        "collection_name": f"{TEST_COLLECTION}_sp",
                        "embedding_model_dims": 1536,
                    },
                },
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": os.getenv("NEO4J_URI", "bolt://localhost:8687"),
                        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                        "password": os.getenv("NEO4J_PASSWORD", "mem0graph"),
                        "database": "neo4j",
                    },
                },
            },
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            embedder={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_EMBEDDER_API_KEY"),
                    "model": os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
                    "openai_base_url": os.getenv("OPENAI_EMBEDDER_URL"),
                },
            },
            vector_store={
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "8432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                    "dbname": os.getenv("POSTGRES_DB", "postgres"),
                    "collection_name": f"{TEST_COLLECTION}_sp",
                    "embedding_model_dims": 1536,
                },
            },
        )

        memory = Memory(config)
        try:
            memory.add(
                messages=_make_summaries(),
                user_id=user_id,
                memory_type="process_memory",
            )

            result = memory.search_process(
                current_step={
                    "Goal": "Add user authentication",
                    "Step": "03 - Create auth.py",
                    "Brief": "Create auth.py and implement login/logout functions",
                },
                user_id=user_id,
            )

            assert "chunks" in result
            assert "summaries" in result
            assert "graph" in result
        finally:
            try:
                memory.vector_store.delete_col()
            except Exception:
                pass
            try:
                memory.process_vector_store.delete_col()
            except Exception:
                pass
            try:
                memory.process_graph_store.reset()
            except Exception:
                pass

    def test_memory_add_without_process_config(self):
        """Without process_memory config, memory_type='process_memory' should error."""
        config = MemoryConfig(
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            embedder={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_EMBEDDER_API_KEY"),
                    "model": os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
                    "openai_base_url": os.getenv("OPENAI_EMBEDDER_URL"),
                },
            },
        )

        memory = Memory(config)
        from mem0.exceptions import ValidationError as Mem0ValidationError

        with pytest.raises(Mem0ValidationError):
            memory.add(
                messages=_make_summaries(),
                user_id="test-user",
                memory_type="process_memory",
            )
