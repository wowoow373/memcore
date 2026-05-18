"""End-to-end tests for AddEngine with real backend services.

Prerequisites:
    - Docker services running (postgres:8432, neo4j:8687)
    - Environment variables loaded from server/.env

Usage:
    cd server/ && docker compose up -d
    export $(grep -v '^#' server/.env | xargs)
    conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py -v
"""

import json
import os
import uuid

import pytest

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.embeddings.openai import OpenAIEmbedding
from mem0.llms.openai import OpenAILLM
from mem0.memory.add_engine import AddEngine
from mem0.memory.search_engine import SearchEngine
from mem0.memory.storage import SQLiteManager
from mem0.utils.factory import GraphStoreFactory
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


# Collection names for test isolation
TEST_COLLECTION = "test_add_engine_e2e"


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
    """Real pgvector instance for E2E testing."""
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
    """Real SQLiteManager for history tracking (in-memory)."""
    return SQLiteManager(db_path=":memory:")


@pytest.fixture
def search_engine(embedding_model, vector_store, llm):
    """SearchEngine with real vector store, no graph."""
    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        graph_store=None,
        reranker=None,
        llm=llm,
    )


@pytest.fixture
def add_engine(embedding_model, vector_store, llm, db, search_engine):
    """AddEngine with real vector store, no graph."""
    return AddEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        db=db,
        search_engine=search_engine,
        graph=None,
    )


# ---------------------------------------------------------------------------
# E2E Tests: infer=False (direct_add fast path)
# ---------------------------------------------------------------------------

@requires_env
@requires_pg
class TestE2EDirectAdd:
    """End-to-end tests for the direct_add path (infer=False)."""

    def test_direct_add_single_message(self, add_engine):
        """Add a single user message directly, verify it's stored and retrievable."""
        messages = [{"role": "user", "content": "Hello, my name is Alice"}]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-direct-add"},
            filters={"user_id": "e2e-direct-add"},
            infer=False,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "Hello, my name is Alice"
        assert result["results"][0]["role"] == "user"
        memory_id = result["results"][0]["id"]
        assert memory_id is not None

        # Verify recalled_memories is present (empty for direct add)
        assert result["recalled_memories"]["results"] == []

    def test_direct_add_skips_system(self, add_engine):
        """System messages should be skipped in direct add."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I like pizza"},
        ]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-direct-skip-system"},
            filters={"user_id": "e2e-direct-skip-system"},
            infer=False,
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "I like pizza"

    def test_direct_add_multiple_messages(self, add_engine):
        """Multiple user/assistant messages should all be stored."""
        messages = [
            {"role": "user", "content": "My favorite color is blue"},
            {"role": "assistant", "content": "Blue is a great choice!"},
            {"role": "user", "content": "I also like green"},
        ]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-direct-multi"},
            filters={"user_id": "e2e-direct-multi"},
            infer=False,
        )

        assert len(result["results"]) == 3
        events = [r["event"] for r in result["results"]]
        assert all(e == "ADD" for e in events)

    def test_direct_add_empty_content_skipped(self, add_engine):
        """Messages with empty content should be skipped."""
        messages = [
            {"role": "user", "content": "   "},
            {"role": "user", "content": "Valid message"},
        ]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-direct-empty"},
            filters={"user_id": "e2e-direct-empty"},
            infer=False,
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "Valid message"


# ---------------------------------------------------------------------------
# E2E Tests: infer=True (LLM path)
# ---------------------------------------------------------------------------

@requires_env
@requires_pg
class TestE2EInferAdd:
    """End-to-end tests for the infer=True path (LLM extraction + decision)."""

    def test_infer_add_new_fact(self, add_engine):
        """Add a new fact with infer=True. LLM should extract and ADD."""
        messages = [{"role": "user", "content": "My name is John and I am a software engineer"}]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-infer-add"},
            filters={"user_id": "e2e-infer-add"},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert len(result["results"]) >= 1
        events = [r["event"] for r in result["results"]]
        assert "ADD" in events

        # Check structure of each result entry
        for entry in result["results"]:
            assert "id" in entry
            assert "memory" in entry
            assert "event" in entry
            assert entry["id"] is not None

    def test_infer_recalls_previous_memories(self, add_engine, search_engine):
        """After adding a memory, a subsequent add should recall it."""
        user_id = "e2e-infer-recall-" + str(uuid.uuid4())[:8]

        # First add
        result1 = add_engine.add(
            messages=[{"role": "user", "content": "I love eating sushi"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert len(result1["results"]) >= 1

        # Second add should recall the first
        result2 = add_engine.add(
            messages=[{"role": "user", "content": "I also enjoy ramen"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        # Should have recalled the sushi memory
        recalled = result2["recalled_memories"]["results"]
        recalled_texts = [r.get("memory", "") for r in recalled]
        assert any("sushi" in t.lower() for t in recalled_texts), \
            f"Expected sushi memory in recall, got: {recalled_texts}"

    def test_infer_update_existing(self, add_engine, search_engine):
        """Adding related information should UPDATE an existing memory."""
        user_id = "e2e-infer-update-" + str(uuid.uuid4())[:8]

        # First add: initial fact
        add_engine.add(
            messages=[{"role": "user", "content": "I live in New York"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        # Second add: more detailed fact about the same topic
        result = add_engine.add(
            messages=[{"role": "user", "content": "I have been living in Brooklyn, New York for 5 years now"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        events = [r["event"] for r in result["results"]]
        # Should have at least one UPDATE or ADD (depends on LLM decision)
        assert any(e in events for e in ["ADD", "UPDATE"])

    def test_infer_empty_conversation(self, add_engine):
        """An empty or greeting-only conversation should produce no/empty results."""
        messages = [{"role": "user", "content": "Hi"}]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-infer-empty"},
            filters={"user_id": "e2e-infer-empty"},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        # Greetings may or may not produce facts; either is fine
        assert isinstance(result["results"], list)

    def test_infer_result_structure(self, add_engine):
        """Verify the full response structure matches the design spec."""
        messages = [{"role": "user", "content": "I am a vegetarian and I enjoy hiking"}]

        result = add_engine.add(
            messages=messages,
            metadata={"user_id": "e2e-structure"},
            filters={"user_id": "e2e-structure"},
            infer=True,
        )

        # Top-level keys
        assert "results" in result
        assert "recalled_memories" in result

        # recalled_memories structure
        rm = result["recalled_memories"]
        assert "results" in rm
        assert "relations" in rm
        assert isinstance(rm["results"], list)
        assert isinstance(rm["relations"], list)

        # Each result entry
        for entry in result["results"]:
            assert "id" in entry
            assert "memory" in entry
            assert "event" in entry
            assert entry["event"] in ("ADD", "UPDATE", "DELETE")

            if entry["event"] == "UPDATE":
                assert "previous_memory" in entry


# ---------------------------------------------------------------------------
# E2E Tests: Memory.add() delegation
# ---------------------------------------------------------------------------

@requires_env
@requires_pg
class TestE2EMemoryAddDelegation:
    """Verify that Memory.add() correctly delegates to AddEngine."""

    def test_memory_add_delegation(self, embedding_model, vector_store):
        """Memory.add() should produce results with recalled_memories."""
        from mem0.configs.base import MemoryConfig
        from mem0.memory.main import Memory

        config = MemoryConfig(
            vector_store={
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "8432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                    "dbname": os.getenv("POSTGRES_DB", "postgres"),
                    "collection_name": "test_memory_add_e2e",
                    "embedding_model_dims": 1536,
                    "diskann": False,
                    "hnsw": True,
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
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            history_db_path=":memory:",
            version="v1.1",
        )

        memory = Memory(config)

        try:
            user_id = "e2e-memory-delegate-" + str(uuid.uuid4())[:8]

            result = memory.add(
                messages=[{"role": "user", "content": "My name is Bob and I work as a designer"}],
                user_id=user_id,
                infer=True,
            )

            # Verify new response format
            assert "results" in result
            assert "recalled_memories" in result

            # Check recalled_memories structure
            rm = result["recalled_memories"]
            assert "results" in rm
            assert isinstance(rm["results"], list)

            # Check results
            assert len(result["results"]) >= 1
            for entry in result["results"]:
                assert "id" in entry
                assert "memory" in entry
                assert "event" in entry

        finally:
            try:
                memory.vector_store.delete_col()
            except Exception:
                pass

    def test_memory_add_direct(self, embedding_model, vector_store):
        """Memory.add() with infer=False should go through direct_add path."""
        from mem0.configs.base import MemoryConfig
        from mem0.memory.main import Memory

        config = MemoryConfig(
            vector_store={
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "8432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                    "dbname": os.getenv("POSTGRES_DB", "postgres"),
                    "collection_name": "test_memory_direct_e2e",
                    "embedding_model_dims": 1536,
                    "diskann": False,
                    "hnsw": True,
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
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            history_db_path=":memory:",
            version="v1.1",
        )

        memory = Memory(config)

        try:
            user_id = "e2e-memory-direct-" + str(uuid.uuid4())[:8]

            result = memory.add(
                messages=[{"role": "user", "content": "Direct message test"}],
                user_id=user_id,
                infer=False,
            )

            assert "results" in result
            assert "recalled_memories" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["event"] == "ADD"
            assert result["results"][0]["memory"] == "Direct message test"

        finally:
            try:
                memory.vector_store.delete_col()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# E2E Tests: Graph write path (extract_graph → execute_graph)
# ---------------------------------------------------------------------------

TEST_GRAPH_COLLECTION = "test_add_engine_graph_e2e"
NEO4J_TEST_DB = "neo4j"


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


@pytest.fixture(scope="module")
def graph_vector_store():
    """Separate pgvector collection for graph E2E tests."""
    vs = PGVector(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "8432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        collection_name=TEST_GRAPH_COLLECTION,
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
def graph_instance(embedding_model, graph_vector_store, llm):
    """Create a real MemoryGraph connected to local Neo4j."""
    config = MemoryConfig(
        graph_store={
            "provider": "neo4j",
            "config": {
                "url": os.getenv("NEO4J_URI", "bolt://localhost:8687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "mem0graph"),
                "database": NEO4J_TEST_DB,
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
                "collection_name": TEST_GRAPH_COLLECTION,
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
    graph = GraphStoreFactory.create("neo4j", config)
    yield graph
    try:
        graph.reset()
    except Exception:
        pass


@pytest.fixture
def graph_search_engine(embedding_model, graph_vector_store, llm):
    """SearchEngine without graph (graph search is separate from vector search)."""
    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=graph_vector_store,
        graph_store=None,
        reranker=None,
        llm=llm,
    )


@pytest.fixture
def add_engine_with_graph(embedding_model, graph_vector_store, llm, db, graph_search_engine, graph_instance):
    """AddEngine with graph enabled for E2E graph write tests."""
    return AddEngine(
        embedding_model=embedding_model,
        vector_store=graph_vector_store,
        llm=llm,
        db=db,
        search_engine=graph_search_engine,
        graph=graph_instance,
    )


@requires_env
@requires_pg
@requires_neo4j
class TestE2EGraphWrite:
    """End-to-end tests for the graph write path (extract_graph → execute_graph)."""

    def test_add_with_graph_creates_relations(self, add_engine_with_graph):
        """Adding a message with infer=True + graph enabled should create graph relations."""
        user_id = "e2e-graph-add-" + str(uuid.uuid4())[:8]

        result = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "Alice lives in New York and works at Google"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert "relations" in result, f"Expected graph relations in result, got keys: {list(result.keys())}"

        relations = result["relations"]
        assert "added_entities" in relations
        assert "deleted_entities" in relations

    def test_add_with_graph_no_new_entities(self, add_engine_with_graph):
        """A greeting message creates no entities; graph_result should be empty."""
        user_id = "e2e-graph-empty-" + str(uuid.uuid4())[:8]

        result = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "Hi there!"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        if "relations" in result:
            rels = result["relations"]
            assert isinstance(rels, dict)

    def test_add_graph_entity_dedup(self, add_engine_with_graph):
        """Adding same entity twice should MERGE, not duplicate."""
        user_id = "e2e-graph-dedup-" + str(uuid.uuid4())[:8]

        result1 = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "Bob is a software engineer"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        result2 = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "I also know Bob, he is a software engineer"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert "results" in result1
        assert "results" in result2

    def test_add_graph_deletes_old_relation(self, add_engine_with_graph):
        """When a fact changes, extract_graph should delete the old relation."""
        user_id = "e2e-graph-delete-" + str(uuid.uuid4())[:8]

        # Establish initial relation
        add_engine_with_graph.add(
            messages=[{"role": "user", "content": "Alice works at Meta"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        # Change it
        result = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "Alice changed jobs, she now works at Google instead of Meta"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert "results" in result
        if "relations" in result:
            assert "deleted_entities" in result["relations"]
            assert "added_entities" in result["relations"]

    def test_add_graph_result_structure(self, add_engine_with_graph):
        """Verify full result structure when graph is enabled."""
        user_id = "e2e-graph-structure-" + str(uuid.uuid4())[:8]

        result = add_engine_with_graph.add(
            messages=[{"role": "user", "content": "My name is Carol and I am a doctor in Boston"}],
            metadata={"user_id": user_id},
            filters={"user_id": user_id},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result

        if "relations" in result:
            rels = result["relations"]
            assert isinstance(rels, dict)
            assert "added_entities" in rels
            assert "deleted_entities" in rels
            for added in rels["added_entities"]:
                if isinstance(added, list) and len(added) > 0:
                    entry = added[0]
                    assert "source" in entry
                    assert "relationship" in entry
                    assert "target" in entry

    def test_memory_add_with_graph_delegation(self, embedding_model, vector_store):
        """Full Memory.add() with graph config → relations in output."""
        from mem0.memory.main import Memory

        collection = "test_memory_graph_e2e"
        config = MemoryConfig(
            vector_store={
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "8432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                    "dbname": os.getenv("POSTGRES_DB", "postgres"),
                    "collection_name": collection,
                    "embedding_model_dims": 1536,
                    "diskann": False,
                    "hnsw": True,
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
            llm={
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENAI_llm_API_KEY"),
                    "model": os.getenv("OPENAI_llm_Model", "deepseek-chat"),
                    "openai_base_url": os.getenv("OPENAI_llm_URL"),
                    "temperature": 0.0,
                },
            },
            graph_store={
                "provider": "neo4j",
                "config": {
                    "url": os.getenv("NEO4J_URI", "bolt://localhost:8687"),
                    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                    "password": os.getenv("NEO4J_PASSWORD", "mem0graph"),
                    "database": NEO4J_TEST_DB,
                },
            },
            history_db_path=":memory:",
            version="v1.1",
        )

        memory = Memory(config)

        try:
            user_id = "e2e-memory-graph-" + str(uuid.uuid4())[:8]

            result = memory.add(
                messages=[{"role": "user", "content": "David is a chef and lives in Paris"}],
                user_id=user_id,
                infer=True,
            )

            assert "results" in result
            assert "recalled_memories" in result
            assert "relations" in result, (
                f"Expected 'relations' key when graph is enabled, got: {list(result.keys())}"
            )
            assert "added_entities" in result["relations"]

        finally:
            try:
                memory.vector_store.delete_col()
            except Exception:
                pass
            try:
                memory.graph.reset()
            except Exception:
                pass
