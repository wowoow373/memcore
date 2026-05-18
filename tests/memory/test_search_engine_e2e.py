"""End-to-end tests for SearchEngine with real backend services.

Prerequisites:
    - Docker services running (postgres:8432, neo4j:8687)
    - Environment variables loaded from server/.env
"""

import os
import uuid
from unittest.mock import MagicMock

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.openai import OpenAIEmbedding
from mem0.memory.search_engine import SearchEngine
from mem0.vector_stores.pgvector import PGVector


@pytest.fixture(scope="module")
def embedding_model():
    """Create a real embedding model instance."""
    config = BaseEmbedderConfig(
        api_key=os.getenv("OPENAI_EMBEDDER_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-v4"),
        openai_base_url=os.getenv("OPENAI_EMBEDDER_URL"),
    )
    return OpenAIEmbedding(config)


@pytest.fixture(scope="module")
def vector_store():
    """Create a real pgvector instance for E2E testing."""
    vs = PGVector(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "8432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        collection_name="test_search_engine_e2e",
        embedding_model_dims=1536,
        diskann=False,
        hnsw=True,
    )
    yield vs
    # Cleanup: drop the test collection
    try:
        vs.delete_col()
    except Exception:
        pass


@pytest.fixture
def search_engine(embedding_model, vector_store):
    """Create a SearchEngine with real vector backend (no graph)."""
    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        graph_store=None,
        reranker=None,
    )


class TestE2EVectorRecall:
    """E2E tests for vector search pipeline with real pgvector + OpenAI embedding."""

    def test_e2e_vector_recall_basic(self, search_engine, vector_store, embedding_model):
        """Insert a memory via vector_store directly and verify search recalls it."""
        test_id = str(uuid.uuid4())
        test_text = "I love hiking in the mountains on weekends"

        # Generate embedding and insert directly
        embedding = embedding_model.embed(test_text, "add")
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "test-hash-1",
                "user_id": "e2e-test-user",
                "created_at": "2024-01-01T00:00:00",
            }],
        )

        # Search for semantically related query
        result = search_engine.search(
            query="What are my weekend hobbies?",
            filters={"user_id": "e2e-test-user"},
            limit=10,
        )

        assert "results" in result
        assert "relations" not in result
        assert len(result["results"]) >= 1

        # Find our inserted memory in results
        ids = [r["id"] for r in result["results"]]
        assert test_id in ids

        # Verify structure
        matched = [r for r in result["results"] if r["id"] == test_id][0]
        assert matched["memory"] == test_text
        assert matched["user_id"] == "e2e-test-user"
        assert "score" in matched

    def test_e2e_vector_threshold_filtering(self, search_engine, vector_store, embedding_model):
        """Verify that threshold filters out low-score results."""
        user_id = f"e2e-threshold-{uuid.uuid4().hex[:8]}"

        # Insert two memories with very different semantic content
        text1 = "Python is my favorite programming language for building software"
        text2 = "The weather forecast says it will rain tomorrow afternoon"
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())

        emb1 = embedding_model.embed(text1, "add")
        emb2 = embedding_model.embed(text2, "add")

        vector_store.insert(
            vectors=[emb1, emb2],
            ids=[id1, id2],
            payloads=[
                {"data": text1, "hash": "h1", "user_id": user_id},
                {"data": text2, "hash": "h2", "user_id": user_id},
            ],
        )

        # Search for programming-related query (should match text1 strongly, text2 weakly)
        result_all = search_engine.search(
            query="coding and software development",
            filters={"user_id": user_id},
            limit=10,
            threshold=None,
        )
        assert len(result_all["results"]) == 2
        scores = {r["id"]: r["score"] for r in result_all["results"]}

        # Pick a threshold between the two scores to filter out the weaker match
        score1 = scores.get(id1, 0)
        score2 = scores.get(id2, 0)
        mid_threshold = (score1 + score2) / 2

        # Apply threshold — should keep only the stronger match
        result_filtered = search_engine.search(
            query="coding and software development",
            filters={"user_id": user_id},
            limit=10,
            threshold=mid_threshold,
        )
        filtered_ids = [r["id"] for r in result_filtered["results"]]

        # The programming text should have higher score than weather text
        if score1 > score2:
            assert id1 in filtered_ids
            assert id2 not in filtered_ids
        else:
            assert id2 in filtered_ids
            assert id1 not in filtered_ids

    def test_e2e_vector_merge_dedup(self, search_engine, vector_store, embedding_model):
        """Verify duplicate IDs are deduplicated (highest score kept)."""
        user_id = f"e2e-dedup-{uuid.uuid4().hex[:8]}"
        test_id = str(uuid.uuid4())
        test_text = "I enjoy reading science fiction novels"

        embedding = embedding_model.embed(test_text, "add")

        # Insert once
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "h1",
                "user_id": user_id,
            }],
        )

        # Search should return exactly one result for this id
        result = search_engine.search(
            query="What kind of books do I like?",
            filters={"user_id": user_id},
            limit=10,
        )

        matched_ids = [r["id"] for r in result["results"]]
        assert matched_ids.count(test_id) == 1

    def test_e2e_vector_empty_results(self, search_engine):
        """Search with a user that has no memories should return empty results."""
        result = search_engine.search(
            query="Something completely unrelated",
            filters={"user_id": f"nonexistent-{uuid.uuid4().hex[:8]}"},
            limit=10,
        )

        assert "results" in result
        assert result["results"] == []


class Neo4jGraphStore:
    """Minimal graph store wrapper that queries Neo4j directly via Cypher.

    Bypasses MemoryGraph's LLM-based entity extraction for stable E2E testing.
    """

    def __init__(self, driver):
        self.driver = driver

    def search_nodes(self, node_names, filters, depth=2, limit=100):
        """Pure graph traversal from given node names (new interface)."""
        names = [n.lower().replace(" ", "_") for n in node_names if n]
        if not names:
            return []
        safe_depth = max(1, int(depth))

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n:__Entity__ {{user_id: $user_id}})
                WHERE n.name IN $names
                MATCH path = (n)-[*1..{safe_depth}]-(m:__Entity__ {{user_id: $user_id}})
                UNWIND relationships(path) AS rel
                WITH DISTINCT startNode(rel) AS src, rel, endNode(rel) AS dst
                WHERE src.user_id = $user_id AND dst.user_id = $user_id
                RETURN src.name AS source, type(rel) AS relationship, dst.name AS destination
                LIMIT $limit
                """,
                user_id=filters.get("user_id", "user"),
                names=names,
                limit=limit,
            )
            return [dict(record) for record in result]

    def search(self, query, filters, limit):
        """Backward-compatible wrapper."""
        if isinstance(query, str):
            node_names = [query.strip()]
        elif isinstance(query, (list, tuple, set)):
            node_names = [str(n).strip() for n in query if str(n).strip()]
        else:
            node_names = [str(query).strip()] if str(query).strip() else []
        return self.search_nodes(node_names, filters, depth=limit, limit=limit * 20)

    def delete_all(self, filters):
        """Delete all nodes for a given user."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n:__Entity__ {user_id: $user_id})
                DETACH DELETE n
                """,
                user_id=filters.get("user_id", "user"),
            )


@pytest.fixture(scope="module")
def neo4j_driver():
    """Create a direct Neo4j driver connection."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:8687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "mem0graph"),
        ),
    )
    yield driver
    driver.close()


@pytest.fixture
def search_engine_with_graph(embedding_model, vector_store, neo4j_driver):
    """Create a SearchEngine with real vector + real neo4j graph backend (no LLM)."""
    graph_store = Neo4jGraphStore(neo4j_driver)
    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        graph_store=graph_store,
        reranker=None,
    )


@pytest.fixture
def mock_llm_for_entity_extraction():
    """Mock LLM that extracts entities via tool-call (mirrors real LLM behavior)."""
    llm = MagicMock()
    llm.__class__.__name__ = "OpenAILLM"
    llm.generate_response.return_value = {
        "tool_calls": [
            {
                "name": "extract_entities",
                "arguments": {
                    "entities": [
                        {"entity": "Alice", "entity_type": "person"},
                        {"entity": "Bob", "entity_type": "person"},
                    ]
                },
            }
        ]
    }
    return llm


@pytest.fixture
def search_engine_with_graph_and_llm(
    embedding_model, vector_store, neo4j_driver, mock_llm_for_entity_extraction
):
    """Create a SearchEngine with real vector + real neo4j + mock LLM for entity extraction."""
    graph_store = Neo4jGraphStore(neo4j_driver)
    return SearchEngine(
        embedding_model=embedding_model,
        vector_store=vector_store,
        graph_store=graph_store,
        reranker=None,
        llm=mock_llm_for_entity_extraction,
    )


class TestE2EGraphRecall:
    """E2E tests for graph traversal pipeline with real Neo4j."""

    def test_e2e_graph_recall_basic(
        self, search_engine_with_graph, vector_store, embedding_model, neo4j_driver
    ):
        """Insert vector + graph data, verify both are recalled."""
        user_id = f"e2e-graph-{uuid.uuid4().hex[:8]}"
        test_text = "Alice is a good friend of Bob"
        test_id = str(uuid.uuid4())

        # 1. Insert vector data
        embedding = embedding_model.embed(test_text, "add")
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "h1",
                "user_id": user_id,
            }],
        )

        # 2. Insert graph data directly via Cypher
        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (a:__Entity__ {name: 'alice', user_id: $user_id})
                CREATE (b:__Entity__ {name: 'bob', user_id: $user_id})
                CREATE (a)-[:KNOWS]->(b)
                """,
                user_id=user_id,
            )

        try:
            # 3. Search with graph_depth=2
            result = search_engine_with_graph.search(
                query="alice",
                filters={"user_id": user_id},
                limit=10,
                graph_depth=2,
            )

            # 4. Verify vector results
            assert "results" in result
            assert "relations" in result

            # Vector result should contain our inserted memory
            ids = [r["id"] for r in result["results"]]
            assert test_id in ids

            # Graph result should contain the KNOWS relation
            assert len(result["relations"]) >= 1
            rel = result["relations"][0]
            assert rel["source"] == "alice"
            assert rel["relationship"] == "KNOWS"
            assert rel["destination"] == "bob"
        finally:
            # Cleanup graph nodes
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )

    def test_e2e_graph_depth_zero_skips_graph(
        self, search_engine_with_graph, vector_store, embedding_model, neo4j_driver
    ):
        """graph_depth=0 should not traverse graph and omit relations."""
        user_id = f"e2e-graph-d0-{uuid.uuid4().hex[:8]}"
        test_text = "Charlie works with Diana"
        test_id = str(uuid.uuid4())

        # Insert vector data
        embedding = embedding_model.embed(test_text, "add")
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "h1",
                "user_id": user_id,
            }],
        )

        # Insert graph data
        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (c:__Entity__ {name: 'charlie', user_id: $user_id})
                CREATE (d:__Entity__ {name: 'diana', user_id: $user_id})
                CREATE (c)-[:WORKS_WITH]->(d)
                """,
                user_id=user_id,
            )

        try:
            # Search with graph_depth=0
            result = search_engine_with_graph.search(
                query="charlie",
                filters={"user_id": user_id},
                limit=10,
                graph_depth=0,
            )

            # Should have vector results and empty relations (enable_graph=True)
            assert "results" in result
            assert "relations" in result
            assert result["relations"] == []
            assert test_id in [r["id"] for r in result["results"]]
        finally:
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )

    def test_e2e_graph_multi_hop_traversal(
        self, search_engine_with_graph, vector_store, embedding_model, neo4j_driver
    ):
        """Traverse multiple hops: alice -> bob -> carol."""
        user_id = f"e2e-graph-multi-{uuid.uuid4().hex[:8]}"
        test_text = "Alice knows Bob who knows Carol"
        test_id = str(uuid.uuid4())

        embedding = embedding_model.embed(test_text, "add")
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "h1",
                "user_id": user_id,
            }],
        )

        # Create a 3-node chain
        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (a:__Entity__ {name: 'alice', user_id: $user_id})
                CREATE (b:__Entity__ {name: 'bob', user_id: $user_id})
                CREATE (c:__Entity__ {name: 'carol', user_id: $user_id})
                CREATE (a)-[:KNOWS]->(b)
                CREATE (b)-[:KNOWS]->(c)
                """,
                user_id=user_id,
            )

        try:
            # graph_depth=2 should traverse alice->bob->carol
            result = search_engine_with_graph.search(
                query="alice",
                filters={"user_id": user_id},
                limit=10,
                graph_depth=2,
            )

            assert "relations" in result
            assert len(result["relations"]) >= 2

            sources = {r["source"] for r in result["relations"]}
            destinations = {r["destination"] for r in result["relations"]}
            assert "alice" in sources
            assert "carol" in destinations
        finally:
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )


class TestE2EGraphRecallWithLLMExtraction:
    """E2E tests verifying LLM entity extraction moved to SearchEngine layer."""

    def test_e2e_graph_recall_uses_llm_extraction(
        self,
        search_engine_with_graph_and_llm,
        mock_llm_for_entity_extraction,
        vector_store,
        embedding_model,
        neo4j_driver,
    ):
        """LLM extracts entities from natural language query, then search_nodes traverses graph."""
        user_id = f"e2e-llm-extract-{uuid.uuid4().hex[:8]}"
        test_text = "Alice is a good friend of Bob"
        test_id = str(uuid.uuid4())

        # 1. Insert vector data
        embedding = embedding_model.embed(test_text, "add")
        vector_store.insert(
            vectors=[embedding],
            ids=[test_id],
            payloads=[{
                "data": test_text,
                "hash": "h1",
                "user_id": user_id,
            }],
        )

        # 2. Insert graph data directly via Cypher
        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (a:__Entity__ {name: 'alice', user_id: $user_id})
                CREATE (b:__Entity__ {name: 'bob', user_id: $user_id})
                CREATE (a)-[:KNOWS]->(b)
                """,
                user_id=user_id,
            )

        try:
            # 3. Search with natural language query
            result = search_engine_with_graph_and_llm.search(
                query="Alice and Bob are friends",
                filters={"user_id": user_id},
                limit=10,
                graph_depth=2,
            )

            # 4. Verify vector results
            assert "results" in result
            assert "relations" in result
            ids = [r["id"] for r in result["results"]]
            assert test_id in ids

            # 5. Verify graph results contain the KNOWS relation
            assert len(result["relations"]) >= 1
            rel = result["relations"][0]
            assert rel["source"] == "alice"
            assert rel["relationship"] == "KNOWS"
            assert rel["destination"] == "bob"

            # 6. Verify LLM was called for entity extraction
            mock_llm_for_entity_extraction.generate_response.assert_called()
            call_messages = mock_llm_for_entity_extraction.generate_response.call_args[1]["messages"]
            assert any("Alice and Bob are friends" in m.get("content", "") for m in call_messages)
        finally:
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )

    def test_e2e_llm_extraction_self_reference(
        self,
        search_engine_with_graph_and_llm,
        mock_llm_for_entity_extraction,
        neo4j_driver,
    ):
        """LLM system prompt contains user_id for self-reference resolution."""
        user_id = f"e2e-self-ref-{uuid.uuid4().hex[:8]}"

        with neo4j_driver.session() as session:
            session.run(
                """
                CREATE (a:__Entity__ {name: 'alice', user_id: $user_id})
                CREATE (b:__Entity__ {name: 'bob', user_id: $user_id})
                CREATE (a)-[:KNOWS]->(b)
                """,
                user_id=user_id,
            )

        # Override mock to simulate self-reference extraction
        mock_llm_for_entity_extraction.generate_response.return_value = {
            "tool_calls": [
                {
                    "name": "extract_entities",
                    "arguments": {
                        "entities": [
                            {"entity": user_id, "entity_type": "user"},
                            {"entity": "Bob", "entity_type": "person"},
                        ]
                    },
                }
            ]
        }

        try:
            result = search_engine_with_graph_and_llm.search(
                query="I know Bob",
                filters={"user_id": user_id},
                limit=10,
                graph_depth=2,
            )

            assert "relations" in result
            # LLM system prompt should contain user_id
            call_messages = mock_llm_for_entity_extraction.generate_response.call_args[1]["messages"]
            system_content = call_messages[0]["content"]
            assert user_id in system_content
        finally:
            with neo4j_driver.session() as session:
                session.run(
                    "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
                    user_id=user_id,
                )
