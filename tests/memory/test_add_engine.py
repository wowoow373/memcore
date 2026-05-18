"""Unit tests for AddEngine — each node method is independently testable."""

import json
import logging
from unittest.mock import MagicMock, call, patch

import pytest

from mem0.memory.add_engine import (
    DECIDE_MEMORY_SYSTEM_PROMPT,
    EXTRACT_GRAPH_SYSTEM_PROMPT,
    EXTRACT_QUERIES_PROMPT,
    AddEngine,
    AddState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    embedding_model=None,
    vector_store=None,
    llm=None,
    db=None,
    search_engine=None,
    graph=None,
):
    return AddEngine(
        embedding_model=embedding_model or MagicMock(),
        vector_store=vector_store or MagicMock(),
        llm=llm or MagicMock(),
        db=db or MagicMock(),
        search_engine=search_engine or MagicMock(),
        graph=graph,
    )


def _make_state(**overrides):
    defaults: AddState = {
        "messages": [],
        "metadata": {},
        "filters": {},
        "infer": True,
        "parsed_messages": "",
        "search_queries": [],
        "recalled_memories": {"results": [], "relations": []},
        "decisions": [],
        "entity_type_map": {},
        "relations": [],
        "to_be_deleted": [],
        "graph_result": {},
        "results": [],
        "final_results": {},
        "error": None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Init & graph construction
# ---------------------------------------------------------------------------

class TestAddEngineInit:
    def test_init_basic(self):
        engine = _make_engine()
        assert engine.embedding_model is not None
        assert engine.vector_store is not None
        assert engine.llm is not None
        assert engine.db is not None
        assert engine.search_engine is not None
        assert engine.enable_graph is False
        assert engine.graph is None
        assert engine.add_graph is not None  # compiled LangGraph

    def test_init_with_graph(self):
        mock_graph = MagicMock()
        engine = _make_engine(graph=mock_graph)
        assert engine.enable_graph is True
        assert engine.graph is mock_graph

    def test_init_compiles_graph(self):
        engine = _make_engine()
        assert hasattr(engine.add_graph, "invoke")


# ---------------------------------------------------------------------------
# Conditional edge logic
# ---------------------------------------------------------------------------

class TestConditionalEdges:
    def test_should_infer_true(self):
        engine = _make_engine()
        state = _make_state(infer=True)
        assert engine._should_infer(state) == "extract_queries"

    def test_should_infer_false(self):
        engine = _make_engine()
        state = _make_state(infer=False)
        assert engine._should_infer(state) == "direct_add"

    def test_should_extract_graph_with_graph(self):
        mock_graph = MagicMock()
        engine = _make_engine(graph=mock_graph)
        state = _make_state()
        assert engine._should_extract_graph(state) == "extract_graph"

    def test_should_extract_graph_without_graph(self):
        engine = _make_engine(graph=None)
        state = _make_state()
        assert engine._should_extract_graph(state) == "assemble_result"


# ---------------------------------------------------------------------------
# Node: preprocess
# ---------------------------------------------------------------------------

class TestPreprocessMessages:
    def test_simple_user_message(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hello world"}]
        result = engine._preprocess_messages(messages)
        assert result == "user: Hello world"

    def test_multiple_messages(self):
        engine = _make_engine()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = engine._preprocess_messages(messages)
        assert result == "user: Hi\nassistant: Hello!\nuser: How are you?"

    def test_skips_system_messages(self):
        engine = _make_engine()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = engine._preprocess_messages(messages)
        assert result == "user: Hello"
        assert "system" not in result

    def test_empty_messages(self):
        engine = _make_engine()
        result = engine._preprocess_messages([])
        assert result == ""

    def test_message_with_empty_content(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": ""}]
        result = engine._preprocess_messages(messages)
        assert result == "user: "

    def test_unknown_role_skipped(self):
        engine = _make_engine()
        messages = [
            {"role": "tool", "content": "tool output"},
            {"role": "user", "content": "Hello"},
        ]
        result = engine._preprocess_messages(messages)
        assert result == "user: Hello"


class TestNodePreprocess:
    def test_node_preprocess(self):
        engine = _make_engine()
        state = _make_state(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        result = engine._node_preprocess(state)
        assert result["parsed_messages"] == "user: Hi\nassistant: Hello!"


# ---------------------------------------------------------------------------
# Node: direct_add
# ---------------------------------------------------------------------------

class TestDirectAddMessages:
    def test_single_message(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_vector = MagicMock()
        mock_db = MagicMock()

        engine = _make_engine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            db=mock_db,
        )

        messages = [{"role": "user", "content": "Hello world"}]
        metadata = {"user_id": "u1"}

        results = engine._direct_add_messages(messages, metadata)

        assert len(results) == 1
        assert results[0]["memory"] == "Hello world"
        assert results[0]["event"] == "ADD"
        assert "id" in results[0]
        mock_embedder.embed.assert_called_once_with("Hello world", "add")
        mock_vector.insert.assert_called_once()
        mock_db.add_history.assert_called_once()

    def test_multiple_messages(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        engine = _make_engine(embedding_model=mock_embedder)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        metadata = {"user_id": "u1"}

        results = engine._direct_add_messages(messages, metadata)

        assert len(results) == 2
        assert results[0]["role"] == "user"
        assert results[1]["role"] == "assistant"

    def test_skips_system(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1]
        engine = _make_engine(embedding_model=mock_embedder)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ]
        results = engine._direct_add_messages(messages, {"user_id": "u1"})

        assert len(results) == 1
        assert results[0]["memory"] == "Hi"

    def test_extracts_actor_name(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1]
        engine = _make_engine(embedding_model=mock_embedder)

        messages = [{"role": "user", "content": "Hi", "name": "Alice"}]
        results = engine._direct_add_messages(messages, {"user_id": "u1"})

        assert results[0]["actor_id"] == "Alice"

    def test_skips_empty_content(self, caplog):
        mock_embedder = MagicMock()
        engine = _make_engine(embedding_model=mock_embedder)

        messages = [{"role": "user", "content": "   "}]
        with caplog.at_level(logging.WARNING):
            results = engine._direct_add_messages(messages, {"user_id": "u1"})

        assert len(results) == 0
        mock_embedder.embed.assert_not_called()

    def test_skips_invalid_format(self, caplog):
        mock_embedder = MagicMock()
        engine = _make_engine(embedding_model=mock_embedder)

        messages = [{"no_role": True}]
        with caplog.at_level(logging.WARNING):
            results = engine._direct_add_messages(messages, {"user_id": "u1"})

        assert len(results) == 0


class TestNodeDirectAdd:
    def test_node_direct_add(self):
        engine = _make_engine()
        messages = [{"role": "user", "content": "Hi"}]
        metadata = {"user_id": "u1"}
        state = _make_state(messages=messages, metadata=metadata)

        with patch.object(engine, "_direct_add_messages", return_value=[{"id": "abc", "memory": "Hi", "event": "ADD"}]) as mock_method:
            result = engine._node_direct_add(state)
            mock_method.assert_called_once_with(messages, metadata)
            assert result["results"] == [{"id": "abc", "memory": "Hi", "event": "ADD"}]


# ---------------------------------------------------------------------------
# Node: extract_queries
# ---------------------------------------------------------------------------

class TestExtractSearchQueries:
    def test_successful_extraction(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"queries": ["用户喜欢什么食物", "用户住哪里"]}'
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: 我喜欢吃披萨\nassistant: 好的")
        assert queries == ["用户喜欢什么食物", "用户住哪里"]
        mock_llm.generate_response.assert_called_once()

    def test_empty_parsed_messages(self):
        mock_llm = MagicMock()
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("")
        assert queries == []
        mock_llm.generate_response.assert_not_called()

    def test_fallback_to_parsed_messages_on_empty_response(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = ""
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: Hello")
        assert queries == ["user: Hello"]

    def test_fallback_on_json_parse_error(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "not valid json"
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: Hello")
        assert queries == ["user: Hello"]

    def test_uses_parsed_messages_as_query_when_queries_empty(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"queries": []}'
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: Hi")
        assert queries == ["user: Hi"]

    def test_llm_error_fallback(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = RuntimeError("LLM down")
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: Hi")
        assert queries == ["user: Hi"]

    def test_filters_empty_query_strings(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"queries": ["good query", "", "  "]}'
        engine = _make_engine(llm=mock_llm)

        queries = engine._extract_search_queries("user: Hi")
        assert queries == ["good query"]

    def test_prompt_contains_parsed_messages(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"queries": ["q1"]}'
        engine = _make_engine(llm=mock_llm)

        engine._extract_search_queries("user: test content")

        call_args = mock_llm.generate_response.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == EXTRACT_QUERIES_PROMPT
        assert "test content" in messages[1]["content"]


class TestNodeExtractQueries:
    def test_node_extract_queries(self):
        engine = _make_engine()
        state = _make_state(parsed_messages="user: Hello world")

        with patch.object(engine, "_extract_search_queries", return_value=["Hello world"]) as mock_method:
            result = engine._node_extract_queries(state)
            mock_method.assert_called_once_with("user: Hello world")
            assert result["search_queries"] == ["Hello world"]


# ---------------------------------------------------------------------------
# Node: search
# ---------------------------------------------------------------------------

class TestSearchMemories:
    def test_search_single_query(self):
        mock_search = MagicMock()
        mock_search.search.return_value = {
            "results": [{"id": "mem-1", "memory": "text", "score": 0.9}],
            "relations": [],
        }
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["query1"], {"user_id": "u1"})

        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "mem-1"
        mock_search.search.assert_called_once_with(
            query="query1", filters={"user_id": "u1"},
            limit=10, threshold=None, graph_depth=2, rerank=True,
        )

    def test_search_multiple_queries_merges_results(self):
        mock_search = MagicMock()
        mock_search.search.side_effect = [
            {"results": [{"id": "mem-1", "memory": "A", "score": 0.9}], "relations": []},
            {"results": [{"id": "mem-2", "memory": "B", "score": 0.8}], "relations": []},
        ]
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["q1", "q2"], {"user_id": "u1"})

        assert len(result["results"]) == 2
        assert mock_search.search.call_count == 2

    def test_dedups_by_id_keeps_highest_score(self):
        mock_search = MagicMock()
        mock_search.search.side_effect = [
            {"results": [{"id": "mem-1", "memory": "First", "score": 0.6}], "relations": []},
            {"results": [{"id": "mem-1", "memory": "Duplicate", "score": 0.95}], "relations": []},
        ]
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["q1", "q2"], {"user_id": "u1"})

        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.95

    def test_dedups_graph_relations(self):
        mock_search = MagicMock()
        mock_search.search.side_effect = [
            {
                "results": [],
                "relations": [{"source": "A", "relationship": "r", "destination": "B"}],
            },
            {
                "results": [],
                "relations": [{"source": "A", "relationship": "r", "destination": "B"}],
            },
        ]
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["q1", "q2"], {"user_id": "u1"})

        assert len(result["relations"]) == 1

    def test_search_error_continues(self):
        mock_search = MagicMock()
        mock_search.search.side_effect = [
            RuntimeError("search failed"),
            {"results": [{"id": "mem-1", "memory": "B", "score": 0.8}], "relations": []},
        ]
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["q1", "q2"], {"user_id": "u1"})

        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "mem-1"

    def test_empty_queries(self):
        mock_search = MagicMock()
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories([], {"user_id": "u1"})

        assert result == {"results": [], "relations": []}
        mock_search.search.assert_not_called()

    def test_skips_result_without_id(self):
        mock_search = MagicMock()
        mock_search.search.return_value = {
            "results": [
                {"memory": "no id"},
                {"id": "mem-1", "memory": "with id", "score": 0.7},
            ],
            "relations": [],
        }
        engine = _make_engine(search_engine=mock_search)

        result = engine._search_memories(["q1"], {"user_id": "u1"})

        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "mem-1"


class TestNodeSearch:
    def test_node_search(self):
        engine = _make_engine()
        state = _make_state(
            search_queries=["q1"],
            filters={"user_id": "u1"},
        )

        with patch.object(engine, "_search_memories", return_value={"results": [{"id": "x"}], "relations": []}) as mock_method:
            result = engine._node_search(state)
            mock_method.assert_called_once_with(["q1"], {"user_id": "u1"})
            assert result["recalled_memories"] == {"results": [{"id": "x"}], "relations": []}


# ---------------------------------------------------------------------------
# Node: decide_memory
# ---------------------------------------------------------------------------

class TestDecideMemoryActions:
    def test_add_new_fact_no_existing(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": None, "text": "Name is John", "event": "ADD"}
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions(
            "user: My name is John",
            [],  # no recalled memories
        )

        assert len(decisions) == 1
        assert decisions[0]["id"] is None
        assert decisions[0]["text"] == "Name is John"
        assert decisions[0]["event"] == "ADD"

    def test_add_fact_with_existing_memories(self):
        existing = [
            {"id": "uuid-1", "memory": "Likes pizza", "score": 0.9},
        ]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": None, "text": "Name is John", "event": "ADD"},
                {"id": "uuid-1", "text": "Likes pizza", "event": "NONE"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: My name is John", existing)

        assert len(decisions) == 2
        add_decision = [d for d in decisions if d["event"] == "ADD"][0]
        assert add_decision["id"] is None
        none_decision = [d for d in decisions if d["event"] == "NONE"][0]
        assert none_decision["id"] == "uuid-1"

    def test_update_existing(self):
        existing = [
            {"id": "uuid-1", "memory": "Likes to play cricket", "score": 0.8},
        ]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": "uuid-1", "text": "Loves to play cricket with friends", "event": "UPDATE", "old_memory": "Likes to play cricket"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions(
            "user: I love to play cricket with friends", existing
        )

        assert len(decisions) == 1
        assert decisions[0]["event"] == "UPDATE"
        assert decisions[0]["id"] == "uuid-1"
        assert decisions[0]["old_memory"] == "Likes to play cricket"

    def test_delete_existing(self):
        existing = [
            {"id": "uuid-1", "memory": "Loves cheese pizza", "score": 0.9},
        ]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": "uuid-1", "text": "Loves cheese pizza", "event": "DELETE"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions(
            "user: I dislike cheese pizza", existing
        )

        assert len(decisions) == 1
        assert decisions[0]["event"] == "DELETE"
        assert decisions[0]["id"] == "uuid-1"

    def test_validates_event_types(self):
        existing = [{"id": "uuid-1", "memory": "test", "score": 0.9}]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": "uuid-1", "text": "test", "event": "INVALID"},
                {"id": None, "text": "new", "event": "ADD"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: Hi", existing)

        assert len(decisions) == 1  # invalid filtered out
        assert decisions[0]["event"] == "ADD"

    def test_ensures_add_id_is_null(self):
        existing = [{"id": "uuid-1", "memory": "test", "score": 0.9}]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": "some-fake-id", "text": "new", "event": "ADD"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: Hi", existing)

        assert decisions[0]["id"] is None  # forced to None for ADD

    def test_update_with_invalid_id_converts_to_none(self):
        existing = [{"id": "uuid-1", "memory": "test", "score": 0.9}]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "memory": [
                {"id": "uuid-nonexistent", "text": "blah", "event": "UPDATE"},
            ]
        })
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: Hi", existing)

        assert decisions[0]["event"] == "NONE"

    def test_empty_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = ""
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: Hi", [])
        assert decisions == []

    def test_llm_call_error(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = RuntimeError("LLM error")
        engine = _make_engine(llm=mock_llm)

        decisions = engine._decide_memory_actions("user: Hi", [])
        assert decisions == []

    def test_prompt_includes_existing_memories_with_score(self):
        existing = [{"id": "uuid-1", "memory": "test", "score": 0.85}]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({"memory": []})
        engine = _make_engine(llm=mock_llm)

        engine._decide_memory_actions("user: Hi", existing)

        call_args = mock_llm.generate_response.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert "uuid-1" in user_content
        assert "0.85" in user_content
        assert "test" in user_content


class TestNodeDecideMemory:
    def test_node_decide_memory(self):
        engine = _make_engine()
        state = _make_state(
            parsed_messages="user: Hello",
            recalled_memories={"results": [{"id": "uuid-1", "memory": "old"}], "relations": []},
        )

        with patch.object(engine, "_decide_memory_actions", return_value=[{"id": None, "text": "new", "event": "ADD"}]) as mock_method:
            result = engine._node_decide_memory(state)
            mock_method.assert_called_once_with("user: Hello", [{"id": "uuid-1", "memory": "old"}])
            assert result["decisions"] == [{"id": None, "text": "new", "event": "ADD"}]


# ---------------------------------------------------------------------------
# Node: execute_vector
# ---------------------------------------------------------------------------

class TestExecuteVectorOperations:
    def test_add_operation(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        engine = _make_engine(embedding_model=mock_embedder)

        decisions = [{"id": None, "text": "New fact", "event": "ADD"}]
        metadata = {"user_id": "u1"}

        with patch.object(engine, "_create_memory", return_value="new-uuid") as mock_create:
            results = engine._execute_vector_operations(decisions, metadata)
            mock_create.assert_called_once()
            assert len(results) == 1
            assert results[0]["id"] == "new-uuid"
            assert results[0]["memory"] == "New fact"
            assert results[0]["event"] == "ADD"

    def test_update_operation(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.3, 0.4]
        engine = _make_engine(embedding_model=mock_embedder)

        decisions = [{
            "id": "uuid-1",
            "text": "Updated fact",
            "event": "UPDATE",
            "old_memory": "Old fact",
        }]
        metadata = {"user_id": "u1"}

        with patch.object(engine, "_update_memory", return_value="uuid-1") as mock_update:
            results = engine._execute_vector_operations(decisions, metadata)
            mock_update.assert_called_once()
            assert results[0]["id"] == "uuid-1"
            assert results[0]["event"] == "UPDATE"
            assert results[0]["previous_memory"] == "Old fact"

    def test_delete_operation(self):
        engine = _make_engine()

        decisions = [{"id": "uuid-1", "text": "", "event": "DELETE"}]
        metadata = {"user_id": "u1"}

        with patch.object(engine, "_delete_memory", return_value="uuid-1") as mock_delete:
            results = engine._execute_vector_operations(decisions, metadata)
            mock_delete.assert_called_once_with("uuid-1")
            assert results[0]["event"] == "DELETE"

    def test_none_skipped(self):
        engine = _make_engine()

        decisions = [{"id": "uuid-1", "text": "no change", "event": "NONE"}]
        results = engine._execute_vector_operations(decisions, {"user_id": "u1"})

        assert len(results) == 0

    def test_multiple_operations(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        engine = _make_engine(embedding_model=mock_embedder)

        decisions = [
            {"id": None, "text": "New", "event": "ADD"},
            {"id": "uuid-1", "text": "Updated", "event": "UPDATE", "old_memory": "Old"},
            {"id": "uuid-2", "text": "", "event": "DELETE"},
            {"id": "uuid-3", "text": "noop", "event": "NONE"},
        ]

        with patch.object(engine, "_create_memory", return_value="new-id"), \
             patch.object(engine, "_update_memory", return_value="uuid-1"), \
             patch.object(engine, "_delete_memory", return_value="uuid-2"):
            results = engine._execute_vector_operations(decisions, {"user_id": "u1"})

        assert len(results) == 3  # NONE excluded
        events = [r["event"] for r in results]
        assert events == ["ADD", "UPDATE", "DELETE"]

    def test_operation_error_logged_not_raised(self, caplog):
        engine = _make_engine()

        decisions = [{"id": "uuid-1", "text": "", "event": "DELETE"}]
        with patch.object(engine, "_delete_memory", side_effect=ValueError("not found")):
            with caplog.at_level(logging.ERROR):
                results = engine._execute_vector_operations(decisions, {"user_id": "u1"})

        assert len(results) == 0
        assert "Error executing DELETE" in caplog.text


class TestNodeExecuteVector:
    def test_node_execute_vector(self):
        engine = _make_engine()
        state = _make_state(
            decisions=[{"id": None, "text": "new", "event": "ADD"}],
            metadata={"user_id": "u1"},
        )

        with patch.object(engine, "_execute_vector_operations", return_value=[{"id": "abc", "memory": "new", "event": "ADD"}]) as mock_method:
            result = engine._node_execute_vector(state)
            mock_method.assert_called_once()
            assert result["results"] == [{"id": "abc", "memory": "new", "event": "ADD"}]


# ---------------------------------------------------------------------------
# Node: extract_graph
# ---------------------------------------------------------------------------

class TestExtractGraphData:
    def test_extract_entities_and_relations(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "entities": [{"entity": "Alice", "entity_type": "person"}],
            "relations": [{"source": "Alice", "relationship": "lives_in", "destination": "New York"}],
            "to_be_deleted": [],
        })
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, to_be_deleted = engine._extract_graph_data(
            "user: Alice lives in New York",
            {"user_id": "u1"},
            existing_relations=[],
        )

        assert entity_type_map == {"alice": "person"}
        assert relations == [{"source": "alice", "relationship": "lives_in", "destination": "new_york"}]
        assert to_be_deleted == []

    def test_extract_with_delete(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "entities": [{"entity": "Alice", "entity_type": "person"}],
            "relations": [{"source": "Alice", "relationship": "lives_in", "destination": "Boston"}],
            "to_be_deleted": [{"source": "Alice", "relationship": "lives_in", "destination": "New York"}],
        })
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, to_be_deleted = engine._extract_graph_data(
            "user: Alice moved from New York to Boston",
            {"user_id": "u1"},
            existing_relations=[{"source": "Alice", "relationship": "lives_in", "destination": "New York"}],
        )

        assert entity_type_map == {"alice": "person"}
        assert relations == [{"source": "alice", "relationship": "lives_in", "destination": "boston"}]
        assert to_be_deleted == [{"source": "alice", "relationship": "lives_in", "destination": "new_york"}]

    def test_no_existing_relations(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "entities": [],
            "relations": [],
            "to_be_deleted": [],
        })
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, to_be_deleted = engine._extract_graph_data(
            "user: Hello", {"user_id": "u1"}, []
        )

        assert entity_type_map == {}
        assert relations == []
        assert to_be_deleted == []
        # Verify "(No existing relationships)" in prompt
        user_content = mock_llm.generate_response.call_args[1]["messages"][1]["content"]
        assert "No existing relationships" in user_content

    def test_llm_error_graceful(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = RuntimeError("LLM failed")
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, to_be_deleted = engine._extract_graph_data(
            "user: Hi", {"user_id": "u1"}, []
        )

        assert entity_type_map == {}
        assert relations == []
        assert to_be_deleted == []

    def test_json_parsing_error_graceful(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "not valid json"
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, to_be_deleted = engine._extract_graph_data(
            "user: Hi", {"user_id": "u1"}, []
        )

        assert entity_type_map == {}
        assert relations == []
        assert to_be_deleted == []

    def test_normalizes_entities(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "entities": [{"entity": "Alice Smith", "entity_type": "human being"}],
            "relations": [{"source": "Alice Smith", "relationship": "Lives In", "destination": "New York City"}],
            "to_be_deleted": [],
        })
        engine = _make_engine(llm=mock_llm)

        entity_type_map, relations, _ = engine._extract_graph_data(
            "user: Hi", {"user_id": "u1"}, []
        )

        assert entity_type_map == {"alice_smith": "human_being"}
        assert relations == [{"source": "alice_smith", "relationship": "lives_in", "destination": "new_york_city"}]

    def test_user_id_in_system_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = json.dumps({
            "entities": [], "relations": [], "to_be_deleted": [],
        })
        engine = _make_engine(llm=mock_llm)

        engine._extract_graph_data("user: Hi", {"user_id": "test-user-123"}, [])

        system_msg = mock_llm.generate_response.call_args[1]["messages"][0]["content"]
        assert "test-user-123" in system_msg


class TestNodeExtractGraph:
    def test_node_extract_graph(self):
        engine = _make_engine()
        state = _make_state(
            parsed_messages="user: Hi",
            filters={"user_id": "u1"},
            recalled_memories={"results": [], "relations": []},
        )

        expected = ({"a": "person"}, [{"source": "a", "relationship": "r", "destination": "b"}], [])
        with patch.object(engine, "_extract_graph_data", return_value=expected) as mock_method:
            result = engine._node_extract_graph(state)
            mock_method.assert_called_once_with("user: Hi", {"user_id": "u1"}, [])
            assert result["entity_type_map"] == {"a": "person"}
            assert result["relations"] == [{"source": "a", "relationship": "r", "destination": "b"}]
            assert result["to_be_deleted"] == []


# ---------------------------------------------------------------------------
# Node: execute_graph
# ---------------------------------------------------------------------------

class TestExecuteGraphWrite:
    def test_calls_ingest(self):
        mock_graph = MagicMock()
        mock_graph.ingest.return_value = {"deleted_entities": [], "added_entities": [{"source": "a"}]}
        engine = _make_engine(graph=mock_graph)

        result = engine._execute_graph_write(
            {"alice": "person"},
            [{"source": "alice", "relationship": "knows", "destination": "bob"}],
            {"user_id": "u1"},
            [],
        )

        mock_graph.ingest.assert_called_once_with(
            entity_type_map={"alice": "person"},
            relations=[{"source": "alice", "relationship": "knows", "destination": "bob"}],
            filters={"user_id": "u1"},
            to_be_deleted=None,
        )
        assert "added_entities" in result

    def test_empty_relations_skips(self):
        mock_graph = MagicMock()
        engine = _make_engine(graph=mock_graph)

        result = engine._execute_graph_write({}, [], {"user_id": "u1"}, [])
        assert result == {}
        mock_graph.ingest.assert_not_called()

    def test_with_deletes(self):
        mock_graph = MagicMock()
        mock_graph.ingest.return_value = {"deleted_entities": [{}], "added_entities": []}
        engine = _make_engine(graph=mock_graph)

        to_be_deleted = [{"source": "a", "relationship": "r", "destination": "b"}]
        result = engine._execute_graph_write(
            {"a": "person"}, [], {"user_id": "u1"}, to_be_deleted
        )

        mock_graph.ingest.assert_called_once_with(
            entity_type_map={"a": "person"},
            relations=[],
            filters={"user_id": "u1"},
            to_be_deleted=to_be_deleted,
        )
        assert "deleted_entities" in result

    def test_graph_error_graceful(self):
        mock_graph = MagicMock()
        mock_graph.ingest.side_effect = RuntimeError("graph error")
        engine = _make_engine(graph=mock_graph)

        result = engine._execute_graph_write(
            {"a": "person"},
            [{"source": "a", "relationship": "r", "destination": "b"}],
            {"user_id": "u1"},
            [],
        )
        assert result == {}


class TestNodeExecuteGraph:
    def test_node_execute_graph(self):
        engine = _make_engine(graph=MagicMock())
        state = _make_state(
            entity_type_map={"a": "person"},
            relations=[{"source": "a", "relationship": "r", "destination": "b"}],
            filters={"user_id": "u1"},
            to_be_deleted=[],
        )

        expected = {"deleted_entities": [], "added_entities": [{"source": "a"}]}
        with patch.object(engine, "_execute_graph_write", return_value=expected) as mock_method:
            result = engine._node_execute_graph(state)
            assert result["graph_result"] == expected


# ---------------------------------------------------------------------------
# Node: assemble_result
# ---------------------------------------------------------------------------

class TestAssembleFinalResult:
    def test_minimal(self):
        engine = _make_engine()
        result = engine._assemble_final_result(
            [], {"results": [], "relations": []}
        )
        assert result == {
            "results": [],
            "recalled_memories": {"results": [], "relations": []},
        }

    def test_with_vector_results(self):
        engine = _make_engine()
        result = engine._assemble_final_result(
            [{"id": "1", "memory": "text", "event": "ADD"}],
            {"results": [{"id": "recalled-1", "memory": "old"}], "relations": []},
        )
        assert len(result["results"]) == 1
        assert len(result["recalled_memories"]["results"]) == 1

    def test_with_graph_result(self):
        engine = _make_engine()
        result = engine._assemble_final_result(
            [{"id": "1", "memory": "text", "event": "ADD"}],
            {"results": [], "relations": []},
            graph_result={"deleted_entities": [], "added_entities": [{"source": "a"}]},
        )
        assert "relations" in result
        assert result["relations"]["added_entities"] == [{"source": "a"}]

    def test_empty_graph_result_omitted(self):
        engine = _make_engine()
        result = engine._assemble_final_result(
            [],
            {"results": [], "relations": []},
            graph_result={"deleted_entities": [], "added_entities": []},
        )
        assert "relations" not in result

    def test_none_graph_result_omitted(self):
        engine = _make_engine()
        result = engine._assemble_final_result(
            [],
            {"results": [], "relations": []},
            graph_result=None,
        )
        assert "relations" not in result


class TestNodeAssembleResult:
    def test_node_assemble_without_graph(self):
        engine = _make_engine(graph=None)
        state = _make_state(
            results=[{"id": "1", "memory": "text", "event": "ADD"}],
            recalled_memories={"results": [], "relations": []},
        )
        result = engine._node_assemble_result(state)
        assert "relations" not in result["final_results"]
        assert result["final_results"]["results"] == state["results"]

    def test_node_assemble_with_graph(self):
        mock_graph = MagicMock()
        engine = _make_engine(graph=mock_graph)
        state = _make_state(
            results=[{"id": "1", "memory": "text", "event": "ADD"}],
            recalled_memories={"results": [], "relations": []},
            graph_result={"deleted_entities": [], "added_entities": [{"source": "x"}]},
        )
        result = engine._node_assemble_result(state)
        assert "relations" in result["final_results"]


# ---------------------------------------------------------------------------
# Vector store helper methods
# ---------------------------------------------------------------------------

class TestCreateMemory:
    def test_create_memory_inserts_with_embeddings(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        mock_vector = MagicMock()
        mock_db = MagicMock()
        engine = _make_engine(embedding_model=mock_embedder, vector_store=mock_vector, db=mock_db)

        memory_id = engine._create_memory("test data", {"test data": [0.1, 0.2]}, {"user_id": "u1"})

        assert memory_id is not None
        mock_vector.insert.assert_called_once()
        insert_args = mock_vector.insert.call_args
        assert insert_args[1]["ids"][0] == memory_id
        assert insert_args[1]["payloads"][0]["data"] == "test data"
        assert insert_args[1]["payloads"][0]["user_id"] == "u1"
        mock_db.add_history.assert_called_once_with(
            memory_id, None, "test data", "ADD",
            created_at=insert_args[1]["payloads"][0]["created_at"],
            actor_id=None, role=None,
        )

    def test_create_memory_embeds_if_not_present(self):
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.5, 0.6]
        engine = _make_engine(embedding_model=mock_embedder)

        engine._create_memory("new data", {}, {"user_id": "u1"})
        mock_embedder.embed.assert_called_once_with("new data", memory_action="add")


class TestUpdateMemory:
    def test_update_memory_success(self):
        mock_vector = MagicMock()
        existing_mock = MagicMock()
        existing_mock.payload = {"data": "old content", "user_id": "u1", "created_at": "2024-01-01T00:00:00+00:00"}
        mock_vector.get.return_value = existing_mock
        mock_db = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.3, 0.4]
        engine = _make_engine(embedding_model=mock_embedder, vector_store=mock_vector, db=mock_db)

        memory_id = engine._update_memory("uuid-1", "new content", {"new content": [0.3, 0.4]}, {"agent_id": "a1"})

        assert memory_id == "uuid-1"
        mock_vector.update.assert_called_once()
        mock_db.add_history.assert_called_once()
        # Verify history event is UPDATE
        assert mock_db.add_history.call_args[0][3] == "UPDATE"

    def test_update_memory_not_found_raises(self):
        mock_vector = MagicMock()
        mock_vector.get.return_value = None
        engine = _make_engine(vector_store=mock_vector)

        with pytest.raises(ValueError, match="not found"):
            engine._update_memory("uuid-1", "content", {}, {})


class TestDeleteMemory:
    def test_delete_memory_success(self):
        mock_vector = MagicMock()
        existing_mock = MagicMock()
        existing_mock.payload = {"data": "to delete", "actor_id": "actor1", "role": "user"}
        mock_vector.get.return_value = existing_mock
        mock_db = MagicMock()
        engine = _make_engine(vector_store=mock_vector, db=mock_db)

        memory_id = engine._delete_memory("uuid-1")

        assert memory_id == "uuid-1"
        mock_vector.delete.assert_called_once_with(vector_id="uuid-1")
        mock_db.add_history.assert_called_once()
        assert mock_db.add_history.call_args[0][3] == "DELETE"

    def test_delete_memory_not_found_raises(self):
        mock_vector = MagicMock()
        mock_vector.get.return_value = None
        engine = _make_engine(vector_store=mock_vector)

        with pytest.raises(ValueError, match="not found"):
            engine._delete_memory("uuid-1")


# ---------------------------------------------------------------------------
# Full add() method (integration of all nodes via LangGraph)
# ---------------------------------------------------------------------------

class TestAddPublicApi:
    def test_add_infer_true_returns_recalled_memories(self):
        """End-to-end test of add() with infer=True, all dependencies mocked."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]

        mock_vector = MagicMock()
        mock_llm = MagicMock()

        # First LLM call: extract queries
        # Second LLM call: decide memory actions
        mock_llm.generate_response.side_effect = [
            '{"queries": ["test query"]}',
            '{"memory": [{"id": null, "text": "Name is John", "event": "ADD"}]}',
        ]

        mock_search = MagicMock()
        mock_search.search.return_value = {
            "results": [{"id": "recalled-1", "memory": "old memory", "score": 0.9}],
            "relations": [],
        }

        mock_db = MagicMock()

        engine = _make_engine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            llm=mock_llm,
            db=mock_db,
            search_engine=mock_search,
            graph=None,
        )

        result = engine.add(
            messages=[{"role": "user", "content": "My name is John"}],
            metadata={"user_id": "u1"},
            filters={"user_id": "u1"},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert result["recalled_memories"]["results"] == [{"id": "recalled-1", "memory": "old memory", "score": 0.9}]

        # verify results contain an ADD
        assert len(result["results"]) >= 1
        add_events = [r for r in result["results"] if r["event"] == "ADD"]
        assert len(add_events) >= 1

    def test_add_infer_false_direct_path(self):
        """End-to-end test of add() with infer=False."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]
        mock_vector = MagicMock()
        mock_db = MagicMock()

        engine = _make_engine(
            embedding_model=mock_embedder,
            vector_store=mock_vector,
            db=mock_db,
        )

        result = engine.add(
            messages=[{"role": "user", "content": "Hello world"}],
            metadata={"user_id": "u1"},
            filters={"user_id": "u1"},
            infer=False,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "Hello world"

    def test_add_with_graph_enabled(self):
        """End-to-end with graph enabled."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2]

        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = [
            '{"queries": ["test"]}',
            '{"memory": [{"id": null, "text": "New", "event": "ADD"}]}',
            json.dumps({
                "entities": [{"entity": "Alice", "entity_type": "person"}],
                "relations": [{"source": "Alice", "relationship": "knows", "destination": "Bob"}],
                "to_be_deleted": [],
            }),
        ]

        mock_search = MagicMock()
        mock_search.search.return_value = {"results": [], "relations": []}

        mock_graph = MagicMock()
        mock_graph.ingest.return_value = {"deleted_entities": [], "added_entities": [{"source": "alice"}]}

        engine = _make_engine(
            embedding_model=mock_embedder,
            llm=mock_llm,
            search_engine=mock_search,
            graph=mock_graph,
        )

        result = engine.add(
            messages=[{"role": "user", "content": "Alice knows Bob"}],
            metadata={"user_id": "u1"},
            filters={"user_id": "u1"},
            infer=True,
        )

        assert "results" in result
        assert "recalled_memories" in result
        assert "relations" in result  # graph was enabled
        mock_graph.ingest.assert_called_once()
