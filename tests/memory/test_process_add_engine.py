import json
from unittest.mock import MagicMock

import pytest

from mem0.memory.process_add_engine import ProcessMemoryAddEngine


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _make_engine(**overrides):
    """Build a ProcessMemoryAddEngine with MagicMock dependencies."""
    defaults = {
        "embedding_model": MagicMock(),
        "vector_store": MagicMock(),
        "llm": MagicMock(),
        "db": MagicMock(),
        "search_engine": MagicMock(),
        "graph_store": MagicMock(),
    }
    defaults.update(overrides)
    return ProcessMemoryAddEngine(**defaults)


def _sample_summaries():
    return [
        {
            "Goal": "Add user auth",
            "Step": "01 - Read main.py",
            "Action": "read_file(path='main.py')",
            "Dependency": [],
            "Brief": "Read main.py to understand entry point",
        },
        {
            "Goal": "Add user auth",
            "Step": "03 - Create auth.py",
            "Action": "create_file(path='auth.py')",
            "Dependency": [
                {"step_id": "01 - Read main.py", "description": "Parse entry logic"}
            ],
            "Brief": "Create auth.py and implement login function",
        },
        {
            "Goal": "Setup database",
            "Step": "02 - Create config.py",
            "Action": "create_file(path='config.py')",
            "Dependency": [],
            "Brief": "Create config.py with DB connection settings",
        },
    ]


# ══════════════════════════════════════════════════════════════════════════
# Constructor tests
# ══════════════════════════════════════════════════════════════════════════


class TestProcessAddEngineInit:
    def test_init_stores_dependencies(self):
        emb = MagicMock()
        vs = MagicMock()
        llm = MagicMock()
        db = MagicMock()
        se = MagicMock()
        gs = MagicMock()

        engine = ProcessMemoryAddEngine(emb, vs, llm, db, se, gs)

        assert engine.embedding_model is emb
        assert engine.vector_store is vs
        assert engine.llm is llm
        assert engine.db is db
        assert engine.search_engine is se
        assert engine.graph_store is gs

    def test_compiles_langgraph(self):
        engine = _make_engine()

        assert engine.add_graph is not None
        # The compiled graph should have invoke method
        assert hasattr(engine.add_graph, "invoke")


# ══════════════════════════════════════════════════════════════════════════
# _node_preprocess tests
# ══════════════════════════════════════════════════════════════════════════


class TestPreprocessNode:
    def test_extracts_goals_deduplicated(self):
        engine = _make_engine()
        summaries = _sample_summaries()

        result = engine._node_preprocess({
            "summaries": summaries, "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        assert result["goals"] == ["Add user auth", "Setup database"]

    def test_extracts_steps_with_fields(self):
        engine = _make_engine()
        summaries = _sample_summaries()

        result = engine._node_preprocess({
            "summaries": summaries, "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        steps = result["steps"]
        assert len(steps) == 3
        assert steps[0]["name"] == "01 - Read main.py"
        assert steps[0]["goal"] == "Add user auth"
        assert steps[1]["name"] == "03 - Create auth.py"
        assert steps[1]["action"] == "create_file(path='auth.py')"

    def test_extracts_dependencies(self):
        engine = _make_engine()
        summaries = _sample_summaries()

        result = engine._node_preprocess({
            "summaries": summaries, "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        deps = result["dependencies"]
        assert len(deps) >= 1
        assert deps[0]["source"] == "01 - Read main.py"
        assert deps[0]["target"] == "03 - Create auth.py"
        assert deps[0]["relationship"] == "DEPENDS_ON"

    def test_builds_entity_type_map(self):
        engine = _make_engine()
        summaries = _sample_summaries()

        result = engine._node_preprocess({
            "summaries": summaries, "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        em = result["entity_type_map"]
        assert em.get("01 - Read main.py") == "Step"
        assert em.get("03 - Create auth.py") == "Step"

    def test_builds_task_description(self):
        engine = _make_engine()
        summaries = _sample_summaries()

        result = engine._node_preprocess({
            "summaries": summaries, "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        assert "Add user auth" in result["task_description"]
        assert "Setup database" in result["task_description"]

    def test_empty_summaries(self):
        engine = _make_engine()

        result = engine._node_preprocess({
            "summaries": [], "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        assert result["goals"] == []
        assert result["steps"] == []
        assert result["dependencies"] == []

    def test_skips_non_dict_summaries(self):
        engine = _make_engine()

        result = engine._node_preprocess({
            "summaries": ["not a dict", None, 123], "metadata": {}, "filters": {},
            "goals": [], "task_description": "", "steps": [], "dependencies": [],
            "entity_type_map": {},
        })

        assert result["goals"] == []
        assert result["steps"] == []


# ══════════════════════════════════════════════════════════════════════════
# _node_search tests
# ══════════════════════════════════════════════════════════════════════════


class TestSearchNode:
    def test_delegates_to_search_engine(self):
        mock_se = MagicMock()
        mock_se.search_for_dedup.return_value = {
            "graph": {"chains": [{"name": "01"}]},
            "chunks": [{"goal": "Add user auth", "id": "c1"}],
            "summaries": [],
        }
        engine = _make_engine(search_engine=mock_se)

        result = engine._node_search({
            "goals": ["Add user auth"], "task_description": "Add user auth",
            "filters": {"user_id": "u1"},
        })

        mock_se.search_for_dedup.assert_called_once_with(
            goals=["Add user auth"],
            task_description="Add user auth",
            filters={"user_id": "u1"},
        )
        assert result["recalled"]["graph"]["chains"] == [{"name": "01"}]
        assert len(result["recalled"]["chunks"]) == 1

    def test_error_returns_empty_recalled(self):
        mock_se = MagicMock()
        mock_se.search_for_dedup.side_effect = RuntimeError("search down")
        engine = _make_engine(search_engine=mock_se)

        result = engine._node_search({
            "goals": ["goal1"], "task_description": "desc",
            "filters": {"user_id": "u1"},
        })

        assert result["recalled"] == {"graph": {"chains": []}, "chunks": [], "summaries": []}


# ══════════════════════════════════════════════════════════════════════════
# _node_decide tests
# ══════════════════════════════════════════════════════════════════════════


class TestDecideNode:
    def _make_engine_for_decide(self, llm_response=None):
        mock_llm = MagicMock()
        if llm_response is not None:
            mock_llm.generate_response.return_value = llm_response
        else:
            mock_llm.generate_response.return_value = json.dumps({
                "graph": {
                    "steps": [{"name": "01 - Read main.py", "event": "ADD", "goal": "Add user auth", "brief": "...", "action": "..."}],
                    "edges": []
                },
                "chunks": [
                    {"goal": "Add user auth", "event": "ADD", "steps": [{"step": "01 - Read main.py", "brief": "..."}]}
                ],
                "summary": {"event": "ADD", "task_description": "Add user auth system", "full_chain": []}
            })
        return _make_engine(llm=mock_llm), mock_llm

    def test_calls_llm_with_context(self):
        engine, mock_llm = self._make_engine_for_decide()

        engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        mock_llm.generate_response.assert_called_once()
        # Verify system prompt was passed
        call_args = mock_llm.generate_response.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "process memory manager" in messages[0]["content"].lower()

    def test_returns_decisions_on_valid_json(self):
        engine, _ = self._make_engine_for_decide()

        result = engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        decisions = result["decisions"]
        assert "graph" in decisions
        assert "chunks" in decisions
        assert "summary" in decisions

    def test_add_forces_null_id(self):
        engine, _ = self._make_engine_for_decide(
            llm_response=json.dumps({
                "graph": {
                    "steps": [{"name": "01", "event": "ADD", "id": "should_be_null", "goal": "g", "brief": "b", "action": "a"}],
                    "edges": []
                },
                "chunks": [{"goal": "g", "event": "ADD", "steps": []}],
                "summary": {"event": "ADD", "task_description": "td", "full_chain": []}
            })
        )

        result = engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        step = result["decisions"]["graph"]["steps"][0]
        assert step["id"] is None

    def test_update_with_invalid_id_falls_back_to_add(self):
        engine, _ = self._make_engine_for_decide(
            llm_response=json.dumps({
                "graph": {
                    "steps": [{"name": "01", "event": "UPDATE", "id": "nonexistent", "goal": "g", "brief": "b", "action": "a"}],
                    "edges": []
                },
                "chunks": [],
                "summary": {}
            })
        )

        result = engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {
                "graph": {"chains": []},  # empty — no valid IDs
                "chunks": [],
                "summaries": [],
            },
        })

        step = result["decisions"]["graph"]["steps"][0]
        assert step["event"] == "ADD"
        assert step["id"] is None

    def test_llm_error_returns_empty_decisions(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.side_effect = RuntimeError("LLM down")
        engine = _make_engine(llm=mock_llm)

        result = engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        assert result["decisions"] == {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}}

    def test_invalid_json_returns_empty_decisions(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "not valid json {{{"
        engine = _make_engine(llm=mock_llm)

        result = engine._node_decide({
            "summaries": _sample_summaries(),
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        assert result["decisions"] == {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}}


# ══════════════════════════════════════════════════════════════════════════
# _node_execute tests
# ══════════════════════════════════════════════════════════════════════════


class TestExecuteNode:
    def test_graph_write_called_with_correct_params(self):
        mock_graph = MagicMock()
        mock_graph.ingest.return_value = {"added_entities": ["01"], "deleted_entities": []}
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1] * 10
        engine = _make_engine(graph_store=mock_graph, embedding_model=mock_emb, vector_store=MagicMock())

        decisions = {
            "graph": {
                "steps": [
                    {"name": "01 - Read main.py", "event": "ADD", "goal": "g", "brief": "read file", "action": "read()"},
                ],
                "edges": [
                    {"source": "01 - Read main.py", "target": "03 - Create auth.py", "relationship": "DEPENDS_ON", "event": "ADD"},
                ]
            },
            "chunks": [],
            "summary": {},
        }
        entity_type_map = {"01 - Read main.py": "Step", "03 - Create auth.py": "Step"}

        state = {
            "decisions": decisions,
            "entity_type_map": entity_type_map,
            "dependencies": [],
            "filters": {"user_id": "u1"},
            "metadata": {"user_id": "u1"},
        }

        engine._node_execute(state)

        mock_graph.ingest.assert_called_once()
        call_kwargs = mock_graph.ingest.call_args.kwargs
        assert "node_properties" in call_kwargs
        assert call_kwargs["node_properties"] is not None
        assert "01 - Read main.py" in call_kwargs["node_properties"]

    def test_chunk_add_inserts_to_vector_store(self):
        import uuid
        mock_vs = MagicMock()
        mock_vs.get.return_value = None
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1] * 10
        mock_db = MagicMock()
        engine = _make_engine(vector_store=mock_vs, embedding_model=mock_emb, db=mock_db)

        decisions = {
            "graph": {"steps": [], "edges": []},
            "chunks": [
                {"goal": "Add user auth", "event": "ADD", "steps": [{"step": "01", "brief": "..."}]},
            ],
            "summary": {},
        }

        state = {
            "decisions": decisions,
            "entity_type_map": {},
            "dependencies": [],
            "filters": {"user_id": "u1"},
            "metadata": {"user_id": "u1"},
        }

        result = engine._node_execute(state)

        mock_vs.insert.assert_called_once()
        mock_db.add_history.assert_called()
        assert len(result["chunk_results"]) == 1
        assert result["chunk_results"][0]["event"] == "ADD"
        assert result["chunk_results"][0]["goal"] == "Add user auth"

    def test_summary_add_writes_to_vector_store(self):
        mock_vs = MagicMock()
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1] * 10
        engine = _make_engine(vector_store=mock_vs, embedding_model=mock_emb)

        decisions = {
            "graph": {"steps": [], "edges": []},
            "chunks": [],
            "summary": {"event": "ADD", "task_description": "Build auth system", "full_chain": []},
        }

        state = {
            "decisions": decisions,
            "entity_type_map": {},
            "dependencies": [],
            "filters": {"user_id": "u1"},
            "metadata": {"user_id": "u1"},
        }

        result = engine._node_execute(state)

        mock_vs.insert.assert_called()
        assert result["summary_result"]["event"] == "ADD"

    def test_summary_empty_no_write(self):
        mock_vs = MagicMock()
        engine = _make_engine(vector_store=mock_vs)

        decisions = {
            "graph": {"steps": [], "edges": []},
            "chunks": [],
            "summary": {},
        }

        state = {
            "decisions": decisions,
            "entity_type_map": {},
            "dependencies": [],
            "filters": {"user_id": "u1"},
            "metadata": {"user_id": "u1"},
        }

        result = engine._node_execute(state)

        # No vector store write for empty summary
        assert result["summary_result"] == {}


# ══════════════════════════════════════════════════════════════════════════
# _node_assemble tests
# ══════════════════════════════════════════════════════════════════════════


class TestAssembleNode:
    def test_assembles_final_result_structure(self):
        engine = _make_engine()

        result = engine._node_assemble({
            "graph_result": {"added_entities": ["s1"]},
            "chunk_results": [{"id": "c1", "goal": "g", "event": "ADD"}],
            "summary_result": {"id": "s1", "event": "ADD", "task_description": "td"},
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
        })

        final = result["final_results"]
        assert "results" in final
        assert "recalled" in final
        assert final["results"]["graph"] == {"added_entities": ["s1"]}
        assert final["results"]["chunks"][0]["event"] == "ADD"
        assert final["results"]["summary"]["event"] == "ADD"


# ══════════════════════════════════════════════════════════════════════════
# add() method tests
# ══════════════════════════════════════════════════════════════════════════


class TestAddMethod:
    def test_add_invokes_langgraph(self):
        engine = _make_engine()
        # Mock the compiled graph
        engine.add_graph = MagicMock()
        engine.add_graph.invoke.return_value = {
            "final_results": {
                "results": {"graph": {}, "chunks": [], "summary": {}},
                "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
            },
            "error": None,
        }

        result = engine.add(
            summaries=_sample_summaries(),
            metadata={"user_id": "u1"},
            filters={"user_id": "u1"},
        )

        engine.add_graph.invoke.assert_called_once()
        assert "results" in result
        assert "recalled" in result

    def test_add_propagates_error(self):
        engine = _make_engine()
        engine.add_graph = MagicMock()
        engine.add_graph.invoke.return_value = {
            "final_results": {},
            "error": "Something went wrong",
        }

        result = engine.add(
            summaries=[],
            metadata={},
            filters={"user_id": "u1"},
        )

        assert result == {}


# ══════════════════════════════════════════════════════════════════════════
# Helper function tests
# ══════════════════════════════════════════════════════════════════════════


class TestMergeSteps:
    def test_merges_deduplicating_by_step_name(self):
        from mem0.memory.process_add_engine import _merge_steps

        existing = [{"step": "01", "brief": "old"}, {"step": "02", "brief": "old2"}]
        incoming = [{"step": "01", "brief": "new"}, {"step": "03", "brief": "new3"}]

        merged = _merge_steps(existing, incoming)

        assert len(merged) == 3
        by_name = {s["step"]: s["brief"] for s in merged}
        assert by_name["01"] == "new"  # incoming wins
        assert by_name["02"] == "old2"
        assert by_name["03"] == "new3"

    def test_handles_empty_lists(self):
        from mem0.memory.process_add_engine import _merge_steps

        assert _merge_steps([], []) == []
        assert len(_merge_steps([], [{"step": "01", "brief": "b"}])) == 1
        assert len(_merge_steps([{"step": "01", "brief": "b"}], [])) == 1

    def test_skips_non_dict_entries(self):
        from mem0.memory.process_add_engine import _merge_steps

        existing = [{"step": "01", "brief": "ok"}]
        incoming = ["string", None, 123, {"step": "02", "brief": "new"}]

        merged = _merge_steps(existing, incoming)

        assert len(merged) == 2


class TestNormalizeTimestamp:
    def test_utc_timestamp_returned_unchanged(self):
        from mem0.memory.process_add_engine import _normalize_iso_timestamp_to_utc

        ts = "2024-01-15T10:30:00+00:00"
        result = _normalize_iso_timestamp_to_utc(ts)
        assert "+00:00" in result or "Z" in result

    def test_none_returns_none(self):
        from mem0.memory.process_add_engine import _normalize_iso_timestamp_to_utc

        assert _normalize_iso_timestamp_to_utc(None) is None

    def test_empty_string_returns_empty(self):
        from mem0.memory.process_add_engine import _normalize_iso_timestamp_to_utc

        assert _normalize_iso_timestamp_to_utc("") == ""
