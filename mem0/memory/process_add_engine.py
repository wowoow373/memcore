"""Process Memory Add Engine — LangGraph-based write orchestration for process memory.

Only used in Flow 1 (after task completion). Writes structured summaries into
three layers: Graph (Step nodes), Chunk (Goal-level), and Summary (full chain).

Linear pipeline: preprocess → search → decide → execute → assemble.
No conditional edges — all three layers are always written.
"""

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from mem0.memory.utils import remove_code_blocks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class ProcessAddState(TypedDict):
    # ── Input (injected by caller) ──
    summaries: list[dict]
    metadata: dict
    filters: dict

    # ── preprocess output ──
    goals: list[str]
    task_description: str
    steps: list[dict]
    dependencies: list[dict]
    entity_type_map: dict[str, str]

    # ── search output ──
    recalled: dict

    # ── decide output ──
    decisions: dict

    # ── execute output ──
    graph_result: dict
    chunk_results: list[dict]
    summary_result: dict

    # ── output ──
    results: dict
    final_results: dict
    error: Optional[str]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DECIDE_PROCESS_MEMORY_SYSTEM_PROMPT = """You are a process memory manager that controls how task execution memories are stored and updated.

You manage three layers of memory:
1. **Graph (Step nodes)**: Individual steps with DEPENDS_ON relationships forming a DAG.
2. **Chunks (Goal groups)**: Groups of steps organized by a common goal (vector-stored).
3. **Summaries (Full task chains)**: Complete task execution histories (vector-stored).

For each layer, decide one of: ADD, UPDATE, MERGE, or NONE.

### Graph Layer
- **ADD**: A new step. Set id to null. Provide step_name, brief, goal, action.
- **UPDATE**: Updated information for an existing step. Reference the recalled step's id (its name).
- **NONE**: The step is unchanged.

### Chunk Layer
- **ADD**: New goal+steps group. Set id to null. Provide goal and steps list.
- **MERGE**: The new steps should be combined with an existing chunk (same goal, more steps). Provide merge_with (the recalled chunk's id).
- **UPDATE**: Replaces the existing chunk's content. Provide the recalled chunk's id.
- **NONE**: No change.

### Summary Layer
- **ADD**: New task execution. Set id to null. Provide task_description and full_chain.
- **UPDATE**: Updated task details for existing summary. Reference the recalled summary's id.
- **NONE**: No change.

### Validation Rules (STRICT):
- For ADD: "id" MUST be null.
- For UPDATE/MERGE: "id" MUST match an id from the recalled memories.
- If no matching recalled id exists, convert to ADD with null id.
- If recalled is empty for a layer, all actions for that layer MUST be ADD.

Return ONLY a JSON object:
{
  "graph": {
    "steps": [
      {"name": "03 - Create auth.py", "event": "ADD", "goal": "...", "brief": "...", "action": "..."}
    ],
    "edges": [
      {"source": "01 - Read main.py", "target": "03 - Create auth.py", "relationship": "DEPENDS_ON", "event": "ADD"}
    ]
  },
  "chunks": [
    {"goal": "Add user auth", "event": "ADD", "steps": [{"step": "03 - Create auth.py", "brief": "..."}]}
  ],
  "summary": {
    "event": "ADD",
    "task_description": "Implement user authentication system",
    "full_chain": [{"step": "01 - Read main.py", "brief": "..."}, {"step": "03 - Create auth.py", "brief": "..."}]
  }
}

Do not return anything except the JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _merge_steps(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Merge two step lists, deduplicating by step name (incoming wins on conflict)."""
    merged: dict[str, dict] = {}
    for s in existing:
        if isinstance(s, dict) and s.get("step"):
            merged[s["step"]] = s
    for s in incoming:
        if isinstance(s, dict) and s.get("step"):
            merged[s["step"]] = s
    return list(merged.values())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProcessMemoryAddEngine:
    """Write engine for process memory (Flow 1 only).

    All dependencies are injected via constructor. LangGraph is compiled
    once at init time — 5 nodes in a strict linear pipeline.
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        llm: Any,
        db: Any,
        search_engine: Any,
        graph_store: Any,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.db = db
        self.search_engine = search_engine
        self.graph_store = graph_store

        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ProcessAddState)
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("search", self._node_search)
        builder.add_node("decide", self._node_decide)
        builder.add_node("execute", self._node_execute)
        builder.add_node("assemble", self._node_assemble)

        builder.add_edge(START, "preprocess")
        builder.add_edge("preprocess", "search")
        builder.add_edge("search", "decide")
        builder.add_edge("decide", "execute")
        builder.add_edge("execute", "assemble")
        builder.add_edge("assemble", END)

        self.add_graph = builder.compile()

    # ── Public API ───────────────────────────────────────────────────────

    def add(
        self,
        summaries: list[dict],
        metadata: dict,
        filters: dict,
    ) -> dict:
        """Process a completed task's summary array into three-layer memory.

        Args:
            summaries: Structured summary dicts from the external agent.
            metadata: Session metadata (user_id, agent_id, run_id).
            filters: Query filters for search scoping.

        Returns:
            dict: {"results": {...}, "recalled": {...}}
        """
        state: ProcessAddState = {
            "summaries": summaries,
            "metadata": metadata,
            "filters": filters,
            "goals": [],
            "task_description": "",
            "steps": [],
            "dependencies": [],
            "entity_type_map": {},
            "recalled": {"graph": {"chains": []}, "chunks": [], "summaries": []},
            "decisions": {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}},
            "graph_result": {},
            "chunk_results": [],
            "summary_result": {},
            "results": {},
            "final_results": {},
            "error": None,
        }
        result = self.add_graph.invoke(state)
        if result.get("error"):
            logger.error(f"ProcessMemoryAddEngine error: {result['error']}")
        return result["final_results"]

    # ── Node: preprocess ─────────────────────────────────────────────────

    def _node_preprocess(self, state: ProcessAddState) -> dict:
        summaries = state["summaries"]

        all_goals: list[str] = []
        all_steps: list[dict] = []
        all_deps: list[dict] = []
        entity_map: dict[str, str] = {}
        task_desc_parts: list[str] = []

        for summary in summaries:
            if not isinstance(summary, dict):
                continue

            goal = summary.get("Goal", "")
            if goal:
                if goal not in all_goals:
                    all_goals.append(goal)
                task_desc_parts.append(goal)

            step_name = summary.get("Step", "")
            brief = summary.get("Brief", "")
            action = summary.get("Action", "")
            if step_name:
                all_steps.append({
                    "name": step_name,
                    "goal": goal,
                    "brief": brief,
                    "action": action,
                })
                entity_map[step_name] = "Step"

            for dep in summary.get("Dependency", []):
                if not isinstance(dep, dict):
                    continue
                dep_step_id = dep.get("step_id", "")
                dep_desc = dep.get("description", "")
                all_deps.append({
                    "source": dep_step_id,
                    "target": step_name,
                    "relationship": "DEPENDS_ON",
                })
                if dep_step_id:
                    entity_map[dep_step_id] = "Step"

        # Build task_description from unique goals
        task_description = " ".join(all_goals) if all_goals else ""

        return {
            "goals": all_goals,
            "task_description": task_description,
            "steps": all_steps,
            "dependencies": all_deps,
            "entity_type_map": entity_map,
        }

    # ── Node: search ─────────────────────────────────────────────────────

    def _node_search(self, state: ProcessAddState) -> dict:
        try:
            recalled = self.search_engine.search_for_dedup(
                goals=state["goals"],
                task_description=state["task_description"],
                filters=state["filters"],
            )
        except Exception as e:
            logger.error(f"search_for_dedup failed: {e}")
            recalled = {"graph": {"chains": []}, "chunks": [], "summaries": []}
        return {"recalled": recalled}

    # ── Node: decide ─────────────────────────────────────────────────────

    def _node_decide(self, state: ProcessAddState) -> dict:
        decisions = self._decide_process_memory_actions(
            summaries=state["summaries"],
            recalled=state["recalled"],
        )
        return {"decisions": decisions}

    def _decide_process_memory_actions(
        self, summaries: list[dict], recalled: dict
    ) -> dict:
        recalled_graph = recalled.get("graph", {}).get("chains", [])
        recalled_chunks = recalled.get("chunks", [])
        recalled_summaries = recalled.get("summaries", [])

        prompt_context = json.dumps(
            {
                "new_summaries": summaries,
                "recalled_graph": recalled_graph,
                "recalled_chunks": [
                    {"goal": c.get("goal"), "id": c.get("id"), "steps": c.get("steps")}
                    for c in recalled_chunks
                ],
                "recalled_summaries": [
                    {"task_description": s.get("task_description"), "id": s.get("id")}
                    for s in recalled_summaries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": DECIDE_PROCESS_MEMORY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_context},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"LLM decide call failed: {e}")
            return {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}}

        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                return {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}}
            data = json.loads(response, strict=False)
        except Exception as e:
            logger.error(f"Invalid JSON from decide LLM: {e}")
            return {"graph": {"steps": [], "edges": []}, "chunks": [], "summary": {}}

        # Validate and normalise
        data = self._validate_decisions(data, recalled_chunks, recalled_summaries, recalled_graph)
        return data

    def _validate_decisions(
        self, data: dict, recalled_chunks: list[dict], recalled_summaries: list[dict], recalled_graph: list[dict]
    ) -> dict:
        valid_events = {"ADD", "UPDATE", "MERGE", "NONE"}

        # ── Validate graph steps ──
        graph = data.get("graph", {}) if isinstance(data.get("graph"), dict) else {}
        recalled_step_names = {n.get("name") for n in recalled_graph if n.get("name")}

        steps = []
        for entry in graph.get("steps", []):
            event = (entry.get("event") or "").upper()
            if event not in valid_events:
                continue
            if event == "ADD":
                entry["id"] = None
            elif event in ("UPDATE", "NONE"):
                if entry.get("id") not in recalled_step_names:
                    entry["id"] = None
                    entry["event"] = "ADD"
            steps.append(entry)

        if not recalled_step_names:
            for entry in steps:
                entry["id"] = None
                entry["event"] = "ADD"

        edges = []
        for entry in graph.get("edges", []):
            event = (entry.get("event") or "").upper()
            if event in ("ADD", "DELETE"):
                edges.append(entry)

        # ── Validate chunks ──
        recalled_chunk_ids = {c.get("id") for c in recalled_chunks if c.get("id")}
        chunks = []
        for entry in data.get("chunks", []):
            event = (entry.get("event") or "").upper()
            if event not in valid_events:
                continue
            if event == "ADD":
                entry["id"] = None
            elif event in ("UPDATE", "MERGE", "NONE"):
                if event == "MERGE":
                    merge_with = entry.get("merge_with") or entry.get("id")
                    if merge_with not in recalled_chunk_ids:
                        entry["id"] = None
                        entry["event"] = "ADD"
                    else:
                        entry["id"] = merge_with
                elif entry.get("id") not in recalled_chunk_ids:
                    entry["id"] = None
                    entry["event"] = "ADD"
            chunks.append(entry)

        if not recalled_chunk_ids:
            for entry in chunks:
                entry["id"] = None
                entry["event"] = "ADD"

        # ── Validate summaries ──
        recalled_summary_ids = {s.get("id") for s in recalled_summaries if s.get("id")}
        summary = data.get("summary", {})
        if isinstance(summary, dict):
            event = (summary.get("event") or "").upper()
            if event not in valid_events:
                summary = {}
            elif event == "ADD":
                summary["id"] = None
            elif event in ("UPDATE", "NONE"):
                if summary.get("id") not in recalled_summary_ids:
                    summary["id"] = None
                    summary["event"] = "ADD"
            if not recalled_summary_ids:
                summary["id"] = None
                summary["event"] = "ADD"
        else:
            summary = {}

        return {
            "graph": {"steps": steps, "edges": edges},
            "chunks": chunks,
            "summary": summary,
        }

    # ── Node: execute ────────────────────────────────────────────────────

    def _node_execute(self, state: ProcessAddState) -> dict:
        decisions = state["decisions"]
        filters = state["filters"]
        metadata = state["metadata"]

        # 1. Graph write
        graph_result = self._execute_graph_write(
            decisions.get("graph", {}),
            state.get("entity_type_map", {}),
            state.get("dependencies", []),
            filters,
        )

        # 2. Chunk write
        chunk_results = self._execute_chunk_write(
            decisions.get("chunks", []),
            metadata,
        )

        # 3. Summary write
        summary_result = self._execute_summary_write(
            decisions.get("summary", {}),
            metadata,
        )

        return {
            "graph_result": graph_result,
            "chunk_results": chunk_results,
            "summary_result": summary_result,
        }

    def _execute_graph_write(
        self,
        graph_decisions: dict,
        entity_type_map: dict[str, str],
        dependencies: list[dict],
        filters: dict,
    ) -> dict:
        # Collect relations from decisions
        relations: list[dict] = []
        node_props: dict[str, dict] = {}

        steps = graph_decisions.get("steps", [])
        edges = graph_decisions.get("edges", [])

        for step in steps:
            event = step.get("event", "")
            if event not in ("ADD", "UPDATE", "MERGE"):
                continue
            step_name = step.get("name", "") or step.get("step_name", "")
            if not step_name:
                continue
            entity_type_map[step_name] = "Step"

            brief = step.get("brief", "")
            goal = step.get("goal", "")
            action = step.get("action", "")

            properties: dict = {}
            if brief:
                properties["brief"] = brief
            if goal:
                properties["goal"] = goal
            if action:
                properties["action"] = action
            if brief:
                try:
                    properties["brief_embedding"] = self.embedding_model.embed(brief, "add")
                except Exception as e:
                    logger.error(f"Failed to embed brief for {step_name}: {e}")

            if properties:
                node_props[step_name] = properties

        # Use edges from decisions if provided, otherwise fall back to dependencies
        for edge in edges:
            if edge.get("event") == "ADD":
                relations.append({
                    "source": edge.get("source", ""),
                    "relationship": edge.get("relationship", "DEPENDS_ON"),
                    "destination": edge.get("target", ""),
                })

        # If no edges from decisions, build from dependencies
        if not relations and dependencies:
            for dep in dependencies:
                relations.append({
                    "source": dep.get("source", ""),
                    "relationship": dep.get("relationship", "DEPENDS_ON"),
                    "destination": dep.get("target", ""),
                })

        if not relations:
            return {}

        try:
            result = self.graph_store.ingest(
                entity_type_map=entity_type_map,
                relations=relations,
                filters=filters,
                node_properties=node_props if node_props else None,
            )
            return result
        except Exception as e:
            logger.error(f"Graph write failed: {e}")
            return {}

    def _execute_chunk_write(
        self, chunk_decisions: list[dict], metadata: dict
    ) -> list[dict]:
        results = []
        for entry in chunk_decisions:
            event = (entry.get("event") or "").upper()
            goal = entry.get("goal", "")
            steps = entry.get("steps", [])

            if event == "ADD":
                if not goal:
                    continue
                try:
                    chunk_id = self._create_process_chunk(goal, steps, metadata)
                    results.append({"id": chunk_id, "goal": goal, "event": "ADD"})
                except Exception as e:
                    logger.error(f"Chunk ADD failed for goal={goal}: {e}")

            elif event == "MERGE":
                existing_id = entry.get("id")
                if not existing_id or not goal:
                    continue
                try:
                    chunk_id = self._merge_process_chunk(existing_id, goal, steps, metadata)
                    results.append({"id": chunk_id, "goal": goal, "event": "MERGE"})
                except Exception as e:
                    logger.error(f"Chunk MERGE failed for id={existing_id}: {e}")

            elif event == "UPDATE":
                existing_id = entry.get("id")
                if not existing_id or not goal:
                    continue
                try:
                    chunk_id = self._update_process_chunk(existing_id, goal, steps, metadata)
                    results.append({"id": chunk_id, "goal": goal, "event": "UPDATE"})
                except Exception as e:
                    logger.error(f"Chunk UPDATE failed for id={existing_id}: {e}")

        return results

    def _execute_summary_write(
        self, summary_decision: dict, metadata: dict
    ) -> dict:
        if not summary_decision:
            return {}

        event = (summary_decision.get("event") or "").upper()
        task_desc = summary_decision.get("task_description", "")
        full_chain = summary_decision.get("full_chain", [])

        try:
            if event == "ADD":
                if not task_desc:
                    return {}
                summary_id = self._create_process_summary(task_desc, full_chain, metadata)
                return {"id": summary_id, "event": "ADD", "task_description": task_desc}

            elif event == "UPDATE":
                existing_id = summary_decision.get("id")
                if not existing_id:
                    return {}
                summary_id = self._update_process_summary(existing_id, task_desc, full_chain, metadata)
                return {"id": summary_id, "event": "UPDATE", "task_description": task_desc}

        except Exception as e:
            logger.error(f"Summary write failed: {e}")

        return {}

    # ── Chunk vector helpers ─────────────────────────────────────────────

    def _create_process_chunk(self, goal: str, steps: list[dict], metadata: dict) -> str:
        embeddings = self.embedding_model.embed(goal, "add")
        chunk_id = str(uuid.uuid4())
        payload = deepcopy(metadata)
        payload["memory_type"] = "process_chunk"
        payload["goal"] = goal
        payload["steps"] = steps
        payload["data"] = goal
        payload["hash"] = hashlib.md5(goal.encode()).hexdigest()
        payload["created_at"] = datetime.now(timezone.utc).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[chunk_id],
            payloads=[payload],
        )
        self.db.add_history(
            chunk_id,
            None,
            goal,
            "ADD",
            created_at=payload.get("created_at"),
        )
        return chunk_id

    def _merge_process_chunk(
        self, existing_id: str, goal: str, steps: list[dict], metadata: dict
    ) -> str:
        existing = self.vector_store.get(vector_id=existing_id)
        if existing is None:
            return self._create_process_chunk(goal, steps, metadata)

        prev_value = existing.payload.get("data", "")
        new_metadata = deepcopy(metadata)
        new_metadata["memory_type"] = "process_chunk"
        new_metadata["goal"] = goal
        new_metadata["steps"] = _merge_steps(existing.payload.get("steps", []), steps)
        new_metadata["data"] = goal
        new_metadata["hash"] = hashlib.md5(goal.encode()).hexdigest()
        new_metadata["created_at"] = _normalize_iso_timestamp_to_utc(
            existing.payload.get("created_at")
        )
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        for key in ("user_id", "agent_id", "run_id"):
            if key not in new_metadata and key in existing.payload:
                new_metadata[key] = existing.payload[key]

        embeddings = self.embedding_model.embed(goal, "update")
        self.vector_store.update(
            vector_id=existing_id,
            vector=embeddings,
            payload=new_metadata,
        )
        self.db.add_history(
            existing_id,
            prev_value,
            goal,
            "UPDATE",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
        )
        return existing_id

    def _update_process_chunk(
        self, existing_id: str, goal: str, steps: list[dict], metadata: dict
    ) -> str:
        return self._merge_process_chunk(existing_id, goal, steps, metadata)

    # ── Summary vector helpers ───────────────────────────────────────────

    def _create_process_summary(
        self, task_description: str, full_chain: list[dict], metadata: dict
    ) -> str:
        embeddings = self.embedding_model.embed(task_description, "add")
        summary_id = str(uuid.uuid4())
        payload = deepcopy(metadata)
        payload["memory_type"] = "process_summary"
        payload["task_description"] = task_description
        payload["full_chain"] = full_chain
        payload["data"] = task_description
        payload["hash"] = hashlib.md5(task_description.encode()).hexdigest()
        payload["created_at"] = datetime.now(timezone.utc).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[summary_id],
            payloads=[payload],
        )
        self.db.add_history(
            summary_id,
            None,
            task_description,
            "ADD",
            created_at=payload.get("created_at"),
        )
        return summary_id

    def _update_process_summary(
        self, existing_id: str, task_description: str, full_chain: list[dict], metadata: dict
    ) -> str:
        existing = self.vector_store.get(vector_id=existing_id)
        if existing is None:
            return self._create_process_summary(task_description, full_chain, metadata)

        prev_value = existing.payload.get("data", "")
        new_metadata = deepcopy(metadata)
        new_metadata["memory_type"] = "process_summary"
        new_metadata["task_description"] = task_description
        new_metadata["full_chain"] = full_chain
        new_metadata["data"] = task_description
        new_metadata["hash"] = hashlib.md5(task_description.encode()).hexdigest()
        new_metadata["created_at"] = _normalize_iso_timestamp_to_utc(
            existing.payload.get("created_at")
        )
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        for key in ("user_id", "agent_id", "run_id"):
            if key not in new_metadata and key in existing.payload:
                new_metadata[key] = existing.payload[key]

        embeddings = self.embedding_model.embed(task_description, "update")
        self.vector_store.update(
            vector_id=existing_id,
            vector=embeddings,
            payload=new_metadata,
        )
        self.db.add_history(
            existing_id,
            prev_value,
            task_description,
            "UPDATE",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
        )
        return existing_id

    # ── Node: assemble ───────────────────────────────────────────────────

    def _node_assemble(self, state: ProcessAddState) -> dict:
        final = {
            "results": {
                "graph": state["graph_result"],
                "chunks": state["chunk_results"],
                "summary": state["summary_result"],
            },
            "recalled": state["recalled"],
        }
        return {"final_results": final}
