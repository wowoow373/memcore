"""Add Engine — LangGraph-based write orchestration for Memory.add().

Decouples the add flow into independently testable nodes:
  preprocess → [extract_queries → search → decide_memory → execute_vector]
  → [extract_graph → execute_graph] → assemble_result

When infer=False, takes the fast path: preprocess → direct_add → assemble_result.
"""

import hashlib
import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from mem0.memory.utils import (
    extract_json,
    remove_code_blocks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AddState(TypedDict):
    # ── Input ──
    messages: list[dict]
    metadata: dict
    filters: dict
    infer: bool

    # ── Intermediate ──
    parsed_messages: str
    search_queries: list[str]
    recalled_memories: dict          # {"results": [MemoryItem], "relations": [...]}
    decisions: list[dict]            # [{"id": str|null, "text": str, "event": str, "old_memory": str}]
    entity_type_map: dict            # {"entity": "type"}
    relations: list[dict]            # [{"source", "relationship", "destination"}]
    to_be_deleted: list[dict]        # same shape as relations
    graph_result: dict               # graph.ingest() return value

    # ── Output ──
    results: list[dict]
    final_results: dict
    error: Optional[str]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXTRACT_QUERIES_PROMPT = """You are a memory retrieval assistant. Given a conversation, generate queries for searching existing memories.
- Extract all key topics, entities, and personal information from the conversation.
- Each query should be self-contained and semantically complete.
- If the conversation contains very little information or no extractable content, return an empty list.
- Return JSON format: {"queries": ["query1", "query2"]}"""


DECIDE_MEMORY_SYSTEM_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Analyze the conversation text to identify factual information about the user/agent, then compare against existing memories.

There are specific guidelines to select which operation to perform:

1. **Add**: If the conversation contains new factual information not present in existing memories, add it with a null id.
- **Example**:
    - Existing Memory (empty)
    - Conversation: "My name is John"
    - Result: {"memory": [{"id": null, "text": "Name is John", "event": "ADD"}]}

2. **Update**: If the conversation contains information that is related to an existing memory but provides more detail or updated information, update the existing memory.
Example (a) -- if the existing memory is "User likes to play cricket" and the conversation reveals "Loves to play cricket with friends", then update the memory.
Example (b) -- if the existing memory is "Likes cheese pizza" and the conversation reveals "Loves cheese pizza", then you do not need to update it because they convey the same information.
When updating, use the existing memory's ID and provide the old memory content in old_memory field.

3. **Delete**: If the conversation contradicts existing memories, delete the existing memory.
Example -- existing memory "Loves cheese pizza", conversation reveals "Dislikes cheese pizza" → DELETE.

4. **No Change**: If the conversation does not contain new, updated, or contradictory information relative to existing memories, mark as NONE.

Return ONLY a JSON object with the following structure:
{
    "memory": [
        {
            "id": "<existing-id-or-null>",
            "text": "<content>",
            "event": "<ADD|UPDATE|DELETE|NONE>",
            "old_memory": "<previous content, only for UPDATE>"
        }
    ]
}

Important:
- For ADD events, set "id" to null (a new UUID will be assigned automatically).
- For UPDATE/DELETE/NONE events, use the exact ID from the existing memories provided.
- Only include entries that require action. If nothing needs to change, return {"memory": []}.
- Do not return anything except the JSON."""


EXTRACT_GRAPH_SYSTEM_PROMPT = """You are a graph memory manager. Extract entities, relationships, and deletion decisions from conversations.

Return a JSON object with three fields:
{{
  "entities": [{{"entity": "EntityName", "entity_type": "type"}}],
  "relations": [{{"source": "EntityA", "relationship": "rel_type", "destination": "EntityB"}}],
  "to_be_deleted": [{{"source": "EntityA", "relationship": "rel_type", "destination": "EntityB"}}]
}}

Guidelines:
- Only extract explicitly stated entities and relationships.
- Use "{user_id}" as the source entity for self-references ("I", "me", "my").
- Use consistent, timeless relationship types (e.g., "lives_in" not "became_lives_in").
- Do NOT delete relationships just because a new one of the same type exists with a different target.
- Only DELETE relationships that are directly contradicted by the new conversation.
- If there are no entities to extract, return empty arrays.
- If there are no deletions needed, set to_be_deleted to []."""


# ---------------------------------------------------------------------------
# Add Engine
# ---------------------------------------------------------------------------

class AddEngine:
    """LangGraph-based write orchestration engine for Memory.add().

    Composes the add flow into independently testable nodes.
    Internally builds a LangGraph StateGraph; exposed via a simple ``add()`` method.
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        llm: Any,
        db: Any,                          # SQLiteManager for history
        search_engine: Any,               # SearchEngine instance
        graph: Optional[Any] = None,      # Graph store (optional)
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.db = db
        self.search_engine = search_engine
        self.graph = graph
        self.enable_graph = graph is not None

        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AddState)

        # Register nodes
        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("direct_add", self._node_direct_add)
        builder.add_node("extract_queries", self._node_extract_queries)
        builder.add_node("search", self._node_search)
        builder.add_node("decide_memory", self._node_decide_memory)
        builder.add_node("execute_vector", self._node_execute_vector)
        builder.add_node("extract_graph", self._node_extract_graph)
        builder.add_node("execute_graph", self._node_execute_graph)
        builder.add_node("assemble_result", self._node_assemble_result)

        # Build edges
        builder.add_edge(START, "preprocess")

        # Conditional: preprocess → direct_add (infer=False) or extract_queries (infer=True)
        builder.add_conditional_edges(
            "preprocess",
            self._should_infer,
            {"direct_add": "direct_add", "extract_queries": "extract_queries"},
        )

        # direct_add → assemble_result
        builder.add_edge("direct_add", "assemble_result")

        # infer=True path
        builder.add_edge("extract_queries", "search")
        builder.add_edge("search", "decide_memory")
        builder.add_edge("decide_memory", "execute_vector")

        # Conditional: execute_vector → extract_graph (if graph enabled) or assemble_result
        builder.add_conditional_edges(
            "execute_vector",
            self._should_extract_graph,
            {"extract_graph": "extract_graph", "assemble_result": "assemble_result"},
        )

        builder.add_edge("extract_graph", "execute_graph")
        builder.add_edge("execute_graph", "assemble_result")
        builder.add_edge("assemble_result", END)

        self.add_graph = builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        messages: list[dict],
        metadata: dict,
        filters: dict,
        infer: bool = True,
    ) -> dict:
        """Run the full add flow.

        Args:
            messages: Normalized list of message dicts (already validated upstream).
            metadata: Storage metadata (includes user_id/agent_id/run_id).
            filters: Query filters for search scoping.
            infer: Whether to use LLM for fact extraction & decision.

        Returns:
            dict with keys ``results``, ``recalled_memories``, and optionally ``relations``.
        """
        state: AddState = {
            "messages": messages,
            "metadata": metadata,
            "filters": filters,
            "infer": infer,
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
        result = self.add_graph.invoke(state)
        return result["final_results"]

    # ------------------------------------------------------------------
    # Conditional edge logic
    # ------------------------------------------------------------------

    def _should_infer(self, state: AddState) -> str:
        if state.get("infer"):
            return "extract_queries"
        return "direct_add"

    def _should_extract_graph(self, state: AddState) -> str:
        if self.enable_graph:
            return "extract_graph"
        return "assemble_result"

    # ------------------------------------------------------------------
    # Node: preprocess
    # ------------------------------------------------------------------

    def _preprocess_messages(self, messages: list[dict]) -> str:
        """Concatenate user/assistant messages into a single parsed string."""
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role in ("user", "assistant"):
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _node_preprocess(self, state: AddState) -> dict:
        parsed = self._preprocess_messages(state["messages"])
        return {"parsed_messages": parsed}

    # ------------------------------------------------------------------
    # Node: direct_add (infer=False fast path)
    # ------------------------------------------------------------------

    def _direct_add_messages(
        self,
        messages: list[dict],
        metadata: dict,
    ) -> list[dict]:
        """Embed each message and insert into vector store directly. No LLM, no graph."""
        results = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") is None or msg.get("content") is None:
                logger.warning(f"Skipping invalid message format: {msg}")
                continue

            if msg["role"] == "system":
                continue

            per_msg_meta = deepcopy(metadata)
            per_msg_meta["role"] = msg["role"]

            actor_name = msg.get("name")
            if actor_name:
                per_msg_meta["actor_id"] = actor_name

            content = msg["content"]
            if not content or not content.strip():
                logger.warning("Skipping message with empty content")
                continue

            embeddings = self.embedding_model.embed(content, "add")
            memory_id = self._create_memory(content, {content: embeddings}, per_msg_meta)

            results.append({
                "id": memory_id,
                "memory": content,
                "event": "ADD",
                "actor_id": actor_name if actor_name else None,
                "role": msg["role"],
            })
        return results

    def _node_direct_add(self, state: AddState) -> dict:
        results = self._direct_add_messages(state["messages"], state["metadata"])
        return {"results": results}

    # ------------------------------------------------------------------
    # Node: extract_queries
    # ------------------------------------------------------------------

    def _extract_search_queries(self, parsed_messages: str) -> list[str]:
        """Use LLM to extract search queries from conversation text."""
        if not parsed_messages or not parsed_messages.strip():
            return [parsed_messages] if parsed_messages else []

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": EXTRACT_QUERIES_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{parsed_messages}"},
                ],
                response_format={"type": "json_object"},
            )

            response = remove_code_blocks(response)
            if not response or not response.strip():
                return [parsed_messages]

            data = json.loads(response, strict=False)
            queries = data.get("queries", [])
            if not queries:
                return [parsed_messages]

            return [q for q in queries if q and q.strip()]
        except Exception as e:
            logger.error(f"Error extracting search queries: {e}")
            return [parsed_messages]

    def _node_extract_queries(self, state: AddState) -> dict:
        queries = self._extract_search_queries(state["parsed_messages"])
        return {"search_queries": queries}

    # ------------------------------------------------------------------
    # Node: search
    # ------------------------------------------------------------------

    def _search_memories(
        self,
        search_queries: list[str],
        filters: dict,
    ) -> dict:
        """Call SearchEngine for each query and merge results."""
        all_vector_results = []
        all_relations = []

        for query in search_queries:
            try:
                result = self.search_engine.search(
                    query=query,
                    filters=filters,
                    limit=10,
                    threshold=None,
                    graph_depth=2,
                    rerank=True,
                )
                all_vector_results.extend(result.get("results", []))
                all_relations.extend(result.get("relations", []))
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                continue

        # Dedup vector results by id, keeping highest score
        deduped_vectors = {}
        for item in all_vector_results:
            item_id = item.get("id")
            if item_id is None:
                continue
            existing = deduped_vectors.get(item_id)
            if existing is None:
                deduped_vectors[item_id] = item
            else:
                existing_score = existing.get("score") or 0.0
                item_score = item.get("score") or 0.0
                if item_score > existing_score:
                    deduped_vectors[item_id] = item

        # Dedup graph relations by (source, relationship, destination)
        seen_relations = set()
        deduped_relations = []
        for rel in all_relations:
            key = (rel.get("source"), rel.get("relationship"), rel.get("destination"))
            if key not in seen_relations:
                seen_relations.add(key)
                deduped_relations.append(rel)

        return {
            "results": list(deduped_vectors.values()),
            "relations": deduped_relations,
        }

    def _node_search(self, state: AddState) -> dict:
        recalled = self._search_memories(state["search_queries"], state["filters"])
        return {"recalled_memories": recalled}

    # ------------------------------------------------------------------
    # Node: decide_memory
    # ------------------------------------------------------------------

    def _decide_memory_actions(
        self,
        parsed_messages: str,
        recalled_results: list[dict],
    ) -> list[dict]:
        """One LLM call to decide ADD/UPDATE/DELETE/NONE for each piece of info.

        Feeds raw conversation text (not extracted facts!) and existing memories
        to the LLM. Returns decisions using real UUIDs (no temp mapping).
        """
        if recalled_results:
            existing_for_llm = []
            for item in recalled_results:
                entry = {
                    "id": item.get("id"),
                    "text": item.get("memory", ""),
                }
                score = item.get("score")
                if score is not None:
                    entry["score"] = score
                existing_for_llm.append(entry)

            memory_context = json.dumps(existing_for_llm, ensure_ascii=False, indent=2)
        else:
            memory_context = "[]"

        prompt = (
            f"Below is the current content of my memory which I have collected till now:\n\n"
            f"{memory_context}\n\n"
            f"The following is a conversation between a user and an assistant. "
            f"You have to analyze the conversation, identify factual information about the user/agent, "
            f"compare with existing memories, and decide ADD/UPDATE/DELETE/NONE for each.\n\n"
            f"Conversation:\n{parsed_messages}"
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": DECIDE_MEMORY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Error calling LLM for memory decisions: {e}")
            return []

        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                logger.warning("Empty response from LLM, no memories to process")
                return []

            data = json.loads(response, strict=False)
        except Exception as e:
            logger.error(f"Invalid JSON response from decide_memory LLM: {e}")
            return []

        decisions = []
        valid_events = {"ADD", "UPDATE", "DELETE", "NONE"}
        recall_ids = {item.get("id") for item in recalled_results if item.get("id")}

        for entry in data.get("memory", []):
            event = entry.get("event", "").upper()
            if event not in valid_events:
                logger.warning(f"Invalid event type from LLM: {event}, skipping")
                continue

            text = entry.get("text", "")

            decision = {
                "id": entry.get("id"),
                "text": text,
                "event": event,
            }

            if event == "UPDATE":
                decision["old_memory"] = entry.get("old_memory", "")

            # Validate: ADD must have null id
            if event == "ADD":
                decision["id"] = None
            # UPDATE/DELETE/NONE must have valid id from recall set
            elif event in ("UPDATE", "DELETE", "NONE"):
                if decision["id"] not in recall_ids:
                    logger.warning(
                        f"LLM returned id '{decision['id']}' not found in recalled memories, "
                        f"converting NONE/UPDATE/DELETE to NONE"
                    )
                    decision["event"] = "NONE"

            decisions.append(decision)

        # If no recalled memories, all new facts should be ADD (LLM may return them)
        if not recalled_results:
            for d in decisions:
                d["id"] = None
                d["event"] = "ADD"

        return decisions

    def _node_decide_memory(self, state: AddState) -> dict:
        decisions = self._decide_memory_actions(
            state["parsed_messages"],
            state["recalled_memories"]["results"],
        )
        return {"decisions": decisions}

    # ------------------------------------------------------------------
    # Node: execute_vector
    # ------------------------------------------------------------------

    def _execute_vector_operations(
        self,
        decisions: list[dict],
        metadata: dict,
    ) -> list[dict]:
        """Execute ADD/UPDATE/DELETE/NONE operations on the vector store."""
        results = []
        for decision in decisions:
            event = decision["event"]
            text = decision.get("text", "")

            try:
                if event == "ADD":
                    embeddings = self.embedding_model.embed(text, "add")
                    memory_id = self._create_memory(
                        text, {text: embeddings}, deepcopy(metadata)
                    )
                    results.append({"id": memory_id, "memory": text, "event": "ADD"})

                elif event == "UPDATE":
                    memory_id = decision["id"]
                    old_memory = decision.get("old_memory", "")
                    embeddings = self.embedding_model.embed(text, "update")
                    self._update_memory(
                        memory_id, text, {text: embeddings}, deepcopy(metadata)
                    )
                    results.append({
                        "id": memory_id,
                        "memory": text,
                        "event": "UPDATE",
                        "previous_memory": old_memory,
                    })

                elif event == "DELETE":
                    memory_id = decision["id"]
                    self._delete_memory(memory_id)
                    results.append({
                        "id": memory_id,
                        "memory": text,
                        "event": "DELETE",
                    })

                elif event == "NONE":
                    # Skip — no session ID update in new design
                    logger.debug(f"NOOP for memory id={decision.get('id')}")

            except Exception as e:
                logger.error(f"Error executing {event} for decision {decision}: {e}")

        return results

    def _node_execute_vector(self, state: AddState) -> dict:
        results = self._execute_vector_operations(
            state["decisions"], state["metadata"]
        )
        return {"results": results}

    # ------------------------------------------------------------------
    # Node: extract_graph
    # ------------------------------------------------------------------

    def _extract_graph_data(
        self,
        parsed_messages: str,
        filters: dict,
        existing_relations: list[dict],
    ) -> tuple[dict, list[dict], list[dict]]:
        """One LLM call with 3 tools: extract entities, establish relations, delete old relations.

        Returns (entity_type_map, relations, to_be_deleted).
        """
        user_id = filters.get("user_id", "user")
        system_prompt = EXTRACT_GRAPH_SYSTEM_PROMPT.format(user_id=user_id)

        # Format existing relations as context
        if existing_relations:
            existing_lines = []
            for rel in existing_relations:
                s = rel.get("source", "")
                r = rel.get("relationship", "")
                d = rel.get("destination", "")
                existing_lines.append(f"{s} -- {r} -- {d}")
            existing_text = "\n".join(existing_lines)
        else:
            existing_text = "(No existing relationships)"

        user_prompt = (
            f"Existing relationships:\n{existing_text}\n\n"
            f"Conversation:\n{parsed_messages}"
        )

        entity_type_map = {}
        relations = []
        to_be_deleted = []

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Error in extract_graph LLM call: {e}")
            return entity_type_map, relations, to_be_deleted

        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                return entity_type_map, relations, to_be_deleted

            data = json.loads(response, strict=False)

            for item in data.get("entities", []):
                entity = item.get("entity", "")
                entity_type = item.get("entity_type", "")
                if entity:
                    entity_type_map[entity] = entity_type

            for item in data.get("relations", []):
                source = item.get("source", "")
                relationship = item.get("relationship", "")
                destination = item.get("destination", "")
                if source and destination and relationship:
                    relations.append({
                        "source": source,
                        "relationship": relationship,
                        "destination": destination,
                    })

            for item in data.get("to_be_deleted", []):
                source = item.get("source", "")
                relationship = item.get("relationship", "")
                destination = item.get("destination", "")
                if source and destination and relationship:
                    to_be_deleted.append({
                        "source": source,
                        "relationship": relationship,
                        "destination": destination,
                    })
        except Exception as e:
            logger.error(f"Error parsing extract_graph JSON response: {e}")

        # Normalize: lowercase + spaces to underscores
        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        for item in relations:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        for item in to_be_deleted:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")

        return entity_type_map, relations, to_be_deleted

    def _node_extract_graph(self, state: AddState) -> dict:
        entity_type_map, relations, to_be_deleted = self._extract_graph_data(
            state["parsed_messages"],
            state["filters"],
            state["recalled_memories"]["relations"],
        )
        return {
            "entity_type_map": entity_type_map,
            "relations": relations,
            "to_be_deleted": to_be_deleted,
        }

    # ------------------------------------------------------------------
    # Node: execute_graph
    # ------------------------------------------------------------------

    def _execute_graph_write(
        self,
        entity_type_map: dict,
        relations: list[dict],
        filters: dict,
        to_be_deleted: list[dict],
    ) -> dict:
        """Call graph.ingest() with pre-extracted data. No LLM calls."""
        if not relations and not to_be_deleted:
            return {}

        try:
            result = self.graph.ingest(
                entity_type_map=entity_type_map,
                relations=relations,
                filters=filters,
                to_be_deleted=to_be_deleted if to_be_deleted else None,
            )
            return result
        except Exception as e:
            logger.error(f"Error in graph.ingest(): {e}")
            return {}

    def _node_execute_graph(self, state: AddState) -> dict:
        graph_result = self._execute_graph_write(
            state["entity_type_map"],
            state["relations"],
            state["filters"],
            state["to_be_deleted"],
        )
        return {"graph_result": graph_result}

    # ------------------------------------------------------------------
    # Node: assemble_result
    # ------------------------------------------------------------------

    def _assemble_final_result(
        self,
        results: list[dict],
        recalled_memories: dict,
        graph_result: Optional[dict] = None,
    ) -> dict:
        """Merge all node outputs into the final response dict."""
        final: dict = {
            "results": results,
            "recalled_memories": recalled_memories,
        }

        if graph_result and (graph_result.get("deleted_entities") or graph_result.get("added_entities")):
            final["relations"] = graph_result

        return final

    def _node_assemble_result(self, state: AddState) -> dict:
        graph_result = state.get("graph_result") if self.enable_graph else None
        final = self._assemble_final_result(
            state["results"],
            state["recalled_memories"],
            graph_result,
        )
        return {"final_results": final}

    # ------------------------------------------------------------------
    # Vector store helpers (mirror Memory._create/_update/_delete_memory)
    # ------------------------------------------------------------------

    def _create_memory(self, data: str, existing_embeddings: dict, metadata: dict = None) -> str:
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _update_memory(self, memory_id: str, data: str, existing_embeddings: dict, metadata: dict = None) -> str:
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory
        for key in ("user_id", "agent_id", "run_id", "actor_id", "role"):
            if key not in new_metadata and key in existing_memory.payload:
                new_metadata[key] = existing_memory.payload[key]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id: str) -> str:
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id


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
