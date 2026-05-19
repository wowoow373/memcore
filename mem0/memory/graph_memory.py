import logging

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class MemoryGraph:
    def __init__(self, config, node_label=None):
        self.config = config
        self.graph = Neo4jGraph(
            url=self.config.graph_store.config.url,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
            database=self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )

        if node_label is not None:
            self.node_label = node_label
        elif self.config.graph_store.config.base_label:
            self.node_label = ":`__Entity__`"
        else:
            self.node_label = ""

        if self.config.graph_store.config.base_label and node_label is None:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        # print("======entity_type_map!!!!!!!:", entity_type_map) 
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        #print("!!!!!!to_be_added!!!!!!!:", to_be_added)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        print("search_output:", search_output)
        print("=====data:====", data)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
 
        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)
        # print("added_entities:", added_entities)
        # print("deleted_entities:", deleted_entities)


        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search_nodes(self, node_names, filters, depth=2, limit=100):
        """Pure graph traversal from given node names.

        Args:
            node_names: Pre-extracted node names to start traversal from.
            filters: Scope filters (user_id, agent_id, run_id).
            depth: Traversal depth (hops).
            limit: Max number of relations to return.

        Returns:
            list[dict]: Relations with keys source/relationship/destination.
        """
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = 1
        if depth <= 0:
            return []
        max_results = depth * 20

        # Normalize node names.
        node_list = [str(node).strip().lower().replace(" ", "_") for node in node_names if str(node).strip()]
        if not node_list:
            return []

        # Traverse the graph from the resolved start nodes.
        search_output = self._search_graph_db_by_depth(
            node_list=node_list,
            filters=filters,
            depth=depth,
            limit=max_results,
        )

        if not search_output:
            return []

        search_results = []
        for item in search_output:
            search_results.append({"source": item["source"], "relationship": item["relationship"], "destination": item["destination"]})

        logger.info(f"Returned {len(search_results)} graph search results")

        return search_results

    def search_nodes_by_embedding(
        self,
        embedding: list[float],
        filters: dict,
        top_k: int = 10,
        threshold: float = 0.6,
    ) -> list[dict]:
        """Search nodes by embedding cosine similarity on brief_embedding property.

        Receives a pre-computed embedding and matches against Step nodes that
        have a brief_embedding property. The caller (ProcessMemorySearchEngine)
        is responsible for computing the embedding externally.

        Args:
            embedding: Pre-computed embedding vector.
            filters: Scope filters; must contain at least user_id.
            top_k: Max number of nodes to return.
            threshold: Minimum cosine similarity score (0.0-1.0).

        Returns:
            list[dict]: Nodes with keys name, brief, goal, step, action, score.
        """
        if not embedding:
            return []
        user_id = filters.get("user_id")
        if not user_id:
            return []

        conditions = ["n.user_id = $user_id", "n.brief_embedding IS NOT NULL"]
        if filters.get("agent_id"):
            conditions.append("n.agent_id = $agent_id")
        if filters.get("run_id"):
            conditions.append("n.run_id = $run_id")
        where_clause = " AND ".join(conditions)

        cypher = f"""
        MATCH (n {self.node_label})
        WHERE {where_clause}
        WITH n, vector.similarity.cosine(n.brief_embedding, $embedding) AS score
        WHERE score >= $threshold
        RETURN n.name AS name,
               n.brief AS brief,
               n.goal AS goal,
               n.step AS step,
               n.action AS action,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """

        params = {
            "embedding": embedding,
            "user_id": user_id,
            "threshold": threshold,
            "top_k": top_k,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        try:
            return self.graph.query(cypher, params=params)
        except Exception as e:
            logger.error(f"search_nodes_by_embedding failed: {e}")
            return []

    def search(self, query, filters, limit=100):
        """Backward-compatible wrapper. No LLM extraction.

        - query is list/tuple/set: pass directly to search_nodes.
        - query is str: split by comma (or keep as single item) -> search_nodes.

        Args:
            query (str | list[str]): Fact text or node name(s) to search from.
            filters (dict): Scope filters; must contain at least `user_id`.
            limit (int): **Traversal depth** (hops) in the graph.  Defaults to 100.

        Returns:
            list[dict]: Relations, each with keys:
                - "source"        : str, start node name
                - "relationship"  : str, edge type
                - "destination"   : str, end node name
        """
        try:
            depth = int(limit)
        except (TypeError, ValueError):
            depth = 1
        if depth <= 0:
            return []

        if isinstance(query, (list, tuple, set)):
            node_names = [str(node).strip() for node in query if str(node).strip()]
        elif isinstance(query, str):
            node_names = [node.strip() for node in query.split(",") if node.strip()] if "," in query else [query.strip()]
        else:
            node_names = [str(query).strip()] if str(query).strip() else []

        return self.search_nodes(node_names, filters, depth=depth, limit=limit)

    def _search_graph_db_by_depth(self, node_list, filters, depth=1, limit=20):
        """Search relations by traversing from given nodes up to a fixed depth."""
        result_relations = []

        # Build WHERE filters for start and traversed nodes.
        start_conditions = ["n.user_id = $user_id", "n.name IN $node_list"]
        traversed_conditions = ["src.user_id = $user_id", "dst.user_id = $user_id"]
        if filters.get("agent_id"):
            start_conditions.append("n.agent_id = $agent_id")
            traversed_conditions.append("src.agent_id = $agent_id")
            traversed_conditions.append("dst.agent_id = $agent_id")
        if filters.get("run_id"):
            start_conditions.append("n.run_id = $run_id")
            traversed_conditions.append("src.run_id = $run_id")
            traversed_conditions.append("dst.run_id = $run_id")

        start_where = " AND ".join(start_conditions)
        traversed_where = " AND ".join(traversed_conditions)
        safe_depth = max(1, int(depth))

        cypher_query = f"""
        MATCH (n {self.node_label})
        WHERE {start_where}
        MATCH path = (n)-[*1..{safe_depth}]-(m {self.node_label})
        UNWIND relationships(path) AS rel
        WITH DISTINCT startNode(rel) AS src, rel, endNode(rel) AS dst
        WHERE {traversed_where}
        RETURN src.name AS source,
               elementId(src) AS source_id,
               type(rel) AS relationship,
               elementId(rel) AS relation_id,
               dst.name AS destination,
               elementId(dst) AS destination_id
        LIMIT $limit
        """

        params = {
            "node_list": node_list,
            "user_id": filters["user_id"],
            "limit": limit,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        ans = self.graph.query(cypher_query, params=params)
        result_relations.extend(ans)
        return result_relations

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query.

        DEPRECATED: Entity extraction should be performed upstream by the add engine.
        Use `ingest()` which accepts pre-extracted `entity_type_map`.
        """
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )
        print("======search_results from entity extraction:=======", search_results)
        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call.get("arguments", {}).get("entities", []):
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes.

        DEPRECATED: Relation extraction should be performed upstream by the add engine.
        Use `ingest()` which accepts pre-extracted `relations`.
        """

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations.

        DEPRECATED: Embedding-based similarity search is no longer needed in the add flow.
        Use `ingest()` which relies on exact MERGE by node name.
        """
        result_relations = []

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)
        print("node_props_str:", node_props_str)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity // denormalize for backward compatibility
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                WITH n  
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output.

        DEPRECATED: Deletion decisions should be made upstream by the add engine.
        Use `ingest()` and pass `to_be_deleted` directly.
        """
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist.

        DEPRECATED: Replaced by `ingest()` which uses exact MERGE without
        embedding similarity search. Use `ingest()` for new code.
        """
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = entity_type_map.get(destination, "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        # DEPRECATED: Embedding-based node lookup is no longer needed.
        # Use `ingest()` which relies on exact MERGE by node name.
        # Build WHERE conditions
        where_conditions = ["source_candidate.embedding IS NOT NULL", "source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("source_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE {where_clause}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # DEPRECATED: Embedding-based node lookup is no longer needed.
        # Use `ingest()` which relies on exact MERGE by node name.
        # Build WHERE conditions
        where_conditions = ["destination_candidate.embedding IS NOT NULL", "destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("destination_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE {where_clause}

            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def ingest(self, entity_type_map, relations, filters, to_be_deleted=None, node_properties=None):
        """Add-engine friendly interface: pure graph write without LLM/embedding calls.

        Receives pre-extracted structured data and performs exact MERGE
        operations. No entity extraction, no relation extraction, no
        embedding similarity search.

        Args:
            entity_type_map: dict of {entity_name: entity_type}.
            relations: list of dicts, each with 'source', 'relationship', 'destination'.
            filters: dict with at least 'user_id'; optional 'agent_id', 'run_id'.
            to_be_deleted: optional list of dicts, each with 'source', 'relationship', 'destination'.
            node_properties: optional dict of {node_name: {prop: value}}. When provided,
                extra properties (brief, goal, action, brief_embedding) are set on
                matched/created nodes. brief_embedding is stored as a vector property.

        Returns:
            dict: {'deleted_entities': [...], 'added_entities': [...]}
        """
        # 1. Handle deletions
        deleted = []
        if to_be_deleted:
            deleted = self._delete_entities(to_be_deleted, filters)

        # 2. Normalize inputs
        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        for item in relations:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = sanitize_relationship_for_cypher(
                item["relationship"].lower().replace(" ", "_")
            )
            item["destination"] = item["destination"].lower().replace(" ", "_")

        # 3. Write relations
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        run_id = filters.get("run_id")

        added = []
        for item in relations:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            source_label = self.node_label if self.node_label else f":`{source_type}`"
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Build node_properties extra SET clauses and params
            source_np = (node_properties or {}).get(source, {})
            dest_np = (node_properties or {}).get(destination, {})

            source_create_set = ""
            source_match_set = ""
            source_vector_call = ""
            source_vector_with = ""
            dest_create_set = ""
            dest_match_set = ""
            dest_vector_call = ""
            dest_vector_with = ""

            if source_np:
                for key in ("brief", "goal", "action"):
                    if key in source_np:
                        source_create_set += f",\n                            source.{key} = $source_{key}"
                        source_match_set += f",\n                            source.{key} = $source_{key}"
                if "brief_embedding" in source_np:
                    source_vector_call = (
                        "\n            CALL db.create.setNodeVectorProperty("
                        "source, 'brief_embedding', $source_brief_embedding)"
                    )
                    source_vector_with = "\n            WITH source"

            if dest_np:
                for key in ("brief", "goal", "action"):
                    if key in dest_np:
                        dest_create_set += f",\n                            destination.{key} = $dest_{key}"
                        dest_match_set += f",\n                            destination.{key} = $dest_{key}"
                if "brief_embedding" in dest_np:
                    dest_vector_call = (
                        "\n            CALL db.create.setNodeVectorProperty("
                        "destination, 'brief_embedding', $dest_brief_embedding)"
                    )
                    dest_vector_with = "\n            WITH destination"

            cypher = f"""
            MERGE (source {source_label} {{{source_props_str}}})
            ON CREATE SET source.created = timestamp(),
                        source.mentions = 1
                        {source_extra_set}{source_create_set}
            ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1{source_match_set}{source_vector_call}{source_vector_with}
            MERGE (destination {destination_label} {{{dest_props_str}}})
            ON CREATE SET destination.created = timestamp(),
                        destination.mentions = 1
                        {destination_extra_set}{dest_create_set}
            ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1{dest_match_set}{dest_vector_call}{dest_vector_with}
            WITH source, destination
            MERGE (source)-[rel:{relationship}]->(destination)
            ON CREATE SET rel.created = timestamp(), rel.mentions = 1
            ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
            RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
            """

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Add node_properties params
            for prefix, np in [("source", source_np), ("dest", dest_np)]:
                for key in ("brief", "goal", "action", "brief_embedding"):
                    if key in np:
                        params[f"{prefix}_{key}"] = np[key]

            result = self.graph.query(cypher, params=params)
            added.append(result)

        return {"deleted_entities": deleted, "added_entities": added}

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
