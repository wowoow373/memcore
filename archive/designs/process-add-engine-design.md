# ProcessMemoryAddEngine 详细设计

## 一、概述

ProcessMemoryAddEngine 是流程记忆的**写入**引擎。仅用于 Flow 1（任务完成后），通过 LangGraph 编排五个节点：preprocess → search → decide → execute → assemble。

与标准 `AddEngine` 的核心差异：
- 输入不是原始对话 messages，而是已结构化的 summary 数组
- 三层并行写入（Graph / Chunk / Summary），不是先向量后图
- 去重逻辑是 Goal 级别的 merge，不是 fact 级别的 ADD/UPDATE/DELETE

---

## 二、初始化与依赖

```python
class ProcessMemoryAddEngine:
    def __init__(
        self,
        embedding_model,    # EmbedderFactory 创建的实例
        vector_store,       # VectorStoreBase 子类实例（Chunk + Summary 共用）
        llm,                # LlmFactory 创建的 LLM 实例
        db,                 # SQLiteManager 实例，用于 history 记录
        search_engine,      # ProcessMemorySearchEngine 实例
        graph_store,        # MemoryGraph 实例（Step node_label 配置）
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.db = db
        self.search_engine = search_engine
        self.graph_store = graph_store

        # 编译 LangGraph
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
```

所有依赖外部注入，Engine 内部不创建。LangGraph 编译在 `__init__` 完成。

---

## 三、公开接口

```python
def add(
    self,
    summaries: list[dict],    # 完整 summary 数组，按 step 执行顺序排列
    metadata: dict,           # {"user_id": str, "agent_id": str|None, "run_id": str|None}
) -> dict:
```

`summaries` 的每项：

```json
{
    "Goal": "添加用户认证功能",
    "Step": "03 - 创建 auth.py",
    "Action": "create_file(path='auth.py')",
    "Dependency": [
        {"step_id": "01", "description": "解析 main.py 入口逻辑"}
    ],
    "Brief": "创建 auth.py 并实现 login 函数框架"
}
```

**返回**：

```python
{
    "results": {
        "graph": {
            "added_steps": [...],
            "updated_steps": [...],
            "added_edges": [...],
            "deleted_edges": [...],
        },
        "chunks": [
            {"goal": "...", "event": "ADD|MERGE", "id": "..."},
        ],
        "summary": {
            "event": "ADD|UPDATE",
            "id": "...",
            "task_description": "...",
        },
    },
    "recalled": {
        "graph": {"chains": [...]},
        "chunks": [...],
        "summaries": [...],
    },
}
```

---

## 四、LangGraph State

```python
class ProcessAddState(TypedDict):
    # ── 输入（外部传入）──
    summaries: list[dict]
    metadata: dict
    filters: dict

    # ── preprocess 产出 ──
    goals: list[str]                   # 去重的 Goal 列表
    task_description: str              # 任务宏观描述（由 goals + briefs 拼接）
    steps: list[dict]                  # [{"name", "goal", "brief", "action", "dependency"}, ...]
    dependencies: list[dict]           # [{"source", "target", "relationship"}, ...]
    entity_type_map: dict              # {step_name: "Step"} 用于 graph_store.ingest()

    # ── search 产出 ──
    recalled: dict                     # ProcessMemorySearchEngine.search_for_dedup() 返回

    # ── decide 产出 ──
    decisions: dict                    # LLM 决策 JSON，结构见第六节

    # ── execute 产出 ──
    graph_result: dict                 # graph_store.ingest() 返回
    chunk_results: list[dict]          # 每条 chunk 的写入结果
    summary_result: dict               # summary 的写入结果

    # ── 输出 ──
    results: dict                      # 合并的三层写入结果
    final_results: dict                # {"results": ..., "recalled": ...}
    error: Optional[str]
```

---

## 五、节点详细设计

### 5.1 preprocess —— 纯解析，无 LLM

```python
def _preprocess_summaries(
    self, summaries: list[dict]
) -> tuple[list[str], str, list[dict], list[dict], dict]:
```

**逻辑**：

1. **提取 steps**：遍历 summaries，构造 step 节点：
   ```python
   step = {
       "name": summary["Step"],           # "03 - 创建 auth.py"
       "goal": summary["Goal"],           # "添加用户认证功能"
       "brief": summary["Brief"],         # "创建 auth.py 并实现 login 函数框架"
       "action": summary["Action"],       # "create_file(path='auth.py')"
   }
   ```

2. **提取 dependencies**：遍历每个 summary 的 `Dependency` 数组，构造边：
   ```python
   for dep in summary["Dependency"]:
       dependencies.append({
           "source": dep["step_id"],       # "01"
           "target": summary["Step"],      # "03 - 创建 auth.py"
           "relationship": "DEPENDS_ON",
       })
   ```
   注意：`dep["step_id"]` 是外部 agent 的 step_id 格式（如 "01"），需要映射到完整的 step name（如 "01 - 阅读 main.py"）。如果 summaries 中能找到对应 step name，则使用完整名称；否则使用原始 step_id。

3. **提取 goals**：去重收集所有 `Goal`。

4. **生成 task_description**：拼接所有 Goal + 关键 Brief 生成一个宏观描述文本。无需 LLM。

5. **构造 entity_type_map**：`{step["name"]: "Step" for step in steps}` —— 用于后续 `graph_store.ingest()` 调用。

```python
def _node_preprocess(self, state: ProcessAddState) -> dict:
    goals, task_desc, steps, deps, entity_type_map = self._preprocess_summaries(
        state["summaries"]
    )
    return {
        "goals": goals,
        "task_description": task_desc,
        "steps": steps,
        "dependencies": deps,
        "entity_type_map": entity_type_map,
    }
```

---

### 5.2 search —— 调 SearchEngine 去重

```python
def _node_search(self, state: ProcessAddState) -> dict:
    recalled = self.search_engine.search_for_dedup(
        goals=state["goals"],
        task_description=state["task_description"],
        filters=state["filters"],
    )
    return {"recalled": recalled}
```

**边界**：Search 节点只做调用和状态写入，检索逻辑完全在 ProcessMemorySearchEngine 内部。

---

### 5.3 decide —— 一次 LLM，统一决策三层

```python
def _decide_process_memory(
    self,
    summaries: list[dict],      # 原始输入
    goals: list[str],           # 去重后的 Goal
    task_description: str,      # 任务宏观描述
    recalled: dict,             # 三层召回结果
    filters: dict,
) -> dict:
```

**逻辑**：

1. 将 `summaries` 和 `recalled` 格式化为 prompt 文本
2. 一次 LLM 调用，返回三层决策 JSON
3. 校验 LLM 返回：ADD 的 id 为 null，UPDATE/MERGE 的 id 必须在 recalled 中存在，否则降级为 NONE

**Prompt 结构**（核心内容，具体措辞实现时细化）：

```
System: 你是流程记忆管理器。分析新任务与已有记忆，决定三层存储的变更。

User:
## 已有记忆
### Graph（已有步骤链）
{recalled.graph.chains}

### Chunks（已有子目标）
{recalled.chunks}

### Summaries（已有任务总览）
{recalled.summaries}

## 新任务
### Steps
{summaries}
### Goals
{goals}
### 任务描述
{task_description}

请返回 JSON，对三层分别决策：
- Graph steps: ADD/UPDATE/NONE；Graph edges: ADD/DELETE/NONE
- Chunks: ADD/MERGE/NONE（同 Goal 聚合到同一 chunk）
- Summary: ADD/UPDATE/NONE
```

**LLM 输出结构**：

```json
{
  "graph": {
    "steps": [
      {"name": "01 - 阅读 main.py", "event": "ADD", "goal": "添加用户认证", "brief": "...", "action": "..."},
      {"name": "03 - 创建 auth.py", "event": "UPDATE", "id": "existing_node_123", "goal": "添加用户认证", "brief": "更新后的 brief"},
      {"name": "05 - 测试", "event": "NONE"}
    ],
    "edges": [
      {"source": "01 - 阅读 main.py", "target": "03 - 创建 auth.py", "relationship": "DEPENDS_ON", "event": "ADD"},
      {"source": "01 - 阅读 main.py", "target": "04 - 旧步骤", "relationship": "DEPENDS_ON", "event": "DELETE"}
    ]
  },
  "chunks": [
    {
      "goal": "添加用户认证功能",
      "event": "MERGE",
      "merge_with": "existing_chunk_id_456",
      "steps": [{"step": "01 - 阅读 main.py", "brief": "..."}, {"step": "03 - 创建 auth.py", "brief": "..."}]
    },
    {
      "goal": "配置数据库连接",
      "event": "ADD",
      "steps": [{"step": "02 - 创建 config.py", "brief": "..."}]
    }
  ],
  "summary": {
    "event": "ADD",
    "task_description": "实现完整的用户认证系统，包括登录、注册、密码重置",
    "full_chain": ["01 - 阅读 main.py", "02 - 创建 config.py", "03 - 创建 auth.py", "04 - 修改 main.py", "05 - 测试"]
  }
}
```

**校验规则**（与 AddEngine.decide_memory 一致）：

- ADD event：强制 `id = null`
- UPDATE/MERGE event：`id` / `merge_with` 必须在 recalled 中存在，否则降级为 ADD
- 无效 event 类型 → 丢弃并 log
- recalled 为空 → 所有变为 ADD

```python
def _node_decide(self, state: ProcessAddState) -> dict:
    decisions = self._decide_process_memory(
        summaries=state["summaries"],
        goals=state["goals"],
        task_description=state["task_description"],
        recalled=state["recalled"],
        filters=state["filters"],
    )
    return {"decisions": decisions}
```

---

### 5.4 execute —— 三层独立并行写入

```python
def _execute_all(
    self,
    decisions: dict,
    entity_type_map: dict,
    filters: dict,
    metadata: dict,
) -> tuple[dict, list[dict], dict]:
```

**逻辑**：三步独立，可并行（内部用 `concurrent.futures.ThreadPoolExecutor` 或顺序执行均可，初期量小）：

#### 5.4.1 图写入

使用扩展后的 `MemoryGraph.ingest()`，传入 `node_properties` 携带 Brief embedding 和业务属性：

```python
# 准备数据
graph_entity_type_map = {
    step["name"]: "Step" for step in decisions["graph"]["steps"] if step["event"] in ("ADD", "UPDATE")
}

graph_relations = [
    edge for edge in decisions["graph"]["edges"] if edge["event"] == "ADD"
]

graph_to_be_deleted = [
    edge for edge in decisions["graph"]["edges"] if edge["event"] == "DELETE"
]

# 构造 node_properties：{node_name: {"brief", "goal", "action", "brief_embedding"}}
graph_node_properties = {}
for step in decisions["graph"]["steps"]:
    if step["event"] in ("ADD", "UPDATE"):
        graph_node_properties[step["name"]] = {
            "brief": step["brief"],
            "goal": step["goal"],
            "action": step.get("action", ""),
            "brief_embedding": self.embedding_model.embed(step["brief"], "add"),
        }

# 调扩展后的 MemoryGraph.ingest
graph_result = self.graph_store.ingest(
    entity_type_map=graph_entity_type_map,
    relations=graph_relations,
    filters=filters,
    to_be_deleted=graph_to_be_deleted if graph_to_be_deleted else None,
    node_properties=graph_node_properties,    # 新增参数
)
```

**`MemoryGraph.ingest()` 扩展**：

签名新增可选参数 `node_properties: dict[str, dict] | None = None`：

```python
def ingest(self, entity_type_map, relations, filters,
           to_be_deleted=None, node_properties=None):
```

当 `node_properties` 不为 None 且包含某节点名时，在 MERGE Cypher 的 ON CREATE SET / ON MATCH SET 中附加写入 `brief`、`goal`、`action`、`brief_embedding` 属性。标准记忆调用（不传 `node_properties`）行为完全不变。

```cypher
-- 扩展后的 Cypher（仅 node_properties 非空且含此节点时追加 SET 子句）
MERGE (source {source_label} {{name: $source_name, user_id: $user_id}})
ON CREATE SET source.created = timestamp(), source.mentions = 1,
              source.brief = $source_brief, source.goal = $source_goal,
              source.action = $source_action,
              source.brief_embedding = $source_brief_embedding
ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1,
              source.brief = $source_brief, source.goal = $source_goal,
              source.action = $source_action,
              source.brief_embedding = $source_brief_embedding
```

当 `node_properties=None` 时，回退到现有 Cypher（只 SET mentions），向后兼容。

#### 5.4.2 Chunk 写入

```python
for chunk_decision in decisions["chunks"]:
    if chunk_decision["event"] == "ADD":
        # embed Goal 文本
        embeddings = self.embedding_model.embed(chunk_decision["goal"], "add")
        chunk_id = str(uuid.uuid4())
        payload = deepcopy(metadata)
        payload["memory_type"] = "process_chunk"
        payload["goal"] = chunk_decision["goal"]
        payload["steps"] = chunk_decision["steps"]
        payload["data"] = chunk_decision["goal"]  # 向量库的 data 字段
        payload["hash"] = hashlib.md5(chunk_decision["goal"].encode()).hexdigest()
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        self.vector_store.insert(vectors=[embeddings], ids=[chunk_id], payloads=[payload])
        self.db.add_history(chunk_id, None, chunk_decision["goal"], "ADD", ...)
        chunk_results.append({"id": chunk_id, "goal": chunk_decision["goal"], "event": "ADD"})

    elif chunk_decision["event"] == "MERGE":
        # 更新已有 chunk
        existing_id = chunk_decision["merge_with"]
        embeddings = self.embedding_model.embed(chunk_decision["goal"], "update")
        existing = self.vector_store.get(vector_id=existing_id)
        new_payload = deepcopy(existing.payload)
        # 合并 steps：按 step name 去重
        new_payload["steps"] = _merge_steps(existing.payload.get("steps", []), chunk_decision["steps"])
        new_payload["data"] = chunk_decision["goal"]
        new_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.vector_store.update(vector_id=existing_id, vector=embeddings, payload=new_payload)
        self.db.add_history(existing_id, existing.payload.get("data"), chunk_decision["goal"], "UPDATE", ...)
        chunk_results.append({"id": existing_id, "goal": chunk_decision["goal"], "event": "MERGE"})
```

#### 5.4.3 Summary 写入

```python
summary_decision = decisions["summary"]
if summary_decision["event"] == "ADD":
    embeddings = self.embedding_model.embed(summary_decision["task_description"], "add")
    summary_id = str(uuid.uuid4())
    payload = deepcopy(metadata)
    payload["memory_type"] = "process_summary"
    payload["task_description"] = summary_decision["task_description"]
    payload["full_chain"] = summary_decision["full_chain"]
    payload["data"] = summary_decision["task_description"]
    payload["hash"] = hashlib.md5(summary_decision["task_description"].encode()).hexdigest()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    self.vector_store.insert(vectors=[embeddings], ids=[summary_id], payloads=[payload])
    self.db.add_history(summary_id, None, summary_decision["task_description"], "ADD", ...)
    summary_result = {"id": summary_id, "event": "ADD", "task_description": summary_decision["task_description"]}

elif summary_decision["event"] == "UPDATE":
    # 类似 UPDATE 逻辑
    ...
```

```python
def _node_execute(self, state: ProcessAddState) -> dict:
    graph_result, chunk_results, summary_result = self._execute_all(
        decisions=state["decisions"],
        entity_type_map=state["entity_type_map"],
        filters=state["filters"],
        metadata=state["metadata"],
    )
    return {
        "graph_result": graph_result,
        "chunk_results": chunk_results,
        "summary_result": summary_result,
    }
```

---

### 5.5 assemble —— 组装返回

```python
def _assemble_final_result(
    self,
    graph_result: dict,
    chunk_results: list[dict],
    summary_result: dict,
    recalled: dict,
) -> dict:
    results = {
        "graph": graph_result,
        "chunks": chunk_results,
        "summary": summary_result,
    }
    return {
        "results": results,
        "recalled": recalled,
    }

def _node_assemble(self, state: ProcessAddState) -> dict:
    final = self._assemble_final_result(
        graph_result=state["graph_result"],
        chunk_results=state["chunk_results"],
        summary_result=state["summary_result"],
        recalled=state["recalled"],
    )
    return {"final_results": final}
```

---

## 六、LangGraph 图结构

```
START
  │
  ▼
preprocess     ← 无 LLM，纯解析 summaries → goals/steps/deps/task_desc/entity_type_map
  │
  ▼
search         ← 调用 ProcessMemorySearchEngine.search_for_dedup()
  │
  ▼
decide         ← 一次 LLM：summaries + recalled → 三层决策 JSON
  │
  ▼
execute        ← 三层独立并行写入（Graph via MemoryGraph.ingest / Chunk via vector_store / Summary via vector_store）
  │
  ▼
assemble       ← 合并 results + recalled → final_results
  │
  ▼
END
```

共 **5 个节点**，**0 个条件分支**（流程记忆始终走全流程，无 infer 开关，三层始终全写）。

---

## 七、节点职责边界总结

| 节点 | 职责 | LLM 调用 | 写操作 | 外部依赖 |
|------|------|---------|--------|---------|
| `preprocess` | 解析 summaries → 结构化中间数据 | 无 | 无 | 无 |
| `search` | 委托 ProcessMemorySearchEngine 做三层去重召回 | 无 | 无 | `ProcessMemorySearchEngine` |
| `decide` | 一次 LLM 决策三层的 ADD/UPDATE/MERGE/NONE | **1 次** | 无 | `self.llm` |
| `execute` | 三层独立并行写入 | 无 | **有** | `MemoryGraph.ingest`, `vector_store`, `SQLiteManager` |
| `assemble` | 合并 results + recalled → 最终返回 | 无 | 无 | 无 |

---

## 八、与标准 AddEngine 的差异

| 维度 | AddEngine（标准记忆） | ProcessMemoryAddEngine |
|------|---------------------|----------------------|
| 输入 | 原始对话 messages | 结构化 summary 数组 |
| LLM 调用点 | extract_queries + decide_memory + extract_graph（3 次） | decide（1 次） |
| 条件分支 | infer + enable_graph（2 个） | 无 |
| 图写入方式 | extract_graph(LLM) → execute_graph(ingest) | decide(LLM) → execute(ingest) |
| 向量写入层级 | 单层（facts） | 两层（Chunk + Summary） |
| 去重方式 | fact 级 ADD/UPDATE/DELETE/NONE | Goal 级 ADD/MERGE/NONE + step 级 ADD/UPDATE/NONE |

---

## 九、已确定的技术决策

1. **`MemoryGraph.ingest()` 扩展** ✅：新增可选参数 `node_properties: dict[str, dict] | None = None`。非空时在 MERGE 的 ON CREATE/MATCH SET 中附加写入 `brief`/`goal`/`action`/`brief_embedding`。标准记忆调用不传该参数，行为完全不变。
2. **`MemoryGraph.search_nodes_by_embedding`** ✅：新增方法，接收外部预计算的 embedding，不自己做 embed。调用方（ProcessMemorySearchEngine）负责在外部完成 embedding。
3. **execute 内部并行**：跳过，顺序执行。后续量大再优化。
4. **task_description 生成**：暂留，当前用 goals + briefs 拼接，不调 LLM。
