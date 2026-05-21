# Add Engine 设计规格

## 一、总览

Add Engine 是 Memory 写入层的 LangGraph 编排引擎。它将当前 `main.py` 中 `add()` 的内联逻辑拆分为独立、可测试的节点，通过 LangGraph 状态机编排。

### 核心设计原则

- **Search 是 Add 的依赖**：Add Engine 内部调用 SearchEngine 获取已有记忆，SearchEngine 不感知 Add
- **recalled_memories 随 Add 返回**：让调用方感知系统"联想"了哪些记忆
- **图写入与向量写入分离**：用 graph.ingest() 替代旧的 graph.add()，LLM 实体/关系提取上提到 Add Engine
- **不做流程记忆**：`memory_type="procedural_memory"` 暂不实现，留 TODO

### 流程对比

```
OLD add():
  messages → parse → extract_facts(LLM) → per-fact vector search(N+1)
  → temp UUID mapping → decide(LLM) → execute ADD/UPDATE/DELETE
  → graph.add(独立LLM管线: 提取实体→提取关系→embedding搜索→写入)

NEW add():
  messages → preprocess → extract_queries(LLM) → search(SearchEngine)
  → decide_memory(LLM) → execute_vector → extract_graph(LLM) → execute_graph(graph.ingest)
  → assemble_result(含 recalled_memories)
```

### LangGraph 图结构

```
START
  │
  ▼
preprocess ──┬── infer=False ──▶ direct_add ──▶ assemble_result ──▶ END
  │
  ▼ (infer=True)
extract_queries
  │
  ▼
search
  │
  ▼
decide_memory
  │
  ▼
execute_vector
  │
  ├── enable_graph=False ──▶ assemble_result ──▶ END
  │
  ▼ (enable_graph=True)
extract_graph
  │
  ▼
execute_graph
  │
  ▼
assemble_result ──▶ END
```

共 9 个节点，2 个条件分支。

---

## 二、接口与文件

### 2.1 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `mem0/memory/add_engine.py` | **新增** | Add Engine 主体，含 AddState + LangGraph + 核心方法 |
| `mem0/memory/main.py` | **修改** | `Memory.add()` 委托给 AddEngine；删除 `_add_to_vector_store`、`_add_to_graph` 旧方法 |
| `tests/memory/test_add_engine.py` | **新增** | Add Engine 单元测试（各核心方法独立可测） |
| `tests/memory/test_add_engine_e2e.py` | **新增** | Add Engine E2E 测试（需 Docker 服务） |

不改动：`graph_memory.py`、`search_engine.py`、`utils.py`、`factory.py`、`prompts.py`、`storage.py`、`telemetry.py`

### 2.2 AddEngine 类签名

```python
class AddEngine:
    def __init__(
        self,
        embedding_model: Any,
        vector_store: Any,
        graph: Optional[Any],
        llm: Any,
        db: Any,                          # SQLiteManager for history
        search_engine: Any,               # SearchEngine instance
    ):
        ...

    def add(
        self,
        messages: list[dict],
        metadata: dict,
        filters: dict,
        infer: bool = True,
    ) -> dict:
        ...
```

### 2.3 Memory.add() 委托方式

```python
# main.py Memory.__init__
self.add_engine = AddEngine(
    embedding_model=self.embedding_model,
    vector_store=self.vector_store,
    graph=self.graph if self.enable_graph else None,
    llm=self.llm,
    db=self.db,
    search_engine=self.search_engine,
)

# main.py Memory.add()
def add(self, messages, *, user_id=None, agent_id=None, run_id=None,
        metadata=None, infer=True, memory_type=None, prompt=None):
    # 参数校验（保留在 Memory 层）
    processed_metadata, effective_filters = _build_filters_and_metadata(...)
    # 归一化 messages 格式（保留在 Memory 层）
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        messages = [messages]
    # 委托给 AddEngine
    return self.add_engine.add(
        messages=messages,
        metadata=processed_metadata,
        filters=effective_filters,
        infer=infer,
    )
```

参数校验、messages 归一化、`_build_filters_and_metadata`、vision message 解析、telemetry 均在 Memory 层完成。AddEngine 只处理核心写入逻辑。

### 2.4 对外返回结构

```python
# 有图
{
    "results": [
        {"id": "uuid", "memory": "text", "event": "ADD"},
        {"id": "uuid", "memory": "text", "event": "UPDATE", "previous_memory": "old text"},
        {"id": "uuid", "memory": "text", "event": "DELETE"},
    ],
    "recalled_memories": {
        "results": [MemoryItem, ...],
        "relations": [{"source": "..", "relationship": "..", "destination": ".."}, ...]
    },
    "relations": {"deleted_entities": [...], "added_entities": [...]}
}

# 无图
{
    "results": [...],
    "recalled_memories": {"results": [MemoryItem, ...]}
}
```

### 2.5 AddState 定义

```python
class AddState(TypedDict):
    # ── 输入 ──
    messages: list[dict]
    metadata: dict
    filters: dict
    infer: bool

    # ── 中间产物 ──
    parsed_messages: str
    search_queries: list[str]
    recalled_memories: dict          # {"results": [MemoryItem], "relations": [...]}
    decisions: list[dict]            # [{"id": str, "text": str, "event": str, "old_memory": str}]
    entity_type_map: dict            # {"entity": "type"}
    relations: list[dict]            # [{"source", "relationship", "destination"}]
    to_be_deleted: list[dict]        # 同上格式
    graph_result: dict               # graph.ingest() 返回值

    # ── 输出 ──
    results: list[dict]
    final_results: dict
    error: Optional[str]
```

---

## 三、节点详细边界与行为

### 3.1 preprocess — 预处理

**职责**：校验输入，拼接消息文本，设置状态初始值。

**输入**：`messages`, `metadata`, `filters`, `infer`

**行为**：
1. 拼接 messages 文本：将 messages 中 role 为 user/assistant 的 content 提取并格式化为 `"user: content\nassistant: content\n"` → 存入 `parsed_messages`
2. 将 filters 中的 `user_id` 补充到 filters（若缺失且无其他标识可用时不做 fallback，由上游保证）

**输出**：
```python
{
    "parsed_messages": "user: ...\nassistant: ...\n",
}
```

**边界**：
- 不做参数校验（上游 Memory.add() 已完成）
- 不做 messages 类型归一化
- system role 消息被跳过

---

### 3.2 direct_add — 直接写入（infer=False）

**职责**：当 `infer=False` 时，跳过所有 LLM 步骤，逐条 message embed + 写入向量库。

**输入**：`messages`, `metadata`

**行为**：
1. 遍历 messages，跳过 role="system" 的消息
2. 对每条消息：
   - 构建 per-message metadata：copy metadata + role + actor_id（若有 name 字段）
   - embed content → 生成 UUID → `vector_store.insert`
   - 记录到 `db.add_history`
   - 加入 results 列表：`{"id": uuid, "memory": content, "event": "ADD", "actor_id": ..., "role": ...}`
3. **不写入图**

**输出**：
```python
{
    "results": [{"id": "uuid", "memory": "...", "event": "ADD", ...}, ...]
}
```

**边界**：
- 不做图写入（infer=False 是快速路径，不触发图操作）
- 复用已有的 `Memory._create_memory` 逻辑
- 空 content、无效格式的 message 跳过并 log warning

---

### 3.3 extract_queries — 提取检索查询

**职责**：用 LLM 从原始对话中提取用于检索已有记忆的 query 列表。

**输入**：`parsed_messages`

**行为**：
1. 构造 prompt，要求 LLM 分析对话内容，提取 N 个独立 query，用于从记忆库中检索相关的已有记忆
2. prompt 要点：
   - 告诉 LLM 这段对话涉及的主题、实体、事实是什么
   - 要求生成简洁、语义完整的 query 字符串
   - 不要生成过于泛化的 query（如 "用户偏好"）
3. 调用 `llm.generate_response(response_format={"type": "json_object"})`
4. 解析返回 JSON：`{"queries": ["query1", "query2", ...]}`
5. 若 LLM 返回空或解析失败 → 回退为 `[parsed_messages]` 整体作为单个 query

**输出**：
```python
{
    "search_queries": ["用户喜欢什么食物", "用户住在哪里", ...]
}
```

**Prompt 设计方向**（写死在代码中）：
```
你是一个记忆检索助手。给定一段对话，生成用于搜索已有记忆的查询。
- 提取对话中涉及的所有关键主题、实体和个人信息
- 每个查询应独立、语义完整
- 若对话内容少或无法提取有效查询，返回空列表
- 返回 JSON 格式 {"queries": ["query1", "query2"]}
```

**边界**：
- 不做 fact 提取 — fact 提取是旧设计，现在只生成 search query
- query 数量不设上限，由 LLM 自行判断
- 返回空列表时，回退到用 `parsed_messages` 整体做搜索

---

### 3.4 search — 召回已有记忆

**职责**：用 extract_queries 的结果调用 SearchEngine 获取已有记忆。

**输入**：`search_queries`, `filters`

**行为**：
1. 对每个 query 调用 `search_engine.search(query, filters, limit=10, threshold=None, graph_depth=2, rerank=True)`
2. 合并多 query 结果：
   - **向量结果**：按 `id` 去重，保留 score 最高的
   - **图关系**：按 `(source, relationship, destination)` 去重
3. 将合并结果存入 `recalled_memories`

**输出**：
```python
{
    "recalled_memories": {
        "results": [MemoryItem, ...],     # 去重后的向量召回
        "relations": [{"source": "...", "relationship": "...", "destination": "..."}, ...]  # 去重后的图关系
    }
}
```

**边界**：
- SearchEngine.search() 参数：`limit=10`（比外部 search 默认 100 小，控制上下文长度）、`graph_depth=2`、`rerank=True`
- 若 `search_queries` 为空 → recalled_memories 为空

---

### 3.5 decide_memory — 记忆决策

**职责**：一次 LLM 调用，基于原始对话 + 已召回的记忆 → 决定每条信息 ADD / UPDATE / DELETE / NONE。

**输入**：`parsed_messages`, `recalled_memories["results"]`

**行为**：
1. 将 `recalled_memories["results"]` 格式化为 LLM 可读的列表：
   ```
   [
     {"id": "real-uuid-1", "text": "用户喜欢披萨", "score": 0.85},
     {"id": "real-uuid-2", "text": "用户住在纽约", "score": 0.72},
   ]
   ```
   使用真实 UUID，不做 temp mapping
2. 构造 prompt（改造现有 `get_update_memory_messages` / `DEFAULT_UPDATE_MEMORY_PROMPT`）：
   - 输入原始对话文本（不是提取后的 facts！）
   - 输入已召回的旧记忆列表（含真实 ID）
   - 要求 LLM 分析对话中蕴含的事实信息，对比旧记忆，决定 ADD/UPDATE/DELETE/NONE
3. 调用 `llm.generate_response(response_format={"type": "json_object"})`
4. 解析返回 JSON：
   ```json
   {
     "memory": [
       {"id": "real-uuid-or-null", "text": "内容", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "旧内容(仅UPDATE需要)"}
     ]
   }
   ```
5. 校验：event 必须在 {ADD, UPDATE, DELETE, NONE} 内；ADD 的 id 为 null；UPDATE/DELETE/NONE 的 id 必须在 recalled results 中存在

**输出**：
```python
{
    "decisions": [
        {"id": None, "text": "新事实", "event": "ADD"},
        {"id": "uuid-1", "text": "更新的内容", "event": "UPDATE", "old_memory": "旧内容"},
        {"id": "uuid-2", "text": "", "event": "DELETE"},
        {"id": "uuid-3", "text": "", "event": "NONE"},
    ]
}
```

**边界**：
- 不涉及图操作 — 只做向量记忆决策
- 真实 UUID，无 temp mapping hack
- 若 `recalled_memories["results"]` 为空 → 全部新事实标记为 ADD
- 若 LLM 返回空或 JSON 解析失败 → decisions 为空，log error
- ADD 操作的 id 字段为 None，由 execute_vector 分配 UUID

---

### 3.6 execute_vector — 执行向量操作

**职责**：按 decisions 执行向量库的 ADD / UPDATE / DELETE / NONE。

**输入**：`decisions`, `metadata`

**行为**：

| event | 操作 |
|-------|------|
| ADD | embed text → 生成 UUID → `vector_store.insert(payload=metadata + data + hash + created_at)` → `db.add_history("ADD")` |
| UPDATE | embed text → `vector_store.update(vector_id=id, payload=merge(metadata, data, updated_at))` → `db.add_history("UPDATE")` |
| DELETE | `vector_store.delete(vector_id=id)` → `db.add_history("DELETE")` |
| NONE | 跳过（不更新 session IDs，简化处理） |

对每条 decision 构造 result 条目：
```python
# ADD
{"id": new_uuid, "memory": text, "event": "ADD"}

# UPDATE
{"id": id, "memory": text, "event": "UPDATE", "previous_memory": old_memory}

# DELETE
{"id": id, "memory": "", "event": "DELETE"}

# NONE
# 不加入 results
```

**输出**：
```python
{
    "results": [{"id": "...", "memory": "...", "event": "ADD|UPDATE|DELETE", ...}]
}
```

**边界**：
- 复用 `Memory._create_memory`、`Memory._update_memory`、`Memory._delete_memory` 的核心逻辑（embed + vector_store + history）
- 不做 NONE 的 session_id 更新（简化，旧实现会更新）
- 失败只 log error，不回滚，不重试

---

### 3.7 extract_graph — 提取图实体与关系

**职责**：一次 LLM 调用，同时完成实体提取、关系提取、删除决策，为 graph.ingest() 准备数据。

**输入**：`parsed_messages`, `filters`, `recalled_memories["relations"]`

**行为**：

1. 构造 system prompt，包含三项任务的说明：
   - 从对话中提取实体及其类型
   - 提取实体间的关系
   - 基于已存在的图关系，判断哪些旧关系需要删除
2. 将 `recalled_memories["relations"]` 格式化为文本嵌入 prompt：
   ```
   已存在的图关系：
   alice -- lives_in -- new_york
   bob -- works_with -- alice
   ```
   若为空则注明"无已存在关系"
3. 提供三个 tool 给 LLM（一次调用中可依次调用多个 tool）：
   - `extract_entities` — `EXTRACT_ENTITIES_TOOL` / `EXTRACT_ENTITIES_STRUCT_TOOL`
   - `establish_relations` — `RELATIONS_TOOL` / `RELATIONS_STRUCT_TOOL`
   - `delete_graph_memory` — `DELETE_MEMORY_TOOL_GRAPH` / `DELETE_MEMORY_STRUCT_TOOL_GRAPH`
4. 根据 llm provider 选择 structured 或普通 tool 格式
5. 调用 `llm.generate_response(messages=[system, user], tools=[...])`
6. 解析 tool_calls 返回值：
   - 从 `extract_entities` tool call → 构建 `entity_type_map`
   - 从 `establish_relations` tool call → 构建 `relations`
   - 从 `delete_graph_memory` tool call → 构建 `to_be_deleted`
7. 对所有字段做规范化：小写 + 空格转下划线

**输出**：
```python
{
    "entity_type_map": {"alice": "person", "new_york": "city"},
    "relations": [{"source": "alice", "relationship": "lives_in", "destination": "new_york"}],
    "to_be_deleted": [{"source": "alice", "relationship": "lives_in", "destination": "boston"}]
}
```

**边界**：
- 复用已有 `graph_memory.py` 中的 tools（`EXTRACT_ENTITIES_TOOL`, `RELATIONS_TOOL`, `DELETE_MEMORY_TOOL_GRAPH` 等），不新建 tool
- 一次网络往返完成三项任务，LLM 在上下文中同时看到原始对话和已存在关系，决策更准确
- 若 LLM 未调用某个 tool（例如无需删除时没有 `delete_graph_memory` 调用）→ 对应字段为空列表
- 若 LLM 返回空或解析失败 → 三个字段均为空，log error，不影响向量侧结果
- 提取失败不阻塞 pipeline（图写入是增强功能）

---

### 3.8 execute_graph — 执行图写入

**职责**：调用 `graph.ingest()` 执行图写入。

**输入**：`entity_type_map`, `relations`, `filters`, `to_be_deleted`

**行为**：
1. 若 `relations` 和 `to_be_deleted` 都为空 → 跳过，返回空 dict
2. 调用 `graph.ingest(entity_type_map, relations, filters, to_be_deleted)`
3. 返回结果

**输出**：
```python
{
    "graph_result": {"deleted_entities": [...], "added_entities": [...]}
}
```

**边界**：
- 使用 `graph.ingest()` 而非旧 `graph.add()`（不涉及 LLM 和 embedding 搜索）
- 失败只 log error，不回滚，不重试
- 确保传入的参数已经过 extract_graph 的规范化

---

### 3.9 assemble_result — 组装返回

**职责**：合并各节点产出为最终返回格式。

**输入**：`results`, `recalled_memories`, `graph_result`（可选）

**行为**：
1. 构建 response dict：
   - 始终包含 `"results"` 和 `"recalled_memories"`
   - 若有 `graph_result` 且有内容 → 添加 `"relations"` 字段
2. 若 `error` 非空 → 附加到返回结构中

**输出**：
```python
{
    "final_results": {
        "results": [...],
        "recalled_memories": {"results": [...], "relations": [...]},
        "relations": {"deleted_entities": [...], "added_entities": [...]}  # 可选
    }
}
```

**边界**：
- 纯数据组装，无副作用
- recalled_memories 直接透传 SearchEngine 的返回，不做二次处理

---

## 四、未纳入本次实现的功能

| 功能 | 状态 | 说明 |
|------|------|------|
| `memory_type="procedural_memory"` | ❌ TODO | 流程记忆分支暂不实现，调用时若传入则按 infer=False 路径处理或 log warning |
| `add_recall_limit` 配置项 | ❌ TODO | Add 调用 Search 时的 limit 硬编码为 10，后续配置化 |
| 图/向量写入失败回滚 | ❌ 不做 | 仅 log error，不考虑重试和回滚 |
| Code Agent Summary 双写 | ❌ TODO | Summary 的 Step+DEPENDS_ON 图写入 + Brief 向量写入未实现 |
| AsyncMemory 改造 | ❌ TODO | 本次仅实现同步 Memory，AsyncMemory 后续参照改造 |
