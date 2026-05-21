# Add Engine 实现文档

## 一、概述

Add Engine 是 mem0 Memory 写入层的 LangGraph 编排引擎。它将原先 `main.py` 中 `add()` 的内联逻辑拆分为独立、可测试的节点，通过 LangGraph 状态机编排，实现 **Search 是 Add 的依赖**、**recalled_memories 随 Add 返回**、**图写入与向量写入分离** 三大设计目标。

### 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `mem0/memory/add_engine.py` | **新增** | Add Engine 主体，含 AddState + LangGraph + 核心方法 |
| `mem0/memory/main.py` | **修改** | `Memory.add()` 委托给 AddEngine；删除 `_add_to_vector_store`、`_add_to_graph` |
| `tests/memory/test_add_engine.py` | **新增** | 85 个单元测试（每个节点独立 mock） |
| `tests/memory/test_add_engine_e2e.py` | **新增** | 17 个 E2E 测试（含直接写入、LLM 推理、图写入、Memory 委托） |

---

## 二、架构设计

### 2.1 LangGraph 图结构

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

共 **9 个节点**，**2 个条件分支**。

### 2.2 调用关系

```
Memory.add()
  ├── 参数校验（user_id/agent_id/run_id 至少一个）
  ├── messages 归一化（str → [dict], dict → [dict]）
  ├── vision 消息解析
  ├── _build_filters_and_metadata()
  └── add_engine.add()                    ← LangGraph 编排
        ├── extract_queries (LLM)         ← 提取检索 query
        ├── search (SearchEngine)         ← 统一召回已有记忆
        ├── decide_memory (LLM)           ← 用原始对话决策 ADD/UPDATE/DELETE/NONE
        ├── execute_vector                ← 操作向量库 + history
        ├── extract_graph (LLM)           ← 提取实体 + 关系 + 删除决策
        ├── execute_graph (graph.ingest)  ← 纯 MERGE，无 LLM
        └── assemble_result               ← 组装 results + recalled_memories + relations
```

### 2.3 新旧流程对比

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

关键改进：
- **不再逐 fact 做 vector search**（消除 N+1 问题），改为 LLM 提取 query 后统一调用 SearchEngine
- **不再有 temp UUID mapping hack**，使用真实 UUID
- **图写入分离**：LLM 实体/关系提取上提到 extract_graph 节点，execute_graph 只做纯 Cypher MERGE
- **输入是原始对话文本**，不再先提取 facts 再决策

---

## 三、节点详细说明

### 3.1 preprocess（预处理）

**职责**：拼接 messages 文本。

**行为**：
- 遍历 messages，跳过 `role="system"` 的消息
- 将 `user` 和 `assistant` 消息格式化为 `"user: content\nassistant: content\n"` → `parsed_messages`

**边界**：不做参数校验（上游 Memory.add() 已完成），不做 messages 类型归一化。

### 3.2 direct_add（直接写入，infer=False）

**职责**：跳过所有 LLM 步骤，逐条 message embed + 写入向量库。

**行为**：
1. 遍历 messages，跳过 system 角色和空 content
2. 每条的 metadata 复制 + 补充 role 和 actor_id（若 name 字段存在）
3. embed → 生成 UUID → `vector_store.insert` → `db.add_history`
4. 返回 `[{"id": uuid, "memory": content, "event": "ADD", "actor_id": ..., "role": ...}]`

**关键**：不写入图，不做 LLM 调用。

### 3.3 extract_queries（提取检索查询）

**职责**：用 LLM 从对话中提取用于检索已有记忆的 query 列表。

**行为**：
1. 构造 prompt，要求 LLM 提取 N 个独立 query
2. 调用 `llm.generate_response(response_format={"type": "json_object"})`
3. 解析 `{"queries": ["query1", "query2", ...]}`
4. 若 LLM 返回空或解析失败 → 回退为 `[parsed_messages]` 整体作 query

### 3.4 search（召回已有记忆）

**职责**：调用 SearchEngine 获取已有记忆。

**行为**：
1. 对每个 query 调用 `search_engine.search(query, filters, limit=10, graph_depth=2, rerank=True)`
2. 合并多 query 结果：向量结果按 `id` 去重保留最高 score；图关系按 `(source, relationship, destination)` 去重

### 3.5 decide_memory（记忆决策）

**职责**：一次 LLM 调用，基于**原始对话文本** + 已召回记忆 → 决定 ADD/UPDATE/DELETE/NONE。

**关键设计决策**：
- 输入是 `parsed_messages`（原始对话），不是提取后的 facts
- 使用真实 UUID，不做 temp mapping
- 校验：ADD 的 id 强制为 None；UPDATE/DELETE/NONE 的 id 必须在 recalled results 中存在
- 若 recalled memories 为空 → 所有新事实标记为 ADD

### 3.6 execute_vector（执行向量操作）

**职责**：按 decisions 执行向量库操作。

| event | 操作 |
|-------|------|
| ADD | embed text → 生成 UUID → `vector_store.insert` → `db.add_history("ADD")` |
| UPDATE | embed text → `vector_store.update` → `db.add_history("UPDATE")` |
| DELETE | `vector_store.delete` → `db.add_history("DELETE")` |
| NONE | 跳过 |

失败只 log error，不回滚，不重试。

### 3.7 extract_graph（提取图实体与关系）

**职责**：一次 LLM 调用，返回结构化 JSON 包含 entities、relations、to_be_deleted。

**实现细节**：
- 使用 `response_format={"type": "json_object"}`（非 tool calling），因为 DeepSeek 的并行 tool calling 不可靠
- 将 `recalled_memories["relations"]` 格式化为文本嵌入 prompt
- LLM 返回：
  ```json
  {
    "entities": [{"entity": "Alice", "entity_type": "person"}],
    "relations": [{"source": "Alice", "relationship": "lives_in", "destination": "NYC"}],
    "to_be_deleted": [{"source": "Alice", "relationship": "lives_in", "destination": "Boston"}]
  }
  ```
- 所有字段规范化：小写 + 空格转下划线

### 3.8 execute_graph（执行图写入）

**职责**：调用 `graph.ingest()` 执行图写入。

**行为**：
1. 若 `relations` 和 `to_be_deleted` 都为空 → 跳过
2. 调用 `graph.ingest(entity_type_map, relations, filters, to_be_deleted)`
3. `ingest()` 只做纯 Cypher MERGE，不涉及 LLM 和 embedding 搜索

### 3.9 assemble_result（组装返回）

**职责**：合并各节点产出为最终返回结构。

```python
# 无图
{
    "results": [{"id": "uuid", "memory": "text", "event": "ADD"}],
    "recalled_memories": {"results": [MemoryItem, ...], "relations": [...]}
}

# 有图（graph_result 非空）
{
    "results": [...],
    "recalled_memories": {"results": [...], "relations": [...]},
    "relations": {"deleted_entities": [...], "added_entities": [...]}
}
```

---

## 四、main.py 修改

### 4.1 Memory.__init__ 新增 AddEngine 初始化

```python
from mem0.memory.add_engine import AddEngine
self.add_engine = AddEngine(
    embedding_model=self.embedding_model,
    vector_store=self.vector_store,
    llm=self.llm,
    db=self.db,
    search_engine=self.search_engine,
    graph=self.graph if self.enable_graph else None,
)
```

### 4.2 Memory.add() 委托方式

```python
def add(self, messages, *, user_id=None, agent_id=None, run_id=None, ...):
    # 参数校验（保留在 Memory 层）
    processed_metadata, effective_filters = _build_filters_and_metadata(...)

    # messages 归一化（保留在 Memory 层）
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # procedural_memory 暂走旧路径（TODO）
    if agent_id and memory_type == "procedural_memory":
        return self._create_procedural_memory(...)

    # vision 解析（保留在 Memory 层）
    messages = parse_vision_messages(messages, ...)

    # 委托给 AddEngine
    result = self.add_engine.add(
        messages=messages,
        metadata=processed_metadata,
        filters=effective_filters,
        infer=infer,
    )

    # telemetry（保留在 Memory 层）
    capture_event("mem0.add", self, {...})
    return result
```

### 4.3 删除的旧方法

- `Memory._add_to_vector_store` — 已移除（AsyncMemory 保留自身副本）
- `Memory._add_to_graph` — 已移除（AsyncMemory 保留自身副本）

---

## 五、AddState 定义

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
    decisions: list[dict]            # [{"id": str|null, "text": str, "event": str, "old_memory": str}]
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

## 六、测试覆盖

### 6.1 单元测试（85 个）

所有测试使用 `unittest.mock.MagicMock` 模拟依赖，每个核心方法独立可测。

| 测试类 | 测试数 | 覆盖范围 |
|--------|--------|---------|
| `TestAddEngineInit` | 3 | 初始化、图启用/不启用、LangGraph 编译 |
| `TestConditionalEdges` | 4 | infer 分支、graph 分支 |
| `TestPreprocessMessages` | 6 | user/assistant 拼接、system 跳过、空消息、未知 role |
| `TestNodePreprocess` | 1 | LangGraph 节点包装 |
| `TestDirectAddMessages` | 6 | 单条/多条消息、system 跳过、actor name、空 content、无效格式 |
| `TestNodeDirectAdd` | 1 | 节点包装 |
| `TestExtractSearchQueries` | 8 | 正常提取、空输入、空响应、JSON 解析失败、LLM 异常、空 query 过滤、prompt 验证 |
| `TestNodeExtractQueries` | 1 | 节点包装 |
| `TestSearchMemories` | 7 | 单 query、多 query 合并、向量去重、关系去重、搜索异常、空 queries、无 id 跳过 |
| `TestNodeSearch` | 1 | 节点包装 |
| `TestDecideMemoryActions` | 8 | 无已有记忆 ADD、有已有记忆 ADD+NONE、UPDATE、DELETE、无效 event 过滤、ADD id 强制 null、无效 id 转 NONE、空响应、LLM 异常、prompt 验证 |
| `TestNodeDecideMemory` | 1 | 节点包装 |
| `TestExecuteVectorOperations` | 6 | ADD、UPDATE、DELETE、NONE 跳过、多操作、异常 log |
| `TestNodeExecuteVector` | 1 | 节点包装 |
| `TestExtractGraphData` | 7 | entities+relations、带删除、空已有关系、LLM 异常、JSON 解析异常、归一化、user_id 在 prompt |
| `TestNodeExtractGraph` | 1 | 节点包装 |
| `TestExecuteGraphWrite` | 4 | 正常调用 ingest、空跳过、带删除、异常容错 |
| `TestNodeExecuteGraph` | 1 | 节点包装 |
| `TestAssembleFinalResult` | 5 | 最小化、带向量结果、带图结果、空图结果省略、None 省略 |
| `TestNodeAssembleResult` | 2 | 无图、有图 |
| `TestCreateMemory` | 2 | 正常创建、embed 调用 |
| `TestUpdateMemory` | 2 | 正常更新、找不到抛异常 |
| `TestDeleteMemory` | 2 | 正常删除、找不到抛异常 |
| `TestAddPublicApi` | 3 | infer=True 全流程、infer=False、图启用 |

### 6.2 E2E 测试（17 个）

使用真实基础设施：PostgreSQL + pgvector、Neo4j、DeepSeek LLM (deepseek-chat)、DashScope Embedding (text-embedding-v4)。

#### TestE2EDirectAdd（4 个）— infer=False

| 测试 | 覆盖 |
|------|------|
| `test_direct_add_single_message` | 单条 user 消息直接写入，验证 event=ADD、role=user、UUID 存在 |
| `test_direct_add_skips_system` | system 消息被跳过 |
| `test_direct_add_multiple_messages` | user→assistant→user 三轮全存 |
| `test_direct_add_empty_content_skipped` | 纯空格 content 被跳过 |

#### TestE2EInferAdd（5 个）— infer=True

| 测试 | 覆盖 |
|------|------|
| `test_infer_add_new_fact` | 全新用户提取事实 + ADD |
| `test_infer_recalls_previous_memories` | **两次 add，第二次的 recalled_memories 包含第一条记忆** |
| `test_infer_update_existing` | 同 topic 补充细节，触发 ADD 或 UPDATE |
| `test_infer_empty_conversation` | 纯打招呼 "Hi"，不崩溃 |
| `test_infer_result_structure` | 返回结构完全符合设计规格 |

#### TestE2EMemoryAddDelegation（2 个）— Memory.add() 全链路

| 测试 | 覆盖 |
|------|------|
| `test_memory_add_delegation` | 通过 `Memory.add(user_id=.., infer=True)` 调用，验证参数校验 + 委托全链路 |
| `test_memory_add_direct` | `Memory.add(infer=False)` 走 direct_add 路径 |

#### TestE2EGraphWrite（6 个）— 图写入路径

| 测试 | 覆盖 |
|------|------|
| `test_add_with_graph_creates_relations` | 实体关系消息 → extract_graph → execute_graph → relations 在返回中 |
| `test_add_with_graph_no_new_entities` | "Hi!" 无实体，不崩溃 |
| `test_add_graph_entity_dedup` | 同一实体两次 add，MERGE 语义不重复 |
| `test_add_graph_deletes_old_relation` | Alice 从 works_at Meta → Google，旧关系被标记删除 |
| `test_add_graph_result_structure` | added_entities 中每项含 source/relationship/target |
| `test_memory_add_with_graph_delegation` | Memory 全链路含 graph_store 配置，输出含 relations |

---

## 七、运行方法

### 7.1 单元测试（不需要外部服务）

```bash
conda run -n mem0 pytest tests/memory/test_add_engine.py -v
conda run -n mem0 pytest tests/memory/test_search_engine.py tests/memory/test_add_engine.py -v
```

### 7.2 E2E 测试（需要 Docker + 环境变量）

```bash
# 1. 启动数据库服务
cd server/
docker compose up -d

# 2. 加载环境变量
set -a && source server/.env && set +a

# 3. 运行全部 E2E 测试
conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py -v

# 4. 分类运行
conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py::TestE2EDirectAdd -v
conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py::TestE2EInferAdd -v
conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py::TestE2EMemoryAddDelegation -v
conda run -n mem0 pytest tests/memory/test_add_engine_e2e.py::TestE2EGraphWrite -v
```

### 7.3 依赖服务端口

| 服务 | 宿主机端口 | 用途 |
|------|-----------|------|
| PostgreSQL + pgvector | 8432 | 向量数据库 |
| Neo4j Bolt | 8687 | 图数据库 |
| Neo4j HTTP | 8474 | Neo4j 浏览器 |

---

## 八、关键设计决策

| 决策 | 说明 |
|------|------|
| **Search 是 Add 的依赖** | AddEngine 内部调用 `search_engine.search()`，SearchEngine 不感知 Add |
| **recalled_memories 必须返回** | Add 返回结构中始终包含 `recalled_memories`，让调用方感知"联想" |
| **真实 UUID，无 temp mapping** | decide_memory 使用 recalled results 的真实 UUID，不再做整数映射 |
| **输入原始对话，不先提取 facts** | decide_memory 直接接收 `parsed_messages`（原始对话文本），而非先从 LLM 提取 facts 再决策 |
| **图写入用 JSON 非 tool calling** | extract_graph 使用 `response_format={"type": "json_object"}`，因为 DeepSeek 的并行 tool calling 不稳定 |
| **graph.ingest() 替代 graph.add()** | execute_graph 调用 `graph.ingest()`（纯 Cypher MERGE，无 LLM），与旧 `graph.add()`（内含 4 次 LLM 调用）完全不同 |
| **参数校验/归一化/telemetry 保留在 Memory 层** | AddEngine 只处理核心写入逻辑，输入已由 Memory.add() 预处理 |

---

## 九、未纳入本次实现

| 功能 | 状态 | 说明 |
|------|------|------|
| `memory_type="procedural_memory"`、Code Agent Summary 双写 | ❌ TODO | Memory.add() 中暂走旧的 `_create_procedural_memory` 旁路 |
| `add_recall_limit` 配置项 | ❌ TODO | Add 调用 Search 时的 limit 硬编码为 10 |
| 图/向量写入失败回滚 | ❌ 不做 | 仅 log error |
| AsyncMemory 改造 | ❌ 不做 | 仅实现同步 Memory，AsyncMemory 保留旧实现 |
