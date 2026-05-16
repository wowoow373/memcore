# Memory 模块 LangGraph 重构与功能增强计划

## 需求本质分析

抛开现有代码，从需求本身出发，核心诉求是：

1. **Search = 所有召回能力的统一出口**：无论是向量相似度召回还是图关系跨步召回，全部由 Search 负责
2. **Add = 写操作，但写之前必须先读**：Add 内部不再自己做任何检索，而是显式调用 Search 获取已有记忆，基于此做去重决策
3. **Add 返回时要带回召回记忆**：让调用方（外部 Agent）感知到"联想"了哪些相关记忆
4. **流程记忆融入标准分支**：流程记忆（procedural memory）也要先 Search 召回已有记忆，然后去重对比，不是独立旁路
5. **Code Agent Summary 双写**：Summary 里的 Dependency 关系写入图数据库，Brief 内容切片写入向量数据库

---

## Search 流程设计

### Search 的职责边界

Search 是**纯只读**的记忆召回接口，负责回答一个问题："给定一个 query，哪些已有记忆与之相关？"

**Search 不负责**：
- 任何写入操作（ADD/UPDATE/DELETE）
- 去重决策（这是 Add 的职责）
- LLM fact 提取

**Search 负责**：
- 从向量数据库召回语义相似的记忆（不区分记忆类型，procedural/semantic/episodic 一视同仁）
- 从图数据库按可配置深度进行跨步关系遍历召回
- 将多源召回结果合并去重后返回

### Search 的输入

```python
search(query: str, metadata: dict, limit: int, graph_depth: int, rerank: bool)
```

- `query`：查询文本（Add 调用时会将 messages 拼接为 query）
- `metadata`：作用域过滤（user_id / agent_id / run_id / 其他自定义 metadata），为日后扩展预留
- `limit`：返回数量上限
- `graph_depth`：图跨步检索深度（默认 2，0 表示不查图）
- `rerank`：是否启用重排序

### Search 的输出

```python
{
    "results": [MemoryItem, ...],   # 召回的记忆列表（多种排序结果，让下游决定）
    "relations": [...]               # 节点关联的完整关系（由 Add 写入时收敛，Search 直接返回）
}
```

- `MemoryItem`：包含与去重、存储相关的必要字段（如 id, content, memory_type, score 等），属于易修改结构，按需扩展
- `relations`：返回节点及其所有关联关系，不做收敛过滤

### Search 内部步骤

Search 内部用 LangGraph 编排，但对外暴露的是一个简单方法。内部可分为：

1. **向量召回**：embedding query → 向量库相似度搜索（召回所有类型的记忆）
2. **图跨步召回**：提取 query 中的实体 → 从实体出发沿关系链遍历 `graph_depth` 步 → 收集途经的节点和关系
3. **合并**：向量结果和图结果去重（同一记忆在两种召回中出现时合并）
4. **重排序**（可选）：reranker 对合并后的结果重排
5. **协同召回**可能会需要图数据和向量结果互相帮助召回

### 图跨步检索的设计

"跨步"指的是在图数据库中沿着关系边从一个节点跳到相邻节点，跳 `graph_depth` 次。

- `graph_depth=0`：不查图，只走向量召回
- `graph_depth=1`：查与 query 实体直接相连的节点
- `graph_depth=2`：查直接相连节点，以及这些节点的邻居
- 以此类推

这意味着图召回的结果质量取决于图中已经建立了什么关系。标准记忆通过实体-关系-实体建立图结构，Code Agent Summary 通过 Step-DEPENDS_ON-Step 建立链式结构。

---

## Add 流程设计

### Add 的职责边界

Add 是**写操作**接口，负责将新的信息持久化到记忆系统中，同时避免重复存储。

**Add 不负责**：
- 任何检索逻辑的具体实现（全部委托给 Search）
- 返回结果的 rerank（Search 已做）

**Add 负责**：
- 接收外部输入（messages + 可选的 summary）
- 调用 Search 召回相关已有记忆（**包括流程记忆**）
- 基于召回结果决定每条新信息是 ADD / UPDATE / DELETE / NONE
- 执行存储（向量 + 图）
- 返回操作结果 + 召回的记忆

### Add 的输入

```python
add(
    messages: list[dict],           # 统一输入字段：标准对话消息或 Code Agent summary 消息
    metadata: dict,                # 作用域信息（user_id / agent_id / run_id / 其他自定义字段）
    infer: bool,                   # 是否用 LLM 提取 facts / 生成总结
    memory_type: str,              # "procedural_memory" 或 None（决定如何解析 messages）
)
```

- `messages` 是**唯一输入字段**，不区分标准消息和 Code Agent summary
- 当 `memory_type == "procedural_memory"` 时，按流程记忆格式解析 messages；否则按标准对话格式解析

### Add 的输出

```python
{
    "results": [...],              # 实际执行的操作（ADD/UPDATE/DELETE）列表
    "recalled_memories": [...],    # Search 召回的相关记忆（关键新增）
    "relations": [...]             # 图存储结果（若启用）
}
```

`recalled_memories` 是核心新增字段，让外部 Agent 能感知到系统"联想"了哪些记忆。

### Add 内部步骤

1. **预处理**：验证输入、归一化 messages、构建 filters
2. **分支判断**：
   - 如果 `infer=False` → 直接存储分支（不走 LLM，每条 message 直接存）
   - 否则 → 标准分支（走下面的步骤）
3. **召回已有记忆**（标准分支）：将 messages 拼接为 query，调用 **Search** 召回相关记忆（**包含 procedural memories**）
4. **提取内容**（标准分支，infer=True）：
   - 如果 `memory_type != "procedural_memory"` → LLM 从 messages 中提取结构化 facts 列表
   - 如果 `memory_type == "procedural_memory"` → 对summary流程进行切片-总结等预处理
5. **决策**：
   - 如果 `memory_type != "procedural_memory"` → LLM 基于提取的 facts + 召回的已有记忆，决定每个 fact 是 ADD / UPDATE / DELETE / NONE
   - 如果 `memory_type == "procedural_memory"` → LLM 基于当前流程 + 召回的已有 procedural memories，决定是 ADD / UPDATE / DELETE /NONE
6. **执行**：按决策结果操作向量数据库
7. **图存储**：
   - 标准 messages：提取实体-关系 → 写入图数据库
   - Code Agent Summaries：每个 summary 先 search 图库查重 → 创建/更新 Step 节点 → 建立 DEPENDS_ON 关系
   - **Add 负责决定每条记忆在向量库和图库分别如何存储**，Search 只负责读取
8. **组装返回**：收集所有结果 + 步骤 3 的 recalled_memories → 返回

### 流程记忆融入标准分支的关键点

**为什么流程记忆也要走标准分支**：
- 流程记忆本质上也是一种记忆，存储时同样需要避免重复
- 一个 agent 多次执行相似任务时，流程记忆应该被 UPDATE 而非重复 ADD
- 因此流程记忆也需要先 Search 召回已有记忆，然后做去重对比

**流程记忆和标准记忆的区别**：

| 阶段 | 标准记忆 | 流程记忆 |
|------|---------|---------|
| 步骤 4 提取 | LLM 提取 facts 列表 | chunk 切片（将完整流程切成可召回的小步骤）+ LLM 生成流程总结文本 |
| 步骤 5 决策 | 每个 fact 独立决策 ADD/UPDATE/DELETE/NONE | 对总结和 chunks 分别决策 ADD/UPDATE/DELETE/NONE |
| 步骤 6 执行 | 每个 fact 独立操作 | 总结和每个 chunk 分别作为独立记忆操作 |

**流程记忆不再走独立旁路**：`memory_type="procedural_memory"` 只影响步骤 4 的"提取内容"方式，不影响整体流程框架。

### Add 调用 Search 的关键细节

Add 在步骤 3 调用 Search 时：
- query = messages 拼接后的文本（或提取的关键信息）
- metadata = 与 Add 相同的 metadata（至少包含 user_id/agent_id/run_id 等 filter 数据，确保只召回同作用域的记忆）
- limit = 由配置决定（比如 10）
- graph_depth = 由配置决定

Search 返回的 recalled_memories 会原封不动地放入 Add 的返回结果中。

---

## Code Agent Summary 处理设计

### Summary 的数据格式

外部 Code Agent 每一步输出：

```json
{
    "Goal": "整体任务目标",
    "Step": "01 - 步骤名称",
    "Action": "tool_call(params)",
    "Dependency": [
        {"step_id": "01", "description": "依赖描述"}
    ],
    "Brief": "本步骤的简短语义描述"
}
```

Add 接收的是多个这样的 summary 组成的列表（一个完整任务的多步执行记录）。

### Summary 的图数据库存储

**目标**：把步骤作为节点、依赖关系作为边存入图数据库，支持后续的跨步检索。

**存储逻辑**：
1. **Search 查重**：按语义相似度搜索图库，检查是否已存在相同步骤，或者先在向量数据库中语义相似度召回片段和整体流程的总结后获得唯一id去图数据中召回
2. **节点写入**：
   - 若不存在 → 创建 `(s:Step {step_id, goal, action, brief})`
   - 若存在 → 更新属性（或根据策略跳过/覆盖）
3. **关系写入**：对 `Dependency` 中的每个依赖项：
   - 查找 `(prev:Step {step_id: dependency.step_id})`
   - 创建 `(prev)-[:DEPENDS_ON]->(current)`

**为什么必须先 search 后 write**：
避免重复创建节点。Code Agent 可能会多次报告相同步骤（比如重试、增量更新），先 search 可以识别已有节点并更新而非重复创建。

### Summary 的向量数据库存储

**目标**：让每个步骤的 Brief 内容可被语义搜索召回。

**存储逻辑**（具体切片规则实现时讨论，此处定义接口）：
1. 对每个 summary，取 `Brief` 字段作为核心文本
2. 可选拼接 `Goal` 提供上下文
3. 生成 embedding 存入向量库
4. 元数据携带：`step_id`, `memory_type="step_summary"`, `user_id`, `agent_id`, `run_id`

**为什么 Brief 是切片单元**：
Brief 本身就是"简短语义化描述"，天然适合作为 embedding 的基本单元。

### Summary 存储与 Add 标准流程的关系

Code Agent Summary 的处理是 Add 流程的一个子分支：
- 当 `memory_type == "procedural_memory"` 时，messages 按 Code Agent summary 格式解析，触发 Summary 的图+向量双写



---

## LangGraph 的使用范围

### 为什么用 LangGraph

需求要求"用 LangGraph 编排"，其价值在于：
- 把 Add 和 Search 的复杂流程拆分为独立的、可观测的步骤
- 支持条件分支（infer=True/False, procedural/standard, summaries present/absent）
- 便于日后扩展（比如在两个步骤之间插入新步骤）

### LangGraph 编排什么

**Search 图**：编排召回流程内部的各步骤（向量召回 → 图召回 → 合并 → rerank）

**Add 图**：编排写入流程内部的各步骤（预处理 → 分支 → 召回 → 提取内容 → 决策 → 执行 → 图存储 → 组装）

### LangGraph 不编排什么

- 不编排异步流程（本次不涉及 AsyncMemory）
- 不把 Search 和 Add 编排在同一张大图里（Add 调用 Search 是方法调用关系，不是图内部的边关系）

---

## 配置设计

需要新增的配置项：

1. **`graph_search_depth`**：Search 的图跨步检索默认深度（默认 2）
2. **`add_recall_limit`**：Add 调用 Search 时的默认召回数量（默认 10）
3. **`enable_langgraph`**：LangGraph 编排开关（便于回退调试）

---

## 关键设计决策

1. **Search 是 Add 的依赖，不是反过来**：Add 调用 Search，Search 不感知 Add 的存在
2. **recalled_memories 必须随 Add 返回**：这是"add-with-search-back"的核心语义
3. **流程记忆融入标准分支**：`memory_type` 只影响"提取内容"的方式，不影响整体"召回→决策→执行"框架
4. **Summary 图存储必须先 search 后 write**：避免重复节点，支持增量更新
5. **图跨步深度在 Search 层配置**：Add 调用 Search 时透传，不在 Add 层重复定义
6. **向量召回高分作为图遍历起点**：Search 的协同召回策略（具体实现内部决定）
7. **Search 返回多种排序结果**：向量召回和图召回分别返回，不融合排序，让下游（Add 的决策步骤）自行决定如何使用
