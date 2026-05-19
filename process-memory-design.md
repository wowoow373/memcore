# Process Memory 整体设计

## 一、背景与目标

外部 Code Agent 每一步产出结构化记录（`summary`）：

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

一次完整任务从开始到 `action: "finish"` 产出一组 summary 数组，形成一条有依赖关系的步骤链（DAG）。

Process Memory 将执行经验持久化为三层颗粒度的记忆，在后续任务中辅助决策。

## 二、两层流程

### Flow 1：任务完成后 —— 写入

- **时机**：外部 Agent 完成任务，交付完整 summary 数组
- **操作**：Search（去重） → Add/Update
- **目的**：将完整执行流程沉淀为持久记忆。先 search 后 add，由 LLM 决定新增、更新、合并或跳过

### Flow 2：任务进行中 —— 仅检索

- **时机**：外部 Agent 执行某一步时
- **输入**：当前 step 的 summary + 前一步的 summary（用于语义再筛选）
- **操作**：仅 Search，不写
- **目的**：检索相关历史经验，辅助当前决策。中途不写入，半成品不污染记忆

## 三、三层记忆颗粒度

### 第一层：Graph（图谱）—— Step 级依赖

| 项目 | 说明 |
|------|------|
| **模型** | `(Step)-[:DEPENDS_ON]->(Step)` DAG |
| **粒度** | 细 —— 单步依赖 |
| **回答** | "做这步之前/之后需要做什么" |
| **写入** | 复用 `MemoryGraph.ingest()`，Step 作为实体类型、DEPENDS_ON 作为关系类型，`brief`/`goal`/`action`/`brief_embedding` 通过 `node_properties` 参数传入 |
| **Flow 1 搜索** | 先从 Chunk 召回中取已有 step 名称 → `MemoryGraph.search_nodes(step_names)` 沿 DEPENDS_ON 遍历获取完整链路 |
| **Flow 2 搜索** | Brief embedding → `MemoryGraph.search_nodes_by_embedding()` 语义匹配节点 → 1-hop 扩展 → 前一步 Brief 语义再筛选 |

### 第二层：Chunk（向量）—— Goal 级子目标

| 项目 | 说明 |
|------|------|
| **模型** | 同一 Goal 下的所有 steps 聚合为一个 chunk，以 Goal 文本做 embedding |
| **粒度** | 中 —— 子目标 |
| **回答** | "实现这个子目标需要怎么做" |
| **写入** | 向量存储，以 Goal 为 semantic key。同 Goal 去重时 merge |
| **Flow 1 搜索** | 按 Goal 向量召回相似 chunk（结果同时作为图搜索的 step 名称来源） |
| **Flow 2 搜索** | 按当前 step 的 Goal 做向量召回 |

### 第三层：Summary（向量）—— 完整链路总览

| 项目 | 说明 |
|------|------|
| **模型** | 整个完整流程的宏观摘要文本 + 完整 step 链路 payload |
| **粒度** | 粗 —— 完整任务 |
| **回答** | "这个任务整体上应该长什么样" |
| **写入** | 向量存储，task_description 做 embedding |
| **Flow 1 搜索** | 按任务宏观描述向量召回相似 summary |
| **Flow 2 搜索** | 根据已有 step 推断任务类型 → 向量检索 → 返回完整链路参考 |

## 四、数据流

### Flow 1

```
外部 Agent 产出完整 summary 数组
  → preprocess: 解析 summaries → steps / deps / goals / task_description
  → search_for_dedup(goals, task_desc, filters):
        Chunk: Goal 向量召回 → 获得已有 step 名称
        Graph: 用 Chunk 返回的 step 名称 → MemoryGraph.search_nodes() 遍历完整链路
        Summary: task_description 向量召回（独立，与 Chunk/Graph 可并行）
  → LLM decide: 新 summaries + 三层召回 → 三层各自 ADD/UPDATE/MERGE/NONE
  → execute: 三层独立并行写入
        Graph: MemoryGraph.ingest(entity_type_map, relations, filters, node_properties)
        Chunk: vector_store.insert/update
        Summary: vector_store.insert/update
  → assemble: 返回写入结果 + recalled
```

### Flow 2

```
外部 Agent 执行某一步
  → 输入: 当前 step summary + 前一步 summary
  → search_for_step(current_step, previous_step, filters):
        Graph: Brief embedding → MemoryGraph.search_nodes_by_embedding() 语义匹配
               → MemoryGraph.search_nodes() 1-hop 扩展
               → 前一步 Brief 语义再筛选（独立）
        Chunk: Goal 向量召回（独立）
        Summary: 任务推断 → 向量召回（独立）
  → 合并返回给外部 Agent
```

## 五、与标准记忆的隔离

| 维度 | 标准记忆 | Process Memory |
|------|---------|---------------|
| 图模型 | `(:Entity)-[:RELATION]->(:Entity)` | `(:Step)-[:DEPENDS_ON]->(:Step)` |
| 向量语义 | 事实粒度 | 任务/步骤粒度 |
| 写入引擎 | AddEngine | ProcessMemoryAddEngine |
| 检索引擎 | SearchEngine | ProcessMemorySearchEngine |
| 图存储 | MemoryGraph（Entity 标签） | 复用 MemoryGraph（Step 标签，同一 Neo4j 实例，标签隔离） |
| 向量存储 | 主 collection | 独立 collection（通过 memory_type metadata 隔离） |

## 六、关键设计决策

1. **Flow 1 / Flow 2 读写分离**：半成品不污染记忆
2. **前一步再筛选是语义的**：用前一步 Brief 的 embedding 与候选节点做相似度筛选，不是 ID 匹配
3. **去重决策由 LLM 统一处理**：和 AddEngine.decide_memory 模式一致 —— 召回 + 新输入 → LLM → ADD/UPDATE/MERGE/NONE
4. **三层写入完全独立**：无先后依赖，可并行执行
5. **图存储复用 MemoryGraph**：使用 `ingest()`、`search_nodes()` 等非 LLM 接口，通过 `node_label` 区分 Entity 节点和 Step 节点
6. **filters 直接注入**：`user_id`/`agent_id`/`run_id` 在入口处已确定，后续所有检索和写入都在同 ID 下进行
7. **Flow 1 图搜索依赖 Chunk 结果**：不新增 MemoryGraph 按属性查询的方法。先从 Chunk 向量召回获取已有 step 名称，再通过 `MemoryGraph.search_nodes(step_names)` 做精准图遍历
8. **Flow 2 图搜索新增 `search_nodes_by_embedding`**：接收外部预计算的 Brief embedding，在 Cypher 层做余弦相似度匹配。embedding 由 ProcessMemorySearchEngine 外部计算
