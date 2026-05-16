# Memory 模块 LangGraph 重构 — 目录结构与分层设计

## 重构范围

**仅拆 `mem0/memory/main.py`，其余现有文件一律不大改，只会按需更改。**

- 不重构 `graph_memory.py`、`memgraph_memory.py`、`kuzu_memory.py`、`apache_age_memory.py`
- 理论上不改 `utils/factory.py`
- 理论上不改 `vector_stores/*`、`llms/*`、`embeddings/*`、`reranker/*`、`graphs/*`
- 理论上不改 `memory/base.py`、`memory/storage.py`、`memory/telemetry.py`、`memory/utils.py`、`memory/setup.py`

---

## 目标目录结构

从**任务**出发：本次重构只做两件事——**Search 统一召回** 和 **Add 写入（含 Summary）**。因此只需要两个 Engine 文件，各自把 LangGraph 状态机收在内部。

```
mem0/memory/
├── main.py                  # Memory + AsyncMemory：接口层、初始化、委托
├── base.py                  # MemoryBase（已有，不动）
├── storage.py               # SQLiteManager（已有，不动）
├── telemetry.py             # 遥测（已有，不动）
├── utils.py                 # 通用工具（已有，不动）
├── graph_memory.py          # 已有
├── memgraph_memory.py       # 已有
├── kuzu_memory.py           # 已有
├── apache_age_memory.py     # 已有
├── setup.py                 # 已有
│
├── search_engine.py         # 【新增】Search 统一召回引擎（含 LangGraph 状态机）
└── add_engine.py            # 【新增】Add 写入引擎（标准 + 流程 + Summary，含 LangGraph 状态机）
```

---

## 各文件职责

### main.py（瘦身）

职责：对外暴露接口，参数校验，初始化两个 Engine，委托调用。

保留内容：
- `Memory` / `AsyncMemory` 类定义
- `__init__` 中的组件初始化（embedding_model, vector_store, llm, graph, db, reranker）
- `search()` / `add()` / `get()` / `get_all()` / `update()` / `delete()` / `history()` / `reset()` 方法签名
- `search()` 和 `add()` 改为委托给 Engine，自身只保留参数校验和结果组装

其余内容视情况保留，最后删除不需要的方法

### search_engine.py

职责：**Search 统一召回引擎**。向量召回 + 图召回 + 合并去重 + rerank + 格式化。LangGraph 状态机收在本文件内部，对外暴露一个简单的 `search()` 方法。

内部 LangGraph 状态机：
- 节点：embed → vector_search → graph_search → merge_dedup → rerank
- 条件：graph_depth > 0 时走 graph_search，否则跳过；rerank=True 时走 rerank，否则结束
- 对外无感知，编译后的图在 Engine 初始化时生成

### add_engine.py

职责：**Add 写入引擎**。标准记忆、流程记忆、Code Agent Summary 均走 Add 流程的不同分支。LangGraph 状态机收在本文件内部，对外暴露一个简单的 `add()` 方法。

内部 LangGraph 状态机：
- 节点：preprocess → recall_memories → extract_content → decide_actions → execute_vector → execute_graph → assemble_result
- 条件分支：
  - infer=False → 直接写入分支（跳过提取和决策）
  - summaries 存在 → Summary 双写分支（Brief → 向量库，Step/DEPENDS_ON → 图库）
  - memory_type="procedural_memory" → 流程记忆提取节点（chunk + 总结）
  - 默认 → 标准记忆提取节点（LLM 提取 facts）

关键点：流程记忆不再走独立旁路。`memory_type="procedural_memory"` 只影响"提取内容"的方式，不影响整体"召回 → 决策 → 执行"框架。Summary 处理是 Add 的一个子分支，不单独拆文件。

---

## main.py 与 Engine 的调用关系

```
Memory.search()
    └── search_engine.search()
            ├── embedding_model.embed()
            ├── vector_store.search()
            ├── graph.search() 
            └── reranker.rerank() (可选)

Memory.add()
    └── add_engine.add()  (内部 LangGraph 编排)
            ├── search_engine.search()  (召回已有记忆)
            ├── llm.generate_response()  (提取 / 决策)
            ├── vector_store.insert/update/delete()
            └── graph.add()
```

---

## 现有文件改动说明

| 文件 | 改动 |
|------|------|
| `memory/main.py` | 大改。删除 `_search_vector_store`、`_add_to_vector_store`、`_add_to_graph`、`_create_procedural_memory` 等方法的具体逻辑，改为初始化 Engine 并委托调用。保留接口签名、参数校验、底层存储操作方法、工具函数。 |
| `memory/search_engine.py` | 新增。含 Search 召回逻辑 + 内部 LangGraph 状态机。 |
| `memory/add_engine.py` | 新增。含 Add 写入逻辑（标准 + 流程 + Summary）+ 内部 LangGraph 状态机。 |
| `memory/base.py` | 理论上不动 |
| `memory/storage.py` | 理论上不动 |
| `memory/telemetry.py` | 理论上不动 |
| `memory/utils.py` | 理论上不动 |
| `memory/graph_memory.py` | 视需求改动 |
| `memory/memgraph_memory.py` | 理论上不动 |
| `memory/kuzu_memory.py` | 理论上不动 |
| `memory/apache_age_memory.py` | 理论上不动 |
| `memory/setup.py` | 理论上不动 |
| `utils/factory.py` | 理论上不动 |
| `vector_stores/*` | 视需求改动 |
| `llms/*` | 理论上不动 |
| `embeddings/*` | 理论上不动 |
| `reranker/*` | 理论上不动 |
| `graphs/*` | 理论上不动 |

---

## 关键设计决策

1. **Search 是 Add 的依赖**：Add Engine 内部调用 Search Engine，Search Engine 不感知 Add。
2. **recalled_memories 必须随 Add 返回**：Add Engine 的返回结构中包含 `recalled_memories` 字段。
3. **流程记忆融入标准分支**：`memory_type` 只影响"提取内容"方式，不影响整体"召回 → 决策 → 执行"框架。
4. **Summary 是 Add 的子分支**：Code Agent Summary 的双写逻辑在 Add Engine 内处理，不单独拆文件。
5. **LangGraph 状态机收在 Engine 内部**：不单独拆 langgraph_ops 文件，状态和节点与 Engine 放在一起，对外只暴露简单的 `search()` / `add()` 方法。
6. **现有 GraphStore 接口不变**：`graph.search()` 和 `graph.add()` 的调用方式保持现有签名，不改内部实现。
