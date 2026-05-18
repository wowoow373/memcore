# MemoryGraph.ingest() —— 面向 Add-Engine 的图写入接口

## 概述

`ingest()` 是一个纯图写入接口，专为上层的 **add 引擎** 设计。
它接收 **预提取** 的结构化数据，并执行精确的 `MERGE` 操作，
不涉及任何 LLM 调用或嵌入相似性搜索。

它替代了已弃用的 `add()` 内部流水线
（`_retrieve_nodes_from_data` -> `_establish_nodes_relations_from_data` ->
`_search_graph_db` -> `_get_delete_entities_from_search_output` -> `_add_entities`）。

---

## 函数签名

```python
def ingest(
    self,
    entity_type_map: dict[str, str],
    relations: list[dict],
    filters: dict,
    to_be_deleted: list[dict] | None = None,
) -> dict:
```

---

## 参数

### `entity_type_map`

| 属性 | 类型 | 说明 |
|----------|------|-------------|
| 键 | `str` | 实体名称（例如 `"Alice"`、`"Bob"`）。这些名称将在内部被**规范化**为 `alice`、`bob`。 |
| 值 | `str` | 实体类型（例如 `"person"`、`"hobby"`）。这些类型也会被规范化为小写。 |

**`ingest()` 内部应用的规范化规则：**
- 键：`lower().replace(" ", "_")`
- 值：`lower().replace(" ", "_")`

**示例：**
```python
{"Alice": "Person", "New York": "City"}
# 内部变为 {"alice": "person", "new_york": "city"}
```

如果 `relations` 中引用的实体在此映射中缺失，则默认类型为 `"__User__"`。

---

### `relations`

关系字典列表，每一项描述一条要创建（或合并）的有向边。

| 键 | 类型 | 必填 | 说明 |
|-----|------|----------|-------------|
| `source` | `str` | 是 | 源实体名称。在内部规范化。 |
| `relationship` | `str` | 是 | 关系类型。规范化并清理为合法的 Cypher 格式。 |
| `destination` | `str` | 是 | 目标实体名称。在内部规范化。 |

**`ingest()` 内部应用的规范化规则：**
- `source`、`destination`：`lower().replace(" ", "_")`
- `relationship`：`lower().replace(" ", "_")`，然后执行 `sanitize_relationship_for_cypher()`

**示例：**
```python
[
    {"source": "Alice", "relationship": "lives_in", "destination": "New York"},
    {"source": "Bob", "relationship": "works_with", "destination": "Alice"},
]
```

---

### `filters`

范围过滤条件，用于标识图分区。至少需要提供 `user_id`。

| 键 | 类型 | 必填 | 说明 |
|-----|------|----------|-------------|
| `user_id` | `str` | **是** | 图分区的所有者。 |
| `agent_id` | `str` | 否 | 可选的代理范围。 |
| `run_id` | `str` | 否 | 可选的运行范围。 |

**示例：**
```python
{"user_id": "u1", "agent_id": "agent_a", "run_id": "run_42"}
```

这些过滤条件会嵌入到每一条 Cypher `MERGE` 中，以确保节点和边的正确作用域。

---

### `to_be_deleted`

可选的关系列表，用于在创建新关系**之前**删除。它替代了已弃用的基于 LLM 的删除步骤 `_get_delete_entities_from_search_output`。

模式与 `relations` 相同：

| 键 | 类型 | 必填 | 说明 |
|-----|------|----------|-------------|
| `source` | `str` | 是 | 要删除的关系的源实体名称。 |
| `relationship` | `str` | 是 | 要删除的关系类型。 |
| `destination` | `str` | 是 | 要删除的关系的目标实体名称。 |

**示例：**
```python
[
    {"source": "Alice", "relationship": "lives_in", "destination": "Boston"},
]
```

使用内部的 `_delete_entities()` 辅助方法，执行 `MATCH ... DELETE r`。

---

## 返回值

```python
{
    "deleted_entities": list[list[dict]],
    "added_entities": list[list[dict]],
}
```

### `deleted_entities`

删除阶段的 Cypher 查询结果列表。`to_be_deleted` 中的每一项对应一个条目。
每个条目是一个记录列表，包含以下键：

- `source` (`str`)：源节点名称
- `target` (`str`)：目标节点名称
- `relationship` (`str`)：被删除边的类型

如果 `to_be_deleted` 为 `None`，则为空列表。

### `added_entities`

MERGE 阶段的 Cypher 查询结果列表。`relations` 中的每一项对应一个条目。
每个条目是一个记录列表，包含以下键：

- `source` (`str`)：源节点名称
- `relationship` (`str`)：边的类型
- `target` (`str`)：目标节点名称

---

## Cypher 行为

对于每一条关系，`ingest()` 执行一条单一的 Cypher 查询，该查询：

1. 通过 `(name, user_id, [agent_id, run_id])` `MERGE` 源节点
   - `ON CREATE`：设置 `created = timestamp()`、`mentions = 1`
   - `ON MATCH`：`mentions` 自增
2. 通过相同的键 `MERGE` 目标节点
   - 使用相同的 `ON CREATE` / `ON MATCH` 逻辑
3. `MERGE` 有向关系
   - `ON CREATE`：设置 `created = timestamp()`、`mentions = 1`
   - `ON MATCH`：`mentions` 自增

**不会写入 `embedding` 属性。** 这是有意为之 —— `ingest()` 不执行嵌入相似性搜索，因此对于通过此接口创建的节点，嵌入字段是不必要的。

---

## 与 `add()` 的对比

| 方面 | 旧的 `add()` | 新的 `ingest()` |
|--------|-------------|----------------|
| LLM 实体提取 | `_retrieve_nodes_from_data`（内部） | 在**上游**执行 |
| LLM 关系提取 | `_establish_nodes_relations_from_data`（内部） | 在**上游**执行 |
| 嵌入相似性搜索 | `_search_graph_db`、`_search_source_node`、`_search_destination_node` | **无** —— 精确 MERGE |
| LLM 删除判断 | `_get_delete_entities_from_search_output`（内部） | `to_be_deleted` **直接传入** |
| 节点写入逻辑 | 基于嵌入搜索结果的四路分支 | 单一的 `MERGE` 路径 |
| 嵌入存储 | `db.create.setNodeVectorProperty` | **无** |
| 向后兼容 | 是（未改变） | 仅新接口 |

---

## 使用示例

```python
from mem0.memory.graph_memory import MemoryGraph

graph = MemoryGraph(config)

# 这些通常来自上游 add-engine 节点
entity_type_map = {"alice": "person", "new_york": "city"}
relations = [
    {"source": "alice", "relationship": "lives_in", "destination": "new_york"},
]
to_be_deleted = [
    {"source": "alice", "relationship": "lives_in", "destination": "boston"},
]

result = graph.ingest(
    entity_type_map=entity_type_map,
    relations=relations,
    filters={"user_id": "u1"},
    to_be_deleted=to_be_deleted,
)

print(result["deleted_entities"])  # [[{"source": "alice", ...}]]
print(result["added_entities"])    # [[{"source": "alice", ...}]]
```

---

## 给 Add-Engine 作者的注意事项

1. **规范化是幂等的。** 你可以传入已规范化的字符串或原始字符串；两者都能正常工作。
2. **关系名称会被清理。** Cypher 关系类型中的无效特殊字符会被自动处理。
3. **类型回退。** 任何在 `entity_type_map` 中找不到的实体都会获得类型 `"__User__"`。
4. **节点标签策略。** 如果设置了 `self.node_label`（即配置了 `base_label`），所有节点使用通用的 `__Entity__` 标签，并通过额外的 `source`/`destination` 属性来表示类型。否则，每个节点直接使用其类型作为标签。
5. **暂无可异步变体。** 如果未来添加 `AsyncMemoryGraph`，`ingest()` 应该有对应的 `aingest()` 方法。
