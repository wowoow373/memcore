"""
mem0 MCP Server — 把 FastAPI REST 端点包装为 MCP 工具
═══════════════════════════════════════════════════════════════════

【文件定位】
  本文件是"协议适配层"：它不实现任何记忆逻辑，只负责把外部的 MCP 协议请求
  翻译为 HTTP 请求，转发给运行在 localhost:8888 的 FastAPI 后端。

【架构概览 — 两层协议的边界】

  ┌──────────────────────────────────────┐
  │        MCP Client（AI 助手）          │  ← 说 "JSON-RPC over HTTP"
  │   Claude Desktop / Cursor / etc.     │     调 tools/call
  └──────────────┬───────────────────────┘
                 │ streamable-http
                 │ http://localhost:8765/mcp
  ┌──────────────▼───────────────────────┐
  │        本文件: mcp_server.py          │  ← MCP 协议层（你在这里）
  │   FastMCP + 12 个 @mcp.tool()        │     类型注解自动生成 JSON Schema
  │   httpx.AsyncClient → 转发 HTTP      │     写 docstring 给 LLM 看
  └──────────────┬───────────────────────┘
                 │ HTTP (httpx)
                 │ http://127.0.0.1:8888
  ┌──────────────▼───────────────────────┐
  │       server/main.py (FastAPI)       │  ← REST 协议层
  │   @app.post / @app.get ...           │     Pydantic 做参数校验
  │   Memory.add() / search() / ...      │     Header 鉴权
  └──────────────────────────────────────┘

【为什么这样分层？】
  1. 不改动现有 FastAPI 代码 — 零侵入。
  2. 能清晰对比 FastAPI 和 MCP 两种协议：一个收 HTTP，一个转 HTTP。
  3. MCP 和 REST 各自独立启停，互不影响。

【阅读顺序建议】
  1. 先看底部 mcp = FastMCP(...) — 理解 MCP Server 怎么创建
  2. 再看 lifespan — 理解资源生命周期（httpx 连接池）
  3. 看 _request / _require_scope — 两个核心工具函数
  4. 挑一个简单的 tool（如 get_memory）看装饰器怎么工作
  5. 看 add_memory — 最复杂的 tool，理解参数映射
  6. 看最后的 if __name__ == "__main__" — 理解传输协议选择

【核心概念速查表】
  ┌────────────────────┬──────────────────────────────────┐
  │ 概念                │ 本文件中的体现                    │
  ├────────────────────┼──────────────────────────────────┤
  │ 装饰器即注册         │ @mcp.tool() 把函数注册为 MCP 工具│
  │ 类型注解 → Schema   │ str, int, Dict → JSON Schema     │
  │ docstring → LLM    │ "When to use" 是 AI 调用的唯一依据 │
  │ stateless_http     │ 每次请求独立，无 session 状态      │
  │ json_response      │ 返回单个 JSON，方便 curl 调试      │
  │ streamable-http    │ HTTP POST 收发 JSON-RPC           │
  │ lifespan           │ async context manager 管理资源     │
  └────────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 第一部分：配置常量 — 通过环境变量注入，有默认值兜底
# ═══════════════════════════════════════════════════════════════════════════════
# 这几个变量控制 MCP Server 的行为：
#   MEM0_BASE_URL  → FastAPI 后端地址
#   MEM0_API_KEY   → 调用 FastAPI 时带的鉴权头
#   MCP_PORT       → MCP Server 自己的监听端口
#
# 【FastAPI 学习点】在 server/main.py 中，ADMIN_API_KEY 也是同样的模式：
#   环境变量 → 默认值 → 生产环境 warning。
#   这是 12-factor app 的标准做法：配置和代码分离。

MEM0_BASE_URL = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:8888")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "my_very_long_custom_key_123456")
MCP_PORT = int(os.getenv("MEM0_MCP_PORT", "8765"))

if MEM0_API_KEY == "my_very_long_custom_key_123456":
    logger.warning(
        "MEM0_API_KEY 正在使用默认值。生产环境请设置 MEM0_API_KEY 环境变量。"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 第二部分：模块级状态 — httpx AsyncClient 的存储
# ═══════════════════════════════════════════════════════════════════════════════
# 【为什么要用模块级变量而不是参数传递？】
#   FastMCP 在 stateless_http 模式下，工具函数没有 "request context"。
#   每个工具函数被调用时，除了自己的参数外拿不到额外的上下文。
#   所以我们在 lifespan 中创建 httpx.AsyncClient，存在模块级 _http 里，
#   所有工具函数通过闭包捕获这个变量来共享同一个 HTTP 连接池。
#
# 【为什么必须用 httpx.AsyncClient 而不是 httpx.Client？】
#   FastMCP 的工具是 async 函数，内部运行在 asyncio 事件循环中。
#   如果在这里用同步的 httpx.Client，会阻塞整个事件循环，
#   导致其他并发的工具调用全部卡住。AsyncClient 在等 HTTP 响应时
#   会主动让出控制权（yield），允许事件循环调度其他协程。
#
# 【type hint 解释】
#   _http: httpx.AsyncClient | None = None
#   意思是：_http 要么是一个 AsyncClient 实例，要么是 None（启动前/关闭后）。

_http: httpx.AsyncClient | None = None

# 默认的请求头，每次 HTTP 转发都会带上 API Key 鉴权
DEFAULT_HEADERS = {"X-API-Key": MEM0_API_KEY}


# ═══════════════════════════════════════════════════════════════════════════════
# 第三部分：lifespan — 资源生命周期管理
# ═══════════════════════════════════════════════════════════════════════════════
# 【这是什么？】
#   lifespan 是 FastMCP 的"启动/关闭钩子"。它是一个 async context manager：
#     - yield 之前的代码 → 服务器启动时执行
#     - yield 之后的代码 → 服务器关闭时执行
#
# 【为什么用 @contextlib.asynccontextmanager？】
#   这是 Python 标准库提供的装饰器，把一个 async generator 函数
#   变成一个 async context manager。等价于写一个带 __aenter__/__aexit__ 的类，
#   但写法简洁很多。
#
# 【FastAPI 对比学习】
#   FastAPI 也有类似的 lifespan 模式（fastapi.FastAPI(lifespan=...) ）。
#   两种框架的设计思路一致：启动时初始化连接 → 运行时复用 → 关闭时清理。
#
# 【连接池的意义】
#   httpx.AsyncClient 内部维护了一个 HTTP 连接池。如果不复用 client，
#   每次工具调用都要新建 TCP 连接、TLS 握手，延迟增加 100-300ms。
#   通过 lifespan 创建一个长生命周期的 client，所有工具调用共享它。

@contextlib.asynccontextmanager
async def lifespan(server: FastMCP) -> None:  # noqa: ARG001 — server 参数由框架传入，我们暂不需要
    """
    MCP Server 的资源生命周期管理器。

    启动时创建一个带连接池的 httpx AsyncClient，
    关闭时自动清理连接。
    """
    global _http
    async with httpx.AsyncClient(
        base_url=MEM0_BASE_URL,       # 基础 URL，后续只需要传相对路径如 "/memories"
        headers=DEFAULT_HEADERS,       # 每次请求自动附带 API Key
        timeout=httpx.Timeout(30.0),   # 30 秒超时（mem0 add 流程可能很慢）
    ) as http:
        _http = http
        logger.info("MCP httpx 客户端已连接到 %s", MEM0_BASE_URL)
        # ── yield 是分界线：上面是"启动时"，下面是"关闭时" ──
        yield
    # 走到这里说明服务器正在关闭，清理全局引用
    _http = None


# ═══════════════════════════════════════════════════════════════════════════════
# 第四部分：FastMCP 实例 — 整个 MCP Server 的核心
# ═══════════════════════════════════════════════════════════════════════════════
# 【FastMCP 是什么？】
#   FastMCP 是 mcp 库提供的"高级 API"。它封装了底层的 JSON-RPC 协议细节，
#   让你用 Python 装饰器的形式注册工具，不用手写 JSON-RPC 消息处理。
#
#   类比：FastMCP 之于 MCP 协议 ≈ FastAPI 之于 HTTP 协议。
#   两者都是"装饰器驱动开发"：用 @app.get / @mcp.tool() 声明接口。
#
# 【关键参数解释】
#
#   stateless_http=True
#     → 每次 HTTP 请求都是一个独立的 MCP 会话。服务器不保留 session 状态。
#     → 适合这种"纯透传代理"场景：MCP Server 无状态，状态在 FastAPI 那边。
#     → 如果不设置这个，FastMCP 会要求客户端先发送 initialize 并维护 session ID。
#
#   json_response=True
#     → 返回单个 JSON 对象而不是 SSE (Server-Sent Events) 流。
#     → 方便用 curl 直接调试：一个请求过去，一个 JSON 回来。
#     → 适合工具调用这种"一问一答"的场景。
#
#   host="0.0.0.0"
#     → 监听所有网络接口。在 WSL2 中，0.0.0.0 的端口会自动转发到 Windows。
#     → 如果用 "127.0.0.1"，新版 WSL2 也能转发，但 0.0.0.0 兼容性更好。
#
#   lifespan=lifespan
#     → 把上面定义的资源管理器挂载到 Server 上。

mcp = FastMCP(
    "mem0-mcp",              # Server 名称，会出现在 initialize 响应的 serverInfo.name 中
    stateless_http=True,     # 无状态模式：每次请求独立
    json_response=True,      # 返回 JSON 而非 SSE 流
    host="0.0.0.0",          # 监听所有网络接口（WSL2 转发需要）
    port=MCP_PORT,           # 监听端口，默认 8765
    lifespan=lifespan,       # 资源生命周期管理器
)


# ═══════════════════════════════════════════════════════════════════════════════
# 第五部分：工具函数 — 两个"螺丝"，所有 tool 都用它们
# ═══════════════════════════════════════════════════════════════════════════════

async def _request(method: str, path: str, **kw: Any) -> dict:
    """
    统一的 HTTP 请求转发器。

    【设计意图】
      12 个 MCP Tool 的核心逻辑完全一样：调用 FastAPI 的某个端点，
      把响应返回。与其在每个 tool 里重复 try/except，不如抽一个函数。

    【为什么返回 dict 而不是抛异常？】
      LLM（大语言模型）会读取 tool 的返回值来决定下一步。
      如果抛异常，MCP 协议层会返回一个 JSON-RPC error，LLM 看到的
      是错误码和堆栈，难以理解。返回一个带 "error": True 的 dict，
      LLM 可以读懂错误内容并调整策略。例如：
        {"error": True, "status": 400, "detail": "user_id is required"}
      比 "InternalError: HTTPStatusError" 有用得多。

    【参数说明】
      method: "GET" | "POST" | "PUT" | "DELETE"
      path:   路径，如 "/memories" 或 "/memories/abc123"
      **kw:   透传给 httpx.AsyncClient.request()，通常用 json={...} 或 params={...}
    """
    try:
        r = await _http.request(method, path, **kw)
        r.raise_for_status()  # 4xx/5xx 会触发 HTTPStatusError
        return r.json()
    except httpx.HTTPStatusError as e:
        # 后端返回了错误状态码（如 400、500）
        return {"error": True, "status": e.response.status_code, "detail": e.response.text}
    except httpx.RequestError as e:
        # 网络层错误（连接失败、超时等）
        return {"error": True, "status": None, "detail": str(e)}


def _require_scope(user_id: Any, agent_id: Any, run_id: Any) -> None:
    """
    作用域校验：至少提供一个标识符。

    【为什么在 MCP 层校验而不依赖 FastAPI 的校验？】
      FastAPI 也会做这个校验（见 server/main.py 各端点），但如果我们
      等到 HTTP 请求发过去才发现错误，来回多一圈延迟（~10ms）。
      更重要的是：在 MCP 层直接抛出 ValueError，FastMCP 会将其
      转换为清晰的错误消息返回给 LLM，LLM 可以立即修正参数重试。
      如果等 FastAPI 返回 400，LLM 需要解析 JSON 错误体，多一步理解成本。

    【设计思考：为什么用 ValueError 而不是返回错误 dict？】
      与 _request 不同——_request 的错误是"运行时意外"（网络问题、后端异常），
      而缺少 scope 是"调用方用错了"——这是一个编程/提示词错误。
      用异常抛出可以让 LLM 明确知道：这不是"操作失败了"，而是"你参数没给对"。
    """
    if not any([user_id, agent_id, run_id]):
        raise ValueError(
            "至少需要提供 user_id、agent_id 或 run_id 中的一个。"
        )


def _drop_none(d: dict) -> dict:
    """
    过滤掉字典中值为 None 的键。

    【为什么需要这个？】
      FastAPI 的端点（如 GET /memories）对 None 值的处理不一致：
      如果传了 user_id=None，有些版本会把 None 序列化为字符串 "null"。
      与其依赖后端处理，不如在发送前就把 None 键去掉。
      同时也避免在 URL query string 中出现 ?user_id=None 这种脏参数。

    例如：{"user_id": "u1", "agent_id": None} → {"user_id": "u1"}
    """
    return {k: v for k, v in d.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════════════════
# 第六部分：12 个 MCP Tool — 每个对应一个 FastAPI 端点
# ═══════════════════════════════════════════════════════════════════════════════
#
# 【装饰器的魔法 — @mcp.tool() 做了什么？】
#
#   当你写：
#     @mcp.tool()
#     async def add_memory(messages: ..., user_id: ...) -> dict:
#         """Store new memories..."""
#         ...
#
#   FastMCP 在背后做了 3 件事：
#     1. 注册：把函数名 "add_memory" 加入工具列表（tools/list 会返回它）
#     2. Schema 生成：根据类型注解（List[Dict[str,str]], str, int）
#        自动生成 JSON Schema，告诉 LLM "这个工具接受什么参数"
#     3. 路由：当收到 tools/call {"name": "add_memory", "arguments": {...}}
#        时，自动把 arguments 映射到函数参数并调用
#
#   【FastAPI 对比】
#     FastAPI:  @app.post("/memories")  + Pydantic BaseModel → OpenAPI Schema
#     FastMCP:  @mcp.tool()            + Python type hints  → JSON Schema
#
#     两者的设计哲学完全一致：声明式编程，框架负责"协议翻译"，
#     你只负责写业务逻辑（这里是 HTTP 转发）。
#
# 【docstring 写给谁看？】
#   写给 LLM（大语言模型）看的！LLM 根据 docstring 中的描述来决定
#   什么时候调用哪个工具。所以要写清楚：
#     1. 这个工具做什么（一句概括）
#     2. When to use（什么时候该调用它）
#     3. Example（示例参数，帮助 LLM 理解参数格式）
#
# 【参数类型与 JSON Schema 的对应关系】
#   Python 类型注解              →  JSON Schema type
#   ─────────────────────────      ─────────────────
#   str                            →  "string"
#   int                            →  "integer"
#   bool                           →  "boolean"
#   List[Dict[str,str]]            →  "array" of "object"
#   Dict[str,Any]                  →  "object" (无固定字段)
#   Optional[str] (= str | None)   →  "string" (非必填)
#
#   这就是 FastMCP 的核心机制：你写 Python 类型，它自动生成 JSON Schema。
#   不需要手写 schema 文件，不需要 Pydantic model。
#
# 【工具分组说明】
#   下面按 4 组排列，与 server/main.py 的 openapi_tags 一一对应：
#     Configuration (1), Memories (7), Summaries (2), Maintenance (1)
#   再加上 1 个 Search，共 12 个。

# ─── Configuration（配置）────────────────────────────────────────────────────
# 只有 1 个工具：configure。动态修改后端记忆引擎的配置。

@mcp.tool()
async def configure(config: Dict[str, Any]) -> dict:
    """运行时动态修改后端记忆引擎配置。

    When to use: 当用户需要更换 LLM、embedder 或向量数据库而不重启服务时。

    参数 config 是一个字典，结构与 mem0 的 MemoryConfig 一致。
    示例：
      {
        "vector_store": {"provider": "qdrant", "config": {...}},
        "llm": {"provider": "openai", "config": {"model": "gpt-4o", ...}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", ...}}
      }

    【注意】这个操作会替换整个全局 Memory 实例，影响所有后续请求。
    """
    return await _request("POST", "/configure", json=config)


# ─── Memories — Create（创建记忆）────────────────────────────────────────────
# add_memory 是 12 个工具中最复杂的一个，参数最多，背后的流程最长。

@mcp.tool()
async def add_memory(
    messages: List[Dict[str, str]],
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    infer: bool = True,
    memory_type: str | None = None,
    prompt: str | None = None,
    vector_return_number: int = 2,
    graph_return_depth: int = 2,
) -> dict:
    """存储新的记忆。

    When to use: 每次用户分享关于自己的信息、偏好、或任何值得在未来对话中
    回忆的内容时使用。用户显式要求记住某事时也要用。

    【后端处理流程（fast path vs inference path）】
      infer=True（默认）→ 完整流程：
        1. LLM 从 messages 中提取事实
        2. 向量检索已有相关记忆 → 去重
        3. 对每条事实决定 ADD / UPDATE / DELETE / NONE
        4. 执行决策，写入向量库和图数据库

      infer=False → 直接写入：
        跳过 LLM 推理，直接把 messages 原样存入向量库。
        适合已经预处理过的内容。

    【参数速查】
      messages             — 对话消息列表，每条 {"role":"user/assistant","content":"..."}
      user_id/agent_id/run_id — 至少提供一个作为记忆的作用域
      metadata             — 自定义元数据，可包含 memory_mode="process_flow" 等控制标志
      infer                — True=走事实提取+CRUD决策，False=直接写入
      memory_type          — "procedural_memory" 与 agent_id 配合使用
      vector_return_number — add 流程中向量召回的 top-k 候选数
      graph_return_depth   — add 流程中图搜索的跳数

    示例：
      messages=[{"role":"user","content":"I prefer dark roast coffee and short morning workouts."}]
      user_id="u_demo_001"
    """
    # ── 第一步：参数校验 ──
    # 在发送 HTTP 请求之前，先在 MCP 层校验。失败时 LLM 立刻收到清晰提示。
    _require_scope(user_id, agent_id, run_id)

    # ── 第二步：构造请求体 ──
    # _drop_none 确保值为 None 的字段不会出现在 JSON 中，
    # 避免把 user_id=null 发给后端。
    body = _drop_none({
        "messages": messages,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": metadata,
        "infer": infer,
        "memory_type": memory_type,
        "prompt": prompt,
        "vector_return_number": vector_return_number,
        "graph_return_depth": graph_return_depth,
    })

    # ── 第三步：HTTP 转发 ──
    # POST /memories 对应 server/main.py 的 add_memory 端点
    return await _request("POST", "/memories", json=body)


# ─── Memories — Read（读取记忆）──────────────────────────────────────────────
# 3 个读取工具：list_memories（全量）、get_memory（单条）、memory_history（变更历史）

@mcp.tool()
async def list_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """获取指定作用域下的所有记忆。

    When to use: 需要了解某个用户/agent/run 已经记住了什么，再做后续决策时。
    这是"概览"工具，返回列表。

    示例：
      user_id="u_demo_001"
    """
    _require_scope(user_id, agent_id, run_id)
    # GET 请求的参数放在 URL query string 中（通过 params= 传入）
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request("GET", "/memories", params=params)


@mcp.tool()
async def get_memory(memory_id: str) -> dict:
    """按 ID 获取一条具体的记忆。

    When to use: 你已经知道某条记忆的 ID（从之前的 search 或 list 结果中拿到），
    需要查看它的完整内容时。

    示例：
      memory_id="abc123-def456"

    【URL 参数 vs Query 参数 — FastAPI 学习点】
      这个工具对应 GET /memories/{memory_id}。
      路径中的 {memory_id} 叫"路径参数"（path parameter），
      用 Python 的 f-string 拼入 URL。
      而 list_memories 的 user_id 叫"查询参数"（query parameter），
      通过 httpx 的 params= 传入，会附加到 URL 的 ? 后面。
      在 FastAPI 中，路径参数用 @app.get("/memories/{memory_id}") 定义，
      查询参数用函数签名中的 Optional[str] = None 定义。
    """
    return await _request("GET", f"/memories/{memory_id}")


@mcp.tool()
async def memory_history(memory_id: str) -> dict:
    """获取某条记忆的变更历史。

    When to use: 需要追踪一条记忆从创建以来的所有修改记录
    （ADD、UPDATE、DELETE 事件）时。

    示例：
      memory_id="abc123-def456"
    """
    return await _request("GET", f"/memories/{memory_id}/history")


# ─── Memories — Update（更新记忆）────────────────────────────────────────────
@mcp.tool()
async def update_memory(memory_id: str, updated: Dict[str, Any]) -> dict:
    """更新一条已有记忆的内容。

    When to use: 用户纠正或修正之前存储的信息时。
    比如用户之前说喜欢深烘咖啡，现在说改喜欢浅烘了。

    示例：
      memory_id="abc123-def456"
      updated={"memory": "I prefer light roast coffee now."}

    【PUT 语义 — FastAPI 学习点】
      对应 FastAPI 的 @app.put("/memories/{memory_id}")。
      PUT 在 REST 中语义是"完整替换"或"幂等更新"。
      这里传的 updated dict 会直接传给 Memory.update() 方法。
    """
    return await _request("PUT", f"/memories/{memory_id}", json=updated)


# ─── Memories — Delete（删除记忆）────────────────────────────────────────────
# 2 个删除工具：delete_memory（单条）、delete_all_memories（批量）

@mcp.tool()
async def delete_memory(memory_id: str) -> dict:
    """按 ID 删除一条记忆。

    When to use: 用户要求忘记某条具体信息，或发现某条记忆不正确需要删除时。

    示例：
      memory_id="abc123-def456"
    """
    return await _request("DELETE", f"/memories/{memory_id}")


@mcp.tool()
async def delete_all_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """删除指定作用域下的所有记忆。

    When to use: 用户要求"清空所有记忆"或"重置我的数据"时。
    这是批量操作，影响范围由 user_id/agent_id/run_id 确定。

    【安全设计】
      必须提供至少一个标识符才能执行。这防止了"误删全库"的事故。
      如果想清空整个数据库（所有用户的所有记忆），用 reset_all。

    示例：
      user_id="u_demo_001"
    """
    _require_scope(user_id, agent_id, run_id)
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request("DELETE", "/memories", params=params)


# ─── Search（搜索记忆）───────────────────────────────────────────────────────
# 这是使用频率最高的工具。LLM 在回答用户问题前，应该先 search 相关记忆。

@mcp.tool()
async def search_memories(
    query: str,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 100,
    filters: Dict[str, Any] | None = None,
) -> dict:
    """语义搜索已存储的记忆。

    When to use: 每当用户问的问题可能跟之前存储的信息有关时。
    这是最主要的记忆检索工具。

    【搜索模式】
      Normal 模式（默认）：
        向量相似度搜索 + 图数据库并行查询。
        filters 可以用来限定范围，比如 {"channel": "web"}。

      Process-flow 模式：
        设置 filters={"memory_mode":"process_flow"} 启用。
        会在 Neo4j 内部做向量召回 + 图遍历，适合流程类记忆。
        可选 filters.graph_depth 控制图遍历跳数（默认 1）。

    【limit 参数】
      默认 100。对于 process-flow 模式，limit 控制向量召回的步节点数。

    【为什么 limit 默认 100 而不是更小的值？】
      mem0 后端会做去重和排序。返回更多候选给 LLM，
      让 LLM 自己判断相关性，比在 MCP 层截断更可靠。

    示例：
      query="coffee preferences"
      user_id="u_demo_001"
      limit=5
      filters={"memory_mode": "process_flow", "graph_depth": 2}
    """
    _require_scope(user_id, agent_id, run_id)
    body = _drop_none({
        "query": query,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "limit": limit,
        "filters": filters,
    })
    return await _request("POST", "/search", json=body)


# ─── Summaries（记忆总结）────────────────────────────────────────────────────
# 2 个工具：start_summary（触发后台生成）、get_summary（读取结果）。
# 这是异步模式：触发 → 后台跑 → 轮询获取结果。

@mcp.tool()
async def start_summary(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    limit: int = 200,
    trigger: str = "manual",
) -> dict:
    """触发后台记忆总结生成。

    When to use: 一段对话结束后，用户可能希望把最近的记忆压缩成摘要。
    这个工具只负责触发，不等待结果。用 get_summary 获取生成的结果。

    【异步任务模式 — FastAPI 学习点】
      start_summary → 触发后台线程/任务 → 立即返回（可能是 "already_running"）
      get_summary    → 读取已生成的总结（可能为空）
      这种"触发 + 轮询"是 REST API 中常见的异步任务处理模式。

    示例：
      user_id="u_demo_001"
      limit=200
      trigger="manual"
    """
    _require_scope(user_id, agent_id, run_id)
    body = _drop_none({
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "limit": limit,
        "trigger": trigger,
    })
    return await _request("POST", "/start_mem_summary", json=body)


@mcp.tool()
async def get_summary(
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """读取最新生成的记忆总结。

    When to use: 调用 start_summary 后，或在对话开始时检查是否已有总结。
    如果还没有总结，返回空结构。

    示例：
      user_id="u_demo_001"
    """
    _require_scope(user_id, agent_id, run_id)
    params = _drop_none({"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
    return await _request("GET", "/get_summary", params=params)


# ─── Maintenance（维护操作）───────────────────────────────────────────────────
# 只有 1 个工具：reset_all。清空整个记忆库。

@mcp.tool()
async def reset_all() -> dict:
    """完全重置整个记忆存储。

    When to use: 仅当用户明确要求清空所有记忆时使用。
    这是不可逆的破坏性操作。

    【为什么这个工具没有 user_id 等参数？】
      因为 reset() 是全局操作——清空整个向量数据库和所有图数据。
      如果需要只清空某个用户的数据，应该用 delete_all_memories。
    """
    return await _request("POST", "/reset")


# ═══════════════════════════════════════════════════════════════════════════════
# 第七部分：启动入口
# ═══════════════════════════════════════════════════════════════════════════════
# 【为什么是 __name__ == "__main__" 而不是通过 uvicorn 启动？】
#   FastMCP 在 run() 方法内部自己创建和管理 uvicorn 实例。
#   不需要也不能用 uvicorn main:app 这种方式启动。
#   这也意味着不能加 --reload 参数（那不是 mcp.run() 支持的用法）。
#
# 【transport="streamable-http" 是什么？】
#   MCP 协议有 3 种传输方式：
#     1. stdio           — 标准输入输出，适合本地子进程（如 Claude Desktop 拉起）
#     2. SSE             — Server-Sent Events，单向推送 + 独立 POST 通道
#     3. streamable-http — 双向 HTTP，每个请求含完整的 JSON-RPC 请求/响应
#
#   这里选 streamable-http 是因为：
#     - MCP 和 FastAPI 是独立进程，不在同一台机器上也行
#     - WSL2 → Windows 的 localhost 端口转发天然支持 HTTP
#     - 用 curl 可以直接调试，不需要特殊的 MCP 客户端
#
# 【访问路径】
#   MCP 端点挂在 /mcp 路径下，不是 / 根路径。
#   客户端应该连接 http://localhost:8765/mcp。
#   如果访问 http://localhost:8765/ 会返回 404。

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
