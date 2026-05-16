"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys

═══════════════════════════════════════════════════════════════════════════════
【并发与多用户模型总览】
───────────────────────────────────────────────────────────────────────────────
这份代码的并发由 3 个层次共同支撑：

1. FastAPI + asyncio（协程级并发）
   - uvicorn ASGI 服务器在单进程内运行事件循环
   - 每个 HTTP 请求/SSE 连接是一个独立的协程（async def）
   - 多个 MCP Client 可以同时建立 SSE 长连接，互不阻塞

2. contextvars（请求上下文隔离）
   - user_id_var / client_name_var 是每个协程独立的"储物柜"
   - 即使多个用户同时请求，各自的 user_id 也不会互相覆盖
   - 这是 asyncio 中替代全局变量的标准做法

3. mem0 Memory 单例共享
   - get_memory_client() 返回全局单例 _memory_client
   - 所有请求共享同一个 Memory 实例
   - Memory 内部组件的并发安全性：
     • vector_store / embedding_model / llm：HTTP 客户端，天然线程安全
     • db (SQLiteManager)：使用 threading.Lock()，但在 asyncio 中会阻塞事件循环
     • _summary_executor：ThreadPoolExecutor，后台线程池执行总结任务

⚠️ 注意：Tool 函数是 async def，但 memory_client.add() 等是同步方法。
   在 asyncio 中调用同步代码会阻塞整个事件循环，影响并发处理能力。
   生产环境建议使用 asyncio.to_thread() 包裹或改用 AsyncMemory。

多用户数据隔离：
   - URL 路径参数区分用户：/{client_name}/sse/{user_id}
   - mem0 内部通过 user_id + filters 在向量数据库层面隔离
   - 关系型数据库通过外键和查询条件隔离
═══════════════════════════════════════════════════════════════════════════════
"""

import contextvars
import datetime
import json
import logging
import uuid

from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter

# ═══════════════════════════════════════════════════════════════════════════════
# 【MCP SERVER 框架代码】导入 MCP 库
# ───────────────────────────────────────────────────────────────────────────────
# 这 2 个导入是纯 MCP 协议相关的，和具体业务无关。
# - FastMCP: 用于创建 MCP Server 实例，管理 Tool 注册
# - SseServerTransport: 用于 MCP 协议的 SSE 传输层
#
# 【为谁服务？】→ 为 MCP SERVER 服务
# 【作用？】让 Python 代码具备 MCP Server 的能力
#
# 【重要：MCP 本身没有并发能力】
#   mcp 库只是一个"协议解析器 + 工具注册表 + 消息路由器"。
#   FastMCP 不是真正的服务器，mcp._mcp_server.run() 只是顺序处理消息的事件循环。
#   如果用 stdio 传输（本地子进程），只能顺序处理，没有并发。
#   只有 FastAPI + asyncio 这层包装才带来了协程级并发能力。
#   可以把 mcp 库想象成"快递分拣系统"，它只负责把包裹分到正确的地方，
#   至于能同时处理多少包裹，取决于外面的运输网络（FastAPI）。
# ═══════════════════════════════════════════════════════════════════════════════
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# 【MCP SERVER 框架代码】创建 MCP Server 实例
# ───────────────────────────────────────────────────────────────────────────────
# mcp = FastMCP("...") 创建一个 MCP Server 对象。
# 这行代码本身就是在"启动" MCP Server —— 后续所有 @mcp.tool() 都是向这个 Server 注册工具。
#
# 【为谁服务？】→ 为 MCP SERVER 服务
# 【谁会用？】MCP Client（如 Claude Desktop、Cursor、或其他 AI 应用）会连接到这个 Server
# 【作用？】这是 Server 的"核心引擎"，负责：
#   1. 维护已注册的 Tool 列表
#   2. 处理 MCP 协议消息（JSON-RPC）
#   3. 将 Client 的 Tool 调用请求分发给对应的 Python 函数
# ═══════════════════════════════════════════════════════════════════════════════
mcp = FastMCP("mem0-mcp-server")


# ───────────────────────────────────────────────────────────────────────────────
# 【非 MCP 业务逻辑】安全获取 memory client
# ───────────────────────────────────────────────────────────────────────────────
# 这个函数和 MCP 协议无关，是 OpenMemory 应用自身的业务逻辑。
# 它只是给 get_memory_client() 加了一个 try/except 包装。
#
# 【为谁服务？】→ 为 OpenMemory 应用服务（不是为 MCP 服务）
# 【作用？】获取 mem0 的 Memory 实例，失败时返回 None 而不是抛异常
#
# 【并发相关】单例模式说明：
#   get_memory_client() 内部使用全局变量 _memory_client 缓存实例。
#   第一次调用时创建 Memory.from_config(config)，后续直接返回缓存。
#   这意味着所有并发请求共享同一个 Memory 实例。
#
#   单例的好处：
#     1. 避免重复初始化（LLM 连接、向量数据库连接很耗时）
#     2. 减少资源消耗（连接池复用）
#   单例的风险：
#     1. Memory 内部状态变化会影响所有请求
#     2. 如果 Memory 不是线程安全的，并发请求可能互相干扰
#
#   好消息是：Memory 的主要组件（vector_store、llm、embedder）
#   都是无状态的 HTTP 客户端，天然支持并发。
# ───────────────────────────────────────────────────────────────────────────────
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id and client_name
# 使用 contextvars 在同一次 HTTP 请求的不同协程间共享用户身份信息
#
# 【并发安全核心机制】
#   contextvars 是 Python 的"协程级全局变量"——每个协程看到不同的值。
#   在 asyncio 中，100 个并发请求 = 100 个协程 = 100 份独立的 user_id。
#
#   为什么不能用普通全局变量？
#     global_user_id = None  # ❌ 所有请求共享同一个变量
#     # 请求 A 设置 user_id="Alice"
#     # 请求 B 同时设置 user_id="Bob"
#     # 结果：A 可能读到 "Bob"，互相覆盖！
#
#   contextvars 如何解决？
#     user_id_var.set("Alice")  # 只影响当前协程的副本
#     user_id_var.set("Bob")    # 只影响另一个协程的副本
#     # 两个请求互不干扰 ✅
#
#   生命周期管理：
#     handle_sse() 中 set() → Tool 函数中 get() → finally 中 reset()
#     reset() 确保请求结束后清理，避免污染后续复用的协程
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# ═══════════════════════════════════════════════════════════════════════════════
# 【MCP SERVER 框架代码】初始化 SSE 传输层
# ───────────────────────────────────────────────────────────────────────────────
# SseServerTransport 是 MCP 协议的一部分，负责"传输"层。
# MCP 协议不规定必须用 HTTP，也可以用 stdio、WebSocket 等。
# 这里选择 SSE 是因为它是基于 HTTP 的，易于穿透防火墙和浏览器支持。
#
# 【为谁服务？】→ 为 MCP SERVER 服务（具体说是为"传输层"服务）
# 【作用？】建立和维护 Server 与 Client 之间的通信通道：
#   - Server → Client: 通过 SSE 长连接推送响应
#   - Client → Server: 通过 POST /messages/ 发送请求
#
# 【类比】就像快递公司的"配送网络"，负责把包裹（JSON-RPC 消息）送到目的地
# ═══════════════════════════════════════════════════════════════════════════════
sse = SseServerTransport("/mcp/messages/")


# ═══════════════════════════════════════════════════════════════════════════════
# 【MCP SERVER 框架代码】注册 MCP Tool
# ───────────────────────────────────────────────────────────────────────────────
# @mcp.tool(description="...") 是 MCP Server 框架提供的装饰器。
# 被装饰的函数会变成 MCP Server 的"工具"，供 MCP Client 调用。
#
# 【为谁服务？】→ 框架层面为 MCP SERVER 服务；函数体为具体业务服务
# 【谁会用？】→ MCP Client（AI 助手）会调用这些 Tool
# 【调用流程】
#   1. AI 助手（Client）分析用户意图，决定调用某个 Tool
#   2. Client 通过 MCP 协议发送 JSON-RPC 请求：{"method": "tools/call", "params": {"name": "add_memories", "arguments": {"text": "..."}}}
#   3. MCP Server 收到请求，找到对应的函数（add_memories），传入参数执行
#   4. 函数返回的字符串结果，通过 MCP 协议返回给 Client
#   5. Client 将结果展示给 AI，AI 继续对话
#
# 【重要】description 是写给 AI 看的！AI 靠 description 决定要不要调用这个工具。
#
# 【关于 async def 的准确解释】
#   MCP 框架层确实支持 async def Tool！
#   FastMCP 源码（func_metadata.py 第 92-95 行）：
#     if fn_is_async:
#         return await fn(...)    ← async def → MCP 用 await 调用 ✅
#     else:
#         return fn(...)          ← def → MCP 直接调用      ❌
#
#   这些函数声明了 async def：
#     ✅ MCP 层面使用 await 调用（异步感知）
#     ✅ MCP Server 在 await 时可以处理其他消息
#     ❌ 但函数体内部没有任何 await（全部是同步调用）
#     ❌ memory_client.add() / .search() / .get_all() / .delete() 都是 def
#     ❌ 函数体被阻塞，事件循环仍然卡住
#
#   关键认知：
#     async def ≠ 不会阻塞
#     async def 只表示"可以被 await"
#     只有内部有 await 时，才会真正挂起协程释放事件循环
#     否则 async def 和 def 行为完全一样！
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(description="Add a new memory. This method is called everytime the user informs anything about themselves, their preferences, or anything that has any relevant information which can be useful in the future conversation. This can also be called when the user asks you to remember something.")
async def add_memories(text: str) -> str:
    """
    【MCP TOOL 实现】添加记忆 —— 这是具体业务逻辑

    【为谁服务？】→ 为 MCP CLIENT 服务（被 AI 调用）
    【什么时候被调用？】当 AI 判断用户说了需要记住的信息时
    【调用方是谁？】→ MCP Client（如 Claude Desktop），不是人类用户直接调用
    【返回什么？】JSON 字符串，告诉 Client 操作结果

    业务逻辑：
    1. 从 contextvars 获取当前用户（由 HTTP 路由层注入）
    2. 调用 mem0 核心库 memory_client.add() 保存记忆
    3. 同步更新关系型数据库（Memory、MemoryStatusHistory 表）
    4. 返回 JSON 结果给 MCP Client
    """
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            if not app.is_active:
                return f"Error: App {app.name} is currently paused on OpenMemory. Cannot create new memories."

            # ⚠️【并发注意】memory_client.add() 是同步方法，会阻塞整个 asyncio 事件循环
            #
            # 【阻塞机制深度解析】
            #   asyncio 是单线程事件循环。当协程执行到同步代码时，事件循环被"卡住"，
            #   无法调度其他协程，直到同步代码返回。
            #
            #   add() 内部调用链（全部同步）：
            #     memory_client.add()
            #       └─► LangGraph workflow.stream()  [Python 层面同步]
            #           ├─► _node_extract_facts() ──► llm.generate_response()
            #           │     └─► openai.OpenAI().chat.completions.create()
            #           │           └─► httpx.Client.send()  [HTTP 请求，阻塞等待响应]
            #           ├─► _node_retrieve_memories() ──► vector_store.search()
            #           │     └─► qdrant_client.query_points()  [HTTP/本地调用，阻塞]
            #           └─► _node_execute_add() ──► vector_store.insert()  [阻塞]
            #
            #   所有这些调用都是同步的，任何一个步骤都会阻塞事件循环。
            #
            # 【关于"HTTP 是否真阻塞"的澄清】
            #   httpx.Client 的 .send() 确实会释放 GIL（在 C 层面做 socket IO 时），
            #   但这只对其他*线程*有用。对于 asyncio 的单线程事件循环来说：
            #   - 协程 A 调用 httpx.Client.send() → Python 线程被阻塞
            #   - 事件循环无法切换协程 B，因为调度器在 Python 层面被卡住了
            #   - 所以"GIL 释放"对 asyncio 没有意义，事件循环仍然被阻塞
            #
            # 【time.sleep() 也是同理】
            #   time.sleep(5) 会阻塞 Python 线程，事件循环完全暂停 5 秒。
            #   这 5 秒内：新连接无法建立、已有连接的请求无法处理、心跳无法响应。
            #   asyncio.sleep(5) 才会让出控制权，允许其他协程运行。
            #
            # 生产环境优化建议：
            #   response = await asyncio.to_thread(memory_client.add, text, user_id=uid, ...)
            #   或者改用 mem0 的 AsyncMemory 类
            response = memory_client.add(text,
                                         user_id=uid,
                                         metadata={
                                            "source_app": "openmemory",
                                            "mcp_client": client_name,
                                        })

            if isinstance(response, dict) and 'results' in response:
                for result in response['results']:
                    memory_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()

                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                state=MemoryState.active
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']

                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if memory else None,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.UTC)
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted
                            )
                            db.add(history)

                db.commit()

            return json.dumps(response)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return f"Error adding to memory: {e}"


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str) -> str:
    """
    【MCP TOOL 实现】搜索记忆 —— 具体业务逻辑

    【为谁服务？】→ 为 MCP CLIENT 服务（被 AI 调用）
    【什么时候被调用？】AI 认为需要检索用户过往记忆来回答问题时
    【调用方是谁？】→ MCP Client（AI 助手）
    """
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            filters = {"user_id": uid}

            # ⚠️【并发注意】以下两个调用都是同步方法，会阻塞事件循环
            #
            # embed() 调用链：
            #   OpenAIEmbedding.embed()
            #     └─► openai.OpenAI().embeddings.create()
            #           └─► httpx.Client.send()  [同步 HTTP，阻塞]
            #
            # search() 调用链（以 Qdrant 为例）：
            #   Qdrant.search()
            #     └─► qdrant_client.query_points()
            #           └─► 内部 HTTP 请求到 Qdrant 服务器  [同步，阻塞]
            #
            # 这两个操作通常很快（嵌入 100-500ms，搜索 50-200ms），
            # 对并发影响相对较小。但如果 Qdrant 服务器负载高或网络延迟大，
            # 同样会导致事件循环被长时间阻塞。
            embeddings = memory_client.embedding_model.embed(query, "search")

            hits = memory_client.vector_store.search(
                query=query,
                vectors=embeddings,
                limit=10,
                filters=filters,
            )

            allowed = set(str(mid) for mid in accessible_memory_ids) if accessible_memory_ids else None

            results = []
            for h in hits:
                id, score, payload = h.id, h.score, h.payload
                if allowed and h.id is None or h.id not in allowed:
                    continue

                results.append({
                    "id": id,
                    "memory": payload.get("data"),
                    "hash": payload.get("hash"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "score": score,
                })

            for r in results:
                if r.get("id"):
                    access_log = MemoryAccessLog(
                        memory_id=uuid.UUID(r["id"]),
                        app_id=app.id,
                        access_type="search",
                        metadata_={
                            "query": query,
                            "score": r.get("score"),
                            "hash": r.get("hash"),
                        },
                    )
                    db.add(access_log)
            db.commit()

            return json.dumps({"results": results}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    """【MCP TOOL 实现】列出用户所有记忆"""
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # ⚠️【并发注意】memory_client.get_all() 是同步方法
            # 它会从向量数据库读取该用户的所有记忆，可能涉及大量 IO。
            # 在协程中直接调用会阻塞事件循环。
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={"hash": memory_data.get('hash')}
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={"hash": memory.get('hash')}
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="Delete specific memories by their IDs")
async def delete_memories(memory_ids: list[str]) -> str:
    """【MCP TOOL 实现】根据ID列表删除指定记忆"""
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            requested_ids = [uuid.UUID(mid) for mid in memory_ids]
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            ids_to_delete = [mid for mid in requested_ids if mid in accessible_memory_ids]

            if not ids_to_delete:
                return "Error: No accessible memories found with provided IDs"

            # ⚠️【并发注意】memory_client.delete() 是同步方法，会阻塞事件循环
            for memory_id in ids_to_delete:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            now = datetime.datetime.now(datetime.UTC)
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now

                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.active,
                        new_state=MemoryState.deleted
                    )
                    db.add(history)

                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="delete",
                        metadata_={"operation": "delete_by_id"}
                    )
                    db.add(access_log)

            db.commit()
            return f"Successfully deleted {len(ids_to_delete)} memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    """【MCP TOOL 实现】删除用户所有可访问的记忆"""
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # ⚠️【并发注意】批量删除涉及多次同步 IO 调用，阻塞时间较长
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()
            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 【FastAPI HTTP 适配层】将 MCP 协议映射到 HTTP 端点
# ───────────────────────────────────────────────────────────────────────────────
# 以下代码和 MCP 协议本身无关，是"适配器"代码 —— 让 MCP Server 可以通过 HTTP 被访问。
# 如果没有这些路由，MCP Server 只能在 stdio（本地进程）模式下运行。
#
# 【为谁服务？】→ 为网络通信服务（让 MCP Server 可以被远程 Client 访问）
# 【作用？】把 HTTP 请求转换成 MCP 协议消息，再交给 mcp._mcp_server 处理
#
# 通信流程（仔细看这个！）：
#   ┌─────────────┐  HTTP GET   ┌────────────────────────┐
#   │ MCP Client  │ ──────────► │  handle_sse()          │
#   │ (Claude等)  │             │  建立 SSE 长连接        │
#   └─────────────┘             └────────────────────────┘
#           ▲                                    │
#           │         SSE 推送响应               │
#           └────────────────────────────────────┘
#           │         POST 发送请求
#           └────────────────────────────────────┘
#
# 注意：SSE 是"单向"的（Server → Client），所以 Client 发请求要用 POST。
# ═══════════════════════════════════════════════════════════════════════════════

@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """
    【FastAPI HTTP 适配】处理 SSE 连接请求 —— MCP Client 的"入口"

    【为谁服务？】→ 为 MCP CLIENT 服务（Client 必须先连这个端点才能开始通信）
    【什么时候被调用？】MCP Client 启动时，发送 GET 请求建立长连接
    【作用？】
      1. 从 URL 提取 user_id、client_name，存入 contextvars
      2. 建立 SSE 长连接（Server 用这条连接给 Client 推送消息）
      3. 启动 mcp._mcp_server.run() —— 这是 MCP Server 的"主循环"

    mcp._mcp_server.run() 内部会：
      - 读取 Client 发来的 JSON-RPC 请求
      - 找到对应的 Tool 函数执行
      - 把结果通过 SSE 推回给 Client

    【并发与多用户接入】
      这个函数是 async def，意味着每个 SSE 连接是一个独立的协程。
      FastAPI + uvicorn 可以同时处理成百上千个这样的连接。

      多用户如何隔离？
        - 用户 A 连接：GET /mcp/app1/sse/user_A → contextvars 存 user_A
        - 用户 B 连接：GET /mcp/app2/sse/user_B → contextvars 存 user_B
        - 两个协程并行运行，互不干扰

      一个 MCP Client 连接的生命周期：
        1. 连接建立（这个函数被调用）→ mcp._mcp_server.run() 开始事件循环
        2. Client 发送工具调用请求 → POST /messages/
        3. Server 执行 Tool → 结果通过 SSE 推回
        4. 连接保持直到 Client 断开

    【调用机制深度解析：不是队列化，是并发竞争】
      很多人误以为"单例 = 队列化调用"，实际上不是。
      真实场景是：多个协程同时争夺同一个 memory_client。

      假设用户 A 和用户 B 同时请求 add_memories：

        时间线 →
        ├─ 协程 A: handle_sse() ──► mcp._mcp_server.run() ──► add_memories() ──► memory_client.add() ──┐
        │                                                                                                 │
        │  协程 B: handle_sse() ──► mcp._mcp_server.run() ──► add_memories() ──► memory_client.add() ──►┤
        │                                                                                                 │
        └─ 两个协程同时运行，同时调用同一个 memory_client 实例！                                        │
           memory_client.add() 是同步方法，协程 A 先抢到 GIL 开始执行，协程 B 被阻塞等待。            │
           当协程 A 执行 HTTP 请求时（socket IO），C 层面释放 GIL，协程 B 才有机会执行。              │
           这不是"队列化"，这是"竞争 + 协作式多任务"。                                                 │

      初始化时机：
        1. 模块导入：mcp = FastMCP(...) 创建工具注册表（无 Memory 实例）
        2. FastAPI 启动：setup_mcp_server(app) 挂载路由（仍无 Memory 实例）
        3. 第一次 Tool 调用：get_memory_client() 懒加载创建 Memory 单例
        4. 后续所有调用：直接返回缓存的 _memory_client，共享使用

      为什么是单例而不是每个请求创建一个？
        - Memory 初始化耗时：连接向量数据库 + 加载 LLM 客户端 ≈ 数秒
        - 连接池复用：HTTP 客户端有连接池，频繁创建/销毁效率低
        - 内存占用：每个实例都持有 embedding_model、vector_store、llm 等重对象
        - 配置一致性：所有请求使用相同的配置，避免状态分裂
    """
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")

    try:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # mcp._mcp_server.run() 是 MCP 协议的核心处理循环
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_post_message(request: Request):
    return await handle_post_message(request)

async def handle_post_message(request: Request):
    """
    【FastAPI HTTP 适配】处理 POST 消息 —— Client 发送 JSON-RPC 请求的"入口"

    【为谁服务？】→ 为 MCP CLIENT 服务（Client 通过 POST 发送 Tool 调用请求）
    【什么时候被调用？】当 AI 决定调用某个 Tool 时，Client 发送 POST 请求到这里
    【作用？】把 HTTP POST body（JSON-RPC 格式）转发给 MCP Server 处理

    JSON-RPC 请求示例（Client 发来的）：
    {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "tools/call",
      "params": {
        "name": "search_memory",
        "arguments": {"query": "用户喜欢什么"}
      }
    }
    """
    try:    
        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            return {}

        await sse.handle_post_message(request.scope, receive, send)

        return {"status": "ok"}
    finally:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 【并发与性能总结】
# ───────────────────────────────────────────────────────────────────────────────
# 这份代码的并发模型简图：
#
#   ┌─────────────┐      ┌─────────────┐      ┌─────────────────────┐
#   │  MCP Client │      │  MCP Client │      │  MCP Client (用户N)  │
#   │   (用户A)    │      │   (用户B)    │      │                     │
#   └──────┬──────┘      └──────┬──────┘      └──────────┬──────────┘
#          │                    │                        │
#          │ SSE 长连接          │ SSE 长连接              │ SSE 长连接
#          ▼                    ▼                        ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │                    FastAPI + uvicorn + asyncio                       │
#   │              （单进程事件循环，协程级并发）                             │
#   │                                                                     │
#   │   协程A(handle_sse)   协程B(handle_sse)   协程N(handle_sse)        │
#   │        │                    │                    │                 │
#   │        ▼                    ▼                    ▼                 │
#   │   contextvars A      contextvars B       contextvars N             │
#   │   user_id="Alice"    user_id="Bob"       user_id="Carol"           │
#   │        │                    │                    │                 │
#   │        └────────────────────┼────────────────────┘                 │
#   │                             ▼                                      │
#   │              共享同一个 memory_client（单例）                       │
#   │              ┌───────────────────────────────┐                     │
#   │              │ vector_store（HTTP客户端，线程安全）│ ◄── 并发安全    │
#   │              │ llm（HTTP客户端，线程安全）        │ ◄── 并发安全    │
#   │              │ db（SQLite + threading.Lock）   │ ◄── 会阻塞事件循环│
#   │              └───────────────────────────────┘                     │
#   └─────────────────────────────────────────────────────────────────────┘
#
# ⚠️ 性能瓶颈分析：
#   1. 同步调用阻塞：memory_client.add() 等同步方法阻塞整个事件循环
#   2. SQLite 锁：threading.Lock() 在 asyncio 中会暂停所有协程
#   3. LLM API 延迟：调用 OpenAI/Ollama 可能需要数秒，期间无法处理其他请求
#
# 【同步阻塞的本质：Python 线程 vs asyncio 事件循环】
#   很多人混淆了"线程阻塞"和"事件循环阻塞"：
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  time.sleep(5)                                                      │
#   │    → 阻塞 Python 线程                                               │
#   │    → GIL 被持有，其他线程无法执行 Python 代码                        │
#   │    → asyncio 事件循环完全暂停 5 秒                                  │
#   │    → ✅ 所有协程都无法运行                                          │
#   │                                                                     │
#   │  httpx.Client.send() (HTTP 请求)                                    │
#   │    → Python 线程被阻塞等待 socket 响应                              │
#   │    → C 层面释放 GIL（允许其他线程执行）                             │
#   │    → 但 asyncio 事件循环的调度器在 Python 层面，仍然被卡住          │
#   │    → ✅ 事件循环仍然暂停，其他协程无法被调度                        │
#   │    → ❌ "GIL 释放"只对多线程有用，对 asyncio 单线程没用            │
#   │                                                                     │
#   │  asyncio.sleep(5)                                                   │
#   │    → 挂起当前协程，把控制权交还给事件循环                           │
#   │    → 事件循环可以调度其他协程                                       │
#   │    → ✅ 这是 asyncio 的正确写法                                     │
#   └─────────────────────────────────────────────────────────────────────┘
#
# 【用户问题的精准答案】
#   是的！这就是最主要的问题。
#
#   FastAPI 基于 asyncio（单线程事件循环），但 mem0 内部全是同步代码。
#   当协程执行同步 HTTP 请求时：
#     1. Python 线程进入 socket 等待状态
#     2. C 层面的 socket IO 释放 GIL（Global Interpreter Lock）
#     3. 操作系统可以把 CPU 调度给其他线程
#     4. ❌ 但是！asyncio 的事件循环调度器在 Python 层面
#     5. ❌ 调度器所在的线程被卡住了，无法 yield 切换到其他协程
#     6. ❌ 结果：计算资源（CPU）确实让出来了，但 asyncio 的协程调度完全停滞
#
#   类比理解：
#     asyncio 事件循环 = 一个餐厅的前台调度员（只有 1 个人）
#     协程 = 等待服务的顾客
#     GIL 释放 = 调度员暂时离开前台去上厕所（CPU 可以做别的事）
#     但问题是：调度员"上厕所"是被迫的（因为同步阻塞），他不是主动交班的
#     所以他没有告诉其他顾客"请稍等，我先处理下一位"
#     所有顾客都在前台傻等，即使后厨（CPU）闲着。
#
#     如果换成 async HTTP（如 httpx.AsyncClient）：
#     调度员主动说"这位顾客需要等 3 秒，我先去服务下一位"，然后继续调度。
#
#   所以：在 asyncio 中调用任何同步阻塞代码（time.sleep、同步 HTTP、同步数据库），
#   无论它是否释放 GIL，都会阻塞事件循环，导致其他协程无法执行。
#
# 【用户追问：FastAPI 接口还活着吗？外部请求会怎样？】
#   问得很好！答案是：接口"半活着"——能接收到，但处理不了。
#
#   具体情况：
#     1. 新 TCP 连接：
#        → 操作系统内核的 listen backlog 队列可以暂存新连接（默认 128 个）
#        → 但 uvicorn 的事件循环被卡住，无法执行 accept()
#        → 新连接在内核队列中排队等待
#        → 如果队列满了，客户端会收到 "Connection refused"
#
#     2. 已建立的 SSE 长连接：
#        → SSE 需要定期发送心跳保持连接
#        → 事件循环被阻塞，心跳无法发送
#        → 客户端在等待响应时超时断开
#
#     3. 新 HTTP 请求：
#        → TCP 连接建立了，但请求数据读不到（因为 read 回调被阻塞）
#        → 如果前面有 nginx/负载均衡，超时后会返回 502/504
#
#   类比：餐厅前台被卡住了
#     - 新顾客可以走进餐厅（TCP 连接建立）
#     - 但前台调度员被卡在 A 顾客那里（事件循环阻塞）
#     - B 顾客点不了菜（请求处理不了）
#     - C 顾客等太久离开了（超时断开）
#     - D 顾客连门都进不来（backlog 队列满）
#
# ✅ 优化建议（生产环境）：
#   方案A：使用 asyncio.to_thread() 包裹同步调用
#     response = await asyncio.to_thread(memory_client.add, text, user_id=uid, ...)
#
#   方案B：改用 AsyncMemory（mem0 提供的异步版本）
#     from mem0 import AsyncMemory
#     memory_client = AsyncMemory.from_config(config)
#     response = await memory_client.add(...)
#
#   方案C：多进程部署
#     启动多个 uvicorn worker（uvicorn main:app --workers 4）
#     每个进程有独立的事件循环和 Memory 实例
#
#   方案D：连接池调优
#     增加 vector_store 和 LLM 客户端的连接池大小
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 【FastAPI 集成】将 MCP 路由挂载到 FastAPI 应用
# ───────────────────────────────────────────────────────────────────────────────
# 这个函数在 FastAPI 应用启动时调用，把 MCP 相关的 HTTP 路由注册进去。
# 这样 MCP Server 就和普通的 REST API 共存在一个应用中。
#
# 【为谁服务？】→ 为 FastAPI 应用服务（启动注册）
# ═══════════════════════════════════════════════════════════════════════════════
def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"
    app.include_router(mcp_router)
