"""
LLM + MCP Agent — openai-agents SDK 原生 MCP 集成 + 第三方 LLM 配置
═══════════════════════════════════════════════════════════════════

【核心代码量对比】

  旧写法 (手动 ReAct 循环):
    ① 手动调 tools/list 拿工具列表
    ② 手动写 MCP → OpenAI Function Calling 格式转换函数
    ③ 手动写 while msg.tool_calls: 循环
    ④ 手动把 tool result 序列化塞回 messages
    ⑤ 手动管理 tool_call_id 对应关系
    → ~80 行胶水代码

  新写法 (openai-agents SDK):
    async with MCPServerStreamableHttp(...) as mcp_server:
        agent = Agent(mcp_servers=[mcp_server], ...)
        result = await Runner.run(agent, user_input)
    → 3 行核心代码，其余都是注释

【第三方 LLM 兼容配置】

  SDK 默认使用 OpenAI 专有的 Responses API，与 DeepSeek / 阿里云等不兼容。
  只需两行配置切换到 Chat Completions API（详见 README.md 中的完整分析）：

    set_default_openai_api("chat_completions")
    set_default_openai_client(AsyncOpenAI(api_key=..., base_url=...))

  这两行之后，Runner.run() 完全走行业标准的 Chat Completions 路径，
  DeepSeek、阿里云、Ollama 等任何 OpenAI-compatible 提供商都能用。

【阅读顺序】
  1. _configure_deepseek() — 理解如何把 SDK 切换到第三方 LLM
  2. MCPServerStreamableHttp 参数 — 一行 url 连接 MCP Server
  3. Agent(mcp_servers=[...]) — MCP Server 直接当工具源注入
  4. Runner.run() — 整个对话循环被这一行替代
  5. main() — 交互式对话循环
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.mcp import MCPServerStreamableHttp

load_dotenv()


def _configure_deepseek() -> tuple[str, str]:
    """
    配置 openai-agents SDK 使用 DeepSeek 作为 LLM 后端。

    返回 (model_name, base_url) 供显示用。

    【为什么需要这两步？】
      openai-agents SDK 默认：
        - 使用 OpenAI 专有的 /v1/responses API（DeepSeek 不支持）
        - 连接 api.openai.com（DeepSeek 在 api.deepseek.com）

      set_default_openai_api("chat_completions")
        → 强制 SDK 使用 /v1/chat/completions API（行业标准，所有 LLM 都支持）

      set_default_openai_client(custom_client)
        → 把底层 HTTP 客户端指向 DeepSeek 的 base_url + api_key

      这两步之后，SDK 的所有内部调用链（模型推理、工具调用决策等）
      全部通过 DeepSeek 完成，不需要 OpenAI 账号。

    【如果用其他 LLM？】
      只需改 base_url 和 api_key：
        - 阿里云 DashScope: base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        - 本地 Ollama:      base_url = "http://localhost:11434/v1"
        - OpenRouter:       base_url = "https://openrouter.ai/api/v1"
    """
    api_key = os.getenv("OPENAI_llm_API_KEY")
    base_url = (os.getenv("OPENAI_llm_URL") or "https://api.deepseek.com") + "/v1"

    if not api_key:
        raise RuntimeError("请设置 OPENAI_llm_API_KEY 环境变量")

    # ① 创建指向 DeepSeek 的异步 HTTP 客户端
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ② 设为全局默认（所有 Agent 都用它）
    set_default_openai_client(client, use_for_tracing=False)
    #    use_for_tracing=False: DeepSeek 的 API key 不能向 OpenAI 上传 trace 数据

    # ③ 关键：切换到 Chat Completions API（DeepSeek 支持，OpenAI Responses 不支持）
    set_default_openai_api("chat_completions")

    model = os.getenv("OPENAI_llm_Model", "deepseek-chat")
    return model, base_url


async def main(user_id: str = "u_agent_demo") -> None:
    model, base_url = _configure_deepseek()

    print(f"LLM: {model}")
    print(f"API: {base_url}")
    print(f"User ID: {user_id}")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # MCP Server 直接作为工具源注入 Agent
    # ═══════════════════════════════════════════════════════════════════════
    #
    # MCPServerStreamableHttp 是 openai-agents SDK 内置的 MCP 客户端。
    # 它内部自动做了：
    #   1. 连接 MCP Server → initialize 握手
    #   2. tools/list → 获取所有工具定义
    #   3. MCP 工具定义 → OpenAI Function Calling 格式转换
    #   4. tools/call → 执行 LLM 决定的工具调用
    #
    # 这些步骤全部由 SDK 内部消化，业务代码不需要关心。

    async with MCPServerStreamableHttp(
        params={
            "url": "http://localhost:8765/mcp",  # MCP Server 地址
            "timeout": 30.0,                       # HTTP 超时
        },
        cache_tools_list=True,     # 缓存工具列表（12 个工具不会变）
        client_session_timeout_seconds=30.0,  # MCP 工具执行允许更长等待
    ) as mcp_server:

        # ── 创建 Agent ──
        # mcp_servers=[mcp_server]：把 MCP Server 当工具源注入。
        # SDK 会自动向 MCP Server 请求工具列表并注入到 LLM 的上下文中。
        agent = Agent(
            name="Memory Assistant",
            instructions=(
                f"You are an AI assistant with persistent memory. "
                f"The current user_id is \"{user_id}\". "
                f"EVERY time the user shares anything about themselves — "
                f"preferences, facts, plans, experiences — use add_memory. "
                f"Don't wait to be asked. "
                f"Before answering personal questions, use search_memories first. "
                f"For search, prefer limit=5-10."
            ),
            mcp_servers=[mcp_server],  # ← MCP Server 直接注入
            model=model,
        )

        # ── 对话循环 ──
        # Runner.run() 内部自动完成：
        #   user input → LLM 读 tools → LLM 决定调哪个 → MCP 执行 → 结果还给 LLM
        #   → LLM 可能继续调更多工具 → 最终返回自然语言回复
        print("开始对话（输入 'quit' 或 'exit' 退出）")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 退出对话")
                break

            if user_input.lower() in ("quit", "exit"):
                print("👋 退出对话")
                break

            result = await Runner.run(agent, user_input)
            print(f"\n🤖 Assistant: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
