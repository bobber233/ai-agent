import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any, cast, AsyncGenerator
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionToolMessageParam
)
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from src.agents.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """# Role
你是一个具备深度思考、全网检索与本地工具调用能力的 AI 智能体。

# Workflow
必须严格按以下链路循环执行，直至获取最终答案：
Thought -> Action (Tool Calls) -> Observation -> Final Answer

# Two-Stage Retrieval Tactics
环境提供【互联网检索】和【网页读取】工具时，面对技术报错/复杂查询须严格执行：
1. 广度探路：优先调用互联网检索工具，目标是获取高价值的源头链接（URL），严禁单方面依赖摘要。
2. 深度破局：挑选最权威的 URL，立即调用网页读取工具抓取全量文本，消灭知识盲区。

# Constraints
1. 搜索预算：联网检索工具调用上限绝对不能超过 3 次。
2. 严禁无脑复读：检索不佳时，须更换关键词维度或直接改用全文读取工具。
3. 降级退出：2-3 次检索深挖后若无果，立即停止调用网络工具，基于已有碎片信息进行严密逻辑推导，并在回答中坦诚说明。
4. 当前年份：2026 年。

# Output Style
- 直奔主题，拒绝一切客套话。
- 极度专业、严谨，优先使用 Markdown（代码块、列表、表格）。"""

openai_client = AsyncOpenAI(base_url=settings.MODEL_BASE_URL, api_key="ollama")

async def _call_mcp_tool(
    tool_call: ChatCompletionMessageCustomToolCallParam,
    tool_to_session: dict[str, ClientSession],
) -> str:
    """Call an MCP tool with robust error handling and retries."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            tool_id = tool_call["id"]
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            
            # 解析参数
            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
            except json.JSONDecodeError:
                tool_args = {}
            
            # 路由寻找绑定的 MCP 实例
            target_session = tool_to_session.get(tool_name)
            
            if target_session:
                try:
                    # 物理触发 MCP 服务的 Tool 能力
                    mcp_result = await target_session.call_tool(tool_name, tool_args)
                    # 取出服务返回的纯文本
                    execution_result = mcp_result.content[0].text
                except Exception as e:
                    execution_result = f"<error>本地 MCP 工具执行失败: {str(e)}</error>"
            else:
                execution_result = f"<error>网关中未注册名为 [{tool_name}] 的服务</error>"
            return execution_result
        except Exception as e:
            logger.warning("Attempt %s to call tool %s failed: %s", attempt, tool_name, e)
            if attempt == max_retries:
                raise
            await asyncio.sleep(1)
    return ""

def _prepare_mcp_and_messages() -> dict[str, StdioServerParameters]:
    """ 动态发现 MCP 服务
    
    Returns:
        dict: MCP 服务名到启动参数的映射表
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.normpath(os.path.join(current_dir, "..", "server"))
    server_configs: dict[str, StdioServerParameters] = {}
    if not os.path.exists(server_dir):
        logger.info(f"错误: 找不到指定的 server 目录路径 -> {server_dir}")
    
    for file in os.listdir(server_dir):
        file_path = Path(file)
        if (
            file.startswith("mcp_server")
            and file_path.suffix == ".py"
        ):
            server_name = file_path.stem.replace("mcp_server", "")
            server_path = os.path.join(server_dir, file)
            server_configs[server_name] = StdioServerParameters(
                command=sys.executable,
                args=[server_path]
            )

    return server_configs

async def initialize_all_servers(
    server_configs: dict[str, StdioServerParameters],
    stack: AsyncExitStack
) -> tuple[dict[str, ClientSession], list[ChatCompletionToolParam]]:
    """ 并行初始化所有动态发现的 MCP 服务，并生成统一的工具路由表
    
    Args:
        server_configs: 包含服务名和启动参数的字典
        stack: 外部传入的异步上下文管理器栈

    Returns:
        tuple: 工具名到会话的路由映射以及可供 llm 使用的工具参数列表
    """
    tool_to_session: dict[str, ClientSession] = {}
    llm_tools: list[ChatCompletionToolParam] = []

    for server_name, server_params in server_configs.items():
        try:
            read_channel, write_channel = await stack.enter_async_context(stdio_client(server_params))
            server_session = await stack.enter_async_context(ClientSession(read_channel, write_channel))
            await server_session.initialize()
            logger.info(f"  ├─ [连接成功]: 服务 {server_name}")
            mcp_tools = await server_session.list_tools()
            for tool in mcp_tools.tools:
                tool_to_session[tool.name] = server_session
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or '',
                        "parameters": tool.inputSchema,
                    },
                })
        except Exception as e:
            logger.info(f"[启动失败]: 服务 {server_name} 发生异常, 错误原因: {e}")

    return tool_to_session, llm_tools

async def run_agent(
    user_message: str,
    max_iterations: int = 5
) -> AsyncGenerator[str, None]:
    server_configs = _prepare_mcp_and_messages()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    async with AsyncExitStack() as stack:
        tool_to_session, llm_tools = await initialize_all_servers(server_configs, stack)
        for _ in range(1, max_iterations + 1):
            text_content = ""
            calls_buffer = {}
            tool_calls: list[ChatCompletionMessageCustomToolCallParam] | None = None
            try:
                response = await openai_client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=messages,
                    tools=llm_tools if llm_tools else [],
                    tool_choice="auto",
                    stream=True,
                    extra_body={
                        "options": {
                            "num_ctx": 4096,       # 限制上下文，防止 6G 显存频繁 OOM 导致卡顿
                            "num_predict": 2048,
                            "temperature": 0.3,
                        }
                    }
                )
            except Exception as e:
                logger.error(f"调用模型接口失败: {e}")
                yield f"\n[系统提示]: 模型接口调用失败，错误信息: {e}"
                return

            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    text_content += delta.content
                    yield delta.content
                
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in calls_buffer:
                            calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            calls_buffer[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            yield f"\n[Agent 状态]: 准备执行 -> 【{tc.function.name}】...\n"
                            calls_buffer[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            calls_buffer[idx]["arguments"] += tc.function.arguments
                
            if calls_buffer:
                tool_calls = cast(list[ChatCompletionMessageCustomToolCallParam], [
                    {
                        "id": v["id"],
                        "type": "function",
                        "function": {"name": v["name"], "arguments": v["arguments"]}
                    }
                    for v in calls_buffer.values()
                ])
                
            # 4. 组装 Assistant 消息并存入历史
            assistant_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": text_content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                for tool_call in tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["function"]["name"]
                    yield f"[Agent 状态]: 正在执行【{tool_name}】，请稍候...\n"
                    execution_result = await _call_mcp_tool(tool_call, tool_to_session)
                    yield f"[Agent 状态]: 【{tool_name}】执行完毕。正在读入观测数据...\n"
                    # 按照 OpenAI 严格的标准，将执行结果包装成 tool 角色塞回上下文中
                    tool_message: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": execution_result
                    }
                    messages.append(tool_message)
            else:
                # 如果没有工具调用，说明模型输出了最终答案（Final Answer）或者是纯文本交互
                if text_content.strip():
                    messages.append(assistant_message)
                return # 只有在确认完成了本轮纯文本输出后才退出
            
            

        yield f"\n[系统提示]: 已达到最大思考步数限制（{max_iterations} 步），任务强行终止。"
