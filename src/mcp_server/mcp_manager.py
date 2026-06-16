import os
import sys
import asyncio
import json
from pathlib import Path
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai.types.chat import (
    ChatCompletionToolParam,
)
from src.utils.logger import logger

def _prepare_mcp_and_messages() -> dict[str, StdioServerParameters]:
    """ 动态发现 MCP 服务
    
    Returns:
        dict: MCP 服务名到启动参数的映射表
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.normpath(os.path.join(current_dir, "..", "mcp_server"))
    server_configs: dict[str, StdioServerParameters] = {}
    if not os.path.exists(server_dir):
        logger.info(f"错误: 找不到指定的 server 目录路径 -> {server_dir}")
    
    for file in os.listdir(server_dir):
        file_path = Path(file)
        if (
            file.startswith("mcp_server")
            and file_path.suffix == ".py"
        ):
            server_name = file_path.stem
            server_path = os.path.join(server_dir, file)
            server_configs[server_name] = StdioServerParameters(
                command=sys.executable,
                args=[server_path]
            )

    return server_configs

async def initialize_all_servers(stack: AsyncExitStack) -> tuple[dict[str, ClientSession], list[ChatCompletionToolParam]]:
    """ 动态发现并初始化 MCP 服务，并生成统一的工具路由表
    
    Args:
        stack: 外部传入的异步上下文管理器栈
    Returns:
        tuple: 工具名到会话的路由映射以及可供 llm 使用的工具参数列表
    """
    server_configs: dict[str, StdioServerParameters] = _prepare_mcp_and_messages()
    tool_to_session: dict[str, ClientSession] = {}
    llm_tools: list[ChatCompletionToolParam] = []

    for server_name, server_params in server_configs.items():
        try:
            read_channel, write_channel = await stack.enter_async_context(stdio_client(server_params))
            server_session = await stack.enter_async_context(ClientSession(read_channel, write_channel))
            await server_session.initialize()
            logger.info(f"  ├─ [连接成功]: MCP服务 {server_name}")
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

async def call_mcp_tool(tool_call, tool_to_session: dict[str, ClientSession]) -> str:
    """ 执行 mcp 服务下的工具获取工具执行结果，并返回给 LLM
    Args:
        tool_call: 模型输出的工具调用参数
        tool_to_session: 工具名到 MCP 会话的映射表
    Returns:
        str: 工具执行结果文本
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
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
                    execution_result = ""
                    if isinstance(mcp_result.content[0], TextContent):
                        execution_result = mcp_result.content[0].text
                except Exception as e:
                    execution_result = f"<error>本地 MCP 工具执行失败: {str(e)}</error>"
            else:
                execution_result = f"<error>网关中未注册名为 [{tool_name}] 的服务</error>"
            return execution_result
        except Exception as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(1)
    return ""
