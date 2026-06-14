import os
import sys
import asyncio
import json
import logging
from typing import Any, Dict, List, AsyncGenerator
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from src.agents.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

openai_client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")

async def _call_mcp_tool(
    session: ClientSession,
    tool_name: str,
    tool_args: Dict[str, Any]
) -> str:
    """Call an MCP tool with robust error handling and retries."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            result = await session.call_tool(tool_name, arguments=tool_args)
            if hasattr(result, "content") and result.content:
                first = result.content[0]
                if isinstance(first, str):
                    return first
            return str(result)
        except Exception as e:
            logger.warning("Attempt %s to call tool %s failed: %s", attempt, tool_name, e)
            if attempt == max_retries:
                raise
            await asyncio.sleep(1)
    return ""

async def _prepare_mcp_and_messages(user_message: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "..", "server", "weather_server.py")
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path]
    )
    return server_params

async def run_agent(user_message: str, stream: bool = False) -> AsyncGenerator[str, None] | str:
    if stream:
        return run_agent_stream(user_message)
    else:
        return await run_agent_sync(user_message)

async def run_agent_sync(user_message: str) -> str:
    server_params = await _prepare_mcp_and_messages(user_message)

    async with stdio_client(server_params) as (read_channel, write_channel):
        async with ClientSession(read_channel, write_channel) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            llm_tools: List[ChatCompletionToolParam] = []

            for tool in mcp_tools.tools:
                parameters: dict[str, object] = tool.inputSchema
                description: str = tool.description or ''
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": parameters,
                    },
                })

            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": user_message}]
            
            response = await openai_client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                tools=llm_tools,
                tool_choice="auto",
            )
            response_msg = response.choices[0].message

            if response_msg.tool_calls:
                assistant_msg: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": response_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (response_msg.tool_calls or [])
                        if isinstance(tc, ChatCompletionMessageToolCall)
                    ],
                }
                messages.append(assistant_msg)
                
                for tool_call in response_msg.tool_calls:
                    if isinstance(tool_call, ChatCompletionMessageToolCall):
                        tool_output = await _call_mcp_tool(session, tool_call.function.name, json.loads(tool_call.function.arguments))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output,
                        })

                final_response = await openai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=messages,
                )
                return final_response.choices[0].message.content or ""
            else:
                return response_msg.content or ""

async def run_agent_stream(user_message: str) -> AsyncGenerator[str, None]:
    server_params = await _prepare_mcp_and_messages(user_message)

    async with stdio_client(server_params) as (read_channel, write_channel):
        async with ClientSession(read_channel, write_channel) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            llm_tools: List[ChatCompletionToolParam] = []

            for tool in mcp_tools.tools:
                parameters: dict[str, object] = tool.inputSchema
                description: str = tool.description or ''
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": parameters,
                    },
                })

            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            
            response = await openai_client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                tools=llm_tools,
                tool_choice="auto",
            )
            response_msg = response.choices[0].message

            if response_msg.tool_calls:
                assistant_msg: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": response_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (response_msg.tool_calls or [])
                        if isinstance(tc, ChatCompletionMessageToolCall)
                    ],
                }
                messages.append(assistant_msg)
                
                for tool_call in response_msg.tool_calls:
                    if isinstance(tool_call, ChatCompletionMessageToolCall):
                        tool_output = await _call_mcp_tool(session, tool_call.function.name, json.loads(tool_call.function.arguments))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output,
                        })
                stream_response = await openai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=messages,
                    stream=True,
                )
                async for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response_msg.content or ""

if __name__ == "__main__":
    reply = asyncio.run(run_agent("帮我查一下北京的天气怎么样？"))
    print(reply)
