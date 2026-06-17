from contextlib import AsyncExitStack
from typing import AsyncGenerator
from openai import AsyncOpenAI
from mcp import ClientSession
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from src.agents.core.config import settings
from src.utils.logger import logger
from src.mcp_server.mcp_manager import call_mcp_tool
from src.utils.intent_router import intent_router
from src.mcp_server.mcp_server_knowledge import _blocking_vector_search

openai_client = AsyncOpenAI(base_url=settings.MODEL_BASE_URL, api_key="ollama")

async def consume_and_parse(response):
    """ 消费模型的流式输出，动态解析文本内容和工具调用指令，并以增量方式 yield 给前端
    Args:
    response: 模型接口返回的异步生成器，包含增量文本和工具调用指令

    Yields:
    str: 模型输出的增量文本或工具调用状态更新
    tuple: 最终的 (text_content, tool_calls) 供后续处理工具
    """
    text_content = ""
    tool_calls: list[ChatCompletionMessageToolCallParam] = []
    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            text_content += delta.content
            yield delta.content
        
        if delta.tool_calls:
            calls_buffer: dict = {}
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in calls_buffer:
                    calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    calls_buffer[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    yield f"\n准备执行 -> 【{tc.function.name}】...\n"
                    calls_buffer[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    calls_buffer[idx]["arguments"] += tc.function.arguments
            if calls_buffer:
                tool_calls = [{
                    "id": v["id"],
                    "type": "function",
                    "function": {"name": v["name"], "arguments": v["arguments"]}
                } for v in calls_buffer.values()]

    yield text_content, tool_calls

async def execute_all_tools(
    tool_calls: list[ChatCompletionMessageToolCallParam],
    tool_to_session: dict[str, ClientSession],
    messages: list[ChatCompletionMessageParam]
) -> AsyncGenerator[str, None]:
    """ 遍历工具调用列表，执行每个工具并将结果封装成消息追加到上下文中
    Args:
    tool_calls: 模型输出的工具调用列表
    tool_to_session: 工具名到 MCP 会话的映射表
    messages: 当前的对话消息列表，工具执行结果将以 tool 角色的消息追加到其中
    """
    for tool_call in tool_calls:
        tool_id = tool_call.get('id')
        tool_name = tool_call.get('function').get('name')
        yield f"正在执行【{tool_name}】，请稍候...\n"
        execution_result = await call_mcp_tool(tool_call, tool_to_session)
        yield f"【{tool_name}】执行完毕。正在思考...\n"
        # 按照 OpenAI 严格的标准，将执行结果包装成 tool 角色塞回上下文中
        tool_message: ChatCompletionToolMessageParam = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": execution_result
        }
        messages.append(tool_message)

async def run_agent(
    user_message: str,
    system_prompt: str,
    *,
    tool_to_session: dict[str, ClientSession],
    llm_tools: list[ChatCompletionToolParam],
    max_iterations: int = 5
) -> AsyncGenerator[str, None]:
    """ 主函数：运行智能体思考流程，处理工具调用并返回最终答案
    Args:
        user_message: 用户输入的初始消息
        max_iterations: 智能体思考的最大循环次数，防止死循环
    Yields:
        str: 每次模型输出的增量文本或工具调用状态更新
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    async with AsyncExitStack():
        for _ in range(1, max_iterations + 1):
            text_content = ""
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            try:
                logger.info(f"  ├─ [思考循环]: 第 {_} 轮思考开始...")
                response = await openai_client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=messages,
                    tools=llm_tools if llm_tools else [],
                    tool_choice="auto",
                    stream=True,
                )
            except Exception as e:
                logger.error(f"调用模型接口失败: {e}")
                yield f"\n[系统提示]: 模型接口调用失败，错误信息: {e}"
                return

            async for item in consume_and_parse(response):
                if isinstance(item, str):
                    yield item
                elif isinstance(item, tuple):
                    text_content, tool_calls = item
                
            # 4. 组装 Assistant 消息并存入历史
            assistant_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": text_content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                async for item in execute_all_tools(tool_calls, tool_to_session, messages):
                    yield item
            else:
                # 如果没有工具调用，说明模型输出了最终答案（Final Answer）或者是纯文本交互
                if text_content.strip():
                    messages.append(assistant_message)
                return # 只有在确认完成了本轮纯文本输出后才退出

        yield f"\n[系统提示]: 已达到最大思考步数限制（{max_iterations} 步），任务强行终止。"
