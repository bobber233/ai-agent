import asyncio
from typing import Annotated
# 核心变化：导入最新的 ddgs 统一入口
from ddgs import DDGS 
from mcp.shared.exceptions import McpError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ErrorData,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from pydantic import BaseModel, Field

class SearchSchema(BaseModel):
    """DuckDuckGo 搜索参数校验模型"""
    query: Annotated[
        str, 
        Field(description="需要搜索的关键字或短语，支持高级搜索语法")
    ]
    max_results: Annotated[
        int,
        Field(
            default=5,
            description="返回的最大搜索结果条数",
            gt=0,
            le=20,
        ),
    ]

def _blocking_ddg_search(query: str, max_results: int) -> str:
    """同步阻塞的 DuckDuckGo 检索核心逻辑"""
    try:
        # 使用最新版的统一 DDGS 上下文管理器
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            
            if not results:
                return f"未能在互联网上找到关于 '{query}' 的任何相关结果。"
            
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] 标题: {r.get('title', '无标题')}\n"
                    f"    链接: {r.get('href', '无链接')}\n"
                    f"    摘要: {r.get('body', '无摘要')}\n"
                )
            return "\n".join(formatted_results)
            
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"DuckDuckGo 引擎检索发生异常: {str(e)}"
        ))

async def execute_ddg_search(query: str, max_results: int) -> str:
    """使用 asyncio.to_thread 将同步请求桥接到异步事件循环中，防止阻塞 stdio 管道"""
    return await asyncio.to_thread(_blocking_ddg_search, query, max_results)

async def serve() -> None:
    """初始化并运行 DuckDuckGo MCP 服务器"""
    server = Server("mcp-duckduckgo")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="web_search",
                description="""使用 DuckDuckGo 检索全网。支持并强烈建议大模型使用以下高级搜索语法来提高精准度：
                1. site:域名 -> 限制在特定网站内搜索。技术问题优先使用 site:github.com, site:huggingface.co, site:reddit.com。
                2. "关键词" -> 双引号进行精确匹配，防止词组被拆散。例如搜索特定报错：\"TypeError: not all arguments converted\"。
                3. intitle:关键词 -> 强迫网页标题必须包含该词。
                4. filetype:pdf/md -> 寻找技术白皮书或文档。
                示例：如果你要找 Qwen2.5-Coder 的量化特性，应该组合输入：'site:github.com "qwen2.5-coder" 7b instruct'""",
                inputSchema=SearchSchema.model_json_schema(),
            )
        ]

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="web_search",
                description="向互联网检索指定主题的最新技术演进或事实信息",
                arguments=[
                    PromptArgument(
                        name="query", description="搜索关键词", required=True
                    )
                ],
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name != "web_search":
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"未知工具: {name}"))
            
        try:
            args = SearchSchema(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"参数解析失败: {str(e)}"))

        query = args.query.strip()
        if not query:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="搜索关键词 'query' 不能为空"))

        content = await execute_ddg_search(query, args.max_results)
        return [TextContent(type="text", text=content)]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
        if name != "web_search":
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"未知 Prompt: {name}"))
            
        if not arguments or "query" not in arguments:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="参数 'query' 是必填项"))

        query = arguments["query"]
        
        try:
            content = await execute_ddg_search(query, max_results=5)
        except McpError as e:
            return GetPromptResult(
                description=f"对 '{query}' 执行互联网检索失败",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"检索失败原因: {str(e)}"),
                    )
                ],
            )

        return GetPromptResult(
            description=f"关于 '{query}' 的互联网检索结果",
            messages=[
                PromptMessage(
                    role="user", 
                    content=TextContent(type="text", text=f"以下是通过 DuckDuckGo 检索到的最新上下文，请基于此进行分析解答:\n\n{content}")
                )
            ],
        )

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=False)

async def main():
    await serve()

if __name__ == "__main__":
    asyncio.run(main())