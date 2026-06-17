from typing import Annotated, Tuple
from urllib.parse import urlparse, urlunparse
import asyncio
import markdownify
import readabilipy.simple_json
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
from protego import Protego
from pydantic import BaseModel, Field, AnyUrl

DEFAULT_USER_AGENT_AUTONOMOUS = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
DEFAULT_USER_AGENT_MANUAL = "ModelContextProtocol/1.0 (User-Specified; +https://github.com/modelcontextprotocol/servers)"


def extract_content_from_html(html: str) -> str:
    """提取 HTML 内容并转换为 Markdown 格式。"""
    ret = readabilipy.simple_json.simple_json_from_html_string(
        html, use_readability=True
    )
    if not ret["content"]:
        return "<error>页面 HTML 结构简化失败，无法解析核心正文。</error>"
    content = markdownify.markdownify(
        ret["content"],
        heading_style=markdownify.ATX,
    )
    return content


def get_robots_txt_url(url: str) -> str:
    """获取给定网站的 robots.txt URL。"""
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    return robots_url


async def check_may_autonomously_fetch_url(
    url: str,
    user_agent: str,
    proxy_url: str | None = None
) -> None:
    """检查是否允许自主抓取该 URL。"""
    from httpx import AsyncClient, HTTPError

    robot_txt_url = get_robots_txt_url(url)

    async with AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                robot_txt_url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except HTTPError:
            raise McpError(ErrorData(
                code=INTERNAL_ERROR,
                message=f"由于连接问题，获取 robots.txt ({robot_txt_url}) 失败",
            ))
        if response.status_code in (401, 403):
            raise McpError(ErrorData(
                code=INTERNAL_ERROR,
                message=f"抓取 robots.txt ({robot_txt_url}) 时返回状态码 {response.status_code}。系统默认不允许自主抓取，请让用户通过 UI 界面手动触发 fetch 提示词模板来获取内容。",
            ))
        elif 400 <= response.status_code < 500:
            return
        robot_txt = response.text
    processed_robot_txt = "\n".join(
        line for line in robot_txt.splitlines() if not line.strip().startswith("#")
    )
    robot_parser = Protego.parse(processed_robot_txt)
    if not robot_parser.can_fetch(str(url), user_agent):
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"该网站的 robots.txt ({robot_txt_url}) 声明了禁止自主抓取当前页面。\n"
            f"<useragent>{user_agent}</useragent>\n"
            f"<url>{url}</url>\n"
            f"你必须立即告诉用户：由于目标网站的 robots.txt 限制，自主抓取失败。引导用户通过 UI 界面手动触发 fetch 模板来绕过此限制。",
        ))


async def fetch_url(
    url: str,
    user_agent: str,
    force_raw: bool = False,
    proxy_url: str | None = None
) -> Tuple[str, str]:
    """执行网页请求并返回清洗后的文本内容。"""
    from httpx import AsyncClient, HTTPError

    async with AsyncClient(proxy=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
        except HTTPError as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"请求 URL 失败 {url}: {e!r}"))
        if response.status_code >= 400:
            raise McpError(ErrorData(
                code=INTERNAL_ERROR,
                message=f"请求 URL 失败 {url} - 状态码 {response.status_code}",
            ))

        page_raw = response.text

    content_type = response.headers.get("content-type", "")
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html and not force_raw:
        return extract_content_from_html(page_raw), ""

    return (
        page_raw,
        f"当前内容类型 ({content_type}) 无法简化为 Markdown，已为你展示原始文本数据：\n",
    )


class Fetch(BaseModel):
    """网页深度读取参数校验模型"""
    url: Annotated[AnyUrl, Field(description="需要精读和全量提取内容的网页 URL 链接")]
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="本轮单次返回的最大字符数，防止上下文窗口过载。",
            gt=0,
            lt=1000000,
        ),
    ]
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="返回内容的起始字符索引。如果上一次读取被截断，可传入此参数继续向下追加读取。",
            ge=0,
        ),
    ]
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="是否强制获取未经任何清洗简化的原生 HTML 源码。",
        ),
    ]


async def serve(
    custom_user_agent: str | None = None,
    ignore_robots_txt: bool = False,
    proxy_url: str | None = None,
) -> None:
    server = Server("mcp-fetch")
    user_agent_autonomous = custom_user_agent or DEFAULT_USER_AGENT_AUTONOMOUS
    user_agent_manual = custom_user_agent or DEFAULT_USER_AGENT_MANUAL

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="fetch",
                # 💡 核心修改：改为纯中文，并与你的系统提示词战术形成联动暗示
                description="""深度网页/文档通读器。输入一个具体的 URL 网址，全量提取其核心内容并转换为高信息密度的 Markdown 纯文本。
【战术定位】当广度检索工具返回了高价值的线索链接（如官方文档、GitHub 核心 Issue、技术白皮书）后，你必须调用本工具直接点进该网址通读全文，以彻底消灭两三行检索摘要带来的信息盲区。""",
                inputSchema=Fetch.model_json_schema(),
            )
        ]

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="fetch",
                description="精读指定 URL 网页并将核心正文提取为 Markdown 文本",
                arguments=[
                    PromptArgument(
                        name="url", description="需要精读的目标网页 URL 链接", required=True
                    )
                ],
            )
        ]

    @server.call_tool()
    async def call_tool(name, arguments: dict) -> list[TextContent]:
        try:
            args = Fetch(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        url = str(args.url)
        if not url:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="参数错误：URL 不能为空"))

        if not ignore_robots_txt:
            await check_may_autonomously_fetch_url(url, user_agent_autonomous, proxy_url)

        content, prefix = await fetch_url(
            url, user_agent_autonomous, force_raw=args.raw, proxy_url=proxy_url
        )
        original_length = len(content)
        if args.start_index >= original_length:
            content = "<error>没有更多内容可供读取。</error>"
        else:
            truncated_content = content[args.start_index : args.start_index + args.max_length]
            if not truncated_content:
                content = "<error>没有更多内容可供读取。</error>"
            else:
                content = truncated_content
                actual_content_length = len(truncated_content)
                remaining_content = original_length - (args.start_index + actual_content_length)
                
                # 💡 提示中文化
                if actual_content_length == args.max_length and remaining_content > 0:
                    next_start = args.start_index + actual_content_length
                    content += f"\n\n<error>【提示】由于单次长度限制，内容已被截断。若想继续阅读后续正文，请保持其他参数不变，传入 start_index={next_start} 再次调用本工具。</error>"
        return [TextContent(type="text", text=f"{prefix}网址 {url} 的全量网页干货内容如下：\n{content}")]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
        if not arguments or "url" not in arguments:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="参数错误：URL 不能为空"))

        url = arguments["url"]

        try:
            content, prefix = await fetch_url(url, user_agent_manual, proxy_url=proxy_url)
        except McpError as e:
            return GetPromptResult(
                description=f"抓取网页失败 {url}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"网页拉取发生异常: {str(e)}"),
                    )
                ],
            )
        return GetPromptResult(
            description=f"网址 {url} 的页面干货内容",
            messages=[
                PromptMessage(
                    role="user", content=TextContent(type="text", text=f"{prefix}{content}")
                )
            ],
        )

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=False)

async def main():
    await serve(
        custom_user_agent = None,
        ignore_robots_txt = False,
        proxy_url = None,
    )

if __name__ == "__main__":
    asyncio.run(main())