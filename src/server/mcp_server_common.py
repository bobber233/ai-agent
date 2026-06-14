import asyncio
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

server = Server("mcp-common")

# 1. 注册工具
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="fetch_weather",
            description="获取指定城市的实时天气数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如：北京、东京"}
                },
                "required": ["city"],
            },
        )
    ]

# 2. 实现工具逻辑
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    if name == "fetch_weather":
        city = arguments.get("city", "未知城市")
        # 这里本应调用真实的气象 API，现在用模拟数据替代
        weather_info = f"{city}今天的天气是：晴，22°C，微风。"
        return [types.TextContent(type="text", text=weather_info)]
    raise ValueError(f"找不到工具: {name}")

async def main():
    # 启动服务器，使用 stdio 传输协议
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="weather-service",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())