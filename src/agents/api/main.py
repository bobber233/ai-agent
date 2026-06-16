from typing import Optional
from pydantic import BaseModel
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from src.agents.client import run_agent
from src.mcp_server.mcp_manager import initialize_all_servers
from src.utils.logger import logger

class AppState:
    def __init__(self):
        self.tool_to_session = {}
        self.llm_tools = []
        self.stack = AsyncExitStack()

state = AppState()

@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动时初始化 MCP 服务器和工具映射
    state.tool_to_session, state.llm_tools = await initialize_all_servers(state.stack)
    
    yield
    
    # 关闭时清理资源
    await state.stack.aclose()
    logger.info("  ├─ [服务关闭]: MCP 服务器已关闭，资源已清理。")

app = FastAPI(title="AI Agent API", lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = run_agent(
        request.message,
        tool_to_session=state.tool_to_session,
        llm_tools=state.llm_tools
    )
    return StreamingResponse(reply, media_type="text/event-stream")
