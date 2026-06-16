import os
import sqlite3
from typing import Optional
from pydantic import BaseModel
from contextlib import AsyncExitStack, asynccontextmanager
from functools import lru_cache
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
    user_type: str = "standard"
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str

@lru_cache(maxsize=10)
def get_system_prompt(user_type: str) -> str:
    db_path = "src/data/agent_core.db"
    if not os.path.exists(db_path):
        return "You are a helpful AI assistant."
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    prompt_key = "professional_assistant" if user_type == "professional" else "standard_assistant"
    
    cursor.execute("SELECT content FROM system_prompts WHERE prompt_key = ?", (prompt_key,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    return "You are a helpful AI assistant."

@app.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = get_system_prompt(request.user_type)
    reply = run_agent(
        request.message,
        system_prompt=system_prompt,
        tool_to_session=state.tool_to_session,
        llm_tools=state.llm_tools
    )
    return StreamingResponse(reply, media_type="text/event-stream")
