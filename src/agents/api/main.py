from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from src.agents.agent_client import run_agent

app = FastAPI(title="AI Agent API")

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = run_agent(request.message)
    return StreamingResponse(reply, media_type="text/event-stream")
