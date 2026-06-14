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
    if request.stream:
        # returns an AsyncGenerator
        generator = await run_agent(request.message, stream=True)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        reply = await run_agent(request.message, stream=False)
        return ChatResponse(reply=reply if isinstance(reply, str) else '')
