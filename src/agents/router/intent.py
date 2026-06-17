from typing import Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from src.agents.core.config import settings
from src.utils.logger import logger
from src.agents.core.config import settings

# 1. 定义路由分类的结构化 Schema
class RoutingDecision(BaseModel):
    """大模型意图路由分类决策"""
    intent: Literal["knowledge_base", "general_chat", "other_tools"] = Field(
        description="判断用户的请求应该路由到哪里：\n"
                    "- 'knowledge_base': 涉及公司内部业务、政策、专属技术文档、产品手册等专有背景知识。\n"
                    "- 'general_chat': 日常打招呼、闲聊、通用常识问答、简单数学计算。\n"
                    "- 'other_tools': 明确要求使用其他外部工具（如搜索、天气等）。"
    )
    reason: str = Field(description="做出该路由决策的简短理由，用于日志审计。")

class IntentRouter:
    def __init__(self):
        self.client = AsyncOpenAI(base_url=settings.MODEL_BASE_URL, api_key=settings.API_KEY)
        self.router_model = settings.INDENT_MODEL_NAME

    async def route_intent(self, user_message: str) -> str:
        """ 分析用户消息意图，返回路由决策：'knowledge_base'、'general_chat' 或 'other_tools'
        """
        system_prompt = (
            "你是一个极其严格的高性能意图路由网关。你的职责是分析用户的输入，"
            "并精准将其分类到最匹配的业务流水线中。请仔细观察分类规则，不要带任何主观情绪。"
        )

        try:
            # 配合 response_format 参数，强制小模型输出标准的 JSON 且符合 Pydantic 结构
            completion = await self.client.beta.chat.completions.parse(
                model=self.router_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请对以下输入进行意图路由划分：\n{user_message}"}
                ],
                response_format=RoutingDecision,
                temperature=0.0  # 设为0以保证分类的确定性
            )
            
            result = completion.choices[0].message.parsed or RoutingDecision(intent="general_chat", reason="模型未返回有效分类，默认降级到 general_chat")
            logger.info(f"【意图路由成功】 决策: {result.intent} | 原因: {result.reason}")
            return result.intent
            
        except Exception as e:
            logger.error(f"意图路由网关异常，默认降级走通用对话流程: {e}")
            return "general_chat"

# 实例化单例
intent_router = IntentRouter()