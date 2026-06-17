import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式，绝对不联网 check
import asyncio
from typing import Annotated
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import (
    ErrorData,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from src.utils.logger import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
encoder = SentenceTransformer(MODEL_NAME)

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="agent_knowledge")

class KnowledgeQuerySchema(BaseModel):
    """知识库检索参数校验模型"""
    query: Annotated[
        str, 
        Field(description="需要从本地向量知识库中检索背景信息的查询词、关键字或语义句")
    ]
    top_k: Annotated[
        int, 
        Field(default=3, description="返回的最相关文本片段数量，最大支持10条", gt=0, le=10)
    ]

server = Server("mcp-knowledge")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """向大模型声明当前知识库服务可用的工具表"""
    return [
        Tool(
            name="query_knowledge_base",
            description=(
                "当用户询问关于内部政策、专属技术文档、产品说明或未知的背景知识时"
                "调用此工具从向量知识库中检索相关的匹配片段。"
            ),
            inputSchema=KnowledgeQuerySchema.model_json_schema()
        )
    ]

def _blocking_vector_search(query: str, top_k: int) -> str:
    """内部同步阻塞的语义向量生成与向量库查询核心逻辑"""
    try:
        query_vector = encoder.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        # documents = results.get("documents", [[]])[0]
        raw_docs = results.get("documents")
        documents = raw_docs[0] if raw_docs is not None else []
        if not documents:
            return "在本地知识库中未找到任何相关的参考资料。"
            
        context_list = []
        for i, doc in enumerate(documents, 1):
            context_list.append(f"【参考片段 {i}】:\n{doc}")
            
        return "\n\n".join(context_list)
    except Exception as e:
        logger.error(f"向量计算或数据库检索失败: {e}")
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"数据库检索出现内部异常: {str(e)}",
        ))
        
# 7. 绑定工具调用的具体执行分发器
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    """处理大模型下发的工具调用指令"""
    if name != "query_knowledge_base":
        raise McpError(ErrorData(
            code=INVALID_PARAMS,
            message=f"未知的工具名称: {name}"
        ))

    if not arguments:
        raise McpError(ErrorData(
            code=INVALID_PARAMS,
            message="缺少必需的参数结构"
        ))

    try:
        # 使用 Pydantic 进行参数校验
        args = KnowledgeQuerySchema(**arguments)
    except Exception as e:
        raise McpError(ErrorData(
            code=INVALID_PARAMS,
            message=f"参数校验不通过: {str(e)}"
        ))

    # 使用 asyncio.to_thread 将同步阻塞的推理和向量库请求转交到线程池，防止挂起主执行管道
    try:
        search_result = await asyncio.to_thread(
            _blocking_vector_search, 
            query=args.query, 
            top_k=args.top_k
        )
        return [TextContent(type="text", text=search_result)]
    except McpError as mcp_err:
        raise mcp_err
    except Exception as e:
        raise McpError(ErrorData(
            code=INTERNAL_ERROR,
            message=f"执行检索时发生未预期错误: {str(e)}"
        ))

# 8. 启动 stdio 通信流
async def serve() -> None:
    """初始化并运行标准输入输出 Stdio 管道服务"""
    logger.info("Vector Knowledge Base MCP server starting...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(serve())