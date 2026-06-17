import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
# from utils.logger import logger

print("正在初始化本地向量模型...")

COLLECTION_NAME = "knowledge_base"
CHROMA_DATA_PATH = "src/data/chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

embedding_model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    """简单文本切片函数：将大文本按照固定字数切块，带有一点重叠防止上下文断开"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def ingest_directory(docs_dir: str):
    """扫描指定文件夹，将里面的 .txt / .md 文件存入向量数据库"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"错误: 文件夹 {docs_dir} 不存在！")
        return
    
    global_id_counter = collection.count()
    
    # 遍历文件夹下的 md 和 txt 文件
    for file_path in docs_path.glob("**/*"):
        if file_path.suffix in [".txt", ".md"]:
            print(f"\n正在处理文件: {file_path.name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 文本切片
                chunks = split_text_into_chunks(content, chunk_size=400, overlap=50)
                print(f"-> 成功切分为 {len(chunks)} 个片段")
                
                # 开始批量生成向量并写入
                for i, chunk in enumerate(chunks):
                    # 生成当前切片的嵌入向量
                    embedding = embedding_model.encode(chunk).tolist()
                    
                    doc_id = f"doc_{global_id_counter}"
                    # 写入 ChromaDB
                    collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"source": file_path.name, "chunk_index": i}]
                    )
                    global_id_counter += 1
                    
            except Exception as e:
                print(f"处理文件 {file_path.name} 失败: {e}")

    print(f"\n🎉 灌入完成！当前知识库中总共有 {collection.count()} 条知识片段。")

if __name__ == "__main__":
    # 在这里指定你的原始文档存放目录
    DATA_SOURCE_DIR = "src/knowledge"
    
    # 自动创建测试目录
    if not os.path.exists(DATA_SOURCE_DIR):
        os.makedirs(DATA_SOURCE_DIR)
        print(f"已为你创建存放文档的空文件夹 '{DATA_SOURCE_DIR}'。")
        print("请往该文件夹下丢入几个 .txt 或 .md 文件，然后重新运行此脚本！")
    else:
        ingest_directory(DATA_SOURCE_DIR)