#!/usr/bin/env python3
"""
kb_ingest_advanced.py - 高级文档向量灌入工具

相比 src/scripts/ingest_docs.py，支持更丰富的参数控制，如自定义目录、自定义切片参数等。

用法:
    python kb_ingest_advanced.py --source /path/to/my/docs --chunk-size 400 --overlap 50
"""

import os
import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

COLLECTION_NAME = "knowledge_base"
CHROMA_DATA_PATH = str(project_root / "src" / "data" / "chroma_db")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def split_text_into_chunks(text: str, chunk_size: int, overlap: int):
    """文本切片"""
    chunks = []
    start = 0
    if chunk_size <= 0:
        return [text]
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if start >= len(text) or (chunk_size - overlap) <= 0:
            break
    return chunks

def main():
    parser = argparse.ArgumentParser(description="高级文档向量灌入工具")
    parser.add_argument("--source", type=str, default=str(project_root / "src" / "knowledge"), help="要灌入的文档目录或文件路径")
    parser.add_argument("--chunk-size", type=int, default=400, help="文本切分字数")
    parser.add_argument("--overlap", type=int, default=50, help="相邻分段的重叠字数，避免上下文断档")
    args = parser.parse_args()

    # 1. 检查依赖
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        print(f"[错误] 依赖缺失: {e}。请确认在正确的虚拟环境下运行。")
        sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[错误] 路径不存在: {args.source}")
        sys.exit(1)

    print("正在加载向量模型...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"正在连接 ChromaDB ({CHROMA_DATA_PATH})...")
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 递归查找 .txt 与 .md 文件
    target_files = []
    if source_path.is_file():
        if source_path.suffix in [".txt", ".md"]:
            target_files.append(source_path)
    else:
        for p in source_path.glob("**/*"):
            if p.is_file() and p.suffix in [".txt", ".md"]:
                target_files.append(p)

    if not target_files:
        print(f"未在 '{args.source}' 下找到有效的 .txt 或 .md 文件。")
        sys.exit(0)

    print(f"共发现 {len(target_files)} 个文档需要处理...")
    global_id_counter = collection.count()
    success_count = 0

    for file_path in target_files:
        print(f"\n正在处理文件: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = split_text_into_chunks(content, args.chunk_size, args.overlap)
            print(f"-> 成功切分为 {len(chunks)} 个片段")

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                # 生成嵌入向量
                embedding = model.encode(chunk).tolist()
                doc_id = f"doc_{global_id_counter}"
                
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": file_path.name, "chunk_index": i}]
                )
                global_id_counter += 1
            success_count += 1
            print(f"-> 文件 {file_path.name} 写入完毕")
        except Exception as e:
            print(f"[错误] 处理文件 {file_path.name} 失败: {e}")

    print("-" * 60)
    print(f"🎉 灌入完成！成功导入了 {success_count} 个文档。")
    print(f"当前知识库中总共有 {collection.count()} 条知识片段。")

if __name__ == "__main__":
    main()
