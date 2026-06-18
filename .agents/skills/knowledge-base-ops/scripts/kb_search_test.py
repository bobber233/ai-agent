#!/usr/bin/env python3
"""
kb_search_test.py - 知识库检索质量验证工具

用法:
    python kb_search_test.py --query "报销流程是什么" --top-k 3
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

CHROMA_DATA_PATH = str(project_root / "src" / "data" / "chroma_db")
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def main():
    parser = argparse.ArgumentParser(description="知识库检索质量验证工具")
    parser.add_argument("--query", type=str, required=True, help="检索测试的关键字或问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索出的最匹配文档片段数量")
    args = parser.parse_args()

    # 1. 尝试导入依赖
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError as e:
        print(f"[错误] 依赖缺失: {e}。请确认在正确的虚拟环境下运行。")
        sys.exit(1)

    print(f"正在加载向量模型: {MODEL_NAME} ...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        sys.exit(1)

    print(f"正在连接本地向量库: {CHROMA_DATA_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"[错误] Collection '{COLLECTION_NAME}' 未找到。请先运行灌入脚本。")
        sys.exit(1)

    print(f"\n正在计算查询向量并检索...")
    query_vector = model.encode(args.query).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=args.top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("未找到任何相关知识匹配片段。")
        return

    print("=" * 60)
    print(f"检索测试结果: 【{args.query}】")
    print("=" * 60)
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        source = meta.get("source", "未知来源")
        chunk_idx = meta.get("chunk_index", "?")
        print(f"【匹配 {i}】 距离分数(距离越小越相似): {dist:.4f}")
        print(f"  来源: {source} (分片 #{chunk_idx})")
        print("-" * 60)
        print(doc)
        print("=" * 60)

if __name__ == "__main__":
    main()
