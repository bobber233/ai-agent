#!/usr/bin/env python3
"""
kb_status.py - 知识库状态查询工具

用法:
    python kb_status.py              # 打印状态摘要
    python kb_status.py --export f.json   # 同时导出完整快照到 JSON
    python kb_status.py --verbose    # 打印前 20 条文档片段详情

必须在项目根目录下执行:
    cd /home/yanqing/projects/ai-agent
    .venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_status.py
"""

import sys
import json
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

CHROMA_DATA_PATH = str(project_root / "src" / "data" / "chroma_db")
COLLECTION_NAME = "knowledge_base"


def get_collection():
    try:
        import chromadb
    except ImportError:
        print("[错误] 找不到 chromadb，请确认在正确的虚拟环境下运行。")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"[警告] Collection '{COLLECTION_NAME}' 不存在，知识库为空或尚未初始化。")
        sys.exit(0)
    return col


def print_status(col, verbose: bool = False):
    count = col.count()
    print("=" * 55)
    print(f"  知识库状态报告")
    print("=" * 55)
    print(f"  ChromaDB 路径  : {CHROMA_DATA_PATH}")
    print(f"  Collection 名  : {COLLECTION_NAME}")
    print(f"  文档片段总数   : {count}")
    print("=" * 55)

    if count == 0:
        print("  知识库为空，请先运行 ingest_docs.py 灌入文档。")
        return

    # 拉取最新 5 条（倒序）
    sample = col.get(limit=min(5, count), include=["documents", "metadatas"])
    print(f"\n  最近入库的 {len(sample['ids'])} 条片段：")
    print("-" * 55)
    for i, (doc_id, doc, meta) in enumerate(
        zip(sample["ids"], sample["documents"], sample["metadatas"]), 1
    ):
        source = meta.get("source", "未知来源")
        chunk_idx = meta.get("chunk_index", "?")
        preview = doc[:80].replace("\n", " ") + ("..." if len(doc) > 80 else "")
        print(f"  [{i}] ID: {doc_id}")
        print(f"       来源: {source}  (chunk #{chunk_idx})")
        print(f"       内容: {preview}")
        print()

    if verbose and count > 5:
        extra = col.get(limit=min(20, count), include=["documents", "metadatas"])
        print(f"\n  [--verbose] 完整前 {len(extra['ids'])} 条记录：")
        print("-" * 55)
        for doc_id, doc, meta in zip(
            extra["ids"], extra["documents"], extra["metadatas"]
        ):
            print(f"  {doc_id} | {meta.get('source','?')} | {doc[:60]}...")


def export_snapshot(col, output_path: str):
    count = col.count()
    if count == 0:
        print("[警告] 知识库为空，跳过导出。")
        return

    # ChromaDB 的 get() 没有 offset，分批拉取
    batch_size = 200
    all_records = []
    offset_ids = None

    # 简单全量拉取（小型知识库可直接全取）
    data = col.get(limit=count, include=["documents", "metadatas"])
    for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        all_records.append({"id": doc_id, "document": doc, "metadata": meta})

    snapshot = {
        "collection": COLLECTION_NAME,
        "chroma_path": CHROMA_DATA_PATH,
        "total_count": count,
        "records": all_records,
    }

    out = Path(output_path)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[导出成功] {count} 条记录已保存到 {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="知识库状态查询工具")
    parser.add_argument("--export", metavar="FILE", help="将快照导出为 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="显示前 20 条片段详情")
    args = parser.parse_args()

    col = get_collection()
    print_status(col, verbose=args.verbose)

    if args.export:
        export_snapshot(col, args.export)


if __name__ == "__main__":
    main()
