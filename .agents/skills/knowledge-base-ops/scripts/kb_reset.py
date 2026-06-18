#!/usr/bin/env python3
"""
kb_reset.py - 知识库清空与重建工具

用法:
    python kb_reset.py --confirm
"""

import sys
import argparse
import shutil
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

CHROMA_DATA_PATH = project_root / "src" / "data" / "chroma_db"
COLLECTION_NAME = "knowledge_base"

def main():
    parser = argparse.ArgumentParser(description="知识库清空与重建工具")
    parser.add_argument("--confirm", action="store_true", help="确认清空知识库，跳过交互保护。警告：操作不可逆！")
    args = parser.parse_args()

    if not args.confirm:
        print("=" * 60)
        print("⚠️  警告: 本操作将彻底删除本地向量数据库内所有已入库的文档和向量数据！")
        print(f"数据目录: {CHROMA_DATA_PATH}")
        print("=" * 60)
        user_input = input("确定要清空吗？输入 [yes] 确认: ")
        if user_input.strip().lower() != 'yes':
            print("操作已取消。")
            sys.exit(0)

    try:
        import chromadb
    except ImportError:
        # 如果无法导入，但目录存在，提供彻底删除目录的降级方案
        if CHROMA_DATA_PATH.exists():
            print("[信息] 找不到 chromadb，尝试物理删除数据目录...")
            try:
                shutil.rmtree(CHROMA_DATA_PATH)
                print(f"🎉 物理删除数据目录成功: {CHROMA_DATA_PATH}")
            except Exception as e:
                print(f"[错误] 删除失败: {e}")
            sys.exit(0)
        else:
            print("[警告] 目录不存在，无需清理。")
            sys.exit(0)

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DATA_PATH))
        # 尝试删除集合
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"🎉 成功删除 Chroma 集合 '{COLLECTION_NAME}'。")
        except ValueError:
            print(f"[警告] 集合 '{COLLECTION_NAME}' 本就不存在。")
        
        # 再次尝试物理清空文件夹，避免产生无效缓存文件或损坏导致 SQLite 错误
        client = None # 显式释放连接
        
        # 等待一小会儿确保 sqlite3 已关闭
        import time
        time.sleep(0.5)
        
        if CHROMA_DATA_PATH.exists():
            shutil.rmtree(CHROMA_DATA_PATH)
            print("🎉 物理数据目录已彻底清理。下次灌入时会重新构建。")

    except Exception as e:
        print(f"[错误] 清理向量库异常: {e}")
        # 降级删除
        if CHROMA_DATA_PATH.exists():
            try:
                shutil.rmtree(CHROMA_DATA_PATH)
                print("🎉 已通过强制删除数据目录进行降级恢复。")
            except Exception as ex:
                print(f"[错误] 强制物理删除失败: {ex}")

if __name__ == "__main__":
    main()
