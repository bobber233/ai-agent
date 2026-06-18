---
name: knowledge-base-ops
description: >
  AI Agent 项目的向量知识库运维 Skill。当需要管理 ChromaDB 向量知识库时触发，
  包括：知识库状态查询、文档灌入、知识库清空/重建、搜索质量验证、
  知识库迁移等操作。关键词：知识库、ChromaDB、向量数据库、ingest、embedding、
  文档入库、知识管理。
---

# Knowledge Base Ops Skill

本 Skill 用于管理 `src/data/chroma_db` 中的 ChromaDB 向量知识库。

## 项目配置速查

| 配置项 | 值 |
|---|---|
| ChromaDB 路径 | `src/data/chroma_db` |
| Collection 名 | `knowledge_base` |
| 嵌入模型 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| 文档源目录 | `src/knowledge/` |
| 现有灌入脚本 | `src/scripts/ingest_docs.py` |

> 所有脚本必须在项目根目录 `/home/yanqing/projects/ai-agent` 下执行，
> 使用 `.venv/bin/python` 以确保依赖正确。

---

## 任务 1：查询知识库状态

使用 `scripts/kb_status.py` 脚本。

```bash
cd /home/yanqing/projects/ai-agent
.venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_status.py
```

**输出示例**: 当前 collection 名、文档片段总数、最近入库的 5 条 metadata。

---

## 任务 2：灌入新文档

1. 将 `.txt` 或 `.md` 文件放到 `src/knowledge/` 目录下
2. 运行现有灌入脚本：

```bash
cd /home/yanqing/projects/ai-agent
.venv/bin/python src/scripts/ingest_docs.py
```

如需指定自定义目录，使用扩展版脚本：

```bash
.venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_ingest_advanced.py \
  --source /path/to/docs \
  --chunk-size 400 \
  --overlap 50
```

---

## 任务 3：验证检索质量

```bash
.venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_search_test.py \
  --query "你的测试问题" \
  --top-k 5
```

---

## 任务 4：清空知识库（危险操作，不可逆）

```bash
.venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_reset.py --confirm
```

---

## 任务 5：导出知识库快照

```bash
.venv/bin/python .agents/skills/knowledge-base-ops/scripts/kb_status.py --export snapshot.json
```

---

## 常见问题排查

### 问题: SentenceTransformer 模型加载很慢
原因: 首次使用时下载模型。设置了 HF_HUB_OFFLINE=1 后必须确保模型缓存已存在。
检查: `ls ~/.cache/huggingface/hub/`

### 问题: ChromaDB 报 sqlite3.DatabaseError
原因: ChromaDB 数据文件损坏。
解法: 备份后执行 `kb_reset.py`，重新灌入。

### 问题: 检索结果质量差
原因: chunk_size 设置不合理，或文档语言与模型不匹配。
解法: 用 `kb_search_test.py` 测试并调整 chunk_size（推荐 300-600）。
