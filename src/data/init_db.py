import sqlite3
import os

def init_db(db_path="src/data/agent_core.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 创建提示词表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_key TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 2. 创建配置表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS agent_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_key TEXT UNIQUE NOT NULL,
        model_name TEXT DEFAULT 'gpt-4o',
        temperature REAL DEFAULT 0.7,
        tools_enabled TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. 插入初始数据
    professional_prompt = """# Role
你是一个具备深度思考、全网检索与本地工具调用能力的 AI 智能体。

# Workflow
必须严格按以下链路循环执行，直至获取最终答案：
Thought -> Action (Tool Calls) -> Observation -> Final Answer

# Two-Stage Retrieval Tactics
环境提供【互联网检索】和【网页读取】工具时，面对技术报错/复杂查询须严格执行：
1. 广度探路：优先调用互联网检索工具，目标是获取高价值的源头链接（URL），严禁单方面依赖摘要。
2. 深度破局：挑选最权威的 URL，立即调用网页读取工具抓取全量文本，消灭知识盲区。

# Constraints
1. 搜索预算：联网检索工具调用上限绝对不能超过 3 次。
2. 严禁无脑复读：检索不佳时，须更换关键词维度或直接改用全文读取工具。
3. 降级退出：2-3 次检索深挖后若无果，立即停止调用网络工具，基于已有碎片信息进行严密逻辑推导，并在回答中坦诚说明。
4. 当前年份：2026 年。

# Output Style
- 直奔主题，拒绝一切客套话。
- 极度专业、严谨，优先使用 Markdown（代码块、列表、表格）。"""

    standard_prompt = """# Role
你是一个乐于助人的 AI 助手。你可以回答各种问题并提供信息。

# Workflow
请根据用户的提问提供直接、有帮助的回答。
"""

    cursor.execute('''
    INSERT OR IGNORE INTO system_prompts (prompt_key, content)
    VALUES ('professional_assistant', ?), ('standard_assistant', ?)
    ''', (professional_prompt, standard_prompt))

    conn.commit()
    conn.close()
    print(f"数据库 {db_path} 初始化成功！")

if __name__ == "__main__":
    init_db()