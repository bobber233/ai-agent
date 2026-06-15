import sqlite3

def init_db(db_path="agent_core.db"):
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

    conn.commit()
    conn.close()
    print(f"数据库 {db_path} 初始化成功！")

if __name__ == "__main__":
    init_db()