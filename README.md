# AI Agent

## 安装依赖
```shell
poetry config virtualenvs.in-project true

poetry install
```

## 运行
poetry run uvicorn src.agents.api.main:app --reload

## 测试
```shell
 curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "2026年6月15日热点新闻有哪些"}'
```

