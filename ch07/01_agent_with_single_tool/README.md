# 7장 · 도구를 쓰는 기상학자 에이전트 (목업)

책 7장 `01_agent_with_single_tool` 예제의 구조(`Tool` → `create_react_agent` → `AgentExecutor`)를 그대로 두고,
올라마와 Tavily 대신 **외부 호출이 없는 테스트 모델과 고정 날씨 도구**(`vivo_demo.py`)를 붙였어요.

```bash
cd ch07/01_agent_with_single_tool
uv run main.py
```

| 설치된 LangChain | 결과 |
|---|---|
| 0.3.25 (책 출간 당시) | `Seoul: sunny, 22 C` |
| 1.x (최신) | `ImportError: cannot import name 'Tool' from 'langchain.agents'` |
