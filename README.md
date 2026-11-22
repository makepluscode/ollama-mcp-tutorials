# ollama-mcp-tutorials · drift-demo (목업)

> 이 브랜치는 **목업**이에요. 『올라마와 오픈소스 LLM을 활용한 AI 에이전트 개발 입문』(위키북스) 예제 저장소
> [godstale/ollama-mcp-tutorials](https://github.com/godstale/ollama-mcp-tutorials)를 포크해,
> 라이브러리 버전이 바뀌어 책 예제가 멈추는 상황만 재현하도록 **예제 두 개만 남겼어요.**
> 실제 실습은 원본 저장소를 사용하세요.

제1회 라이너 해커톤 `.EXE-THON` 출품작 **Book Drift Inspector**의 데모 입력이에요.

## 재현하는 문제

책이 나온 뒤 LangChain이 1.0으로 바뀌면서, 책대로 만든 에이전트 예제가 첫 줄 import에서 멈춰요.

```text
ModuleNotFoundError: No module named 'langchain.prompts'          # 4장
ImportError: cannot import name 'Tool' from 'langchain.agents'      # 7장
```

## 남긴 예제

| 경로 | 내용 |
|---|---|
| [`ch04/01_prompt_template`](ch04/01_prompt_template) | 프롬프트 → 모델 → 출력 파서 체인. `langchain.prompts` import에서 멈춰요 |
| [`ch07/01_agent_with_single_tool`](ch07/01_agent_with_single_tool) | 날씨 도구를 쓰는 ReAct 에이전트. 올라마·Tavily 대신 고정 테스트 모델과 날씨 도구를 써요 |

나머지 장과 원본 커밋 이력은 원본 저장소와 이 포크의 `main` 브랜치에 그대로 있어요.
