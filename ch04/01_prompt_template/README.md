# 4장 · 프롬프트 템플릿으로 기상학자 만들기 (목업)

책 4장 `01_prompt_template` 예제의 구조(`ChatPromptTemplate` → 모델 → `StrOutputParser` 체인)를 그대로 두고,
올라마 대신 **받은 프롬프트를 되돌려 주는 테스트 모델**(`vivo_demo.py`)을 붙였어요.

```bash
cd ch04/01_prompt_template
uv run main.py
```

| 설치된 LangChain | 결과 |
|---|---|
| 0.3.25 (책 출간 당시) | `기상학자 답변 ← 당신은 유능한 기상학자입니다. 다음 질문에 답해주세요. <질문>: 서울 날씨는?` |
| 1.x (최신) | `ModuleNotFoundError: No module named 'langchain.prompts'` |
