# MCP 기반 RAG 검색 서비스 (MCP-based RAG Search Service)

이 프로젝트는 **공식 MCP(Model Context Protocol) Python SDK**를 사용하여 구현된 RAG(Retrieval-Augmented Generation) 검색 서비스 예제입니다.

ChromaDB에 저장된 기술 문서를 검색하는 도구(`search_knowledge_base`)를 Claude Desktop과 같은 MCP 클라이언트에 노출합니다.

## 📋 주요 기능

-   **HTML 문서 처리**: 웹 페이지 콘텐츠를 가져와 H2 태그 기준으로 청킹합니다.
-   **벡터 검색**: Ollama(`bge-m3`) 임베딩과 ChromaDB를 사용하여 유사도 검색을 수행합니다.
-   **Official MCP 지원**: `mcp` 라이브러리를 사용하여 표준 MCP 서버를 구현했습니다.

## 🛠️ 사전 준비 사항 (Prerequisites)

이 프로젝트를 실행하기 위해 다음 도구들이 필요합니다.

1.  **Python 3.10+**: 파이썬 런타임
2.  **uv**: 최신 파이썬 패키지 매니저
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3.  **Ollama**: 로컬 LLM 및 임베딩 실행 도구 (실행 중이어야 함)
    - 임베딩 모델 다운로드 (`bge-m3`):
        ```bash
        ollama pull bge-m3
        ```

## 🚀 설치 및 설정 (Setup)

### 1. 의존성 설치
```bash
uv sync
```

### 2. 기술 문서 데이터베이스 구축 (Setup DB)
서버를 실행하기 전에 검색 대상이 될 문서를 ChromaDB에 인덱싱해야 합니다.
```bash
uv run python setup_db.py
```
> **주의**: 이 프로젝트는 `FastAPI` + `uvicorn` 방식을 사용하지 않습니다. `mcp` 라이브러리를 통해 직접 실행되므로 `uvicorn` 명령어를 사용하지 마세요.

## 🖥️ 사용 방법 (Usage)

### 방법 A: Claude Desktop 연동 (권장)

Claude Desktop 앱에서 이 서버를 사용하려면 설정 파일(`claude_desktop_config.json`)에 다음을 추가하세요.

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "/absolute/path/to/ollama-mcp-tutorials/mcp-rag-service/mcp_server.py"
      ]
    }
  }
}
```
> **주의**: `uv`와 `mcp_server.py`의 경로는 반드시 **절대 경로**로 입력해야 합니다.

### 방법 B: 테스트 클라이언트로 실행 (검증용)

포함된 `mcp_client.py`를 사용하여 터미널에서 서버 동작을 바로 확인할 수 있습니다.

```bash
uv run python mcp_client.py "What is QoS?"
```

**실행 결과 예시:**
```text
🔌 MCP 서버 연결 중...
✅ 세션 초기화 완료
🛠️ 발견된 도구: ['search_knowledge_base']
🔍 검색 실행: 'What is QoS?'

============================================================
검색 결과 ('What is QoS?' 관련):

--- 결과 1 ---
제목: 1.8Durability
출처: https://...
내용: The Durability QoS policy controls whether...
============================================================
```

## 📂 프로젝트 구조

-   `setup_db.py`: HTML 문서를 가져와 ChromaDB를 구축하는 ETL 스크립트
-   `mcp_server.py`: 공식 MCP SDK를 사용한 서버 (tool 노출)
-   `mcp_client.py`: 서버 테스트를 위한 간단한 MCP 클라이언트
-   `pyproject.toml`: 프로젝트 및 의존성 설정

## ❓ 자주 묻는 질문 (FAQ)

### Q: `mcp_server.py`를 따로 실행하지 않아도 되나요?
**A: 네, 그렇습니다. 이것이 MCP의 핵심인 'Stdio Transport' 방식입니다.**

`mcp_client.py`(또는 Claude Desktop)를 실행하면, 내부적으로 `mcp_server.py`를 **하위 프로세스(Subprocess)** 로 자동 실행합니다.
그리고 네트워크 포트가 아닌 **표준 입출력(stdin/stdout)** 을 통해 안전하고 빠르게 통신합니다.

따라서 웹 서버처럼 미리 실행해 둘 필요가 없으며, 클라이언트가 종료되면 서버도 함께 종료됩니다.