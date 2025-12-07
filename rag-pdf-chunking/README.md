# PDF 문서 전처리: 다양한 청킹(Chunking) 전략

PDF 문서를 RAG(Retrieval-Augmented Generation)에 사용하기 위해 전처리하는 고급 예제입니다. LangChain에서 제공하는 다양한 텍스트 분할(Text Splitting) 전략을 적용하고, 각 결과를 상세한 리포트 파일로 생성합니다.

---

## 개요

이 프로젝트는 단일 PDF 파일을 로드하여, 세 가지 서로 다른 청킹 전략을 동시에 실행합니다.
- `RecursiveCharacterTextSplitter`
- `CharacterTextSplitter`
- `TokenTextSplitter`

각 전략의 실행 결과는 별도의 `chunks-[전략명].txt` 파일로 저장됩니다. 이 파일에는 상세한 메타데이터, 실행 시간, 통계 정보가 포함되어 있어 각 전략의 특성을 비교하고 분석하는 데 유용합니다.

### 주요 특징
- ✅ **세 가지 청킹 전략 비교**: Recursive, Character, Token 기반 분할을 모두 실행합니다.
- ✅ **상세 리포트 생성**: 각 전략의 결과는 별도 파일에 풍부한 메타데이터와 함께 저장됩니다.
- ✅ **분리된 아키텍처**: `PdfDocumentLoader`와 `ChunkerFactory`를 통해 로직이 명확하게 분리되어 있습니다.
- ✅ **효율적인 실행**: PDF 문서는 한 번만 로드하여 각 전략에 재사용합니다.

---

## 빠른 시작

### 1. 의존성 설치
`uv` 패키지 매니저를 사용하여 필요한 라이브러리를 설치합니다.
```bash
uv sync
```

### 2. 프로그램 실행
아래 명령어를 실행하면 프로젝트 루트 디렉토리에 3개의 결과 파일이 생성됩니다.
```bash
uv run python main.py
```
- **실행 결과**:
  - `chunks-recursive.txt`
  - `chunks-character.txt`
  - `chunks-token.txt`

---

## 청킹 전략 상세

`main.py`의 `ChunkerFactory` 클래스는 각 전략에 맞는 `TextSplitter` 객체를 생성합니다.

### 1. RecursiveCharacterTextSplitter (`recursive`)
의미적으로 관련된 텍스트가 함께 유지되도록 노력하는 가장 추천되는 방식입니다. 구분자(separator) 리스트(`["\n\n", "\n", " ", ""]`)를 사용하여 계층적으로 텍스트를 분할합니다.

**API 사용:**
```python
# From ChunkerFactory in main.py
return RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

### 2. CharacterTextSplitter (`character`)
단순히 하나의 특정 문자(기본값: `\n\n`)를 기준으로 텍스트를 분할합니다. 가장 간단한 분할 방식입니다.

**API 사용:**
```python
# From ChunkerFactory in main.py
return CharacterTextSplitter(
    separator="\n\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    is_separator_regex=False,
)
```

### 3. TokenTextSplitter (`token`)
LLM(언어 모델)이 텍스트를 처리하는 방식과 유사하게, 토큰(token)을 기준으로 텍스트를 분할합니다. `tiktoken` 라이브러리를 사용하여 토큰 수를 계산하며, 모델의 컨텍스트 윈도우에 더 정확하게 맞출 수 있습니다.

**API 사용:**
```python
# From ChunkerFactory in main.py
encoding = tiktoken.get_encoding("cl100k_base")
return TokenTextSplitter(
    encoding_name=encoding.name,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
```

---

## 기술 스택

| 컴포넌트 | 기술 | 용도 |
|---|---|---|
| 문서 로딩 | `PyPDFLoader` | PDF 파싱 |
| 텍스트 분할 | `RecursiveCharacterTextSplitter` | 계층적 텍스트 분할 |
| | `CharacterTextSplitter` | 단일 문자 기준 분할 |
| | `TokenTextSplitter` | 토큰 기준 분할 |
| 토큰 계산 | `tiktoken` | `TokenTextSplitter`를 위한 토큰 수 계산 |
| 의존성 관리 | `uv` | 패키지 설치 및 관리 |

---

## 프로젝트 구조

```
rag-pdf-chunking/
├── main.py                      # 메인 프로그램 (모든 전략 실행)
├── DDSI-RTPS-v2.5.pdf           # 샘플 PDF
├── chunks-recursive.txt         # Recursive 전략 결과 (자동 생성)
├── chunks-character.txt         # Character 전략 결과 (자동 생성)
├── chunks-token.txt             # Token 전략 결과 (자동 생성)
├── README.md                    # 이 파일
├── pyproject.toml               # 의존성 설정
└── uv.lock                      # 잠금 파일
```

---

## 설정

`main.py` 상단에서 주요 설정을 수정할 수 있습니다.

```python
# 처리할 PDF 파일 경로
PDF_FILE_PATH = "DDSI-RTPS-v2.5.pdf"

# 문서 분할 설정
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70
```

---

## 참고 자료

- [LangChain | Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [tiktoken documentation](https://github.com/openai/tiktoken)