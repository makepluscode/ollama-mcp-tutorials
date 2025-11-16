# RAG PDF - Enhanced Example

PDF 문서 기반 RAG(Retrieval-Augmented Generation) 시스템 예제입니다.

---

## 개요

PDF 문서를 벡터 데이터베이스로 변환하고, 사용자 질문에 대해 관련 문서를 검색하여 LLM 기반 답변을 생성하는 시스템입니다.

**특징**:
- ✅ 간결하고 읽기 쉬운 코드 구조
- ✅ 클래스 기반 아키텍처
- ✅ 상세한 로깅
- ✅ 자동 벡터 DB 생성/로드
- ✅ 사용자 친화적 CLI 인터페이스

---

## 빠른 시작

### 1. 사전 요구 사항

**Ollama 모델 다운로드**:
```bash
ollama pull qwen3:8b     # LLM 모델
ollama pull bge-m3       # 임베딩 모델
```

### 2. 설치 및 실행

```bash
# 의존성 설치
uv sync

# 프로그램 실행
uv run python main.py
```

### 3. 사용 방법

```
질문> 동영상 촬영 방법은?

[답변]
--------------------------------------------------------------------------------
동영상 촬영 방법은 다음과 같습니다:
1. 촬영 모드에서 동영상을 선택한 후...
...
--------------------------------------------------------------------------------
응답 시간: 16.41초 | 길이: 562자
--------------------------------------------------------------------------------

질문> exit  (종료)
```

**명령어**:
- 일반 질문: PDF 내용에 대한 질문 입력
- `/help`: 도움말
- `/stats`: 벡터 DB 통계
- `exit`, `quit`, `끝`, `q`: 종료

---

## 주요 기능

### 벡터 DB 자동 관리
- **최초 실행**: PDF에서 벡터 DB 자동 생성 (~60초)
- **이후 실행**: 저장된 벡터 DB 로드 (~0.03초)

### 빠른 검색
- 검색 속도: 0.16~0.23초
- 검색 문서: 4개 (유사도 높은 순)
- 컨텍스트: 2,000~2,600자

### 정확한 답변
- LLM: qwen3:8b
- 응답 시간: 5~17초 (평균 10초)
- 출력 방식: 스트리밍

---

## 기술 스택

| 컴포넌트 | 기술 | 용도 |
|---------|------|------|
| 문서 로딩 | PyPDFLoader | PDF 파싱 |
| 텍스트 분할 | RecursiveCharacterTextSplitter | 청크 생성 (700자) |
| 임베딩 | bge-m3 (Ollama) | 벡터 변환 |
| 벡터 DB | FAISS | 유사도 검색 |
| LLM | qwen3:8b (Ollama) | 답변 생성 |
| 체인 | LangChain LCEL | 파이프라인 구성 |

---

## 프로젝트 구조

```
ch05/11_rag_ex1/
├── main.py                      # 메인 프로그램 (605줄)
├── samsung-s22-camera.pdf       # 샘플 PDF
├── faiss_index/                 # 벡터 DB (자동 생성)
│   ├── index.faiss
│   └── index.pkl
├── README.md                    # 이 파일
├── ARCHITECTURE.md              # 아키텍처 설명 (다이어그램)
├── REPORT.md                    # 테스트 및 분석 보고서
├── pyproject.toml               # 의존성 설정
└── uv.lock                      # 잠금 파일
```

---

## 문서

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: 시스템 아키텍처 및 Mermaid 다이어그램
- **[REPORT.md](REPORT.md)**: 테스트 결과 및 성능 분석

---

## 개발 도구

### 코드 포맷팅
```bash
uv run black main.py
uv run isort main.py
```

### 통계 확인
프로그램 실행 후:
```
질문> /stats

벡터 DB 통계 정보
================================================================================
  총 벡터 수      : 23개
  벡터 차원       : 1,024D
  인덱스 타입     : IndexFlatL2
  저장 크기       : 0.12 MB
  ...
```

---

## 설정

`main.py` 상단에서 수정 가능:

```python
# 벡터 DB 경로
VECTOR_DB_PATH = "faiss_index"
PDF_FILE_PATH = "samsung-s22-camera.pdf"

# 청크 설정
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70

# 모델 설정
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen3:8b"
LLM_TEMPERATURE = 0
```

---

## 성능 특성

### 초기화
- 벡터 DB 생성 (최초): ~60초
- 벡터 DB 로드 (이후): ~0.03초

### 검색
- 문서 검색: 0.2초 이하
- 검색 정확도: 50% (테스트 기준)

### 답변 생성
- 응답 시간: 5~17초
- 평균: ~10초

---

## 알려진 제한사항

1. **특정 키워드 검색**: "4K", "8K" 같은 구체적 스펙 검색 실패 가능
2. **복합 질문**: 여러 주제를 포함한 질문 처리 미흡
3. **응답 시간**: LLM 생성 시간이 다소 긴 편 (평균 10초)

개선 방안은 [REPORT.md](REPORT.md)를 참조하세요.

---

## 라이선스

이 프로젝트는 학습 목적의 예제 코드입니다.

---

## 참고 자료

- [LangChain 문서](https://python.langchain.com/)
- [FAISS 문서](https://github.com/facebookresearch/faiss)
- [Ollama 문서](https://ollama.com/)
