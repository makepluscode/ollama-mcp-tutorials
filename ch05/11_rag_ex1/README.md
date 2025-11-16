# RAG PDF (Enhanced with Detailed Logging)

PDF 문서를 활용한 RAG(Retrieval-Augmented Generation) 예제입니다.

## 설명

이 예제는 PDF 문서를 로드하고 처리하여 벡터 스토어에 저장한 후, 사용자 질문에 대해 PDF 내용을 검색하여 답변을 생성하는 RAG 시스템을 구현합니다. 문서 기반 질의응답 시스템의 기본 구조를 보여주며, 각 단계별로 상세한 로그를 출력하여 프로세스를 이해하기 쉽게 합니다.

## 주요 구성 요소

- **PDF Processing**: PDF 문서 로드 및 텍스트 추출
- **Document Chunking**: 문서를 적절한 크기로 분할
- **Vector Store**: FAISS를 사용한 벡터 저장소 구축 및 관리
- **RAG Implementation**: PDF 내용 기반 질의응답 시스템
- **Detailed Logging**: 각 단계별 실행 시간 및 상세 정보 로깅

## 사전 요구 사항

1. Ollama 설치 및 실행
2. 필요한 모델 다운로드:
   ```bash
   ollama pull qwen3:8b
   ollama pull bge-m3
   ```
3. PDF 파일 준비 (예제에는 `samsung-s22-camera.pdf` 포함)

## 실행 방법

1. 프로젝트 디렉토리로 이동:
   ```bash
   cd ch05/11_rag_ex1
   ```

2. 의존성 설치:
   ```bash
   uv sync
   ```

3. 프로그램 실행:
   ```bash
   uv run python main.py
   ```

4. PDF 내용에 관한 질문을 입력하고 답변을 확인하세요.

5. 종료하려면 `끝` 또는 `exit`를 입력하세요.

## 로그 출력 예시

프로그램 실행 시 다음과 같은 상세 로그를 확인할 수 있습니다:

- **벡터 DB 생성 시** (최초 실행):
  - [단계 1/5] PDF 문서 로딩 (페이지 수, 소요 시간)
  - [단계 2/5] 문서 분할 (청크 수, 평균 크기, 소요 시간)
  - [단계 3/5] 임베딩 모델 초기화
  - [단계 4/5] 벡터 저장소 구축 (소요 시간)
  - [단계 5/5] 벡터 DB 저장 (소요 시간)

- **질의응답 시**:
  - 질문 내용
  - 검색된 문서 수 및 소요 시간
  - 결합된 컨텍스트 크기
  - LLM 응답 생성 시간 및 응답 길이

## 개발 도구

코드 품질 유지를 위한 린트 도구가 포함되어 있습니다:

```bash
# 코드 포맷팅
uv run black main.py
uv run isort main.py

# 또는 한번에
uv run black . && uv run isort .
```

## 기술 스택

- **LangChain**: RAG 파이프라인 구성
- **FAISS**: 벡터 저장 및 검색
- **Ollama**: 로컬 LLM 실행 (qwen3:8b)
- **BGE-M3**: 임베딩 모델
- **PyPDF**: PDF 문서 파싱