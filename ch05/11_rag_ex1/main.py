import logging
import os
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 벡터 DB 파일 경로
VECTOR_DB_PATH = "faiss_index"
PDF_FILE_PATH = "samsung-s22-camera.pdf"


# 1. 벡터 DB 파일이 없으면 생성 후 vector_store 리턴
def create_vector_db():
    logger.info("=" * 80)
    logger.info("벡터 DB 생성 프로세스 시작")
    logger.info("=" * 80)

    # 1-1. 문서 로딩 (Document Loading)
    logger.info(f"[단계 1/5] PDF 문서 로딩 시작: {PDF_FILE_PATH}")
    start_time = datetime.now()
    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"✓ 문서 로딩 완료 - 페이지 수: {len(docs)}, 소요 시간: {elapsed:.2f}초"
    )
    for i, doc in enumerate(docs):
        logger.debug(
            f"  페이지 {i+1}: 문자 수 {len(doc.page_content)}, "
            f"메타데이터: {doc.metadata}"
        )

    # 1-2. 문서 분할 (Splitting)
    logger.info("[단계 2/5] 문서 분할 시작")
    logger.info(f"  설정: chunk_size=700, chunk_overlap=70")
    start_time = datetime.now()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    splits = text_splitter.split_documents(docs)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"✓ 문서 분할 완료 - 청크 수: {len(splits)}, 소요 시간: {elapsed:.2f}초"
    )
    logger.info(
        f"  평균 청크 크기: {sum(len(s.page_content) for s in splits) / len(splits):.0f} 문자"
    )

    # 1-3. 임베딩 생성 (Embedding)
    logger.info("[단계 3/5] 임베딩 모델 초기화")
    logger.info("  모델: bge-m3")
    embeddings = OllamaEmbeddings(model="bge-m3")
    logger.info("✓ 임베딩 모델 초기화 완료")

    # 1-4. 벡터 저장소 구축 (Vector Database)
    logger.info("[단계 4/5] 벡터 저장소 구축 시작")
    logger.info(f"  {len(splits)}개 청크에 대한 임베딩 생성 중...")
    start_time = datetime.now()
    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ 벡터 저장소 구축 완료 - 소요 시간: {elapsed:.2f}초")

    # 1-5. 벡터 DB를 로컬에 저장
    logger.info(f"[단계 5/5] 벡터 DB 저장 시작: {VECTOR_DB_PATH}")
    start_time = datetime.now()
    vector_store.save_local(VECTOR_DB_PATH)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ 벡터 DB 저장 완료 - 소요 시간: {elapsed:.2f}초")
    logger.info("=" * 80)

    return vector_store


# 2. 메인 로직
logger.info("RAG 시스템 초기화 시작")
if os.path.exists(VECTOR_DB_PATH):
    logger.info(f"기존 벡터 DB 로드: {VECTOR_DB_PATH}")
    start_time = datetime.now()
    embeddings = OllamaEmbeddings(model="bge-m3")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # 믿을 수 있는 소스임을 확인
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ 벡터 DB 로드 완료 - 소요 시간: {elapsed:.2f}초")
else:
    logger.info("벡터 DB가 존재하지 않음 - 새로운 벡터 DB 생성")
    vector_store = create_vector_db()


# 3. 쿼리 저장소 검색을 위한 retriever 생성
logger.info("Retriever 생성")
retriever = vector_store.as_retriever()
logger.info("✓ Retriever 생성 완료")

# 4. PROMPT Template 생성
logger.info("프롬프트 템플릿 생성")
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요. 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.
#Question:
{question}
#Context:
{context}
#Answer:"""
)
logger.info("✓ 프롬프트 템플릿 생성 완료")

# 5. Ollama 초기화
logger.info("LLM 모델 초기화: qwen3:8b")
llm = ChatOllama(model="qwen3:8b", temperature=0)
logger.info("✓ LLM 모델 초기화 완료")

# 6. 체인을 생성합니다.
logger.info("RAG 체인 생성")
chain = prompt | llm | StrOutputParser()
logger.info("✓ RAG 체인 생성 완료")

logger.info("\n" + "=" * 80)
logger.info("RAG 시스템 준비 완료 - 질문을 입력하세요 (종료: '끝' 또는 'exit')")
logger.info("=" * 80 + "\n")

# 7. chain 실행 및 결과 출력을 반복
while True:
    # 7-1. 사용자의 입력을 기다림
    question = input("\n\n당신: ")
    if question == "끝" or question == "exit":
        logger.info("프로그램 종료")
        break

    # 7-2. 쿼리 처리 (Query-Retriever) : 벡터 DB 에서 참고할 문서 검색
    logger.info(f"[검색] 질문: {question}")
    start_time = datetime.now()
    retrieved_docs = retriever.invoke(question)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"✓ 문서 검색 완료 - 검색된 문서 수: {len(retrieved_docs)}, "
        f"소요 시간: {elapsed:.2f}초"
    )

    for i, doc in enumerate(retrieved_docs):
        logger.debug(
            f"  검색 문서 {i+1}: 페이지 {doc.metadata.get('page', 'N/A')}, "
            f"길이 {len(doc.page_content)} 문자"
        )

    combined_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)
    logger.info(f"  결합된 컨텍스트 크기: {len(combined_docs)} 문자")

    # 7-3. 검색된 문서를 첨부해서 PROMPT 생성
    formatted_prompt = {"context": combined_docs, "question": question}

    # 7-4. 체인을 실행하고 결과를 stream 형태로 출력
    logger.info("[생성] LLM 응답 생성 시작")
    print("\nAI: ", end="", flush=True)
    start_time = datetime.now()
    result = ""
    for chunk in chain.stream(formatted_prompt):
        print(chunk, end="", flush=True)
        result += chunk
    elapsed = (datetime.now() - start_time).total_seconds()
    print()  # 줄바꿈
    logger.info(
        f"✓ 응답 생성 완료 - 응답 길이: {len(result)} 문자, 소요 시간: {elapsed:.2f}초"
    )
