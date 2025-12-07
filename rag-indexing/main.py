"""
RAG (Retrieval-Augmented Generation) 시스템 구현

PDF 문서를 벡터 데이터베이스로 변환하고, 사용자 질문에 대해
관련 문서를 검색하여 답변을 생성하는 시스템입니다.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# 설정 (Configuration)
# ============================================================================

# 벡터 데이터베이스 설정
VECTOR_DB_PATH = "faiss_index"
PDF_FILE_PATH = "samsung-s22-camera.pdf"

# 문서 분할 설정
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70

# 모델 설정
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen3:8b"
LLM_TEMPERATURE = 0

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# RAG 시스템 클래스 (RAG System Class)
# ============================================================================


class RAGSystem:
    """PDF 문서 기반 RAG 시스템을 관리하는 클래스"""

    def __init__(
        self,
        pdf_path: str = PDF_FILE_PATH,
        vector_db_path: str = VECTOR_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
    ):
        """
        RAG 시스템 초기화

        Args:
            pdf_path: PDF 파일 경로
            vector_db_path: 벡터 DB 저장 경로
            embedding_model: 임베딩 모델 이름
            llm_model: LLM 모델 이름
        """
        self.pdf_path = pdf_path
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # 컴포넌트 초기화 (나중에 설정됨)
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[BaseRetriever] = None
        self.chain: Optional[Runnable] = None

    def setup(self) -> None:
        """RAG 시스템의 모든 컴포넌트를 초기화"""
        logger.info("RAG 시스템 초기화 시작")

        # 1. 벡터 스토어 로드 또는 생성
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = self._load_or_create_vector_store()

        # 2. Retriever 생성
        self.retriever = self._create_retriever()

        # 3. RAG 체인 생성
        self.chain = self._create_rag_chain()

        logger.info("=" * 80)
        logger.info("RAG 시스템 준비 완료")
        logger.info("=" * 80)

    def _load_or_create_vector_store(self) -> FAISS:
        """
        벡터 스토어를 로드하거나 새로 생성

        Returns:
            FAISS 벡터 스토어 인스턴스
        """
        if os.path.exists(self.vector_db_path):
            return self._load_vector_store()
        else:
            return self._create_vector_store()

    def _load_vector_store(self) -> FAISS:
        """
        기존 벡터 스토어 로드

        Returns:
            로드된 FAISS 벡터 스토어
        """
        logger.info(f"기존 벡터 DB 로드: {self.vector_db_path}")
        start_time = datetime.now()

        vector_store = FAISS.load_local(
            self.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 벡터 DB 로드 완료 - 소요 시간: {elapsed:.2f}초")

        return vector_store

    def _create_vector_store(self) -> FAISS:
        """
        PDF 문서로부터 새로운 벡터 스토어 생성

        Returns:
            생성된 FAISS 벡터 스토어
        """
        logger.info("=" * 80)
        logger.info("벡터 DB 생성 프로세스 시작")
        logger.info("=" * 80)

        # 1. 문서 로딩
        docs = self._load_pdf_documents()

        # 2. 문서 분할
        splits = self._split_documents(docs)

        # 3. 벡터 스토어 구축
        vector_store = self._build_vector_store(splits)

        # 4. 벡터 스토어 저장
        self._save_vector_store(vector_store)

        logger.info("=" * 80)
        return vector_store

    def _load_pdf_documents(self) -> List[Document]:
        """
        PDF 문서 로딩

        Returns:
            로드된 문서 리스트
        """
        logger.info(f"[단계 1/4] PDF 문서 로딩: {self.pdf_path}")
        start_time = datetime.now()

        loader = PyPDFLoader(self.pdf_path)
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

        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """
        문서를 작은 청크로 분할

        Args:
            docs: 원본 문서 리스트

        Returns:
            분할된 문서 청크 리스트
        """
        logger.info("[단계 2/4] 문서 분할")
        logger.info(f"  설정: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
        start_time = datetime.now()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)

        elapsed = (datetime.now() - start_time).total_seconds()
        avg_chunk_size = sum(len(s.page_content) for s in splits) / len(splits)

        logger.info(
            f"✓ 문서 분할 완료 - 청크 수: {len(splits)}, 소요 시간: {elapsed:.2f}초"
        )
        logger.info(f"  평균 청크 크기: {avg_chunk_size:.0f} 문자")

        return splits

    def _build_vector_store(self, splits: List[Document]) -> FAISS:
        """
        문서 청크로부터 벡터 스토어 구축

        Args:
            splits: 분할된 문서 청크 리스트

        Returns:
            구축된 FAISS 벡터 스토어
        """
        logger.info("[단계 3/4] 벡터 저장소 구축")
        logger.info(f"  모델: {self.embedding_model}")
        logger.info(f"  {len(splits)}개 청크에 대한 임베딩 생성 중...")
        start_time = datetime.now()

        vector_store = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings,
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 벡터 저장소 구축 완료 - 소요 시간: {elapsed:.2f}초")

        return vector_store

    def _save_vector_store(self, vector_store: FAISS) -> None:
        """
        벡터 스토어를 로컬에 저장

        Args:
            vector_store: 저장할 FAISS 벡터 스토어
        """
        logger.info(f"[단계 4/4] 벡터 DB 저장: {self.vector_db_path}")
        start_time = datetime.now()

        vector_store.save_local(self.vector_db_path)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 벡터 DB 저장 완료 - 소요 시간: {elapsed:.2f}초")

    def _get_vector_store_stats(self) -> dict:
        """
        벡터 스토어 통계 정보 수집

        Returns:
            통계 정보 딕셔너리
        """
        if self.vector_store is None:
            return {}

        try:
            # FAISS 인덱스 정보
            index = self.vector_store.index
            stats = {
                "total_vectors": index.ntotal,
                "vector_dimension": index.d,
                "index_type": str(type(index).__name__),
            }

            # 저장된 파일 크기 확인
            if os.path.exists(self.vector_db_path):
                total_size = 0
                for root, dirs, files in os.walk(self.vector_db_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)

            return stats
        except Exception as e:
            logger.warning(f"통계 정보 수집 중 오류: {e}")
            return {}

    def _create_retriever(self) -> BaseRetriever:
        """
        문서 검색을 위한 Retriever 생성

        Returns:
            생성된 Retriever 인스턴스
        """
        logger.info("Retriever 생성")
        retriever = self.vector_store.as_retriever()
        logger.info("✓ Retriever 생성 완료")
        return retriever

    def _create_rag_chain(self) -> Runnable:
        """
        RAG 체인 생성 (Prompt + LLM + OutputParser)

        Returns:
            생성된 RAG 체인
        """
        logger.info("RAG 체인 생성")

        # 1. 프롬프트 템플릿 생성
        prompt = self._create_prompt_template()

        # 2. LLM 초기화
        llm = self._initialize_llm()

        # 3. 체인 구성
        chain = prompt | llm | StrOutputParser()

        logger.info("✓ RAG 체인 생성 완료")
        return chain

    def _create_prompt_template(self) -> PromptTemplate:
        """
        질문-답변을 위한 프롬프트 템플릿 생성

        Returns:
            생성된 프롬프트 템플릿
        """
        logger.info("프롬프트 템플릿 생성")

        template = """당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요. 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""

        prompt = PromptTemplate.from_template(template)
        logger.info("✓ 프롬프트 템플릿 생성 완료")
        return prompt

    def _initialize_llm(self) -> ChatOllama:
        """
        LLM 모델 초기화

        Returns:
            초기화된 ChatOllama 인스턴스
        """
        logger.info(f"LLM 모델 초기화: {self.llm_model}")
        llm = ChatOllama(model=self.llm_model, temperature=LLM_TEMPERATURE)
        logger.info("✓ LLM 모델 초기화 완료")
        return llm

    def query(self, question: str) -> str:
        """
        사용자 질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            생성된 답변 문자열
        """
        # 1. 관련 문서 검색
        retrieved_docs = self._retrieve_documents(question)

        # 2. 검색된 문서를 컨텍스트로 결합
        context = self._combine_documents(retrieved_docs)

        # 3. LLM을 통해 답변 생성
        answer = self._generate_answer(question, context)

        return answer

    def _retrieve_documents(self, question: str) -> List[Document]:
        """
        질문과 관련된 문서 검색

        Args:
            question: 사용자 질문

        Returns:
            검색된 문서 리스트
        """
        logger.info(f"[검색] 질문: {question}")
        start_time = datetime.now()

        retrieved_docs = self.retriever.invoke(question)

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

        return retrieved_docs

    def _combine_documents(self, docs: List[Document]) -> str:
        """
        검색된 문서들을 하나의 컨텍스트로 결합

        Args:
            docs: 검색된 문서 리스트

        Returns:
            결합된 컨텍스트 문자열
        """
        combined = "\n\n".join(doc.page_content for doc in docs)
        logger.info(f"  결합된 컨텍스트 크기: {len(combined)} 문자")
        return combined

    def _generate_answer(self, question: str, context: str) -> str:
        """
        LLM을 사용하여 답변 생성 (스트리밍)

        Args:
            question: 사용자 질문
            context: 검색된 컨텍스트

        Returns:
            생성된 답변
        """
        logger.info("[생성] LLM 응답 생성 시작")
        print("\n" + "-" * 80)
        print("[답변]")
        print("-" * 80)

        start_time = datetime.now()
        result = ""

        # 스트리밍으로 답변 생성
        for chunk in self.chain.stream({"context": context, "question": question}):
            print(chunk, end="", flush=True)
            result += chunk

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "-" * 80)
        print(f"응답 시간: {elapsed:.2f}초 | 길이: {len(result)}자")
        print("-" * 80 + "\n")

        logger.info(
            f"✓ 응답 생성 완료 - 응답 길이: {len(result)} 문자, 소요 시간: {elapsed:.2f}초"
        )

        return result


# ============================================================================
# 메인 실행 함수 (Main Execution)
# ============================================================================


def run_interactive_session(rag_system: RAGSystem) -> None:
    """
    대화형 세션 실행

    Args:
        rag_system: 초기화된 RAG 시스템 인스턴스
    """
    # 환영 메시지 및 사용 안내
    print("\n" + "=" * 80)
    print("RAG 시스템 - PDF 문서 기반 질의응답")
    print("=" * 80)
    print(f"\n[문서] {rag_system.pdf_path}")

    # 벡터 DB 통계
    stats = rag_system._get_vector_store_stats()
    print(
        f"[정보] 벡터 DB: {stats.get('total_vectors', 'N/A')}개 청크, "
        f"{stats.get('storage_size_mb', 'N/A')}MB"
    )

    print("\n" + "-" * 80)
    print("사용 방법:")
    print("-" * 80)
    print("\n[1] 질문하기")
    print("    PDF 내용에 대한 질문을 입력하세요")
    print("    예시: 카메라 해상도는 얼마인가요?")
    print()
    print("[2] 명령어")
    print("    /help    - 도움말")
    print("    /stats   - 통계 정보")
    print()
    print("[3] 종료")
    print("    exit, quit, 끝, q 중 하나 입력")
    print("    또는 Ctrl+C")
    print("-" * 80)
    print()

    while True:
        try:
            # 사용자 입력 받기
            user_input = input("질문> ").strip()

            # 빈 입력 건너뛰기
            if not user_input:
                print("질문을 입력하세요. (/help: 도움말)\n")
                continue

            # 종료 명령 확인
            if user_input.lower() in ["끝", "exit", "quit", "q"]:
                print("\n프로그램을 종료합니다.\n")
                logger.info("프로그램 종료")
                break

            # 명령어 처리
            if user_input.startswith("/"):
                handle_command(rag_system, user_input)
            else:
                # 질문 처리 및 답변 생성
                rag_system.query(user_input)

        except KeyboardInterrupt:
            # Ctrl+C 처리
            print("\n\n프로그램을 종료합니다. (Ctrl+C)\n")
            logger.info("프로그램 종료 (KeyboardInterrupt)")
            break
        except EOFError:
            # EOF 처리
            print("\n\n프로그램을 종료합니다.\n")
            logger.info("프로그램 종료 (EOF)")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"\n[오류] {e}")
            print("다시 시도해주세요.\n")


def handle_command(rag_system: RAGSystem, command: str) -> None:
    """
    사용자 명령어 처리

    Args:
        rag_system: RAG 시스템 인스턴스
        command: 사용자가 입력한 명령어
    """
    parts = command.lower().split()
    cmd = parts[0]

    if cmd == "/stats":
        # 통계 정보 표시
        stats = rag_system._get_vector_store_stats()
        print("\n" + "=" * 80)
        print("벡터 DB 통계 정보")
        print("=" * 80)
        print(f"  총 벡터 수      : {stats.get('total_vectors', 'N/A'):,}개")
        print(f"  벡터 차원       : {stats.get('vector_dimension', 'N/A'):,}D")
        print(f"  인덱스 타입     : {stats.get('index_type', 'N/A')}")
        print(f"  저장 크기       : {stats.get('storage_size_mb', 'N/A')} MB")
        print(f"  PDF 파일        : {rag_system.pdf_path}")
        print(f"  벡터 DB 경로    : {rag_system.vector_db_path}")
        print(f"  임베딩 모델     : {rag_system.embedding_model}")
        print(f"  LLM 모델        : {rag_system.llm_model}")
        print("=" * 80 + "\n")

    elif cmd == "/help":
        # 도움말 표시
        print("\n" + "=" * 80)
        print("도움말 - 사용 가능한 명령어")
        print("=" * 80)
        print("\n[질문하기]")
        print("  일반 질문을 입력하면 됩니다")
        print("  예시: 카메라 해상도는?")
        print("        촬영 모드 종류는?")
        print()
        print("[명령어]")
        print("  /help              - 도움말 표시")
        print("  /stats             - 통계 정보 확인")
        print()
        print("[종료]")
        print("  exit, quit, 끝, q  - 프로그램 종료")
        print("  Ctrl+C             - 강제 종료")
        print()
        print("[참고]")
        print("  벡터 DB는 첫 실행 시 자동으로 생성/저장됩니다")
        print("=" * 80 + "\n")

    else:
        print(f"\n[오류] 알 수 없는 명령어: {cmd}")
        print("       /help 로 사용 가능한 명령어를 확인하세요.\n")


def main() -> None:
    """메인 실행 함수"""
    # RAG 시스템 초기화
    rag_system = RAGSystem()
    rag_system.setup()

    # 대화형 세션 시작
    run_interactive_session(rag_system)


if __name__ == "__main__":
    main()
