"""
============================================================================
ONE-TIME SETUP SCRIPT: Build ChromaDB (데이터베이스 구축 스크립트)
============================================================================

이 스크립트는 웹 페이지에서 콘텐츠를 가져와 청킹(Chunking)하고,
Ollama 임베딩을 사용하여 ChromaDB 데이터베이스를 구축하는 일회성 설정 스크립트입니다.

mcp_server.py를 실행하기 전에 반드시 이 스크립트를 먼저 실행하여
'chroma_db_html' 디렉토리에 데이터베이스를 생성해야 합니다.

실행 방법:
uv run python setup_db.py
============================================================================
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup, NavigableString
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# ============================================================================
# 설정 (Configuration)
# ============================================================================

# 처리할 웹 페이지 URL
URL = "https://community.rti.com/static/documentation/connext-dds/current/doc/manuals/connext_dds_professional/qos_reference/qos_reference/BasicQoS.htm"

# Ollama 임베딩 모델 설정
EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ChromaDB 설정
CHROMA_DB_PATH = "chroma_db_html"
COLLECTION_NAME = "html_chunks"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. HTML 콘텐츠 로드 및 파싱 (Load and Parse HTML)
# ============================================================================


def fetch_html(url: str) -> Optional[str]:
    """
    주어진 URL에서 HTML 콘텐츠를 가져옵니다.

    Args:
        url (str): 가져올 웹 페이지의 URL

    Returns:
        Optional[str]: HTML 콘텐츠 문자열. 실패 시 None 반환.
    """
    logger.info(f"🌐 웹 페이지 콘텐츠 로드 시작: {url}")
    try:
        response = requests.get(url, timeout=10)  # 타임아웃 설정 추가
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        logger.info("✅ 웹 페이지 콘텐츠 로드 성공")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 웹 페이지 로드 중 오류 발생: {e}")
        return None


# ============================================================================
# 2. H2 태그 기준 청킹 (Chunking by H2 Tags)
# ============================================================================


def chunk_by_h2(html_content: str, url: str) -> List[Document]:
    """
    H2 태그를 기준으로 HTML 콘텐츠를 청킹합니다.

    Args:
        html_content (str): 파싱할 HTML 콘텐츠
        url (str): 출처 URL (메타데이터용)

    Returns:
        List[Document]: LangChain Document 객체 리스트
    """
    logger.info("✂️ H2 태그 기준 청킹 프로세스 시작")
    soup = BeautifulSoup(html_content, "html.parser")
    chunks = []

    # --- 2-1. 문서의 도입부 추출 (첫 H2 태그 이전 내용) ---
    first_h2 = soup.find("h2")
    if first_h2:
        content_before_first_h2 = []
        for elem in first_h2.find_all_previous(string=True):
            # 불필요한 태그(script, style 등)와 공백 제외
            if (
                elem.parent.name
                not in ["style", "script", "head", "title", "meta", "[document]"]
                and elem.strip()
            ):
                content_before_first_h2.append(elem.strip())

        # 역순으로 수집된 텍스트를 원래 순서대로 결합
        initial_content = " ".join(reversed(content_before_first_h2))
        if initial_content:
            chunks.append(
                Document(
                    page_content=initial_content,
                    metadata={"source": url, "header": "Introduction"},
                )
            )

    # --- 2-2. H2 태그별 본문 추출 ---
    for h2_tag in soup.find_all("h2"):
        header_text = h2_tag.get_text(strip=True)
        content_parts = []

        # 현재 H2 태그 이후의 형제 노드들을 순회
        for sibling in h2_tag.find_next_siblings():
            if sibling.name == "h2":
                break  # 다음 H2 태그를 만나면 현재 섹션 종료

            # 텍스트 노드인 경우
            if isinstance(sibling, NavigableString) and sibling.strip():
                content_parts.append(sibling.strip())
            # 태그 노드인 경우 (자식 태그가 없거나 있는 경우 모두 처리)
            elif sibling.name:
                # script, style 태그 제외
                if sibling.name in ["script", "style"]:
                    continue
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    content_parts.append(text)

        full_content = " ".join(content_parts)

        # 내용이 있는 경우에만 청크로 추가
        if full_content:
            chunks.append(
                Document(
                    page_content=f"{header_text}: {full_content}",
                    metadata={"source": url, "header": header_text},
                )
            )

    logger.info(f"✅ 청킹 완료 - 총 {len(chunks)}개의 청크 생성됨")
    return chunks


# ============================================================================
# 3. ChromaDB에 저장 (Save to ChromaDB)
# ============================================================================


def save_to_chromadb(
    chunks: List[Document], embeddings: OllamaEmbeddings
) -> Optional[Chroma]:
    """
    청크를 임베딩하여 ChromaDB에 저장합니다.

    Args:
        chunks (List[Document]): 저장할 문서 청크 리스트
        embeddings (OllamaEmbeddings): 임베딩 모델 객체

    Returns:
        Optional[Chroma]: 생성된 ChromaDB 객체. 실패 시 None.
    """
    logger.info("💾 ChromaDB에 데이터 저장 시작...")
    if not chunks:
        logger.warning("⚠️ 저장할 청크가 없습니다. 프로세스를 종료합니다.")
        return None

    # 기존 컬렉션 초기화 (중복 방지)
    try:
        # DB 경로가 존재하면 초기화 시도
        if os.path.exists(CHROMA_DB_PATH):
            logger.info("♻️ 기존 데이터베이스 초기화 중...")
            # 간단하게는 로컬 파일을 삭제하거나, Chroma API로 reset
            import shutil

            shutil.rmtree(CHROMA_DB_PATH)
            logger.info("🗑️ 기존 데이터 삭제 완료")
    except Exception as e:
        logger.warning(f"⚠️ 기존 데이터 삭제 중 오류 (무시됨): {e}")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
        )
        logger.info(
            f"✅ ChromaDB 저장 완료! (위치: '{CHROMA_DB_PATH}', 컬렉션: '{COLLECTION_NAME}')"
        )
        return vectorstore
    except Exception as e:
        logger.error(f"❌ ChromaDB 저장 중 치명적 오류 발생: {e}")
        return None


# ============================================================================
# 메인 실행 함수 (Main Execution)
# ============================================================================


def main() -> None:
    """전체 데이터 처리 파이프라인을 실행합니다."""
    logger.info("=" * 60)
    logger.info("🚀 RAG 데이터베이스 구축 시작")
    logger.info(f"대상 URL: {URL}")
    logger.info("=" * 60)

    # 1. HTML 콘텐츠 가져오기
    html_content = fetch_html(URL)
    if not html_content:
        logger.error("데이터 수집 실패로 프로그램을 종료합니다.")
        return

    # 2. H2 태그 기준으로 청킹
    documents = chunk_by_h2(html_content, URL)
    if not documents:
        logger.error("청킹된 문서가 없습니다. 프로그램을 종료합니다.")
        return

    # 청킹 결과 샘플링 (디버깅용)
    logger.info(f"--- 첫 2개 청크 미리보기 ---")
    for i, doc in enumerate(documents[:2]):
        logger.info(f"[Chunk #{i+1}] 헤더: {doc.metadata.get('header')}")
        logger.info(f"내용(일부): {doc.page_content[:100]}...")
    logger.info("-" * 40)

    # 3. 임베딩 모델 초기화
    logger.info(f"🔌 Ollama 임베딩 모델 연결 시도 ({EMBEDDING_MODEL})...")
    try:
        ollama_embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        # 연결 테스트 (가벼운 쿼리로 테스트)
        ollama_embeddings.embed_query("connection test")
        logger.info(f"✅ Ollama 모델 연결 성공")
    except Exception as e:
        logger.error(
            f"❌ Ollama 연결 실패. Ollama가 실행 중이고 '{EMBEDDING_MODEL}' 모델이 다운로드되었는지 확인하세요."
        )
        logger.error(f"상세 오류: {e}")
        return

    # 4. ChromaDB에 저장
    vectorstore = save_to_chromadb(documents, ollama_embeddings)

    if vectorstore:
        # 5. 저장 검증 (간단한 검색 테스트)
        logger.info("\n🔎 저장 데이터 검증 테스트 (검색)")
        test_query = "What is Durability QoS policy?"
        logger.info(f"질의: '{test_query}'")

        try:
            results = vectorstore.similarity_search(test_query, k=2)
            logger.info("--- 검색 결과 ---")
            for i, doc in enumerate(results):
                logger.info(f"[결과 #{i+1}] {doc.metadata.get('header', 'N/A')}")
                logger.info(f"내용: {doc.page_content[:150]}...")
        except Exception as e:
            logger.error(f"검색 테스트 실패: {e}")

    logger.info("=" * 60)
    logger.info("✨ 모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
