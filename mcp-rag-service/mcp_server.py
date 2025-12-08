"""
============================================================================
MCP Server: HTML Knowledge Base Search (Official MCP SDK)
============================================================================

이 서버는 공식 Python MCP SDK(mcp)를 사용하여 구현된 RAG 검색 서버입니다.
FastMCP를 사용하여 간단하게 도구(Tool)를 정의하고 노출합니다.

주요 기능:
- `search_knowledge_base`: chroma_db_html에서 관련 문서를 검색합니다.

실행 방법 (Claude Desktop 설정):
{
  "mcpServers": {
    "rag-search": {
      "command": "uv",
      "args": ["run", "mcp_server.py"]
    }
  }
}
"""

import logging
import os
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

# ============================================================================
# 설정 (Configuration)
# ============================================================================

# FastMCP 서버 초기화
mcp = FastMCP("RAG Search Service")

# ChromaDB 설정
CHROMA_DB_PATH = "chroma_db_html"
COLLECTION_NAME = "html_chunks"
EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# 전역 변수로 리소스 유지
vectorstore: Optional[Chroma] = None
ollama_embeddings: Optional[OllamaEmbeddings] = None

# 로깅 설정 (stderr로 출력하여 stdio 통신 방해 금지)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 리소스 초기화 (Initialization)
# ============================================================================


def initialize_resources():
    """임베딩 모델과 ChromaDB를 초기화합니다."""
    global vectorstore, ollama_embeddings

    if vectorstore is not None:
        return

    logger.info("🛠️ 리소스 초기화 중...")

    try:
        # 1. Ollama 임베딩
        ollama_embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # 2. ChromaDB 로드
        if not os.path.exists(CHROMA_DB_PATH):
            logger.error(f"❌ DB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
            logger.error("먼저 'uv run python setup_db.py'를 실행하세요.")
            return

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ollama_embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        logger.info("✅ 리소스 초기화 완료")

    except Exception as e:
        logger.error(f"❌ 리소스 초기화 실패: {e}")


# ============================================================================
# 도구 정의 (Tools)
# ============================================================================


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    Search the HTML knowledge base for relevant information.
    Use this tool to find answers about Connext DDS QoS policies or other indexed content.

    Args:
        query: The question or search term to look up.
    """
    # Lazy initialization (첫 요청 시 로드)
    initialize_resources()

    if vectorstore is None:
        return "오류: 데이터베이스가 초기화되지 않았습니다. 서버 로그를 확인하거나 관리자에게 문의하세요."

    logger.info(f"🔍 검색 요청 수신: '{query}'")

    try:
        results: List[Document] = vectorstore.similarity_search(query, k=3)

        if not results:
            return "검색 결과가 없습니다."

        response_parts = [f"검색 결과 ('{query}' 관련):"]
        for i, doc in enumerate(results):
            header = doc.metadata.get("header", "N/A")
            source = doc.metadata.get("source", "N/A")
            content = doc.page_content

            response_parts.append(
                f"\n--- 결과 {i+1} ---\n"
                f"제목: {header}\n"
                f"출처: {source}\n"
                f"내용: {content}"
            )

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"


# ============================================================================
# 메인 실행 (Main Execution)
# ============================================================================

if __name__ == "__main__":
    # MCP 서버 실행 (stdio 모드가 기본값)
    mcp.run()
