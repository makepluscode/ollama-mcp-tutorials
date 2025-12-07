"""
PDF 문서 파싱 및 청킹

이 스크립트는 지정된 PDF 파일을 로드하고, 텍스트를 추출한 후,
미리 정의된 모든 청킹 전략을 사용하여 의미 있는 단위(청크)로 분할하고,
각 전략의 결과를 별도의 텍스트 파일로 저장합니다.
"""

import logging
import os
from datetime import datetime
from typing import List, Literal, Tuple, Dict, Any

import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

# ============================================================================ 
# 설정 (Configuration)
# ============================================================================ 

# 처리할 PDF 파일 경로
PDF_FILE_PATH = "DDSI-RTPS-v2.5.pdf"

# Ollama 임베딩 모델 설정
EMBEDDING_MODEL = "bge-m3"

# 문서 분할 설정 (Semantic Chunker는 이 값을 직접 사용하지 않음)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================ 
# 1. 문서 로더 (Document Loader)
# ============================================================================ 


class PdfDocumentLoader:
    """PDF 문서를 로드하는 클래스"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self) -> Tuple[List[Document], float]:
        """
        PDF 문서를 로딩하여 Document 객체 리스트와 소요 시간을 반환합니다.
        """
        logger.info(f"PDF 문서 로딩 시작: {self.pdf_path}")
        start_time = datetime.now()

        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✓ 문서 로딩 완료 - 페이지 수: {len(docs)}, 소요 시간: {elapsed:.2f}초"
        )
        return docs, elapsed


# ============================================================================ 
# 2. 텍스트 분할기 팩토리 (Text Splitter Factory)
# ============================================================================ 


class ChunkerFactory:
    """다양한 텍스트 분할기(Chunker)를 생성하는 팩토리 클래스"""

    @staticmethod
    def get_text_splitter(
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embeddings: Any = None,
    ) -> TextSplitter:
        """
        주어진 전략에 맞는 TextSplitter 인스턴스를 생성하여 반환합니다.
        """
        logger.info(f"'{strategy}' 청킹 전략에 대한 TextSplitter 생성")
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        elif strategy == "character":
            return CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                is_separator_regex=False,
            )
        elif strategy == "token":
            encoding = tiktoken.get_encoding("cl100k_base")
            return TokenTextSplitter(
                encoding_name=encoding.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif strategy == "semantic":
            if not embeddings:
                raise ValueError("'semantic' 전략은 임베딩 모델이 필요합니다.")
            logger.info("SemanticChunker는 chunk_size/overlap을 직접 사용하지 않고, 임베딩을 통해 의미론적 분할을 수행합니다.")
            return SemanticChunker(embeddings=embeddings)
        else:
            raise ValueError(f"알 수 없는 청킹 전략: {strategy}")


# ============================================================================ 
# 3. 청크 처리 및 저장 (Chunk Processing and Saving)
# ============================================================================ 


def split_documents(
    text_splitter: TextSplitter, docs: List[Document]
) -> Tuple[List[Document], float, float]:
    """
    주어진 TextSplitter를 사용하여 문서를 청크로 분할하고 통계 정보를 반환합니다.
    """
    logger.info("문서 분할 시작")
    start_time = datetime.now()

    # SemanticChunker는 매우 큰 문서를 한 번에 처리할 때 메모리 문제가 발생할 수 있습니다.
    # 여기서는 문서를 페이지별로 분할하여 처리합니다.
    if isinstance(text_splitter, SemanticChunker):
        logger.info("SemanticChunker는 페이지별로 문서를 분할하여 처리합니다.")
        all_splits = []
        for doc in docs:
            all_splits.extend(text_splitter.split_documents([doc]))
        splits = all_splits
    else:
        splits = text_splitter.split_documents(docs)

    elapsed = (datetime.now() - start_time).total_seconds()
    avg_chunk_size = (
        sum(len(s.page_content) for s in splits) / len(splits) if splits else 0
    )

    logger.info(f"✓ 문서 분할 완료 - 청크 수: {len(splits)}, 소요 시간: {elapsed:.2f}초")
    logger.info(f"  평균 청크 크기: {avg_chunk_size:.0f} 문자")
    return splits, elapsed, avg_chunk_size


def find_common_metadata(splits: List[Document]) -> Dict[str, Any]:
    """모든 청크에 공통적으로 나타나는 메타데이터를 찾습니다."""
    if not splits:
        return {}
    
    common_metadata = dict(splits[0].metadata)
    
    for chunk in splits[1:]:
        for key in list(common_metadata.keys()):
            if key not in chunk.metadata or common_metadata[key] != chunk.metadata[key]:
                del common_metadata[key]
    return common_metadata


def save_chunks_to_file(
    output_path: str,
    pdf_path: str,
    strategy: str,
    chunk_config: Dict[str, Any],
    splits: List[Document],
    times: Dict[str, float],
    avg_chunk_size: float,
) -> None:
    """
    분할된 청크 정보를 풍부한 메타데이터와 함께 텍스트 파일에 저장합니다.
    """
    logger.info(f"청크 정보 저장 시작: {output_path}")
    start_time = datetime.now()

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # --- 전체 메타데이터 헤더 ---
            f.write("=" * 80 + "\n")
            f.write(" PDF Chunking Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Source PDF:         {os.path.basename(pdf_path)}\n")
            f.write(f"Full Path:          {os.path.abspath(pdf_path)}\n")
            f.write(f"Processing Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n")
            f.write("Chunking Configuration:\n")
            f.write(f"  - Strategy:         {strategy}\n")
            if strategy != "semantic":
                f.write(f"  - Chunk Size:       {chunk_config['size']}\n")
                f.write(f"  - Chunk Overlap:    {chunk_config['overlap']}\n")
            else:
                f.write(f"  - Embeddings:       {chunk_config['embeddings']}\n")
                f.write("  - Breakpoint Threshold: Default (95th percentile)\n")
            f.write("-" * 80 + "\n")
            f.write("Execution Times:\n")
            f.write(f"  - PDF Loading:      {times['loading']:.4f} seconds\n")
            f.write(f"  - Text Splitting:   {times['splitting']:.4f} seconds\n")
            f.write("-" * 80 + "\n")
            f.write("Result Statistics:\n")
            f.write(f"  - Total Chunks:     {len(splits)}\n")
            f.write(f"  - Average Size:     {avg_chunk_size:.2f} characters\n")
            
            common_metadata = find_common_metadata(splits)
            if common_metadata:
                f.write("Common Metadata:\n")
                for key, value in common_metadata.items():
                    f.write(f"  - {key}: {value}\n")
            
            f.write("=" * 80 + "\n\n")

            # --- 개별 청크 ---
            for i, chunk in enumerate(splits):
                f.write(f"--- Chunk {i+1} / {len(splits)} (Length: {len(chunk.page_content)} chars) ---\n")
                
                unique_metadata = {k: v for k, v in chunk.metadata.items() if k not in common_metadata}
                if unique_metadata:
                    f.write("  Metadata:\n")
                    for key, value in unique_metadata.items():
                        f.write(f"    - {key}: {value}\n")
                
                f.write("-" * 20 + "\n")
                f.write(chunk.page_content.strip())
                f.write("\n\n" + "=" * 80 + "\n\n")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ 청크 정보 저장 완료 - 소요 시간: {elapsed:.2f}초")

    except IOError as e:
        logger.error(f"청크 정보 파일 저장 중 오류 발생: {e}")


# ============================================================================ 
# 메인 실행 함수 (Main Execution)
# ============================================================================ 


def main() -> None:
    """
    모든 정의된 청킹 전략을 실행하고 각각의 결과를 별도 파일로 저장하는 메인 함수
    """
    strategies: List[Literal["recursive", "character", "token", "semantic"]] = [
        "recursive",
        "character",
        "token",
        "semantic",
    ]

    logger.info("=" * 80)
    logger.info("모든 청킹 전략에 대한 PDF 처리 및 저장 프로세스 시작")
    logger.info(f"대상 PDF: {PDF_FILE_PATH}")
    logger.info("=" * 80)
    
    # 1. 문서 로딩 (한 번만 수행)
    loader = PdfDocumentLoader(pdf_path=PDF_FILE_PATH)
    docs, loading_time = loader.load()

    # 2. 임베딩 모델 준비 (Semantic Chunker에 필요)
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        ollama_embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=ollama_base_url,
        )
        # 간단한 테스트로 Ollama 서버 연결 확인
        ollama_embeddings.embed_query("test")
        logger.info(f"Ollama 임베딩 모델 '{EMBEDDING_MODEL}'에 연결되었습니다. (URL: {ollama_base_url})")
    except Exception as e:
        logger.error(f"Ollama 임베딩 모델에 연결할 수 없습니다 (URL: {ollama_base_url}).")
        logger.error("Ollama가 WSL 호스트(Windows)에서 실행 중인 경우, OLLAMA_BASE_URL 환경 변수를 설정해야 할 수 있습니다.")
        logger.error("예: export OLLAMA_BASE_URL=http://<Windows_IP>:11434")
        logger.error(f"'semantic' 전략은 건너뜁니다. 오류: {e}")
        strategies.remove("semantic")


    for strategy in strategies:
        print("\n")
        logger.info("-" * 80)
        logger.info(f"전략 '{strategy}' 실행 시작")
        logger.info("-" * 80)

        # 3. 선택된 전략에 따라 TextSplitter 생성
        text_splitter = ChunkerFactory.get_text_splitter(
            strategy=strategy,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embeddings=ollama_embeddings if strategy == "semantic" else None,
        )

        # 4. 문서 분할
        splits, splitting_time, avg_chunk_size = split_documents(text_splitter, docs)

        # 5. 결과 파일 경로 생성
        output_path = f"chunks-{strategy}.txt"
        
        # 6. 청킹 설정 정보 구성
        chunk_config = {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
        if strategy == "semantic":
            chunk_config = {"embeddings": EMBEDDING_MODEL}


        # 7. 분할된 청크를 파일에 저장
        save_chunks_to_file(
            output_path=output_path,
            pdf_path=PDF_FILE_PATH,
            strategy=strategy,
            chunk_config=chunk_config,
            splits=splits,
            times={"loading": loading_time, "splitting": splitting_time},
            avg_chunk_size=avg_chunk_size,
        )
        logger.info(f"전략 '{strategy}' 실행 완료. 결과 파일: {output_path}")
    
    logger.info("=" * 80)
    logger.info("모든 프로세스가 성공적으로 완료되었습니다.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
