# Project: PDF Document Chunking for RAG

## Project Overview

This project contains a Python script (`main.py`) designed to preprocess a PDF document for use in a Retrieval-Augmented Generation (RAG) pipeline. Its primary function is to load a specified PDF, parse its content, and split it into smaller, manageable text "chunks." The resulting chunks and their metadata are then saved to a text file (`chunks.txt`).

This serves as a foundational step for creating a vector database for semantic search.

The key technologies used are:
- **Python** (3.10+)
- **LangChain** for document loading (`PyPDFLoader`) and text splitting (`RecursiveCharacterTextSplitter`).
- **uv** as the package manager.

## Building and Running

This is a self-contained Python script.

**Prerequisites:**
1.  **Python 3.10+**
2.  **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

**To run the script:**
1.  Install dependencies using `uv`:
    ```bash
    uv sync
    ```
2.  Run the main script:
    ```bash
    uv run python main.py
    ```
This will read the `DDSI-RTPS-v2.5.pdf` file and generate/overwrite the `chunks.txt` file in the same directory.

## Development Conventions

- **Configuration:** The script can be configured by modifying the constants at the top of `main.py`, such as `PDF_FILE_PATH`, `OUTPUT_FILE_PATH`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`.
- **Dependency Management:** The project uses `uv` to manage dependencies, which are defined in `pyproject.toml`.
- **Output:** The script logs its progress to the console and writes the final chunked text to `chunks.txt`.
