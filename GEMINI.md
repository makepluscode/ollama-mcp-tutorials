# Project: AI Agent Programming Guide

## Project Overview

This repository contains the source code for the "AI Agent Programming Guide: LangChain & LangGraph with MCP". It's a collection of educational Python projects organized into chapters, guiding users from basic chatbot concepts to advanced multi-agent systems.

The key technologies used are:
- **Python** (3.10+)
- **LangChain & LangGraph** for building AI agent workflows.
- **Ollama** for running local large language models.
- **MCP (Model Context Protocol)** for advanced agent communication.
- **uv** as the package manager.

The project is structured into chapters (e.g., `ch03`, `ch04`), each containing one or more self-contained Python projects.

## Building and Running

Each sub-project within a chapter is independent and can be run separately.

**Prerequisites:**
1.  **Python 3.10+**
2.  **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
3.  **Ollama** installed and running.

**To run a specific project:**
1.  Navigate to the project's directory:
    ```bash
    # Example for the meeting note pipeline
    cd ch11
    ```
2.  Install dependencies using `uv`:
    ```bash
    uv sync
    ```
3.  Run the main script:
    ```bash
    uv run python main.py
    ```
    Some projects may require command-line arguments. For example:
    ```bash
    uv run python main.py --audio-path path/to/audio.mp3 --save
    ```

## Development Conventions

- **Modular Structure:** Each chapter is a module, and subdirectories within chapters are individual, runnable projects.
- **Dependency Management:** Each project has its own `pyproject.toml` and `uv.lock` file for managing dependencies with `uv`.
- **Linting & Formatting:** The project uses `black` and `isort` for code formatting and `mypy` for type checking. These can be run via `uv run <tool> .`.
- **Environment Variables:** Some projects require API keys (e.g., `OPENAI_API_KEY`, `LANGCHAIN_API_KEY`) to be set as environment variables.
