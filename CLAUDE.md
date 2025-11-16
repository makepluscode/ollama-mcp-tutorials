# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains example code for an AI agent programming guide using LangChain and LangGraph with MCP (Model Context Protocol). The project is organized into progressive chapters (ch03-ch11) that teach AI agent development from basic Ollama chat to advanced MCP-enabled multi-agent systems.

### Educational Progression
- **ch03**: Basic LangChain concepts with Ollama
- **ch04**: Prompt engineering and chat history
- **ch05**: RAG (Retrieval-Augmented Generation) implementations
- **ch06**: Observability and tracing with LangSmith
- **ch07**: Tool-enabled agents
- **ch08**: Multi-agent systems with LangGraph
- **ch09**: MCP protocol integration
- **ch10**: Advanced MCP use cases
- **ch11**: Real-world application (meeting transcription)

## Development Commands

### Running Individual Projects
Each chapter contains independent Python projects with their own dependencies:

```bash
# Navigate to any chapter/project directory
cd ch11  # or ch09/01_mcp_agent, etc.

# Install dependencies
uv sync

# Run the main script
uv run python main.py

# For projects with CLI arguments (like ch11)
uv run python main.py --audio-path path/to/audio.mp3 --save
```

### Development Tools (where available)
Some projects include development dependencies:

```bash
# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Testing
uv run pytest
```

### Common Issues and Solutions

#### MCP Projects (ch09-ch10)
- **Error**: "MCP client initialization failed"
  - Ensure `mcp_config.json` exists in project directory
  - Verify MCP server paths in configuration are correct

- **Async/await patterns**: MCP projects use `nest_asyncio` to enable nested event loops
  - Main entry must use `asyncio.run()` for async functions
  - Always cleanup MCP clients in finally blocks

#### Ollama Integration
- **Error**: "Model not found" or connection errors
  - Verify Ollama is running: Check that Ollama service is active
  - Pull required model: `ollama pull qwen2.5:7b` or `ollama pull qwen3:8b`
  - Common models used: `qwen2.5:7b`, `qwen3:8b`, `llama3:8b`

## Architecture Overview

### Project Structure Pattern
Most chapters follow this structure:
- `main.py` - Entry point and CLI interface
- `pyproject.toml` - Dependencies and project configuration (using uv package manager)
- `src/` - Core implementation modules (in more complex projects like ch11)
- `README.md` - Chapter-specific instructions and explanations
- `.env` - Environment variables (API keys, optional depending on project)

### Key Architectural Patterns

#### 1. LangChain Integration (ch03-ch07)
- Uses `ChatOllama` for local LLM integration
- Implements chains with LCEL (LangChain Expression Language)
- Progressive complexity: basic chat → tools → agents

#### 2. LangGraph Multi-Agent Systems (ch08)
- Supervisor pattern for agent coordination
- State management with `AgentState` TypedDict
- Conditional routing between specialized agents
- Memory persistence with checkpointers

#### 3. MCP (Model Context Protocol) Integration (ch09-ch10)
- `mcp_manager.py` - MCP client lifecycle management with async context managers
  - `initialize_mcp_client()` - Creates and enters MCP client context
  - `cleanup_mcp_client()` - Safely exits client (handles CancelledError)
  - `load_mcp_config()` - Reads from `mcp_config.json`
- `mcp_config.json` - Server configuration defining MCP servers and their settings
- `mcp_prompt.py` - Specialized prompts for MCP-enabled agents
- `langchain_mcp_adapters.client.MultiServerMCPClient` for tool integration
- Async patterns for MCP communication (requires `nest_asyncio` for Jupyter compatibility)
- MCP server implementations in `mcp_server/` subdirectories

#### 4. Audio Processing Pipeline (ch11)
- LangGraph workflow with sequential nodes:
  1. `validate_input` - Path validation
  2. `transcribe` - Audio → text via Faster Whisper
  3. `summarize` - Text → structured summary via LLM
  4. `save_file` - Write markdown output
- Components in `src/`:
  - `transcriber.py` - `AudioTranscriber` class with lazy model loading
  - `summarizer.py` - `MeetingSummarizer` with structured prompt templates
  - `file_handler.py` - `FileHandler` for markdown generation
  - `pipeline.py` - `MeetingNotesPipeline` orchestrating LangGraph workflow
- State management via `MeetingState` TypedDict
- Graph visualization saved as `graph.png` during initialization

### Common Dependencies
- **LangChain ecosystem**: `langchain-core`, `langchain-ollama`, `langchain-community`
- **LangGraph**: State graphs, checkpointing, prebuilt agents
- **MCP**: `mcp`, `langchain-mcp-adapters` for protocol integration
- **Local LLM**: Ollama for running models locally

### Development Patterns

#### Error Handling
- Async context managers for MCP clients (`__aenter__`/`__aexit__`)
- Try-catch blocks with cleanup in finally blocks
- Graceful degradation when services unavailable

#### Configuration Management
- Environment variables via `python-dotenv`
- JSON configuration files for MCP servers
- CLI argument parsing with `argparse`

#### State Management
- TypedDict for structured state in LangGraph
- Message annotation with `add_messages`
- Thread-based conversation memory

## Prerequisites

### Required Services
- **Ollama**: Install and run locally for LLM inference
  - Download from https://ollama.com/download
  - Start service (runs automatically on macOS/Windows, systemd on Linux)
- **Models**: Pull required models before running projects
  - `ollama pull qwen2.5:7b` (commonly used in early chapters)
  - `ollama pull qwen3:8b` (used in ch11 and some ch09/ch10 projects)
  - `ollama pull llama3:8b` (alternative for ch11)

### Environment Setup
- **Python 3.10+** (some projects require 3.12+)
- **UV package manager** for dependency management
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - UV replaces pip/venv and handles project dependencies via `pyproject.toml`

### Optional API Keys (set in `.env` file)
- **OpenAI API**: `OPENAI_API_KEY` - For projects using GPT models (some ch08-ch10 projects)
- **LangSmith**: `LANGCHAIN_API_KEY` - For tracing and monitoring (ch06)
- **Tavily**: `TAVILY_API_KEY` - For web search tools (ch07-ch08)
- **Notion**: `NOTION_API_KEY` and `NOTION_DATABASE_ID` - For ch10/01_mcp_agent_notion

## Chapter-Specific Notes

### ch03-ch05: Foundation Concepts
- **ch03**: Basic Ollama integration, streaming, and output parsing
- **ch04**: Prompt templates and conversation memory
- **ch05**: Vector stores, embeddings, and RAG patterns

### ch06: Observability
- LangSmith integration for tracing and monitoring
- Dataset management and evaluation

### ch07: Tool-Enabled Agents
- Single and multi-tool agent implementations
- ReAct pattern with LangChain agents

### ch08: Multi-Agent Systems
- Complex state routing logic
- Supervisor pattern for agent coordination
- Graph visualization capabilities

### ch09-ch10: MCP Integration
- Require proper MCP server configuration in `mcp_config.json`
- Use async/await patterns throughout
- Handle MCP client lifecycle carefully
- **ch09**: Basic MCP agent with weather tools
- **ch10**: Advanced MCP integrations (Notion, news aggregation)

### ch11: Real-world Application  
- **ch11**: Meeting transcription system using LangGraph workflows
- Features audio processing pipeline with structured state management
- Implements modular components: transcriber, summarizer, file handler
- Uses Faster Whisper for speech-to-text and Ollama for summarization

## Development Commands by Chapter

### Chapter 11 (Current - Meeting Notes Pipeline)
```bash
# Navigate to ch11 directory
cd ch11

# Install dependencies 
uv sync

# Run with default sample audio
uv run python main.py

# Process custom audio file with save
uv run python main.py --audio-path path/to/audio.mp3 --save

# Specify output location
uv run python main.py --audio-path input.mp3 --output meeting_notes.md --save

# Use different Whisper model size
uv run python main.py --model-size large --llm-model llama3:8b

# Development tools (when available)
uv run black .
uv run isort .
uv run mypy .
```

### Code Style Guidelines
Ch11 follows specific Korean coding conventions defined in `ch11/coding_rules.md`:
- **Language**: Comments, docstrings, and user-facing text in Korean; code identifiers in English
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Documentation**: Google-style docstrings in Korean with Args/Returns/Raises sections
- **Error handling**: User-friendly Korean error messages with actionable solutions
- **String formatting**: F-strings preferred over % or .format()
- **Code organization**: Sequential step numbering in comments (`# 1.`, `# 2.`, etc.)
- **Type hints**: Full type annotations using `typing` module (Optional, Dict, Any, etc.)
- **Design patterns**: Lazy loading for heavy resources (models), class-based component design

Note: Other chapters use English documentation and follow standard Python conventions.

## Key Concepts by Chapter Type

### Agent Creation Patterns
- **ch07**: Uses `create_react_agent()` from LangChain with single/multiple tools
- **ch08**: Uses `create_react_agent()` within LangGraph workflows with state management
- **ch09-ch10**: Combines LangGraph + MCP tools via `langchain_mcp_adapters`
  - Agent nodes created with `create_react_agent(model=chat_model, tools=mcp_tools)`
  - Supervisor pattern for routing between specialized agents

### State Management in LangGraph
- Define state with `TypedDict` and `Annotated[list, add_messages]` for message handling
- Create workflow with `StateGraph(StateClass)`
- Add nodes with `graph_builder.add_node("node_name", node_function)`
- Connect nodes with `add_edge()` for sequential flow or `add_conditional_edges()` for routing
- Compile with `graph_builder.compile()` (optionally with `checkpointer` for memory)

### Async Patterns (MCP Projects)
```python
# Standard MCP initialization pattern
async def main():
    client, tools = None, None
    try:
        client, tools = await initialize_mcp_client()
        # Use tools with agent
        agent = create_agent(mcp_tools=tools)
        # Run agent
        result = await agent.ainvoke(...)
    finally:
        await cleanup_mcp_client(client)

# Entry point
if __name__ == "__main__":
    nest_asyncio.apply()  # For Jupyter compatibility
    asyncio.run(main())
```

## Learning Path Recommendations

1. **Start with ch03** for basic LangChain concepts
2. **Progress through ch04-ch05** for foundational patterns
3. **Use ch06** to understand observability early in development
4. **Explore ch07-ch08** for agent architectures
5. **Advance to ch09-ch10** for cutting-edge MCP integration
6. **Complete with ch11** for real-world application implementation