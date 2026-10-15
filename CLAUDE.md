# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses `uv` (Python 3.13). Always use `uv` to run Python files (`uv run python <file>`) and to manage dependencies (`uv add <pkg>`); never call `pip` or bare `python` directly. Requires `ANTHROPIC_API_KEY` in a root `.env` (see `.env.example`).

```bash
uv sync                      # install dependencies
./run.sh                     # start server on :8000 (web UI at /, API docs at /docs)
cd backend && uv run uvicorn app:app --reload --port 8000   # same, manually
```

There are no tests, linter, or build step configured. `main.py` at the root is an unused uv placeholder.

The server must be started from `backend/`: `app.py` uses relative paths `../docs` and `../frontend`, and ChromaDB persists to `backend/chroma_db/`.

## Architecture

A course-materials RAG chatbot: FastAPI backend + vanilla JS frontend served as static files by the same app.

**Query flow** — Retrieval is driven by Claude tool use, not by a fixed retrieve-then-generate step:
1. `POST /api/query` (`app.py`) → `RAGSystem.query()` (`rag_system.py`), the orchestrator that wires all components.
2. `AIGenerator.generate_response()` makes a first Claude call with the `search_course_content` tool available. Claude decides whether to search.
3. If `stop_reason == "tool_use"`, `ToolManager` dispatches to `CourseSearchTool` → `VectorStore.search()`, then a second Claude call is made with the tool results and **no tools** — so at most one search round per query.
4. Sources for the UI are not returned through the call chain: `CourseSearchTool` stores them in `last_sources` as a side effect, and `RAGSystem` reads and resets them via `ToolManager` after generation.

**Conversation history** is held in memory by `SessionManager` (lost on restart, last `MAX_HISTORY` exchanges) and injected as text into the **system prompt**, not as prior `messages`.

**Vector store** (`vector_store.py`) has two ChromaDB collections:
- `course_catalog` — one doc per course (title is the ID); lessons stored as a JSON string in metadata (`lessons_json`). Used to semantically resolve a fuzzy `course_name` to an exact title.
- `course_content` — text chunks with `course_title` / `lesson_number` metadata, filtered via a Chroma `where` clause.

**Ingestion** — On startup, every file in `docs/` is parsed by `DocumentProcessor` and added unless a course with the same title already exists. Editing an existing course file therefore has no effect until `backend/chroma_db/` is deleted (or `add_course_folder(..., clear_existing=True)` is used). Files are read as plain text even though `.pdf`/`.docx` pass the extension filter.

Expected document format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<content...>
```
Chunks are sentence-based (`CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py`) and prefixed with lesson/course context before embedding.

**Adding a tool**: subclass `Tool` in `search_tools.py` (Anthropic tool-definition dict + `execute`) and register it with `ToolManager` in `RAGSystem.__init__`. Note `AIGenerator` only executes one round of tool calls.

All tunables (Claude model, embedding model, chunk size, result count, history length) live in `backend/config.py`.
