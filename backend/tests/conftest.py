import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_store import SearchResults, VectorStore

BACKEND_DIR = Path(__file__).resolve().parent.parent
LIVE_CHROMA_PATH = BACKEND_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MCP_COURSE = "MCP: Build Rich-Context AI Apps with Anthropic"


# Fake Anthropic SDK objects: only the attributes AIGenerator reads


def make_text(text):
    return SimpleNamespace(type="text", text=text)


def make_thinking(thinking):
    return SimpleNamespace(type="thinking", thinking=thinking)


def make_tool_use(id, name, input):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def make_response(stop_reason, blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def make_results(docs_and_meta, error=None):
    """Build SearchResults from [(document, metadata), ...]"""
    return SearchResults(
        documents=[doc for doc, _ in docs_and_meta],
        metadata=[meta for _, meta in docs_and_meta],
        distances=[0.1] * len(docs_and_meta),
        error=error,
    )


@pytest.fixture
def mock_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    return store


@pytest.fixture(scope="session")
def chroma_copy(tmp_path_factory):
    """Copy of the live ChromaDB so tests use real data without touching it"""
    if not (LIVE_CHROMA_PATH / "chroma.sqlite3").exists():
        pytest.skip("backend/chroma_db is empty: start the server once to ingest docs/")
    path = tmp_path_factory.mktemp("chroma") / "chroma_db"
    shutil.copytree(LIVE_CHROMA_PATH, path)
    return path


@pytest.fixture(scope="session")
def real_store(chroma_copy):
    return VectorStore(str(chroma_copy), EMBEDDING_MODEL, max_results=5)


@pytest.fixture
def test_config(chroma_copy):
    """Config with the config.py fields, independent of .env"""
    return SimpleNamespace(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="test-model",
        LLM_BASE_URL="",
        EMBEDDING_MODEL=EMBEDDING_MODEL,
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        COURSE_MATCH_MAX_DISTANCE=1.25,
        CHROMA_PATH=str(chroma_copy),
    )


# FastAPI app


@pytest.fixture
def mock_rag():
    """RAGSystem stand-in for the API: mocked query/analytics, real SessionManager"""
    from rag_system import RAGSystem
    from session_manager import SessionManager

    rag = MagicMock(spec=RAGSystem)
    rag.session_manager = SessionManager(max_history=2)
    rag.query.return_value = ("An answer.", [])
    rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
    return rag


@pytest.fixture
def api_client(mock_rag, monkeypatch):
    """TestClient over the real app.py, imported with RAGSystem replaced by mock_rag.

    app.py builds its RAGSystem and mounts ../frontend at import time, relative to
    the cwd, so it is imported fresh from backend/. The client is not used as a
    context manager, so the startup event (ingestion of docs/) does not run.
    """
    import importlib
    import sys

    from fastapi.testclient import TestClient

    import rag_system

    monkeypatch.chdir(BACKEND_DIR)
    monkeypatch.setattr(rag_system, "RAGSystem", lambda config: mock_rag)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    app_module = importlib.import_module("app")
    yield TestClient(app_module.app)
    sys.modules.pop("app", None)
