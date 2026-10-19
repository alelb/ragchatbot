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
