import dataclasses
import urllib.request
from unittest.mock import MagicMock

import pytest

from rag_system import RAGSystem
from conftest import make_response, make_text, make_tool_use, MCP_COURSE

CONTENT_QUESTION = "What problem does MCP solve according to lesson 1?"


@pytest.fixture
def rag(test_config):
    system = RAGSystem(test_config)
    system.ai_generator.client = MagicMock()
    return system


def script(rag, *responses):
    rag.ai_generator.client.messages.create.side_effect = list(responses)


def calls(rag):
    return [c.kwargs for c in rag.ai_generator.client.messages.create.call_args_list]


def search_then_answer(answer="MCP standardizes how apps give context to LLMs."):
    return (
        make_response("tool_use", [make_tool_use(
            "toolu_1", "search_course_content",
            {"query": "what problem does MCP solve", "course_name": "MCP", "lesson_number": 1},
        )]),
        make_response("end_turn", [make_text(answer)]),
    )


def general_answer(answer="Paris."):
    return (make_response("end_turn", [make_text(answer)]),)


class TestContentQuery:
    def test_tool_result_contains_real_course_chunks(self, rag):
        script(rag, *search_then_answer())
        rag.query(CONTENT_QUESTION)
        tool_result = calls(rag)[1]["messages"][-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["content"].startswith(f"[{MCP_COURSE} - Lesson 1]")

    def test_returns_answer_and_sources(self, rag):
        script(rag, *search_then_answer())
        answer, sources = rag.query(CONTENT_QUESTION)
        assert answer == "MCP standardizes how apps give context to LLMs."
        assert sources == [{
            "text": f"{MCP_COURSE} - Lesson 1",
            "url": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp",
        }]

    def test_user_query_reaches_the_model(self, rag):
        script(rag, *search_then_answer())
        rag.query(CONTENT_QUESTION)
        first = calls(rag)[0]
        assert CONTENT_QUESTION in first["messages"][0]["content"]
        assert {t["name"] for t in first["tools"]} == {"search_course_content", "get_course_outline"}

    def test_sources_are_reset_after_query(self, rag):
        script(rag, *search_then_answer(), *general_answer())
        rag.query(CONTENT_QUESTION)
        _, sources = rag.query("What is the capital of France?")
        assert sources == []

    def test_general_question_has_no_sources(self, rag):
        script(rag, *general_answer())
        answer, sources = rag.query("What is the capital of France?")
        assert answer == "Paris."
        assert sources == []
        assert len(calls(rag)) == 1

    def test_sources_do_not_leak_after_failed_query(self, rag):
        script(rag,
               search_then_answer()[0], RuntimeError("LLM connection dropped"),
               *general_answer())
        with pytest.raises(RuntimeError):
            rag.query(CONTENT_QUESTION)
        _, sources = rag.query("What is the capital of France?")
        assert sources == []


class TestSessionHistory:
    def test_exchange_is_recorded_and_sent_next_time(self, rag):
        session_id = rag.session_manager.create_session()
        script(rag, *search_then_answer("First answer."), *general_answer("Second answer."))
        rag.query(CONTENT_QUESTION, session_id)
        rag.query("And lesson 2?", session_id)
        system = calls(rag)[-1]["system"]
        assert CONTENT_QUESTION in system
        assert "First answer." in system


def test_course_analytics(rag):
    analytics = rag.get_course_analytics()
    assert analytics["total_courses"] == 4
    assert MCP_COURSE in analytics["course_titles"]


# --- Live: the LLM configured in .env decides on its own whether to search ---

def _llm_reachable(cfg):
    if cfg.LLM_BASE_URL:
        try:
            urllib.request.urlopen(f"{cfg.LLM_BASE_URL}/api/tags", timeout=3)
            return True
        except OSError:
            return False
    return bool(cfg.ANTHROPIC_API_KEY)


@pytest.mark.live
@pytest.mark.parametrize("question", [
    CONTENT_QUESTION,
    "What is covered in lesson 1 of the MCP course?",
    "How does query expansion work in the Chroma retrieval course?",
])
def test_live_content_question_uses_search_tool(chroma_copy, question):
    from config import config
    if not _llm_reachable(config):
        pytest.skip("configured LLM not reachable")
    rag = RAGSystem(dataclasses.replace(config, CHROMA_PATH=str(chroma_copy)))
    rag.tool_manager.execute_tool = MagicMock(wraps=rag.tool_manager.execute_tool)

    answer, sources = rag.query(question)

    print(f"\nQ: {question}\nTool calls: {rag.tool_manager.execute_tool.call_args_list}\nA: {answer!r}\nSources: {sources}")
    assert rag.tool_manager.execute_tool.called, "model answered without searching"
    assert answer.strip(), "empty answer"
    assert sources, "no sources returned"
