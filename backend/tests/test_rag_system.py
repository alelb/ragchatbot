import dataclasses
import urllib.request
from unittest.mock import MagicMock

import pytest
from conftest import MCP_COURSE, make_response, make_text, make_tool_use

from rag_system import RAGSystem

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
        make_response(
            "tool_use",
            [
                make_tool_use(
                    "toolu_1",
                    "search_course_content",
                    {
                        "query": "what problem does MCP solve",
                        "course_name": "MCP",
                        "lesson_number": 1,
                    },
                )
            ],
        ),
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
        assert sources == [
            {
                "text": f"{MCP_COURSE} - Lesson 1",
                "url": "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp",
            }
        ]

    def test_user_query_reaches_the_model(self, rag):
        script(rag, *search_then_answer())
        rag.query(CONTENT_QUESTION)
        first = calls(rag)[0]
        assert CONTENT_QUESTION in first["messages"][0]["content"]
        assert {t["name"] for t in first["tools"]} == {
            "search_course_content",
            "get_course_outline",
        }

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
        script(
            rag,
            search_then_answer()[0],
            RuntimeError("LLM connection dropped"),
            *general_answer(),
        )
        with pytest.raises(RuntimeError):
            rag.query(CONTENT_QUESTION)
        _, sources = rag.query("What is the capital of France?")
        assert sources == []


class TestSequentialToolCalls:
    def test_outline_then_search_returns_sources_of_both(self, rag):
        script(
            rag,
            make_response(
                "tool_use",
                [make_tool_use("t1", "get_course_outline", {"course_name": "MCP"})],
            ),
            make_response(
                "tool_use",
                [
                    make_tool_use(
                        "t2",
                        "search_course_content",
                        {"query": "Creating An MCP Server"},
                    )
                ],
            ),
            make_response(
                "end_turn", [make_text("Lesson 4 is about building an MCP server.")]
            ),
        )
        answer, sources = rag.query("What does lesson 4 of the MCP course cover?")

        assert answer == "Lesson 4 is about building an MCP server."
        assert len(calls(rag)) == 3
        # Round 2 sees the real outline, including lesson 4's title
        assert (
            "Lesson 4: Creating An MCP Server"
            in calls(rag)[1]["messages"][-1]["content"][0]["content"]
        )
        assert sources[0]["text"] == MCP_COURSE
        assert sources[0]["url"].startswith("https://")
        assert any(" - Lesson " in s["text"] for s in sources[1:])
        assert len({s["text"] for s in sources}) == len(sources)


class TestSessionHistory:
    def test_exchange_is_recorded_and_sent_next_time(self, rag):
        session_id = rag.session_manager.create_session()
        script(
            rag, *search_then_answer("First answer."), *general_answer("Second answer.")
        )
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
@pytest.mark.parametrize(
    "question",
    [
        CONTENT_QUESTION,
        "What is covered in lesson 1 of the MCP course?",
        "How does query expansion work in the Chroma retrieval course?",
    ],
)
def test_live_content_question_uses_search_tool(chroma_copy, question):
    from config import config

    if not _llm_reachable(config):
        pytest.skip("configured LLM not reachable")
    rag = RAGSystem(dataclasses.replace(config, CHROMA_PATH=str(chroma_copy)))
    rag.tool_manager.execute_tool = MagicMock(wraps=rag.tool_manager.execute_tool)

    answer, sources = rag.query(question)

    print(
        f"\nQ: {question}\nTool calls: {rag.tool_manager.execute_tool.call_args_list}\nA: {answer!r}\nSources: {sources}"
    )
    assert rag.tool_manager.execute_tool.called, "model answered without searching"
    assert answer.strip(), "empty answer"
    assert sources, "no sources returned"


TWO_STEP_QUESTION = (
    "Search for a course that discusses the same topic as lesson 4 "
    "of the 'Advanced Retrieval for AI with Chroma' course"
)


@pytest.mark.live
def test_live_two_step_question_makes_sequential_tool_calls(chroma_copy):
    """
    Lesson 4 of the Chroma course is "Cross Encoder Re Ranking": the model must first learn
    the title (outline, or a lesson-filtered search), then search with it in a later round.
    May be flaky: a small model can stop after one call or reuse the first call's filters.
    """
    from config import config

    if not _llm_reachable(config):
        pytest.skip("configured LLM not reachable")
    rag = RAGSystem(dataclasses.replace(config, CHROMA_PATH=str(chroma_copy)))
    client = rag.ai_generator.client
    client.messages.create = MagicMock(wraps=client.messages.create)
    rag.tool_manager.execute_tool = MagicMock(wraps=rag.tool_manager.execute_tool)

    answer, sources = rag.query(TWO_STEP_QUESTION)

    tool_calls = rag.tool_manager.execute_tool.call_args_list
    print(
        f"\nQ: {TWO_STEP_QUESTION}\nTool calls: {tool_calls}\n"
        f"API calls: {client.messages.create.call_count}\nA: {answer!r}\nSources: {sources}"
    )
    assert len(tool_calls) == 2, "model did not make two tool calls"
    # One call per round (not two parallel calls), then the final answer without tools
    assert client.messages.create.call_count == 3
    name, query = (
        tool_calls[1].args[0],
        str(tool_calls[1].kwargs.get("query", "")).lower(),
    )
    assert name == "search_course_content"
    assert any(
        word in query for word in ("rank", "cross", "encoder")
    ), "round 2 did not use lesson 4's title"
    assert answer.strip(), "empty answer"
    assert sources, "no sources returned"
