from unittest.mock import MagicMock, call

import pytest

from ai_generator import AIGenerator
from search_tools import ToolError
from conftest import make_response, make_text, make_thinking, make_tool_use

SEARCH_TOOL_DEF = {
    "name": "search_course_content",
    "description": "Search course materials",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model="test-model")
    gen.client = MagicMock()
    return gen


@pytest.fixture
def tool_manager():
    manager = MagicMock()
    manager.execute_tool.return_value = "[MCP - Lesson 1]\nMCP standardizes tool access."
    return manager


def script(generator, *responses):
    generator.client.messages.create.side_effect = list(responses)


def calls(generator):
    return [c.kwargs for c in generator.client.messages.create.call_args_list]


TOOL_USE = make_tool_use("toolu_1", "search_course_content", {"query": "what is MCP", "course_name": "MCP"})


class TestFirstCall:
    def test_sends_tools_system_and_query(self, generator, tool_manager):
        script(generator, make_response("end_turn", [make_text("hi")]))
        generator.generate_response("What is MCP?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)
        first = calls(generator)[0]
        assert first["tools"] == [SEARCH_TOOL_DEF]
        assert first["tool_choice"] == {"type": "auto"}
        assert first["system"] == AIGenerator.SYSTEM_PROMPT
        assert first["messages"] == [{"role": "user", "content": "What is MCP?"}]
        assert first["model"] == "test-model"

    def test_history_is_appended_to_system_prompt(self, generator):
        script(generator, make_response("end_turn", [make_text("hi")]))
        generator.generate_response("q", conversation_history="User: hello\nAssistant: hi")
        assert calls(generator)[0]["system"].endswith("Previous conversation:\nUser: hello\nAssistant: hi")

    def test_no_tools_key_without_tools(self, generator):
        script(generator, make_response("end_turn", [make_text("hi")]))
        generator.generate_response("q")
        assert "tools" not in calls(generator)[0]

    def test_ollama_disables_thinking(self):
        gen = AIGenerator(api_key="", model="gemma4", base_url="http://localhost:11434")
        assert gen.base_params["thinking"] == {"type": "disabled"}


TOOL_RESULT = "[MCP - Lesson 1]\nMCP standardizes tool access."
OUTLINE_USE = make_tool_use("toolu_0", "get_course_outline", {"course_name": "X"})
SEARCH_USE = make_tool_use("toolu_1", "search_course_content", {"query": "T"})


def ask(generator, tool_manager, query="q", **kwargs):
    return generator.generate_response(query, tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager, **kwargs)


class TestSingleRound:
    def test_direct_answer_makes_one_call(self, generator, tool_manager):
        script(generator, make_response("end_turn", [make_text("Paris")]))
        answer = ask(generator, tool_manager, "Capital of France?")
        assert answer == "Paris"
        assert len(calls(generator)) == 1
        tool_manager.execute_tool.assert_not_called()

    def test_tool_use_executes_search_tool_with_model_input(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [make_text("Searching."), TOOL_USE]),
               make_response("end_turn", [make_text("MCP is a protocol.")]))
        ask(generator, tool_manager, "What is MCP?")
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="what is MCP", course_name="MCP")

    def test_one_tool_round_then_answer(self, generator, tool_manager):
        first_blocks = [make_text("Searching."), TOOL_USE]
        script(generator,
               make_response("tool_use", first_blocks),
               make_response("end_turn", [make_text("MCP is a protocol.")]))
        answer = ask(generator, tool_manager, "What is MCP?")

        assert answer == "MCP is a protocol."
        assert len(calls(generator)) == 2
        second = calls(generator)[1]
        # The model may still make a second tool call after seeing the result
        assert second["tools"] == [SEARCH_TOOL_DEF]
        assert second["tool_choice"] == {"type": "auto"}
        assert second["system"] == AIGenerator.SYSTEM_PROMPT
        assert second["messages"] == [
            {"role": "user", "content": "What is MCP?"},
            {"role": "assistant", "content": first_blocks},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": TOOL_RESULT,
            }]},
        ]

    def test_multiple_tool_uses_each_get_a_result_in_order(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = ["result one", "result two"]
        script(generator,
               make_response("tool_use", [
                   make_tool_use("t1", "search_course_content", {"query": "a"}),
                   make_tool_use("t2", "get_course_outline", {"course_name": "MCP"}),
               ]),
               make_response("end_turn", [make_text("done")]))
        ask(generator, tool_manager)
        results = calls(generator)[1]["messages"][-1]["content"]
        assert [(r["tool_use_id"], r["content"]) for r in results] == [("t1", "result one"), ("t2", "result two")]

    def test_tool_use_detected_regardless_of_stop_reason(self, generator, tool_manager):
        script(generator,
               make_response("end_turn", [TOOL_USE]),
               make_response("end_turn", [make_text("MCP is a protocol.")]))
        assert ask(generator, tool_manager) == "MCP is a protocol."
        tool_manager.execute_tool.assert_called_once()

    def test_tool_use_without_tool_manager_is_not_executed(self, generator):
        script(generator, make_response("tool_use", [make_text("Let me search."), TOOL_USE]))
        answer = generator.generate_response("q", tools=[SEARCH_TOOL_DEF])
        assert answer == "Let me search."
        assert len(calls(generator)) == 1


class TestSequentialRounds:
    def script_two_rounds(self, generator, tool_manager, final_blocks):
        tool_manager.execute_tool.side_effect = ["Lesson 4: T", TOOL_RESULT]
        script(generator,
               make_response("tool_use", [OUTLINE_USE]),
               make_response("tool_use", [SEARCH_USE]),
               make_response("end_turn", final_blocks))

    def test_two_rounds_then_final_call_without_tools(self, generator, tool_manager):
        self.script_two_rounds(generator, tool_manager, [make_text("Course Y covers T.")])
        answer = ask(generator, tool_manager, "Same topic as lesson 4 of X?")

        assert answer == "Course Y covers T."
        assert tool_manager.execute_tool.call_args_list == [
            call("get_course_outline", course_name="X"),
            call("search_course_content", query="T"),
        ]
        first, second, final = calls(generator)
        assert first["tools"] == second["tools"] == [SEARCH_TOOL_DEF]
        assert "tools" not in final
        assert "tool_choice" not in final
        assert [m["role"] for m in final["messages"]] == ["user", "assistant", "user", "assistant", "user"]
        assert final["messages"][-1]["content"][0]["content"] == TOOL_RESULT

    def test_second_round_sees_first_round_result(self, generator, tool_manager):
        self.script_two_rounds(generator, tool_manager, [make_text("done")])
        ask(generator, tool_manager)
        second = calls(generator)[1]
        assert second["messages"][-1]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_0", "content": "Lesson 4: T"},
        ]

    def test_stops_after_two_rounds_even_if_model_wants_more(self, generator, tool_manager):
        more = make_tool_use("toolu_2", "search_course_content", {"query": "more"})
        self.script_two_rounds(generator, tool_manager, [make_text("partial"), more])
        assert ask(generator, tool_manager) == "partial"
        assert len(calls(generator)) == 3
        assert tool_manager.execute_tool.call_count == 2

    def test_all_calls_share_system_prompt_with_history(self, generator, tool_manager):
        self.script_two_rounds(generator, tool_manager, [make_text("done")])
        ask(generator, tool_manager, conversation_history="User: hi\nAssistant: hello")
        systems = {c["system"] for c in calls(generator)}
        assert len(systems) == 1
        assert systems.pop().endswith("Previous conversation:\nUser: hi\nAssistant: hello")

    def test_tool_call_budget_skips_extra_calls(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [
                   make_tool_use("t1", "search_course_content", {"query": "a"}),
                   make_tool_use("t2", "search_course_content", {"query": "b"}),
                   make_tool_use("t3", "search_course_content", {"query": "c"}),
               ]),
               make_response("end_turn", [make_text("done")]))
        ask(generator, tool_manager)
        assert tool_manager.execute_tool.call_count == 2
        results = calls(generator)[1]["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
        assert results[2]["content"].startswith("Skipped")

    def test_repeated_identical_call_is_not_reexecuted(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [SEARCH_USE]),
               make_response("tool_use", [make_tool_use("toolu_9", "search_course_content", {"query": "T"})]),
               make_response("end_turn", [make_text("done")]))
        assert ask(generator, tool_manager) == "done"
        tool_manager.execute_tool.assert_called_once()
        final = calls(generator)[2]
        assert "tools" not in final
        assert final["messages"][-1]["content"][0]["content"].startswith("Already called")


class TestToolErrors:
    @pytest.mark.parametrize("error", [
        "Tool 'nope' not found",
        "Error executing tool 'search_course_content': boom",
        "Search error: database is locked",
    ])
    def test_tool_error_ends_rounds_with_a_final_answer(self, generator, tool_manager, error):
        tool_manager.execute_tool.return_value = ToolError(error)
        script(generator,
               make_response("tool_use", [TOOL_USE]),
               make_response("end_turn", [make_text("Sorry, the search failed.")]))
        assert ask(generator, tool_manager) == "Sorry, the search failed."
        second = calls(generator)[1]
        assert "tools" not in second
        result = second["messages"][-1]["content"][0]
        assert result["content"] == error
        assert result["is_error"] is True

    def test_tool_exception_still_produces_an_answer(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = TypeError("execute() got an unexpected keyword argument 'limit'")
        script(generator,
               make_response("tool_use", [TOOL_USE]),
               make_response("end_turn", [make_text("Sorry, the search failed.")]))
        assert ask(generator, tool_manager) == "Sorry, the search failed."
        second = calls(generator)[1]
        assert "tools" not in second
        assert second["messages"][-1]["content"][0]["is_error"] is True

    def test_no_results_is_not_a_failure(self, generator, tool_manager):
        tool_manager.execute_tool.return_value = "No course found matching 'Y'"
        script(generator,
               make_response("tool_use", [TOOL_USE]),
               make_response("end_turn", [make_text("No such course.")]))
        ask(generator, tool_manager)
        second = calls(generator)[1]
        assert second["tools"] == [SEARCH_TOOL_DEF]
        assert "is_error" not in second["messages"][-1]["content"][0]

    def test_malformed_tool_input_is_reported_as_error(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [make_tool_use("t1", "search_course_content", "oops")]),
               make_response("end_turn", [make_text("Sorry.")]))
        assert ask(generator, tool_manager) == "Sorry."
        tool_manager.execute_tool.assert_not_called()
        second = calls(generator)[1]
        assert "tools" not in second
        assert second["messages"][-1]["content"][0]["is_error"] is True

    def test_json_string_tool_input_is_decoded(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [make_tool_use("t1", "search_course_content", '{"query": "a"}')]),
               make_response("end_turn", [make_text("done")]))
        ask(generator, tool_manager)
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="a")


class TestEmptyAnswer:
    def test_empty_direct_answer_returns_fallback(self, generator, tool_manager):
        script(generator, make_response("end_turn", [make_thinking("hmm")]))
        assert ask(generator, tool_manager) == AIGenerator.FALLBACK_ANSWER

    def test_empty_answer_after_tools_returns_fallback(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [TOOL_USE]),
               make_response("end_turn", [make_text("  ")]))
        assert ask(generator, tool_manager) == AIGenerator.FALLBACK_ANSWER


class TestExtractText:
    def test_skips_thinking_and_tool_use_blocks(self):
        response = make_response("end_turn", [make_thinking("hmm"), make_text("A"), TOOL_USE, make_text("B")])
        assert AIGenerator._extract_text(response) == "AB"
