from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
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


class TestToolCall:
    def test_direct_answer_makes_one_call(self, generator, tool_manager):
        script(generator, make_response("end_turn", [make_text("Paris")]))
        answer = generator.generate_response("Capital of France?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)
        assert answer == "Paris"
        assert len(calls(generator)) == 1
        tool_manager.execute_tool.assert_not_called()

    def test_tool_use_executes_search_tool_with_model_input(self, generator, tool_manager):
        script(generator,
               make_response("tool_use", [make_text("Searching."), TOOL_USE]),
               make_response("end_turn", [make_text("MCP is a protocol.")]))
        generator.generate_response("What is MCP?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="what is MCP", course_name="MCP")

    def test_second_call_carries_tool_result_and_no_tools(self, generator, tool_manager):
        first_blocks = [make_text("Searching."), TOOL_USE]
        script(generator,
               make_response("tool_use", first_blocks),
               make_response("end_turn", [make_text("MCP is a protocol.")]))
        answer = generator.generate_response("What is MCP?", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)

        assert answer == "MCP is a protocol."
        assert len(calls(generator)) == 2
        second = calls(generator)[1]
        assert "tools" not in second
        assert second["system"] == AIGenerator.SYSTEM_PROMPT
        assert second["messages"] == [
            {"role": "user", "content": "What is MCP?"},
            {"role": "assistant", "content": first_blocks},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "[MCP - Lesson 1]\nMCP standardizes tool access.",
            }]},
        ]

    def test_multiple_tool_uses_are_all_executed(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = ["result one", "result two"]
        script(generator,
               make_response("tool_use", [
                   make_tool_use("t1", "search_course_content", {"query": "a"}),
                   make_tool_use("t2", "get_course_outline", {"course_name": "MCP"}),
               ]),
               make_response("end_turn", [make_text("done")]))
        generator.generate_response("q", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)
        results = calls(generator)[1]["messages"][-1]["content"]
        assert [(r["tool_use_id"], r["content"]) for r in results] == [("t1", "result one"), ("t2", "result two")]

    def test_tool_error_still_produces_an_answer(self, generator, tool_manager):
        tool_manager.execute_tool.side_effect = TypeError("execute() got an unexpected keyword argument 'limit'")
        script(generator,
               make_response("tool_use", [TOOL_USE]),
               make_response("end_turn", [make_text("Sorry, the search failed.")]))
        answer = generator.generate_response("q", tools=[SEARCH_TOOL_DEF], tool_manager=tool_manager)
        assert answer == "Sorry, the search failed."
        result = calls(generator)[1]["messages"][-1]["content"][0]
        assert result["is_error"] is True


class TestExtractText:
    def test_skips_thinking_and_tool_use_blocks(self):
        response = make_response("end_turn", [make_thinking("hmm"), make_text("A"), TOOL_USE, make_text("B")])
        assert AIGenerator._extract_text(response) == "AB"
