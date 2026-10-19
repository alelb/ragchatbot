import pytest

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from conftest import make_results, MCP_COURSE

MCP_COURSE_LINK = "https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/"
MCP_LESSON_1_LINK = "https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp"


# --- CourseSearchTool.execute with a mocked store ---

class TestExecuteFormatting:
    def test_forwards_arguments_to_store(self, mock_store):
        mock_store.search.return_value = make_results([])
        CourseSearchTool(mock_store).execute("tools", course_name="MCP", lesson_number=2)
        mock_store.search.assert_called_once_with(query="tools", course_name="MCP", lesson_number=2)

    def test_formats_results_with_course_and_lesson_header(self, mock_store):
        mock_store.search.return_value = make_results([
            ("chunk one", {"course_title": "Course A", "lesson_number": 1}),
            ("chunk two", {"course_title": "Course A", "lesson_number": 2}),
        ])
        output = CourseSearchTool(mock_store).execute("q")
        assert output == "[Course A - Lesson 1]\nchunk one\n\n[Course A - Lesson 2]\nchunk two"

    def test_header_without_lesson_number(self, mock_store):
        mock_store.search.return_value = make_results([
            ("chunk", {"course_title": "Course A", "lesson_number": None}),
        ])
        assert CourseSearchTool(mock_store).execute("q") == "[Course A]\nchunk"

    def test_returns_store_error_verbatim(self, mock_store):
        mock_store.search.return_value = SearchResults.empty("No course found matching 'xyz'")
        tool = CourseSearchTool(mock_store)
        assert tool.execute("q", course_name="xyz") == "No course found matching 'xyz'"
        assert tool.last_sources == []


class TestExecuteEmptyResults:
    @pytest.mark.parametrize("kwargs, expected", [
        ({}, "No relevant content found."),
        ({"course_name": "MCP"}, "No relevant content found in course 'MCP'."),
        ({"lesson_number": 3}, "No relevant content found in lesson 3."),
        ({"course_name": "MCP", "lesson_number": 3}, "No relevant content found in course 'MCP' in lesson 3."),
        ({"lesson_number": 0}, "No relevant content found in lesson 0."),
    ])
    def test_empty_message_mentions_filters(self, mock_store, kwargs, expected):
        mock_store.search.return_value = make_results([])
        assert CourseSearchTool(mock_store).execute("q", **kwargs) == expected


class TestExecuteSources:
    def test_one_source_per_lesson_with_lesson_link(self, mock_store):
        mock_store.search.return_value = make_results([
            ("a", {"course_title": "Course A", "lesson_number": 1}),
            ("b", {"course_title": "Course A", "lesson_number": 1}),
            ("c", {"course_title": "Course A", "lesson_number": 2}),
        ])
        mock_store.get_lesson_link.side_effect = lambda title, n: f"https://lesson/{n}"
        tool = CourseSearchTool(mock_store)
        tool.execute("q")
        assert tool.last_sources == [
            {"text": "Course A - Lesson 1", "url": "https://lesson/1"},
            {"text": "Course A - Lesson 2", "url": "https://lesson/2"},
        ]

    def test_falls_back_to_course_link(self, mock_store):
        mock_store.search.return_value = make_results([
            ("a", {"course_title": "Course A", "lesson_number": 1}),
        ])
        mock_store.get_course_link.return_value = "https://course"
        tool = CourseSearchTool(mock_store)
        tool.execute("q")
        assert tool.last_sources == [{"text": "Course A - Lesson 1", "url": "https://course"}]


class TestExecuteLLMArguments:
    """Arguments come from the model, which (especially local ones) may not respect the schema"""

    def test_string_lesson_number_is_coerced_to_int(self, mock_store):
        mock_store.search.return_value = make_results([])
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        manager.execute_tool("search_course_content", query="q", lesson_number="1")
        assert mock_store.search.call_args.kwargs["lesson_number"] == 1

    def test_unexpected_argument_does_not_raise(self, mock_store):
        mock_store.search.return_value = make_results([])
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        result = manager.execute_tool("search_course_content", query="q", limit=3)
        assert isinstance(result, str)
        assert "limit" not in mock_store.search.call_args.kwargs

    def test_missing_required_argument_returns_error(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        result = manager.execute_tool("search_course_content", course_name="MCP")
        assert result.startswith("Error executing tool 'search_course_content'")

    def test_store_exception_returns_error(self, mock_store):
        mock_store.search.side_effect = RuntimeError("db locked")
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        assert manager.execute_tool("search_course_content", query="q") == \
            "Error executing tool 'search_course_content': db locked"


# --- ToolManager ---

class TestToolManager:
    def test_dispatches_by_name(self, mock_store):
        mock_store.search.return_value = make_results([("doc", {"course_title": "C", "lesson_number": 1})])
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        assert manager.execute_tool("search_course_content", query="q") == "[C - Lesson 1]\ndoc"

    def test_unknown_tool(self):
        assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"

    def test_tool_definition_requires_query(self, mock_store):
        definition = CourseSearchTool(mock_store).get_tool_definition()
        assert definition["name"] == "search_course_content"
        assert definition["input_schema"]["required"] == ["query"]

    def test_last_sources_and_reset(self, mock_store):
        mock_store.search.return_value = make_results([("doc", {"course_title": "C", "lesson_number": 1})])
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        manager.execute_tool("search_course_content", query="q")
        assert manager.get_last_sources() == [{"text": "C - Lesson 1", "url": None}]
        manager.reset_sources()
        assert manager.get_last_sources() == []


# --- CourseSearchTool.execute against the real (copied) ChromaDB ---

class TestExecuteRealStore:
    def test_course_filter_returns_only_that_course(self, real_store):
        tool = CourseSearchTool(real_store)
        output = tool.execute("what is MCP", course_name="MCP")
        assert output.startswith(f"[{MCP_COURSE}")
        headers = [part.split("\n", 1)[0] for part in output.split("\n\n[")]
        assert all(MCP_COURSE in h for h in headers)
        assert tool.last_sources
        assert all(s["url"] and s["url"].startswith("https://") for s in tool.last_sources)

    def test_lesson_filter_returns_only_that_lesson(self, real_store):
        tool = CourseSearchTool(real_store)
        tool.execute("why use MCP", course_name="MCP", lesson_number=1)
        assert tool.last_sources == [{"text": f"{MCP_COURSE} - Lesson 1", "url": MCP_LESSON_1_LINK}]

    def test_search_without_filters_returns_content(self, real_store):
        output = CourseSearchTool(real_store).execute("query expansion with an LLM")
        assert output.startswith("[")
        assert "No relevant content" not in output

    def test_unknown_course_is_reported(self, real_store):
        output = CourseSearchTool(real_store).execute("pasta recipes", course_name="Italian cooking")
        assert output == "No course found matching 'Italian cooking'"


class TestCourseNameResolution:
    @pytest.mark.parametrize("name, expected", [
        ("MCP", MCP_COURSE),
        ("the mcp course", MCP_COURSE),
        ("Chroma retrieval", "Advanced Retrieval for AI with Chroma"),
        ("prompt compression", "Prompt Compression and Query Optimization"),
        ("computer use", "Building Towards Computer Use with Anthropic"),
    ])
    def test_real_names_resolve(self, real_store, name, expected):
        assert real_store._resolve_course_name(name) == expected

    @pytest.mark.parametrize("name", ["Italian cooking", "pasta recipes", "history of Rome", "Python basics"])
    def test_unrelated_names_do_not_resolve(self, real_store, name):
        assert real_store._resolve_course_name(name) is None
