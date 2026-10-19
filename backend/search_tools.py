import re
from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class ToolError(str):
    """Tool output that marks a failed call; still a plain str for the model and existing callers"""


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Chroma'); omit to search all courses"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors: an unknown course is a normal result the model can retry, a store failure is not
        if results.error:
            if results.error.startswith("Search error"):
                return ToolError(results.error)
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number is not None:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI: {"text", "url"}, one per lesson
        seen = set()

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI; several chunks often come from the same lesson
            source_text = header[1:-1]
            if source_text not in seen:
                seen.add(source_text)
                # Link to the lesson video, falling back to the course page
                url = None
                if lesson_num is not None:
                    url = self.store.get_lesson_link(course_title, lesson_num)
                if not url:
                    url = self.store.get_course_link(course_title)
                sources.append({"text": source_text, "url": url})

            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting a course's title, link and complete lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last lookup

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get a course outline: course title, course link and the complete list of lessons (number and title). Use for questions about a course's outline, structure or lesson list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Chroma')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline lookup for the given course.

        Args:
            course_name: Course title, possibly partial

        Returns:
            Formatted course outline or error message
        """
        outline = self.store.get_course_outline(course_name)
        if not outline:
            return f"No course found matching '{course_name}'."

        lines = [
            f"Course: {outline['title']}",
            f"Course Link: {outline['course_link'] or 'not available'}",
            f"Lessons ({len(outline['lessons'])}):"
        ]
        for lesson in outline['lessons']:
            lines.append(f"Lesson {lesson['lesson_number']}: {lesson['lesson_title']}")

        # Store the course as the source for the UI
        self.last_sources = [{"text": outline['title'], "url": outline['course_link']}]

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
        self.sources = []  # Sources of all successful calls since the last reset, in call order
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return ToolError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        # Arguments come from the model, which may add unknown keys or send wrong types
        properties = tool.get_tool_definition()["input_schema"].get("properties", {})
        args = {
            name: self._coerce(value, properties[name].get("type"))
            for name, value in kwargs.items()
            if name in properties
        }
        # Clear the tool's own sources so an early return can't leave those of a previous call
        if hasattr(tool, 'last_sources'):
            tool.last_sources = []
        try:
            result = tool.execute(**args)
        except Exception as e:
            return ToolError(f"Error executing tool '{tool_name}': {e}")

        if not isinstance(result, ToolError):
            for source in getattr(tool, 'last_sources', []):
                if source not in self.sources:
                    self.sources.append(source)
        return result

    @staticmethod
    def _coerce(value: Any, schema_type: Optional[str]) -> Any:
        """Convert a scalar to its JSON schema type when that is unambiguous, e.g. "1" -> 1"""
        if schema_type == "integer":
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif schema_type == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value
    
    def get_last_sources(self) -> list:
        """Get the sources of all tool calls since the last reset, without duplicates"""
        return list(self.sources)

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        self.sources = []
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []