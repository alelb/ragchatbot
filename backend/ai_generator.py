import json
import anthropic
from typing import List, Optional, Dict, Any, Tuple

from search_tools import ToolError

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **get_course_outline**(course_name): course title, link and the numbered list of lessons with their titles
- **search_course_content**(query, course_name?, lesson_number?): passages from the course materials; omit course_name to search all courses
- Course-specific questions: call a tool before answering. General knowledge questions: answer directly without tools
- You can use tools in **up to 2 steps**. After each tool result, decide:
 - the results answer the question -> answer now
 - you need information that depends on that result -> make one more tool call that uses it
- Example: "Which course covers the same topic as lesson 4 of course X?"
 - Step 1: get_course_outline(course_name="X") -> lesson 4 is titled "T"
 - Step 2: search_course_content(query="T") -> passages from the courses that discuss T
 - Then answer, naming the course(s)
- Never say you are going to search: call the tool
- Do not repeat a tool call with the same arguments
- If a tool returns an error or nothing relevant, say so briefly; do not guess
- When no tools are available, answer from the tool results you already have
- Synthesize tool results into accurate, fact-based responses

Outline Responses (when the user asks for a course outline):
- Include the course title, the course link, and every lesson with its number and title
- List all lessons in order; do not omit or summarize any
- Format as Markdown, exactly in this shape (blank lines included):

### <course title>
[View course](<course link>)

**Lessons**
- **Lesson 0:** <lesson title>
- **Lesson 1:** <lesson title>

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Tool rounds (API calls with tools) and tool executions allowed per query
    MAX_TOOL_ROUNDS = 2
    MAX_TOOL_CALLS = 2

    # Returned when the model produces no text, so the UI never shows an empty answer
    FALLBACK_ANSWER = "Sorry, I couldn't produce an answer to that question."

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if base_url:
            # Local Anthropic-compatible server (e.g. Ollama): the SDK requires a key, the server ignores it
            self.client = anthropic.Anthropic(api_key=api_key or "ollama", base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

        # Local models may think by default, which spends the token budget and adds thinking blocks
        if base_url:
            self.base_params["thinking"] = {"type": "disabled"}

    @staticmethod
    def _extract_text(response) -> str:
        """Join the text blocks of a response, skipping thinking/tool_use blocks"""
        return "".join(block.text for block in response.content if block.type == "text")
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        The model may use tools in up to MAX_TOOL_ROUNDS sequential rounds, each a separate
        API call that sees the previous results. Rounds stop early when the model answers
        or a tool call fails; a final call without tools then produces the answer.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        messages = [{"role": "user", "content": query}]
        use_tools = bool(tools) and tool_manager is not None
        executed = set()  # (name, args) of the tool calls run so far

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._create(messages, system_content, tools if use_tools else None)

            # Local models may return tool_use blocks with a stop_reason other than "tool_use"
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses or not use_tools:
                return self._extract_text(response).strip() or self.FALLBACK_ANSWER

            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed, ran_any = self._run_tools(tool_uses, tool_manager, executed)
            messages.append({"role": "user", "content": tool_results})

            # A failed call or only repeated/skipped calls: further rounds won't help
            if failed or not ran_any:
                break

        # Last call without tools, so the model answers from the results it has
        final_response = self._create(messages, system_content, None)
        return self._extract_text(final_response).strip() or self.FALLBACK_ANSWER

    def _create(self, messages: List[Dict[str, Any]], system: str, tools: Optional[List]):
        """Make one API call; tools are omitted entirely to stop tool use (tool_choice "none" is not portable)"""
        api_params = {
            **self.base_params,
            # Copy: later rounds append to the list, and each call must keep the messages it was sent
            "messages": list(messages),
            "system": system
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**api_params)

    def _run_tools(self, tool_uses, tool_manager, executed: set) -> Tuple[List[Dict[str, Any]], bool, bool]:
        """
        Execute the tool calls of one round, one tool_result per tool_use block.

        Args:
            tool_uses: tool_use blocks of the model response
            tool_manager: Manager to execute tools
            executed: (name, args) of the calls already run for this query; updated in place

        Returns:
            (tool results, whether any call failed, whether any call was actually executed)
        """
        tool_results = []
        failed = ran_any = False
        for block in tool_uses:
            result = {"type": "tool_result", "tool_use_id": block.id}
            args = block.input
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            key = (block.name, json.dumps(args, sort_keys=True, default=str))

            if not isinstance(args, dict):
                output = ToolError(f"Invalid arguments for tool '{block.name}': expected an object")
            elif key in executed:
                output = "Already called with the same arguments; the result is above."
            elif len(executed) >= self.MAX_TOOL_CALLS:
                output = "Skipped: tool call limit for this question reached."
            else:
                executed.add(key)
                ran_any = True
                try:
                    output = tool_manager.execute_tool(block.name, **args)
                except Exception as e:
                    # Let the model answer anyway instead of failing the whole query
                    output = ToolError(f"Error executing tool '{block.name}': {e}")

            result["content"] = str(output)
            if isinstance(output, ToolError):
                result["is_error"] = True
                failed = True
            tool_results.append(result)

        return tool_results, failed, ran_any
