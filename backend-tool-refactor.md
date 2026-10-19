Refactor @backend/ai_generator.py to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds.

Current behavior:
- Claude makes 1 tool call → tools are removed from API params → final response
- If Claude wants another tool call after seeing results, it can't (gets empty response)

Desired behavior:
- Each tool call should be a separate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when information from different courses/lessons is needed

Example flow:
1. User: "Search for a course that discusses the same topic as lesson 4 of course X"
2. Claude: get course outline for course X → gets title of lesson 4
3. Claude: uses the title to search for a course that discusses the same topic → returns course information
4. Claude: provides complete answer

Requirements:
- Maximum 2 sequential rounds per user query
- Terminate when: (a) 2 rounds completed, (b) Claude's response has no tool_use blocks, or (c) tool call fails
- Preserve conversation context between rounds
- Handle tool execution errors gracefully
- Define what counts as a failed tool call: `ToolManager.execute_tool` no longer raises; it returns error strings ("Error executing tool ...", "Tool '...' not found"). On a failure, still make one final call without tools so the model answers instead of returning an empty response
- Accumulate sources across rounds: today each tool overwrites its own `last_sources`, and `ToolManager.get_last_sources` returns only the first tool's, so a second search or an outline + search flow loses sources

Notes: 
- The model in the app may be a local one (currently gemma4 via Ollama, with thinking disabled), not Claude: the design must not rely on Claude-level tool chaining
- Keep requests within Ollama's context window: the server uses `OLLAMA_CONTEXT_LENGTH` / `num_ctx`, which can be far below the model's 131072 tokens, and silently drops the start of the conversation when it is exceeded. Two rounds of search results plus system prompt and history add up, so check the configured value and keep tool results compact
- Don't rely on `tool_choice` values other than `auto` (e.g. `any`, `none`): Ollama's Anthropic-compatible endpoint may ignore them. To stop tool use in the last round, omit `tools` from the request, as the current code does
- Add a `@pytest.mark.live` test in @backend/tests/test_rag_system.py with a two-step question (e.g. the example flow above) that checks the configured model actually makes two sequential tool calls
- Update the system prompt in @backend/ai_generator.py 
- Update the test @backend/tests/test_ai_generator.py
- Write tests that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details. 

Use two parallel subagents to brainstorm possible plans. Do not implement any code.
