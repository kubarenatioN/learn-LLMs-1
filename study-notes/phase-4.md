# Phase 4: Agents & Tools

---

### Lesson 1 — Tools & Tool Calling

**The shift from chains to agents:** chains follow predetermined steps; agents use the LLM to decide which tools to call, with what arguments, and in what order.

**Defining tools with `tool()` from `@langchain/core/tools`:**
- `tool(fn, { name, description, schema })` — function + metadata
- `name` — unique identifier the LLM uses to reference the tool
- `description` — the LLM reads this to decide WHEN to use the tool (prompt engineering for tools)
- `schema` — Zod schema defining the input; LLM generates arguments matching this
- Tools are just functions — the LLM doesn't get the code, only the metadata

**`bindTools()` — telling the LLM what's available:**
- `llm.bindTools([tool1, tool2])` — sends tool metadata (name, description, schema) to the LLM
- LLM decides per-query whether to use a tool or answer directly
- "Weather in Tokyo?" → tool call; "Say hello in Japanese" → direct answer

**Tool call response structure:**
- `response.content` — empty string when tool is requested
- `response.tool_calls` — array of `{ name, args, id }`
- `id` is unique per request — used to link results back to the call

**The full tool-calling loop:**
1. LLM receives query + available tools → returns `tool_calls` (may request multiple tools)
2. Your code executes each tool → wraps results in `ToolMessage({ content, tool_call_id })`
3. Send back: `[HumanMessage, AIMessage with tool_calls, ...ToolMessages]`
4. LLM now has tool results → composes final natural language answer

**Key insight:** tool calling is a PROTOCOL, not execution. The LLM says "I want to call X with Y." Your code actually runs it. This separation is what makes it safe and controllable.

---

### Lesson 2 — Agents & the ReAct Pattern

**ReAct = Reasoning + Acting** — an automated loop:
1. LLM reasons about what to do → 2. Calls a tool → 3. Observes result → 4. Repeats until done

**Manual agent loop (built from scratch):**
- `while` loop: invoke LLM → check `tool_calls` → execute tools → add `ToolMessage` to messages → repeat
- Break when LLM returns content with no `tool_calls` (final answer)
- `MAX_STEPS` safety limit prevents infinite loops
- Messages array IS the agent's memory within one task

**`createToolCallingAgent` + `AgentExecutor` from `@langchain/classic/agents`:**
- Prompt requires `{input}` and `MessagesPlaceholder('agent_scratchpad')`
- `agent_scratchpad` = accumulated intermediate steps (tool calls + results) — same as our manual messages array
- `createToolCallingAgent({ llm, tools, prompt })` — creates the agent brain
- `AgentExecutor({ agent, tools, maxIterations })` — runs the loop automatically
- Single `.invoke({ input })` → returns `{ output }` with the final answer
- `verbose: true` shows full reasoning (extremely detailed); `verbose: false` for clean output

**Agent behaviors observed:**
- LLM can request multiple tools in parallel in one step
- LLM may skip tools when it can answer from its own knowledge (e.g., simple math, philosophical questions)
- LLM can recover from tool errors by falling back to its training data

**Tool description engineering:**
- Tool description is the LLM's only instruction manual
- Specifying "expression MUST be formatted for JavaScript" fixed `arccos` → `Math.acos` errors
- The `.describe()` on Zod fields also helps the LLM generate correct arguments
