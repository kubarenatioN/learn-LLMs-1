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

**Legacy: `createToolCallingAgent` + `AgentExecutor` from `@langchain/classic/agents` (DEPRECATED):**
- Prompt requires `{input}` and `MessagesPlaceholder('agent_scratchpad')`
- `agent_scratchpad` = accumulated intermediate steps (tool calls + results) — same as our manual messages array
- `createToolCallingAgent({ llm, tools, prompt })` — creates the agent brain
- `AgentExecutor({ agent, tools, maxIterations })` — runs the loop automatically
- Single `.invoke({ input })` → returns `{ output }` with the final answer
- `verbose: true` shows full reasoning (extremely detailed); `verbose: false` for clean output
- Deprecated since LangChain v1.0 (October 2025) — still works from `@langchain/classic` but not maintained

**Modern: `createAgent` from `"langchain"` (CURRENT):**
- `import { createAgent } from "langchain"` — the new single entry point
- Replaces both `AgentExecutor` and `createReactAgent` from `@langchain/langgraph/prebuilt`
- `createAgent({ model, tools, systemPrompt })` — much simpler setup
- `model` — accepts a `ChatModel` instance OR a string like `"openai:gpt-4o"` for auto-init
- `systemPrompt` — just a plain string (no prompt template, no `{input}`, no `{agent_scratchpad}`)
- Messages-based I/O: `agent.invoke({ messages: [{ role: 'user', content: query }] })`
- Output: `result.messages` array — last message is the final answer
- Messages count reveals steps: 4 messages = tool was used (Human → AI tool_call → Tool result → AI answer), 2 messages = direct answer (Human → AI answer)

**LangChain vs LangGraph — they're layers, not competitors:**
- LangChain = foundation (LLM wrappers, prompts, parsers, tools, chains, RAG components)
- LangGraph = orchestration on top (agent loops as graphs, state management, memory, human-in-the-loop)
- `createAgent` from `"langchain"` wraps LangGraph internally — you get both

**Agent behaviors observed:**
- LLM can request multiple tools in parallel in one step
- LLM may skip tools when it can answer from its own knowledge (e.g., simple math, philosophical questions)
- LLM can recover from tool errors by falling back to its training data

**Tool description engineering:**
- Tool description is the LLM's only instruction manual
- Specifying "expression MUST be formatted for JavaScript" fixed `arccos` → `Math.acos` errors
- The `.describe()` on Zod fields also helps the LLM generate correct arguments

---

### Lesson 3 — Custom Tools (Real Tools)

**From fake to real:** Lessons 1-2 used hardcoded data. Real agents need tools that interact with the outside world — network, filesystem, APIs.

**Web Fetcher tool — design decisions:**
- `fetch()` with `AbortSignal.timeout(10_000)` — always set a timeout; network requests can hang forever
- Strip HTML (scripts, styles, tags) → give the LLM clean text, not raw markup
- Truncate output — web pages are 50-100k chars; LLM context is finite. Trade-off: more chars = better analysis, fewer chars = fits more tool results
- Return error strings, never throw — `"Fetch failed: ..."` lets the LLM reason about the failure and adapt
- `.url()` Zod validator — catches malformed URLs before making the request

**File Reader tool — guardrails pattern:**
- **Allowlist a base directory** — `path.resolve(ALLOWED_DIR, filename)`, then verify the resolved path starts with `ALLOWED_DIR`
- **Path traversal protection** — `../../.env` resolves to a path outside the allowed dir → blocked
- **File size limit** — `stat.size` check prevents sending massive files to the LLM
- **Hardcoded file list in description** — the LLM only sees tool descriptions, so listing available files there helps it know what to request. In production, use a separate `list_files` tool.
- **Graceful errors** — `ENOENT` → `"File not found"`, not a crash. The agent receives the error text and reports it to the user.

**Multi-tool agent — combining tools:**
- Pass both tools to `createAgent({ model, tools: [...], systemPrompt })`
- The agent reads all tool descriptions and picks the right one per query
- Agent gracefully handles tool errors (e.g., file not found) — reports the issue to the user instead of crashing

**Key design principles for custom tools:**
1. **Return strings, not exceptions** — the LLM can reason about error text
2. **Validate inputs with Zod** — catch bad args before they cause runtime errors
3. **Constrain scope** — sandbox file access, set timeouts, truncate output
4. **Descriptive metadata** — the description is the LLM's only documentation for your tool
