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

---

### Lesson 4 — Agent Memory

**The stateless problem:** each `agent.invoke()` starts fresh — no knowledge of previous interactions. Follow-up questions like "the previous city" cause hallucinations (the agent invents data rather than admitting it doesn't know).

**`MemorySaver` from `@langchain/langgraph`:**
- In-memory checkpointer — stores full conversation state (messages + tool call history)
- Pass to `createAgent({ ..., checkpointer: memory })`
- Automatic — you send only the new message per `.invoke()`, the checkpointer handles history accumulation

**Thread-based conversations via `thread_id`:**
- Pass `{ configurable: { thread_id: 'some-id' } }` as the second argument to `.invoke()`
- Same `thread_id` = continued conversation (agent remembers everything)
- Different `thread_id` = fresh, isolated conversation
- One agent instance can serve many parallel conversations — each thread is independent
- Core pattern for multi-user apps: one agent, many threads, no state leakage

**MemorySaver is in-memory only** — data is lost when the process exits. For persistence across restarts, you'd use database-backed checkpointers (Postgres, SQLite, Redis). Same API, different storage backend.

---

### Lesson 5 — Error Handling & Guardrails

**Default error behavior in `createAgent`:**
- Modern `createAgent` (LangGraph-based) catches tool exceptions internally by default
- Error message is passed back to the LLM as a tool result — LLM can recover gracefully
- Without retries, the LLM falls back to training data (may hallucinate)

**`toolRetryMiddleware` — automatic retries with backoff:**
- `import { toolRetryMiddleware } from "langchain"` — pass in `middleware` array of `createAgent`
- `maxRetries` — how many additional attempts after first failure
- `initialDelayMs` / `backoffFactor` — exponential backoff (100ms → 200ms → 400ms)
- `onFailure: 'continue'` — if all retries exhausted, pass error to LLM (graceful). `'error'` throws instead.
- Transparent to the LLM — it never sees the retries, just gets the successful result

**`modelCallLimitMiddleware` — preventing infinite loops:**
- `runLimit` — max LLM calls per single `.invoke()` (prevents loop within one task)
- `threadLimit` — max LLM calls across entire conversation (cost budget)
- `exitBehavior: 'end'` — gracefully stops and returns current state. `'error'` throws.
- Essential for agents with tools that can trick the LLM into looping ("call me again")

**`toolCallLimitMiddleware` — also available:**
- Limits how many times a specific tool (or all tools) can be called
- Per-run and per-thread limits, same pattern as model call limit

**System prompt guardrails — behavioral constraints:**
- Explicit rules in the system prompt: what to answer, what to decline, how to handle missing data
- "Do NOT make up data" prevents hallucination when tools return no results
- Topic restrictions ("ONLY answer weather questions") keep the agent focused
- System prompt is the first line of defense — middleware handles mechanical failures, system prompt handles behavioral ones

**Layers of defense (inner to outer):**
1. **Zod schemas** — validate tool inputs at the schema level
2. **Tool-level error handling** — return error strings, not exceptions
3. **Middleware** — retries, call limits, cost budgets
4. **System prompt** — behavioral rules, topic restrictions, honesty constraints
5. **Application-level** — try/catch around `.invoke()`, timeouts, logging
