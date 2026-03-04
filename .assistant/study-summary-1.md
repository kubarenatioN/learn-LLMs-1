# Study Summary — Context for Continuation

## Who is the student

- Experienced JavaScript/Node.js developer, zero hands-on AI experience, knows some theory
- Learning to build AI applications with Node.js + HuggingFace + LangChain
- Prefers step-by-step lessons with code split into semantic blocks, not full scripts at once
- Doesn't want the next topic started until they say so
- Wants study notes updated after each lesson in `study-notes/phase-N.md`

## Project setup

- Workspace: Node.js project with `"type": "module"` (ES imports)
- Dependencies: `dotenv`, `@huggingface/inference`, `langchain`, `@langchain/core`, `@langchain/community`, `@langchain/openai`, `@langchain/langgraph`, `zod`
- Auth: HuggingFace token in `.env` as `HF_TOKEN`
- Structure: `phase-1/`, `phase-2/`, `phase-3/` folders with lesson files, `study-notes/` folder with per-phase markdown
- Always comment "passed" lesson blocks inside main() function and leave only the current studied function calls
- Use hf-inference as model provider explicitly

## Course plan

Full plan in `course-plan.md`. Five phases: Foundations → LangChain Core → RAG → Agents & Tools → Real Applications.

## What's been completed

### Phase 1: Foundations (3 lessons)

- **Lesson 1:** Raw `fetch()` to HuggingFace Inference API. OpenAI-compatible chat completions format at `https://router.huggingface.co/v1/chat/completions`. Messages array with roles (system/user/assistant), temperature, max_tokens.
- **Lesson 2:** `@huggingface/inference` SDK — `InferenceClient`, `chatCompletion()`, `chatCompletionStream()` (streaming with `for await`), `summarization()` with `facebook/bart-large-cnn`. Cold starts on free tier.
- **Lesson 3:** Model naming conventions (org/model, param counts, Instruct/Chat variants), HuggingFace tasks, provider vs model distinction (model = what runs, provider = where it runs, router picks automatically).

### Phase 2: LangChain Core (4 lessons)

- **Lesson 1:** `ChatOpenAI` wrapper pointed at HuggingFace via `configuration.baseURL`. `ChatPromptTemplate` with `{variables}`. `.pipe()` chaining. `StringOutputParser`. Core mental model: `input → component.pipe(component) → output`.
- **Lesson 2:** Structured output with Zod schemas + `.withStructuredOutput()`. Enforced JSON responses at API level. Works in chains.
- **Lesson 3:** Sequential chains, `RunnableLambda` as adapter between incompatible steps, `RunnableSequence.from([])`, accumulating state pattern (`{ ...input, newField }`).
- **Lesson 4:** Memory — LLMs are stateless, "memory" = sending full conversation history. Manual array approach, `InMemoryChatMessageHistory`, `trimMessages` for window memory (keep recent, drop old). Three strategies: full history, window, summary.

### Phase 3: RAG (5 lessons, COMPLETE)

- **Lesson 1:** Embeddings via `hf.featureExtraction()` with `ibm-granite/granite-embedding-small-english-r2` and `provider: 'hf-inference'`. Manual cosine similarity. Semantic search ranking documents by relevance. HuggingFace `/v1` router doesn't support `/v1/embeddings` — only chat completions.
- **Lesson 2:** Vector stores. `HuggingFaceInferenceEmbeddings` as LangChain wrapper (`embedDocuments`, `embedQuery`). `MemoryVectorStore` from `@langchain/classic/vectorstores/memory` (moved from `langchain/`). `fromTexts()` for one-step create+store. `similaritySearchWithScore(query, k)` replaces manual cosine loop. `Document` object (`pageContent` + `metadata`). Incremental `addDocuments()`. Metadata filtering with function predicate. `asRetriever(k)` converts store to chain-friendly retriever (`invoke(query) → Document[]`). Student extracted `asRetriever` demo into separate file `2-as-retriever.js`.
- **Lesson 3:** Document loaders & text splitters. `TextLoader` from `@langchain/classic/document_loaders/fs/text` reads files into Documents. `RecursiveCharacterTextSplitter` from `@langchain/textsplitters` splits on `\n\n` → `\n` → ` ` → chars recursively. Key params: `chunkSize` (max chars), `chunkOverlap` (shared chars at boundaries). Sweet spot 300-800 chars, overlap 10-20%. Metadata preserved + `loc.lines` added. Full pipeline: load → split → `fromDocuments()` → search. Sample docs in `phase-3/sample-docs/`.
- **Lesson 4:** Retrieval chains. Manual RAG: retrieve → format context → inject into prompt → LLM answers. RAG prompt pattern: "Answer ONLY based on context" prevents hallucination (tested: LLM refused "capital of Japan"). Chain-based RAG with `RunnablePassthrough.assign()` — single `.invoke({ question })`. RAG with sources: chained `.assign()` keeps `docs` in pipeline alongside `answer`, enabling source citations with file + line numbers. `buildVectorStore()` loads multiple files into one store.
- **Lesson 5:** Conversational RAG. Problem: follow-ups fail without memory ("What types are there?" has no context). Solution: question rephrasing — LLM rewrites follow-ups into standalone questions using `MessagesPlaceholder('history')`. Full flow: rephrase → retrieve → answer → update history. Original question stored in history (not rephrased). Topic switches and grounding both work in conversational mode. Student added extra negative test ("How many sugars in coke?" → "I don't have enough information").

### Phase 4: Agents & Tools (4 lessons done, in progress)

- **Lesson 1:** Tools & tool calling. `tool(fn, {name, description, schema})` from `@langchain/core/tools` — Zod schemas for input. `llm.bindTools([...])` sends metadata to LLM. LLM decides per-query: tool call vs direct answer. Full loop: LLM returns `tool_calls` → code executes tools → `ToolMessage({ content, tool_call_id })` → send back to LLM → final answer. LLM can request multiple tools in one turn. Tool calling is a protocol, not execution.
- **Lesson 2:** Agents & ReAct. Manual agent loop: while + tool_calls check + MAX_STEPS. Legacy: `createToolCallingAgent` + `AgentExecutor` from `@langchain/classic/agents` (deprecated since v1.0 Oct 2025). Modern: `createAgent` from `"langchain"` — `createAgent({ model, tools, systemPrompt })`, messages-based I/O. Student noticed AgentExecutor missing from official docs, prompted migration to modern API. LangChain = foundation layer, LangGraph = orchestration layer, `createAgent` wraps both. Tool description engineering: "must be JavaScript" fixed `arccos` → `Math.acos`.
- **Lesson 3:** Custom tools — real tools that interact with the outside world. Web Fetcher: `fetch()` + HTML stripping + truncation + timeout via `AbortSignal.timeout()`. File Reader: sandboxed to allowed directory, `path.resolve()` + startsWith guard blocks path traversal (`../../.env` → denied), `stat.size` prevents huge files. Key design principles: return error strings (not exceptions) so the LLM can reason about failures; validate inputs with Zod; constrain scope (timeouts, size limits, directory sandbox); list available files in tool description since it's the LLM's only documentation. Multi-tool agent with `createAgent` picks the right tool per query and gracefully handles tool errors.
- **Lesson 4:** Agent memory. Stateless agents hallucinate when follow-ups reference prior context ("previous city was 8°C" — invented). Fix: `MemorySaver` from `@langchain/langgraph` — in-memory checkpointer. Pass `checkpointer: memory` to `createAgent`, then `{ configurable: { thread_id } }` as second arg to `.invoke()`. Same thread = continued conversation, different thread = isolated. One agent serves many parallel users via different thread IDs. MemorySaver is in-memory only (lost on restart); database-backed checkpointers exist for persistence.

## What's next

- Phase 4, Lesson 5: Error handling & guardrails
- Phase 5: Building real applications

## Key technical notes

- HuggingFace API moved from `api-inference.huggingface.co` (deprecated, returns 410) to `router.huggingface.co`
- The `/v1` OpenAI-compatible router is **chat completions only** — no embeddings endpoint
- For embeddings, use `@huggingface/inference` SDK (`featureExtraction` method)
- Default chat model used: `Qwen/Qwen2.5-7B-Instruct`
- Default embedding model: `ibm-granite/granite-embedding-small-english-r2` (384 dimensions)
- `--legacy-peer-deps` needed for npm installs due to dotenv v17 vs peer dep conflicts
- `MemoryVectorStore` moved from `langchain/vectorstores/memory` to `@langchain/classic/vectorstores/memory` — `ERR_PACKAGE_PATH_NOT_EXPORTED` means the subpath no longer exists in the package
- `AgentExecutor` + `createToolCallingAgent` deprecated since LangChain v1.0 (Oct 2025). Modern: `import { createAgent } from "langchain"`. Uses `model` (not `llm`), `systemPrompt` (not prompt template), messages-based I/O
- LangChain ecosystem structure: `langchain` (main entry + agents), `@langchain/core` (primitives), `@langchain/openai` (provider), `@langchain/community` (integrations), `@langchain/classic` (legacy/deprecated), `@langchain/langgraph` (graph orchestration)
