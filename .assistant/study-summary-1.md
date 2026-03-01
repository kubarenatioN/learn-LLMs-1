# Study Summary — Context for Continuation

## Who is the student

- Experienced JavaScript/Node.js developer, zero hands-on AI experience, knows some theory
- Learning to build AI applications with Node.js + HuggingFace + LangChain
- Prefers step-by-step lessons with code split into semantic blocks, not full scripts at once
- Doesn't want the next topic started until they say so
- Wants study notes updated after each lesson in `study-notes/phase-N.md`

## Project setup

- Workspace: Node.js project with `"type": "module"` (ES imports)
- Dependencies: `dotenv`, `@huggingface/inference`, `langchain`, `@langchain/core`, `@langchain/community`, `@langchain/openai`, `zod`
- Auth: HuggingFace token in `.env` as `HF_TOKEN`
- Structure: `phase-1/`, `phase-2/`, `phase-3/` folders with lesson files, `study-notes/` folder with per-phase markdown
- Always comment "passed" lesson blocks inside main() function and leave only the current studied function calls

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

### Phase 3: RAG (1 lesson done, in progress)

- **Lesson 1:** Embeddings via `hf.featureExtraction()` with `ibm-granite/granite-embedding-small-english-r2` and `provider: 'hf-inference'`. Manual cosine similarity. Semantic search ranking documents by relevance. HuggingFace `/v1` router doesn't support `/v1/embeddings` — only chat completions.

## What's next

- Phase 3, Lesson 2: Vector stores (in-memory, then persistent)
- Phase 3, Lessons 3-4: Document loaders, text splitters, retrieval chains, conversational RAG
- Phase 4: Agents & Tools
- Phase 5: Building real applications

## Key technical notes

- HuggingFace API moved from `api-inference.huggingface.co` (deprecated, returns 410) to `router.huggingface.co`
- The `/v1` OpenAI-compatible router is **chat completions only** — no embeddings endpoint
- For embeddings, use `@huggingface/inference` SDK (`featureExtraction` method)
- Default chat model used: `Qwen/Qwen2.5-7B-Instruct`
- Default embedding model: `ibm-granite/granite-embedding-small-english-r2` (384 dimensions)
- `--legacy-peer-deps` needed for npm installs due to dotenv v17 vs peer dep conflicts
