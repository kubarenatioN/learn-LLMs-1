# Study Course: Building AI Applications with Node.js

Each lesson follows this pattern:
1. **Brief theory** — what are we doing and why
2. **Code-along** — write the code together, every piece explained
3. **Experiment** — modify params, try different models, break things on purpose
4. **Mini-project** — apply what you learned

---

## Phase 1: Foundations — Talking to LLMs from Node.js

**Goal:** Get comfortable making API calls to LLMs, understand the request/response cycle, tokens, streaming, and basic prompt engineering in practice.

1. Setting up the environment — Node.js project, env variables, API keys
2. Raw HTTP calls to HuggingFace Inference API — no libraries, just `fetch()`. Understand what actually goes over the wire
3. HuggingFace.js SDK (`@huggingface/inference`) — the official JS client, text generation, text-to-image, summarization, etc.
4. Understanding models — what models are on HuggingFace, how to pick one, what "task" means (text-generation, text2text-generation, conversational, etc.)
5. Streaming responses — server-sent events, token-by-token output
6. Prompt engineering basics in practice — system prompts, few-shot examples, temperature/top_p tuning

**Mini-projects:**
- CLI chatbot using HuggingFace Inference API
- Text summarizer script
- Simple prompt tester that compares outputs from different models

---

## Phase 2: LangChain.js Core Concepts

**Goal:** Understand what LangChain gives you on top of raw API calls and when it actually makes sense to use it.

7. LangChain.js setup & architecture overview — models, chains, prompts, output parsers
8. Chat models & LLM wrappers — connecting LangChain to HuggingFace (and optionally OpenAI/Ollama for comparison)
9. Prompt templates — dynamic prompts, partial variables, composing prompts
10. Output parsers — structured output, JSON parsing, comma-separated lists
11. Chains — sequential chains, stuff chain, map-reduce chain. The core idea of "chaining" steps
12. Memory — conversation buffer, summary memory, window memory. Making chatbots that remember context

**Mini-projects:**
- Chatbot with memory using LangChain
- Multi-step content generator (e.g., generate outline → expand each section → summarize)

---

## Phase 3: Retrieval-Augmented Generation (RAG)

**Goal:** Build apps that can answer questions based on your own documents/data — the most common real-world pattern.

13. Embeddings — what they are, HuggingFace embedding models, vector representations
14. Vector stores — in-memory stores, connecting to Chroma/FAISS/Pinecone from Node.js
15. Document loaders — loading PDFs, text files, web pages, markdown
16. Text splitters — chunking strategies, why chunk size matters
17. Retrieval chains — combining retriever + LLM in LangChain
18. Conversational RAG — RAG with memory, follow-up questions

**Mini-projects:**
- "Chat with your docs" app — load markdown/PDF files, ask questions
- Knowledge base Q&A bot

---

## Phase 4: Agents & Tools

**Goal:** Build LLM-powered agents that can use external tools, make decisions, and take actions.

19. Tools concept — giving the LLM abilities (web search, calculator, custom functions)
20. LangChain agents — ReAct pattern, tool-calling agents
21. Custom tools — writing your own tools in JS
22. Agent memory & planning — how agents maintain state across steps
23. Error handling & guardrails — what happens when the LLM goes off-rails

**Mini-projects:**
- Research assistant agent that can search the web and summarize findings
- Task automation agent with custom tools

---

## Phase 5: Building Real Applications

**Goal:** Put it all together into actual apps with proper architecture.

24. API server with Express/Fastify — serving LLM features over HTTP
25. Streaming responses to the frontend — SSE or WebSocket based streaming
26. Rate limiting, caching, error handling — production concerns
27. Cost management — token counting, caching strategies, model selection
28. Evaluation — how to test if your LLM app actually works well

**Final projects:**
- Full-stack AI chat application with RAG
- AI-powered content management tool
- Multi-agent system for a specific domain

---

## Prerequisites Checklist

- [ ] Node.js 18+ installed
- [ ] HuggingFace account + API token (free tier is fine to start)
- [ ] Code editor (Cursor)
- [ ] Optionally: OpenAI API key (useful for comparison, but not required)
