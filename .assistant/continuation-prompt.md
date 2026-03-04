# Continuation Prompt — Paste This to Start a New Chat

---

**Paste everything below this line into a new chat:**

---

I'm studying LLMs using Node.js, HuggingFace ecosystem, LangChain, Ollama, with the goal of building agentic AI applications.

## Context files to read first

Before responding, read these files to understand the full context:

1. `.assistant/study-summary-1.md` — complete history of what I've learned, project setup, technical notes, what's next
2. `study-notes/phase-4.md` — current phase study notes
3. `course-plan.md` — the full course plan

## Teaching style

- I'm an experienced JS/Node.js developer with zero hands-on AI experience
- Give me step-by-step lessons with code split into semantic blocks (not full scripts at once)
- Each block: brief theory → code → I run it → confirm → next block
- Don't start the next topic until I say so
- Comment out previous block function calls in `main()`, leave only the current one active
- Update `study-notes/phase-N.md` after each lesson with concise notes
- Update `.assistant/study-summary-1.md` after each lesson
- Always verify import paths against actual `node_modules` exports before writing code (LangChain moves things between packages frequently)
- Use `provider: 'hf-inference'` for embedding models to avoid "defaulting to auto" noise
- When in doubt about LangChain APIs, check the actual installed package exports — don't trust outdated knowledge

## Where we are

**Phase 4: Agents & Tools — Lesson 3 is next.**

Lessons 1-2 are done:
- Lesson 1: Tools (`tool()` from `@langchain/core/tools`), `bindTools()`, the full tool-calling protocol (LLM requests → code executes → ToolMessage back → LLM answers)
- Lesson 2: Manual agent loop (while + tool_calls), legacy AgentExecutor (deprecated), modern `createAgent` from `"langchain"` with messages-based I/O

**Next up: Phase 4, Lesson 3 — Custom Tools** (building real, useful tools beyond fake data). Then Lessons 4-5 (agent memory & planning, error handling & guardrails), then Phase 5 (real applications).

Let's continue with Lesson 3!
