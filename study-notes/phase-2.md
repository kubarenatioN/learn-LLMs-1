# Phase 2: LangChain.js Core Concepts

---

### Lesson 1 — Setup, Templates, Chains, Parsers

**Chat Model wrapper** — `ChatOpenAI`
- Connects LangChain to any OpenAI-compatible provider
- Point at HuggingFace by setting `configuration.baseURL` to `https://router.huggingface.co/v1`
- Swap `baseURL` + `apiKey` to talk to OpenAI, Groq, Ollama — rest of code stays the same
- `.invoke("text")` returns an `AIMessage` object (not raw JSON)

**Prompt Templates** — `ChatPromptTemplate`
- Reusable prompts with `{variables}` in curly braces
- `ChatPromptTemplate.fromMessages([['role', 'text with {var}'], ...])`
- `.invoke({ var: value })` fills in variables → returns formatted messages
- Define once, reuse with different inputs

**Piping / Chains** — `.pipe()`
- Connects components so data flows automatically between them
- `prompt.pipe(model)` = template fills variables → feeds into model
- `prompt.pipe(model).pipe(parser)` = three-step chain
- Call `.invoke()` on the chain, not individual parts

**Output Parsers** — `StringOutputParser`
- Extracts clean data from model responses
- `StringOutputParser` converts AIMessage → plain string
- For JSON: ask the model to output JSON in the prompt, then `JSON.parse()` the result
- This is fragile — LLMs don't always follow format instructions (solved properly with structured output)

**The core mental model:**
```
input → [component].pipe([component]).pipe([component]) → output
```
Every LangChain component has `.invoke()`. Every component can be piped. That's the whole framework.
