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

---

### Lesson 2 — Structured Output

**Zod schemas** — define the shape of data you want from the model
- `z.object({ field: z.string().describe('hint for the model') })`
- Supported types: `z.string()`, `z.number()`, `z.boolean()`, `z.array()`, `z.enum([...])`
- `.describe()` on each field tells the model what to put there

**`.withStructuredOutput(schema)`** — enforced JSON responses
- Returns a new model instance that always outputs a JS object matching the schema
- No `JSON.parse()`, no `try/catch` — parsing is handled automatically
- Schema is sent to the API via `response_format` parameter (enforced at API level)
- Works in chains: `prompt.pipe(model.withStructuredOutput(schema))`

**Strictness depends on provider** — some providers may return extra fields not in your schema

---

### Component Input/Output Reference

Every component implements the `Runnable` interface (has `.invoke()`).
Rule: **output of step N must be valid input for step N+1.**

| Component | Input | Output |
|---|---|---|
| `ChatPromptTemplate` | `{ key: value }` object | `ChatPromptValue` (messages array) |
| `ChatModel` | `string` or `messages[]` | `AIMessage` |
| `ChatModel.withStructuredOutput()` | `string` or `messages[]` | plain JS object |
| `StringOutputParser` | `AIMessage` | `string` |
| `RunnableLambda` | anything | anything you return |

**Common chain patterns:**
```
PromptTemplate → ChatModel → StringOutputParser          (text out)
PromptTemplate → ChatModel.withStructuredOutput()        (object out)
```

**When unsure about a component's output** — break the chain, log the result:
```js
const result = await someComponent.invoke(input)
console.log(result)
console.log(result.constructor.name)
```

**`RunnableLambda`** — escape hatch to transform data between incompatible steps:
```js
import { RunnableLambda } from '@langchain/core/runnables'
const transform = new RunnableLambda({ func: (input) => transformedOutput })
```

---

### Lesson 3 — Chains in Depth

**Sequential chains** — output of one chain becomes input to the next
- Chain 1 produces a string, but chain 2's template expects `{ key: value }`
- Use `RunnableLambda` as an adapter to reshape data between steps:
  ```js
  chain1
    .pipe(new RunnableLambda({ func: (text) => ({ key: text }) }))
    .pipe(chain2)
  ```

**`RunnableSequence.from([...])`** — compose multiple steps as an array
- Same as chaining `.pipe()` calls, but cleaner for many steps
- Each element is a Runnable (chain, lambda, model, etc.)
  ```js
  const pipeline = RunnableSequence.from([step1, step2, step3])
  ```

**Accumulating state pattern** — carry data forward across steps
- Each step spreads previous state and adds new data: `{ ...input, newField }`
- Later steps have access to everything produced by earlier steps:
  ```
  Step 1: { topic }           → adds title  → { topic, title }
  Step 2: { topic, title }    → adds outline → { topic, title, outline }
  Step 3: { topic, title, outline } → adds intro → { topic, title, outline, intro }
  ```

**Key takeaway:** `.pipe()`, `RunnableLambda`, and `RunnableSequence` are the three building blocks for any multi-step LLM workflow in LangChain.

---

### Lesson 4 — Memory

**The core truth:** LLMs are stateless. "Memory" = sending the entire conversation history with every request.

**Message classes** — `HumanMessage`, `AIMessage`, `SystemMessage`
- Construct messages explicitly: `new HumanMessage('text')`
- The model reads all messages and responds in context
- `msg._getType()` returns `'human'`, `'ai'`, or `'system'`

**Manual memory** — maintain a messages array yourself
- Push `HumanMessage` and `AIMessage` after each turn
- Pass the full array to `model.invoke(history)`
- Simple but doesn't scale — history grows forever, tokens overflow, cost increases

**`InMemoryChatMessageHistory`** — LangChain's message store
- `.addMessage(msg)` to store, `.getMessages()` to retrieve
- Abstraction over the array — can be swapped for a persistent store (Redis, DB) later

**`trimMessages`** — window memory strategy
- Trims history to fit a token budget, keeping only recent messages
- Key options:
  - `maxTokens` — token budget for the trimmed history
  - `strategy: 'last'` — keep the most recent messages
  - `startOn: 'human'` — always start the window on a human message
  - `includeSystem: true` — always keep the system message regardless of trimming
- `tokenCounter` — function to estimate token count (rough: `content.length / 4`)

**Three memory strategies:**
| Strategy | How it works | Tradeoff |
|---|---|---|
| Full history | Send everything | Simple, but overflows on long conversations |
| Window memory (`trimMessages`) | Keep last N messages | Good balance, loses old context |
| Summary memory | LLM summarizes old turns into a paragraph | Saves tokens, loses detail |
