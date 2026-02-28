# Phase 1: Foundations — Talking to LLMs from Node.js

---

### Lesson 1 — Raw fetch() to HuggingFace

- Talking to an LLM is just a **POST request** with JSON body
- HuggingFace Inference API endpoint: `https://router.huggingface.co/v1/chat/completions`
- Request body follows the **OpenAI chat completions format** (same format works across many providers):
  ```
  { model, messages: [{role, content}], max_tokens, temperature }
  ```
- **Roles**: `system` (instructions), `user` (human input), `assistant` (model's previous replies)
- **Temperature**: 0.1 = focused/deterministic, 1.0 = creative/random, 0.7 = common default
- **max_tokens**: caps response length. ~1 token ≈ 0.75 words
- Response: `choices[0].message.content` has the generated text, `usage` has token counts

---

### Lesson 2 — HuggingFace.js SDK (`@huggingface/inference`)

- `InferenceClient(token)` — handles auth, routing, endpoints automatically
- **`client.chatCompletion()`** — same as raw fetch but cleaner. Same input/output shape
- **`client.chatCompletionStream()`** — returns an async iterable, use `for await` to get tokens one by one
  - Each chunk has `choices[0].delta.content` (not `message` — it's a delta/fragment)
  - Use `process.stdout.write()` instead of `console.log()` to avoid newlines between tokens
- **`client.summarization()`** — dedicated task method, different input/output than chat
  - Input: `{ model, inputs: "text" }` → Output: `{ summary_text: "..." }`
- **Cold starts**: free-tier models may take 20-60s to load on first call. SDK retries automatically

---

### Lesson 3 — Understanding Models

- Model naming: `{organization}/{model-name}` (e.g. `Qwen/Qwen2.5-7B-Instruct`)
- Common name patterns:
  - **7B, 13B, 72B** — parameter count in billions. Bigger = smarter but slower/costlier
  - **Instruct** — fine-tuned to follow instructions (vs "base" models that just complete text)
  - **Chat** — fine-tuned for conversation
- **Tasks**: HuggingFace categorizes models by what they do:
  - `text-generation` — chat, text completion
  - `summarization` — compress text
  - `translation` — language translation
  - `text-classification` — categorize text (sentiment, etc.)
  - `feature-extraction` — turn text into vector embeddings (used in RAG)
- **Provider** ≠ **Model**
  - Model = *what* runs (the recipe)
  - Provider = *where* it runs (the kitchen: hf-inference, Together AI, Groq, etc.)
  - HuggingFace Router picks a provider automatically. Same model can run on different providers
