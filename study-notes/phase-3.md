# Phase 3: Retrieval-Augmented Generation (RAG)

---

### Lesson 1 — Embeddings & Semantic Search

**What is RAG?** — Retrieve relevant documents → inject into prompt → LLM answers based on YOUR data
- Alternative to fine-tuning: cheaper, faster, easy to update
- Flow: user query → retrieve relevant chunks → augment prompt with context → generate answer

**Embeddings** — vector representations of text meaning
- An embedding model converts text into an array of numbers (e.g. 384 dimensions)
- Similar meanings → similar vectors, regardless of exact words used
- Used `ibm-granite/granite-embedding-small-english-r2` via `hf.featureExtraction()`
- HuggingFace's `/v1` router only supports chat completions, NOT embeddings — use `@huggingface/inference` SDK instead
- Set `provider: 'hf-inference'` explicitly to avoid "defaulting to auto" log noise

**Cosine similarity** — measures how similar two vectors are
- Returns -1 to 1 (closer to 1 = more similar meaning)
- Standard math: dot product / (magnitude A × magnitude B)
- In practice, vector store libraries handle this for you

**Semantic search flow (manual):**
1. Embed all documents → get vectors
2. Embed the query → get query vector
3. Compare query vector to all document vectors (cosine similarity)
4. Rank by score → top results are most relevant

**Why semantic beats keyword search:**
- "JavaScript backend" matches "JavaScript server side" (different words, same meaning)
- Unrelated topics (geography) score low regardless of shared words
