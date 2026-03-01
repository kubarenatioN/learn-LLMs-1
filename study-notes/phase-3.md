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

---

### Lesson 2 — Vector Stores

**Why vector stores?** — manual cosine similarity doesn't scale. Vector stores handle embedding storage, indexing, and fast similarity search in one place.

**LangChain embeddings wrapper** — `HuggingFaceInferenceEmbeddings` from `@langchain/community/embeddings/hf`
- Wraps `hf.featureExtraction()` in LangChain's interface (`embedDocuments()`, `embedQuery()`)
- Required because vector stores expect an embeddings object, not a raw function
- Set `provider: 'hf-inference'` to avoid "defaulting to auto" log noise

**MemoryVectorStore** — in-memory vector store from `@langchain/classic/vectorstores/memory`
- Import path gotcha: `langchain/vectorstores/memory` no longer exists in newer versions — moved to `@langchain/classic`
- `ERR_PACKAGE_PATH_NOT_EXPORTED` = the package doesn't export that subpath anymore
- `fromTexts(texts, metadatas, embeddings)` — embed + store in one step
- `similaritySearchWithScore(query, k)` — returns `[Document, score][]` sorted by relevance
- Replaces the entire manual embed → cosine similarity → sort loop from Lesson 1

**The Document object** — `{ pageContent: string, metadata: object }`
- Core data structure in LangChain — every piece of text is a Document
- `fromTexts()` creates Documents behind the scenes; `new Document({...})` creates them explicitly
- `addDocuments()` — add to an existing store incrementally (e.g. multiple batches, different sources)

**Metadata & filtering**
- Each document can carry metadata: `{ source, language, section, ... }`
- Tracks WHERE content came from — essential for citations and scoping
- Filter function: `similaritySearchWithScore(query, k, (doc) => doc.metadata.language === 'python')`
- Real-world use: filter by file, version, user permissions, date, etc.

**asRetriever(k)** — converts a vector store into a Retriever
- Retriever has one method: `invoke(query)` → `Document[]` (no scores)
- Simpler interface = pluggable into LangChain chains
- `similaritySearchWithScore()` → for exploration and debugging (shows scores)
- `asRetriever().invoke()` → for plugging into RAG pipelines (Lesson 5)
