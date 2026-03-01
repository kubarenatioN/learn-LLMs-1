/**
 * LESSON 2 (Phase 3) — Vector Stores
 *
 * Moving from manual cosine similarity to proper vector stores.
 * LangChain abstracts embedding + storage + search into one clean API.
 */

import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import 'dotenv/config'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'

// --- LangChain Embeddings Wrapper ---
// Instead of calling hf.featureExtraction() manually,
// LangChain wraps it in an object that vector stores know how to use.

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HF_TOKEN,
  provider: 'hf-inference',
  model: 'ibm-granite/granite-embedding-small-english-r2',
})

// Quick test — same as hf.featureExtraction(), but through LangChain's interface
async function testEmbeddings() {
  console.log('=== LangChain Embeddings Wrapper ===\n')

  // embedDocuments: embed multiple texts at once (for storing)
  const docVectors = await embeddings.embedDocuments(['JavaScript runs on the server with Node.js', 'Paris is the capital of France'])
  console.log('Embedded 2 documents')
  console.log('Vector dimensions:', docVectors[0].length)

  // embedQuery: embed a single query (for searching)
  const queryVector = await embeddings.embedQuery('server-side JavaScript')
  console.log('Query vector dimensions:', queryVector.length)
  console.log('First 5 values:', queryVector.slice(0, 5))
}

// --- MemoryVectorStore: Add Documents & Search ---

async function basicVectorStore() {
  console.log('=== MemoryVectorStore ===\n')

  // Same documents from Lesson 1
  const documents = [
    'JavaScript was created by Brendan Eich in 1995 at Netscape.',
    'Python is a popular language for data science and machine learning.',
    'The Eiffel Tower is located in Paris, France.',
    'Node.js allows running JavaScript on the server side.',
    'React is a library for building user interfaces in JavaScript.',
    'The Great Wall of China is over 13,000 miles long.',
  ]

  // Create the store and add documents in one step.
  // fromTexts() does three things:
  //   1. Calls embeddings.embedDocuments() on all texts
  //   2. Stores the vectors + original text
  //   3. Returns a ready-to-query store
  const store = await MemoryVectorStore.fromTexts(
    documents,
    documents.map((_, i) => ({ id: i })), // metadata for each doc (we'll explore this later)
    embeddings,
  )

  console.log('Stored %d documents\n', documents.length)

  // --- Similarity search ---
  // Compare this to Lesson 1: no manual embed → compare → sort loop!
  const query = 'server-side JavaScript'
  const results = await store.similaritySearchWithScore(query, 3) // k=3: top 3 results

  console.log(`Query: "${query}" (top 3)\n`)
  for (const [doc, score] of results) {
    const bar = '█'.repeat(Math.round(score * 30))
    console.log(`  ${score.toFixed(3)} ${bar} ${doc.pageContent}`)
  }
}

// --- Metadata & Filtering ---

async function metadataAndFiltering() {
  console.log('=== Metadata & Filtering ===\n')

  // In real apps, documents come from files, URLs, databases.
  // Metadata tracks the source so you can tell the user WHERE the answer came from.
  const store = await MemoryVectorStore.fromTexts(
    [
      'JavaScript was created by Brendan Eich in 1995.',
      'Node.js uses the V8 engine under the hood.',
      'Express is the most popular Node.js web framework.',
      'Python was created by Guido van Rossum in 1991.',
      'Django is a popular Python web framework.',
      'Flask is a lightweight Python web framework.',
    ],
    [
      { source: 'history.md', language: 'javascript' },
      { source: 'runtime.md', language: 'javascript' },
      { source: 'frameworks.md', language: 'javascript' },
      { source: 'history.md', language: 'python' },
      { source: 'frameworks.md', language: 'python' },
      { source: 'frameworks.md', language: 'python' },
    ],
    embeddings,
  )

  // Search WITHOUT filter — returns best matches regardless of language
  const query = 'web framework'
  console.log(`Query: "${query}" — no filter (top 4)\n`)
  const allResults = await store.similaritySearchWithScore(query, 4)
  for (const [doc, score] of allResults) {
    console.log(`  ${score.toFixed(3)}  [${doc.metadata.language}] ${doc.pageContent}`)
  }

  // Search WITH filter — only Python documents
  console.log(`\nQuery: "${query}" — filter: python only (top 4)\n`)
  const pythonOnly = await store.similaritySearchWithScore(
    query,
    4,
    (doc) => doc.metadata.language === 'python',
  )
  for (const [doc, score] of pythonOnly) {
    console.log(`  ${score.toFixed(3)}  [${doc.metadata.language}] ${doc.pageContent}`)
  }
}

async function main() {
  await basicVectorStore()
  console.log('\n')
  await metadataAndFiltering()
}

main().catch(console.error)
