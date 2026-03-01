/**
 * LESSON 2 (Phase 3) — Vector Stores
 *
 * Moving from manual cosine similarity to proper vector stores.
 * LangChain abstracts embedding + storage + search into one clean API.
 */

import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { Document } from '@langchain/core/documents'
import 'dotenv/config'

// --- LangChain Embeddings Wrapper ---

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HF_TOKEN,
  model: 'ibm-granite/granite-embedding-small-english-r2',
})

// Quick test — same as hf.featureExtraction(), but through LangChain's interface
async function testEmbeddings() {
  console.log('=== LangChain Embeddings Wrapper ===\n')

  const docVectors = await embeddings.embedDocuments(['JavaScript runs on the server with Node.js', 'Paris is the capital of France'])
  console.log('Embedded 2 documents')
  console.log('Vector dimensions:', docVectors[0].length)

  const queryVector = await embeddings.embedQuery('server-side JavaScript')
  console.log('Query vector dimensions:', queryVector.length)
  console.log('First 5 values:', queryVector.slice(0, 5))
}

// --- MemoryVectorStore: Add Documents & Search ---

async function basicVectorStore() {
  console.log('=== MemoryVectorStore ===\n')

  const documents = [
    'JavaScript was created by Brendan Eich in 1995 at Netscape.',
    'Python is a popular language for data science and machine learning.',
    'The Eiffel Tower is located in Paris, France.',
    'Node.js allows running JavaScript on the server side.',
    'React is a library for building user interfaces in JavaScript.',
    'The Great Wall of China is over 13,000 miles long.',
  ]

  const store = await MemoryVectorStore.fromTexts(
    documents,
    documents.map((_, i) => ({ id: i })),
    embeddings,
  )

  console.log('Stored %d documents\n', documents.length)

  const query = 'server-side JavaScript'
  const results = await store.similaritySearchWithScore(query, 3)

  console.log(`Query: "${query}" (top 3)\n`)
  for (const [doc, score] of results) {
    const bar = '█'.repeat(Math.round(score * 30))
    console.log(`  ${score.toFixed(3)} ${bar} ${doc.pageContent}`)
  }
}

// --- Metadata & Filtering ---

async function metadataAndFiltering() {
  console.log('=== Metadata & Filtering ===\n')

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

  const query = 'web framework'
  console.log(`Query: "${query}" — no filter (top 4)\n`)
  const allResults = await store.similaritySearchWithScore(query, 4)
  for (const [doc, score] of allResults) {
    console.log(`  ${score.toFixed(3)}  [${doc.metadata.language}] ${doc.pageContent}`)
  }

  console.log(`\nQuery: "${query}" — filter: python only (top 4)\n`)
  const pythonOnly = await store.similaritySearchWithScore(query, 4, (doc) => doc.metadata.language === 'python')
  for (const [doc, score] of pythonOnly) {
    console.log(`  ${score.toFixed(3)}  [${doc.metadata.language}] ${doc.pageContent}`)
  }
}

// --- Adding Documents Incrementally ---

async function incrementalDocuments() {
  console.log('=== Incremental Documents & The Document Object ===\n')

  // Create an EMPTY store — no documents yet
  const store = new MemoryVectorStore(embeddings)

  // In LangChain, everything is a Document: { pageContent, metadata }
  // fromTexts() created these for you behind the scenes.
  // Now we build them explicitly.

  const batch1 = [
    new Document({
      pageContent: 'LangChain is a framework for building LLM applications.',
      metadata: { source: 'intro.md', section: 'overview' },
    }),
    new Document({
      pageContent: 'Chains connect multiple LLM calls into a pipeline.',
      metadata: { source: 'intro.md', section: 'concepts' },
    }),
  ]

  // addDocuments embeds and stores them
  await store.addDocuments(batch1)
  console.log('Added batch 1: %d docs', batch1.length)

  // Later, more documents arrive — just add them to the same store
  const batch2 = [
    new Document({
      pageContent: 'Agents use LLMs to decide which tools to call.',
      metadata: { source: 'agents.md', section: 'overview' },
    }),
    new Document({
      pageContent: 'RAG retrieves relevant documents before generating an answer.',
      metadata: { source: 'rag.md', section: 'overview' },
    }),
  ]

  await store.addDocuments(batch2)
  console.log('Added batch 2: %d docs', batch2.length)
  console.log('Total docs in store: %d\n', store.memoryVectors.length)

  // Search across ALL documents (both batches)
  const results = await store.similaritySearchWithScore('how do agents work', 2)
  console.log('Query: "how do agents work" (top 2)\n')
  for (const [doc, score] of results) {
    console.log(`  ${score.toFixed(3)}  [${doc.metadata.source}] ${doc.pageContent}`)
  }
}

async function main() {
  // await testEmbeddings()
  // await basicVectorStore()
  // await metadataAndFiltering()
  await incrementalDocuments()
}

main().catch(console.error)
