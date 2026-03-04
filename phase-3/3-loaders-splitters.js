/**
 * LESSON 3 (Phase 3) — Document Loaders & Text Splitters
 *
 * Loading real files into LangChain Documents, then splitting them
 * into chunks suitable for embedding and vector storage.
 */

import { TextLoader } from '@langchain/classic/document_loaders/fs/text'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import 'dotenv/config'

// --- Loading a file ---

async function loadFile() {
  console.log('=== TextLoader ===\n')

  // TextLoader reads a file and returns an array of Document objects.
  // Each Document has: { pageContent: string, metadata: { source: string } }
  const loader = new TextLoader('phase-3/sample-docs/langchain-overview.md')
  const docs = await loader.load()

  console.log('Documents loaded:', docs.length)
  console.log('Type:', typeof docs[0].pageContent)
  console.log('Metadata:', docs[0].metadata)
  console.log('Content length:', docs[0].pageContent.length, 'characters')
  console.log('\nFirst 200 chars:\n')
  console.log(docs[0].pageContent.slice(0, 200))
  console.log('...')
}

// --- Splitting into chunks ---

async function splitDocument() {
  console.log('=== RecursiveCharacterTextSplitter ===\n')

  // 1. Load the file
  const loader = new TextLoader('phase-3/sample-docs/langchain-overview.md')
  const docs = await loader.load()

  console.log('Before splitting: %d document, %d chars\n', docs.length, docs[0].pageContent.length)

  // 2. Create a splitter
  // chunkSize: max characters per chunk
  // chunkOverlap: characters shared between adjacent chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  })

  // 3. Split the documents
  const chunks = await splitter.splitDocuments(docs)

  console.log('After splitting: %d chunks\n', chunks.length)

  // 4. Inspect each chunk
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]
    console.log(`--- Chunk ${i} (${chunk.pageContent.length} chars) ---`)
    console.log(chunk.pageContent.slice(0, 120) + '...')
    console.log('Metadata:', chunk.metadata)
    console.log()
  }
}

// --- Chunk size experiments ---

async function chunkSizeExperiment() {
  console.log('=== Chunk Size Experiments ===\n')

  const loader = new TextLoader('phase-3/sample-docs/langchain-overview.md')
  const docs = await loader.load()
  const totalChars = docs[0].pageContent.length

  const configs = [
    { chunkSize: 200, chunkOverlap: 0 },
    { chunkSize: 200, chunkOverlap: 40 },
    { chunkSize: 500, chunkOverlap: 50 },
    { chunkSize: 1000, chunkOverlap: 100 },
  ]

  for (const config of configs) {
    const splitter = new RecursiveCharacterTextSplitter(config)
    const chunks = await splitter.splitDocuments(docs)
    const avgSize = Math.round(chunks.reduce((sum, c) => sum + c.pageContent.length, 0) / chunks.length)

    console.log(
      'size=%d overlap=%d → %d chunks (avg %d chars)',
      config.chunkSize,
      config.chunkOverlap,
      chunks.length,
      avgSize,
    )
  }

  console.log('\n--- Guidelines ---')
  console.log('Too small (200): many chunks, context fragmented, embedding loses meaning')
  console.log('Too large (1000+): few chunks, search returns overly broad results')
  console.log('Sweet spot (300-800): depends on your data. Experiment!')
  console.log('Overlap: 10-20%% of chunkSize prevents losing context at boundaries')
}

// --- Full Pipeline: Load → Split → Store → Search ---

async function fullPipeline() {
  console.log('=== Full Pipeline: File → Chunks → Vector Store → Search ===\n')

  // 1. LOAD
  const loader = new TextLoader('phase-3/sample-docs/langchain-overview.md')
  const docs = await loader.load()
  console.log('1. Loaded: %d document (%d chars)', docs.length, docs[0].pageContent.length)

  // 2. SPLIT
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  })
  const chunks = await splitter.splitDocuments(docs)
  console.log('2. Split into: %d chunks', chunks.length)

  // 3. STORE — embed all chunks and add to vector store
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_TOKEN,
    model: 'ibm-granite/granite-embedding-small-english-r2',
    provider: 'hf-inference',
  })

  const store = await MemoryVectorStore.fromDocuments(chunks, embeddings)
  console.log('3. Stored: %d vectors\n', store.memoryVectors.length)

  // 4. SEARCH — ask questions against the file
  const queries = [
    'What are agents and how do they work?',
    'How is RAG different from fine-tuning?',
    'What is memory in LangChain?',
  ]

  for (const query of queries) {
    console.log(`Q: "${query}"`)
    const results = await store.similaritySearchWithScore(query, 2)
    for (const [doc, score] of results) {
      const lines = doc.metadata.loc?.lines
      const lineInfo = lines ? `lines ${lines.from}-${lines.to}` : ''
      console.log(`  ${score.toFixed(3)}  [${lineInfo}] ${doc.pageContent.slice(0, 80)}...`)
    }
    console.log()
  }
}

async function main() {
  // await loadFile()
  // await splitDocument()
  // await chunkSizeExperiment()
  await fullPipeline()
}

main().catch(console.error)
