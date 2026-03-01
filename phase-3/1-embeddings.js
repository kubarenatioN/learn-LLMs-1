/**
 * LESSON 1 (Phase 3) — Embeddings & Semantic Search
 *
 * Turning text into vectors, then using those vectors to find
 * similar/relevant text. This is the foundation of RAG.
 *
 * NOTE: HuggingFace's router.huggingface.co/v1 only supports chat completions.
 * For embeddings, we use the @huggingface/inference SDK (featureExtraction).
 */

import { InferenceClient } from '@huggingface/inference'
import 'dotenv/config'

const hf = new InferenceClient(process.env.HF_TOKEN)

// Helper: get embedding vector for a text string
async function embed(text) {
  const result = await hf.featureExtraction({
    model: 'ibm-granite/granite-embedding-small-english-r2',
    provider: 'hf-inference',
    inputs: text,
  })
  return result
}

// ---- SEEING EMBEDDINGS ----

async function seeEmbeddings() {
  console.log('=== What Embeddings Look Like ===\n')

  const vector = await embed('The cat sat on the mat')

  console.log('Input text: "The cat sat on the mat"')
  console.log('Vector length:', vector.length, 'dimensions')
  console.log('First 10 values:', vector.slice(0, 10))
}

// ---- SEMANTIC SIMILARITY ----

// Cosine similarity: measures how similar two vectors are.
// Returns a value between -1 and 1. Closer to 1 = more similar meaning.
function cosineSimilarity(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

async function semanticSearch() {
  console.log('=== Semantic Search ===\n')

  // 0. Imagine these are chunks from your documents
  const documents = [
    'JavaScript was created by Brendan Eich in 1995 at Netscape.',
    'Python is a popular language for data science and machine learning.',
    'The Eiffel Tower is located in Paris, France.',
    'Node.js allows running JavaScript on the server side.',
    'React is a library for building user interfaces in JavaScript.',
    'The Great Wall of China is over 13,000 miles long.',
  ]

  // 1. Embed all documents
  console.log('Embedding %d documents...', documents.length)
  const docVectors = await Promise.all(documents.map((doc) => embed(doc)))

  // Our query:
  const query = 'Who built the Wall of China?'
  console.log('Query:', query, '\n')

  // 2. Embed the query
  const queryVector = await embed(query)

  // 3. Compare the query to all the documents.
  // Score each document by similarity to the query
  const scored = documents.map((doc, i) => ({
    text: doc,
    score: cosineSimilarity(queryVector, docVectors[i]),
  }))

  // 4. Sort by score descending — most relevant first
  scored.sort((a, b) => b.score - a.score)

  console.log('Results (ranked by relevance):')
  for (const { text, score } of scored) {
    const bar = '█'.repeat(Math.round(score * 30))
    console.log(`  ${score.toFixed(3)} ${bar} ${text}`)
  }
}

async function main() {
  // await seeEmbeddings()
  console.log('\n')
  await semanticSearch()
}

main().catch(console.error)
