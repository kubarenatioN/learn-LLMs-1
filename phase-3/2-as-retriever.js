/**
 * LESSON 2 (Phase 3) — Bridge to RAG chains
 */

import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import 'dotenv/config'

// --- LangChain Embeddings Wrapper ---

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HF_TOKEN,
  provider: 'hf-inference',
  model: 'ibm-granite/granite-embedding-small-english-r2',
})

async function retrieverDemo() {
  console.log('=== asRetriever() ===\n')

  const store = await MemoryVectorStore.fromTexts(
    [
      'Embeddings convert text into numerical vectors.',
      'Vector stores index embeddings for fast similarity search.',
      'Retrievers fetch relevant documents given a query.',
      'Chains connect retrievers and LLMs into a pipeline.',
      'RAG means Retrieval-Augmented Generation.',
      'Agents decide which tools to call at runtime.',
    ],
    [{ topic: 'embeddings' }, { topic: 'vector-stores' }, { topic: 'retrievers' }, { topic: 'chains' }, { topic: 'rag' }, { topic: 'agents' }],
    embeddings,
  )

  // asRetriever(k) wraps the store in a Retriever interface.
  // A retriever has ONE method: invoke(query) → Document[]
  // No scores, no options — just "give me the top k relevant docs."
  // This simplicity is what makes it pluggable into chains.
  const retriever = store.asRetriever(3)

  const docs = await retriever.invoke('What is RAG and how does retrieval work?')

  console.log('Retriever returned %d documents:\n', docs.length)
  for (const doc of docs) {
    console.log(`  [${doc.metadata.topic}] ${doc.pageContent}`)
  }

  // Compare: similaritySearch returns [doc, score] tuples — more control.
  // Retriever returns just docs — simpler, chain-friendly.
  console.log('\n--- Why this matters ---')
  console.log('similaritySearchWithScore() → for exploration, debugging, seeing scores')
  console.log('asRetriever().invoke()       → for plugging into chains (RAG pipeline)')
}

async function main() {
  await retrieverDemo()
}

main().catch(console.error)
