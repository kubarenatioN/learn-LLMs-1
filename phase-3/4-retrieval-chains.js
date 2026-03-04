/**
 * LESSON 4 (Phase 3) — Retrieval Chains
 *
 * Combining retriever + LLM into a RAG pipeline.
 * The LLM answers questions using YOUR documents as context.
 */

import { TextLoader } from '@langchain/classic/document_loaders/fs/text'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { ChatOpenAI } from '@langchain/openai'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { RunnableLambda, RunnablePassthrough } from '@langchain/core/runnables'
import 'dotenv/config'

// --- Shared setup: load docs → split → store ---

async function buildVectorStore() {
  const files = ['phase-3/sample-docs/langchain-overview.md', 'phase-3/sample-docs/nodejs-basics.md']

  const allChunks = []

  for (const file of files) {
    const loader = new TextLoader(file)
    const docs = await loader.load()

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    })
    const chunks = await splitter.splitDocuments(docs)
    allChunks.push(...chunks)
  }

  console.log('Loaded %d files → %d chunks', files.length, allChunks.length)

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_TOKEN,
    model: 'ibm-granite/granite-embedding-small-english-r2',
    provider: 'hf-inference',
  })

  const store = await MemoryVectorStore.fromDocuments(allChunks, embeddings)
  console.log('Vector store ready: %d vectors\n', store.memoryVectors.length)

  return store
}

// --- LLM setup (same as Phase 2) ---

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: {
    baseURL: 'https://router.huggingface.co/v1',
  },
  apiKey: process.env.HF_TOKEN,
})

// --- Manual RAG: see every step ---

async function manualRAG() {
  console.log('=== Manual RAG ===\n')

  const store = await buildVectorStore()
  const query = 'How does the Node.js event loop work?'

  // Prepare retriever
  const retriever = store.asRetriever(3)
  // Step 1: Retrieve relevant chunks
  const relevantDocs = await retriever.invoke(query)

  console.log('Query: "%s"\n', query)
  console.log('Step 1 — Retrieved %d chunks:', relevantDocs.length)
  for (const doc of relevantDocs) {
    console.log('  [%s] \n %s...', doc.metadata.source, doc.pageContent.slice(0, 60))
  }

  // Step 2: Format the context — just join the chunks into one string
  const context = relevantDocs.map((doc) => doc.pageContent).join('\n\n')

  // Step 3: Build the prompt with context injected
  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the question based ONLY on the following context. 
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:`,
  )

  // Step 4: Call the LLM
  const chain = prompt.pipe(llm).pipe(new StringOutputParser())
  const answer = await chain.invoke({ context, question: query })

  console.log('\nStep 2 — Context: %d chars injected into prompt', context.length)
  console.log('\nStep 3 — LLM Answer:\n')
  console.log(answer)
}

// --- Chain-Based RAG ---

async function chainRAG() {
  console.log('=== Chain-Based RAG ===\n')

  const store = await buildVectorStore()
  const retriever = store.asRetriever(3)

  // Helper: format retrieved docs into a single context string
  const formatDocs = new RunnableLambda({
    func: (docs) => docs.map((doc) => doc.pageContent).join('\n\n'),
  })

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:`,
  )

  // The RAG chain:
  // Input: { question: string }
  //   1. RunnablePassthrough passes `question` through unchanged
  //   2. retriever.pipe(formatDocs) turns `question` into a context string
  //   3. Both feed into the prompt template
  //   4. prompt → llm → parser produces the answer
  const ragChain = RunnablePassthrough.assign({
    context: new RunnableLambda({
      func: (input) => input.question,
    }).pipe(retriever).pipe(formatDocs),
  })
    .pipe(prompt)
    .pipe(llm)
    .pipe(new StringOutputParser())

  // Now it's a single .invoke() call — question in, answer out
  const queries = [
    'How does the Node.js event loop work?',
    'What is the difference between chains and agents?',
    'What is the capital of Japan?',
  ]

  for (const question of queries) {
    console.log('Q: %s', question)
    const answer = await ragChain.invoke({ question })
    console.log('A: %s\n', answer)
  }
}

// --- RAG with Sources ---

async function ragWithSources() {
  console.log('=== RAG with Sources ===\n')

  const store = await buildVectorStore()
  const retriever = store.asRetriever(3)

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:`,
  )

  // Instead of a single string chain, we build a chain that returns
  // BOTH the answer AND the source documents.
  const ragChainWithSources = RunnablePassthrough.assign({
    docs: new RunnableLambda({ func: (input) => input.question }).pipe(retriever),
  })
    .assign({
      context: new RunnableLambda({
        func: (input) => input.docs.map((doc) => doc.pageContent).join('\n\n'),
      }),
    })
    .assign({
      answer: prompt.pipe(llm).pipe(new StringOutputParser()),
    })

  // The final output has: { question, docs, context, answer }
  // We can use docs to show sources!

  const question = 'What are Node.js streams used for?'
  console.log('Q: %s\n', question)

  const result = await ragChainWithSources.invoke({ question })

  console.log('A: %s\n', result.answer)
  console.log('Sources:')
  for (const doc of result.docs) {
    const lines = doc.metadata.loc?.lines
    console.log('  - %s (lines %d-%d)', doc.metadata.source, lines?.from, lines?.to)
  }
}

async function main() {
  // await manualRAG()
  // await chainRAG()
  await ragWithSources()
}

main().catch(console.error)
