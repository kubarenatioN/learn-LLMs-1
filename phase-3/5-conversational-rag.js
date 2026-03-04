/**
 * LESSON 5 (Phase 3) — Conversational RAG
 *
 * RAG with memory: follow-up questions, context from previous turns,
 * and question rephrasing so the retriever understands what "it" means.
 */

import { TextLoader } from '@langchain/classic/document_loaders/fs/text'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf'
import { AIMessage, HumanMessage } from '@langchain/core/messages'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import 'dotenv/config'

// --- Shared setup ---

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: { baseURL: 'https://router.huggingface.co/v1' },
  apiKey: process.env.HF_TOKEN,
})

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
    allChunks.push(...(await splitter.splitDocuments(docs)))
  }

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_TOKEN,
    model: 'ibm-granite/granite-embedding-small-english-r2',
    provider: 'hf-inference',
  })

  const store = await MemoryVectorStore.fromDocuments(allChunks, embeddings)
  console.log('Vector store ready: %d chunks\n', store.memoryVectors.length)
  return store
}

// --- The Problem: follow-ups fail ---

async function showTheProblem() {
  console.log('=== The Problem: Follow-ups Without Memory ===\n')

  const store = await buildVectorStore()
  const retriever = store.asRetriever(3)

  // First question works fine
  const q1 = 'What are Node.js streams?'
  const docs1 = await retriever.invoke(q1)
  console.log('Q1: "%s"', q1)
  console.log('Retrieved: %s...\n', docs1[0].pageContent.slice(0, 400))

  // Follow-up: "What types are there?" — the retriever has no idea what "there" means
  const q2 = 'What types are there?'
  const docs2 = await retriever.invoke(q2)
  console.log('Q2: "%s"', q2)
  console.log('Retrieved: %s...', docs2[0].pageContent.slice(0, 400))
  console.log('\n→ The retriever searched for "types" in general, not "types of streams"')
  console.log('→ It has no memory of Q1. Each query is independent.')
}

// --- Question Rephrasing ---

async function questionRephrasing() {
  console.log('=== Question Rephrasing ===\n')

  // This prompt takes conversation history + a follow-up question,
  // and produces a standalone question the retriever can understand.
  const rephrasePrompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `Given the following conversation and a follow-up question, 
rephrase the follow-up into a standalone question. 
Return ONLY the rephrased question, nothing else.`,
    ],
    new MessagesPlaceholder('history'),
    ['human', '{question}'],
  ])

  const rephraseChain = rephrasePrompt.pipe(llm).pipe(new StringOutputParser())

  // Simulate a conversation
  const history = [
    new HumanMessage('What are Node.js streams?'),
    new AIMessage('Streams are a Node.js feature for handling large amounts of data efficiently by processing it in chunks rather than loading everything into memory.'),
  ]

  // Vague follow-up that wouldn't work on its own
  const followUps = ['What types are there?', 'Are they used in Express?', 'How is that different from loading a whole file?']

  for (const question of followUps) {
    const rephrased = await rephraseChain.invoke({ history, question })
    console.log('Original:  "%s"', question)
    console.log('Rephrased: "%s"\n', rephrased)
  }
}

// --- Full Conversational RAG ---

async function conversationalRAG() {
  console.log('=== Full Conversational RAG ===\n')

  const store = await buildVectorStore()
  const retriever = store.asRetriever(3)

  // 1. Rephrase chain: history + follow-up → standalone question
  const rephrasePrompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `Given the following conversation and a follow-up question,
rephrase the follow-up into a standalone question.
Return ONLY the rephrased question, nothing else.`,
    ],
    new MessagesPlaceholder('history'),
    ['human', '{question}'],
  ])
  const rephraseChain = rephrasePrompt.pipe(llm).pipe(new StringOutputParser())

  // 2. Answer chain: context + question → answer
  const answerPrompt = ChatPromptTemplate.fromTemplate(
    `Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {standaloneQuestion}

Answer:`,
  )
  const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser())

  // 3. The conversation loop
  const history = []

  async function ask(question) {
    console.log('User: %s', question)

    // Step A: Rephrase if there's history, otherwise use as-is
    let standaloneQuestion = question
    if (history.length > 0) {
      standaloneQuestion = await rephraseChain.invoke({ history, question })
      console.log('  (rephrased: "%s")', standaloneQuestion)
    }

    // Step B: Retrieve relevant chunks using the standalone question
    const docs = await retriever.invoke(standaloneQuestion)
    const context = docs.map((d) => d.pageContent).join('\n\n')

    // Step C: Generate the answer
    const answer = await answerChain.invoke({ context, standaloneQuestion })
    console.log('Assistant: %s\n', answer)

    // Step D: Add this turn to history for future rephrasing
    history.push(new HumanMessage(question))
    history.push(new AIMessage(answer))

    return answer
  }

  // Multi-turn conversation
  await ask('What are Node.js streams?')
  await ask('What types are there?')
  await ask('What are they useful for?')
  await ask('Now tell me about agents in LangChain')
  await ask('How do they decide what to do?')
  await ask('How many sugars in coke?')
}

async function main() {
  // await showTheProblem()
  // await questionRephrasing()
  await conversationalRAG()
}

main().catch(console.error)
