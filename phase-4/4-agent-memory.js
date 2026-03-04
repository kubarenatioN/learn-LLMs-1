/**
 * LESSON 4 (Phase 4) — Agent Memory & Planning
 *
 * Agents so far are stateless between invocations.
 * Here we add memory so they remember previous conversations.
 */

import { tool } from '@langchain/core/tools'
import { MemorySaver } from '@langchain/langgraph'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'
import { z } from 'zod'

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: { baseURL: 'https://router.huggingface.co/v1' },
  apiKey: process.env.HF_TOKEN,
})

const weatherTool = tool(
  async ({ city }) => {
    const data = { London: 'Cloudy, 12°C', Tokyo: 'Sunny, 24°C', Paris: 'Clear, 20°C' }
    return data[city] || `No data for ${city}`
  },
  {
    name: 'get_weather',
    description: 'Get current weather for a city.',
    schema: z.object({ city: z.string().describe('City name') }),
  },
)

// --- Block 1: The stateless problem ---

async function statelessProblem() {
  console.log('=== Stateless Agent (no memory) ===\n')

  const { createAgent } = await import('langchain')

  const agent = createAgent({
    model: llm,
    tools: [weatherTool],
    systemPrompt: 'You are a helpful assistant. Be concise. In case you dont know the answer, say it directly.',
  })

  // First question
  const r1 = await agent.invoke({
    messages: [{ role: 'user', content: "What's the weather in Tokyo?" }],
  })
  console.log("Q1: What's the weather in Tokyo?")
  console.log('A1: %s\n', r1.messages[r1.messages.length - 1].content)

  // Follow-up — depends on knowing we just talked about Tokyo
  const r2 = await agent.invoke({
    messages: [{ role: 'user', content: 'How about London? Is it warmer or colder than the previous city?' }],
  })
  console.log('Q2: How about London? Is it warmer or colder than the previous city?')
  console.log('A2: %s\n', r2.messages[r2.messages.length - 1].content)

  console.log('→ The agent has no idea what "the previous city" means.\n')
}

// --- Block 2: Stateful agent with MemorySaver ---

async function statefulAgent() {
  console.log('=== Stateful Agent (with MemorySaver) ===\n')

  const { createAgent } = await import('langchain')

  const memory = new MemorySaver()

  const agent = createAgent({
    model: llm,
    tools: [weatherTool],
    systemPrompt: 'You are a helpful assistant. Be concise.',
    checkpointer: memory,
  })

  // First question
  const r1 = await agent.invoke(
    { messages: [{ role: 'user', content: "What's the weather in Tokyo?" }] },
    // thread_id groups messages into a conversation
    { configurable: { thread_id: 'conversation-1' } },
  )
  console.log("Q1: What's the weather in Tokyo?")
  console.log('A1: %s\n', r1.messages[r1.messages.length - 1].content)

  // Follow-up — same thread_id, so the agent remembers Tokyo
  const r2 = await agent.invoke(
    { messages: [{ role: 'user', content: 'How about London? Is it warmer or colder than the previous city?' }] },
    { configurable: { thread_id: 'conversation-1' } },
  )
  console.log('Q2: How about London? Is it warmer or colder than the previous city?')
  console.log('A2: %s\n', r2.messages[r2.messages.length - 1].content)
}

// --- Block 3: Multiple threads — isolated conversations ---

async function multipleThreads() {
  console.log('=== Multiple Threads (isolated conversations) ===\n')

  const { createAgent } = await import('langchain')
  const memory = new MemorySaver()

  const agent = createAgent({
    model: llm,
    tools: [weatherTool],
    systemPrompt: 'You are a helpful assistant. Be concise.',
    checkpointer: memory,
  })

  // Thread A: asking about Tokyo
  const threadA = { configurable: { thread_id: 'user-alice' } }
  await agent.invoke({ messages: [{ role: 'user', content: "What's the weather in Tokyo?" }] }, threadA)
  console.log('[Alice] Asked about Tokyo')

  // Thread B: asking about Paris (completely separate conversation)
  const threadB = { configurable: { thread_id: 'user-bob' } }
  await agent.invoke({ messages: [{ role: 'user', content: "What's the weather in Paris?" }] }, threadB)
  console.log('[Bob]   Asked about Paris')

  // Alice follows up — should remember Tokyo, know nothing about Paris
  const r = await agent.invoke({ messages: [{ role: 'user', content: 'What city did we just discuss?' }] }, threadA)
  const answer = r.messages[r.messages.length - 1].content
  console.log('\n[Alice] "What city did we just discuss?"')
  console.log('[Alice] %s\n', answer)

  console.log("→ Alice sees Tokyo. Bob's Paris conversation is completely isolated.")
}

async function main() {
  // await statelessProblem()
  // await statefulAgent()
  await multipleThreads()
}

main().catch(console.error)
