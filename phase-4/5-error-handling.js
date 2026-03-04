/**
 * LESSON 5 (Phase 4) — Error Handling & Guardrails
 *
 * Making agents robust: handling tool failures, limiting loops,
 * and using middleware to enforce safety.
 */

import { tool } from '@langchain/core/tools'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'
import { z } from 'zod'

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: { baseURL: 'https://router.huggingface.co/v1' },
  apiKey: process.env.HF_TOKEN,
})

// --- A tool that fails sometimes ---

let callCount = 0

const unreliableTool = tool(
  async ({ query }) => {
    callCount++
    console.log('  [unreliableTool] call #%d, query: "%s"', callCount, query)

    if (callCount % 2 === 1) {
      throw new Error('Service temporarily unavailable')
    }
    return `Results for "${query}": Found 3 relevant documents.`
  },
  {
    name: 'search',
    description: 'Search for information. May occasionally fail due to service issues.',
    schema: z.object({ query: z.string().describe('Search query') }),
  },
)

// --- Block 1: Unhandled tool error (what happens by default) ---

async function unhandledToolError() {
  console.log('=== Block 1: Unhandled Tool Error ===\n')

  const { createAgent } = await import('langchain')

  callCount = 0
  const agent = createAgent({
    model: llm,
    tools: [unreliableTool],
    systemPrompt: 'You are a helpful assistant. Use the search tool to answer questions.',
  })

  try {
    const result = await agent.invoke({
      messages: [{ role: 'user', content: 'Search for "LangChain tutorials"' }],
    })
    console.log('Answer: %s', result.messages[result.messages.length - 1].content)
  } catch (err) {
    console.log('Agent crashed: %s', err.message)
    console.log('\n→ Without error handling, a tool exception kills the entire agent.\n')
  }
}

// --- Block 2: toolRetryMiddleware — automatic retries with backoff ---

async function withToolRetry() {
  console.log('=== Block 2: Tool Retry Middleware ===\n')

  const { createAgent, toolRetryMiddleware } = await import('langchain')

  callCount = 0
  const agent = createAgent({
    model: llm,
    tools: [unreliableTool],
    systemPrompt: 'You are a helpful assistant. Use the search tool to answer questions.',
    middleware: [
      toolRetryMiddleware({
        maxRetries: 3,
        initialDelayMs: 100,
        backoffFactor: 2,
        onFailure: 'continue',
      }),
    ],
  })

  const result = await agent.invoke({
    messages: [{ role: 'user', content: 'Search for "LangChain tutorials"' }],
  })

  console.log('\nAnswer: %s', result.messages[result.messages.length - 1].content)
  console.log('\n→ Tool was called %d times (retry middleware handled the failures)\n', callCount)
}

// --- Block 3: modelCallLimitMiddleware — preventing runaway loops ---

const loopyTool = tool(
  async ({ step }) => `Step ${step} done. You must call this tool again with step ${step + 1}.`,
  {
    name: 'next_step',
    description: 'Process the next step. Always call this tool with the next step number.',
    schema: z.object({ step: z.number().describe('Current step number') }),
  },
)

async function withModelCallLimit() {
  console.log('=== Block 3: Model Call Limit Middleware ===\n')

  const { createAgent, modelCallLimitMiddleware } = await import('langchain')

  const agent = createAgent({
    model: llm,
    tools: [loopyTool],
    systemPrompt: 'You are a helpful assistant. Follow tool instructions exactly.',
    middleware: [
      modelCallLimitMiddleware({
        runLimit: 4,
        exitBehavior: 'end',
      }),
    ],
  })

  const result = await agent.invoke({
    messages: [{ role: 'user', content: 'Start processing from step 1.' }],
  })

  const answer = result.messages[result.messages.length - 1].content
  console.log('Final answer: %s', answer)
  console.log('Total messages: %d', result.messages.length)
  console.log('\n→ Agent was forced to stop after 4 model calls (would loop forever otherwise)\n')
}

// --- Block 4: System prompt guardrails ---

const weatherTool = tool(
  async ({ city }) => {
    const data = { London: 'Cloudy, 12°C', Tokyo: 'Sunny, 24°C', Paris: 'Clear, 20°C' }
    return data[city] || `No weather data for ${city}`
  },
  {
    name: 'get_weather',
    description: 'Get current weather for a city.',
    schema: z.object({ city: z.string().describe('City name') }),
  },
)

async function systemPromptGuardrails() {
  console.log('=== Block 4: System Prompt Guardrails ===\n')

  const { createAgent } = await import('langchain')

  const constrainedAgent = createAgent({
    model: llm,
    tools: [weatherTool],
    systemPrompt: [
      'You are a weather assistant. You ONLY answer weather-related questions.',
      'Rules:',
      '- If the user asks about anything other than weather, politely decline.',
      '- If the weather tool returns no data for a city, say so honestly. Do NOT make up weather data.',
      '- Never provide travel advice, restaurant recommendations, or any non-weather information.',
      '- Keep answers to 1-2 sentences maximum.',
    ].join('\n'),
  })

  const queries = [
    'What is the weather in Tokyo?',
    'What are the best restaurants in Tokyo?',
    'What is the weather in Berlin?',
    'Tell me a joke.',
  ]

  for (const query of queries) {
    console.log('Q: %s', query)
    const result = await constrainedAgent.invoke({
      messages: [{ role: 'user', content: query }],
    })
    console.log('A: %s\n', result.messages[result.messages.length - 1].content)
  }
}

async function main() {
  // await unhandledToolError()
  // await withToolRetry()
  // await withModelCallLimit()
  await systemPromptGuardrails()
}

main().catch(console.error)
