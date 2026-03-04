/**
 * LESSON 2 (Phase 4) — Agents & the ReAct Pattern
 *
 * An agent is just a loop: LLM reasons → calls tool → observes result → repeats.
 * We build it manually first, then use LangChain's AgentExecutor.
 */

import { AgentExecutor, createToolCallingAgent } from '@langchain/classic/agents'
import { HumanMessage, ToolMessage } from '@langchain/core/messages'
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts'
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

// --- Tools ---

const weatherTool = tool(
  async ({ city }) => {
    const data = {
      London: 'Cloudy, 12°C',
      Tokyo: 'Sunny, 24°C',
      'New York': 'Rainy, 15°C',
      Paris: 'Clear, 20°C',
    }
    return data[city] || `No data for ${city}`
  },
  {
    name: 'get_weather',
    description: 'Get current weather for a city.',
    schema: z.object({ city: z.string().describe('City name') }),
  },
)

const calculatorTool = tool(
  async ({ expression }) => {
    try {
      return String(new Function(`return ${expression}`)())
    } catch {
      return `Error: ${expression}`
    }
  },
  {
    name: 'calculator',
    description: 'Evaluate a math expression. Returns the numeric result. The expression MUST be formatted for execition in a javascript environment.',
    schema: z.object({ expression: z.string().describe('Math expression like "2+2" or "Math.acos(0.5)"') }),
  },
)

const tools = [weatherTool, calculatorTool]
const toolMap = Object.fromEntries(tools.map((t) => [t.name, t]))

// --- Manual Agent Loop ---

async function manualAgentLoop() {
  console.log('=== Manual Agent Loop ===\n')

  const llmWithTools = llm.bindTools(tools)
  const query = 'What is the temperature in Tokyo and London? What is the distance between the two cities on a spherical earth?'
  console.log('User: %s\n', query)

  // The message history — this IS the agent's "memory" within one task
  const messages = [new HumanMessage(query)]

  let step = 0
  const MAX_STEPS = 5

  while (step < MAX_STEPS) {
    step++
    console.log('--- Step %d ---', step)

    // LLM reasons about what to do next
    const response = await llmWithTools.invoke(messages)
    messages.push(response)

    // If no tool calls, the LLM is giving a final answer
    if (!response.tool_calls || response.tool_calls.length === 0) {
      console.log('Final answer: %s\n', response.content)
      break
    }

    // Execute requested tools
    for (const call of response.tool_calls) {
      console.log('  Calling: %s(%j)', call.name, call.args)
      const result = await toolMap[call.name].invoke(call.args)
      console.log('  Result: %s', result)

      messages.push(new ToolMessage({ content: result, tool_call_id: call.id }))
    }
  }

  if (step >= MAX_STEPS) {
    console.log('⚠ Reached max steps without a final answer')
  }

  console.log('Total steps: %d', step)
}

// --- AgentExecutor: LangChain's Built-in Agent Loop ---

async function agentExecutorDemo() {
  console.log('=== AgentExecutor ===\n')

  // The agent needs a prompt with two placeholders:
  //   {input} — the user's question
  //   {agent_scratchpad} — where the agent's intermediate steps go (tool calls + results)
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant. Use tools when needed. Be concise.'],
    ['human', '{input}'],
    new MessagesPlaceholder('agent_scratchpad'),
  ])

  // createToolCallingAgent wires up:  prompt + LLM + tools
  const agent = createToolCallingAgent({ llm, tools, prompt })

  // AgentExecutor runs the loop we built manually:
  //   LLM → tool calls → execute → LLM → ... → final answer
  const executor = new AgentExecutor({
    agent,
    tools,
    // verbose: true, // shows the agent's reasoning steps
    maxIterations: 5, // same as our MAX_STEPS
  })

  // Single .invoke() — the executor handles the entire loop
  const result = await executor.invoke({
    input: 'What is the weather in Paris? Convert the temperature from Celsius to Fahrenheit.',
  })

  console.log('\n=== Final Result ===')
  console.log(result.output)
}

// --- Multi-query Agent ---

async function multiQueryAgent() {
  console.log('=== Multi-Query Agent ===\n')

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant. Use tools when needed. Be concise.'],
    ['human', '{input}'],
    new MessagesPlaceholder('agent_scratchpad'),
  ])

  const agent = createToolCallingAgent({ llm, tools, prompt })
  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: false,
    maxIterations: 5,
  })

  const queries = [
    'What is the weather in Tokyo?',
    'What is 2 to the power of 10?',
    'If it is 24°C in Tokyo and 12°C in London, what is the average?',
    'What is the meaning of life?',
  ]

  for (const input of queries) {
    console.log('Q: %s', input)
    const { output } = await executor.invoke({ input })
    console.log('A: %s\n', output)
  }
}

// --- Modern API: createAgent from "langchain" ---

async function modernAgent() {
  console.log('=== createAgent (Modern API) ===\n')

  // createAgent replaces both createToolCallingAgent + AgentExecutor (legacy)
  // and createReactAgent from @langchain/langgraph/prebuilt (also deprecated).
  // Import from "langchain" directly — the new single entry point.
  const { createAgent } = await import('langchain')

  // Much simpler setup:
  //   model — accepts a ChatModel instance or a string like "openai:gpt-4o"
  //   tools — same tool() definitions as before
  //   systemPrompt — a plain string (becomes the system message)
  const agent = createAgent({
    model: llm,
    tools,
    systemPrompt: 'You are a helpful assistant. Use tools when needed. Be concise.',
  })

  // Input/output use a `messages` array (not `input`/`output` strings)
  const queries = [
    'What is the weather in Tokyo?',
    'What is 2 to the power of 10?',
    'If it is 24°C in Tokyo and 12°C in London, what is the average?',
    'What is the meaning of life?',
  ]

  for (const query of queries) {
    console.log('Q: %s', query)

    const result = await agent.invoke({
      messages: [{ role: 'user', content: query }],
    })

    const lastMessage = result.messages[result.messages.length - 1]
    console.log('A: %s\n', lastMessage.content)
    console.log('(messages count): %d\n', result.messages.length)
  }
}

async function main() {
  // await manualAgentLoop()
  // await agentExecutorDemo()
  // await multiQueryAgent()
  await modernAgent()
}

main().catch(console.error)
