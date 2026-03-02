/**
 * LESSON 1 (Phase 4) — Tools & Tool Calling
 *
 * Giving LLMs the ability to call functions.
 * The LLM doesn't execute code — it outputs a structured request
 * saying "I want to call function X with arguments Y."
 * YOUR code executes the function and feeds the result back.
 */

import { tool } from '@langchain/core/tools'
import { ChatOpenAI } from '@langchain/openai'
import { z } from 'zod'
import 'dotenv/config'

// --- Defining tools ---

// A tool is just a function + metadata (name, description, input schema).
// The description is CRITICAL — it's what the LLM reads to decide when to use the tool.

const weatherTool = tool(
  async ({ city }) => {
    // In a real app, this would call a weather API.
    // For learning, we fake it.
    const fakeWeather = {
      'London': 'Cloudy, 12°C, light rain',
      'Tokyo': 'Sunny, 24°C, clear skies',
      'New York': 'Partly cloudy, 18°C, humid',
    }
    return fakeWeather[city] || `Weather data not available for ${city}`
  },
  {
    name: 'get_weather',
    description: 'Get the current weather for a city. Use this when the user asks about weather conditions.',
    schema: z.object({
      city: z.string().describe('The city name to get weather for'),
    }),
  },
)

const calculatorTool = tool(
  async ({ expression }) => {
    try {
      const result = new Function(`return ${expression}`)()
      return `${expression} = ${result}`
    } catch {
      return `Error evaluating: ${expression}`
    }
  },
  {
    name: 'calculator',
    description: 'Evaluate a mathematical expression. Use this for any math calculations.',
    schema: z.object({
      expression: z.string().describe('The math expression to evaluate, e.g. "2 + 2" or "Math.sqrt(144)"'),
    }),
  },
)

// Let's see what a tool looks like from LangChain's perspective
async function inspectTools() {
  console.log('=== Tool Inspection ===\n')

  console.log('Name:', weatherTool.name)
  console.log('Description:', weatherTool.description)
  console.log('Schema:', JSON.stringify(weatherTool.schema.shape, null, 2))

  // You can call a tool directly (but normally the LLM decides when to call it)
  console.log('\nDirect call result:', await weatherTool.invoke({ city: 'Tokyo' }))
  console.log('Calculator result:', await calculatorTool.invoke({ expression: '15 * 7 + 3' }))
}

// --- LLM + Tools: bindTools ---

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: { baseURL: 'https://router.huggingface.co/v1' },
  apiKey: process.env.HF_TOKEN,
})

async function toolCalling() {
  console.log('=== Tool Calling — How the LLM Requests Tool Use ===\n')

  // bindTools tells the LLM what tools are available.
  // The LLM doesn't get the function code — only the name, description, and schema.
  const llmWithTools = llm.bindTools([weatherTool, calculatorTool])

  // Ask something that requires a tool
  const response = await llmWithTools.invoke('What is the weather in Tokyo?')

  // The LLM does NOT return a text answer.
  // It returns a message with tool_calls — a structured request.
  console.log('Response content:', JSON.stringify(response.content))
  console.log('\nTool calls:')
  for (const call of response.tool_calls) {
    console.log('  Tool: %s', call.name)
    console.log('  Args: %j', call.args)
    console.log('  ID:   %s', call.id)
  }

  console.log('\n→ The LLM said "call get_weather with city=Tokyo"')
  console.log('→ It did NOT execute the tool. It did NOT know the answer.')
  console.log('→ YOUR code must execute the tool and feed the result back.')

  // Now ask something that DOESN'T need a tool
  console.log('\n--- No tool needed ---\n')
  const response2 = await llmWithTools.invoke('Say hello in Japanese')
  console.log('Response content:', response2.content)
  console.log('Tool calls:', response2.tool_calls?.length || 0)
  console.log('\n→ The LLM answered directly — no tool needed for this question.')
}

// --- The Full Loop: call → execute → respond ---

async function fullToolLoop() {
  console.log('=== Full Tool-Calling Loop ===\n')

  const tools = [weatherTool, calculatorTool]
  const toolMap = Object.fromEntries(tools.map((t) => [t.name, t]))
  const llmWithTools = llm.bindTools(tools)

  const query = 'What is the weather in London, and what is 42 * 17?'
  console.log('User: %s\n', query)

  // Step 1: LLM decides which tools to call
  const aiResponse = await llmWithTools.invoke(query)
  console.log('Step 1 — LLM requested %d tool call(s):', aiResponse.tool_calls.length)
  for (const call of aiResponse.tool_calls) {
    console.log('  %s(%j)', call.name, call.args)
  }

  // Step 2: Execute each tool and collect results as ToolMessages
  // ToolMessage tells the LLM "here's the result of the tool you requested"
  const { ToolMessage } = await import('@langchain/core/messages')

  const toolResults = []
  for (const call of aiResponse.tool_calls) {
    const selectedTool = toolMap[call.name]
    const result = await selectedTool.invoke(call.args)
    toolResults.push(
      new ToolMessage({
        content: result,
        tool_call_id: call.id,
      }),
    )
    console.log('\nStep 2 — Executed %s → "%s"', call.name, result)
  }

  // Step 3: Send everything back to the LLM:
  //   original query + AI's tool request + tool results
  // The LLM now has the data it needed and can compose a final answer.
  const { HumanMessage } = await import('@langchain/core/messages')

  const finalResponse = await llmWithTools.invoke([
    new HumanMessage(query),
    aiResponse,
    ...toolResults,
  ])

  console.log('\nStep 3 — Final answer:\n')
  console.log(finalResponse.content)
}

async function main() {
  // await inspectTools()
  // await toolCalling()
  await fullToolLoop()
}

main().catch(console.error)
