/**
 * LESSON 3 (Phase 4) — Custom Tools
 *
 * Building real, useful tools that interact with the outside world:
 * web fetching, file reading, and combining them in an agent.
 */

import { tool } from '@langchain/core/tools'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'
import fs from 'node:fs/promises'
import path from 'node:path'
import { z } from 'zod'

const llm = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0,
  configuration: { baseURL: 'https://router.huggingface.co/v1' },
  apiKey: process.env.HF_TOKEN,
})

// --- Tool 1: Web Fetcher ---

const webFetchTool = tool(
  async ({ url }) => {
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'LLM-Agent/1.0' },
        signal: AbortSignal.timeout(10_000),
      })

      if (!response.ok) {
        return `HTTP error ${response.status}: ${response.statusText}`
      }

      const html = await response.text()

      // Strip HTML tags, scripts, styles — give the LLM clean text
      const text = html
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<style[\s\S]*?<\/style>/gi, '')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()

      // Truncate — LLMs have context limits, and most pages are huge
      const maxChars = 3000
      if (text.length > maxChars) {
        return text.slice(0, maxChars) + '\n\n[...truncated, page was ' + text.length + ' chars]'
      }

      return text || 'Page loaded but no text content found.'
    } catch (err) {
      return `Fetch failed: ${err.message}`
    }
  },
  {
    name: 'web_fetch',
    description:
      'Fetch a web page and return its text content. Use this to retrieve information from URLs. Returns plain text with HTML stripped. Pages are truncated to 3000 characters.',
    schema: z.object({
      url: z.string().url().describe('The full URL to fetch, e.g. "https://example.com"'),
    }),
  },
)

async function testWebFetcher() {
  console.log('=== Web Fetcher Tool ===\n')

  // Direct call — test the tool works before handing it to an agent
  const result = await webFetchTool.invoke({ url: 'https://httpbin.org/json' })
  console.log('Direct call result:\n%s\n', result)

  // Now let the LLM use it
  const { HumanMessage, ToolMessage } = await import('@langchain/core/messages')
  const llmWithTools = llm.bindTools([webFetchTool])

  const query = 'Fetch the page at https://httpbin.org/json and tell me what data it contains.'
  console.log('User: %s\n', query)

  const messages = [new HumanMessage(query)]
  const response = await llmWithTools.invoke(messages)
  messages.push(response)

  if (response.tool_calls?.length) {
    for (const call of response.tool_calls) {
      console.log('LLM wants to call: %s(%j)', call.name, call.args)
      const toolResult = await webFetchTool.invoke(call.args)
      console.log('Tool returned: %s\n', toolResult.slice(0, 200))
      messages.push(new ToolMessage({ content: toolResult, tool_call_id: call.id }))
    }

    const finalResponse = await llmWithTools.invoke(messages)
    console.log('Final answer:\n%s', finalResponse.content)
  } else {
    console.log('LLM answered directly: %s', response.content)
  }
}

// --- Tool 2: File Reader (sandboxed) ---

const ALLOWED_DIR = path.resolve('phase-4')
// const ALLOWED_DIR = path.resolve('phase-3/sample-docs')
const MAX_FILE_SIZE = 20_000

const fileReaderTool = tool(
  async ({ filename }) => {
    console.log('file reader tool:', filename)

    try {
      const filePath = path.resolve(ALLOWED_DIR, filename)

      // Path traversal guard — resolved path must stay inside ALLOWED_DIR
      if (!filePath.startsWith(ALLOWED_DIR)) {
        return `Access denied: path escapes the allowed directory.`
      }

      const stat = await fs.stat(filePath)
      if (!stat.isFile()) return `"${filename}" is not a file.`
      if (stat.size > MAX_FILE_SIZE) {
        return `File too large (${stat.size} bytes). Max is ${MAX_FILE_SIZE}.`
      }

      const content = await fs.readFile(filePath, 'utf-8')
      return `[File: ${filename}, ${content.length} chars]\n\n${content.slice(0, 200)}`
    } catch (err) {
      if (err.code === 'ENOENT') return `File not found: "${filename}"`
      return `Error reading file: ${err.message}`
    }
  },
  {
    name: 'read_file',
    description: 'Read a file by the provided path.' + 'Returns the full text content of the file.',
    schema: z.object({
      filename: z.string().describe('Name of the file to read, e.g. "nodejs-basics.md"'),
    }),
  },
)

async function testFileReader() {
  console.log('=== File Reader Tool ===\n')

  // Happy path
  const result = await fileReaderTool.invoke({ filename: 'nodejs-basics.md' })
  console.log('Read success:\n%s\n', result.slice(0, 300) + '...')

  // File not found
  const missing = await fileReaderTool.invoke({ filename: 'nope.txt' })
  console.log('Missing file: %s\n', missing)

  // Path traversal attempt — should be blocked
  const escape = await fileReaderTool.invoke({ filename: '../../.env' })
  console.log('Traversal attempt: %s', escape)
}

// --- Multi-Tool Agent ---

async function multiToolAgent() {
  console.log('=== Multi-Tool Agent ===\n')

  const { createAgent } = await import('langchain')

  const agent = createAgent({
    model: llm,
    tools: [webFetchTool, fileReaderTool],
    systemPrompt: 'You are a helpful research assistant. Use the available tools to answer questions. Be concise.',
  })

  const queries = [
    // Should use read_file
    'Read the file about nodejs and give me a 2-sentence summary.',
    // Should use web_fetch
    'Fetch https://httpbin.org/json and describe the data structure.',
    // Should use read_file — but the file doesn't exist. How does the agent handle it?
    'Read the file "1-tools.js" of this workspace and tell me about it.',
  ]

  for (const query of queries) {
    console.log('Q: %s', query)
    const result = await agent.invoke({ messages: [{ role: 'user', content: query }] })
    const answer = result.messages[result.messages.length - 1].content
    console.log('A: %s', answer)
    console.log('   (steps: %d messages)\n', result.messages.length)
  }
}

async function main() {
  // await testWebFetcher()
  // await testFileReader()
  await multiToolAgent()
}

main().catch(console.error)
