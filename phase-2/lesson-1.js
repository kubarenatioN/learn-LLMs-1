/**
 * LESSON 1 (Phase 2) — LangChain.js Setup & First Steps
 *
 * Connecting LangChain to HuggingFace and understanding
 * the difference between using the SDK directly vs through LangChain.
 *
 * KEY INSIGHT: Since HuggingFace now uses the OpenAI-compatible API format
 * (remember router.huggingface.co/v1/chat/completions from Phase 1?),
 * we use LangChain's ChatOpenAI class and point it at HuggingFace's URL.
 * This is a very common pattern — many providers are OpenAI-compatible.
 */

import { ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'

// Compare this to Phase 1:
//   Phase 1: new InferenceClient(token) + manual chatCompletion() calls
//   Phase 2: ChatOpenAI configured once, then .invoke() everywhere
//
// We point ChatOpenAI at HuggingFace's router using configuration.baseURL.
// Same trick works for Ollama, Together AI, Groq, or any OpenAI-compatible provider.
const model = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0.7,
  maxTokens: 200,
  apiKey: process.env.HF_TOKEN,
  configuration: {
    baseURL: 'https://router.huggingface.co/v1',
  },
})

// ---- BASIC INVOCATION ----

async function basicCall() {
  console.log('=== Basic LangChain Call ===\n')

  // .invoke() is THE universal LangChain method. Every component has it.
  // Pass a string — LangChain wraps it as a HumanMessage automatically.
  const response = await model.invoke('What is LangChain in one paragraph, but like Im a 10 y.o. kid?')

  // The response is NOT raw JSON like in Phase 1.
  // It's an AIMessage object — a LangChain abstraction.
  console.log('Full response object type:', response.constructor.name)
  console.log()

  // The actual text is in .content
  console.log('Just the text:', response.content)
  console.log()

  // Token usage in response_metadata
  console.log('Token usage:', response.usage_metadata)
}

// ---- PROMPT TEMPLATES ----

async function promptTemplateDemo() {
  console.log('=== Prompt Templates ===\n')

  // A template with variables in curly braces.
  // This creates a reusable prompt structure — you define it once,
  // then fill in the blanks with different values each time.
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are an expert in {topic}. Explain things clearly and concisely.'],
    ['user', '{question}'],
  ])

  // .invoke() on a template fills in the variables and returns a formatted
  // message array — ready to be sent to a model.
  const formatted = await prompt.invoke({
    topic: 'databases',
    question: 'What is the difference between SQL and NoSQL?',
  })

  // Let's see what the template produced before sending to the model
  console.log('Formatted messages:')
  for (const msg of formatted.messages) {
    console.log(`  [${msg._getType()}]: ${msg.content}`)
  }
  console.log()

  // Now send the formatted prompt to the model
  const response = await model.invoke(formatted)
  console.log('Response:', response.content)
  console.log()

  // The power: same template, different variables — reuse it!
  const formatted2 = await prompt.invoke({
    topic: 'JavaScript',
    question: 'When should I use Map instead of a plain object?',
  })

  const response2 = await model.invoke(formatted2)
  console.log('Same template, different topic:', response2.content)
}

async function main() {
  // await basicCall()
  await promptTemplateDemo()
}

main().catch(console.error)
