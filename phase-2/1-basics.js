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
import { StringOutputParser } from '@langchain/core/output_parsers'
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
    console.log(`  [${msg.type}]: ${msg.content}`)
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

// ---- CHAINING WITH .pipe() ----

async function chainingDemo() {
  console.log('=== Chaining with .pipe() ===\n')

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a naming consultant. Suggest 3 creative names.'],
    ['user', 'Suggest names for a {business} in {city}.'],
  ])

  // .pipe() connects components: prompt → model
  // The output of prompt.invoke() automatically feeds into model.invoke()
  const chain = prompt.pipe(model)

  // Now invoke the CHAIN, not the individual parts.
  // You pass the template variables, and it flows through: template → model
  const response = await chain.invoke({
    business: 'coffee shop',
    city: 'Tokyo',
  })

  console.log('Result:', response.content)
  console.log()

  // Same chain, different input — the whole pipeline reuses
  const response2 = await chain.invoke({
    business: 'bookstore',
    city: 'Paris',
  })

  console.log('Same chain, different input:', response2.content)
}

// ---- OUTPUT PARSERS ----

async function outputParserDemo() {
  console.log('=== Output Parsers ===\n')

  // Without a parser, chain.invoke() returns an AIMessage object.
  // With StringOutputParser, it extracts just the text string.
  // Seems minor, but it matters when you pipe output into the NEXT step.
  const parser = new StringOutputParser()

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'Respond with ONLY valid JSON, no markdown, no explanation.'],
    ['user', 'Give me 3 names for a {business} in {city}. Format: {{"names": ["name1", "name2", "name3"]}}'],
  ])

  // Three-step chain: template → model → parser
  const chain = prompt.pipe(model).pipe(parser)

  // Now the result is a plain string, not an AIMessage
  const result = await chain.invoke({
    business: 'pizza place',
    city: 'New York',
  })

  console.log('Raw result (string):', result)
  console.log('Type:', typeof result)
  console.log()

  // Since it's supposed to be JSON, we can parse it
  try {
    const parsed = JSON.parse(result)
    console.log('Parsed JSON:', parsed)
    console.log('First name:', parsed.names[0])
  } catch {
    // LLMs don't always follow instructions perfectly!
    console.log('Failed to parse as JSON — the model didnt follow the format.')
    console.log('This is a real problem we will solve properly in the next lesson.')
  }
}

async function main() {
  // await basicCall()
  // await promptTemplateDemo()
  // await chainingDemo()
  await outputParserDemo()
}

main().catch(console.error)
