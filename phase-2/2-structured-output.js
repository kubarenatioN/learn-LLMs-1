/**
 * LESSON 2 (Phase 2) — Structured Output
 *
 * Making LLMs return reliable, parseable data instead of freeform text.
 * Uses Zod schemas + LangChain's .withStructuredOutput() method.
 */

import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'
import { z } from 'zod'

const model = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0.7,
  maxTokens: 300,
  apiKey: process.env.HF_TOKEN,
  configuration: {
    baseURL: 'https://router.huggingface.co/v1',
  },
})

// ---- DEFINING A SCHEMA ----

// A Zod schema describes the SHAPE of data you want back from the model.
// Think of it as a contract: "I want an object with these fields, these types."
// The .describe() calls are important — they tell the model what each field means.

const movieReviewSchema = z.object({
  title: z.string().describe('The title of the movie'),
  rating: z.number().min(1).max(10).describe('Rating from 1 to 10'),
  mainHeroes: z.array(z.string()).describe('List of main characters'),
  pros: z.array(z.string()).describe('List of positive aspects'),
  cons: z.array(z.string()).describe('List of negative aspects'),
  oneLinerSummary: z.string().describe('A one-sentence summary of the review'),
})

// ---- STRUCTURED OUTPUT ----

async function structuredDemo() {
  console.log('=== Structured Output ===\n')

  // .withStructuredOutput() returns a NEW model that always outputs
  // data matching the schema. Under the hood, it sends the schema
  // to the API as a "response_format" instruction.
  const structuredModel = model.withStructuredOutput(movieReviewSchema)

  // Now invoke just like before — but the result is a JS object, not an AIMessage
  const result = await structuredModel.invoke('Review the movie "Inception" by Christopher Nolan.')

  // result is already a parsed object matching our schema!
  console.log('Full result:', JSON.stringify(result, null, 2))
  console.log()

  // Access fields directly — no JSON.parse(), no string parsing
  console.log('Title:', result.title)
  console.log('Rating:', result.rating, '/ 10')
  console.log('Pros:', result.pros.join(', '))
  console.log('Cons:', result.cons.join(', '))
  console.log('Summary:', result.oneLinerSummary)
}

// ---- STRUCTURED OUTPUT IN A CHAIN ----

async function structuredChainDemo() {
  console.log('=== Structured Output in a Chain ===\n')

  // Different schema — analyzing a tech concept
  const conceptSchema = z.object({
    name: z.string().describe('Name of the concept'),
    difficulty: z.enum(['beginner', 'intermediate', 'advanced']).describe('Difficulty level'),
    prerequisites: z.array(z.string()).describe('Things you should know first'),
    useCases: z.array(z.string()).describe('Real-world use cases'),
    timeToLearn: z.string().describe('Estimated time to learn the basics'),
  })

  // Notice: .withStructuredOutput() works with .pipe() like any other model
  const { ChatPromptTemplate } = await import('@langchain/core/prompts')

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a tech educator. Analyze the given concept for someone learning {language}.'],
    ['user', 'Analyze this concept: {concept}'],
  ])

  const chain = prompt.pipe(model.withStructuredOutput(conceptSchema))

  const result = await chain.invoke({
    language: 'JavaScript',
    concept: 'Promises and async/await',
  })

  console.log('Concept:', result.name)
  console.log('Difficulty:', result.difficulty)
  console.log('Prerequisites:', result.prerequisites)
  console.log('Use cases:', result.useCases)
  console.log('Time to learn:', result.timeToLearn)
}

async function main() {
  // await structuredDemo()
  console.log('\n')
  await structuredChainDemo()
}

main().catch(console.error)
