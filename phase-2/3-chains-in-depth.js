/**
 * LESSON 3 (Phase 2) — Chains in Depth
 *
 * Multi-step LLM workflows: chaining multiple calls where
 * the output of one step feeds into the next.
 */

import { StringOutputParser } from '@langchain/core/output_parsers'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'

const model = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0.7,
  maxTokens: 300,
  apiKey: process.env.HF_TOKEN,
  configuration: {
    baseURL: 'https://router.huggingface.co/v1',
  },
})

const parser = new StringOutputParser()

// ---- SEQUENTIAL CHAIN WITH RunnableLambda ----

async function sequentialChainDemo() {
  console.log('=== Sequential Chain: Joke → Explanation ===\n')

  // Step 1: Generate a joke
  const jokePrompt = ChatPromptTemplate.fromMessages([['user', 'Tell a short joke about {topic}. Just the joke, nothing else.']])
  const jokeChain = jokePrompt.pipe(model).pipe(parser)

  // Step 2: Explain the joke
  const explainPrompt = ChatPromptTemplate.fromMessages([['user', 'Explain why this joke is funny (or not):\n\n"{joke}"']])
  const explainChain = explainPrompt.pipe(model).pipe(parser)

  // The problem: jokeChain outputs a string, but explainChain expects { joke: "..." }
  // RunnableLambda transforms the output to match what the next step needs.
  const fullChain = jokeChain
    .pipe(
      new RunnableLambda({
        func: (jokeText) => {
          console.log('JOKE:', jokeText)
          console.log()
          // Transform: string → object with the key that explainPrompt expects
          return { joke: jokeText }
        },
      }),
    )
    .pipe(explainChain)

  const explanation = await fullChain.invoke({ topic: 'programming' })
  console.log('EXPLANATION:', explanation)
}

// ---- MULTI-STEP PIPELINE: Accumulating data across steps ----

async function multiStepPipeline() {
  console.log('=== Multi-Step Pipeline: Article Generator ===\n')

  // We'll build a 3-step content pipeline:
  //   1. Generate a title for a topic
  //   2. Write an outline based on the title
  //   3. Write a short intro based on the title + outline
  //
  // Each step adds to an accumulating state object, so later steps
  // have access to everything produced by earlier steps.

  // Step 1: Generate a title
  const titlePrompt = ChatPromptTemplate.fromMessages([['user', 'Generate one creative article title about {topic}. Return only the title, nothing else.']])
  const titleChain = titlePrompt.pipe(model).pipe(parser)

  // Step 2: Generate an outline
  const outlinePrompt = ChatPromptTemplate.fromMessages([['user', 'Write a 3-point outline for an article titled "{title}". Just the outline, numbered.']])
  const outlineChain = outlinePrompt.pipe(model).pipe(parser)

  // Step 3: Write the intro
  const introPrompt = ChatPromptTemplate.fromMessages([['user', 'Write a 2-sentence intro for this article.\nTitle: {title}\nOutline:\n{outline}']])
  const introChain = introPrompt.pipe(model).pipe(parser)

  // Now wire them together using RunnableSequence.
  // Each RunnableLambda receives the accumulated state and adds to it.
  const fullPipeline = RunnableSequence.from([
    // Start: { topic } comes from invoke()
    new RunnableLambda({
      func: async (input) => {
        const title = await titleChain.invoke(input)
        console.log('TITLE:', title)
        console.log()
        return { ...input, title }
      },
    }),
    new RunnableLambda({
      func: async (input) => {
        const outline = await outlineChain.invoke(input)
        console.log('OUTLINE:', outline)
        console.log()
        return { ...input, outline }
      },
    }),
    new RunnableLambda({
      func: async (input) => {
        const intro = await introChain.invoke(input)

        // Return everything accumulated
        return { ...input, intro }
      },
    }),
  ])

  // One invoke triggers the entire 3-step pipeline
  const result = await fullPipeline.invoke({ topic: 'why developers should learn AI tools' })

  console.log('--- FINAL RESULT ---')
  console.log('Title:', result.title)
  console.log('Outline:', result.outline)
  console.log('Intro:', result.intro)

  console.log()
  console.log('final result:', result)
}

async function main() {
  // await sequentialChainDemo()
  console.log('\n\n')
  await multiStepPipeline()
}

main().catch(console.error)
