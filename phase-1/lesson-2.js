/**
 * LESSON 2 — HuggingFace.js SDK (@huggingface/inference)
 *
 * Same thing as Lesson 1 (talking to LLMs), but using the official SDK
 * instead of raw fetch(). Less boilerplate, better errors, built-in streaming.
 */

import { InferenceClient } from '@huggingface/inference'
import 'dotenv/config'

// One line replaces all the URL + headers setup from lesson-1.
// The client stores your token and knows the correct endpoints.
const client = new InferenceClient(process.env.HF_TOKEN)

// ---- CHAT COMPLETION ----

async function basicChat() {
  console.log('=== Basic Chat Completion ===\n')

  const response = await client.chatCompletion({
    model: 'Qwen/Qwen2.5-7B-Instruct',
    messages: [
      { role: 'system', content: 'You are a helpful assistant. Keep answers concise.' },
      { role: 'user', content: 'What is the difference between an API and an SDK?' },
    ],
    max_tokens: 400,
    temperature: 0.7,
  })

  // Same OpenAI-compatible response shape as lesson-1
  console.log('Response:', response.choices[0].message.content)
  console.log()
  console.log('Tokens used:', response.usage)
  console.log()
}

// ---- STREAMING ----

async function streamingChat() {
  console.log('=== Streaming Chat ===\n')

  const stream = client.chatCompletionStream({
    model: 'Qwen/Qwen2.5-7B-Instruct',
    messages: [{ role: 'user', content: 'Count from 1 to 10 slowly, with a fun fact for each number. Keep each count short enough to be said in one breath.' }],
    max_tokens: 300,
    temperature: 0.7,
  })

  // Each chunk contains a small piece of the response (usually 1 token).
  // chunk.choices[0].delta.content is the text fragment.
  // We use process.stdout.write instead of console.log to avoid newlines
  // between chunks — so the text flows naturally like in ChatGPT.
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content
    if (text) {
      process.stdout.write(text)
    }
  }

  console.log('\n\n--- Stream complete ---\n')
}

// ---- SUMMARIZATION (a different task type) ----

async function summarize() {
  console.log('=== Summarization ===\n')

  const longText = `
    The Apollo 11 mission was the first manned mission to land on the Moon.
    It was launched on July 16, 1969, carrying astronauts Neil Armstrong,
    Buzz Aldrin, and Michael Collins. Armstrong and Aldrin landed the lunar
    module Eagle on the Moon's surface on July 20, while Collins orbited
    above in the command module Columbia. Armstrong became the first human
    to step onto the Moon's surface, followed by Aldrin. They spent about
    two and a quarter hours outside the spacecraft, collecting samples and
    taking photographs. The mission returned safely to Earth on July 24,
    splashing down in the Pacific Ocean. The success of Apollo 11 fulfilled
    President Kennedy's goal of landing a man on the Moon before the end
    of the 1960s decade.
  `.trim()

  console.log('INPUT TEXT:', longText.slice(0, 100) + '...\n')

  // summarization() calls a model specifically trained for summarization.
  // Unlike chat models, it takes raw text in and returns a summary — no
  // messages array, no roles, no system prompt.
  //
  // "facebook/bart-large-cnn" is a classic summarization model trained on
  // CNN/DailyMail news articles.
  //
  // Note: dedicated models on HuggingFace free tier can be "cold" — not
  // loaded in memory. First call may take 20-60 seconds while the model boots.
  // The SDK has retry_on_error: true by default, which retries on 503 (loading).

  console.log('(this may take a while if the model is cold-starting...)\n')

  const result = await client.summarization({
    model: 'facebook/bart-large-cnn',
    inputs: longText,
  })

  // Response is simple: just { summary_text: "..." }
  console.log('SUMMARY:', result.summary_text)
  console.log()
}

// Run demos sequentially
async function main() {
  // await basicChat()
  // await streamingChat()
  await summarize()
}

main().catch(console.error)
