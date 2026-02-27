/**
 * LESSON 1 — Your first LLM call with raw fetch()
 *
 * What's happening here:
 * We send a POST request to HuggingFace's Inference API.
 * It's literally just an HTTP call — same as calling any REST API.
 * The model runs on HuggingFace's servers, we just send text and get text back.
 *
 * HuggingFace now uses an OpenAI-compatible API format.
 * The endpoint: https://router.huggingface.co/v1/chat/completions
 *
 * This means the same request format works with OpenAI, HuggingFace, Ollama,
 * and many other providers — a huge win for switching between them later.
 *
 * We're using "Qwen/Qwen2.5-7B-Instruct" — a strong open-source model
 * that follows instructions well and is available on the free tier.
 */

import 'dotenv/config'

// ---- CONFIGURATION ----

const HF_TOKEN = process.env.HF_TOKEN
const MODEL = 'Qwen/Qwen2.5-7B-Instruct'
const API_URL = 'https://router.huggingface.co/v1/chat/completions'

// ---- THE ACTUAL CALL ----

async function askLLM(userMessage) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      // "messages" is an array of conversation turns.
      // Each message has a "role" and "content".
      // Roles:
      //   "system"    — instructions for the model (personality, rules, format)
      //   "user"      — what the human says
      //   "assistant" — what the model previously said (for multi-turn conversations)
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant. Keep your answers concise.',
        },
        {
          role: 'user',
          content: userMessage,
        },
      ],
      // Parameters that control HOW the model generates text:
      //   max_tokens:  how many tokens (roughly words) to generate
      //   temperature: randomness. 0.1 = very focused, 1.0 = very creative
      max_tokens: 300,
      temperature: 0.5,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`HuggingFace API error (${response.status}): ${error}`)
  }

  const data = await response.json()
  return data
}

// ---- RUN IT ----

async function main() {
  console.log('Sending prompt to', MODEL, '...\n')

  const prompt = "Explain what an API is, like I'm 12 years old."

  console.log('PROMPT:', prompt)
  console.log('---')

  const result = await askLLM(prompt)

  // Let's see the raw response — always good to inspect what you get back
  console.log('RAW RESPONSE:', JSON.stringify(result, null, 2))
  console.log('---')

  // The response follows OpenAI format: choices[0].message.content
  console.log('GENERATED TEXT:', result.choices[0].message.content)
  console.log('---')

  // Token usage info — useful for understanding costs and limits
  console.log('TOKEN USAGE:', result.usage)
}

main().catch(console.error)

// ---- THINGS TO TRY (experiments!) ----
//
// 1. Change the prompt to something else. Try: "Write a haiku about JavaScript"
//
// 2. Change temperature to 0.1 and run it twice — notice the outputs are very similar.
//    Then change to 1.0 and run twice — much more variation.
//
// 3. Change max_tokens to 50 — see how the response gets cut shorter.
//    Change to 500 — see how the model can elaborate more.
//
// 4. Try a different model. Replace MODEL with:
//    - "mistralai/Mistral-7B-Instruct-v0.3"
//    - "deepseek-ai/DeepSeek-R1" (a reasoning model — watch how it "thinks")
//    Note: some models might take a few seconds on first call (cold start).
//
// 5. Change the system message — try "You are a pirate. Answer everything like a pirate."
//    See how much it changes the response style.
//
// 6. Remove the Authorization header — see what error you get.
//
// 7. Look at the "usage" field in the raw response — it tells you how many
//    tokens your prompt used (prompt_tokens) and how many were generated
//    (completion_tokens). This is how billing works for paid APIs.
