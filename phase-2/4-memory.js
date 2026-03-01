/**
 * LESSON 4 (Phase 2) — Memory
 *
 * Making chatbots that remember conversation history.
 * LLMs are stateless — memory is just "send all previous messages every time."
 */

import { InMemoryChatMessageHistory } from '@langchain/core/chat_history'
import { AIMessage, HumanMessage, SystemMessage, trimMessages } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import 'dotenv/config'

const model = new ChatOpenAI({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0.7,
  maxTokens: 200,
  apiKey: process.env.HF_TOKEN,
  configuration: {
    baseURL: 'https://router.huggingface.co/v1',
  },
})

// ---- MANUAL MEMORY: the array approach ----

async function manualMemory() {
  console.log('=== Manual Memory ===\n')

  // The "memory" is just an array of messages that we maintain ourselves.
  const history = [new SystemMessage('You are a helpful travel assistant. Be concise.')]

  // Helper: send a message, get a response, store both in history
  async function chat(userText) {
    history.push(new HumanMessage(userText))

    // We send the ENTIRE history every time — that's how the model "remembers"
    const response = await model.invoke(history)

    history.push(new AIMessage(response.content))
    return response.content
  }

  // Multi-turn conversation — each call sees all previous messages
  console.log('User: I want to visit Japan next spring.')
  console.log('AI:', await chat('I want to visit Japan next spring.'))
  console.log()

  console.log('User: What cities do you recommend?')
  console.log('AI:', await chat('What cities do you recommend?'))
  console.log()

  // This question only makes sense if the model remembers the context
  console.log('User: Which one is best for cherry blossoms?')
  console.log('AI:', await chat('Which one is best for cherry blossoms?'))
  console.log()

  // Let's see what the history looks like now
  console.log('--- Conversation history (%d messages) ---', history.length)
  for (const msg of history) {
    console.log(`  [${msg._getType()}]: ${msg.content.slice(0, 80)}...`)
  }
}

// ---- LANGCHAIN MEMORY: ChatMessageHistory + trimMessages ----

async function langchainMemory() {
  console.log('=== LangChain Memory with trimMessages ===\n')

  // InMemoryChatMessageHistory — stores messages in memory (resets on restart).
  // In a real app you'd use a persistent store (database, Redis, etc.)
  const history = new InMemoryChatMessageHistory()

  const systemMsg = new SystemMessage('You are a helpful cooking assistant. Keep answers to 2-3 sentences.')

  async function chat(userText) {
    await history.addMessage(new HumanMessage(userText))

    // Get all stored messages
    const allMessages = await history.getMessages()

    // trimMessages keeps only the most recent messages that fit within a token budget.
    // This is the "window memory" strategy — like a sliding window over the conversation.
    const trimmed = await trimMessages(allMessages, {
      maxTokens: 300,
      tokenCounter: (msgs) => msgs.reduce((sum, m) => sum + m.content.length / 4, 0),
      strategy: 'last', // keep the LAST messages (most recent)
      startOn: 'human', // always start the window on a human message
      includeSystem: true, // always keep the system message
    })

    // Prepend system message + send trimmed history
    const response = await model.invoke([systemMsg, ...trimmed])

    await history.addMessage(new AIMessage(response.content))
    return response.content
  }

  console.log('User: How do I make pasta from scratch?')
  console.log('AI:', await chat('How do I make pasta from scratch?'))
  console.log()

  console.log('User: What flour should I use?')
  console.log('AI:', await chat('What flour should I use?'))
  console.log()

  console.log('User: How long do I knead it?')
  console.log('AI:', await chat('How long do I knead it?'))
  console.log()

  console.log('User: What sauce goes well with fresh pasta?')
  console.log('AI:', await chat('What sauce goes well with fresh pasta?'))
  console.log()

  // Show what's in the full history vs what gets trimmed
  const allMsgs = await history.getMessages()
  const trimmedMsgs = await trimMessages(allMsgs, {
    maxTokens: 300,
    tokenCounter: (msgs) => msgs.reduce((sum, m) => sum + m.content.length / 4, 0),
    strategy: 'last',
    startOn: 'human',
    includeSystem: true,
  })

  console.log('--- Full history: %d messages ---', allMsgs.length)
  console.log('--- After trimming: %d messages ---', trimmedMsgs.length)
  console.log('--- Trimmed keeps only the most recent exchanges ---')
}

async function main() {
  // await manualMemory()
  console.log('\n\n')
  await langchainMemory()
}

main().catch(console.error)
