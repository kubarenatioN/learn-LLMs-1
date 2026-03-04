# LangChain Overview

LangChain is a framework for developing applications powered by large language models (LLMs). It provides tools and abstractions to make it easier to build complex AI applications.

## Core Concepts

### Models
LangChain supports multiple LLM providers including OpenAI, HuggingFace, Anthropic, and local models via Ollama. The framework wraps these providers in a unified interface so you can swap models without changing your application code.

### Prompt Templates
Instead of hardcoding prompts, LangChain uses templates with variables. This makes prompts reusable and composable. Templates can include system messages, few-shot examples, and dynamic content.

### Chains
Chains connect multiple processing steps into a pipeline. A simple chain might be: format prompt → call LLM → parse output. More complex chains can branch, loop, or accumulate state across steps.

### Memory
LLMs are stateless — they don't remember previous conversations. Memory components maintain conversation history and inject it into prompts. Strategies include full history, sliding window, and summary memory.

## RAG (Retrieval-Augmented Generation)

RAG is the most common pattern for building AI apps that work with your own data. Instead of fine-tuning a model, you retrieve relevant documents at query time and include them in the prompt.

### How RAG Works
1. Load your documents (PDFs, markdown, web pages, etc.)
2. Split them into smaller chunks
3. Create embeddings (vector representations) for each chunk
4. Store embeddings in a vector database
5. When a user asks a question, embed the query
6. Find the most similar document chunks
7. Include those chunks in the LLM prompt as context
8. The LLM generates an answer based on the retrieved context

### Why RAG Over Fine-Tuning
- Much cheaper — no GPU training costs
- Easy to update — just add/remove documents
- Transparent — you can see which documents were used
- No catastrophic forgetting — the base model stays intact

## Agents

Agents are LLM-powered decision makers. Unlike chains where the steps are predetermined, agents use the LLM to decide which actions to take. They have access to tools (web search, calculators, APIs, databases) and can call them in any order based on the task.

### Agent Loop
1. Receive user input
2. LLM decides which tool to use (or whether to respond directly)
3. Execute the tool and observe the result
4. LLM decides next step based on the observation
5. Repeat until the task is complete

Agents are powerful but less predictable than chains. Use chains when you know the exact steps; use agents when the task requires dynamic decision-making.
