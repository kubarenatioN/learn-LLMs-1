# Node.js Basics

Node.js is a JavaScript runtime built on Chrome's V8 engine. It allows developers to run JavaScript outside the browser, primarily for building server-side applications.

## Event Loop

The event loop is the core of Node.js's non-blocking architecture. Instead of creating a new thread for each request (like traditional servers), Node.js uses a single thread with an event loop that handles multiple connections concurrently. This makes it highly efficient for I/O-heavy workloads like web servers and APIs.

## npm (Node Package Manager)

npm is the default package manager for Node.js. It hosts over 2 million packages, making it the largest software registry in the world. Common commands include `npm install` to add dependencies, `npm init` to create a new project, and `npm run` to execute scripts defined in package.json.

## Common Use Cases

Node.js excels at:
- REST APIs and GraphQL servers
- Real-time applications (chat, gaming, collaboration tools) using WebSockets
- Microservices architectures
- CLI tools and build systems
- Server-side rendering (SSR) for React, Vue, and other frameworks

## Express.js

Express is the most popular Node.js web framework. It provides routing, middleware support, and a simple API for handling HTTP requests. A basic Express server can be set up in just a few lines of code. Many other frameworks like Nest.js and Koa.js build on top of Express concepts.

## Streams

Streams are a powerful feature for handling large amounts of data efficiently. Instead of loading an entire file into memory, streams process data in chunks. There are four types: Readable, Writable, Duplex, and Transform streams. This is especially useful for file processing, HTTP responses, and data transformation pipelines.
