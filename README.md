# llm-code-sdk

Rust SDK for LLM APIs with agentic tool use.
Inspired by Anthropic's Python SDK, built for Palace.

## Purpose

Provides a unified interface to LLM APIs with first-class support
for tool use, streaming, and code intelligence features.

## Features

- **Messages API**: Chat completions with system/user/assistant messages
- **Streaming**: Real-time token streaming with async iterators
- **Tool Use**: Define tools, execute them, feed results back
- **Extended Thinking**: Support for reasoning traces
- **Token Counting**: Estimate usage before sending

### Optional Features

- `smart`: Five-layer code analysis (AST, Call Graph, CFG, DFG, PDG)
- `e2e`: Playwright integration for browser testing
- `tree-sitter-*`: Language-specific syntax parsing

## Smart Tools (5-Layer Analysis)

When `smart` feature is enabled:

```
Layer 1: AST        - Syntax structure (functions, types)
Layer 2: Call Graph - Who calls whom
Layer 3: CFG        - Control flow, cyclomatic complexity
Layer 4: DFG        - Data flow, variable def-use chains
Layer 5: PDG        - Program dependence, slicing
```

### Key Components

- **SmartRead**: Token-efficient code reading (60-95% savings)
- **SmartWrite**: Structural edits with complexity analysis
- **ChangeTransaction**: Staged commits with validation gates
- **CodeQuery**: Granular, composable code queries

## Usage

```rust
use llm_code_sdk::{Client, Message};

let client = Client::new("http://localhost:1234/v1");

let response = client
    .messages()
    .model("glm-4.7-flash")
    .system("You are a helpful assistant.")
    .user("Hello!")
    .send()
    .await?;
```

## License

AGPL-3.0
