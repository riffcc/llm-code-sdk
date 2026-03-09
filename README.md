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
- **LCS Module and Bin**: Drive DeepWiki-style model-driven repository understanding, sharpened by SmartRead and Hugo-backed live bases from inside the crate

### Optional Features

- `smart`: Five-layer code analysis (AST, Call Graph, CFG, DFG, PDG)
- `e2e`: Playwright integration for browser testing
- `tree-sitter-*`: Language-specific syntax parsing

## LCS Module and Bin

The crate now includes a first-cut `lcs` binary for DeepWiki-style model-driven repository understanding from the same semantic substrate that powers `smart_read`, `smart_write`, `ask_docs`, `ask_code`, and `mr_search`.

The intended shape is simple:

- model-driven structure generation
- model-driven page generation
- SmartRead-enhanced repository decomposition while those steps happen
- live Hugo bases kept hot by an `lcs` daemon
- the same substrate exposed back out through Palace Call

```bash
cargo run -p llm-code-sdk --bin lcs -- site /mnt/riffcastle/castle/garage/lcs
cargo run -p llm-code-sdk --bin lcs -- jetpack /home/wings/projects/jetpack /mnt/riffcastle/garage/jetpack-docs
cargo run -p llm-code-sdk --bin lcs -- daemon jetpack /home/wings/projects/jetpack /mnt/riffcastle/garage/jetpack-docs --poll-seconds 2
```

This is the current dogfood path for Living Code inside Palace. DeepWiki-open is the product-shape reference. The differentiator is that LCS gives the generator a Swiss army knife instead of only a chunked document pile. The deep semantic graph work comes next; the current cut proves the live documentation spine end to end while the imperative tool surface and the repo-scoped `lcs` daemon evolve around it.

`llm-code-sdk` already supports Anthropic-style integrations, so local models, Anthropic-compatible endpoints, OpenRouter, and Minimax-style Anthropic-compatible providers can all sit behind the same higher-level LCS flow through runtime configuration.

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
