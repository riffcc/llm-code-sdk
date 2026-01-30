//! # LLM Code SDK
//!
//! A Rust SDK for LLM APIs with agentic tool use capabilities.
//!
//! This SDK is inspired by and derives patterns from
//! [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) (MIT Licensed)
//! by Anthropic, PBC.
//!
//! ## Features
//!
//! - **Messages API**: Create messages with text, images, and documents
//! - **Streaming**: Real-time streaming responses via SSE
//! - **Tool Use**: Define and execute tools in an agentic loop
//! - **Extended Thinking**: Enable Claude's reasoning process
//! - **Token Counting**: Count tokens before sending requests
//!
//! ## Example
//!
//! ```rust,no_run
//! use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = Client::new("your-api-key")?;
//!
//!     let message = client.messages().create(MessageCreateParams {
//!         model: "glm-4-plus".into(),
//!         max_tokens: 1024,
//!         messages: vec![MessageParam::user("Hello, Claude!")],
//!         ..Default::default()
//!     }).await?;
//!
//!     println!("{:?}", message.text());
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming Example
//!
//! ```rust,no_run
//! use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
//! use tokio_stream::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = Client::new("your-api-key")?;
//!
//!     let mut stream = client.messages().stream(MessageCreateParams {
//!         model: "glm-4-plus".into(),
//!         max_tokens: 1024,
//!         messages: vec![MessageParam::user("Tell me a story")],
//!         ..Default::default()
//!     }).await?;
//!
//!     while let Some(event) = stream.next().await {
//!         if let llm_code_sdk::streaming::StreamEvent::Text { text, .. } = event {
//!             print!("{}", text);
//!         }
//!     }
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod skills;
pub mod streaming;
pub mod tools;
pub mod types;

#[cfg(feature = "e2e")]
pub mod e2e;

pub use client::{
    global_throughput, AdaptiveConfig, AdaptiveStreamConfig, ApiFormat, Client, ClientBuilder,
    ThroughputConfig, ThroughputStats, ThroughputTracker,
};
pub use error::{ApiError, Error, Result};
pub use skills::{
    BetaFeature, Container, LocalSkill, Skill, SkillDefinition, SkillRef, SkillStack,
    SkillType, SkillVersion,
};
pub use streaming::{MessageStream, RawStreamEvent, StreamEvent};
pub use tools::{
    create_editing_tools, create_exploration_tools, BashTool, EditFileTool, FunctionTool,
    GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, Tool, ToolRunner, ToolRunnerConfig,
    WriteFileTool,
};
pub use types::*;
