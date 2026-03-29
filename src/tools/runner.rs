//! Tool runner for agentic loops.
//!
//! The [`ToolRunner`] manages the conversation loop where:
//! 1. Messages are sent to the API
//! 2. If the model requests tool use, tools are executed
//! 3. Tool results are appended and the loop continues
//! 4. The loop ends when the model stops requesting tools

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info, warn};

use super::traits::ToolResultContent as TraitToolResultContent;
use super::{Tool, ToolResult};
use crate::client::{AdaptiveConfig, Client, Result as ClientResult};
use crate::types::{
    ContentBlock, ContentBlockParam, Message, MessageCreateParams, MessageParam, StopReason,
    ToolResultBlock, ToolResultContent,
};

/// Event emitted during tool execution.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Model produced text output.
    Text { text: String },
    /// A tool is about to be called.
    ToolCall {
        name: String,
        input: HashMap<String, serde_json::Value>,
    },
    /// A tool call completed.
    ToolResult {
        name: String,
        success: bool,
        /// The output from the tool.
        output: String,
        /// Optional structured metadata from the tool.
        metadata: Option<serde_json::Value>,
    },
    /// Token usage for this API call.
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        cache_read_tokens: u64,
        cache_creation_tokens: u64,
    },
}

/// Callback type for tool events.
pub type ToolEventCallback = Arc<dyn Fn(ToolEvent) + Send + Sync>;

/// Configuration for the tool runner.
#[derive(Clone)]
pub struct ToolRunnerConfig {
    /// Maximum number of iterations (API calls) before stopping.
    /// None means unlimited.
    pub max_iterations: Option<usize>,

    /// Whether to print debug information about tool calls.
    pub verbose: bool,

    /// Optional callback for tool events.
    pub on_event: Option<ToolEventCallback>,

    /// Adaptive timeout configuration for individual API calls.
    /// Uses half-exponential backoff (1.5x) on timeout/retry.
    pub adaptive_config: AdaptiveConfig,

    /// Cancellation flag — set to true to stop the runner between iterations.
    /// The runner checks this before each API call and tool execution.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for ToolRunnerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRunnerConfig")
            .field("max_iterations", &self.max_iterations)
            .field("verbose", &self.verbose)
            .field("on_event", &self.on_event.is_some())
            .field("adaptive_config", &self.adaptive_config)
            .finish()
    }
}

impl Default for ToolRunnerConfig {
    fn default() -> Self {
        Self {
            max_iterations: Some(50),
            verbose: false,
            on_event: None,
            adaptive_config: AdaptiveConfig::default(),
            cancel: None,
        }
    }
}

/// Runs an agentic loop with automatic tool execution.
///
/// The tool runner manages the conversation with the API, automatically
/// executing tools when requested and continuing until the model stops.
///
/// # Example
///
/// ```rust,no_run
/// use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
/// use llm_code_sdk::tools::{ToolRunner, FunctionTool, ToolRunnerConfig};
/// use llm_code_sdk::types::InputSchema;
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let client = Client::zai("your-api-key")?;
///
/// let read_file = Arc::new(FunctionTool::new(
///     "read_file",
///     "Read a file from disk",
///     InputSchema::object().required_string("path", "File path"),
///     |input| {
///         let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
///         std::fs::read_to_string(path).map_err(|e| e.to_string())
///     },
/// ));
///
/// let mut runner = ToolRunner::new(client, vec![read_file as Arc<dyn llm_code_sdk::tools::Tool>]);
///
/// let final_message = runner.run(MessageCreateParams {
///     model: "glm-4-plus".into(),
///     max_tokens: 4096,
///     messages: vec![MessageParam::user("Read the contents of /etc/hostname")],
///     ..Default::default()
/// }).await?;
///
/// println!("Final response: {:?}", final_message.text());
/// # Ok(())
/// # }
/// ```
pub struct ToolRunner {
    client: Client,
    tools: HashMap<String, Arc<dyn Tool>>,
    config: ToolRunnerConfig,
}

impl ToolRunner {
    /// Create a new tool runner with the given client and tools.
    pub fn new(client: Client, tools: Vec<Arc<dyn Tool>>) -> Self {
        let tools_map = tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();

        Self {
            client,
            tools: tools_map,
            config: ToolRunnerConfig::default(),
        }
    }

    /// Create a tool runner with custom configuration.
    pub fn with_config(
        client: Client,
        tools: Vec<Arc<dyn Tool>>,
        config: ToolRunnerConfig,
    ) -> Self {
        let tools_map = tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();

        Self {
            client,
            tools: tools_map,
            config,
        }
    }

    /// Run the agentic loop until completion.
    ///
    /// Returns the final message from the model (when it stops requesting tools).
    pub async fn run(&self, mut params: MessageCreateParams) -> ClientResult<Message> {
        // Add tool definitions to the request
        params.tools = self.tools.values().map(|t| t.to_param()).collect();

        let mut iteration = 0;
        let max_iterations = self.config.max_iterations.unwrap_or(usize::MAX);

        loop {
            // Check cancellation before each iteration
            if let Some(ref cancel) = self.config.cancel {
                if cancel.load(Ordering::SeqCst) {
                    return Err(crate::client::ClientError::Cancelled);
                }
            }

            if iteration >= max_iterations {
                warn!("Tool runner reached max iterations ({})", max_iterations);
                break;
            }

            iteration += 1;
            debug!("Tool runner iteration {}", iteration);

            // Make API call with adaptive timeout (half-exp backoff on stalls)
            let message = self
                .client
                .messages()
                .create_adaptive(&params, self.config.adaptive_config.clone())
                .await?;

            // Emit usage event
            if let Some(ref callback) = self.config.on_event {
                callback(ToolEvent::Usage {
                    input_tokens: message.usage.input_tokens,
                    output_tokens: message.usage.output_tokens,
                    cache_read_tokens: message.usage.cache_read_input_tokens.unwrap_or(0),
                    cache_creation_tokens: message.usage.cache_creation_input_tokens.unwrap_or(0),
                });
            }

            // Emit text events for any text content
            if let Some(ref callback) = self.config.on_event {
                for block in &message.content {
                    if let ContentBlock::Text(t) = block {
                        if !t.text.is_empty() {
                            callback(ToolEvent::Text {
                                text: t.text.clone(),
                            });
                        }
                    }
                }
            }

            if self.config.verbose {
                info!(
                    "Iteration {}: stop_reason={:?}, tool_uses={}",
                    iteration,
                    message.stop_reason,
                    message.tool_uses().len()
                );
            }

            // Check if we should continue
            if message.stop_reason != Some(StopReason::ToolUse) {
                debug!("Model stopped with reason: {:?}", message.stop_reason);
                return Ok(message);
            }

            // Check cancellation before tool execution
            if let Some(ref cancel) = self.config.cancel {
                if cancel.load(Ordering::SeqCst) {
                    return Err(crate::client::ClientError::Cancelled);
                }
            }

            // Execute tools and collect results
            let tool_results = self.execute_tools(&message).await;

            if tool_results.is_empty() {
                debug!("No tool results generated, stopping");
                return Ok(message);
            }

            // Append assistant message and tool results to conversation
            params.messages.push(MessageParam {
                role: "assistant".to_string(),
                content: crate::types::MessageContent::Blocks(
                    message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(t) => Some(ContentBlockParam::text(&t.text)),
                            ContentBlock::ToolUse(tu) => Some(ContentBlockParam::ToolUse {
                                id: tu.id.clone(),
                                name: tu.name.clone(),
                                input: tu.input.clone(),
                            }),
                            _ => None,
                        })
                        .collect(),
                ),
            });

            params.messages.push(MessageParam {
                role: "user".to_string(),
                content: crate::types::MessageContent::Blocks(
                    tool_results
                        .into_iter()
                        .map(|r| ContentBlockParam::ToolResult {
                            tool_use_id: r.tool_use_id,
                            content: r.content,
                            is_error: r.is_error,
                        })
                        .collect(),
                ),
            });

            // Compact old tool result payloads: keep metadata, drop bulk content.
            // Results from more than 2 iterations ago get their large payloads
            // replaced with a compact summary. The tool call structure, success/error
            // status, and key metadata are preserved — only the raw bulk data
            // (file contents, grep output, bash stdout) is released.
            if iteration > 2 {
                Self::compact_old_payloads(&mut params.messages, iteration);
            }
        }

        // This shouldn't happen, but return the last message if we hit max iterations
        // by making one more call without tools
        params.tools.clear();
        self.client
            .messages()
            .create_adaptive(&params, self.config.adaptive_config.clone())
            .await
    }

    async fn execute_tools(&self, message: &Message) -> Vec<ToolResultBlock> {
        let mut results = Vec::new();

        for tool_use in message.tool_uses() {
            // Emit tool call event
            if let Some(ref callback) = self.config.on_event {
                callback(ToolEvent::ToolCall {
                    name: tool_use.name.clone(),
                    input: tool_use.input.clone(),
                });
            }

            let (result, success, metadata) = if let Some(tool) = self.tools.get(&tool_use.name) {
                if self.config.verbose {
                    info!(
                        "Executing tool: {} with input: {:?}",
                        tool_use.name, tool_use.input
                    );
                }

                match tool.call(tool_use.input.clone()).await {
                    ToolResult::Success { output, metadata } => (
                        ToolResultBlock {
                            tool_use_id: tool_use.id.clone(),
                            content: Some(ToolResultContent::Text(output.into())),
                            is_error: false,
                        },
                        true,
                        metadata,
                    ),
                    ToolResult::Error(e) => {
                        warn!("Tool {} returned error: {}", tool_use.name, e);
                        (
                            ToolResultBlock {
                                tool_use_id: tool_use.id.clone(),
                                content: Some(ToolResultContent::Text(e.into())),
                                is_error: true,
                            },
                            false,
                            None,
                        )
                    }
                    ToolResult::Content(blocks) => {
                        // Convert to string for now
                        let text = blocks
                            .iter()
                            .filter_map(|b| match b {
                                TraitToolResultContent::Text(t) => Some(t.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        (
                            ToolResultBlock {
                                tool_use_id: tool_use.id.clone(),
                                content: Some(ToolResultContent::Text(text.into())),
                                is_error: false,
                            },
                            true,
                            None,
                        )
                    }
                }
            } else {
                warn!("Tool not found: {}", tool_use.name);
                (
                    ToolResultBlock {
                        tool_use_id: tool_use.id.clone(),
                        content: Some(ToolResultContent::Text(format!(
                            "Error: Tool '{}' not found",
                            tool_use.name
                        ).into())),
                        is_error: true,
                    },
                    false,
                    None,
                )
            };

            // Extract output text for the event
            let output_text: String = match &result.content {
                Some(ToolResultContent::Text(t)) => t.to_string(),
                Some(ToolResultContent::Blocks(blocks)) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        crate::types::ToolResultContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                None => String::new(),
            };

            // Emit tool result event
            if let Some(ref callback) = self.config.on_event {
                callback(ToolEvent::ToolResult {
                    name: tool_use.name.clone(),
                    success,
                    output: output_text,
                    metadata,
                });
            }

            results.push(result);
        }

        results
    }

    /// Get the list of available tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Compact bulk payloads in old tool results.
    /// Keeps the last `KEEP_RECENT` iteration-pairs intact.
    /// For older results, replaces content >THRESHOLD bytes with a compact summary
    /// that preserves the byte count so the model knows how much data was there.
    fn compact_old_payloads(messages: &mut [MessageParam], _current_iteration: usize) {
        const THRESHOLD: usize = 2048;
        const KEEP_RECENT_PAIRS: usize = 2; // keep last 2 assistant+user pairs

        let total = messages.len();
        if total <= KEEP_RECENT_PAIRS * 2 + 2 {
            return;
        }

        let compact_up_to = total.saturating_sub(KEEP_RECENT_PAIRS * 2);

        for msg in &mut messages[..compact_up_to] {
            if msg.role != "user" {
                continue;
            }
            if let crate::types::MessageContent::Blocks(blocks) = &mut msg.content {
                for block in blocks.iter_mut() {
                    if let ContentBlockParam::ToolResult { content: Some(content), is_error, .. } = block {
                        let len = match &*content {
                            ToolResultContent::Text(t) => t.len(),
                            ToolResultContent::Blocks(b) => b.iter().map(|b| match b {
                                crate::types::ToolResultContentBlock::Text { text } => text.len(),
                                _ => 0,
                            }).sum(),
                        };
                        if len > THRESHOLD {
                            // Extract first line as a hint of what this was
                            let hint = match &*content {
                                ToolResultContent::Text(t) => {
                                    let first = t.lines().next().unwrap_or("");
                                    if first.len() > 120 { format!("{}...", &first[..117]) } else { first.to_string() }
                                }
                                _ => String::new(),
                            };
                            let status = if *is_error { "error" } else { "ok" };
                            *content = ToolResultContent::Text(
                                format!("[{status}, {len} bytes] {hint}").into()
                            );
                        }
                    }
                }
            }
        }
    }

    /// Add a tool to the runner.
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Remove a tool from the runner.
    pub fn remove_tool(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.remove(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::FunctionTool;
    use crate::types::InputSchema;

    fn make_test_tool() -> Arc<dyn Tool> {
        Arc::new(FunctionTool::new(
            "test_tool",
            "A test tool",
            InputSchema::object().required_string("input", "Test input"),
            |input| {
                let val = input.get("input").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!("Got: {}", val))
            },
        ))
    }

    #[test]
    fn test_tool_runner_creation() {
        let client = Client::new("test-key").unwrap();
        let runner = ToolRunner::new(client, vec![make_test_tool()]);

        assert_eq!(runner.tool_names(), vec!["test_tool"]);
    }

    #[test]
    fn test_tool_runner_add_remove() {
        let client = Client::new("test-key").unwrap();
        let mut runner = ToolRunner::new(client, vec![]);

        assert!(runner.tool_names().is_empty());

        runner.add_tool(make_test_tool());
        assert_eq!(runner.tool_names(), vec!["test_tool"]);

        runner.remove_tool("test_tool");
        assert!(runner.tool_names().is_empty());
    }

    #[test]
    fn test_tool_runner_config() {
        let config = ToolRunnerConfig {
            max_iterations: Some(10),
            verbose: true,
            on_event: None,
            ..Default::default()
        };

        assert_eq!(config.max_iterations, Some(10));
        assert!(config.verbose);
    }
}
