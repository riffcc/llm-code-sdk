//! OpenAI-compatible API types.
//!
//! These types are used for OpenAI Chat Completions API format,
//! compatible with LM Studio, Ollama, and other local LLM servers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ContentBlock, Message, MessageCreateParams, StopReason, TextBlock, ToolUseBlock, Usage};

/// OpenAI chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// OpenAI message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// OpenAI function definition.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// OpenAI chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChatResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created: Option<i64>,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
}

/// OpenAI choice.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChoice {
    pub index: i32,
    pub message: OpenAIMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// OpenAI usage.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

impl From<&MessageCreateParams> for OpenAIChatRequest {
    fn from(params: &MessageCreateParams) -> Self {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(system) = &params.system {
            let content = match system {
                super::SystemPrompt::Text(s) => s.clone(),
                super::SystemPrompt::Blocks(blocks) => {
                    blocks.iter().map(|b| b.text.clone()).collect::<Vec<_>>().join("\n")
                }
            };
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &params.messages {
            match &msg.content {
                super::MessageContent::Text(text) => {
                    messages.push(OpenAIMessage {
                        role: msg.role.clone(),
                        content: Some(text.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                super::MessageContent::Blocks(blocks) => {
                    // Collect text and tool results/uses
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();

                    for block in blocks {
                        match block {
                            super::ContentBlockParam::Text { text, .. } => {
                                text_parts.push(text.clone());
                            }
                            super::ContentBlockParam::ToolResult { tool_use_id, content, .. } => {
                                // Tool results go as separate messages
                                let result_text = match content {
                                    Some(super::ToolResultContent::Text(t)) => t.clone(),
                                    Some(super::ToolResultContent::Blocks(blocks)) => {
                                        blocks.iter().filter_map(|b| {
                                            if let super::ToolResultContentBlock::Text { text } = b {
                                                Some(text.clone())
                                            } else {
                                                None
                                            }
                                        }).collect::<Vec<_>>().join("\n")
                                    }
                                    None => String::new(),
                                };
                                messages.push(OpenAIMessage {
                                    role: "tool".to_string(),
                                    content: Some(result_text),
                                    tool_calls: None,
                                    tool_call_id: Some(tool_use_id.clone()),
                                });
                            }
                            super::ContentBlockParam::ToolUse { id, name, input } => {
                                tool_calls.push(OpenAIToolCall {
                                    id: id.clone(),
                                    call_type: "function".to_string(),
                                    function: OpenAIFunctionCall {
                                        name: name.clone(),
                                        arguments: serde_json::to_string(input).unwrap_or_default(),
                                    },
                                });
                            }
                            _ => {}
                        }
                    }

                    if !text_parts.is_empty() || !tool_calls.is_empty() {
                        messages.push(OpenAIMessage {
                            role: msg.role.clone(),
                            content: if text_parts.is_empty() { None } else { Some(text_parts.join("\n")) },
                            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                            tool_call_id: None,
                        });
                    }
                }
            }
        }

        // Convert tools
        let tools = if params.tools.is_empty() {
            None
        } else {
            Some(params.tools.iter().map(|t| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: serde_json::to_value(&t.input_schema).unwrap_or_default(),
                },
            }).collect())
        };

        OpenAIChatRequest {
            model: params.model.clone(),
            messages,
            max_tokens: Some(params.max_tokens as i32),
            temperature: params.temperature,
            tools,
            tool_choice: if params.tools.is_empty() { None } else { Some("auto".to_string()) },
            stream: params.stream,
        }
    }
}

impl From<OpenAIChatResponse> for Message {
    fn from(response: OpenAIChatResponse) -> Self {
        let choice = response.choices.first();

        let mut content = Vec::new();
        let mut stop_reason = None;

        if let Some(choice) = choice {
            // Add text content
            if let Some(text) = &choice.message.content {
                if !text.is_empty() {
                    content.push(ContentBlock::Text(TextBlock { text: text.clone() }));
                }
            }

            // Add tool calls
            if let Some(tool_calls) = &choice.message.tool_calls {
                for tc in tool_calls {
                    let input: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                    content.push(ContentBlock::ToolUse(ToolUseBlock {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        input,
                    }));
                }
            }

            // Map finish reason
            stop_reason = match choice.finish_reason.as_deref() {
                Some("stop") => Some(StopReason::EndTurn),
                Some("length") => Some(StopReason::MaxTokens),
                Some("tool_calls") => Some(StopReason::ToolUse),
                _ => None,
            };
        }

        Message {
            id: response.id.unwrap_or_else(|| "openai-compat".to_string()),
            message_type: "message".to_string(),
            model: response.model.unwrap_or_else(|| "unknown".to_string()),
            role: "assistant".to_string(),
            content,
            stop_reason,
            stop_sequence: None,
            usage: response.usage.map(|u| Usage {
                input_tokens: u.prompt_tokens as u64,
                output_tokens: u.completion_tokens as u64,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }).unwrap_or_default(),
        }
    }
}
