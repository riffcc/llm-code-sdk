//! OpenAI Responses API types (/v1/responses).
//!
//! Used with ChatGPT OAuth tokens. The Responses API has a different
//! request/response format from Chat Completions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    ContentBlock, Message, MessageCreateParams, StopReason, TextBlock, ToolUseBlock, Usage,
};

/// Request body for the Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    pub store: bool,
    pub stream: bool,
}

/// Reasoning/effort configuration for the Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct ReasoningConfig {
    pub effort: String,
}

/// An item in the input array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponseItemContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

/// Content of a message input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseItemContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

/// Tool definition for the Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Response from the Responses API.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub error: Option<ResponsesError>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
}

/// An output item.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        role: String,
        #[serde(default)]
        content: Vec<OutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)]
        call_id: String,
        #[serde(default)]
        name: String,
        #[serde(default)]
        arguments: String,
    },
}

/// Output content block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        #[serde(default)]
        text: String,
    },
}

/// Usage for Responses API.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokenDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputTokenDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

/// Error in response.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesError {
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

// ── Conversions ──

impl From<&MessageCreateParams> for ResponsesRequest {
    fn from(params: &MessageCreateParams) -> Self {
        let instructions = params.system.as_ref().map(|s| match s {
            super::SystemPrompt::Text(t) => t.clone(),
            super::SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        });

        let mut input = Vec::new();

        for msg in &params.messages {
            match &msg.content {
                super::MessageContent::Text(text) => {
                    input.push(ResponseItem::Message {
                        role: msg.role.clone(),
                        content: ResponseItemContent::Text(text.clone()),
                    });
                }
                super::MessageContent::Blocks(blocks) => {
                    let mut text_parts = Vec::new();
                    for block in blocks {
                        match block {
                            super::ContentBlockParam::Text { text, .. } => {
                                text_parts.push(text.clone());
                            }
                            super::ContentBlockParam::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                let output = match content {
                                    Some(super::ToolResultContent::Text(t)) => t.clone(),
                                    Some(super::ToolResultContent::Blocks(blocks)) => blocks
                                        .iter()
                                        .filter_map(|b| {
                                            if let super::ToolResultContentBlock::Text { text } = b {
                                                Some(text.clone())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n"),
                                    None => String::new(),
                                };
                                let output = if *is_error {
                                    format!("ERROR: {output}")
                                } else {
                                    output
                                };
                                input.push(ResponseItem::FunctionCallOutput {
                                    call_id: tool_use_id.clone(),
                                    output,
                                });
                            }
                            super::ContentBlockParam::ToolUse { id, name, input: args } => {
                                input.push(ResponseItem::FunctionCall {
                                    call_id: id.clone(),
                                    name: name.clone(),
                                    arguments: serde_json::to_string(args).unwrap_or_default(),
                                });
                            }
                            _ => {}
                        }
                    }
                    if !text_parts.is_empty() {
                        let role = if msg.role == "assistant" { "assistant" } else { "user" };
                        input.push(ResponseItem::Message {
                            role: role.to_string(),
                            content: ResponseItemContent::Text(text_parts.join("\n")),
                        });
                    }
                }
            }
        }

        let tools: Vec<ResponsesTool> = params.tools.iter().map(|t| {
            ResponsesTool {
                tool_type: "function".to_string(),
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: serde_json::to_value(&t.input_schema).unwrap_or_default(),
                strict: Some(false),
            }
        }).collect();

        let reasoning = params.reasoning_effort.as_ref().map(|effort| {
            ReasoningConfig { effort: effort.clone() }
        });

        ResponsesRequest {
            model: params.model.clone(),
            input,
            instructions,
            tools,
            temperature: params.temperature,
            reasoning,
            store: false,
            stream: true,
        }
    }
}

impl From<ResponsesResponse> for Message {
    fn from(response: ResponsesResponse) -> Self {
        let mut content = Vec::new();
        let mut has_tool_calls = false;

        for item in &response.output {
            match item {
                OutputItem::Message { content: parts, .. } => {
                    for part in parts {
                        match part {
                            OutputContent::OutputText { text } => {
                                if !text.is_empty() {
                                    content.push(ContentBlock::Text(TextBlock {
                                        text: text.clone(),
                                    }));
                                }
                            }
                        }
                    }
                }
                OutputItem::FunctionCall { call_id, name, arguments } => {
                    has_tool_calls = true;
                    let input: HashMap<String, serde_json::Value> =
                        serde_json::from_str(arguments).unwrap_or_default();
                    content.push(ContentBlock::ToolUse(ToolUseBlock {
                        id: call_id.clone(),
                        name: name.clone(),
                        input,
                    }));
                }
            }
        }

        let stop_reason = if has_tool_calls {
            Some(StopReason::ToolUse)
        } else if response.status == "incomplete" {
            Some(StopReason::MaxTokens)
        } else {
            Some(StopReason::EndTurn)
        };

        let usage = response.usage.map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: u.input_tokens_details.map(|d| d.cached_tokens),
        }).unwrap_or_default();

        Message {
            id: response.id,
            message_type: "message".to_string(),
            model: response.model.unwrap_or_else(|| "unknown".to_string()),
            role: "assistant".to_string(),
            content,
            stop_reason,
            stop_sequence: None,
            usage,
        }
    }
}
