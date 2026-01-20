//! Message creation parameters.

use serde::{Deserialize, Serialize};

use super::{ContentBlockParam, ToolParam};

/// A message parameter for the conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageParam {
    /// The role of the message author ("user" or "assistant").
    pub role: String,

    /// The content of the message.
    pub content: MessageContent,
}

impl MessageParam {
    /// Create a new user message with text content.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a new assistant message with text content.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a new user message with content blocks.
    pub fn user_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }

    /// Create a new assistant message with content blocks.
    pub fn assistant_blocks(blocks: Vec<ContentBlockParam>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(blocks),
        }
    }
}

impl From<(&str, &str)> for MessageParam {
    fn from((role, content): (&str, &str)) -> Self {
        Self {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
        }
    }
}

impl From<(String, String)> for MessageParam {
    fn from((role, content): (String, String)) -> Self {
        Self {
            role,
            content: MessageContent::Text(content),
        }
    }
}

/// Message content - either a string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),

    /// Array of content blocks.
    Blocks(Vec<ContentBlockParam>),
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<Vec<ContentBlockParam>> for MessageContent {
    fn from(blocks: Vec<ContentBlockParam>) -> Self {
        MessageContent::Blocks(blocks)
    }
}

/// Parameters for creating a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCreateParams {
    /// The model to use.
    pub model: String,

    /// The maximum number of tokens to generate.
    pub max_tokens: u32,

    /// Input messages for the conversation.
    pub messages: Vec<MessageParam>,

    /// System prompt (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tools available for the model to use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolParam>,

    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Temperature for sampling (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Custom stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Metadata about the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,

    /// Configuration for extended thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl Default for MessageCreateParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_tokens: 1024,
            messages: Vec::new(),
            system: None,
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            stream: None,
            metadata: None,
            thinking: None,
        }
    }
}

/// System prompt - either a string or array of text blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple text system prompt.
    Text(String),

    /// Array of text blocks (for caching).
    Blocks(Vec<SystemTextBlock>),
}

impl From<&str> for SystemPrompt {
    fn from(s: &str) -> Self {
        SystemPrompt::Text(s.to_string())
    }
}

impl From<String> for SystemPrompt {
    fn from(s: String) -> Self {
        SystemPrompt::Text(s)
    }
}

/// A text block in a system prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemTextBlock {
    /// Type (always "text").
    #[serde(rename = "type")]
    pub block_type: String,

    /// The text content.
    pub text: String,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache control settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheControl {
    /// Cache type.
    #[serde(rename = "type")]
    pub control_type: String,
}

impl CacheControl {
    /// Create an ephemeral cache control.
    pub fn ephemeral() -> Self {
        Self {
            control_type: "ephemeral".to_string(),
        }
    }
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to use tools.
    Auto,

    /// Force the model to use at least one tool.
    Any,

    /// Force the model to use a specific tool.
    Tool {
        /// Name of the tool to use.
        name: String,
    },

    /// Prevent the model from using any tools.
    None,
}

/// Request metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RequestMetadata {
    /// User ID for abuse detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Configuration for extended thinking.
///
/// Extended thinking allows Claude to use additional tokens for internal
/// reasoning before producing a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Enable extended thinking with a token budget.
    Enabled {
        /// The number of tokens Claude can use for thinking.
        /// Must be >= 1024 and less than max_tokens.
        budget_tokens: u32,
    },

    /// Disable extended thinking.
    Disabled,
}

impl ThinkingConfig {
    /// Create an enabled thinking config with the given budget.
    ///
    /// # Panics
    /// Panics if budget_tokens < 1024.
    pub fn enabled(budget_tokens: u32) -> Self {
        assert!(
            budget_tokens >= 1024,
            "budget_tokens must be >= 1024, got {}",
            budget_tokens
        );
        ThinkingConfig::Enabled { budget_tokens }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        ThinkingConfig::Disabled
    }
}

/// Parameters for counting tokens in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensParams {
    /// The model to use for counting.
    pub model: String,

    /// Input messages for the conversation.
    pub messages: Vec<MessageParam>,

    /// System prompt (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tools available for the model to use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolParam>,

    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Configuration for extended thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

impl Default for CountTokensParams {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            system: None,
            tools: Vec::new(),
            tool_choice: None,
            thinking: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_param_user() {
        let msg = MessageParam::user("Hello!");
        assert_eq!(msg.role, "user");
        if let MessageContent::Text(text) = msg.content {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_message_param_from_tuple() {
        let msg: MessageParam = ("user", "Hello!").into();
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn test_message_create_params_serialization() {
        let params = MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 1024,
            messages: vec![MessageParam::user("Hello")],
            system: Some("You are helpful.".into()),
            ..Default::default()
        };

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"model\":\"glm-4-plus\""));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"system\":\"You are helpful.\""));
        // Empty tools should be skipped
        assert!(!json.contains("\"tools\":"));
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::Auto;
        let json = serde_json::to_string(&auto).unwrap();
        assert!(json.contains("\"type\":\"auto\""));

        let tool = ToolChoice::Tool {
            name: "read_file".to_string(),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"tool\""));
        assert!(json.contains("\"name\":\"read_file\""));
    }

    #[test]
    fn test_system_prompt_variants() {
        let text: SystemPrompt = "Hello".into();
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "\"Hello\"");

        let blocks = SystemPrompt::Blocks(vec![SystemTextBlock {
            block_type: "text".to_string(),
            text: "Hello".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        }]);
        let json = serde_json::to_string(&blocks).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"cache_control\""));
    }

    #[test]
    fn test_thinking_config_variants() {
        let enabled = ThinkingConfig::enabled(2048);
        let json = serde_json::to_string(&enabled).unwrap();
        assert!(json.contains("\"type\":\"enabled\""));
        assert!(json.contains("\"budget_tokens\":2048"));

        let disabled = ThinkingConfig::disabled();
        let json = serde_json::to_string(&disabled).unwrap();
        assert_eq!(json, "{\"type\":\"disabled\"}");
    }

    #[test]
    #[should_panic(expected = "budget_tokens must be >= 1024")]
    fn test_thinking_config_budget_validation() {
        ThinkingConfig::enabled(500); // Should panic
    }
}
