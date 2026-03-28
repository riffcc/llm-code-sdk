//! Content block types for messages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A content block in a message response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// A text content block.
    Text(TextBlock),

    /// A tool use content block (model requesting to use a tool).
    ToolUse(ToolUseBlock),

    /// A tool result content block (result of executing a tool).
    ToolResult(ToolResultBlock),

    /// A thinking content block (extended thinking).
    Thinking(ThinkingBlock),

    /// Redacted thinking content (for safety).
    RedactedThinking(RedactedThinkingBlock),
}

impl ContentBlock {
    /// Returns the type name of this content block.
    pub fn block_type(&self) -> &'static str {
        match self {
            ContentBlock::Text(_) => "text",
            ContentBlock::ToolUse(_) => "tool_use",
            ContentBlock::ToolResult(_) => "tool_result",
            ContentBlock::Thinking(_) => "thinking",
            ContentBlock::RedactedThinking(_) => "redacted_thinking",
        }
    }

    /// Returns the text content if this is a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text(block) => Some(&block.text),
            _ => None,
        }
    }

    /// Returns the tool use block if this is a tool use.
    pub fn as_tool_use(&self) -> Option<&ToolUseBlock> {
        match self {
            ContentBlock::ToolUse(block) => Some(block),
            _ => None,
        }
    }
}

/// A text content block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextBlock {
    /// The text content.
    pub text: String,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A tool use content block - the model requesting to use a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUseBlock {
    /// Unique identifier for this tool use.
    pub id: String,

    /// Name of the tool being invoked.
    pub name: String,

    /// Input arguments for the tool.
    pub input: HashMap<String, serde_json::Value>,
}

/// A tool result content block - the result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultBlock {
    /// The ID of the tool use this is a result for.
    pub tool_use_id: String,

    /// The result content (string or array of content blocks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolResultContent>,

    /// Whether the tool execution resulted in an error.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

/// Content of a tool result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple string content. Uses Arc<str> so clones share the allocation.
    Text(std::sync::Arc<str>),

    /// Array of content blocks.
    Blocks(Vec<ToolResultContentBlock>),
}

impl From<String> for ToolResultContent {
    fn from(s: String) -> Self {
        ToolResultContent::Text(std::sync::Arc::from(s.as_str()))
    }
}

impl From<&str> for ToolResultContent {
    fn from(s: &str) -> Self {
        ToolResultContent::Text(std::sync::Arc::from(s))
    }
}

/// A content block within a tool result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text { text: String },
    Image { source: ImageSource },
}

/// Image source for image content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: ImageMediaType,
        data: String,
    },
    Url {
        url: String,
    },
}

impl ImageSource {
    /// Create a base64 image source.
    pub fn base64(media_type: ImageMediaType, data: impl Into<String>) -> Self {
        ImageSource::Base64 {
            media_type,
            data: data.into(),
        }
    }

    /// Create a URL image source.
    pub fn url(url: impl Into<String>) -> Self {
        ImageSource::Url { url: url.into() }
    }
}

/// Supported image media types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageMediaType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

/// Document source for document content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded PDF.
    Base64 {
        media_type: DocumentMediaType,
        data: String,
    },
    /// URL to a PDF document.
    Url { url: String },
    /// Plain text document.
    Text {
        media_type: TextMediaType,
        data: String,
    },
}

impl DocumentSource {
    /// Create a base64 PDF document source.
    pub fn pdf_base64(data: impl Into<String>) -> Self {
        DocumentSource::Base64 {
            media_type: DocumentMediaType::Pdf,
            data: data.into(),
        }
    }

    /// Create a URL PDF document source.
    pub fn pdf_url(url: impl Into<String>) -> Self {
        DocumentSource::Url { url: url.into() }
    }

    /// Create a plain text document source.
    pub fn plain_text(data: impl Into<String>) -> Self {
        DocumentSource::Text {
            media_type: TextMediaType::Plain,
            data: data.into(),
        }
    }
}

/// Supported document media types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentMediaType {
    #[serde(rename = "application/pdf")]
    Pdf,
}

/// Supported text media types for documents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TextMediaType {
    #[serde(rename = "text/plain")]
    Plain,
}

/// A thinking content block (extended thinking feature).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThinkingBlock {
    /// The thinking content.
    pub thinking: String,

    /// Signature for verification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// A redacted thinking block (content removed for safety).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedactedThinkingBlock {
    /// Placeholder data.
    #[serde(default)]
    pub data: String,
}

/// Cache control for content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControlParam {
    /// The cache control type.
    #[serde(rename = "type")]
    pub control_type: String,
}

impl CacheControlParam {
    /// Create an ephemeral cache control.
    pub fn ephemeral() -> Self {
        Self {
            control_type: "ephemeral".to_string(),
        }
    }
}

/// Content block parameter for request messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    /// Text content.
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlParam>,
    },

    /// Image content.
    Image {
        source: ImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlParam>,
    },

    /// Document content (PDF, text).
    Document {
        source: DocumentSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControlParam>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
    },

    /// Tool use (for assistant messages).
    ToolUse {
        id: String,
        name: String,
        input: HashMap<String, serde_json::Value>,
    },

    /// Tool result (for user messages).
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

impl ContentBlockParam {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlockParam::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Create an image content block from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentBlockParam::Image {
            source: ImageSource::url(url),
            cache_control: None,
        }
    }

    /// Create an image content block from base64 data.
    pub fn image_base64(media_type: ImageMediaType, data: impl Into<String>) -> Self {
        ContentBlockParam::Image {
            source: ImageSource::base64(media_type, data),
            cache_control: None,
        }
    }

    /// Create a PDF document content block from base64 data.
    pub fn document_pdf_base64(data: impl Into<String>) -> Self {
        ContentBlockParam::Document {
            source: DocumentSource::pdf_base64(data),
            cache_control: None,
            title: None,
            context: None,
        }
    }

    /// Create a PDF document content block from a URL.
    pub fn document_pdf_url(url: impl Into<String>) -> Self {
        ContentBlockParam::Document {
            source: DocumentSource::pdf_url(url),
            cache_control: None,
            title: None,
            context: None,
        }
    }

    /// Create a plain text document content block.
    pub fn document_plain_text(data: impl Into<String>) -> Self {
        ContentBlockParam::Document {
            source: DocumentSource::plain_text(data),
            cache_control: None,
            title: None,
            context: None,
        }
    }

    /// Create a tool result content block.
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        let s: String = content.into();
        ContentBlockParam::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(ToolResultContent::from(s)),
            is_error: false,
        }
    }

    /// Create an error tool result content block.
    pub fn tool_result_error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        let s: String = error.into();
        ContentBlockParam::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: Some(ToolResultContent::from(s)),
            is_error: true,
        }
    }
}

impl From<&str> for ContentBlockParam {
    fn from(s: &str) -> Self {
        ContentBlockParam::text(s)
    }
}

impl From<String> for ContentBlockParam {
    fn from(s: String) -> Self {
        ContentBlockParam::text(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_block_serialization() {
        let block = ContentBlock::Text(TextBlock::new("Hello, world!"));
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello, world!\""));
    }

    #[test]
    fn test_text_block_deserialization() {
        let json = r#"{"type":"text","text":"Hello, world!"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        assert_eq!(block.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_tool_use_block_serialization() {
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("/tmp/test.txt"));

        let block = ContentBlock::ToolUse(ToolUseBlock {
            id: "toolu_123".to_string(),
            name: "read_file".to_string(),
            input,
        });

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"id\":\"toolu_123\""));
        assert!(json.contains("\"name\":\"read_file\""));
    }

    #[test]
    fn test_tool_use_block_deserialization() {
        let json =
            r#"{"type":"tool_use","id":"toolu_456","name":"bash","input":{"command":"ls -la"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();

        if let ContentBlock::ToolUse(tool_use) = block {
            assert_eq!(tool_use.id, "toolu_456");
            assert_eq!(tool_use.name, "bash");
            assert_eq!(
                tool_use.input.get("command"),
                Some(&serde_json::json!("ls -la"))
            );
        } else {
            panic!("Expected ToolUse block");
        }
    }

    #[test]
    fn test_tool_result_block() {
        let result = ToolResultBlock {
            tool_use_id: "toolu_123".to_string(),
            content: Some("Success!".into()),
            is_error: false,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"tool_use_id\":\"toolu_123\""));
        assert!(json.contains("\"content\":\"Success!\""));
        assert!(!json.contains("is_error")); // Should be skipped when false
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResultBlock {
            tool_use_id: "toolu_123".to_string(),
            content: Some("File not found".into()),
            is_error: true,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"is_error\":true"));
    }

    #[test]
    fn test_content_block_param_from_str() {
        let param: ContentBlockParam = "Hello".into();
        if let ContentBlockParam::Text {
            text,
            cache_control,
        } = param
        {
            assert_eq!(text, "Hello");
            assert!(cache_control.is_none());
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_document_content_block() {
        let doc = ContentBlockParam::document_pdf_base64("base64data");
        let json = serde_json::to_string(&doc).unwrap();
        assert!(json.contains("\"type\":\"document\""));
        assert!(json.contains("\"source\""));
    }

    #[test]
    fn test_image_content_block() {
        let img = ContentBlockParam::image_url("https://example.com/image.png");
        let json = serde_json::to_string(&img).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"url\":\"https://example.com/image.png\""));
    }
}
