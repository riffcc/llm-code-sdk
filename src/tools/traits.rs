//! Tool trait definitions.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::types::ToolParam;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Successful execution with string output.
    Success {
        output: String,
        metadata: Option<serde_json::Value>,
    },

    /// Successful execution with structured content.
    Content(Vec<ToolResultContent>),

    /// Tool execution failed with an error message.
    Error(String),
}

impl ToolResult {
    /// Create a success result from a string.
    pub fn success(s: impl Into<String>) -> Self {
        ToolResult::Success {
            output: s.into(),
            metadata: None,
        }
    }

    /// Create a success result with structured metadata.
    pub fn success_with_metadata(s: impl Into<String>, metadata: serde_json::Value) -> Self {
        ToolResult::Success {
            output: s.into(),
            metadata: Some(metadata),
        }
    }

    /// Create an error result.
    pub fn error(s: impl Into<String>) -> Self {
        ToolResult::Error(s.into())
    }

    /// Returns true if this is an error result.
    pub fn is_error(&self) -> bool {
        matches!(self, ToolResult::Error(_))
    }

    /// Get structured metadata when present.
    pub fn metadata(&self) -> Option<&serde_json::Value> {
        match self {
            ToolResult::Success { metadata, .. } => metadata.as_ref(),
            _ => None,
        }
    }

    /// Get the content as a string for the API.
    pub fn to_content_string(&self) -> String {
        match self {
            ToolResult::Success { output, .. } => output.clone(),
            ToolResult::Content(blocks) => {
                // Convert structured content to string representation
                blocks
                    .iter()
                    .filter_map(|b| match b {
                        ToolResultContent::Text(t) => Some(t.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            ToolResult::Error(e) => e.clone(),
        }
    }
}

impl From<String> for ToolResult {
    fn from(s: String) -> Self {
        ToolResult::success(s)
    }
}

impl From<&str> for ToolResult {
    fn from(s: &str) -> Self {
        ToolResult::success(s.to_string())
    }
}

impl<T: ToString, E: ToString> From<Result<T, E>> for ToolResult {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(v) => ToolResult::success(v.to_string()),
            Err(e) => ToolResult::Error(e.to_string()),
        }
    }
}

/// Content types for structured tool results.
#[derive(Debug, Clone)]
pub enum ToolResultContent {
    /// Text content.
    Text(String),

    /// Image content (base64 encoded).
    Image { media_type: String, data: String },
}

/// A tool that can be invoked by the model.
///
/// Implement this trait to create custom tools that the model can use.
///
/// # Example
///
/// ```rust
/// use llm_code_sdk::tools::{Tool, ToolResult};
/// use llm_code_sdk::types::{ToolParam, InputSchema};
/// use async_trait::async_trait;
/// use std::collections::HashMap;
///
/// struct ReadFileTool;
///
/// #[async_trait]
/// impl Tool for ReadFileTool {
///     fn name(&self) -> &str {
///         "read_file"
///     }
///
///     fn to_param(&self) -> ToolParam {
///         ToolParam::new(
///             "read_file",
///             InputSchema::object()
///                 .required_string("path", "The file path to read"),
///         )
///         .with_description("Read a file from disk")
///     }
///
///     async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
///         let path = input.get("path")
///             .and_then(|v| v.as_str())
///             .unwrap_or("");
///
///         match std::fs::read_to_string(path) {
///             Ok(content) => ToolResult::success(content),
///             Err(e) => ToolResult::error(format!("Failed to read file: {}", e)),
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the name of the tool.
    fn name(&self) -> &str;

    /// Returns the tool parameter definition for the API.
    fn to_param(&self) -> ToolParam;

    /// Execute the tool with the given input.
    ///
    /// The input is a map of parameter names to JSON values.
    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_from_string() {
        let result: ToolResult = "Hello".into();
        assert!(!result.is_error());
        assert_eq!(result.to_content_string(), "Hello");
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert!(result.is_error());
        assert_eq!(result.to_content_string(), "Something went wrong");
    }

    #[test]
    fn test_tool_result_from_std_result() {
        let ok_result: Result<&str, &str> = Ok("Success");
        let tool_result: ToolResult = ok_result.into();
        assert!(!tool_result.is_error());

        let err_result: Result<&str, &str> = Err("Failed");
        let tool_result: ToolResult = err_result.into();
        assert!(tool_result.is_error());
    }
}
