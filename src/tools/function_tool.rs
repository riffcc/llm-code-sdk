//! Function-based tool implementation.

use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// A tool implemented as a function.
///
/// This provides a convenient way to create tools from closures or functions.
///
/// # Example
///
/// ```rust
/// use llm_code_sdk::tools::FunctionTool;
/// use llm_code_sdk::types::InputSchema;
///
/// let tool = FunctionTool::new(
///     "greet",
///     "Greet someone by name",
///     InputSchema::object().required_string("name", "The name to greet"),
///     |input| {
///         let name = input.get("name")
///             .and_then(|v| v.as_str())
///             .unwrap_or("World");
///         Ok(format!("Hello, {}!", name))
///     },
/// );
/// ```
pub struct FunctionTool<F>
where
    F: Fn(HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync + 'static,
{
    name: String,
    description: Option<String>,
    input_schema: InputSchema,
    func: F,
}

impl<F> FunctionTool<F>
where
    F: Fn(HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync + 'static,
{
    /// Create a new function tool.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: InputSchema,
        func: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
            func,
        }
    }

    /// Create a function tool without a description.
    pub fn without_description(
        name: impl Into<String>,
        input_schema: InputSchema,
        func: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
            func,
        }
    }
}

#[async_trait]
impl<F> Tool for FunctionTool<F>
where
    F: Fn(HashMap<String, serde_json::Value>) -> Result<String, String> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn to_param(&self) -> ToolParam {
        let mut param = ToolParam::new(self.name.clone(), self.input_schema.clone());
        if let Some(ref desc) = self.description {
            param = param.with_description(desc.clone());
        }
        param
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        match (self.func)(input) {
            Ok(s) => ToolResult::Success(s),
            Err(e) => ToolResult::Error(e),
        }
    }
}

/// Type alias for async tool functions.
pub type AsyncToolFn = Arc<
    dyn Fn(HashMap<String, serde_json::Value>) -> Pin<Box<dyn Future<Output = ToolResult> + Send>>
        + Send
        + Sync,
>;

/// An async function tool.
pub struct AsyncFunctionTool {
    name: String,
    description: Option<String>,
    input_schema: InputSchema,
    func: AsyncToolFn,
}

impl AsyncFunctionTool {
    /// Create a new async function tool.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: InputSchema,
        func: F,
    ) -> Self
    where
        F: Fn(HashMap<String, serde_json::Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ToolResult> + Send + 'static,
    {
        let func = Arc::new(move |input| {
            let fut = func(input);
            Box::pin(fut) as Pin<Box<dyn Future<Output = ToolResult> + Send>>
        });

        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
            func,
        }
    }
}

#[async_trait]
impl Tool for AsyncFunctionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn to_param(&self) -> ToolParam {
        let mut param = ToolParam::new(self.name.clone(), self.input_schema.clone());
        if let Some(ref desc) = self.description {
            param = param.with_description(desc.clone());
        }
        param
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        (self.func)(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_tool_creation() {
        let tool = FunctionTool::new(
            "test_tool",
            "A test tool",
            InputSchema::object().required_string("input", "Test input"),
            |input| {
                let val = input.get("input").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!("Got: {}", val))
            },
        );

        assert_eq!(tool.name(), "test_tool");

        let param = tool.to_param();
        assert_eq!(param.name, "test_tool");
        assert_eq!(param.description, Some("A test tool".to_string()));
    }

    #[tokio::test]
    async fn test_function_tool_call() {
        let tool = FunctionTool::new(
            "echo",
            "Echo the input",
            InputSchema::object().required_string("message", "Message to echo"),
            |input| {
                let msg = input
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("empty");
                Ok(format!("Echo: {}", msg))
            },
        );

        let mut input = HashMap::new();
        input.insert("message".to_string(), serde_json::json!("Hello"));

        let result = tool.call(input).await;
        assert_eq!(result.to_content_string(), "Echo: Hello");
    }

    #[tokio::test]
    async fn test_function_tool_error() {
        let tool = FunctionTool::new(
            "fail",
            "Always fails",
            InputSchema::object(),
            |_| Err("This always fails".to_string()),
        );

        let result = tool.call(HashMap::new()).await;
        assert!(result.is_error());
        assert_eq!(result.to_content_string(), "This always fails");
    }

    #[tokio::test]
    async fn test_async_function_tool() {
        let tool = AsyncFunctionTool::new(
            "async_echo",
            "Async echo",
            InputSchema::object().required_string("message", "Message"),
            |input| async move {
                let msg = input
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("empty");
                ToolResult::success(format!("Async: {}", msg))
            },
        );

        let mut input = HashMap::new();
        input.insert("message".to_string(), serde_json::json!("Test"));

        let result = tool.call(input).await;
        assert_eq!(result.to_content_string(), "Async: Test");
    }
}
