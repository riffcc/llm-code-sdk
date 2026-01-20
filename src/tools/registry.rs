//! Central tool registry.
//!
//! All tools should be registered here. Use `pal call <tool>` to invoke.

use std::collections::HashMap;
use std::sync::Arc;

use super::{Tool, ToolResult};

/// Central registry for all Palace tools.
///
/// Tools are registered once and looked up by name.
/// This is the ONE place for tool registration - do not duplicate elsewhere.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Register multiple tools.
    pub fn register_all(&mut self, tools: Vec<Arc<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Call a tool by name with JSON input.
    pub async fn call(
        &self,
        name: &str,
        input: HashMap<String, serde_json::Value>,
    ) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.call(input).await,
            None => ToolResult::error(format!(
                "Unknown tool: '{}'\nAvailable: {}",
                name,
                self.list().join(", ")
            )),
        }
    }

    /// List all registered tool names.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Get help text for all tools.
    pub fn help(&self) -> String {
        let mut help = String::from("Available tools:\n\n");
        for name in self.list() {
            if let Some(tool) = self.get(name) {
                let param = tool.to_param();
                help.push_str(&format!(
                    "  {:<16} {}\n",
                    name,
                    param.description.as_deref().unwrap_or("")
                ));
            }
        }
        help
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a registry with standard exploration tools.
pub fn create_exploration_registry(project_root: &std::path::Path) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register_all(super::create_exploration_tools(project_root));
    registry
}

/// Create a registry with standard editing tools.
pub fn create_editing_registry(project_root: &std::path::Path) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register_all(super::create_editing_tools(project_root));
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InputSchema;
    use async_trait::async_trait;

    struct TestTool;

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "test"
        }

        fn to_param(&self) -> crate::types::ToolParam {
            crate::types::ToolParam::new("test", InputSchema::object())
                .with_description("A test tool")
        }

        async fn call(&self, _input: HashMap<String, serde_json::Value>) -> ToolResult {
            ToolResult::success("test result")
        }
    }

    #[tokio::test]
    async fn test_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(TestTool));

        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());

        let result = registry.call("test", HashMap::new()).await;
        assert!(!result.is_error());
        assert_eq!(result.to_content_string(), "test result");
    }
}
