//! JECJIT Examples Tool - provides code examples just-in-time.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::ast::Lang;
use super::examples::{CodeExamples, PatternKind};
use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, PropertySchema, ToolParam};

/// Tool for providing JECJIT code examples.
pub struct ExamplesTool {
    project_root: PathBuf,
    examples: Arc<RwLock<CodeExamples>>,
}

impl ExamplesTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        let root = project_root.into();
        Self {
            project_root: root.clone(),
            examples: Arc::new(RwLock::new(CodeExamples::new(root))),
        }
    }
}

#[async_trait]
impl Tool for ExamplesTool {
    fn name(&self) -> &str {
        "code_examples"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "code_examples",
            InputSchema::object()
                .required_string("language", "Language: rust, nim, lean, python, go, javascript, typescript, perl")
                .property(
                    "patterns",
                    PropertySchema::array(PropertySchema::string())
                        .with_description("Pattern types to include: error, testing, idiom, async, data, imports, api"),
                    false,
                )
                .optional_string("task", "Description of the task for context-aware examples"),
        )
        .with_description(
            "Get idiomatic code examples for a language. Combines built-in patterns with project-specific examples from .claude/examples/ and the codebase.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let lang_str = input
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("rust");

        let lang = match lang_str.to_lowercase().as_str() {
            "rust" | "rs" => Lang::Rust,
            "nim" => Lang::Nim,
            "lean" => Lang::Lean,
            "python" | "py" => Lang::Python,
            "go" | "golang" => Lang::Go,
            "javascript" | "js" => Lang::JavaScript,
            "typescript" | "ts" => Lang::TypeScript,
            "perl" | "pl" => Lang::Perl,
            _ => return ToolResult::error(format!("Unknown language: {}", lang_str)),
        };

        // Parse requested patterns
        let patterns: Vec<PatternKind> = input
            .get("patterns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(parse_pattern)
                    .collect()
            })
            .unwrap_or_else(|| {
                // If task is provided, infer patterns
                if let Some(task) = input.get("task").and_then(|v| v.as_str()) {
                    infer_patterns_from_task(task)
                } else {
                    // Default to most useful patterns
                    vec![
                        PatternKind::ErrorHandling,
                        PatternKind::Idiom,
                        PatternKind::Testing,
                    ]
                }
            });

        let mut examples = self.examples.write().unwrap();
        let context = examples.to_context(lang, &patterns);

        ToolResult::success(context)
    }
}

fn parse_pattern(s: &str) -> Option<PatternKind> {
    match s.to_lowercase().as_str() {
        "error" | "errorhandling" | "error_handling" => Some(PatternKind::ErrorHandling),
        "test" | "testing" => Some(PatternKind::Testing),
        "api" | "apiusage" | "api_usage" => Some(PatternKind::ApiUsage),
        "idiom" | "idioms" => Some(PatternKind::Idiom),
        "async" | "concurrent" => Some(PatternKind::Async),
        "data" | "datastructure" | "data_structure" => Some(PatternKind::DataStructure),
        "import" | "imports" => Some(PatternKind::Imports),
        _ => None,
    }
}

fn infer_patterns_from_task(task: &str) -> Vec<PatternKind> {
    let task_lower = task.to_lowercase();
    let mut patterns = Vec::new();

    if task_lower.contains("error")
        || task_lower.contains("result")
        || task_lower.contains("handle")
    {
        patterns.push(PatternKind::ErrorHandling);
    }
    if task_lower.contains("test") || task_lower.contains("assert") {
        patterns.push(PatternKind::Testing);
    }
    if task_lower.contains("async")
        || task_lower.contains("await")
        || task_lower.contains("concurrent")
    {
        patterns.push(PatternKind::Async);
    }
    if task_lower.contains("import") || task_lower.contains("module") {
        patterns.push(PatternKind::Imports);
    }
    if task_lower.contains("struct") || task_lower.contains("type") || task_lower.contains("class")
    {
        patterns.push(PatternKind::DataStructure);
    }

    // Always include idioms if task is code-related
    if patterns.is_empty() || task_lower.contains("write") || task_lower.contains("implement") {
        patterns.push(PatternKind::Idiom);
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_examples_tool() {
        let tool = ExamplesTool::new("/tmp");

        let mut input = HashMap::new();
        input.insert("language".to_string(), serde_json::json!("nim"));
        input.insert(
            "patterns".to_string(),
            serde_json::json!(["idiom", "error"]),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = result.to_content_string();
        assert!(content.contains("## Nim Code Examples"));
        assert!(content.contains("Idioms"));
    }

    #[tokio::test]
    async fn test_task_inference() {
        let tool = ExamplesTool::new("/tmp");

        let mut input = HashMap::new();
        input.insert("language".to_string(), serde_json::json!("rust"));
        input.insert(
            "task".to_string(),
            serde_json::json!("implement error handling for file operations"),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = result.to_content_string();
        assert!(content.contains("Error Handling"));
    }

    #[test]
    fn test_infer_patterns() {
        let patterns = infer_patterns_from_task("write async http client with error handling");
        assert!(patterns.contains(&PatternKind::Async));
        assert!(patterns.contains(&PatternKind::ErrorHandling));
    }
}
