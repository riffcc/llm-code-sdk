//! SmartWrite tool - writes code using the layercake approach.
//!
//! Uses structural understanding to make surgical edits rather than
//! full file rewrites. The LLM can specify edits at different granularities:
//! - Function level: replace/insert/delete functions
//! - Block level: modify specific code blocks
//! - Line level: precise line edits
//!
//! The tool uses AST awareness to validate edits maintain valid syntax.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::ast::{AstParser, Lang, Symbol};
use super::layers::LayerAnalyzer;
use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// Edit granularity for SmartWrite operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditGranularity {
    /// Replace an entire function/method
    Function,
    /// Replace a specific symbol (struct, enum, etc.)
    Symbol,
    /// Replace lines in a range
    Lines,
    /// Insert at a specific location
    Insert,
    /// Delete a symbol or range
    Delete,
}

/// A structural edit operation.
#[derive(Debug, Clone)]
pub struct StructuralEdit {
    pub granularity: EditGranularity,
    pub target: String,        // Symbol name or line range
    pub new_content: String,   // Replacement content
    pub after: Option<String>, // For Insert: insert after this symbol
}

impl StructuralEdit {
    pub fn replace_function(name: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Function,
            target: name.to_string(),
            new_content: new_content.to_string(),
            after: None,
        }
    }

    pub fn replace_symbol(name: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Symbol,
            target: name.to_string(),
            new_content: new_content.to_string(),
            after: None,
        }
    }

    pub fn insert_after(after_symbol: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Insert,
            target: String::new(),
            new_content: new_content.to_string(),
            after: Some(after_symbol.to_string()),
        }
    }

    pub fn delete_symbol(name: &str) -> Self {
        Self {
            granularity: EditGranularity::Delete,
            target: name.to_string(),
            new_content: String::new(),
            after: None,
        }
    }

    pub fn replace_lines(start: usize, end: usize, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Lines,
            target: format!("{}:{}", start, end),
            new_content: new_content.to_string(),
            after: None,
        }
    }
}

/// SmartWrite tool for structure-aware code editing.
pub struct SmartWriteTool {
    project_root: PathBuf,
    analyzer: Arc<RwLock<LayerAnalyzer>>,
    dry_run: bool,
    read_tracker: super::ReadTracker,
}

impl SmartWriteTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            dry_run: false,
            read_tracker: super::ReadTracker::new(),
        }
    }

    /// Create with a shared read tracker (paired with SmartRead).
    pub fn with_tracker(project_root: impl Into<PathBuf>, tracker: super::ReadTracker) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            dry_run: false,
            read_tracker: tracker,
        }
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }

    /// Apply a structural edit to a file.
    pub fn apply_edit(&self, path: &str, edit: &StructuralEdit) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        // Security check
        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Path must be within project root".to_string());
            }
        }

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path).ok_or_else(|| "Unsupported language".to_string())?;

        let new_content = self.apply_structural_edit(&content, lang, edit)?;

        if !self.dry_run {
            std::fs::write(&full_path, &new_content)
                .map_err(|e| format!("Failed to write file: {}", e))?;
        }

        Ok(new_content)
    }

    fn apply_structural_edit(
        &self,
        source: &str,
        lang: Lang,
        edit: &StructuralEdit,
    ) -> Result<String, String> {
        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(source, lang);

        match edit.granularity {
            EditGranularity::Function | EditGranularity::Symbol => {
                self.apply_symbol_replacement(source, &symbols, &edit.target, &edit.new_content)
            }
            EditGranularity::Insert => {
                let after = edit
                    .after
                    .as_ref()
                    .ok_or("Insert requires 'after' target")?;
                self.apply_insert(source, &symbols, after, &edit.new_content)
            }
            EditGranularity::Delete => self.apply_delete(source, &symbols, &edit.target),
            EditGranularity::Lines => {
                self.apply_line_replacement(source, &edit.target, &edit.new_content)
            }
        }
    }

    fn apply_symbol_replacement(
        &self,
        source: &str,
        symbols: &[Symbol],
        target_name: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == target_name)
            .ok_or_else(|| format!("Symbol '{}' not found", target_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines before the symbol
        for line in &lines[..symbol.start_line - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // New content
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_insert(
        &self,
        source: &str,
        symbols: &[Symbol],
        after_name: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == after_name)
            .ok_or_else(|| format!("Symbol '{}' not found", after_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines up to and including the symbol
        for line in &lines[..symbol.end_line] {
            result.push_str(line);
            result.push('\n');
        }

        // Blank line + new content
        result.push('\n');
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_delete(
        &self,
        source: &str,
        symbols: &[Symbol],
        target_name: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == target_name)
            .ok_or_else(|| format!("Symbol '{}' not found", target_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines before the symbol
        for line in &lines[..symbol.start_line - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // Skip the symbol's lines

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_line_replacement(
        &self,
        source: &str,
        range: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let parts: Vec<&str> = range.split(':').collect();
        if parts.len() != 2 {
            return Err("Line range must be 'start:end'".to_string());
        }

        let start: usize = parts[0].parse().map_err(|_| "Invalid start line")?;
        let end: usize = parts[1].parse().map_err(|_| "Invalid end line")?;

        let lines: Vec<&str> = source.lines().collect();
        if start < 1 || end > lines.len() || start > end {
            return Err(format!("Invalid line range {}:{}", start, end));
        }

        let mut result = String::new();

        // Lines before
        for line in &lines[..start - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // New content
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after
        for line in &lines[end..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    /// Write arbitrary content to a file, creating parent directories as needed.
    fn handle_write(&self, path: &str, content: &str) -> ToolResult {
        let full_path = self.resolve_path(path);

        // Security check
        if let Some(parent) = full_path.parent() {
            if let Ok(canonical) = parent.canonicalize() {
                if !canonical.starts_with(&self.project_root) {
                    return ToolResult::error("Path must be within project root");
                }
            }
        }

        if self.dry_run {
            return ToolResult::success(format!("(dry run) Would write {} bytes to {}", content.len(), path));
        }

        // Create parent directories
        if let Some(parent) = full_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("Failed to create directories: {}", e));
            }
        }

        match std::fs::write(&full_path, content) {
            Ok(()) => ToolResult::success(format!("Wrote {} bytes to {}", content.len(), path)),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }

    /// Handle a batch of writes.
    async fn handle_batch(&self, writes: &[serde_json::Value]) -> ToolResult {
        let mut results = Vec::new();

        for (i, w) in writes.iter().enumerate() {
            let path = w.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = w.get("content").and_then(|v| v.as_str()).unwrap_or("");

            if path.is_empty() {
                results.push(format!("[{}] error: path is required", i));
                continue;
            }

            let result = self.handle_write(path, content);
            if result.is_error() {
                results.push(format!("[{}] error writing {}: {}", i, path, result.to_content_string()));
            } else {
                results.push(format!("[{}] {}", i, result.to_content_string()));
            }
        }

        ToolResult::success(results.join("\n"))
    }

    /// Preview what an edit would produce without writing.
    pub fn preview_edit(&self, path: &str, edit: &StructuralEdit) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path).ok_or_else(|| "Unsupported language".to_string())?;

        self.apply_structural_edit(&content, lang, edit)
    }
}

#[async_trait]
impl Tool for SmartWriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "write",
            InputSchema::object()
                .required_string("path", "File path to write or edit")
                .required_string("content", "File content (for 'write') or new content (for edits)")
                .optional_string("operation", "Edit operation: 'write' (default, creates/overwrites file), 'replace_function', 'replace_symbol', 'insert_after', 'delete', 'replace_lines'")
                .optional_string("target", "Target symbol name or line range (e.g., 'my_function' or '10:20') — required for structural edits")
                .optional_string("after", "For insert_after: symbol to insert after"),
        )
        .with_description(
            "Write or edit files. Default operation creates/overwrites a file. Structural operations (replace_function, replace_symbol, insert_after, delete, replace_lines) make surgical edits.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        // Batch mode: array of writes
        if let Some(writes) = input.get("writes").and_then(|v| v.as_array()) {
            return self.handle_batch(writes).await;
        }

        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        // Enforce read-before-write
        let full_path = self.resolve_path(path);
        if let Err(e) = self.read_tracker.check_write(&full_path) {
            return ToolResult::error(e);
        }

        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("write");

        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        // Plain write: create or overwrite the file
        if operation == "write" {
            return self.handle_write(path, content);
        }

        let target = input.get("target").and_then(|v| v.as_str()).unwrap_or("");
        let after = input.get("after").and_then(|v| v.as_str());

        let edit = match operation {
            "replace_function" => StructuralEdit::replace_function(target, content),
            "replace_symbol" => StructuralEdit::replace_symbol(target, content),
            "insert_after" => {
                let after_target = after.unwrap_or(target);
                StructuralEdit::insert_after(after_target, content)
            }
            "delete" => StructuralEdit::delete_symbol(target),
            "replace_lines" => StructuralEdit::replace_lines(
                target
                    .split(':')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1),
                target
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1),
                content,
            ),
            _ => return ToolResult::error(format!("Unknown operation: {}", operation)),
        };

        match self.apply_edit(path, &edit) {
            Ok(result) => {
                let preview = if result.len() > 500 {
                    format!(
                        "{}...\n(truncated, {} total chars)",
                        &result[..500],
                        result.len()
                    )
                } else {
                    result
                };
                ToolResult::success(format!("Edit applied successfully.\n\n{}", preview))
            }
            Err(e) => ToolResult::error(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_replace_function() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit =
            StructuralEdit::replace_function("hello", "fn hello() {\n    println!(\"Hi!\");\n}");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("Hi!"));
        assert!(result.contains("fn world()"));
        assert!(!result.contains("Hello"));
    }

    #[test]
    fn test_insert_after() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn first() {
    println!("First");
}

fn third() {
    println!("Third");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit =
            StructuralEdit::insert_after("first", "fn second() {\n    println!(\"Second\");\n}");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("fn first()"));
        assert!(result.contains("fn second()"));
        assert!(result.contains("fn third()"));

        // Verify order
        let first_pos = result.find("fn first()").unwrap();
        let second_pos = result.find("fn second()").unwrap();
        let third_pos = result.find("fn third()").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_delete_symbol() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn keep_me() {
    println!("Keep");
}

fn delete_me() {
    println!("Delete");
}

fn also_keep() {
    println!("Also keep");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit = StructuralEdit::delete_symbol("delete_me");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("fn keep_me()"));
        assert!(result.contains("fn also_keep()"));
        assert!(!result.contains("fn delete_me()"));
        assert!(!result.contains("Delete"));
    }

    #[tokio::test]
    async fn test_smart_write_tool() {
        let dir = TempDir::new().unwrap();

        let source = r#"pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
"#;

        fs::write(dir.path().join("calc.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert(
            "operation".to_string(),
            serde_json::json!("replace_function"),
        );
        input.insert("target".to_string(), serde_json::json!("add"));
        input.insert(
            "content".to_string(),
            serde_json::json!(
                "pub fn add(a: i32, b: i32) -> i32 {\n    // Optimized\n    a + b\n}"
            ),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(dir.path().join("calc.rs")).unwrap();
        assert!(content.contains("// Optimized"));
        assert!(content.contains("fn subtract"));
    }
}
