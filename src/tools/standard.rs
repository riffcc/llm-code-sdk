//! Standard tools for agentic exploration and code editing.
//!
//! These tools mirror the capabilities of Anthropic's builtin tools:
//! - [`BashTool`] - Execute shell commands
//! - [`ReadFileTool`] - Read file contents
//! - [`WriteFileTool`] - Write/create files
//! - [`EditFileTool`] - Edit files with str_replace
//! - [`GlobTool`] - Find files by glob pattern
//! - [`GrepTool`] - Search file contents
//! - [`ListDirectoryTool`] - List directory contents

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

fn canonical_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn is_within_project_root(path: &Path, project_root: &Path) -> bool {
    let root = canonical_or_original(project_root);

    if let Ok(canonical) = path.canonicalize() {
        return canonical.starts_with(&root);
    }

    if let Some(parent) = path.parent() {
        let canonical_parent = canonical_or_original(parent);
        return canonical_parent.starts_with(&root);
    }

    false
}

#[cfg(feature = "smart")]
use super::smart::{AskCodeTool, MRSearchTool, SmartReadTool, SmartWriteTool};

#[cfg(feature = "search")]
use super::SearchTool;

/// Tool for executing bash commands.
pub struct BashTool {
    working_dir: PathBuf,
    /// Optional timeout in seconds.
    timeout_secs: Option<u64>,
}

impl BashTool {
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
            timeout_secs: Some(120),
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "bash",
            InputSchema::object().required_string("command", "The bash command to execute"),
        )
        .with_description("Execute a bash command and return the output.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");

        if command.is_empty() {
            return ToolResult::error("command is required");
        }

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&self.working_dir)
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    if stderr.is_empty() {
                        ToolResult::success(stdout.to_string())
                    } else {
                        ToolResult::success(format!("{}\n{}", stdout, stderr))
                    }
                } else {
                    ToolResult::error(format!(
                        "Command failed with exit code {:?}\nstdout: {}\nstderr: {}",
                        output.status.code(),
                        stdout,
                        stderr
                    ))
                }
            }
            Err(e) => ToolResult::error(format!("Failed to execute command: {}", e)),
        }
    }
}

/// Tool for reading file contents.
pub struct ReadFileTool {
    project_root: PathBuf,
}

impl ReadFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "read_file",
            InputSchema::object().required_string("path", "File path to read"),
        )
        .with_description("Read the contents of a file.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        // Reject requests to read from .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        match std::fs::read_to_string(&full_path) {
            Ok(content) => {
                // Truncate very large files
                if content.len() > 100_000 {
                    let truncated: String = content.chars().take(100_000).collect();
                    ToolResult::success(format!(
                        "{}\\n\\n... (truncated, {} bytes total)",
                        truncated,
                        content.len()
                    ))
                } else {
                    ToolResult::success(content)
                }
            }
            Err(e) => ToolResult::error(format!("Failed to read file: {}", e)),
        }
    }
}

/// Tool for writing/creating files.
pub struct WriteFileTool {
    project_root: PathBuf,
}

impl WriteFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "write_file",
            InputSchema::object()
                .required_string("path", "File path to write")
                .required_string("content", "Content to write to the file"),
        )
        .with_description("Write content to a file. Creates the file if it doesn't exist.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("Failed to create directories: {}", e));
            }
        }

        match std::fs::write(&full_path, content) {
            Ok(_) => ToolResult::success(format!("Wrote {} bytes to {}", content.len(), path)),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}

/// Tool for editing files with str_replace.
pub struct EditFileTool {
    project_root: PathBuf,
}

impl EditFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "edit_file",
            InputSchema::object()
                .required_string("path", "File path to edit")
                .required_string("old_string", "The exact string to replace")
                .required_string("new_string", "The replacement string"),
        )
        .with_description(
            "Edit a file by replacing an exact string match. The old_string must match exactly.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }
        if old_string.is_empty() {
            return ToolResult::error("old_string is required");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        // Count occurrences
        let count = content.matches(old_string).count();
        if count == 0 {
            return ToolResult::error("old_string not found in file");
        }
        if count > 1 {
            return ToolResult::error(format!(
                "old_string found {} times - must be unique. Provide more context.",
                count
            ));
        }

        let new_content = content.replace(old_string, new_string);

        match std::fs::write(&full_path, &new_content) {
            Ok(_) => ToolResult::success(format!("Successfully edited {}", path)),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}

/// Tool for finding files by glob pattern.
pub struct GlobTool {
    project_root: PathBuf,
}

impl GlobTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "glob",
            InputSchema::object()
                .required_string("pattern", "Glob pattern (e.g., '**/*.rs', 'src/**/*.ts')"),
        )
        .with_description("Find files matching a glob pattern.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");

        if pattern.is_empty() {
            return ToolResult::error("pattern is required");
        }

        let full_pattern = self.project_root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy();

        match glob::glob(&pattern_str) {
            Ok(paths) => {
                let mut results: Vec<String> = paths
                    .flatten()
                    .filter_map(|p| {
                        p.strip_prefix(&self.project_root)
                            .ok()
                            .map(|rel| rel.to_string_lossy().to_string())
                    })
                    .take(100)
                    .collect();
                results.sort();

                if results.is_empty() {
                    ToolResult::success("No files matched the pattern")
                } else {
                    ToolResult::success(results.join("\n"))
                }
            }
            Err(e) => ToolResult::error(format!("Invalid glob pattern: {}", e)),
        }
    }
}

/// Tool for searching file contents.
pub struct GrepTool {
    project_root: PathBuf,
}

impl GrepTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn search_file(
        &self,
        path: &Path,
        pattern: &str,
        results: &mut Vec<String>,
        max_results: usize,
    ) {
        if results.len() >= max_results {
            return;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };

        let relative = path
            .strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy();

        for (i, line) in content.lines().enumerate() {
            if results.len() >= max_results {
                break;
            }
            if line.to_lowercase().contains(&pattern.to_lowercase()) {
                results.push(format!("{}:{}: {}", relative, i + 1, line.trim()));
            }
        }
    }

    fn search_dir(&self, dir: &Path, pattern: &str, results: &mut Vec<String>, max_results: usize) {
        if results.len() >= max_results {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            if results.len() >= max_results {
                break;
            }

            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            // Skip hidden, node_modules, target
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }

            if path.is_dir() {
                self.search_dir(&path, pattern, results, max_results);
            } else if is_text_file(&path) {
                self.search_file(&path, pattern, results, max_results);
            }
        }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "grep",
            InputSchema::object()
                .required_string("pattern", "Search pattern (case-insensitive)")
                .optional_string(
                    "path",
                    "Directory or file to search (defaults to project root)",
                ),
        )
        .with_description(
            "Search for text in files. Returns matching lines with file:line: prefix.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");

        if pattern.is_empty() {
            return ToolResult::error("pattern is required");
        }

        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        // Reject requests to search in .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let search_path = if path == "." {
            self.project_root.clone()
        } else {
            self.project_root.join(path)
        };

        let mut results = Vec::new();
        let max_results = 50;

        if search_path.is_file() {
            self.search_file(&search_path, pattern, &mut results, max_results);
        } else if search_path.is_dir() {
            self.search_dir(&search_path, pattern, &mut results, max_results);
        } else {
            return ToolResult::error(format!("Path not found: {}", path));
        }

        if results.is_empty() {
            ToolResult::success("No matches found")
        } else {
            let truncated = if results.len() >= max_results {
                "\n... (results truncated)"
            } else {
                ""
            };
            ToolResult::success(format!("{}{}", results.join("\n"), truncated))
        }
    }
}

/// Tool for listing directory contents.
pub struct ListDirectoryTool {
    project_root: PathBuf,
}

impl ListDirectoryTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "list_directory",
            InputSchema::object()
                .optional_string("path", "Directory path (defaults to project root)"),
        )
        .with_description("List files and directories. Directories end with /")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        // Reject requests to list .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let full_path = self.resolve_path(path);

        match std::fs::read_dir(&full_path) {
            Ok(entries) => {
                let mut items: Vec<String> = entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        // Skip hidden files and common noise
                        if name.starts_with('.') || name == "node_modules" || name == "target" {
                            return None;
                        }
                        if e.path().is_dir() {
                            Some(format!("{}/", name))
                        } else {
                            Some(name)
                        }
                    })
                    .collect();
                items.sort();
                ToolResult::success(items.join("\n"))
            }
            Err(e) => ToolResult::error(format!("Failed to list directory: {}", e)),
        }
    }
}

/// Check if a file is likely a text file based on extension.
fn is_text_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "java"
            | "rb"
            | "sh"
            | "md"
            | "txt"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
            | "html"
            | "css"
            | "scss"
            | "vue"
            | "svelte"
            | "sql"
            | "graphql"
            | "proto"
            | "xml"
            | "env"
            | "conf"
            | "cfg"
            | "ini"
            | "lock"
            | "sum"
            | "lean"
    )
}

/// Create standard exploration tools for a project.
pub fn create_exploration_tools(project_root: &Path) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(ListDirectoryTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GlobTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GrepTool::new(project_root)) as Arc<dyn Tool>,
    ];

    #[cfg(feature = "search")]
    tools.push(Arc::new(SearchTool::new(project_root)) as Arc<dyn Tool>);

    #[cfg(feature = "smart")]
    {
        tools.push(Arc::new(SmartReadTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(MRSearchTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(AskCodeTool::new(project_root)) as Arc<dyn Tool>);
    }

    tools
}

/// Create standard editing tools for a project.
pub fn create_editing_tools(project_root: &Path) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(WriteFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(EditFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(ListDirectoryTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GlobTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GrepTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(BashTool::new(project_root)) as Arc<dyn Tool>,
    ];

    #[cfg(feature = "smart")]
    {
        tools.push(Arc::new(SmartReadTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(SmartWriteTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(MRSearchTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(AskCodeTool::new(project_root)) as Arc<dyn Tool>);
    }

    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file_tool() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let tool = ReadFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("test.txt"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        assert_eq!(result.to_content_string(), "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_file_tool() {
        let dir = TempDir::new().unwrap();

        let tool = WriteFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("new_file.txt"));
        input.insert("content".to_string(), serde_json::json!("New content!"));

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(dir.path().join("new_file.txt")).unwrap();
        assert_eq!(content, "New content!");
    }

    #[tokio::test]
    async fn test_edit_file_tool() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let tool = EditFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("test.txt"));
        input.insert("old_string".to_string(), serde_json::json!("World"));
        input.insert("new_string".to_string(), serde_json::json!("Rust"));

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_list_directory_tool() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool::new(dir.path());
        let input = HashMap::new();

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("a.txt"));
        assert!(content.contains("b.txt"));
        assert!(content.contains("subdir/"));
    }

    #[tokio::test]
    async fn test_grep_tool() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("test.rs"),
            "fn main() {\n    println!(\"PLACEHOLDER: fix this\");\n}",
        )
        .unwrap();

        let tool = GrepTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("pattern".to_string(), serde_json::json!("PLACEHOLDER"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("PLACEHOLDER"));
        assert!(content.contains("test.rs:2:"));
    }

    #[tokio::test]
    async fn test_bash_tool() {
        let dir = TempDir::new().unwrap();

        let tool = BashTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("command".to_string(), serde_json::json!("echo hello"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        assert!(result.to_content_string().contains("hello"));
    }
}
