//! SmartRead tool - reads code using the layercake approach.
//!
//! Instead of dumping raw code, presents structured representations
//! at the appropriate layer for token efficiency.
//!
//! Supports tree-style batch reads for efficient multi-file context gathering:
//! ```json
//! {
//!   "reads": [
//!     {"path": "src/main.rs", "layer": "ast"},
//!     {"path": "src/lib.rs", "layer": "call_graph"},
//!     {"path": "src/utils.rs", "symbol": "helper", "layer": "raw"}
//!   ]
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::ast::{AstParser, Lang};
use super::layers::{CodeLayer, LayerAnalyzer, LayerView};
use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// A single read request in a batch.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub path: String,
    pub layer: CodeLayer,
    pub symbol: Option<String>, // Optional: read only this symbol at raw level
}

impl ReadRequest {
    pub fn new(path: &str, layer: CodeLayer) -> Self {
        Self {
            path: path.to_string(),
            layer,
            symbol: None,
        }
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }
}

/// SmartRead tool for token-efficient code reading.
pub struct SmartReadTool {
    project_root: PathBuf,
    analyzer: Arc<RwLock<LayerAnalyzer>>,
}

impl SmartReadTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
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

    /// Read a file at the specified layer.
    pub fn read_at_layer(&self, path: &str, layer: CodeLayer) -> Result<LayerView, String> {
        let full_path = self.resolve_path(path);

        // Security check
        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Path must be within project root".to_string());
            }
        }

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut analyzer = self.analyzer.write().unwrap();
        Ok(analyzer.analyze(path, &content, layer))
    }

    /// Read a file with automatic layer selection.
    pub fn read_smart(&self, path: &str) -> Result<LayerView, String> {
        let full_path = self.resolve_path(path);

        // Security check
        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Path must be within project root".to_string());
            }
        }

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut analyzer = self.analyzer.write().unwrap();
        Ok(analyzer.analyze_efficient(path, &content))
    }

    /// Read an entire folder, returning compressed AST view of all code files.
    ///
    /// If `recursive` is true, includes subdirectories.
    /// Returns a tree-structured view showing files and their symbols.
    pub fn read_folder(&self, path: &str, recursive: bool) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        // Security check
        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Path must be within project root".to_string());
            }
        }

        if !full_path.is_dir() {
            return Err(format!("{} is not a directory", path));
        }

        // Collect all code files
        let mut files: Vec<PathBuf> = Vec::new();
        self.collect_code_files(&full_path, recursive, &mut files)?;

        if files.is_empty() {
            return Ok(format!("📂 {} (empty or no code files)", path));
        }

        // Sort for consistent output
        files.sort();

        // Build compressed output
        let mut output = format!("📂 {} ({} files)\n\n", path, files.len());

        for file_path in &files {
            let relative = file_path.strip_prefix(&self.project_root)
                .unwrap_or(file_path)
                .to_string_lossy();

            // Read at AST layer for compression
            match self.read_at_layer(&relative, CodeLayer::Ast) {
                Ok(view) => {
                    let content = view.to_context();
                    // Extract just the symbols line (compact)
                    let symbols: Vec<&str> = content.lines()
                        .filter(|l| l.starts_with("- ") || l.starts_with("  - "))
                        .collect();

                    if symbols.is_empty() {
                        output.push_str(&format!("├── {}\n", relative));
                    } else {
                        output.push_str(&format!("├── {} ({})\n", relative, symbols.len()));
                        for sym in symbols.iter().take(10) {
                            output.push_str(&format!("│   {}\n", sym));
                        }
                        if symbols.len() > 10 {
                            output.push_str(&format!("│   ... and {} more\n", symbols.len() - 10));
                        }
                    }
                }
                Err(e) => {
                    output.push_str(&format!("├── {} (error: {})\n", relative, e));
                }
            }
        }

        Ok(output)
    }

    /// Collect code files from a directory.
    fn collect_code_files(&self, dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            // Skip hidden files/dirs
            if path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false)
            {
                continue;
            }

            if path.is_dir() {
                // Skip common non-code directories
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(name, "target" | "node_modules" | "dist" | "build" | "__pycache__" | ".git") {
                    continue;
                }
                if recursive {
                    self.collect_code_files(&path, recursive, files)?;
                }
            } else if Self::is_code_file(&path) {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Check if a file is a recognized code file.
    fn is_code_file(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| matches!(ext,
                "rs" | "py" | "js" | "ts" | "tsx" | "jsx" |
                "go" | "java" | "c" | "cpp" | "h" | "hpp" |
                "rb" | "php" | "swift" | "kt" | "scala" | "zig"
            ))
            .unwrap_or(false)
    }

    /// Read a specific symbol from a file (returns raw code for just that symbol).
    pub fn read_symbol(&self, path: &str, symbol_name: &str) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        // Security check
        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Path must be within project root".to_string());
            }
        }

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path)
            .ok_or_else(|| "Unsupported language".to_string())?;

        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(&content, lang);

        let symbol = symbols
            .iter()
            .find(|s| s.name == symbol_name)
            .ok_or_else(|| format!("Symbol '{}' not found in {}", symbol_name, path))?;

        let lines: Vec<&str> = content.lines().collect();
        let symbol_lines: Vec<&str> = lines[symbol.start_line - 1..symbol.end_line].to_vec();

        Ok(format!(
            "## {} (lines {}-{})\n```\n{}\n```",
            symbol_name,
            symbol.start_line,
            symbol.end_line,
            symbol_lines.join("\n")
        ))
    }

    /// Batch read multiple files/symbols with individual granularity.
    /// This is the key efficiency feature - gather diverse context in one call.
    pub fn read_batch(&self, requests: &[ReadRequest]) -> Vec<Result<String, String>> {
        requests
            .iter()
            .map(|req| {
                if let Some(symbol) = &req.symbol {
                    // Symbol-specific read (always raw)
                    self.read_symbol(&req.path, symbol)
                } else {
                    // Full file at specified layer
                    self.read_at_layer(&req.path, req.layer.clone())
                        .map(|view| view.to_context())
                }
            })
            .collect()
    }

    /// Batch read and combine into single context string.
    pub fn read_tree(&self, requests: &[ReadRequest]) -> String {
        let results = self.read_batch(requests);
        let mut output = String::new();

        for (req, result) in requests.iter().zip(results.iter()) {
            output.push_str(&format!("\n─── {} ", req.path));
            if let Some(sym) = &req.symbol {
                output.push_str(&format!("({}) ", sym));
            }
            output.push_str(&format!("[{:?}] ───\n", req.layer));

            match result {
                Ok(content) => output.push_str(content),
                Err(e) => output.push_str(&format!("Error: {}", e)),
            }
            output.push_str("\n");
        }

        output
    }
}

#[async_trait]
impl Tool for SmartReadTool {
    fn name(&self) -> &str {
        "smart_read"
    }

    fn to_param(&self) -> ToolParam {
        use crate::types::PropertySchema;

        // Build the read request item schema
        let read_item = PropertySchema::object()
            .property("path", PropertySchema::string().with_description("File path"), true)
            .property("layer", PropertySchema::string().with_description("Layer: raw, ast, call_graph"), false)
            .property("symbol", PropertySchema::string().with_description("Specific symbol to extract"), false);

        ToolParam::new(
            "smart_read",
            InputSchema::object()
                .optional_string("path", "File or folder path")
                .optional_string("layer", "Layer: 'raw', 'ast', 'call_graph' (default: 'ast')")
                .optional_string("symbol", "Specific symbol to extract (returns raw)")
                .property("recursive", PropertySchema::boolean().with_description("For folders: include subdirectories (default: true)"), false)
                .property("reads", PropertySchema::array(read_item).with_description("Batch reads with individual granularity"), false),
        )
        .with_description(
            "Read code with layered analysis. File: {path, layer?, symbol?}. Folder: {path, recursive?}. Batch: {reads: [...]}.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        // Check for batch mode first
        if let Some(reads) = input.get("reads").and_then(|v| v.as_array()) {
            let requests: Vec<ReadRequest> = reads
                .iter()
                .filter_map(|r| {
                    let path = r.get("path")?.as_str()?;
                    let layer = parse_layer(r.get("layer").and_then(|v| v.as_str()));
                    let symbol = r.get("symbol").and_then(|v| v.as_str()).map(|s| s.to_string());

                    Some(ReadRequest {
                        path: path.to_string(),
                        layer,
                        symbol,
                    })
                })
                .collect();

            if requests.is_empty() {
                return ToolResult::error("reads array is empty or invalid");
            }

            return ToolResult::success(self.read_tree(&requests));
        }

        // Single path mode
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path or reads is required");
        }

        let full_path = self.resolve_path(path);

        // Check if path is a directory -> folder compression
        if full_path.is_dir() {
            let recursive = input.get("recursive")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            return match self.read_folder(path, recursive) {
                Ok(content) => ToolResult::success(content),
                Err(e) => ToolResult::error(e),
            };
        }

        // Check for symbol-specific read
        if let Some(symbol) = input.get("symbol").and_then(|v| v.as_str()) {
            return match self.read_symbol(path, symbol) {
                Ok(content) => ToolResult::success(content),
                Err(e) => ToolResult::error(e),
            };
        }

        let layer = parse_layer(input.get("layer").and_then(|v| v.as_str()));

        match self.read_at_layer(path, layer) {
            Ok(view) => ToolResult::success(view.to_context()),
            Err(e) => ToolResult::error(e),
        }
    }
}

fn parse_layer(s: Option<&str>) -> CodeLayer {
    match s.unwrap_or("ast") {
        "raw" => CodeLayer::Raw,
        "ast" => CodeLayer::Ast,
        "call_graph" | "callgraph" => CodeLayer::CallGraph,
        "cfg" => CodeLayer::Cfg,
        "dfg" => CodeLayer::Dfg,
        "pdg" => CodeLayer::Pdg,
        _ => CodeLayer::Ast,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_smart_read_tool() {
        let dir = TempDir::new().unwrap();

        let source = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

struct Calculator {
    value: i32,
}
"#;

        fs::write(dir.path().join("calc.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Test AST layer
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert("layer".to_string(), serde_json::json!("ast"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("add"));
        assert!(content.contains("Calculator"));

        // Test raw layer
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert("layer".to_string(), serde_json::json!("raw"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("pub fn add(a: i32, b: i32)"));
    }

    #[tokio::test]
    async fn test_symbol_read() {
        let dir = TempDir::new().unwrap();

        let source = r#"pub fn first() {
    println!("First function");
}

pub fn second() {
    println!("Second function");
}

pub fn third() {
    println!("Third function");
}
"#;

        fs::write(dir.path().join("funcs.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Read just one function
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("funcs.rs"));
        input.insert("symbol".to_string(), serde_json::json!("second"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should have the second function
        assert!(content.contains("Second function"));
        // Should NOT have the other functions' content
        assert!(!content.contains("First function"));
        assert!(!content.contains("Third function"));
    }

    #[tokio::test]
    async fn test_batch_read() {
        let dir = TempDir::new().unwrap();

        let math_source = r#"pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"#;

        let utils_source = r#"pub fn helper() {
    println!("Helper");
}

pub fn format_output(s: &str) -> String {
    format!("[{}]", s)
}
"#;

        fs::write(dir.path().join("math.rs"), math_source).unwrap();
        fs::write(dir.path().join("utils.rs"), utils_source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Batch read with different granularities
        let mut input = HashMap::new();
        input.insert("reads".to_string(), serde_json::json!([
            {"path": "math.rs", "layer": "ast"},
            {"path": "utils.rs", "layer": "ast"},
            {"path": "math.rs", "symbol": "multiply"}
        ]));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should have AST view of math.rs
        assert!(content.contains("math.rs"));
        // Should have AST view of utils.rs
        assert!(content.contains("utils.rs"));
        // Should have raw code for multiply specifically
        assert!(content.contains("a * b"));
        // Should have format_output from utils
        assert!(content.contains("format_output"));
    }

    #[test]
    fn test_read_tree() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn main() {
    println!("Main");
}

fn helper() {
    println!("Helper");
}
"#;

        fs::write(dir.path().join("main.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        let requests = vec![
            ReadRequest::new("main.rs", CodeLayer::Ast),
            ReadRequest::new("main.rs", CodeLayer::Raw),
        ];

        let result = tool.read_tree(&requests);

        // Should have both views
        assert!(result.contains("[Ast]"));
        assert!(result.contains("[Raw]"));
        assert!(result.contains("main.rs"));
    }

    #[tokio::test]
    async fn test_read_folder() {
        let dir = TempDir::new().unwrap();

        // Create a src subdirectory
        let src = dir.path().join("src");
        fs::create_dir(&src).unwrap();

        let main_source = r#"fn main() {
    lib::greet();
}
"#;
        let lib_source = r#"pub fn greet() {
    println!("Hello");
}

pub fn farewell() {
    println!("Goodbye");
}
"#;

        fs::write(src.join("main.rs"), main_source).unwrap();
        fs::write(src.join("lib.rs"), lib_source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Read the src folder
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("src"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should show folder structure
        assert!(content.contains("📂 src"));
        assert!(content.contains("2 files"));
        // Should list both files
        assert!(content.contains("main.rs"));
        assert!(content.contains("lib.rs"));
        // Should show symbols from lib.rs
        assert!(content.contains("greet"));
        assert!(content.contains("farewell"));
    }

    #[test]
    fn test_read_folder_non_recursive() {
        let dir = TempDir::new().unwrap();

        // Create nested structure
        let src = dir.path().join("src");
        let nested = src.join("nested");
        fs::create_dir_all(&nested).unwrap();

        fs::write(src.join("top.rs"), "fn top() {}").unwrap();
        fs::write(nested.join("deep.rs"), "fn deep() {}").unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Non-recursive read
        let result = tool.read_folder("src", false).unwrap();

        // Should have top.rs
        assert!(result.contains("top.rs"));
        // Should NOT have deep.rs (it's in a subdirectory)
        assert!(!result.contains("deep.rs"));
    }
}
