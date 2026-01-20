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
                .optional_string("path", "File path (for single read)")
                .optional_string("layer", "Layer: 'raw', 'ast', 'call_graph' (default: 'ast')")
                .optional_string("symbol", "Specific symbol to extract (returns raw)")
                .property("reads", PropertySchema::array(read_item).with_description("Batch reads with individual granularity"), false),
        )
        .with_description(
            "Read code with layered analysis. Single: {path, layer?, symbol?}. Batch: {reads: [{path, layer?, symbol?}, ...]}. Efficiently gathers diverse context in one call.",
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

        // Single file mode
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path or reads is required");
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
        assert!(content.contains("60-80%")); // Token savings note

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
}
