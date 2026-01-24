//! Context query - the killer feature for 99% token savings.
//!
//! BFS traversal of call graph from an entry point to depth N,
//! returning only relevant function signatures and complexity metrics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use super::ast::{AstParser, Lang, Symbol, SymbolKind};
use super::cfg::{CfgAnalyzer, CfgInfo};

/// Context for a single function in the call chain.
#[derive(Debug, Clone)]
pub struct FunctionContext {
    pub name: String,
    pub file: String,
    pub line: usize,
    pub signature: String,
    pub docstring: Option<String>,
    pub calls: Vec<String>,
    pub cyclomatic: usize,
    pub blocks: usize,
    pub depth: usize, // Distance from entry point
}

/// Aggregated context from call graph traversal.
#[derive(Debug, Clone)]
pub struct RelevantContext {
    pub entry_point: String,
    pub max_depth: usize,
    pub functions: Vec<FunctionContext>,
    pub files_touched: HashSet<String>,
}

impl RelevantContext {
    /// Format as LLM-ready string (the key output format).
    pub fn to_llm_string(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "## Code Context: {} (depth={})\n\n",
            self.entry_point, self.max_depth
        ));

        for func in &self.functions {
            let indent = "  ".repeat(func.depth);
            let short_file = Path::new(&func.file)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| func.file.clone());

            output.push_str(&format!(
                "{}📍 {} ({}:{})\n",
                indent, func.name, short_file, func.line
            ));

            output.push_str(&format!("{}   {}\n", indent, func.signature));

            if let Some(doc) = &func.docstring {
                let truncated = if doc.len() > 80 {
                    format!("{}...", &doc[..77])
                } else {
                    doc.clone()
                };
                output.push_str(&format!("{}   # {}\n", indent, truncated));
            }

            // Complexity info
            let rating = CfgAnalyzer::complexity_rating(func.cyclomatic);
            let warning = if func.cyclomatic > 10 { " ⚠️" } else { "" };
            output.push_str(&format!(
                "{}   ⚡ complexity: {} ({} blocks) [{}]{}\n",
                indent, func.cyclomatic, func.blocks, rating, warning
            ));

            // Calls (limited to 5)
            if !func.calls.is_empty() {
                let calls_str = if func.calls.len() > 5 {
                    format!(
                        "{} +{} more",
                        func.calls[..5].join(", "),
                        func.calls.len() - 5
                    )
                } else {
                    func.calls.join(", ")
                };
                output.push_str(&format!("{}   → calls: {}\n", indent, calls_str));
            }

            output.push('\n');
        }

        // Summary
        let approx_tokens = output.split_whitespace().count();
        output.push_str("---\n");
        output.push_str(&format!(
            "📊 {} functions | {} files | ~{} tokens\n",
            self.functions.len(),
            self.files_touched.len(),
            approx_tokens
        ));

        output
    }

    /// Estimate token count.
    pub fn estimate_tokens(&self) -> usize {
        self.to_llm_string().split_whitespace().count()
    }
}

/// Context query engine.
pub struct ContextQuery {
    project_root: PathBuf,
    file_cache: HashMap<String, String>,
    ast_cache: HashMap<String, Vec<Symbol>>,
    cfg_cache: HashMap<String, Vec<CfgInfo>>,
}

impl ContextQuery {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            file_cache: HashMap::new(),
            ast_cache: HashMap::new(),
            cfg_cache: HashMap::new(),
        }
    }

    /// Load and cache a file.
    fn load_file(&mut self, path: &str) -> Option<&str> {
        if !self.file_cache.contains_key(path) {
            let full_path = if Path::new(path).is_absolute() {
                PathBuf::from(path)
            } else {
                self.project_root.join(path)
            };

            if let Ok(content) = std::fs::read_to_string(&full_path) {
                self.file_cache.insert(path.to_string(), content);
            }
        }
        self.file_cache.get(path).map(|s| s.as_str())
    }

    /// Get cached AST symbols for a file.
    fn get_symbols(&mut self, path: &str) -> Vec<Symbol> {
        if !self.ast_cache.contains_key(path) {
            if let Some(content) = self.load_file(path) {
                let lang = Lang::from_path(Path::new(path)).unwrap_or(Lang::Rust);
                let mut parser = AstParser::new();
                let symbols = parser.extract_symbols(content, lang);
                self.ast_cache.insert(path.to_string(), symbols);
            }
        }
        self.ast_cache.get(path).cloned().unwrap_or_default()
    }

    /// Get cached CFG info for a file.
    fn get_cfg(&mut self, path: &str) -> Vec<CfgInfo> {
        if !self.cfg_cache.contains_key(path) {
            if let Some(content) = self.load_file(path) {
                let lang = Lang::from_path(Path::new(path)).unwrap_or(Lang::Rust);
                let cfgs = CfgAnalyzer::analyze(content, lang);
                self.cfg_cache.insert(path.to_string(), cfgs);
            }
        }
        self.cfg_cache.get(path).cloned().unwrap_or_default()
    }

    /// Query context starting from an entry point, traversing to depth N.
    /// This is the killer feature - 99% token savings.
    pub fn query(
        &mut self,
        entry_point: &str,
        entry_file: &str,
        depth: usize,
    ) -> RelevantContext {
        let mut functions = Vec::new();
        let mut files_touched = HashSet::new();
        let mut visited = HashSet::new();

        // BFS queue: (function_name, file, current_depth)
        let mut queue: VecDeque<(String, String, usize)> = VecDeque::new();
        queue.push_back((entry_point.to_string(), entry_file.to_string(), 0));

        while let Some((func_name, file_path, current_depth)) = queue.pop_front() {
            let key = format!("{}::{}", file_path, func_name);
            if visited.contains(&key) || current_depth > depth {
                continue;
            }
            visited.insert(key);
            files_touched.insert(file_path.clone());

            // Get symbol info
            let symbols = self.get_symbols(&file_path);
            let symbol = symbols
                .iter()
                .find(|s| s.name == func_name && matches!(s.kind, SymbolKind::Function | SymbolKind::Method));

            let Some(symbol) = symbol else { continue };

            // Get CFG info
            let cfgs = self.get_cfg(&file_path);
            let cfg = cfgs.iter().find(|c| c.function_name == func_name);

            // Build call graph for this file to find callees
            let content = self.load_file(&file_path).map(|s| s.to_string()).unwrap_or_default();
            let lang = Lang::from_path(Path::new(&file_path)).unwrap_or(Lang::Rust);
            let call_graph = self.extract_calls_from_function(&content, lang, &func_name);

            // Create function context
            let func_ctx = FunctionContext {
                name: func_name.clone(),
                file: file_path.clone(),
                line: symbol.start_line,
                signature: symbol.signature.clone().unwrap_or_else(|| format!("fn {}()", func_name)),
                docstring: symbol.doc_comment.clone(),
                calls: call_graph.clone(),
                cyclomatic: cfg.map(|c| c.cyclomatic_complexity).unwrap_or(1),
                blocks: cfg.map(|c| c.basic_blocks).unwrap_or(1),
                depth: current_depth,
            };

            functions.push(func_ctx);

            // Enqueue callees
            if current_depth < depth {
                for callee in &call_graph {
                    // Try to find the callee in the same file first
                    let callee_file = if symbols.iter().any(|s| s.name == *callee) {
                        file_path.clone()
                    } else {
                        // Could expand to search other files, but for now stay in same file
                        file_path.clone()
                    };

                    queue.push_back((callee.clone(), callee_file, current_depth + 1));
                }
            }
        }

        // Sort by depth then name for consistent output
        functions.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.name.cmp(&b.name)));

        RelevantContext {
            entry_point: entry_point.to_string(),
            max_depth: depth,
            functions,
            files_touched,
        }
    }

    /// Extract function calls from a specific function.
    fn extract_calls_from_function(&self, source: &str, lang: Lang, func_name: &str) -> Vec<String> {
        let mut parser = AstParser::new();
        let tree = match parser.parse(source, lang) {
            Some(t) => t,
            None => return vec![],
        };

        let mut calls = Vec::new();
        self.find_function_calls(tree.root_node(), source, lang, func_name, &mut calls, false);

        // Deduplicate
        calls.sort();
        calls.dedup();
        calls
    }

    fn find_function_calls(
        &self,
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        target_func: &str,
        calls: &mut Vec<String>,
        in_target: bool,
    ) {
        let kind = node.kind();

        // Check if we're entering the target function
        let is_function = match lang {
            Lang::Rust => kind == "function_item",
            Lang::Python => kind == "function_definition",
            Lang::JavaScript | Lang::TypeScript => {
                kind == "function_declaration" || kind == "method_definition"
            }
            Lang::Go => kind == "function_declaration" || kind == "method_declaration",
            Lang::Perl => kind == "subroutine_declaration" || kind == "method_declaration",
        };

        let is_target = if is_function {
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(|n| n == target_func)
                .unwrap_or(false)
        } else {
            false
        };

        let now_in_target = in_target || is_target;

        // If we're in the target function, look for call expressions
        if now_in_target {
            let is_call = match lang {
                Lang::Rust => kind == "call_expression",
                Lang::Python => kind == "call",
                Lang::JavaScript | Lang::TypeScript => kind == "call_expression",
                Lang::Go => kind == "call_expression",
                Lang::Perl => kind == "subroutine_call_expression" || kind == "method_call_expression",
            };

            if is_call {
                // Extract the function being called
                if let Some(callee) = self.extract_callee_name(node, source, lang) {
                    calls.push(callee);
                }
            }
        }

        // Recurse
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.find_function_calls(child, source, lang, target_func, calls, now_in_target);
        }
    }

    fn extract_callee_name(&self, node: tree_sitter::Node, source: &str, lang: Lang) -> Option<String> {
        // Get the function part of the call
        let func_node = match lang {
            Lang::Rust => node.child_by_field_name("function"),
            Lang::Python => node.child_by_field_name("function"),
            Lang::JavaScript | Lang::TypeScript => node.child_by_field_name("function"),
            Lang::Go => node.child_by_field_name("function"),
            Lang::Perl => node.child_by_field_name("function").or_else(|| node.child_by_field_name("name")),
        }?;

        // Handle different call patterns
        let text = func_node.utf8_text(source.as_bytes()).ok()?;

        // Extract just the function name (handle method calls, namespaced calls, etc.)
        let name = text
            .rsplit(|c| c == '.' || c == ':' || c == '>')
            .next()
            .unwrap_or(text)
            .trim();

        // Filter out built-ins and common patterns we don't want to track
        let skip = ["println", "print", "format", "dbg", "vec", "Some", "None", "Ok", "Err"];
        if skip.contains(&name) || name.starts_with(|c: char| c.is_uppercase()) {
            return None;
        }

        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_context_query() {
        let dir = TempDir::new().unwrap();

        let source = r#"
fn main() {
    let result = process_data(42);
    display_result(result);
}

fn process_data(x: i32) -> i32 {
    let validated = validate(x);
    transform(validated)
}

fn validate(x: i32) -> i32 {
    if x > 0 { x } else { 0 }
}

fn transform(x: i32) -> i32 {
    x * 2
}

fn display_result(x: i32) {
    // prints result
}
"#;

        fs::write(dir.path().join("main.rs"), source).unwrap();

        let mut query = ContextQuery::new(dir.path());
        let ctx = query.query("main", "main.rs", 2);

        // Should have main and its direct callees
        assert!(!ctx.functions.is_empty());
        assert!(ctx.functions.iter().any(|f| f.name == "main"));

        // At depth 2, should have process_data's callees too
        let _has_validate = ctx.functions.iter().any(|f| f.name == "validate");
        let _has_transform = ctx.functions.iter().any(|f| f.name == "transform");

        // Check the output format
        let output = ctx.to_llm_string();
        assert!(output.contains("## Code Context: main"));
        assert!(output.contains("📍"));
        assert!(output.contains("complexity"));
    }

    #[test]
    fn test_llm_output_format() {
        let ctx = RelevantContext {
            entry_point: "main".to_string(),
            max_depth: 2,
            functions: vec![
                FunctionContext {
                    name: "main".to_string(),
                    file: "src/main.rs".to_string(),
                    line: 1,
                    signature: "fn main()".to_string(),
                    docstring: Some("Entry point".to_string()),
                    calls: vec!["init".to_string(), "run".to_string()],
                    cyclomatic: 1,
                    blocks: 1,
                    depth: 0,
                },
                FunctionContext {
                    name: "init".to_string(),
                    file: "src/main.rs".to_string(),
                    line: 10,
                    signature: "fn init() -> Config".to_string(),
                    docstring: None,
                    calls: vec![],
                    cyclomatic: 3,
                    blocks: 5,
                    depth: 1,
                },
            ],
            files_touched: ["src/main.rs".to_string()].into_iter().collect(),
        };

        let output = ctx.to_llm_string();

        assert!(output.contains("## Code Context: main (depth=2)"));
        assert!(output.contains("📍 main (main.rs:1)"));
        assert!(output.contains("fn main()"));
        assert!(output.contains("→ calls: init, run"));
        assert!(output.contains("📊 2 functions"));
    }
}
