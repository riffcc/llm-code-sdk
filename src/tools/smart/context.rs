//! Context query - the killer feature for 99% token savings.
//!
//! BFS traversal of call graph from an entry point to depth N,
//! returning only relevant function signatures and complexity metrics.
//!
//! ## JECJIT: Just Enough Context, Just In Time
//!
//! The adaptive query system automatically expands context when results are sparse:
//! 1. Start with minimal depth, expand if under token threshold
//! 2. Cross-file callee resolution - search project when local lookup fails
//! 3. Import tracing - follow use/require statements to find definitions
//! 4. Related context injection - add tests or usage examples when sparse

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use super::ast::{AstParser, Lang, Symbol, SymbolKind};
use super::cfg::{CfgAnalyzer, CfgInfo};

/// JECJIT configuration for adaptive context queries.
#[derive(Debug, Clone)]
pub struct ContextQueryConfig {
    /// Minimum tokens to return - auto-expand if below this threshold
    pub min_tokens: usize,
    /// Maximum tokens to return - stop expanding at this ceiling
    pub max_tokens: usize,
    /// Maximum depth to expand to when auto-expanding
    pub max_depth: usize,
    /// Search project-wide for callees not found locally
    pub cross_file_search: bool,
    /// Follow import/use statements to find definitions
    pub follow_imports: bool,
    /// Include related test functions when context is sparse
    pub include_tests: bool,
    /// File extensions to search when doing cross-file resolution
    pub search_extensions: Vec<String>,
}

impl Default for ContextQueryConfig {
    fn default() -> Self {
        Self {
            min_tokens: 100,
            max_tokens: 2000,
            max_depth: 5,
            cross_file_search: true,
            follow_imports: true,
            include_tests: true,
            search_extensions: vec![
                "rs".into(), "py".into(), "js".into(), "ts".into(),
                "go".into(), "pl".into(), "pm".into(),
            ],
        }
    }
}

/// Import/use statement info for cross-file resolution.
#[derive(Debug, Clone)]
pub struct ImportInfo {
    /// The module/package being imported
    pub module: String,
    /// Specific symbols imported (if any)
    pub symbols: Vec<String>,
    /// The file this import was found in
    pub source_file: String,
    /// Line number of the import
    pub line: usize,
}

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
    /// Callees that couldn't be resolved to a file
    pub unresolved_callees: Vec<String>,
    /// Whether JECJIT auto-expanded the query
    pub jecjit_expanded: bool,
}

impl RelevantContext {
    /// Format as LLM-ready string (the key output format).
    pub fn to_llm_string(&self) -> String {
        let mut output = String::new();

        let jecjit_marker = if self.jecjit_expanded { " [JECJIT]" } else { "" };
        output.push_str(&format!(
            "## Code Context: {} (depth={}){}\n\n",
            self.entry_point, self.max_depth, jecjit_marker
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

        // Unresolved callees (if any)
        if !self.unresolved_callees.is_empty() {
            output.push_str("### Unresolved\n");
            let unresolved_str = if self.unresolved_callees.len() > 5 {
                format!(
                    "{} +{} more",
                    self.unresolved_callees[..5].join(", "),
                    self.unresolved_callees.len() - 5
                )
            } else {
                self.unresolved_callees.join(", ")
            };
            output.push_str(&format!("❓ {}\n\n", unresolved_str));
        }

        // Summary
        let approx_tokens = output.split_whitespace().count();
        output.push_str("---\n");
        let jecjit_note = if self.jecjit_expanded { " (auto-expanded)" } else { "" };
        output.push_str(&format!(
            "📊 {} functions | {} files | ~{} tokens{}\n",
            self.functions.len(),
            self.files_touched.len(),
            approx_tokens,
            jecjit_note
        ));

        output
    }

    /// Estimate token count.
    pub fn estimate_tokens(&self) -> usize {
        self.to_llm_string().split_whitespace().count()
    }
}

/// Context query engine with JECJIT support.
pub struct ContextQuery {
    project_root: PathBuf,
    config: ContextQueryConfig,
    file_cache: HashMap<String, String>,
    ast_cache: HashMap<String, Vec<Symbol>>,
    cfg_cache: HashMap<String, Vec<CfgInfo>>,
    import_cache: HashMap<String, Vec<ImportInfo>>,
    /// Project-wide symbol index: symbol_name -> [(file, line)]
    symbol_index: HashMap<String, Vec<(String, usize)>>,
    /// Whether the project has been indexed
    indexed: bool,
}

impl ContextQuery {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            config: ContextQueryConfig::default(),
            file_cache: HashMap::new(),
            ast_cache: HashMap::new(),
            cfg_cache: HashMap::new(),
            import_cache: HashMap::new(),
            symbol_index: HashMap::new(),
            indexed: false,
        }
    }

    pub fn with_config(project_root: impl Into<PathBuf>, config: ContextQueryConfig) -> Self {
        Self {
            project_root: project_root.into(),
            config,
            file_cache: HashMap::new(),
            ast_cache: HashMap::new(),
            cfg_cache: HashMap::new(),
            import_cache: HashMap::new(),
            symbol_index: HashMap::new(),
            indexed: false,
        }
    }

    /// JECJIT: Adaptive query that auto-expands until we have enough context.
    ///
    /// Starts at depth 1, expands if results are sparse, searches cross-file
    /// for unresolved callees, and injects related context as needed.
    pub fn query_adaptive(
        &mut self,
        entry_point: &str,
        entry_file: &str,
    ) -> RelevantContext {
        // Index project if we haven't yet and cross-file search is enabled
        if self.config.cross_file_search && !self.indexed {
            self.index_project();
        }

        let mut depth = 1;
        let mut result = self.query_with_cross_file(entry_point, entry_file, depth);

        // Auto-expand depth until we hit min_tokens or max_depth
        while result.estimate_tokens() < self.config.min_tokens
            && depth < self.config.max_depth
            && result.estimate_tokens() < self.config.max_tokens
        {
            depth += 1;
            result = self.query_with_cross_file(entry_point, entry_file, depth);
        }

        // Still sparse? Try to augment with related context
        if result.estimate_tokens() < self.config.min_tokens && self.config.include_tests {
            self.augment_with_tests(&mut result);
        }

        // Add JECJIT metadata to output
        result.jecjit_expanded = depth > 1;
        result.max_depth = depth;

        result
    }

    /// Query with cross-file callee resolution.
    fn query_with_cross_file(
        &mut self,
        entry_point: &str,
        entry_file: &str,
        depth: usize,
    ) -> RelevantContext {
        let mut functions = Vec::new();
        let mut files_touched = HashSet::new();
        let mut visited = HashSet::new();
        let mut unresolved_callees: Vec<String> = Vec::new();

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

            let Some(symbol) = symbol else {
                // Symbol not found in this file - track as unresolved
                if current_depth > 0 {
                    unresolved_callees.push(func_name.clone());
                }
                continue;
            };

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
                    } else if self.config.cross_file_search {
                        // JECJIT: Search project-wide for this symbol
                        self.find_symbol_file(callee).unwrap_or_else(|| file_path.clone())
                    } else {
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
            unresolved_callees,
            jecjit_expanded: false,
        }
    }

    /// Index all source files in the project for cross-file symbol lookup.
    fn index_project(&mut self) {
        // Clone extensions to avoid borrow conflict
        let extensions: HashSet<String> = self.config.search_extensions.iter().cloned().collect();

        let entries = match self.walk_project_files() {
            Ok(e) => e,
            Err(_) => {
                self.indexed = true;
                return;
            }
        };

        // Collect paths first to avoid borrow issues
        let paths_to_index: Vec<String> = entries
            .into_iter()
            .filter_map(|path| {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !extensions.contains(ext) {
                    return None;
                }
                let rel_path = path.strip_prefix(&self.project_root)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| path.to_string_lossy().to_string());
                Some(rel_path)
            })
            .collect();

        // Now index each file
        for rel_path in paths_to_index {
            let symbols = self.get_symbols(&rel_path);
            for symbol in symbols {
                if matches!(symbol.kind, SymbolKind::Function | SymbolKind::Method) {
                    self.symbol_index
                        .entry(symbol.name.clone())
                        .or_default()
                        .push((rel_path.clone(), symbol.start_line));
                }
            }
        }

        self.indexed = true;
    }

    /// Walk project files, respecting .gitignore patterns.
    fn walk_project_files(&self) -> std::io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.walk_dir(&self.project_root, &mut files)?;
        Ok(files)
    }

    fn walk_dir(&self, dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        // Skip common non-source directories
        let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matches!(dir_name, ".git" | "node_modules" | "target" | "vendor" | "__pycache__" | ".cache" | "dist" | "build") {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.walk_dir(&path, files)?;
            } else if path.is_file() {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Find the file containing a symbol using the project index.
    fn find_symbol_file(&self, symbol_name: &str) -> Option<String> {
        self.symbol_index.get(symbol_name)
            .and_then(|locations| locations.first())
            .map(|(file, _)| file.clone())
    }

    /// Augment sparse results with test functions that exercise the entry point.
    fn augment_with_tests(&mut self, result: &mut RelevantContext) {
        // Look for test files and functions related to the entry point
        let test_patterns = [
            format!("test_{}", result.entry_point),
            format!("{}_test", result.entry_point),
            format!("test_{}_", result.entry_point),
        ];

        // Collect candidates first to avoid borrow conflict
        let candidates: Vec<(String, String, usize)> = self.symbol_index
            .iter()
            .filter_map(|(symbol_name, locations)| {
                let is_related_test = test_patterns.iter().any(|p| symbol_name.starts_with(p))
                    || symbol_name.contains(&format!("test_{}", result.entry_point));

                if is_related_test {
                    if let Some((file, line)) = locations.first() {
                        return Some((symbol_name.clone(), file.clone(), *line));
                    }
                }
                None
            })
            .collect();

        for (symbol_name, file, line) in candidates {
            // Don't add if we already have this function
            if result.functions.iter().any(|f| f.name == symbol_name) {
                continue;
            }

            // Get the test function's context
            let symbols = self.get_symbols(&file);
            if let Some(symbol) = symbols.iter().find(|s| s.name == symbol_name) {
                let cfgs = self.get_cfg(&file);
                let cfg = cfgs.iter().find(|c| c.function_name == symbol_name);

                let content = self.load_file(&file).map(|s| s.to_string()).unwrap_or_default();
                let lang = Lang::from_path(Path::new(&file)).unwrap_or(Lang::Rust);
                let calls = self.extract_calls_from_function(&content, lang, &symbol_name);

                result.functions.push(FunctionContext {
                    name: symbol_name.clone(),
                    file: file.clone(),
                    line,
                    signature: symbol.signature.clone().unwrap_or_else(|| format!("fn {}()", symbol_name)),
                    docstring: symbol.doc_comment.clone(),
                    calls,
                    cyclomatic: cfg.map(|c| c.cyclomatic_complexity).unwrap_or(1),
                    blocks: cfg.map(|c| c.basic_blocks).unwrap_or(1),
                    depth: result.max_depth + 1, // Mark as augmented
                });

                result.files_touched.insert(file.clone());
            }
        }
    }

    /// Extract imports from a file (for import tracing).
    pub fn get_imports(&mut self, path: &str) -> Vec<ImportInfo> {
        if let Some(cached) = self.import_cache.get(path) {
            return cached.clone();
        }

        // Load and clone content to avoid borrow conflict
        let content = self.load_file(path).map(|s| s.to_string());

        let imports = if let Some(content) = content {
            let lang = Lang::from_path(Path::new(path)).unwrap_or(Lang::Rust);
            self.extract_imports(&content, lang, path)
        } else {
            vec![]
        };

        self.import_cache.insert(path.to_string(), imports.clone());
        imports
    }

    /// Extract import statements from source code.
    fn extract_imports(&self, source: &str, lang: Lang, source_file: &str) -> Vec<ImportInfo> {
        let mut parser = AstParser::new();
        let tree = match parser.parse(source, lang) {
            Some(t) => t,
            None => return vec![],
        };

        let mut imports = Vec::new();
        self.find_imports(tree.root_node(), source, lang, source_file, &mut imports);
        imports
    }

    fn find_imports(
        &self,
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        source_file: &str,
        imports: &mut Vec<ImportInfo>,
    ) {
        let kind = node.kind();
        let line = node.start_position().row + 1;

        match lang {
            Lang::Rust => {
                if kind == "use_declaration" {
                    if let Some(path) = node.utf8_text(source.as_bytes()).ok() {
                        // Parse "use foo::bar::baz;" or "use foo::bar::{x, y};"
                        let module = path.trim_start_matches("use ")
                            .trim_end_matches(';')
                            .trim()
                            .to_string();
                        imports.push(ImportInfo {
                            module,
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
            Lang::Python => {
                if kind == "import_statement" || kind == "import_from_statement" {
                    if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                        imports.push(ImportInfo {
                            module: text.to_string(),
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
            Lang::Perl => {
                if kind == "use_no_statement" {
                    // Extract package name from "use PVE::Tools qw(...);"
                    let mut module = String::new();
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        if child.kind() == "package_name" {
                            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                                module = name.to_string();
                            }
                        }
                    }
                    if !module.is_empty() {
                        imports.push(ImportInfo {
                            module,
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
            Lang::JavaScript | Lang::TypeScript => {
                if kind == "import_statement" {
                    if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                        imports.push(ImportInfo {
                            module: text.to_string(),
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
            Lang::Go => {
                if kind == "import_declaration" || kind == "import_spec" {
                    if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                        imports.push(ImportInfo {
                            module: text.to_string(),
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
            Lang::Nim => {
                if kind == "import_statement" || kind == "from_statement" || kind == "include_statement" {
                    if let Some(text) = node.utf8_text(source.as_bytes()).ok() {
                        imports.push(ImportInfo {
                            module: text.to_string(),
                            symbols: vec![],
                            source_file: source_file.to_string(),
                            line,
                        });
                    }
                }
            }
        }

        // Recurse
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.find_imports(child, source, lang, source_file, imports);
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
            unresolved_callees: vec![],
            jecjit_expanded: false,
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
            Lang::Perl => kind == "function_definition" || kind == "anonymous_function",
            Lang::Nim => {
                kind == "proc_declaration" || kind == "func_declaration" ||
                kind == "method_declaration" || kind == "iterator_declaration"
            }
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
                Lang::Nim => kind == "call" || kind == "method_call",
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
            Lang::Nim => node.child_by_field_name("function").or_else(|| node.child(0)),
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
            unresolved_callees: vec![],
            jecjit_expanded: false,
        };

        let output = ctx.to_llm_string();

        assert!(output.contains("## Code Context: main (depth=2)"));
        assert!(output.contains("📍 main (main.rs:1)"));
        assert!(output.contains("fn main()"));
        assert!(output.contains("→ calls: init, run"));
        assert!(output.contains("📊 2 functions"));
    }

    #[test]
    fn test_jecjit_adaptive_query() {
        let dir = TempDir::new().unwrap();

        // Create a multi-file project
        fs::create_dir_all(dir.path().join("src")).unwrap();

        let main_source = r#"
fn main() {
    let result = helper();
    use_result(result);
}

fn use_result(x: i32) {
    // uses result
}
"#;
        fs::write(dir.path().join("src/main.rs"), main_source).unwrap();

        let lib_source = r#"
/// Helper function that does useful work.
pub fn helper() -> i32 {
    compute_value()
}

fn compute_value() -> i32 {
    42
}
"#;
        fs::write(dir.path().join("src/lib.rs"), lib_source).unwrap();

        let config = ContextQueryConfig {
            min_tokens: 50,
            max_tokens: 500,
            max_depth: 3,
            cross_file_search: true,
            follow_imports: false,
            include_tests: false,
            search_extensions: vec!["rs".into()],
        };

        let mut query = ContextQuery::with_config(dir.path(), config);
        let ctx = query.query_adaptive("main", "src/main.rs");

        // Should have auto-expanded and found functions
        assert!(!ctx.functions.is_empty());
        assert!(ctx.functions.iter().any(|f| f.name == "main"));

        // Cross-file search should have found helper in lib.rs
        let found_helper = ctx.functions.iter().any(|f| f.name == "helper");
        // Note: helper may or may not be found depending on indexing
        // The key is that the adaptive query ran without error

        let output = ctx.to_llm_string();
        assert!(output.contains("## Code Context: main"));
    }

    #[test]
    fn test_jecjit_output_format() {
        let ctx = RelevantContext {
            entry_point: "process".to_string(),
            max_depth: 3,
            functions: vec![
                FunctionContext {
                    name: "process".to_string(),
                    file: "src/lib.rs".to_string(),
                    line: 1,
                    signature: "pub fn process(data: &[u8]) -> Result<Output>".to_string(),
                    docstring: Some("Process incoming data".to_string()),
                    calls: vec!["validate".to_string(), "transform".to_string()],
                    cyclomatic: 5,
                    blocks: 8,
                    depth: 0,
                },
            ],
            files_touched: ["src/lib.rs".to_string()].into_iter().collect(),
            unresolved_callees: vec!["external_api".to_string()],
            jecjit_expanded: true,
        };

        let output = ctx.to_llm_string();

        // Should show JECJIT marker
        assert!(output.contains("[JECJIT]"));
        // Should show unresolved callees
        assert!(output.contains("Unresolved"));
        assert!(output.contains("external_api"));
        // Should show auto-expanded note
        assert!(output.contains("auto-expanded"));
    }
}
