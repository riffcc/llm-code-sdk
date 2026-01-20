//! Granular query API - swiss army knife for code understanding.
//!
//! Each query is small, focused, and composable. The model chooses
//! what combination it needs based on the task at hand.
//!
//! Query types:
//! - Structure: symbols, signatures, types
//! - Complexity: cyclomatic, nesting depth
//! - Flow: callers, callees, data flow
//! - Slicing: backward (what affects X), forward (what X affects)
//! - Search: find symbol, find usage

use std::collections::HashSet;
use std::path::Path;

use super::ast::{AstParser, Lang, Symbol, SymbolKind};
use super::cfg::{CfgAnalyzer, CfgInfo};
use super::dfg::{DfgAnalyzer, DfgInfo, VarRef, RefType};
use super::pdg::{PdgBuilder, PdgInfo};
use super::call_graph::CallGraph;

/// Result of any query - uniform interface.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query: String,
    pub content: String,
    pub tokens: usize,
    pub metadata: QueryMetadata,
}

/// Metadata about the query result.
#[derive(Debug, Clone, Default)]
pub struct QueryMetadata {
    pub symbols_found: usize,
    pub lines_returned: usize,
    pub functions_analyzed: usize,
}

impl QueryResult {
    fn new(query: &str, content: String) -> Self {
        let tokens = content.split_whitespace().count();
        Self {
            query: query.to_string(),
            content,
            tokens,
            metadata: QueryMetadata::default(),
        }
    }

    fn with_metadata(mut self, meta: QueryMetadata) -> Self {
        self.metadata = meta;
        self
    }
}

/// Swiss army knife for code queries.
pub struct CodeQuery {
    source: String,
    lang: Lang,
    // Lazy-loaded analysis results
    symbols: Option<Vec<Symbol>>,
    cfgs: Option<Vec<CfgInfo>>,
    dfgs: Option<Vec<DfgInfo>>,
    pdgs: Option<Vec<PdgInfo>>,
}

impl CodeQuery {
    /// Create a new query context for source code.
    pub fn new(source: &str, path: &Path) -> Option<Self> {
        let lang = Lang::from_path(path)?;
        Some(Self {
            source: source.to_string(),
            lang,
            symbols: None,
            cfgs: None,
            dfgs: None,
            pdgs: None,
        })
    }

    /// Create from source with explicit language.
    pub fn with_lang(source: &str, lang: Lang) -> Self {
        Self {
            source: source.to_string(),
            lang,
            symbols: None,
            cfgs: None,
            dfgs: None,
            pdgs: None,
        }
    }

    // === Lazy loaders ===

    fn symbols(&mut self) -> &[Symbol] {
        if self.symbols.is_none() {
            let mut parser = AstParser::new();
            self.symbols = Some(parser.extract_symbols(&self.source, self.lang));
        }
        self.symbols.as_ref().unwrap()
    }

    fn cfgs(&mut self) -> &[CfgInfo] {
        if self.cfgs.is_none() {
            self.cfgs = Some(CfgAnalyzer::analyze(&self.source, self.lang));
        }
        self.cfgs.as_ref().unwrap()
    }

    fn dfgs(&mut self) -> &[DfgInfo] {
        if self.dfgs.is_none() {
            self.dfgs = Some(DfgAnalyzer::analyze(&self.source, self.lang));
        }
        self.dfgs.as_ref().unwrap()
    }

    fn pdgs(&mut self) -> &[PdgInfo] {
        if self.pdgs.is_none() {
            self.pdgs = Some(PdgBuilder::analyze(&self.source, self.lang));
        }
        self.pdgs.as_ref().unwrap()
    }

    // === STRUCTURE QUERIES ===

    /// List all symbols (functions, types, etc.)
    pub fn symbols_list(&mut self) -> QueryResult {
        let symbols = self.symbols();
        let mut content = String::new();

        for s in symbols {
            content.push_str(&format!("{:?}: {} (L{}-{})\n",
                s.kind, s.name, s.start_line, s.end_line));
        }

        QueryResult::new("symbols_list", content)
            .with_metadata(QueryMetadata {
                symbols_found: symbols.len(),
                ..Default::default()
            })
    }

    /// Get just function signatures (most compact structure view).
    pub fn signatures(&mut self) -> QueryResult {
        let symbols = self.symbols();
        let mut content = String::new();

        for s in symbols {
            if matches!(s.kind, SymbolKind::Function | SymbolKind::Method) {
                if let Some(sig) = &s.signature {
                    content.push_str(&format!("{}\n", sig.lines().next().unwrap_or("")));
                } else {
                    content.push_str(&format!("fn {}()\n", s.name));
                }
            }
        }

        let count = symbols.iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function | SymbolKind::Method))
            .count();

        QueryResult::new("signatures", content)
            .with_metadata(QueryMetadata {
                symbols_found: count,
                ..Default::default()
            })
    }

    /// Get just type definitions.
    pub fn types(&mut self) -> QueryResult {
        let symbols = self.symbols();
        let mut content = String::new();

        for s in symbols {
            if matches!(s.kind, SymbolKind::Struct | SymbolKind::Class |
                               SymbolKind::Enum | SymbolKind::Interface | SymbolKind::Trait) {
                content.push_str(&format!("{:?} {}\n", s.kind, s.name));
            }
        }

        QueryResult::new("types", content)
    }

    /// Find a specific symbol by name.
    pub fn find_symbol(&mut self, name: &str) -> QueryResult {
        let symbols = self.symbols();
        let mut content = String::new();

        for s in symbols {
            if s.name == name {
                content.push_str(&format!("{:?}: {} at L{}-{}\n", s.kind, s.name, s.start_line, s.end_line));
                if let Some(sig) = &s.signature {
                    content.push_str(&format!("  {}\n", sig));
                }
                if let Some(doc) = &s.doc_comment {
                    content.push_str(&format!("  /// {}\n", doc));
                }
            }
        }

        if content.is_empty() {
            content = format!("Symbol '{}' not found\n", name);
        }

        QueryResult::new(&format!("find_symbol({})", name), content)
    }

    // === COMPLEXITY QUERIES ===

    /// Get complexity metrics for all functions.
    pub fn complexity(&mut self) -> QueryResult {
        let cfgs = self.cfgs();
        let mut content = String::new();

        for cfg in cfgs {
            let rating = CfgAnalyzer::complexity_rating(cfg.cyclomatic_complexity);
            let warn = if cfg.cyclomatic_complexity > 10 { " ⚠️" } else { "" };
            content.push_str(&format!("{}: {} [{}]{}\n",
                cfg.function_name, cfg.cyclomatic_complexity, rating, warn));
        }

        QueryResult::new("complexity", content)
            .with_metadata(QueryMetadata {
                functions_analyzed: cfgs.len(),
                ..Default::default()
            })
    }

    /// Get complexity for a specific function.
    pub fn complexity_of(&mut self, func: &str) -> QueryResult {
        let cfgs = self.cfgs();

        for cfg in cfgs {
            if cfg.function_name == func {
                let rating = CfgAnalyzer::complexity_rating(cfg.cyclomatic_complexity);
                return QueryResult::new(
                    &format!("complexity_of({})", func),
                    format!("{}: complexity={}, blocks={}, rating={}\n",
                        func, cfg.cyclomatic_complexity, cfg.basic_blocks, rating)
                );
            }
        }

        QueryResult::new(&format!("complexity_of({})", func),
            format!("Function '{}' not found\n", func))
    }

    /// Find functions above a complexity threshold.
    pub fn complex_functions(&mut self, threshold: usize) -> QueryResult {
        let cfgs = self.cfgs();
        let mut content = String::new();
        let mut count = 0;

        for cfg in cfgs {
            if cfg.cyclomatic_complexity > threshold {
                content.push_str(&format!("{}: {} (>{} threshold)\n",
                    cfg.function_name, cfg.cyclomatic_complexity, threshold));
                count += 1;
            }
        }

        if content.is_empty() {
            content = format!("No functions with complexity > {}\n", threshold);
        }

        QueryResult::new(&format!("complex_functions(>{})", threshold), content)
            .with_metadata(QueryMetadata {
                functions_analyzed: count,
                ..Default::default()
            })
    }

    // === DATA FLOW QUERIES ===

    /// Get all variable definitions and uses.
    pub fn data_flow(&mut self) -> QueryResult {
        let dfgs = self.dfgs();
        let mut content = String::new();

        for dfg in dfgs {
            content.push_str(&format!("{}:\n", dfg.function_name));
            content.push_str(&format!("  vars: {}\n", dfg.variables.join(", ")));
            for edge in &dfg.edges {
                content.push_str(&format!("  {} L{}→L{}\n",
                    edge.variable, edge.def_line, edge.use_line));
            }
        }

        QueryResult::new("data_flow", content)
    }

    /// Track a specific variable's flow.
    pub fn var_flow(&mut self, var_name: &str) -> QueryResult {
        let dfgs = self.dfgs();
        let mut content = String::new();

        for dfg in dfgs {
            let defs: Vec<_> = dfg.refs.iter()
                .filter(|r| r.name == var_name && matches!(r.ref_type, RefType::Definition | RefType::Update))
                .collect();
            let uses: Vec<_> = dfg.refs.iter()
                .filter(|r| r.name == var_name && matches!(r.ref_type, RefType::Use | RefType::Update))
                .collect();

            if !defs.is_empty() || !uses.is_empty() {
                content.push_str(&format!("In {}:\n", dfg.function_name));
                for d in &defs {
                    content.push_str(&format!("  defined at L{}\n", d.line));
                }
                for u in &uses {
                    content.push_str(&format!("  used at L{}\n", u.line));
                }

                // Show relevant flows
                for edge in &dfg.edges {
                    if edge.variable == var_name {
                        content.push_str(&format!("  flow: L{}→L{}\n", edge.def_line, edge.use_line));
                    }
                }
            }
        }

        if content.is_empty() {
            content = format!("Variable '{}' not found\n", var_name);
        }

        QueryResult::new(&format!("var_flow({})", var_name), content)
    }

    /// Find where a variable is defined.
    pub fn var_defs(&mut self, var_name: &str) -> QueryResult {
        let dfgs = self.dfgs();
        let mut lines = Vec::new();

        for dfg in dfgs {
            for r in &dfg.refs {
                if r.name == var_name && matches!(r.ref_type, RefType::Definition) {
                    lines.push(r.line);
                }
            }
        }

        let content = if lines.is_empty() {
            format!("'{}' not defined\n", var_name)
        } else {
            format!("{} defined at: {:?}\n", var_name, lines)
        };

        QueryResult::new(&format!("var_defs({})", var_name), content)
    }

    /// Find where a variable is used.
    pub fn var_uses(&mut self, var_name: &str) -> QueryResult {
        let dfgs = self.dfgs();
        let mut lines = Vec::new();

        for dfg in dfgs {
            for r in &dfg.refs {
                if r.name == var_name && matches!(r.ref_type, RefType::Use) {
                    lines.push(r.line);
                }
            }
        }

        let content = if lines.is_empty() {
            format!("'{}' not used\n", var_name)
        } else {
            format!("{} used at: {:?}\n", var_name, lines)
        };

        QueryResult::new(&format!("var_uses({})", var_name), content)
    }

    // === SLICING QUERIES (highest compression potential) ===

    /// Backward slice: what affects this line?
    pub fn slice_backward(&mut self, line: usize) -> QueryResult {
        let pdgs = self.pdgs();
        let mut all_lines: HashSet<usize> = HashSet::new();

        for pdg in pdgs {
            let slice = pdg.backward_slice(line, None);
            all_lines.extend(slice);
        }

        let mut lines: Vec<_> = all_lines.into_iter().collect();
        lines.sort();

        let content = self.format_slice(&lines);
        let meta = QueryMetadata {
            lines_returned: lines.len(),
            ..Default::default()
        };

        QueryResult::new(&format!("slice_backward(L{})", line), content)
            .with_metadata(meta)
    }

    /// Backward slice for a specific variable.
    pub fn slice_backward_var(&mut self, line: usize, var: &str) -> QueryResult {
        let pdgs = self.pdgs();
        let mut all_lines: HashSet<usize> = HashSet::new();

        for pdg in pdgs {
            let slice = pdg.backward_slice(line, Some(var));
            all_lines.extend(slice);
        }

        let mut lines: Vec<_> = all_lines.into_iter().collect();
        lines.sort();

        let content = self.format_slice(&lines);

        QueryResult::new(&format!("slice_backward(L{}, {})", line, var), content)
            .with_metadata(QueryMetadata {
                lines_returned: lines.len(),
                ..Default::default()
            })
    }

    /// Forward slice: what does this line affect?
    pub fn slice_forward(&mut self, line: usize) -> QueryResult {
        let pdgs = self.pdgs();
        let mut all_lines: HashSet<usize> = HashSet::new();

        for pdg in pdgs {
            let slice = pdg.forward_slice(line, None);
            all_lines.extend(slice);
        }

        let mut lines: Vec<_> = all_lines.into_iter().collect();
        lines.sort();

        let content = self.format_slice(&lines);

        QueryResult::new(&format!("slice_forward(L{})", line), content)
            .with_metadata(QueryMetadata {
                lines_returned: lines.len(),
                ..Default::default()
            })
    }

    /// Forward slice for a specific variable.
    pub fn slice_forward_var(&mut self, line: usize, var: &str) -> QueryResult {
        let pdgs = self.pdgs();
        let mut all_lines: HashSet<usize> = HashSet::new();

        for pdg in pdgs {
            let slice = pdg.forward_slice(line, Some(var));
            all_lines.extend(slice);
        }

        let mut lines: Vec<_> = all_lines.into_iter().collect();
        lines.sort();

        let content = self.format_slice(&lines);

        QueryResult::new(&format!("slice_forward(L{}, {})", line, var), content)
            .with_metadata(QueryMetadata {
                lines_returned: lines.len(),
                ..Default::default()
            })
    }

    /// Get raw source for specific lines only.
    pub fn lines(&self, line_numbers: &[usize]) -> QueryResult {
        let content = self.format_slice(line_numbers);
        QueryResult::new("lines", content)
            .with_metadata(QueryMetadata {
                lines_returned: line_numbers.len(),
                ..Default::default()
            })
    }

    /// Get source for a line range.
    pub fn line_range(&self, start: usize, end: usize) -> QueryResult {
        let lines: Vec<usize> = (start..=end).collect();
        let content = self.format_slice(&lines);
        QueryResult::new(&format!("lines({}-{})", start, end), content)
            .with_metadata(QueryMetadata {
                lines_returned: lines.len(),
                ..Default::default()
            })
    }

    // === DEPENDENCY QUERIES ===

    /// Get dependencies for a function (what it needs).
    pub fn dependencies(&mut self, func: &str) -> QueryResult {
        let pdgs = self.pdgs();
        let mut content = String::new();

        for pdg in pdgs {
            if pdg.function_name == func {
                // Variables used
                content.push_str(&format!("Variables used: {}\n", pdg.dfg.variables.join(", ")));

                // Control dependencies
                let ctrl_count = pdg.edges.iter()
                    .filter(|e| e.dep_type == super::pdg::DependenceType::Control)
                    .count();
                let data_count = pdg.edges.iter()
                    .filter(|e| e.dep_type == super::pdg::DependenceType::Data)
                    .count();

                content.push_str(&format!("Control deps: {}, Data deps: {}\n", ctrl_count, data_count));
            }
        }

        if content.is_empty() {
            content = format!("Function '{}' not found\n", func);
        }

        QueryResult::new(&format!("dependencies({})", func), content)
    }

    // === HELPER METHODS ===

    fn format_slice(&self, lines: &[usize]) -> String {
        let source_lines: Vec<&str> = self.source.lines().collect();
        let mut output = String::new();
        let mut last_line = 0;

        for &line in lines {
            if line == 0 || line > source_lines.len() {
                continue;
            }

            if last_line > 0 && line > last_line + 1 {
                output.push_str("    ...\n");
            }

            output.push_str(&format!("{:4} │ {}\n", line, source_lines[line - 1]));
            last_line = line;
        }

        output
    }

    /// Get raw source (escape hatch).
    pub fn raw(&self) -> QueryResult {
        QueryResult::new("raw", self.source.clone())
    }

    /// Get function body by name.
    pub fn function_body(&mut self, name: &str) -> QueryResult {
        let symbols = self.symbols();

        for s in symbols {
            if s.name == name && matches!(s.kind, SymbolKind::Function | SymbolKind::Method) {
                let lines: Vec<usize> = (s.start_line..=s.end_line).collect();
                return QueryResult::new(&format!("function_body({})", name), self.format_slice(&lines))
                    .with_metadata(QueryMetadata {
                        lines_returned: lines.len(),
                        ..Default::default()
                    });
            }
        }

        QueryResult::new(&format!("function_body({})", name),
            format!("Function '{}' not found\n", name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
/// Processes input data.
pub fn process(input: &str, config: &Config) -> Result<Output, Error> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(Error::Empty);
    }
    let validated = validate(trimmed)?;
    let result = transform(validated);
    Ok(result)
}

fn validate(s: &str) -> Result<&str, Error> {
    if s.len() < 3 {
        return Err(Error::TooShort);
    }
    Ok(s)
}

fn transform(s: &str) -> Output {
    Output { value: s.to_lowercase() }
}
"#;

    #[test]
    fn test_signatures() {
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);
        let result = q.signatures();

        assert!(result.content.contains("process"));
        assert!(result.content.contains("validate"));
        assert!(result.content.contains("transform"));
        assert!(result.tokens < 50); // Very compact
    }

    #[test]
    fn test_complexity() {
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);
        let result = q.complexity();

        assert!(result.content.contains("process"));
        println!("Complexity: {}", result.content);
    }

    #[test]
    fn test_slice_backward() {
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);
        let result = q.slice_backward(8); // validated = validate(...)

        println!("Backward slice from L8:\n{}", result.content);
        assert!(result.metadata.lines_returned < 10); // Should be very focused
    }

    #[test]
    fn test_var_flow() {
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);
        let result = q.var_flow("trimmed");

        println!("Flow of 'trimmed':\n{}", result.content);
        assert!(result.content.contains("defined"));
    }

    #[test]
    fn test_function_body() {
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);
        let result = q.function_body("validate");

        println!("validate body:\n{}", result.content);
        assert!(result.content.contains("len"));
    }

    #[test]
    fn test_composition() {
        // Show how queries compose: find complex functions, then get their deps
        let mut q = CodeQuery::with_lang(SAMPLE, Lang::Rust);

        // First: what functions exist?
        let sigs = q.signatures();
        println!("Signatures ({} tokens):\n{}", sigs.tokens, sigs.content);

        // Then: which are complex?
        let complex = q.complexity();
        println!("Complexity ({} tokens):\n{}", complex.tokens, complex.content);

        // Finally: slice to understand a specific line
        let slice = q.slice_backward(8);
        println!("Slice L8 ({} tokens):\n{}", slice.tokens, slice.content);

        // Total tokens used: sigs + complex + slice (all small!)
        let total = sigs.tokens + complex.tokens + slice.tokens;
        println!("Total: {} tokens (vs {} raw)", total, q.raw().tokens);
    }
}
