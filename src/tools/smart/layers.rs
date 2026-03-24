//! Layer abstractions for the five-layer code analysis.
//!
//! 1. AST - syntax structure
//! 2. Call Graph - function relationships
//! 3. CFG - control flow (cyclomatic complexity)
//! 4. DFG - data flow (variable defs/uses)
//! 5. PDG - program dependence (slicing)
//! Extensions:
//! - Theory Graph - Lean declaration/theorem dependency view

use std::path::Path;

use tree_sitter::Node;

use super::ast::{AstParser, Lang, Symbol, SymbolKind};
use super::call_graph::CallGraph;
use super::cfg::CfgAnalyzer;
use super::dfg::DfgAnalyzer;
use super::lean_graph::{LeanDeclGraph, LeanGraphAnalyzer};
use super::pdg::PdgBuilder;

/// Which layer of analysis to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLayer {
    /// Raw source code (no analysis).
    Raw,
    /// AST layer - structure only.
    Ast,
    /// Call graph layer - function relationships.
    CallGraph,
    /// Control flow graph (not yet implemented).
    Cfg,
    /// Data flow graph (not yet implemented).
    Dfg,
    /// Program dependence graph (not yet implemented).
    Pdg,
    /// Lean declaration / theorem dependency graph.
    TheoryGraph,
}

impl CodeLayer {
    /// Token savings estimate for this layer vs raw code.
    pub fn token_savings(&self) -> &'static str {
        match self {
            CodeLayer::Raw => "0%",
            CodeLayer::Ast => "60-80%",
            CodeLayer::CallGraph => "70-85%",
            CodeLayer::Cfg => "75-90%",
            CodeLayer::Dfg => "80-90%",
            CodeLayer::Pdg => "85-95%",
            CodeLayer::TheoryGraph => "85-95%",
        }
    }
}

/// A view of code at a specific layer.
#[derive(Debug, Clone)]
pub struct LayerView {
    pub path: String,
    pub layer: CodeLayer,
    pub content: String,
    pub symbols: Vec<Symbol>,
    pub call_graph: Option<CallGraph>,
    pub lean_graph: Option<LeanDeclGraph>,
}

impl LayerView {
    /// Create a raw view (just the source).
    pub fn raw(path: &str, content: &str) -> Self {
        Self {
            path: path.to_string(),
            layer: CodeLayer::Raw,
            content: content.to_string(),
            symbols: vec![],
            call_graph: None,
            lean_graph: None,
        }
    }

    /// Create an AST view.
    pub fn ast(path: &str, content: &str, parser: &mut AstParser) -> Self {
        let lang = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension);

        let (symbols, summary) = if let Some(lang) = lang {
            let symbols = parser.extract_symbols(content, lang);
            let summary = parser.summarize(content, lang);
            (symbols, summary)
        } else {
            (vec![], content.to_string())
        };

        Self {
            path: path.to_string(),
            layer: CodeLayer::Ast,
            content: summary,
            symbols,
            call_graph: None,
            lean_graph: None,
        }
    }

    /// Create a call graph view.
    pub fn call_graph(path: &str, content: &str, parser: &mut AstParser) -> Self {
        let mut view = Self::ast(path, content, parser);
        view.layer = CodeLayer::CallGraph;

        let mut cg = CallGraph::new();
        for symbol in &view.symbols {
            if matches!(symbol.kind, SymbolKind::Function | SymbolKind::Method) {
                cg.calls.entry(symbol.name.clone()).or_default();
            }
        }

        if let Some(lang) = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension)
        {
            if let Some(tree) = parser.parse(content, lang) {
                collect_call_sites(tree.root_node(), content, lang, None, &mut cg);
            }
        }

        view.content = format!("{}\n{}", view.content, cg.to_summary());
        view.call_graph = Some(cg);
        view
    }

    /// Create a Lean declaration / theorem dependency view.
    pub fn theory_graph(path: &str, content: &str, parser: &mut AstParser) -> Self {
        let lang = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension);

        match lang {
            Some(Lang::Lean) => {
                if let Some(graph) = LeanGraphAnalyzer::analyze(path, content, parser) {
                    Self {
                        path: path.to_string(),
                        layer: CodeLayer::TheoryGraph,
                        content: graph.to_summary(),
                        symbols: vec![],
                        call_graph: None,
                        lean_graph: Some(graph),
                    }
                } else {
                    Self {
                        path: path.to_string(),
                        layer: CodeLayer::TheoryGraph,
                        content: "Failed to parse Lean file".to_string(),
                        symbols: vec![],
                        call_graph: None,
                        lean_graph: None,
                    }
                }
            }
            _ => Self {
                path: path.to_string(),
                layer: CodeLayer::TheoryGraph,
                content: "Theory graph is currently supported for Lean files only.".to_string(),
                symbols: vec![],
                call_graph: None,
                lean_graph: None,
            },
        }
    }

    /// Get a compact representation suitable for LLM context.
    /// The model knows what layer it requested, so we just return the content.
    pub fn to_context(&self) -> String {
        match self.layer {
            CodeLayer::Raw => {
                format!("## {}\n\n```\n{}\n```", self.path, self.content)
            }
            _ => {
                format!("## {}\n\n{}", self.path, self.content)
            }
        }
    }
}

/// Builder for analyzing code at multiple layers.
pub struct LayerAnalyzer {
    parser: AstParser,
}

impl LayerAnalyzer {
    pub fn new() -> Self {
        Self {
            parser: AstParser::new(),
        }
    }

    /// Analyze a file at the specified layer.
    pub fn analyze(&mut self, path: &str, content: &str, layer: CodeLayer) -> LayerView {
        match layer {
            CodeLayer::Raw => LayerView::raw(path, content),
            CodeLayer::Ast => LayerView::ast(path, content, &mut self.parser),
            CodeLayer::CallGraph => LayerView::call_graph(path, content, &mut self.parser),
            CodeLayer::Cfg => self.analyze_cfg(path, content),
            CodeLayer::Dfg => self.analyze_dfg(path, content),
            CodeLayer::Pdg => self.analyze_pdg(path, content),
            CodeLayer::TheoryGraph => LayerView::theory_graph(path, content, &mut self.parser),
        }
    }

    /// Analyze CFG layer - control flow with complexity metrics.
    fn analyze_cfg(&mut self, path: &str, content: &str) -> LayerView {
        let mut view = LayerView::ast(path, content, &mut self.parser);
        view.layer = CodeLayer::Cfg;

        let lang = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension);

        if let Some(lang) = lang {
            let cfgs = CfgAnalyzer::analyze(content, lang);
            let mut cfg_content = String::new();

            cfg_content.push_str("### Control Flow Analysis\n");
            for cfg in &cfgs {
                let rating = CfgAnalyzer::complexity_rating(cfg.cyclomatic_complexity);
                let warn = if cfg.cyclomatic_complexity > 10 {
                    " ⚠️"
                } else {
                    ""
                };
                cfg_content.push_str(&format!(
                    "- `{}`: complexity {} [{}]{}, {} blocks\n",
                    cfg.function_name, cfg.cyclomatic_complexity, rating, warn, cfg.basic_blocks
                ));
            }

            view.content = format!("{}\n{}", view.content, cfg_content);
        }

        view
    }

    /// Analyze DFG layer - data flow with def-use chains.
    fn analyze_dfg(&mut self, path: &str, content: &str) -> LayerView {
        let mut view = LayerView::ast(path, content, &mut self.parser);
        view.layer = CodeLayer::Dfg;

        let lang = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension);

        if let Some(lang) = lang {
            let dfgs = DfgAnalyzer::analyze(content, lang);
            let mut dfg_content = String::new();

            dfg_content.push_str("### Data Flow Analysis\n");
            for dfg in &dfgs {
                dfg_content.push_str(&format!("**{}**\n", dfg.function_name));
                dfg_content.push_str(&format!("  Variables: {}\n", dfg.variables.join(", ")));
                dfg_content.push_str(&format!("  Def-use chains: {}\n", dfg.edges.len()));

                // Show top flows
                for edge in dfg.edges.iter().take(5) {
                    dfg_content.push_str(&format!(
                        "    {} L{}→L{}\n",
                        edge.variable, edge.def_line, edge.use_line
                    ));
                }
                if dfg.edges.len() > 5 {
                    dfg_content.push_str(&format!("    ... +{} more\n", dfg.edges.len() - 5));
                }
            }

            view.content = format!("{}\n{}", view.content, dfg_content);
        }

        view
    }

    /// Analyze PDG layer - program dependence with slicing capability.
    fn analyze_pdg(&mut self, path: &str, content: &str) -> LayerView {
        let mut view = LayerView::ast(path, content, &mut self.parser);
        view.layer = CodeLayer::Pdg;

        let lang = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Lang::from_extension);

        if let Some(lang) = lang {
            let pdgs = PdgBuilder::analyze(content, lang);
            let mut pdg_content = String::new();

            pdg_content.push_str("### Program Dependence Analysis\n");
            for pdg in &pdgs {
                pdg_content.push_str(&format!("**{}**\n", pdg.function_name));
                pdg_content.push_str(&format!("  {}\n", pdg.summary()));

                // Show control vs data deps
                let ctrl = pdg
                    .edges
                    .iter()
                    .filter(|e| e.dep_type == super::pdg::DependenceType::Control)
                    .count();
                let data = pdg
                    .edges
                    .iter()
                    .filter(|e| e.dep_type == super::pdg::DependenceType::Data)
                    .count();
                pdg_content.push_str(&format!("  Control deps: {}, Data deps: {}\n", ctrl, data));
            }

            view.content = format!("{}\n{}", view.content, pdg_content);
        }

        view
    }

    /// Analyze and return the most efficient layer for context.
    pub fn analyze_efficient(&mut self, path: &str, content: &str) -> LayerView {
        // Start with AST - good balance of savings and usefulness
        self.analyze(path, content, CodeLayer::Ast)
    }
}

fn collect_call_sites(
    node: Node,
    source: &str,
    lang: Lang,
    current_fn: Option<String>,
    graph: &mut CallGraph,
) {
    let next_fn = function_name_for_node(node, source, lang).or(current_fn);

    if let Some(ref caller) = next_fn {
        if is_call_node(node.kind()) {
            if let Some(callee) = call_target_name(node, source) {
                graph.add_call(caller, &callee, node.start_position().row + 1);
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_call_sites(child, source, lang, next_fn.clone(), graph);
    }
}

fn function_name_for_node(node: Node, source: &str, lang: Lang) -> Option<String> {
    let is_fn = match lang {
        Lang::Rust => matches!(node.kind(), "function_item"),
        Lang::Python => matches!(node.kind(), "function_definition"),
        Lang::JavaScript | Lang::TypeScript => {
            matches!(node.kind(), "function_declaration" | "method_definition")
        }
        Lang::Go => matches!(node.kind(), "function_declaration" | "method_declaration"),
        Lang::Perl => matches!(node.kind(), "function_definition"),
        Lang::Nim => matches!(
            node.kind(),
            "proc_declaration"
                | "func_declaration"
                | "method_declaration"
                | "converter_declaration"
                | "iterator_declaration"
                | "template_declaration"
                | "macro_declaration"
        ),
        Lang::Lean => matches!(node.kind(), "def" | "abbrev" | "theorem" | "instance"),
    };

    if !is_fn {
        return None;
    }

    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(text) = name_node.utf8_text(source.as_bytes()) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(
            child.kind(),
            "identifier" | "type_identifier" | "field_identifier" | "property_identifier"
        ) {
            if let Ok(text) = child.utf8_text(source.as_bytes()) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }

    None
}

fn is_call_node(kind: &str) -> bool {
    kind.contains("call")
}

fn call_target_name(node: Node, source: &str) -> Option<String> {
    if let Some(function_node) = node.child_by_field_name("function") {
        if let Ok(text) = function_node.utf8_text(source.as_bytes()) {
            let normalized = normalize_callee_name(text);
            if !normalized.is_empty() {
                return Some(normalized);
            }
        }
    }

    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(text) = name_node.utf8_text(source.as_bytes()) {
            let normalized = normalize_callee_name(text);
            if !normalized.is_empty() {
                return Some(normalized);
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }

        if matches!(
            child.kind(),
            "identifier"
                | "field_identifier"
                | "property_identifier"
                | "scoped_identifier"
                | "type_identifier"
                | "qualified_identifier"
                | "member_expression"
                | "field_expression"
        ) {
            if let Ok(text) = child.utf8_text(source.as_bytes()) {
                let normalized = normalize_callee_name(text);
                if !normalized.is_empty() {
                    return Some(normalized);
                }
            }
        }
    }

    None
}

fn normalize_callee_name(raw: &str) -> String {
    raw.trim()
        .trim_start_matches('&')
        .trim_end_matches("()")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_graph_extracts_relationships_for_rust() {
        let source = r#"
fn helper() {}

fn entry() {
    helper();
}
"#;

        let mut parser = AstParser::new();
        let view = LayerView::call_graph("sample.rs", source, &mut parser);
        let graph = view.call_graph.expect("call graph should exist");

        assert!(graph.calls.contains_key("entry"));
        assert!(graph.get_calls("entry").contains(&"helper"));
    }
}

impl Default for LayerAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_view() {
        let source = r#"
pub fn hello() -> String {
    "Hello".to_string()
}

pub fn world() -> String {
    "World".to_string()
}
"#;

        let mut analyzer = LayerAnalyzer::new();

        let raw = analyzer.analyze("test.rs", source, CodeLayer::Raw);
        assert!(raw.content.contains("pub fn hello"));

        let ast = analyzer.analyze("test.rs", source, CodeLayer::Ast);
        assert!(ast.content.contains("hello"));
        assert!(ast.content.contains("Functions"));
    }
}
