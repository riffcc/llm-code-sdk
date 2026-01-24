//! AST layer - tree-sitter based parsing.
//!
//! Extracts structural information from source code:
//! - Function/method signatures
//! - Class/struct definitions
//! - Import statements
//! - Symbol table

use std::collections::HashMap;
use std::path::Path;

use tree_sitter::{Language, Parser, Tree, Node};

/// Language support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Perl,
}

impl Lang {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Lang::Rust),
            "py" => Some(Lang::Python),
            "js" | "jsx" | "mjs" => Some(Lang::JavaScript),
            "ts" | "tsx" => Some(Lang::TypeScript),
            "go" => Some(Lang::Go),
            "pl" | "pm" | "cgi" | "t" => Some(Lang::Perl),
            _ => None,
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(Self::from_extension)
    }

    fn language(&self) -> Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Perl => tree_sitter_perl::LANGUAGE.into(),
        }
    }
}

/// Kind of symbol in the code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Module,
    Variable,
    Constant,
    Import,
}

/// A symbol extracted from the AST.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: Option<String>,
    pub doc_comment: Option<String>,
}

/// A function signature.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<(String, Option<String>)>, // (name, type)
    pub return_type: Option<String>,
    pub is_async: bool,
    pub is_public: bool,
    pub doc_comment: Option<String>,
}

impl FunctionSignature {
    /// Format as a compact signature string.
    pub fn to_signature_string(&self) -> String {
        let params: Vec<String> = self.params.iter().map(|(name, ty)| {
            match ty {
                Some(t) => format!("{}: {}", name, t),
                None => name.clone(),
            }
        }).collect();

        let ret = match &self.return_type {
            Some(t) => format!(" -> {}", t),
            None => String::new(),
        };

        let prefix = if self.is_async { "async " } else { "" };
        let vis = if self.is_public { "pub " } else { "" };

        format!("{}{}fn {}({}){}", vis, prefix, self.name, params.join(", "), ret)
    }
}

/// An AST node representation.
#[derive(Debug, Clone)]
pub struct AstNode {
    pub kind: String,
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub children: Vec<AstNode>,
}

/// AST parser using tree-sitter.
pub struct AstParser {
    parsers: HashMap<Lang, Parser>,
}

impl AstParser {
    pub fn new() -> Self {
        let mut parsers = HashMap::new();

        for lang in [Lang::Rust, Lang::Python, Lang::JavaScript, Lang::TypeScript, Lang::Go, Lang::Perl] {
            let mut parser = Parser::new();
            parser.set_language(&lang.language()).ok();
            parsers.insert(lang, parser);
        }

        Self { parsers }
    }

    /// Parse source code and return the tree.
    pub fn parse(&mut self, source: &str, lang: Lang) -> Option<Tree> {
        let parser = self.parsers.get_mut(&lang)?;
        parser.parse(source, None)
    }

    /// Extract all symbols from source code.
    pub fn extract_symbols(&mut self, source: &str, lang: Lang) -> Vec<Symbol> {
        let tree = match self.parse(source, lang) {
            Some(t) => t,
            None => return vec![],
        };

        let mut symbols = Vec::new();
        self.extract_symbols_from_node(tree.root_node(), source, lang, &mut symbols);
        symbols
    }

    fn extract_symbols_from_node(
        &self,
        node: Node,
        source: &str,
        lang: Lang,
        symbols: &mut Vec<Symbol>,
    ) {
        // Extract based on node type and language
        match lang {
            Lang::Rust => self.extract_rust_symbol(node, source, symbols),
            Lang::Python => self.extract_python_symbol(node, source, symbols),
            Lang::JavaScript | Lang::TypeScript => self.extract_js_symbol(node, source, symbols),
            Lang::Go => self.extract_go_symbol(node, source, symbols),
            Lang::Perl => self.extract_perl_symbol(node, source, symbols),
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.extract_symbols_from_node(child, source, lang, symbols);
        }
    }

    fn extract_rust_symbol(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>) {
        let kind = node.kind();

        let symbol_kind = match kind {
            "function_item" => Some(SymbolKind::Function),
            "impl_item" => None, // We'll extract methods from inside
            "struct_item" => Some(SymbolKind::Struct),
            "enum_item" => Some(SymbolKind::Enum),
            "trait_item" => Some(SymbolKind::Trait),
            "mod_item" => Some(SymbolKind::Module),
            "const_item" => Some(SymbolKind::Constant),
            "use_declaration" => Some(SymbolKind::Import),
            _ => None,
        };

        if let Some(sk) = symbol_kind {
            // For Rust, the name is in the "name" field
            // - function_item: name is an identifier
            // - struct_item: name is a type_identifier
            // - enum_item: name is a type_identifier
            // - trait_item: name is a type_identifier
            let name = self.find_child_text(node, "name", source)
                .or_else(|| {
                    // Walk children to find identifier or type_identifier
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        match child.kind() {
                            "identifier" | "type_identifier" => {
                                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
                            }
                            _ => {}
                        }
                    }
                    None
                })
                .unwrap_or_else(|| "<anonymous>".to_string());

            let signature = if matches!(sk, SymbolKind::Function) {
                Some(node.utf8_text(source.as_bytes()).unwrap_or("").lines().next().unwrap_or("").to_string())
            } else {
                None
            };

            // Extract doc comments from preceding siblings
            let doc_comment = self.extract_rust_doc_comment(node, source);

            symbols.push(Symbol {
                name,
                kind: sk,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                signature,
                doc_comment,
            });
        }
    }

    /// Extract doc comments (/// or //!) from nodes preceding the given node.
    fn extract_rust_doc_comment(&self, node: Node, source: &str) -> Option<String> {
        let mut comments = Vec::new();
        let mut prev = node.prev_sibling();

        // Walk backwards through siblings looking for doc comments
        while let Some(sibling) = prev {
            match sibling.kind() {
                "line_comment" => {
                    if let Ok(text) = sibling.utf8_text(source.as_bytes()) {
                        // Check if it's a doc comment (/// or //!)
                        if text.starts_with("///") || text.starts_with("//!") {
                            // Strip the comment prefix and trim
                            let content = text.trim_start_matches("///")
                                .trim_start_matches("//!")
                                .trim();
                            comments.push(content.to_string());
                        } else {
                            // Regular comment, stop looking
                            break;
                        }
                    }
                }
                "attribute_item" => {
                    // Handle #[doc = "..."] attributes
                    if let Ok(text) = sibling.utf8_text(source.as_bytes()) {
                        if text.contains("doc") {
                            // Extract the doc string (simplified)
                            if let Some(start) = text.find('"') {
                                if let Some(end) = text.rfind('"') {
                                    if start < end {
                                        comments.push(text[start + 1..end].to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                _ => break, // Stop at non-comment/non-attribute
            }
            prev = sibling.prev_sibling();
        }

        if comments.is_empty() {
            None
        } else {
            // Comments are in reverse order, so reverse them
            comments.reverse();
            Some(comments.join(" "))
        }
    }

    fn extract_python_symbol(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>) {
        let kind = node.kind();

        let symbol_kind = match kind {
            "function_definition" => Some(SymbolKind::Function),
            "class_definition" => Some(SymbolKind::Class),
            "import_statement" | "import_from_statement" => Some(SymbolKind::Import),
            _ => None,
        };

        if let Some(sk) = symbol_kind {
            let name = self.find_child_text(node, "name", source)
                .or_else(|| self.find_child_text(node, "identifier", source))
                .unwrap_or_else(|| "<anonymous>".to_string());

            let signature = if matches!(sk, SymbolKind::Function) {
                Some(node.utf8_text(source.as_bytes()).unwrap_or("").lines().next().unwrap_or("").to_string())
            } else {
                None
            };

            symbols.push(Symbol {
                name,
                kind: sk,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                signature,
                doc_comment: None,
            });
        }
    }

    fn extract_js_symbol(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>) {
        let kind = node.kind();

        let symbol_kind = match kind {
            "function_declaration" | "arrow_function" | "function" => Some(SymbolKind::Function),
            "method_definition" => Some(SymbolKind::Method),
            "class_declaration" => Some(SymbolKind::Class),
            "interface_declaration" => Some(SymbolKind::Interface),
            "import_statement" => Some(SymbolKind::Import),
            "variable_declaration" => Some(SymbolKind::Variable),
            _ => None,
        };

        if let Some(sk) = symbol_kind {
            let name = self.find_child_text(node, "name", source)
                .or_else(|| self.find_child_text(node, "identifier", source))
                .unwrap_or_else(|| "<anonymous>".to_string());

            symbols.push(Symbol {
                name,
                kind: sk,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                signature: None,
                doc_comment: None,
            });
        }
    }

    fn extract_go_symbol(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>) {
        let kind = node.kind();

        let symbol_kind = match kind {
            "function_declaration" | "method_declaration" => Some(SymbolKind::Function),
            "type_declaration" => Some(SymbolKind::Struct),
            "import_declaration" => Some(SymbolKind::Import),
            _ => None,
        };

        if let Some(sk) = symbol_kind {
            let name = self.find_child_text(node, "name", source)
                .or_else(|| self.find_child_text(node, "identifier", source))
                .unwrap_or_else(|| "<anonymous>".to_string());

            symbols.push(Symbol {
                name,
                kind: sk,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                signature: None,
                doc_comment: None,
            });
        }
    }

    fn extract_perl_symbol(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>) {
        let kind = node.kind();

        // Perl node types from tree-sitter-perl grammar
        let symbol_kind = match kind {
            "subroutine_declaration" | "anonymous_subroutine_expression" => Some(SymbolKind::Function),
            "method_declaration" => Some(SymbolKind::Method),
            "package_statement" => Some(SymbolKind::Module),
            "use_statement" | "require_statement" => Some(SymbolKind::Import),
            _ => None,
        };

        if let Some(sk) = symbol_kind {
            // For subroutines, look for the name child
            let name = self.find_child_text(node, "name", source)
                .or_else(|| {
                    // Walk children to find bareword/identifier
                    let mut cursor = node.walk();
                    for child in node.children(&mut cursor) {
                        match child.kind() {
                            "bareword" | "identifier" | "package_name" => {
                                return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
                            }
                            _ => {}
                        }
                    }
                    None
                })
                .unwrap_or_else(|| "<anonymous>".to_string());

            let signature = if matches!(sk, SymbolKind::Function | SymbolKind::Method) {
                Some(node.utf8_text(source.as_bytes()).unwrap_or("").lines().next().unwrap_or("").to_string())
            } else {
                None
            };

            // Extract POD documentation (simplified - just look for preceding comments)
            let doc_comment = self.extract_perl_doc_comment(node, source);

            symbols.push(Symbol {
                name,
                kind: sk,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                signature,
                doc_comment,
            });
        }
    }

    /// Extract Perl documentation (# comments or POD).
    fn extract_perl_doc_comment(&self, node: Node, source: &str) -> Option<String> {
        let mut comments = Vec::new();
        let mut prev = node.prev_sibling();

        while let Some(sibling) = prev {
            match sibling.kind() {
                "comment" => {
                    if let Ok(text) = sibling.utf8_text(source.as_bytes()) {
                        let content = text.trim_start_matches('#').trim();
                        comments.push(content.to_string());
                    }
                }
                _ => break,
            }
            prev = sibling.prev_sibling();
        }

        if comments.is_empty() {
            None
        } else {
            comments.reverse();
            Some(comments.join(" "))
        }
    }

    fn find_child_text(&self, node: Node, field_name: &str, source: &str) -> Option<String> {
        node.child_by_field_name(field_name)
            .map(|n| n.utf8_text(source.as_bytes()).unwrap_or("").to_string())
    }

    /// Extract function signatures from source code.
    pub fn extract_functions(&mut self, source: &str, lang: Lang) -> Vec<FunctionSignature> {
        let symbols = self.extract_symbols(source, lang);
        symbols
            .into_iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function | SymbolKind::Method))
            .map(|s| FunctionSignature {
                name: s.name,
                params: vec![], // TODO: parse parameters
                return_type: None, // TODO: parse return type
                is_async: s.signature.as_ref().map(|sig| sig.contains("async")).unwrap_or(false),
                is_public: s.signature.as_ref().map(|sig| sig.contains("pub ")).unwrap_or(false),
                doc_comment: s.doc_comment,
            })
            .collect()
    }

    /// Generate a compact summary of a source file.
    pub fn summarize(&mut self, source: &str, lang: Lang) -> String {
        let symbols = self.extract_symbols(source, lang);

        let mut output = String::new();

        // Group by kind
        let functions: Vec<_> = symbols.iter().filter(|s| matches!(s.kind, SymbolKind::Function | SymbolKind::Method)).collect();
        let types: Vec<_> = symbols.iter().filter(|s| matches!(s.kind, SymbolKind::Struct | SymbolKind::Class | SymbolKind::Enum | SymbolKind::Interface | SymbolKind::Trait)).collect();
        let imports: Vec<_> = symbols.iter().filter(|s| matches!(s.kind, SymbolKind::Import)).collect();

        if !types.is_empty() {
            output.push_str("### Types\n");
            for t in types {
                output.push_str(&format!("- `{}` ({:?}, lines {}-{})\n", t.name, t.kind, t.start_line, t.end_line));
            }
            output.push('\n');
        }

        if !functions.is_empty() {
            output.push_str("### Functions\n");
            for f in functions {
                if let Some(sig) = &f.signature {
                    output.push_str(&format!("- `{}`\n", sig.trim()));
                } else {
                    output.push_str(&format!("- `{}` (lines {}-{})\n", f.name, f.start_line, f.end_line));
                }
            }
            output.push('\n');
        }

        if !imports.is_empty() {
            output.push_str(&format!("### Imports ({} total)\n", imports.len()));
        }

        output
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_parsing() {
        let source = r#"
pub fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

struct User {
    name: String,
    age: u32,
}

impl User {
    fn new(name: String) -> Self {
        Self { name, age: 0 }
    }
}
"#;

        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(source, Lang::Rust);

        assert!(symbols.iter().any(|s| s.name == "hello" && matches!(s.kind, SymbolKind::Function)));
        assert!(symbols.iter().any(|s| s.name == "User" && matches!(s.kind, SymbolKind::Struct)));
    }

    #[test]
    fn test_python_parsing() {
        let source = r#"
def greet(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
"#;

        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(source, Lang::Python);

        assert!(symbols.iter().any(|s| s.name == "greet" && matches!(s.kind, SymbolKind::Function)));
        assert!(symbols.iter().any(|s| s.name == "Person" && matches!(s.kind, SymbolKind::Class)));
    }

    #[test]
    fn test_summarize() {
        let source = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

struct Calculator;
"#;

        let mut parser = AstParser::new();
        let summary = parser.summarize(source, Lang::Rust);

        assert!(summary.contains("Calculator"));
        assert!(summary.contains("add"));
        assert!(summary.contains("subtract"));
    }
}
