//! Data Flow Graph (Layer 4) - variable definitions, uses, and def-use chains.
//!
//! Tracks how data flows through a function:
//! - Variable definitions (where values are assigned)
//! - Variable uses (where values are read)
//! - Def-use chains (linking definitions to their uses)
//!
//! This enables understanding data dependencies for debugging and slicing.

use std::collections::{HashMap, HashSet};

use super::ast::{AstParser, Lang};

/// A variable reference (definition, use, or update).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarRef {
    pub name: String,
    pub ref_type: RefType,
    pub line: usize,
    pub column: usize,
}

/// Type of variable reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefType {
    /// Variable is being defined/assigned
    Definition,
    /// Variable is being read/used
    Use,
    /// Variable is being updated (both read and write, e.g., x += 1)
    Update,
}

/// A def-use edge linking a definition to a use.
#[derive(Debug, Clone)]
pub struct DataflowEdge {
    pub variable: String,
    pub def_line: usize,
    pub use_line: usize,
    pub def_type: RefType,
    pub use_type: RefType,
}

/// Data flow information for a function.
#[derive(Debug, Clone)]
pub struct DfgInfo {
    pub function_name: String,
    pub refs: Vec<VarRef>,
    pub edges: Vec<DataflowEdge>,
    pub variables: Vec<String>,
}

impl DfgInfo {
    /// Get all definitions for a variable.
    pub fn definitions(&self, var: &str) -> Vec<&VarRef> {
        self.refs.iter()
            .filter(|r| r.name == var && matches!(r.ref_type, RefType::Definition | RefType::Update))
            .collect()
    }

    /// Get all uses for a variable.
    pub fn uses(&self, var: &str) -> Vec<&VarRef> {
        self.refs.iter()
            .filter(|r| r.name == var && matches!(r.ref_type, RefType::Use | RefType::Update))
            .collect()
    }

    /// Get variables defined at a specific line.
    pub fn definitions_at_line(&self, line: usize) -> Vec<&VarRef> {
        self.refs.iter()
            .filter(|r| r.line == line && matches!(r.ref_type, RefType::Definition | RefType::Update))
            .collect()
    }

    /// Get variables used at a specific line.
    pub fn uses_at_line(&self, line: usize) -> Vec<&VarRef> {
        self.refs.iter()
            .filter(|r| r.line == line && matches!(r.ref_type, RefType::Use | RefType::Update))
            .collect()
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        let def_count = self.refs.iter()
            .filter(|r| matches!(r.ref_type, RefType::Definition))
            .count();
        let use_count = self.refs.iter()
            .filter(|r| matches!(r.ref_type, RefType::Use))
            .count();

        format!(
            "{}: {} vars, {} defs, {} uses, {} flows",
            self.function_name,
            self.variables.len(),
            def_count,
            use_count,
            self.edges.len()
        )
    }

    /// Format as LLM-ready context.
    pub fn to_context(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("### Data Flow: {}\n", self.function_name));
        output.push_str(&format!("Variables: {}\n", self.variables.join(", ")));

        // Group by variable
        for var in &self.variables {
            let defs: Vec<_> = self.definitions(var);
            let uses: Vec<_> = self.uses(var);

            if !defs.is_empty() || !uses.is_empty() {
                let def_lines: Vec<_> = defs.iter().map(|d| d.line.to_string()).collect();
                let use_lines: Vec<_> = uses.iter().map(|u| u.line.to_string()).collect();

                output.push_str(&format!(
                    "  {} → def@[{}] use@[{}]\n",
                    var,
                    def_lines.join(","),
                    use_lines.join(",")
                ));
            }
        }

        output
    }
}

/// DFG analyzer using tree-sitter.
pub struct DfgAnalyzer;

impl DfgAnalyzer {
    /// Analyze data flow for all functions in source code.
    pub fn analyze(source: &str, lang: Lang) -> Vec<DfgInfo> {
        let mut parser = AstParser::new();
        let tree = match parser.parse(source, lang) {
            Some(t) => t,
            None => return vec![],
        };

        let mut results = Vec::new();
        Self::analyze_node(tree.root_node(), source, lang, &mut results);
        results
    }

    fn analyze_node(
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        results: &mut Vec<DfgInfo>,
    ) {
        let kind = node.kind();

        // Check if this is a function
        let is_function = match lang {
            Lang::Rust => kind == "function_item",
            Lang::Python => kind == "function_definition",
            Lang::JavaScript | Lang::TypeScript => {
                kind == "function_declaration" || kind == "method_definition" || kind == "arrow_function"
            }
            Lang::Go => kind == "function_declaration" || kind == "method_declaration",
            Lang::Perl => kind == "function_definition" || kind == "anonymous_function",
        };

        if is_function {
            if let Some(dfg) = Self::analyze_function(node, source, lang) {
                results.push(dfg);
            }
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::analyze_node(child, source, lang, results);
        }
    }

    fn analyze_function(node: tree_sitter::Node, source: &str, lang: Lang) -> Option<DfgInfo> {
        // Get function name
        let name = node.child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "<anonymous>".to_string());

        let mut refs = Vec::new();
        let mut variables = HashSet::new();

        // Extract variable references from the function body
        Self::extract_refs(node, source, lang, &mut refs, &mut variables);

        // Build def-use chains
        let edges = Self::build_def_use_chains(&refs);

        let mut var_list: Vec<_> = variables.into_iter().collect();
        var_list.sort();

        Some(DfgInfo {
            function_name: name,
            refs,
            edges,
            variables: var_list,
        })
    }

    fn extract_refs(
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match lang {
            Lang::Rust => Self::extract_rust_refs(node, source, refs, variables),
            Lang::Python => Self::extract_python_refs(node, source, refs, variables),
            Lang::JavaScript | Lang::TypeScript => Self::extract_js_refs(node, source, refs, variables),
            Lang::Go => Self::extract_go_refs(node, source, refs, variables),
            Lang::Perl => Self::extract_perl_refs(node, source, refs, variables),
        }

        // Recurse
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::extract_refs(child, source, lang, refs, variables);
        }
    }

    fn extract_rust_refs(
        node: tree_sitter::Node,
        source: &str,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match kind {
            // let x = ... or let mut x = ...
            "let_declaration" => {
                if let Some(pattern) = node.child_by_field_name("pattern") {
                    if let Ok(name) = pattern.utf8_text(source.as_bytes()) {
                        let name = name.to_string();
                        variables.insert(name.clone());
                        refs.push(VarRef {
                            name,
                            ref_type: RefType::Definition,
                            line: pattern.start_position().row + 1,
                            column: pattern.start_position().column,
                        });
                    }
                }
            }
            // x = ... (assignment)
            "assignment_expression" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Update,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // Compound assignment: x += 1
            "compound_assignment_expr" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Update,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // Variable use (identifier in expression context)
            "identifier" => {
                // Check parent to determine if this is a use
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    // Skip if this is the left side of an assignment or a definition
                    if parent_kind != "let_declaration"
                        && parent_kind != "assignment_expression"
                        && parent_kind != "compound_assignment_expr"
                        && parent_kind != "function_item"
                        && parent_kind != "parameter"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            // Skip keywords and common patterns
                            if !["self", "Self", "true", "false", "None", "Some", "Ok", "Err"].contains(&name) {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // Function parameters are definitions
            "parameter" => {
                if let Some(pattern) = node.child_by_field_name("pattern") {
                    if let Ok(name) = pattern.utf8_text(source.as_bytes()) {
                        let name = name.to_string();
                        variables.insert(name.clone());
                        refs.push(VarRef {
                            name,
                            ref_type: RefType::Definition,
                            line: pattern.start_position().row + 1,
                            column: pattern.start_position().column,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_python_refs(
        node: tree_sitter::Node,
        source: &str,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match kind {
            // x = ... assignment
            "assignment" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Definition,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // x += 1 augmented assignment
            "augmented_assignment" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Update,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // Function parameter (must come before general identifier)
            "identifier" if node.parent().map(|p| p.kind()) == Some("parameters") => {
                if let Ok(name) = node.utf8_text(source.as_bytes()) {
                    let name = name.to_string();
                    variables.insert(name.clone());
                    refs.push(VarRef {
                        name,
                        ref_type: RefType::Definition,
                        line: node.start_position().row + 1,
                        column: node.start_position().column,
                    });
                }
            }
            // Variable use (general identifier)
            "identifier" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    if parent_kind != "assignment"
                        && parent_kind != "augmented_assignment"
                        && parent_kind != "function_definition"
                        && parent_kind != "parameters"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            if !["self", "True", "False", "None"].contains(&name) {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_js_refs(
        node: tree_sitter::Node,
        source: &str,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match kind {
            // let/const/var declaration
            "variable_declarator" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                        let name = name.to_string();
                        variables.insert(name.clone());
                        refs.push(VarRef {
                            name,
                            ref_type: RefType::Definition,
                            line: name_node.start_position().row + 1,
                            column: name_node.start_position().column,
                        });
                    }
                }
            }
            // x = ... assignment
            "assignment_expression" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Update,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // x += 1
            "augmented_assignment_expression" => {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        if let Ok(name) = left.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Update,
                                line: left.start_position().row + 1,
                                column: left.start_position().column,
                            });
                        }
                    }
                }
            }
            // Variable use
            "identifier" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    if parent_kind != "variable_declarator"
                        && parent_kind != "assignment_expression"
                        && parent_kind != "augmented_assignment_expression"
                        && parent_kind != "function_declaration"
                        && parent_kind != "formal_parameters"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            if !["undefined", "null", "true", "false", "this"].contains(&name) {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_go_refs(
        node: tree_sitter::Node,
        source: &str,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match kind {
            // Short variable declaration: x := ...
            "short_var_declaration" => {
                if let Some(left) = node.child_by_field_name("left") {
                    let mut cursor = left.walk();
                    for child in left.children(&mut cursor) {
                        if child.kind() == "identifier" {
                            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                                let name = name.to_string();
                                variables.insert(name.clone());
                                refs.push(VarRef {
                                    name,
                                    ref_type: RefType::Definition,
                                    line: child.start_position().row + 1,
                                    column: child.start_position().column,
                                });
                            }
                        }
                    }
                }
            }
            // var declaration
            "var_spec" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                        let name = name.to_string();
                        variables.insert(name.clone());
                        refs.push(VarRef {
                            name,
                            ref_type: RefType::Definition,
                            line: name_node.start_position().row + 1,
                            column: name_node.start_position().column,
                        });
                    }
                }
            }
            // Assignment: x = ...
            "assignment_statement" => {
                if let Some(left) = node.child_by_field_name("left") {
                    let mut cursor = left.walk();
                    for child in left.children(&mut cursor) {
                        if child.kind() == "identifier" {
                            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                                let name = name.to_string();
                                variables.insert(name.clone());
                                refs.push(VarRef {
                                    name,
                                    ref_type: RefType::Update,
                                    line: child.start_position().row + 1,
                                    column: child.start_position().column,
                                });
                            }
                        }
                    }
                }
            }
            // Variable use
            "identifier" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    if parent_kind != "short_var_declaration"
                        && parent_kind != "var_spec"
                        && parent_kind != "assignment_statement"
                        && parent_kind != "function_declaration"
                        && parent_kind != "parameter_declaration"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            if !["nil", "true", "false"].contains(&name) {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_perl_refs(
        node: tree_sitter::Node,
        source: &str,
        refs: &mut Vec<VarRef>,
        variables: &mut HashSet<String>,
    ) {
        let kind = node.kind();

        match kind {
            // my $x = ..., my @arr = ..., my %hash = ...
            "variable_declaration" => {
                // Walk children to find variable nodes
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "scalar_variable" | "array_variable" | "hash_variable" => {
                            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                                let name = name.to_string();
                                variables.insert(name.clone());
                                refs.push(VarRef {
                                    name,
                                    ref_type: RefType::Definition,
                                    line: child.start_position().row + 1,
                                    column: child.start_position().column,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            // $x = ..., @arr = ..., %hash = ... (assignment)
            "assignment_expression" => {
                if let Some(left) = node.child_by_field_name("left") {
                    match left.kind() {
                        "scalar_variable" | "array_variable" | "hash_variable" => {
                            if let Ok(name) = left.utf8_text(source.as_bytes()) {
                                let name = name.to_string();
                                // Check if this is a new definition or an update
                                let ref_type = if variables.contains(&name) {
                                    RefType::Update
                                } else {
                                    variables.insert(name.clone());
                                    RefType::Definition
                                };
                                refs.push(VarRef {
                                    name,
                                    ref_type,
                                    line: left.start_position().row + 1,
                                    column: left.start_position().column,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Compound assignment: $x += 1, $x .= "str"
            "update_expression" | "concatenation_assignment" => {
                // First child is typically the variable
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    match child.kind() {
                        "scalar_variable" | "array_variable" | "hash_variable" => {
                            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                                let name = name.to_string();
                                variables.insert(name.clone());
                                refs.push(VarRef {
                                    name,
                                    ref_type: RefType::Update,
                                    line: child.start_position().row + 1,
                                    column: child.start_position().column,
                                });
                            }
                            break; // Only first variable
                        }
                        _ => {}
                    }
                }
            }
            // Scalar variable use: $x
            "scalar_variable" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    // Skip if this is on the left side of an assignment or declaration
                    if parent_kind != "variable_declaration"
                        && parent_kind != "assignment_expression"
                        && parent_kind != "update_expression"
                        && parent_kind != "concatenation_assignment"
                        && parent_kind != "subroutine_declaration"
                        && parent_kind != "signature"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            // Skip special variables like $_, $1, $@, etc.
                            if !name.starts_with("$_") && !name.starts_with("$@")
                                && !name.starts_with("$!") && !name.starts_with("$$")
                                && !name.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false)
                            {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // Array variable use: @arr
            "array_variable" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    if parent_kind != "variable_declaration"
                        && parent_kind != "assignment_expression"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            // Skip special arrays like @_, @ARGV
                            if name != "@_" && name != "@ARGV" && name != "@ISA" {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // Hash variable use: %hash
            "hash_variable" => {
                if let Some(parent) = node.parent() {
                    let parent_kind = parent.kind();
                    if parent_kind != "variable_declaration"
                        && parent_kind != "assignment_expression"
                    {
                        if let Ok(name) = node.utf8_text(source.as_bytes()) {
                            // Skip special hashes like %ENV
                            if name != "%ENV" && name != "%SIG" {
                                let name = name.to_string();
                                if variables.contains(&name) {
                                    refs.push(VarRef {
                                        name,
                                        ref_type: RefType::Use,
                                        line: node.start_position().row + 1,
                                        column: node.start_position().column,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            // Subroutine parameters (from signature or @_)
            "signature" => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() == "scalar_variable" {
                        if let Ok(name) = child.utf8_text(source.as_bytes()) {
                            let name = name.to_string();
                            variables.insert(name.clone());
                            refs.push(VarRef {
                                name,
                                ref_type: RefType::Definition,
                                line: child.start_position().row + 1,
                                column: child.start_position().column,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Build def-use chains from variable references.
    fn build_def_use_chains(refs: &[VarRef]) -> Vec<DataflowEdge> {
        let mut edges = Vec::new();

        // Group refs by variable
        let mut by_var: HashMap<&str, Vec<&VarRef>> = HashMap::new();
        for r in refs {
            by_var.entry(&r.name).or_default().push(r);
        }

        // For each variable, link definitions to subsequent uses
        for (var, var_refs) in by_var {
            let mut sorted = var_refs.clone();
            sorted.sort_by_key(|r| (r.line, r.column));

            let mut last_def: Option<&VarRef> = None;

            for r in sorted {
                match r.ref_type {
                    RefType::Definition | RefType::Update => {
                        last_def = Some(r);
                    }
                    RefType::Use => {
                        if let Some(def) = last_def {
                            edges.push(DataflowEdge {
                                variable: var.to_string(),
                                def_line: def.line,
                                use_line: r.line,
                                def_type: def.ref_type,
                                use_type: r.ref_type,
                            });
                        }
                    }
                }
            }
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_dfg() {
        let source = r#"
fn process(input: i32) -> i32 {
    let x = input + 1;
    let y = x * 2;
    let z = x + y;
    z
}
"#;

        let dfgs = DfgAnalyzer::analyze(source, Lang::Rust);
        assert_eq!(dfgs.len(), 1);

        let dfg = &dfgs[0];
        assert_eq!(dfg.function_name, "process");

        // Should track input, x, y, z
        assert!(dfg.variables.contains(&"x".to_string()));
        assert!(dfg.variables.contains(&"y".to_string()));
        assert!(dfg.variables.contains(&"z".to_string()));

        // Should have def-use edges
        assert!(!dfg.edges.is_empty());

        println!("DFG summary: {}", dfg.summary());
        println!("DFG context:\n{}", dfg.to_context());
    }

    #[test]
    fn test_python_dfg() {
        let source = r#"
def calculate(n):
    result = 0
    for i in range(n):
        result += i
    return result
"#;

        let dfgs = DfgAnalyzer::analyze(source, Lang::Python);
        assert_eq!(dfgs.len(), 1);

        let dfg = &dfgs[0];
        assert_eq!(dfg.function_name, "calculate");
        assert!(dfg.variables.contains(&"result".to_string()));
    }
}
