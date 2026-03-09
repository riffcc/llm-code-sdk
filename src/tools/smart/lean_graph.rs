//! Lean declaration and theorem graph analysis.
//!
//! This is intentionally a surface-structure graph built from tree-sitter:
//! it can see declarations, imports, namespace/section scope, and identifier
//! references, but it does not claim elaborated proof truth.

use std::collections::{BTreeMap, BTreeSet};

use tree_sitter::Node;

use super::ast::{AstParser, Lang};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LeanDeclKind {
    Def,
    Abbrev,
    Theorem,
    Instance,
    Axiom,
    Constant,
}

impl LeanDeclKind {
    fn from_node_kind(kind: &str) -> Option<Self> {
        match kind {
            "def" => Some(Self::Def),
            "abbrev" => Some(Self::Abbrev),
            "theorem" => Some(Self::Theorem),
            "instance" => Some(Self::Instance),
            "axiom" => Some(Self::Axiom),
            "constant" => Some(Self::Constant),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Def => "def",
            Self::Abbrev => "abbrev",
            Self::Theorem => "theorem",
            Self::Instance => "instance",
            Self::Axiom => "axiom",
            Self::Constant => "constant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanDepEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone)]
pub struct LeanDecl {
    pub id: String,
    pub name: String,
    pub short_name: String,
    pub kind: LeanDeclKind,
    pub namespace_path: Vec<String>,
    pub section_path: Vec<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: String,
    pub doc_comment: Option<String>,
    pub refs: Vec<String>,
    pub local_deps: Vec<String>,
    pub unresolved_refs: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct LeanDeclGraph {
    pub path: String,
    pub imports: Vec<String>,
    pub namespaces: Vec<String>,
    pub declarations: Vec<LeanDecl>,
    pub edges: Vec<LeanDepEdge>,
    pub unresolved_refs: Vec<String>,
}

impl LeanDeclGraph {
    pub fn to_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("### Lean Theory Graph\n");

        if !self.imports.is_empty() {
            out.push_str(&format!("Imports ({}): ", self.imports.len()));
            out.push_str(
                &self
                    .imports
                    .iter()
                    .map(|module| format!("`{}`", module))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push('\n');
        }

        if !self.namespaces.is_empty() {
            out.push_str(&format!("Namespaces ({}): ", self.namespaces.len()));
            out.push_str(
                &self
                    .namespaces
                    .iter()
                    .map(|name| format!("`{}`", name))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push('\n');
        }

        out.push_str(&format!("Declarations: {}\n\n", self.declarations.len()));

        let theorem_like: Vec<_> = self
            .declarations
            .iter()
            .filter(|decl| matches!(decl.kind, LeanDeclKind::Theorem))
            .collect();
        if !theorem_like.is_empty() {
            out.push_str("### Theorems\n");
            for decl in theorem_like {
                let deps = if decl.local_deps.is_empty() {
                    "no local theorem/decl deps".to_string()
                } else {
                    format!("deps: {}", decl.local_deps.join(", "))
                };
                out.push_str(&format!(
                    "- `{}` (lines {}-{}): {}\n",
                    decl.name, decl.start_line, decl.end_line, deps
                ));
            }
            out.push('\n');
        }

        out.push_str("### Declarations\n");
        for decl in &self.declarations {
            let deps = if decl.local_deps.is_empty() {
                "none".to_string()
            } else {
                decl.local_deps.join(", ")
            };
            out.push_str(&format!(
                "- `[{}]` `{}` lines {}-{} | local deps: {}\n",
                decl.kind.label(),
                decl.name,
                decl.start_line,
                decl.end_line,
                deps
            ));
        }

        if !self.edges.is_empty() {
            out.push_str("\n### Local Dependency Edges\n");
            for edge in &self.edges {
                out.push_str(&format!("- `{}` -> `{}`\n", edge.from, edge.to));
            }
        }

        if !self.unresolved_refs.is_empty() {
            out.push_str(&format!(
                "\n### External / Unresolved References ({})\n",
                self.unresolved_refs.len()
            ));
            for reference in self.unresolved_refs.iter().take(20) {
                out.push_str(&format!("- `{}`\n", reference));
            }
            if self.unresolved_refs.len() > 20 {
                out.push_str(&format!(
                    "- ... +{} more\n",
                    self.unresolved_refs.len() - 20
                ));
            }
        }

        out
    }
}

pub struct LeanGraphAnalyzer;

impl LeanGraphAnalyzer {
    pub fn analyze(path: &str, source: &str, parser: &mut AstParser) -> Option<LeanDeclGraph> {
        let tree = parser.parse(source, Lang::Lean)?;
        let mut graph = LeanDeclGraph {
            path: path.to_string(),
            ..LeanDeclGraph::default()
        };

        Self::collect_node(
            tree.root_node(),
            source,
            &[],
            &[],
            &mut graph.imports,
            &mut graph.namespaces,
            &mut graph.declarations,
        );

        graph.imports.sort();
        graph.imports.dedup();
        graph.namespaces.sort();
        graph.namespaces.dedup();

        let qualified_names: BTreeSet<String> = graph
            .declarations
            .iter()
            .map(|decl| decl.name.clone())
            .collect();
        let unique_short_names = Self::unique_short_names(&graph.declarations);

        let mut edges = BTreeSet::new();
        let mut unresolved_refs = BTreeSet::new();

        for decl in &mut graph.declarations {
            let mut local_deps = BTreeSet::new();
            let mut unresolved = BTreeSet::new();

            for reference in &decl.refs {
                if let Some(target) = Self::resolve_reference(
                    reference,
                    &decl.namespace_path,
                    &qualified_names,
                    &unique_short_names,
                ) {
                    local_deps.insert(target);
                } else {
                    unresolved.insert(reference.clone());
                    unresolved_refs.insert(reference.clone());
                }
            }

            decl.local_deps = local_deps.into_iter().collect();
            decl.unresolved_refs = unresolved.into_iter().collect();

            for dep in &decl.local_deps {
                edges.insert((decl.name.clone(), dep.clone()));
            }
        }

        graph.edges = edges
            .into_iter()
            .map(|(from, to)| LeanDepEdge { from, to })
            .collect();
        graph.unresolved_refs = unresolved_refs.into_iter().collect();

        Some(graph)
    }

    fn collect_node(
        node: Node,
        source: &str,
        namespace_path: &[String],
        section_path: &[String],
        imports: &mut Vec<String>,
        namespaces: &mut Vec<String>,
        declarations: &mut Vec<LeanDecl>,
    ) {
        match node.kind() {
            "import" => {
                if let Some(module_name) = Self::field_text(node, "module", source) {
                    imports.push(module_name);
                }
                return;
            }
            "namespace" => {
                let Some(name) = Self::field_text(node, "name", source)
                    .or_else(|| Self::find_identifier(node, source))
                else {
                    return;
                };

                let mut next_namespace = namespace_path.to_vec();
                next_namespace.push(name);
                namespaces.push(next_namespace.join("."));

                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    Self::collect_node(
                        child,
                        source,
                        &next_namespace,
                        section_path,
                        imports,
                        namespaces,
                        declarations,
                    );
                }
                return;
            }
            "section" => {
                let label = Self::field_text(node, "name", source)
                    .or_else(|| Self::find_identifier(node, source))
                    .unwrap_or_else(|| format!("section@L{}", node.start_position().row + 1));

                let mut next_section = section_path.to_vec();
                next_section.push(label);

                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    Self::collect_node(
                        child,
                        source,
                        namespace_path,
                        &next_section,
                        imports,
                        namespaces,
                        declarations,
                    );
                }
                return;
            }
            _ => {}
        }

        if let Some(kind) = LeanDeclKind::from_node_kind(node.kind()) {
            declarations.push(Self::extract_decl(
                node,
                source,
                kind,
                namespace_path,
                section_path,
            ));
            return;
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_node(
                child,
                source,
                namespace_path,
                section_path,
                imports,
                namespaces,
                declarations,
            );
        }
    }

    fn extract_decl(
        node: Node,
        source: &str,
        kind: LeanDeclKind,
        namespace_path: &[String],
        section_path: &[String],
    ) -> LeanDecl {
        let explicit_name =
            Self::field_text(node, "name", source).or_else(|| Self::find_identifier(node, source));
        let short_name = explicit_name.unwrap_or_else(|| match kind {
            LeanDeclKind::Instance => format!("instance@L{}", node.start_position().row + 1),
            _ => format!("anonymous@L{}", node.start_position().row + 1),
        });
        let name = if namespace_path.is_empty() {
            short_name.clone()
        } else {
            format!("{}.{}", namespace_path.join("."), short_name)
        };

        let id = if section_path.is_empty() {
            name.clone()
        } else {
            format!("{}@{}", name, section_path.join("/"))
        };

        let name_range = node
            .child_by_field_name("name")
            .map(|field| (field.start_byte(), field.end_byte()));

        let binder_names = Self::binder_names(node, source);
        let refs = Self::collect_decl_refs(node, source, name_range, &binder_names);
        let signature = node
            .utf8_text(source.as_bytes())
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();

        LeanDecl {
            id,
            name,
            short_name,
            kind,
            namespace_path: namespace_path.to_vec(),
            section_path: section_path.to_vec(),
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            signature,
            doc_comment: Self::extract_lean_doc_comment(node, source),
            refs,
            local_deps: Vec::new(),
            unresolved_refs: Vec::new(),
        }
    }

    fn unique_short_names(decls: &[LeanDecl]) -> BTreeMap<String, Option<String>> {
        let mut map: BTreeMap<String, Option<String>> = BTreeMap::new();

        for decl in decls {
            match map.get(&decl.short_name) {
                None => {
                    map.insert(decl.short_name.clone(), Some(decl.name.clone()));
                }
                Some(Some(existing)) if existing != &decl.name => {
                    map.insert(decl.short_name.clone(), None);
                }
                _ => {}
            }
        }

        map
    }

    fn resolve_reference(
        reference: &str,
        namespace_path: &[String],
        qualified_names: &BTreeSet<String>,
        unique_short_names: &BTreeMap<String, Option<String>>,
    ) -> Option<String> {
        if qualified_names.contains(reference) {
            return Some(reference.to_string());
        }

        if !reference.contains('.') {
            for idx in (0..=namespace_path.len()).rev() {
                let candidate = if idx == 0 {
                    reference.to_string()
                } else {
                    format!("{}.{}", namespace_path[..idx].join("."), reference)
                };
                if qualified_names.contains(&candidate) {
                    return Some(candidate);
                }
            }
        }

        unique_short_names
            .get(reference)
            .and_then(|resolved| resolved.clone())
    }

    fn collect_decl_refs(
        node: Node,
        source: &str,
        name_range: Option<(usize, usize)>,
        binder_names: &BTreeSet<String>,
    ) -> Vec<String> {
        let mut refs = BTreeSet::new();
        Self::collect_identifiers(node, source, name_range, binder_names, &mut refs);
        refs.into_iter().collect()
    }

    fn collect_identifiers(
        node: Node,
        source: &str,
        name_range: Option<(usize, usize)>,
        binder_names: &BTreeSet<String>,
        refs: &mut BTreeSet<String>,
    ) {
        if node.kind() == "identifier" {
            let range = (node.start_byte(), node.end_byte());
            if Some(range) != name_range {
                if let Ok(text) = node.utf8_text(source.as_bytes()) {
                    if !text.is_empty() && !binder_names.contains(text) {
                        refs.insert(text.to_string());
                    }
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_identifiers(child, source, name_range, binder_names, refs);
        }
    }

    fn binder_names(node: Node, source: &str) -> BTreeSet<String> {
        let mut binders = BTreeSet::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "binders" {
                Self::collect_raw_identifiers(child, source, &mut binders);
            }
        }
        binders
    }

    fn collect_raw_identifiers(node: Node, source: &str, binders: &mut BTreeSet<String>) {
        if node.kind() == "identifier" {
            if let Ok(text) = node.utf8_text(source.as_bytes()) {
                if !text.is_empty() {
                    binders.insert(text.to_string());
                }
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_raw_identifiers(child, source, binders);
        }
    }

    fn field_text(node: Node, field_name: &str, source: &str) -> Option<String> {
        node.child_by_field_name(field_name)
            .and_then(|field| field.utf8_text(source.as_bytes()).ok())
            .map(|text| text.to_string())
    }

    fn find_identifier(node: Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                return child
                    .utf8_text(source.as_bytes())
                    .ok()
                    .map(|text| text.to_string());
            }
        }
        None
    }

    fn extract_lean_doc_comment(node: Node, source: &str) -> Option<String> {
        let mut comments = Vec::new();
        let mut prev = node.prev_sibling();

        while let Some(sibling) = prev {
            if sibling.kind() != "comment" {
                break;
            }

            if let Ok(text) = sibling.utf8_text(source.as_bytes()) {
                if text.starts_with("/--") {
                    comments.push(
                        text.trim_start_matches("/--")
                            .trim_end_matches("-/")
                            .trim()
                            .to_string(),
                    );
                } else if text.starts_with("--") {
                    comments.push(text.trim_start_matches("--").trim().to_string());
                } else {
                    break;
                }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_local_theory_graph() {
        let mut parser = AstParser::new();
        let source = r#"
import Mathlib.Data.Nat.Basic

namespace Demo

/-- Increment by one. -/
def helper (n : Nat) : Nat := Nat.succ n

theorem helper_gt (n : Nat) : helper n > n := by
  exact Nat.lt_succ_self n

theorem use_helper (n : Nat) : helper n > n := by
  exact helper_gt n

end Demo
"#;

        let graph = LeanGraphAnalyzer::analyze("Demo.lean", source, &mut parser)
            .expect("should parse Lean graph");

        assert_eq!(graph.imports, vec!["Mathlib.Data.Nat.Basic"]);
        assert!(graph.namespaces.contains(&"Demo".to_string()));
        assert_eq!(graph.declarations.len(), 3);

        let helper_gt = graph
            .declarations
            .iter()
            .find(|decl| decl.name == "Demo.helper_gt")
            .expect("helper_gt declaration");
        assert!(helper_gt.local_deps.contains(&"Demo.helper".to_string()));
        assert!(
            helper_gt
                .unresolved_refs
                .contains(&"Nat.lt_succ_self".to_string())
                || helper_gt.unresolved_refs.contains(&"Nat".to_string())
        );

        let use_helper = graph
            .declarations
            .iter()
            .find(|decl| decl.name == "Demo.use_helper")
            .expect("use_helper declaration");
        assert!(use_helper.local_deps.contains(&"Demo.helper".to_string()));
        assert!(use_helper.local_deps.contains(&"Demo.helper_gt".to_string()));
    }
}
