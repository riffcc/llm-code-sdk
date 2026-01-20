//! Program Dependence Graph (Layer 5) - unified control + data dependencies.
//!
//! Combines CFG (control dependencies) and DFG (data dependencies) into a single graph.
//! Enables program slicing:
//! - backward_slice: what affects a given line?
//! - forward_slice: what does a given line affect?
//!
//! This is the "killer feature" for debugging - show only relevant code.

use std::collections::{HashMap, HashSet, VecDeque};

use super::cfg::{CfgAnalyzer, CfgInfo, BranchKind};
use super::dfg::{DfgAnalyzer, DfgInfo, RefType};
use super::ast::Lang;

/// A node in the PDG.
#[derive(Debug, Clone)]
pub struct PdgNode {
    pub id: usize,
    pub line: usize,
    pub node_type: PdgNodeType,
    pub defines: Vec<String>,  // Variables defined at this node
    pub uses: Vec<String>,     // Variables used at this node
}

/// Type of PDG node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdgNodeType {
    Entry,
    Exit,
    Statement,
    Branch,
    Loop,
}

/// An edge in the PDG.
#[derive(Debug, Clone)]
pub struct PdgEdge {
    pub from: usize,
    pub to: usize,
    pub dep_type: DependenceType,
    pub label: String,
}

/// Type of dependence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependenceType {
    /// Control dependence (from CFG)
    Control,
    /// Data dependence (from DFG)
    Data,
}

/// Program Dependence Graph for a function.
#[derive(Debug, Clone)]
pub struct PdgInfo {
    pub function_name: String,
    pub cfg: CfgInfo,
    pub dfg: DfgInfo,
    pub nodes: Vec<PdgNode>,
    pub edges: Vec<PdgEdge>,
    /// Map from line number to node IDs
    line_to_nodes: HashMap<usize, Vec<usize>>,
}

impl PdgInfo {
    /// Backward slice: find all lines that can affect the given line.
    /// Optionally filter to a specific variable.
    pub fn backward_slice(&self, line: usize, variable: Option<&str>) -> Vec<usize> {
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from nodes at the given line
        if let Some(node_ids) = self.line_to_nodes.get(&line) {
            for &id in node_ids {
                queue.push_back(id);
            }
        }

        // BFS backward through dependencies
        while let Some(node_id) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            if let Some(node) = self.nodes.get(node_id) {
                result.insert(node.line);
            }

            // Follow incoming edges backward
            for edge in &self.edges {
                if edge.to == node_id {
                    // If filtering by variable, only follow data edges for that variable
                    if let Some(var) = variable {
                        if edge.dep_type == DependenceType::Data && edge.label != var {
                            continue;
                        }
                    }
                    queue.push_back(edge.from);
                }
            }
        }

        let mut lines: Vec<_> = result.into_iter().collect();
        lines.sort();
        lines
    }

    /// Forward slice: find all lines that can be affected by the given line.
    /// Optionally filter to a specific variable.
    pub fn forward_slice(&self, line: usize, variable: Option<&str>) -> Vec<usize> {
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from nodes at the given line
        if let Some(node_ids) = self.line_to_nodes.get(&line) {
            for &id in node_ids {
                queue.push_back(id);
            }
        }

        // BFS forward through dependencies
        while let Some(node_id) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            if let Some(node) = self.nodes.get(node_id) {
                result.insert(node.line);
            }

            // Follow outgoing edges forward
            for edge in &self.edges {
                if edge.from == node_id {
                    // If filtering by variable, only follow data edges for that variable
                    if let Some(var) = variable {
                        if edge.dep_type == DependenceType::Data && edge.label != var {
                            continue;
                        }
                    }
                    queue.push_back(edge.to);
                }
            }
        }

        let mut lines: Vec<_> = result.into_iter().collect();
        lines.sort();
        lines
    }

    /// Get a slice of the source code showing only relevant lines.
    pub fn slice_source(&self, source: &str, lines: &[usize]) -> String {
        let source_lines: Vec<&str> = source.lines().collect();
        let _line_set: HashSet<_> = lines.iter().collect();

        let mut output = String::new();
        let mut last_line = 0;

        for &line in lines {
            if line == 0 || line > source_lines.len() {
                continue;
            }

            // Show gap indicator if lines were skipped
            if last_line > 0 && line > last_line + 1 {
                output.push_str("    ...\n");
            }

            output.push_str(&format!("{:4} │ {}\n", line, source_lines[line - 1]));
            last_line = line;
        }

        output
    }

    /// Format as summary.
    pub fn summary(&self) -> String {
        let control_edges = self.edges.iter()
            .filter(|e| e.dep_type == DependenceType::Control)
            .count();
        let data_edges = self.edges.iter()
            .filter(|e| e.dep_type == DependenceType::Data)
            .count();

        format!(
            "{}: {} nodes, {} control deps, {} data deps",
            self.function_name,
            self.nodes.len(),
            control_edges,
            data_edges
        )
    }

    /// Format as LLM-ready context with slicing info.
    pub fn to_context(&self, focus_line: Option<usize>) -> String {
        let mut output = String::new();

        output.push_str(&format!("### PDG: {}\n", self.function_name));
        output.push_str(&format!("Nodes: {}, Edges: {}\n", self.nodes.len(), self.edges.len()));

        // If a focus line is given, show slices
        if let Some(line) = focus_line {
            let backward = self.backward_slice(line, None);
            let forward = self.forward_slice(line, None);

            output.push_str(&format!("\nFocus: line {}\n", line));
            output.push_str(&format!("← Affects this: lines {:?}\n", backward));
            output.push_str(&format!("→ Affected by this: lines {:?}\n", forward));
        }

        // Show key dependencies
        output.push_str("\nKey dependencies:\n");
        for edge in self.edges.iter().take(10) {
            let dep = if edge.dep_type == DependenceType::Control { "ctrl" } else { "data" };
            output.push_str(&format!(
                "  L{} → L{} [{}:{}]\n",
                self.nodes.get(edge.from).map(|n| n.line).unwrap_or(0),
                self.nodes.get(edge.to).map(|n| n.line).unwrap_or(0),
                dep,
                edge.label
            ));
        }
        if self.edges.len() > 10 {
            output.push_str(&format!("  ... and {} more\n", self.edges.len() - 10));
        }

        output
    }
}

/// PDG builder combining CFG and DFG.
pub struct PdgBuilder;

impl PdgBuilder {
    /// Build PDG for all functions in source code.
    pub fn analyze(source: &str, lang: Lang) -> Vec<PdgInfo> {
        let cfgs = CfgAnalyzer::analyze(source, lang);
        let dfgs = DfgAnalyzer::analyze(source, lang);

        // Match CFG and DFG by function name
        let mut results = Vec::new();

        for cfg in cfgs {
            let dfg = dfgs.iter()
                .find(|d| d.function_name == cfg.function_name)
                .cloned()
                .unwrap_or_else(|| DfgInfo {
                    function_name: cfg.function_name.clone(),
                    refs: vec![],
                    edges: vec![],
                    variables: vec![],
                });

            let pdg = Self::build(cfg, dfg);
            results.push(pdg);
        }

        results
    }

    fn build(cfg: CfgInfo, dfg: DfgInfo) -> PdgInfo {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut line_to_nodes: HashMap<usize, Vec<usize>> = HashMap::new();

        // Create nodes from CFG branches (control points)
        // Start with entry node
        let entry_id = nodes.len();
        nodes.push(PdgNode {
            id: entry_id,
            line: 0,
            node_type: PdgNodeType::Entry,
            defines: vec![],
            uses: vec![],
        });

        // Create nodes for each branch point
        for branch in &cfg.branches {
            let node_type = match branch.kind {
                BranchKind::If | BranchKind::ElseIf | BranchKind::Else => PdgNodeType::Branch,
                BranchKind::Match | BranchKind::MatchArm => PdgNodeType::Branch,
                BranchKind::Loop | BranchKind::While | BranchKind::For => PdgNodeType::Loop,
                _ => PdgNodeType::Statement,
            };

            let node_id = nodes.len();

            // Get defs/uses at this line from DFG
            let defines: Vec<_> = dfg.refs.iter()
                .filter(|r| r.line == branch.line && matches!(r.ref_type, RefType::Definition | RefType::Update))
                .map(|r| r.name.clone())
                .collect();
            let uses: Vec<_> = dfg.refs.iter()
                .filter(|r| r.line == branch.line && matches!(r.ref_type, RefType::Use | RefType::Update))
                .map(|r| r.name.clone())
                .collect();

            nodes.push(PdgNode {
                id: node_id,
                line: branch.line,
                node_type,
                defines,
                uses,
            });

            line_to_nodes.entry(branch.line).or_default().push(node_id);

            // Add control edge from entry or previous control point
            edges.push(PdgEdge {
                from: entry_id,
                to: node_id,
                dep_type: DependenceType::Control,
                label: format!("{:?}", branch.kind),
            });
        }

        // Create nodes for statement lines from DFG
        let mut statement_lines: HashSet<usize> = HashSet::new();
        for r in &dfg.refs {
            statement_lines.insert(r.line);
        }

        for line in statement_lines {
            // Skip if we already have a node for this line
            if line_to_nodes.contains_key(&line) {
                continue;
            }

            let node_id = nodes.len();
            let defines: Vec<_> = dfg.refs.iter()
                .filter(|r| r.line == line && matches!(r.ref_type, RefType::Definition | RefType::Update))
                .map(|r| r.name.clone())
                .collect();
            let uses: Vec<_> = dfg.refs.iter()
                .filter(|r| r.line == line && matches!(r.ref_type, RefType::Use | RefType::Update))
                .map(|r| r.name.clone())
                .collect();

            nodes.push(PdgNode {
                id: node_id,
                line,
                node_type: PdgNodeType::Statement,
                defines,
                uses,
            });

            line_to_nodes.entry(line).or_default().push(node_id);
        }

        // Add data edges from DFG
        for dfg_edge in &dfg.edges {
            // Find nodes at def and use lines
            let from_nodes = line_to_nodes.get(&dfg_edge.def_line).cloned().unwrap_or_default();
            let to_nodes = line_to_nodes.get(&dfg_edge.use_line).cloned().unwrap_or_default();

            for &from in &from_nodes {
                for &to in &to_nodes {
                    edges.push(PdgEdge {
                        from,
                        to,
                        dep_type: DependenceType::Data,
                        label: dfg_edge.variable.clone(),
                    });
                }
            }
        }

        // Add exit node
        let exit_id = nodes.len();
        nodes.push(PdgNode {
            id: exit_id,
            line: 0,
            node_type: PdgNodeType::Exit,
            defines: vec![],
            uses: vec![],
        });

        // Connect nodes to exit (simplified: all nodes connect to exit)
        for node in &nodes {
            if node.node_type != PdgNodeType::Exit && node.node_type != PdgNodeType::Entry {
                edges.push(PdgEdge {
                    from: node.id,
                    to: exit_id,
                    dep_type: DependenceType::Control,
                    label: "exit".to_string(),
                });
            }
        }

        let function_name = cfg.function_name.clone();
        PdgInfo {
            function_name,
            cfg,
            dfg,
            nodes,
            edges,
            line_to_nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdg_construction() {
        let source = r#"
fn process(input: i32) -> i32 {
    let x = input + 1;
    let y = x * 2;
    if y > 10 {
        return y;
    }
    let z = x + y;
    z
}
"#;

        let pdgs = PdgBuilder::analyze(source, Lang::Rust);
        assert_eq!(pdgs.len(), 1);

        let pdg = &pdgs[0];
        assert_eq!(pdg.function_name, "process");
        assert!(!pdg.nodes.is_empty());
        assert!(!pdg.edges.is_empty());

        println!("PDG summary: {}", pdg.summary());
        println!("PDG context:\n{}", pdg.to_context(Some(5)));
    }

    #[test]
    fn test_backward_slice() {
        let source = r#"
fn example() {
    let a = 1;
    let b = 2;
    let c = a + b;
    let d = c * 2;
    d
}
"#;

        let pdgs = PdgBuilder::analyze(source, Lang::Rust);
        let pdg = &pdgs[0];

        // Backward slice from line 6 (d = c * 2) should include lines defining a, b, c
        let slice = pdg.backward_slice(6, None);
        println!("Backward slice from line 6: {:?}", slice);

        // Should contain the line itself
        assert!(slice.contains(&6));
    }

    #[test]
    fn test_forward_slice() {
        let source = r#"
fn example() {
    let a = 1;
    let b = a + 1;
    let c = b + 1;
    c
}
"#;

        let pdgs = PdgBuilder::analyze(source, Lang::Rust);
        let pdg = &pdgs[0];

        // Forward slice from line 3 (a = 1) should include b and c
        let slice = pdg.forward_slice(3, None);
        println!("Forward slice from line 3: {:?}", slice);

        // Should contain the starting line
        assert!(slice.contains(&3));
    }

    #[test]
    fn test_slice_source() {
        let source = r#"fn foo() {
    let a = 1;
    let b = 2;
    let c = 3;
    let d = a + c;
    d
}"#;

        let pdgs = PdgBuilder::analyze(source, Lang::Rust);
        let pdg = &pdgs[0];

        // Create a slice of specific lines
        let lines = vec![2, 4, 5];
        let sliced = pdg.slice_source(source, &lines);

        println!("Sliced source:\n{}", sliced);
        assert!(sliced.contains("let a = 1"));
        assert!(sliced.contains("let c = 3"));
        assert!(sliced.contains("let d = a + c"));
        assert!(!sliced.contains("let b = 2")); // b is not in the slice
    }
}
