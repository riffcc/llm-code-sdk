//! Control Flow Graph (Layer 3) - cyclomatic complexity and basic blocks.
//!
//! Analyzes control flow within functions to provide:
//! - Cyclomatic complexity (decision points + 1)
//! - Basic block count
//! - Branch analysis

use super::ast::{AstParser, Lang};

/// Control flow information for a function.
#[derive(Debug, Clone)]
pub struct CfgInfo {
    pub function_name: String,
    pub cyclomatic_complexity: usize,
    pub basic_blocks: usize,
    pub branches: Vec<Branch>,
    pub has_early_return: bool,
    pub max_nesting_depth: usize,
}

/// A branch point in control flow.
#[derive(Debug, Clone)]
pub struct Branch {
    pub kind: BranchKind,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchKind {
    If,
    ElseIf,
    Else,
    Match,
    MatchArm,
    Loop,
    While,
    For,
    TryCatch,
    Return,
    Break,
    Continue,
}

/// CFG analyzer using tree-sitter.
pub struct CfgAnalyzer;

impl CfgAnalyzer {
    /// Analyze control flow for all functions in source code.
    pub fn analyze(source: &str, lang: Lang) -> Vec<CfgInfo> {
        let mut parser = AstParser::new();
        let tree = match parser.parse(source, lang) {
            Some(t) => t,
            None => return vec![],
        };

        let mut results = Vec::new();
        Self::extract_cfg_recursive(tree.root_node(), source, lang, &mut results, 0);
        results
    }

    fn extract_cfg_recursive(
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        results: &mut Vec<CfgInfo>,
        depth: usize,
    ) {
        let kind = node.kind();

        // Check if this is a function definition
        let is_function = match lang {
            Lang::Rust => kind == "function_item",
            Lang::Python => kind == "function_definition",
            Lang::JavaScript | Lang::TypeScript => {
                kind == "function_declaration" || kind == "arrow_function" || kind == "method_definition"
            }
            Lang::Go => kind == "function_declaration" || kind == "method_declaration",
            Lang::Perl => kind == "function_definition" || kind == "anonymous_function",
        };

        if is_function {
            if let Some(cfg) = Self::analyze_function(node, source, lang) {
                results.push(cfg);
            }
        }

        // Recurse into children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::extract_cfg_recursive(child, source, lang, results, depth + 1);
        }
    }

    fn analyze_function(node: tree_sitter::Node, source: &str, lang: Lang) -> Option<CfgInfo> {
        // Get function name
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .unwrap_or("<anonymous>")
            .to_string();

        let mut branches = Vec::new();
        let mut max_depth = 0;
        let mut has_early_return = false;

        Self::collect_branches(node, source, lang, &mut branches, 0, &mut max_depth, &mut has_early_return);

        // Cyclomatic complexity = decision points + 1
        let decision_points = branches
            .iter()
            .filter(|b| matches!(
                b.kind,
                BranchKind::If | BranchKind::ElseIf | BranchKind::While |
                BranchKind::For | BranchKind::Loop | BranchKind::MatchArm |
                BranchKind::TryCatch
            ))
            .count();

        let cyclomatic = decision_points + 1;

        // Basic blocks ≈ branches + 1
        let basic_blocks = branches.len() + 1;

        Some(CfgInfo {
            function_name: name,
            cyclomatic_complexity: cyclomatic,
            basic_blocks,
            branches,
            has_early_return,
            max_nesting_depth: max_depth,
        })
    }

    fn collect_branches(
        node: tree_sitter::Node,
        source: &str,
        lang: Lang,
        branches: &mut Vec<Branch>,
        depth: usize,
        max_depth: &mut usize,
        has_early_return: &mut bool,
    ) {
        let kind = node.kind();
        let line = node.start_position().row + 1;

        *max_depth = (*max_depth).max(depth);

        // Detect branch kinds based on language
        let branch_kind = match lang {
            Lang::Rust => match kind {
                "if_expression" => Some(BranchKind::If),
                "else_clause" => Some(BranchKind::Else),
                "match_expression" => Some(BranchKind::Match),
                "match_arm" => Some(BranchKind::MatchArm),
                "loop_expression" => Some(BranchKind::Loop),
                "while_expression" => Some(BranchKind::While),
                "for_expression" => Some(BranchKind::For),
                "return_expression" => {
                    *has_early_return = true;
                    Some(BranchKind::Return)
                }
                "break_expression" => Some(BranchKind::Break),
                "continue_expression" => Some(BranchKind::Continue),
                _ => None,
            },
            Lang::Python => match kind {
                "if_statement" => Some(BranchKind::If),
                "elif_clause" => Some(BranchKind::ElseIf),
                "else_clause" => Some(BranchKind::Else),
                "match_statement" => Some(BranchKind::Match),
                "case_clause" => Some(BranchKind::MatchArm),
                "while_statement" => Some(BranchKind::While),
                "for_statement" => Some(BranchKind::For),
                "try_statement" => Some(BranchKind::TryCatch),
                "return_statement" => {
                    *has_early_return = true;
                    Some(BranchKind::Return)
                }
                "break_statement" => Some(BranchKind::Break),
                "continue_statement" => Some(BranchKind::Continue),
                _ => None,
            },
            Lang::JavaScript | Lang::TypeScript => match kind {
                "if_statement" => Some(BranchKind::If),
                "else_clause" => Some(BranchKind::Else),
                "switch_statement" => Some(BranchKind::Match),
                "switch_case" => Some(BranchKind::MatchArm),
                "while_statement" => Some(BranchKind::While),
                "for_statement" | "for_in_statement" => Some(BranchKind::For),
                "try_statement" => Some(BranchKind::TryCatch),
                "return_statement" => {
                    *has_early_return = true;
                    Some(BranchKind::Return)
                }
                "break_statement" => Some(BranchKind::Break),
                "continue_statement" => Some(BranchKind::Continue),
                _ => None,
            },
            Lang::Go => match kind {
                "if_statement" => Some(BranchKind::If),
                "expression_switch_statement" | "type_switch_statement" => Some(BranchKind::Match),
                "expression_case" | "type_case" | "default_case" => Some(BranchKind::MatchArm),
                "for_statement" => Some(BranchKind::For),
                "return_statement" => {
                    *has_early_return = true;
                    Some(BranchKind::Return)
                }
                "break_statement" => Some(BranchKind::Break),
                "continue_statement" => Some(BranchKind::Continue),
                _ => None,
            },
            Lang::Perl => match kind {
                "if_statement" | "if_modifier_statement" => Some(BranchKind::If),
                "elsif_clause" => Some(BranchKind::ElseIf),
                "else_clause" => Some(BranchKind::Else),
                "unless_statement" | "unless_modifier_statement" => Some(BranchKind::If),
                "given_statement" => Some(BranchKind::Match),
                "when_clause" | "default_clause" => Some(BranchKind::MatchArm),
                "while_statement" | "while_modifier_statement" => Some(BranchKind::While),
                "until_statement" | "until_modifier_statement" => Some(BranchKind::While),
                "for_statement" | "foreach_statement" => Some(BranchKind::For),
                "eval_expression" => Some(BranchKind::TryCatch),
                "return_expression" => {
                    *has_early_return = true;
                    Some(BranchKind::Return)
                }
                "last_expression" => Some(BranchKind::Break),
                "next_expression" => Some(BranchKind::Continue),
                "redo_expression" => Some(BranchKind::Continue),
                _ => None,
            },
        };

        // Recurse with increased depth for control structures
        let new_depth = if branch_kind.is_some() { depth + 1 } else { depth };

        if let Some(bk) = branch_kind {
            branches.push(Branch { kind: bk, line });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::collect_branches(child, source, lang, branches, new_depth, max_depth, has_early_return);
        }
    }

    /// Get complexity rating.
    pub fn complexity_rating(cyclomatic: usize) -> &'static str {
        match cyclomatic {
            1..=5 => "simple",
            6..=10 => "moderate",
            11..=20 => "complex",
            21..=50 => "very complex",
            _ => "untestable",
        }
    }

    /// Format CFG info for LLM context.
    pub fn to_llm_string(info: &CfgInfo) -> String {
        let rating = Self::complexity_rating(info.cyclomatic_complexity);
        let warning = if info.cyclomatic_complexity > 10 { " ⚠️" } else { "" };

        format!(
            "⚡ complexity: {} ({} blocks) [{}]{}",
            info.cyclomatic_complexity,
            info.basic_blocks,
            rating,
            warning
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_cfg() {
        let source = r#"
fn simple() {
    println!("hello");
}

fn with_branch(x: i32) -> i32 {
    if x > 0 {
        x * 2
    } else {
        x * -1
    }
}

fn complex(x: i32) -> i32 {
    if x > 100 {
        return x;
    }

    for i in 0..x {
        if i % 2 == 0 {
            println!("even");
        } else if i % 3 == 0 {
            println!("div3");
        }
    }

    match x {
        0 => 0,
        1 => 1,
        _ => x,
    }
}
"#;

        let cfgs = CfgAnalyzer::analyze(source, Lang::Rust);

        assert_eq!(cfgs.len(), 3);

        // simple() - no branches
        let simple = cfgs.iter().find(|c| c.function_name == "simple").unwrap();
        assert_eq!(simple.cyclomatic_complexity, 1);

        // with_branch() - one if
        let branch = cfgs.iter().find(|c| c.function_name == "with_branch").unwrap();
        assert_eq!(branch.cyclomatic_complexity, 2); // if + 1

        // complex() - if + for + if + else if + match with 3 arms
        let complex = cfgs.iter().find(|c| c.function_name == "complex").unwrap();
        assert!(complex.cyclomatic_complexity >= 5);
        assert!(complex.has_early_return);
    }

    #[test]
    fn test_python_cfg() {
        let source = r#"
def simple():
    print("hello")

def with_loops(items):
    for item in items:
        if item > 0:
            print(item)
        elif item < 0:
            print(-item)

    while True:
        break
"#;

        let cfgs = CfgAnalyzer::analyze(source, Lang::Python);

        assert_eq!(cfgs.len(), 2);

        let loops = cfgs.iter().find(|c| c.function_name == "with_loops").unwrap();
        assert!(loops.cyclomatic_complexity >= 4); // for + if + elif + while
    }
}
