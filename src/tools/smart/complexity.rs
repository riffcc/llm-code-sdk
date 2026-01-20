//! Edit complexity analyzer for SmartWrite.
//!
//! Uses CFG and PDG analysis to determine the complexity of a proposed edit,
//! which is then used to select the appropriate model tier via AssuranceLevel.
//!
//! Complexity factors:
//! - Cyclomatic complexity of affected functions (CFG)
//! - Number of dependencies affected (PDG backward/forward slice)
//! - Structural scope (lines, functions, files)
//! - Breaking change risk (signature changes, deletions)

use super::ast::{AstParser, Lang, Symbol, SymbolKind};
use super::cfg::CfgAnalyzer;
use super::pdg::PdgBuilder;
use std::path::Path;

/// Edit complexity level, maps to AssuranceLevel for model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditComplexity {
    /// Trivial: rename, comment, formatting. Use FAST or Devstral.
    Trivial,
    /// Simple: add single function, fix obvious bug. Use FLASH.
    Simple,
    /// Medium: refactor function, change logic. Use MINI.
    Medium,
    /// Complex: multi-function changes, architectural. Use FULL.
    Complex,
    /// Critical: breaking changes, deletions, API changes. Use FULL+OC.
    Critical,
}

impl EditComplexity {
    /// Get recommended AssuranceLevel name for this complexity.
    pub fn recommended_level(&self) -> &'static str {
        match self {
            EditComplexity::Trivial => "Fast",
            EditComplexity::Simple => "Flash",
            EditComplexity::Medium => "Mini",
            EditComplexity::Complex => "Full",
            EditComplexity::Critical => "FullLowOC",
        }
    }

    /// Get description of this complexity level.
    pub fn description(&self) -> &'static str {
        match self {
            EditComplexity::Trivial => "Trivial edit (rename, comment, formatting)",
            EditComplexity::Simple => "Simple edit (add function, fix obvious bug)",
            EditComplexity::Medium => "Medium edit (refactor, change logic)",
            EditComplexity::Complex => "Complex edit (multi-function, architectural)",
            EditComplexity::Critical => "Critical edit (breaking changes, API changes)",
        }
    }
}

/// Analysis result for an edit's complexity.
#[derive(Debug, Clone)]
pub struct ComplexityAnalysis {
    /// Overall complexity level.
    pub complexity: EditComplexity,
    /// Cyclomatic complexity of affected functions.
    pub cyclomatic_complexity: usize,
    /// Number of affected lines.
    pub affected_lines: usize,
    /// Number of affected functions.
    pub affected_functions: usize,
    /// Number of backward dependencies (what affects the edit).
    pub backward_deps: usize,
    /// Number of forward dependencies (what the edit affects).
    pub forward_deps: usize,
    /// Whether this is a breaking change (signature change, deletion).
    pub is_breaking: bool,
    /// Explanation of the complexity rating.
    pub explanation: String,
}

impl ComplexityAnalysis {
    /// Create a new trivial analysis.
    pub fn trivial(reason: &str) -> Self {
        Self {
            complexity: EditComplexity::Trivial,
            cyclomatic_complexity: 1,
            affected_lines: 1,
            affected_functions: 0,
            backward_deps: 0,
            forward_deps: 0,
            is_breaking: false,
            explanation: reason.to_string(),
        }
    }

    /// Calculate overall complexity from the factors.
    fn calculate_complexity(&mut self) {
        // Breaking changes are always at least Complex
        if self.is_breaking {
            self.complexity = EditComplexity::Critical;
            return;
        }

        let mut score = 0;

        // Cyclomatic complexity contribution
        score += match self.cyclomatic_complexity {
            0..=2 => 0,
            3..=5 => 1,
            6..=10 => 2,
            11..=20 => 3,
            _ => 4,
        };

        // Affected lines contribution
        score += match self.affected_lines {
            0..=5 => 0,
            6..=20 => 1,
            21..=50 => 2,
            51..=100 => 3,
            _ => 4,
        };

        // Affected functions contribution
        score += match self.affected_functions {
            0..=1 => 0,
            2..=3 => 1,
            4..=6 => 2,
            _ => 3,
        };

        // Dependencies contribution (backward + forward)
        let total_deps = self.backward_deps + self.forward_deps;
        score += match total_deps {
            0..=2 => 0,
            3..=5 => 1,
            6..=10 => 2,
            _ => 3,
        };

        // Map score to complexity
        self.complexity = match score {
            0..=1 => EditComplexity::Trivial,
            2..=4 => EditComplexity::Simple,
            5..=7 => EditComplexity::Medium,
            8..=10 => EditComplexity::Complex,
            _ => EditComplexity::Critical,
        };
    }
}

/// Analyzer for determining edit complexity.
pub struct EditComplexityAnalyzer {
    parser: AstParser,
}

impl EditComplexityAnalyzer {
    pub fn new() -> Self {
        Self {
            parser: AstParser::new(),
        }
    }

    /// Analyze complexity of replacing a symbol.
    pub fn analyze_replace(
        &mut self,
        source: &str,
        lang: Lang,
        target_symbol: &str,
        new_content: &str,
    ) -> ComplexityAnalysis {
        let symbols = self.parser.extract_symbols(source, lang);

        // Find the target symbol
        let target = symbols.iter().find(|s| s.name == target_symbol);

        let Some(target) = target else {
            return ComplexityAnalysis::trivial("Symbol not found, new addition");
        };

        let old_lines = target.end_line - target.start_line + 1;
        let new_lines = new_content.lines().count();

        // Analyze CFG complexity of the function being replaced
        let cfgs = CfgAnalyzer::analyze(source, lang);
        let target_cfg = cfgs.iter().find(|c| c.function_name == target_symbol);
        let cyclomatic = target_cfg.map(|c| c.cyclomatic_complexity).unwrap_or(1);

        // Analyze PDG for dependencies
        let pdgs = PdgBuilder::analyze(source, lang);
        let target_pdg = pdgs.iter().find(|p| p.function_name == target_symbol);

        let (backward_deps, forward_deps) = if let Some(pdg) = target_pdg {
            // Slice from middle of the function
            let mid_line = (target.start_line + target.end_line) / 2;
            (
                pdg.backward_slice(mid_line, None).len(),
                pdg.forward_slice(mid_line, None).len(),
            )
        } else {
            (0, 0)
        };

        // Check if it's a breaking change
        let is_breaking = self.is_breaking_change(target, new_content, lang);

        let mut analysis = ComplexityAnalysis {
            complexity: EditComplexity::Simple,
            cyclomatic_complexity: cyclomatic,
            affected_lines: old_lines.max(new_lines),
            affected_functions: 1,
            backward_deps,
            forward_deps,
            is_breaking,
            explanation: format!(
                "Replacing {} ({} lines → {} lines, complexity {})",
                target_symbol, old_lines, new_lines, cyclomatic
            ),
        };

        analysis.calculate_complexity();
        analysis
    }

    /// Analyze complexity of inserting new code.
    pub fn analyze_insert(
        &mut self,
        source: &str,
        lang: Lang,
        new_content: &str,
    ) -> ComplexityAnalysis {
        let new_lines = new_content.lines().count();

        // Parse the new content to see what's being added
        let new_symbols = self.parser.extract_symbols(new_content, lang);
        let new_functions = new_symbols
            .iter()
            .filter(|s| matches!(s.kind, SymbolKind::Function | SymbolKind::Method))
            .count();

        // Analyze CFG of new content
        let cfgs = CfgAnalyzer::analyze(new_content, lang);
        let max_complexity = cfgs.iter().map(|c| c.cyclomatic_complexity).max().unwrap_or(1);

        let mut analysis = ComplexityAnalysis {
            complexity: EditComplexity::Simple,
            cyclomatic_complexity: max_complexity,
            affected_lines: new_lines,
            affected_functions: new_functions,
            backward_deps: 0,
            forward_deps: 0,
            is_breaking: false,
            explanation: format!(
                "Inserting {} lines, {} functions (max complexity {})",
                new_lines, new_functions, max_complexity
            ),
        };

        analysis.calculate_complexity();
        analysis
    }

    /// Analyze complexity of deleting a symbol.
    pub fn analyze_delete(
        &mut self,
        source: &str,
        lang: Lang,
        target_symbol: &str,
    ) -> ComplexityAnalysis {
        let symbols = self.parser.extract_symbols(source, lang);

        let target = symbols.iter().find(|s| s.name == target_symbol);

        let Some(target) = target else {
            return ComplexityAnalysis::trivial("Symbol not found");
        };

        let lines = target.end_line - target.start_line + 1;

        // Deletions are always potentially breaking
        let mut analysis = ComplexityAnalysis {
            complexity: EditComplexity::Complex,
            cyclomatic_complexity: 0,
            affected_lines: lines,
            affected_functions: 1,
            backward_deps: 0,
            forward_deps: 0,
            is_breaking: true, // Deletions are always breaking
            explanation: format!(
                "Deleting {} ({} lines) - potentially breaking",
                target_symbol, lines
            ),
        };

        analysis.calculate_complexity();
        analysis
    }

    /// Analyze complexity of a line range edit.
    pub fn analyze_line_edit(
        &mut self,
        source: &str,
        lang: Lang,
        start_line: usize,
        end_line: usize,
        new_content: &str,
    ) -> ComplexityAnalysis {
        let symbols = self.parser.extract_symbols(source, lang);

        // Find functions affected by this line range
        let affected_funcs: Vec<&Symbol> = symbols
            .iter()
            .filter(|s| {
                matches!(s.kind, SymbolKind::Function | SymbolKind::Method)
                    && s.start_line <= end_line
                    && s.end_line >= start_line
            })
            .collect();

        let old_lines = end_line - start_line + 1;
        let new_lines = new_content.lines().count();

        // Get PDG slices for the affected range
        let pdgs = PdgBuilder::analyze(source, lang);
        let mut total_backward = 0;
        let mut total_forward = 0;

        for line in start_line..=end_line {
            for pdg in &pdgs {
                total_backward += pdg.backward_slice(line, None).len();
                total_forward += pdg.forward_slice(line, None).len();
            }
        }

        // Deduplicate (rough estimate)
        total_backward = total_backward / old_lines.max(1);
        total_forward = total_forward / old_lines.max(1);

        // Get max cyclomatic complexity of affected functions
        let cfgs = CfgAnalyzer::analyze(source, lang);
        let max_complexity = affected_funcs
            .iter()
            .filter_map(|f| cfgs.iter().find(|c| c.function_name == f.name))
            .map(|c| c.cyclomatic_complexity)
            .max()
            .unwrap_or(1);

        let mut analysis = ComplexityAnalysis {
            complexity: EditComplexity::Simple,
            cyclomatic_complexity: max_complexity,
            affected_lines: old_lines.max(new_lines),
            affected_functions: affected_funcs.len(),
            backward_deps: total_backward,
            forward_deps: total_forward,
            is_breaking: false,
            explanation: format!(
                "Editing lines {}-{} ({} → {} lines), {} functions affected",
                start_line,
                end_line,
                old_lines,
                new_lines,
                affected_funcs.len()
            ),
        };

        analysis.calculate_complexity();
        analysis
    }

    /// Check if replacing a symbol is a breaking change.
    fn is_breaking_change(&mut self, old_symbol: &Symbol, new_content: &str, lang: Lang) -> bool {
        // Parse the new content
        let new_symbols = self.parser.extract_symbols(new_content, lang);

        // Find the replacement symbol
        let new_symbol = new_symbols.iter().find(|s| s.name == old_symbol.name);

        let Some(new_symbol) = new_symbol else {
            // Name changed - breaking
            return true;
        };

        // For functions/methods, check signature
        if matches!(
            old_symbol.kind,
            SymbolKind::Function | SymbolKind::Method
        ) {
            // Compare signatures if available
            if let (Some(old_sig), Some(new_sig)) =
                (&old_symbol.signature, &new_symbol.signature)
            {
                // Signature changed - breaking
                if old_sig != new_sig {
                    return true;
                }
            }
        }

        // For structs/enums, check if fields changed
        if matches!(old_symbol.kind, SymbolKind::Struct | SymbolKind::Enum) {
            // This would need deeper analysis, assume non-breaking for now
            // unless the entire content is significantly different
            let old_len = old_symbol.end_line - old_symbol.start_line;
            let new_len = new_symbol.end_line - new_symbol.start_line;
            if (old_len as i32 - new_len as i32).abs() > 5 {
                return true;
            }
        }

        false
    }
}

impl Default for EditComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Suggests how to split a complex edit into simpler sub-edits.
#[derive(Debug, Clone)]
pub struct EditSplit {
    /// Sub-edits that can be done by simpler models.
    pub simple_edits: Vec<SubEdit>,
    /// Sub-edits that need complex models.
    pub complex_edits: Vec<SubEdit>,
    /// Sub-edits that need review by OC models.
    pub critical_edits: Vec<SubEdit>,
}

/// A sub-edit from splitting a complex edit.
#[derive(Debug, Clone)]
pub struct SubEdit {
    /// Description of what this sub-edit does.
    pub description: String,
    /// Recommended complexity level.
    pub complexity: EditComplexity,
    /// Target symbol or line range.
    pub target: String,
    /// Whether this sub-edit depends on others.
    pub depends_on: Vec<usize>,
}

impl EditSplit {
    /// Create an empty split.
    pub fn empty() -> Self {
        Self {
            simple_edits: vec![],
            complex_edits: vec![],
            critical_edits: vec![],
        }
    }

    /// Total number of sub-edits.
    pub fn total(&self) -> usize {
        self.simple_edits.len() + self.complex_edits.len() + self.critical_edits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_edit() {
        let mut analyzer = EditComplexityAnalyzer::new();

        let source = r#"
fn hello() {
    println!("Hello");
}
"#;

        let new_content = r#"fn hello() {
    // Just a comment
    println!("Hello");
}"#;

        let analysis = analyzer.analyze_replace(source, Lang::Rust, "hello", new_content);

        // Small change to simple function should be trivial/simple
        assert!(
            analysis.complexity <= EditComplexity::Simple,
            "Expected trivial/simple, got {:?}",
            analysis.complexity
        );
    }

    #[test]
    fn test_complex_edit() {
        let mut analyzer = EditComplexityAnalyzer::new();

        let source = r#"
fn process(input: &str) -> Result<String, Error> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(Error::Empty);
    }
    if trimmed.len() < 3 {
        return Err(Error::TooShort);
    }
    if trimmed.len() > 100 {
        return Err(Error::TooLong);
    }
    let validated = validate(trimmed)?;
    let normalized = normalize(validated);
    let transformed = transform(normalized)?;
    Ok(transformed)
}
"#;

        let new_content = r#"fn process(input: &str, config: &Config) -> Result<Output, ProcessError> {
    let pipeline = Pipeline::new(config);
    pipeline.run(input)
}"#;

        let analysis = analyzer.analyze_replace(source, Lang::Rust, "process", new_content);

        // Signature change is breaking
        assert!(
            analysis.is_breaking,
            "Should be breaking due to signature change"
        );
        assert!(
            analysis.complexity >= EditComplexity::Complex,
            "Expected complex+, got {:?}",
            analysis.complexity
        );
    }

    #[test]
    fn test_delete_is_breaking() {
        let mut analyzer = EditComplexityAnalyzer::new();

        let source = r#"
fn keep_me() {
    println!("Keep");
}

fn delete_me() {
    println!("Delete");
}
"#;

        let analysis = analyzer.analyze_delete(source, Lang::Rust, "delete_me");

        assert!(analysis.is_breaking, "Deletions should be breaking");
        assert!(
            analysis.complexity >= EditComplexity::Complex,
            "Deletions should be complex+"
        );
    }

    #[test]
    fn test_insert_new_function() {
        let mut analyzer = EditComplexityAnalyzer::new();

        let source = "";
        let new_content = r#"fn simple_add(a: i32, b: i32) -> i32 {
    a + b
}"#;

        let analysis = analyzer.analyze_insert(source, Lang::Rust, new_content);

        // Simple function addition
        assert!(
            analysis.complexity <= EditComplexity::Simple,
            "Simple function insert should be simple, got {:?}",
            analysis.complexity
        );
        assert!(!analysis.is_breaking);
    }
}
