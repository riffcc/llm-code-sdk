//! SmartWrite tool - writes code using the layercake approach.
//!
//! Uses structural understanding to make surgical edits rather than
//! full file rewrites. The LLM can specify edits at different granularities:
//! - Function level: replace/insert/delete functions
//! - Block level: modify specific code blocks
//! - Line level: precise line edits
//!
//! The tool uses AST awareness to validate edits maintain valid syntax.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::ast::{AstParser, Lang, Symbol};
use super::layers::LayerAnalyzer;
use crate::tools::{Tool, ToolResult};

fn diff_line_counts(before: &str, after: &str) -> (usize, usize) {
    let before_lines: Vec<&str> = before.lines().collect();
    let after_lines: Vec<&str> = after.lines().collect();

    let mut prefix = 0usize;
    while prefix < before_lines.len()
        && prefix < after_lines.len()
        && before_lines[prefix] == after_lines[prefix]
    {
        prefix += 1;
    }

    let mut before_suffix = before_lines.len();
    let mut after_suffix = after_lines.len();
    while before_suffix > prefix
        && after_suffix > prefix
        && before_lines[before_suffix - 1] == after_lines[after_suffix - 1]
    {
        before_suffix -= 1;
        after_suffix -= 1;
    }

    (after_suffix.saturating_sub(prefix), before_suffix.saturating_sub(prefix))
}

/// Build a unified-style diff with line numbers using Myers-like LCS diff.
/// Each entry is a JSON object: {"op": " "/"+"/"-", "line": <1-indexed>, "text": "..."}
/// For context/removed lines, `line` is the line number in the original file.
/// For added lines, `line` is the line number in the new file.
/// Produces interleaved -/+ pairs for changed lines instead of a massive
/// block of removes followed by a massive block of adds.
fn diff_lines(before: &str, after: &str, context: usize) -> Vec<serde_json::Value> {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();

    // Build edit script using simple O(ND) LCS
    let edits = lcs_diff(&a, &b);

    // Convert edit script to hunk entries with context
    let mut raw: Vec<(char, usize, &str)> = Vec::new(); // (op, line_no, text)
    let mut ai = 0usize;
    let mut bi = 0usize;

    for edit in &edits {
        match edit {
            DiffOp::Equal => {
                raw.push((' ', ai + 1, a[ai]));
                ai += 1;
                bi += 1;
            }
            DiffOp::Remove => {
                raw.push(('-', ai + 1, a[ai]));
                ai += 1;
            }
            DiffOp::Add => {
                raw.push(('+', bi + 1, b[bi]));
                bi += 1;
            }
        }
    }

    // Now filter to only changed regions + context lines around them
    let change_indices: Vec<usize> = raw.iter().enumerate()
        .filter(|(_, (op, _, _))| *op != ' ')
        .map(|(i, _)| i)
        .collect();

    if change_indices.is_empty() {
        return Vec::new();
    }

    // Build set of indices to include (changed + context)
    let mut include = vec![false; raw.len()];
    for &ci in &change_indices {
        let start = ci.saturating_sub(context);
        let end = (ci + context + 1).min(raw.len());
        for i in start..end {
            include[i] = true;
        }
    }

    let mut entries = Vec::new();
    let mut in_hunk = false;

    for (i, (op, line_no, text)) in raw.iter().enumerate() {
        if include[i] {
            if !in_hunk && i > 0 && !entries.is_empty() {
                // Gap marker between hunks
                entries.push(serde_json::json!({"op": "...", "line": 0, "text": ""}));
            }
            in_hunk = true;
            let op_str = match op {
                ' ' => " ",
                '-' => "-",
                '+' => "+",
                _ => " ",
            };
            entries.push(serde_json::json!({"op": op_str, "line": line_no, "text": text}));
        } else {
            in_hunk = false;
        }
    }

    entries
}

#[derive(Debug, Clone, Copy)]
enum DiffOp {
    Equal,
    Remove,
    Add,
}

/// Simple O(ND) Myers diff producing an edit script.
fn lcs_diff<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<DiffOp> {
    let n = a.len();
    let m = b.len();

    if n == 0 {
        return vec![DiffOp::Add; m];
    }
    if m == 0 {
        return vec![DiffOp::Remove; n];
    }

    // For very large diffs, fall back to the simple prefix/suffix approach
    // to avoid O(N*M) memory/time
    if n + m > 10_000 {
        return simple_diff(a, b);
    }

    // LCS table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to build edit script
    let mut ops = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && a[i - 1] == b[j - 1] {
            ops.push(DiffOp::Equal);
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            ops.push(DiffOp::Add);
            j -= 1;
        } else {
            ops.push(DiffOp::Remove);
            i -= 1;
        }
    }
    ops.reverse();
    ops
}

/// Fallback for large files: prefix/suffix match, then bulk remove/add.
fn simple_diff<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<DiffOp> {
    let mut prefix = 0;
    while prefix < a.len() && prefix < b.len() && a[prefix] == b[prefix] {
        prefix += 1;
    }

    let mut a_suffix = a.len();
    let mut b_suffix = b.len();
    while a_suffix > prefix && b_suffix > prefix && a[a_suffix - 1] == b[b_suffix - 1] {
        a_suffix -= 1;
        b_suffix -= 1;
    }

    let mut ops = Vec::new();
    for _ in 0..prefix {
        ops.push(DiffOp::Equal);
    }
    for _ in prefix..a_suffix {
        ops.push(DiffOp::Remove);
    }
    for _ in prefix..b_suffix {
        ops.push(DiffOp::Add);
    }
    for _ in a_suffix..a.len() {
        ops.push(DiffOp::Equal);
    }
    ops
}

fn diffstat_metadata(path: &str, operation: &str, before: &str, after: &str) -> serde_json::Value {
    let (added_lines, removed_lines) = diff_line_counts(before, after);
    let diff = diff_lines(before, after, 3);
    serde_json::json!({
        "path": path,
        "operation": operation,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "hunks": diff
    })
}
use crate::types::{InputSchema, PropertySchema, ToolParam};

/// Edit granularity for SmartWrite operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditGranularity {
    /// Replace an entire function/method
    Function,
    /// Replace a specific symbol (struct, enum, etc.)
    Symbol,
    /// Replace lines in a range
    Lines,
    /// Insert at a specific location
    Insert,
    /// Delete a symbol or range
    Delete,
}

/// A structural edit operation.
#[derive(Debug, Clone)]
pub struct StructuralEdit {
    pub granularity: EditGranularity,
    pub target: String,        // Symbol name or line range
    pub new_content: String,   // Replacement content
    pub after: Option<String>, // For Insert: insert after this symbol
}

impl StructuralEdit {
    pub fn replace_function(name: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Function,
            target: name.to_string(),
            new_content: new_content.to_string(),
            after: None,
        }
    }

    pub fn replace_symbol(name: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Symbol,
            target: name.to_string(),
            new_content: new_content.to_string(),
            after: None,
        }
    }

    pub fn insert_after(after_symbol: &str, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Insert,
            target: String::new(),
            new_content: new_content.to_string(),
            after: Some(after_symbol.to_string()),
        }
    }

    pub fn delete_symbol(name: &str) -> Self {
        Self {
            granularity: EditGranularity::Delete,
            target: name.to_string(),
            new_content: String::new(),
            after: None,
        }
    }

    pub fn replace_lines(start: usize, end: usize, new_content: &str) -> Self {
        Self {
            granularity: EditGranularity::Lines,
            target: format!("{}:{}", start, end),
            new_content: new_content.to_string(),
            after: None,
        }
    }
}

/// Hook for coordination-aware writes. The host (Replay) implements this
/// to integrate with the coordination stream and section claims.
pub trait CoordinationHook: Send + Sync {
    /// Called before a write. Returns Err with a message if the write
    /// should be blocked (e.g. claim conflict with another agent).
    /// `agent_id` identifies the writing agent.
    /// `file_path` is the relative path being written.
    /// `start_line`/`end_line` are 1-indexed inclusive line range being modified.
    fn check_write(
        &self,
        agent_id: &str,
        file_path: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<(), String>;

    /// Called after a successful write. Emits a record to the coordination
    /// stream so other agents know what changed.
    fn notify_write(
        &self,
        agent_id: &str,
        file_path: &str,
        start_line: usize,
        end_line: usize,
        purpose: &str,
    );
}

/// SmartWrite tool for structure-aware code editing.
pub struct SmartWriteTool {
    project_root: PathBuf,
    analyzer: Arc<RwLock<LayerAnalyzer>>,
    dry_run: bool,
    read_tracker: super::ReadTracker,
    coordination: Option<Arc<dyn CoordinationHook>>,
    agent_id: String,
}

impl SmartWriteTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            dry_run: false,
            read_tracker: super::ReadTracker::new(),
            coordination: None,
            agent_id: "main".to_string(),
        }
    }

    /// Create with a shared read tracker (paired with SmartRead).
    pub fn with_tracker(project_root: impl Into<PathBuf>, tracker: super::ReadTracker) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            dry_run: false,
            read_tracker: tracker,
            coordination: None,
            agent_id: "main".to_string(),
        }
    }

    /// Set the coordination hook for multi-agent awareness.
    pub fn with_coordination(mut self, hook: Arc<dyn CoordinationHook>, agent_id: &str) -> Self {
        self.coordination = Some(hook);
        self.agent_id = agent_id.to_string();
        self
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }

    /// Apply a structural edit to a file.
    pub fn apply_edit(&self, path: &str, edit: &StructuralEdit) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path).ok_or_else(|| "Unsupported language".to_string())?;

        let new_content = self.apply_structural_edit(&content, lang, edit)?;

        if !self.dry_run {
            std::fs::write(&full_path, &new_content)
                .map_err(|e| format!("Failed to write file: {}", e))?;
            self.read_tracker.mark_written(&full_path);
        }

        Ok(new_content)
    }

    fn apply_structural_edit(
        &self,
        source: &str,
        lang: Lang,
        edit: &StructuralEdit,
    ) -> Result<String, String> {
        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(source, lang);

        match edit.granularity {
            EditGranularity::Function | EditGranularity::Symbol => {
                self.apply_symbol_replacement(source, &symbols, &edit.target, &edit.new_content)
            }
            EditGranularity::Insert => {
                let after = edit
                    .after
                    .as_ref()
                    .ok_or("Insert requires 'after' target")?;
                self.apply_insert(source, &symbols, after, &edit.new_content)
            }
            EditGranularity::Delete => self.apply_delete(source, &symbols, &edit.target),
            EditGranularity::Lines => {
                self.apply_line_replacement(source, &edit.target, &edit.new_content)
            }
        }
    }

    fn apply_symbol_replacement(
        &self,
        source: &str,
        symbols: &[Symbol],
        target_name: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == target_name)
            .ok_or_else(|| format!("Symbol '{}' not found", target_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines before the symbol
        for line in &lines[..symbol.start_line - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // New content
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_insert(
        &self,
        source: &str,
        symbols: &[Symbol],
        after_name: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == after_name)
            .ok_or_else(|| format!("Symbol '{}' not found", after_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines up to and including the symbol
        for line in &lines[..symbol.end_line] {
            result.push_str(line);
            result.push('\n');
        }

        // Blank line + new content
        result.push('\n');
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_delete(
        &self,
        source: &str,
        symbols: &[Symbol],
        target_name: &str,
    ) -> Result<String, String> {
        let symbol = symbols
            .iter()
            .find(|s| s.name == target_name)
            .ok_or_else(|| format!("Symbol '{}' not found", target_name))?;

        let lines: Vec<&str> = source.lines().collect();
        let mut result = String::new();

        // Lines before the symbol
        for line in &lines[..symbol.start_line - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // Skip the symbol's lines

        // Lines after the symbol
        for line in &lines[symbol.end_line..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    fn apply_line_replacement(
        &self,
        source: &str,
        range: &str,
        new_content: &str,
    ) -> Result<String, String> {
        let parts: Vec<&str> = range.split(':').collect();
        if parts.len() != 2 {
            return Err("Line range must be 'start:end'".to_string());
        }

        let start: usize = parts[0].parse().map_err(|_| "Invalid start line")?;
        let end: usize = parts[1].parse().map_err(|_| "Invalid end line")?;

        let lines: Vec<&str> = source.lines().collect();
        if start < 1 || end > lines.len() || start > end {
            return Err(format!("Invalid line range {}:{}", start, end));
        }

        let mut result = String::new();

        // Lines before
        for line in &lines[..start - 1] {
            result.push_str(line);
            result.push('\n');
        }

        // New content
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }

        // Lines after
        for line in &lines[end..] {
            result.push_str(line);
            result.push('\n');
        }

        Ok(result)
    }

    /// Tree-sitter powered edit: find a node by target expression, then replace/delete/insert_after.
    fn handle_node_edit(
        &self,
        path: &str,
        operation: &str,
        target: &str,
        line_hint: Option<usize>,
        content: &str,
        reason: Option<&str>,
    ) -> ToolResult {
        let full_path = self.resolve_path(path);
        let source = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        let lang = match Lang::from_path(&full_path) {
            Some(l) => l,
            None => {
                // Non-code file: fall back to text anchor search
                return ToolResult::error(format!(
                    "Cannot use '{}' on non-code file '{}'. Use 'replace_lines' with anchors instead.",
                    operation, path
                ));
            }
        };

        let mut parser = AstParser::new();
        let matches = parser.query_nodes(&source, lang, target);

        if matches.is_empty() {
            return ToolResult::error(format!("Target not found: '{}'", target));
        }

        // Pick the match closest to line_hint, or first match
        let matched = if let Some(hint) = line_hint {
            matches.iter().min_by_key(|m| {
                (m.start_line as isize - hint as isize).unsigned_abs()
            }).unwrap()
        } else {
            if matches.len() > 1 {
                let locations: Vec<String> = matches.iter().map(|m| {
                    format!("  line {}: {} {}", m.start_line, m.kind, m.name.as_deref().unwrap_or(""))
                }).collect();
                return ToolResult::error(format!(
                    "Target '{}' matched {} nodes. Provide 'line' to disambiguate:\n{}",
                    target, matches.len(), locations.join("\n")
                ));
            }
            &matches[0]
        };

        // Section-level write check: only block if this section overlaps a dirty one
        {
            use super::read_tracker::LineRange;
            let section = LineRange::new(matched.start_line, matched.end_line);
            if let Err(e) = self.read_tracker.check_write_section(&full_path, Some(&section)) {
                return ToolResult::error(e);
            }
        }

        // Coordination check: are any other agents claiming this region?
        if let Some(ref hook) = self.coordination {
            if let Err(e) = hook.check_write(&self.agent_id, path, matched.start_line, matched.end_line) {
                return ToolResult::error(e);
            }
        }

        // Apply the edit
        let before = &source;
        let result = match operation {
            "replace" => {
                let mut result = String::new();
                result.push_str(&source[..matched.start_byte]);
                result.push_str(content);
                if !content.ends_with('\n') && matched.end_byte < source.len() {
                    result.push('\n');
                }
                result.push_str(&source[matched.end_byte..]);
                result
            }
            "delete" => {
                let mut result = String::new();
                result.push_str(&source[..matched.start_byte]);
                // Skip trailing newline after deleted node
                let skip_to = if matched.end_byte < source.len()
                    && source.as_bytes()[matched.end_byte] == b'\n'
                {
                    matched.end_byte + 1
                } else {
                    matched.end_byte
                };
                result.push_str(&source[skip_to..]);
                result
            }
            "insert_after" => {
                let mut result = String::new();
                result.push_str(&source[..matched.end_byte]);
                result.push_str("\n\n");
                result.push_str(content);
                if !content.ends_with('\n') {
                    result.push('\n');
                }
                result.push_str(&source[matched.end_byte..]);
                result
            }
            _ => return ToolResult::error(format!("Unknown node operation: {}", operation)),
        };

        let metadata = diffstat_metadata(path, operation, before, &result);

        if self.dry_run {
            return ToolResult::success_with_metadata(
                format!("(dry run) Would {} '{}' at line {} in {}", operation, target, matched.start_line, path),
                metadata,
            );
        }

        match std::fs::write(&full_path, &result) {
            Ok(()) => {
                // Section-level dirty tracking: mark only the affected range
                let new_line_count = result.lines().count().saturating_sub(
                    before.lines().count().saturating_sub(
                        (matched.end_line - matched.start_line + 1)
                    )
                );
                use super::read_tracker::LineRange;
                self.read_tracker.mark_section_written(
                    &full_path,
                    LineRange::new(matched.start_line, matched.end_line),
                    content.lines().count().max(1),
                );
                // Auto-summary: human-readable
                let kind_word = match matched.kind.as_str() {
                    "function_item" | "function_declaration" | "function_definition"
                    | "method_definition" | "method_declaration" => "fn",
                    "impl_item" => "impl",
                    "struct_item" | "type_declaration" => "struct",
                    "enum_item" => "enum",
                    "trait_item" => "trait",
                    "mod_item" | "module" => "mod",
                    "use_declaration" | "import_statement" | "import_from_statement"
                    | "import_declaration" => "import",
                    "class_declaration" | "class_definition" => "class",
                    "interface_declaration" => "interface",
                    "match_expression" | "match_statement" => "match",
                    "if_expression" | "if_statement" => "if block",
                    "for_expression" | "for_statement" => "for loop",
                    "while_expression" | "while_statement" => "while loop",
                    other => other,
                };
                let name_part = matched.name.as_deref().unwrap_or("");
                let target_desc = if name_part.is_empty() {
                    format!("{kind_word} at lines {}-{}", matched.start_line, matched.end_line)
                } else {
                    format!("{kind_word} `{name_part}`")
                };
                let verb = match operation {
                    "replace" => "replaced",
                    "delete" => "deleted",
                    "insert_after" => "inserted after",
                    _ => operation,
                };
                let auto_summary = format!("{verb} {target_desc}");

                // Notify coordination stream
                if let Some(ref hook) = self.coordination {
                    hook.notify_write(
                        &self.agent_id, path,
                        matched.start_line, matched.end_line,
                        &auto_summary,
                    );
                }

                self.success_with_summary(path, &auto_summary, reason, metadata)
            }
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }

    /// Run a lint/compile check on a file. Returns Ok(()) if it passes,
    /// Err(diagnostic) if it fails. The caller is responsible for reverting.
    fn lint_check(&self, path: &str) -> Result<(), String> {
        let full_path = self.resolve_path(path);
        let ext = full_path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let (cmd, args): (&str, Vec<&str>) = match ext {
            "rs" => {
                // Find the nearest Cargo.toml
                let mut dir = full_path.parent();
                while let Some(d) = dir {
                    if d.join("Cargo.toml").exists() {
                        let output = std::process::Command::new("cargo")
                            .arg("check")
                            .arg("--message-format=short")
                            .current_dir(d)
                            .output();
                        return match output {
                            Ok(o) if o.status.success() => Ok(()),
                            Ok(o) => {
                                let stderr = String::from_utf8_lossy(&o.stderr);
                                let errors: Vec<&str> = stderr.lines()
                                    .filter(|l| l.contains("error"))
                                    .take(5)
                                    .collect();
                                Err(format!("Lint failed:\n{}", errors.join("\n")))
                            }
                            Err(e) => Err(format!("Failed to run cargo check: {e}")),
                        };
                    }
                    dir = d.parent();
                }
                return Ok(()); // No Cargo.toml found, skip
            }
            "py" => ("python3", vec!["-m", "py_compile"]),
            "js" | "jsx" | "ts" | "tsx" => {
                // Try node --check for syntax
                let output = std::process::Command::new("node")
                    .arg("--check")
                    .arg(&full_path)
                    .output();
                return match output {
                    Ok(o) if o.status.success() => Ok(()),
                    Ok(o) => Err(format!("Lint failed: {}", String::from_utf8_lossy(&o.stderr))),
                    Err(_) => Ok(()), // node not available, skip
                };
            }
            _ => return Ok(()), // Unknown language, skip
        };

        let mut command = std::process::Command::new(cmd);
        for arg in &args {
            command.arg(arg);
        }
        command.arg(&full_path);

        match command.output() {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => Err(format!("Lint failed: {}", String::from_utf8_lossy(&o.stderr))),
            Err(_) => Ok(()), // Tool not available, skip
        }
    }

    /// If lint was requested and the write succeeded, run lint check.
    /// Reverts file on lint failure.
    fn maybe_lint(&self, path: &str, lint: bool, original: Option<&str>, result: ToolResult) -> ToolResult {
        if !lint || result.is_error() {
            return result;
        }

        if let Err(diagnostic) = self.lint_check(path) {
            // Revert
            let full_path = self.resolve_path(path);
            if let Some(orig) = original {
                let _ = std::fs::write(&full_path, orig);
            } else {
                // New file — remove it
                let _ = std::fs::remove_file(&full_path);
            }
            return ToolResult::error(format!(
                "Write reverted — lint check failed:\n{diagnostic}"
            ));
        }

        result
    }

    fn success_with_summary(
        &self,
        path: &str,
        auto_summary: &str,
        reason: Option<&str>,
        mut metadata: serde_json::Value,
    ) -> ToolResult {
        // Embed summary into metadata for display layer
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert("summary".to_string(), serde_json::Value::String(auto_summary.to_string()));
            if let Some(r) = reason {
                obj.insert("reason".to_string(), serde_json::Value::String(r.to_string()));
            }
        }

        let display = match reason {
            Some(r) => format!("{}\n  ↳ {}", auto_summary, r),
            None => auto_summary.to_string(),
        };

        ToolResult::success_with_metadata(display, metadata)
    }

    /// Resolve an anchor to a 1-indexed line number.
    ///
    /// `text` — substring to match against file lines (authoritative).
    /// `line_hint` — expected line number, used to disambiguate when `text`
    ///   appears multiple times. Picks the closest match to the hint.
    ///   If `text` is empty/None, `line_hint` is used directly (for blank-line regions).
    fn resolve_anchor(
        text: Option<&str>,
        line_hint: Option<usize>,
        lines: &[&str],
        search_from: usize,
    ) -> Result<usize, String> {
        let text = text.map(|t| t.trim()).filter(|t| !t.is_empty());

        match (text, line_hint) {
            (Some(needle), Some(hint)) => {
                // Find ALL matching lines, pick the one closest to hint
                let matches: Vec<usize> = lines[search_from..]
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.contains(needle))
                    .map(|(i, _)| search_from + i + 1) // 1-indexed
                    .collect();

                if matches.is_empty() {
                    return Err(format!("Anchor text not found: '{}'", needle));
                }

                // Closest to hint
                Ok(*matches.iter().min_by_key(|&&m| {
                    (m as isize - hint as isize).unsigned_abs()
                }).unwrap())
            }
            (Some(needle), None) => {
                // Text only — first match from search_from
                if let Some(idx) = lines[search_from..].iter().position(|l| l.contains(needle)) {
                    Ok(search_from + idx + 1)
                } else {
                    Err(format!("Anchor text not found: '{}'", needle))
                }
            }
            (None, Some(hint)) => {
                // Line number only (blank-line regions)
                if hint >= 1 && hint <= lines.len() {
                    Ok(hint)
                } else {
                    Err(format!("Line {} out of range (file has {} lines)", hint, lines.len()))
                }
            }
            (None, None) => Err("Anchor requires at least 'anchor' text or 'line' number".to_string()),
        }
    }

    /// Apply multiple edits (replace, delete, move) atomically against the original file state.
    /// Targets can be tree-sitter expressions (via 'target') or text anchors (via 'anchor'+'line').
    /// All positions are resolved against the original content, then applied bottom-up.
    fn handle_compound_edit(&self, path: &str, edits: &[serde_json::Value], reason: Option<&str>) -> ToolResult {
        let full_path = self.resolve_path(path);
        let source = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };
        let file_lines: Vec<&str> = source.lines().collect();
        let lang = Lang::from_path(&full_path);

        // Phase 1: resolve all edits to concrete line ranges against original content
        struct ResolvedEdit {
            action: String,
            start: usize,  // 1-indexed
            end: usize,    // 1-indexed, inclusive
            content: String,
            dest: usize,   // 1-indexed, for move — insert before this line
        }

        // Parse tree once for all edits if we have a language
        let mut parser = AstParser::new();
        let node_matches_cache: Option<Vec<(String, Vec<super::ast::NodeMatch>)>> = None;
        let _ = node_matches_cache; // suppress warning

        // Helper: resolve a target (tree-sitter) or anchor (text+line) to start/end lines
        let mut resolve_target = |target: Option<&str>, anchor: Option<&str>, line: Option<usize>,
                              end_anchor: Option<&str>, end_line: Option<usize>, idx: usize|
                              -> Result<(usize, usize), String> {
            // Try tree-sitter target first
            if let (Some(t), Some(l)) = (target, lang) {
                if !t.is_empty() {
                    let matches = parser.query_nodes(&source, l, t);
                    if matches.is_empty() {
                        return Err(format!("Edit [{}]: target not found: '{}'", idx, t));
                    }
                    let m = if let Some(hint) = line {
                        matches.iter().min_by_key(|m| {
                            (m.start_line as isize - hint as isize).unsigned_abs()
                        }).unwrap()
                    } else if matches.len() > 1 {
                        let locs: Vec<String> = matches.iter()
                            .map(|m| format!("  line {}: {}", m.start_line, m.kind))
                            .collect();
                        return Err(format!(
                            "Edit [{}]: target '{}' matched {} nodes, provide 'line' to disambiguate:\n{}",
                            idx, t, matches.len(), locs.join("\n")
                        ));
                    } else {
                        &matches[0]
                    };
                    return Ok((m.start_line, m.end_line));
                }
            }

            // Fall back to text anchor + line
            if anchor.is_none() && line.is_none() {
                return Err(format!("Edit [{}]: needs 'target' (tree-sitter) or 'anchor'+'line'", idx));
            }
            let start = Self::resolve_anchor(anchor, line, &file_lines, 0)
                .map_err(|e| format!("Edit [{}] anchor: {}", idx, e))?;
            let end = if end_anchor.is_some() || end_line.is_some() {
                Self::resolve_anchor(end_anchor, end_line, &file_lines, start.saturating_sub(1))
                    .map_err(|e| format!("Edit [{}] end_anchor: {}", idx, e))?
            } else {
                start
            };
            Ok((start, end))
        };

        let mut resolved = Vec::new();
        for (i, e) in edits.iter().enumerate() {
            let action = e.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let target_expr = e.get("target").and_then(|v| v.as_str());
            let anchor_text = e.get("anchor").and_then(|v| v.as_str());
            let line_hint = e.get("line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let end_text = e.get("end_anchor").and_then(|v| v.as_str());
            let end_line_hint = e.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let content = e.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let dest_target = e.get("dest").and_then(|v| v.as_str());
            let dest_line_hint = e.get("dest_line").and_then(|v| v.as_u64()).map(|n| n as usize);

            let (start, end) = match resolve_target(target_expr, anchor_text, line_hint, end_text, end_line_hint, i) {
                Ok(r) => r,
                Err(e) => return ToolResult::error(e),
            };

            if end < start {
                return ToolResult::error(format!(
                    "Edit [{}]: end ({}) is before start ({})", i, end, start
                ));
            }

            let dest = match action {
                "move" => {
                    let (d, _) = match resolve_target(dest_target, None, dest_line_hint, None, None, i) {
                        Ok(r) => r,
                        Err(_) => return ToolResult::error(format!(
                            "Edit [{}]: 'move' requires 'dest' (target expression) or 'dest_line'", i
                        )),
                    };
                    d
                }
                "replace" | "delete" => 0,
                _ => return ToolResult::error(format!(
                    "Edit [{}]: unknown action '{}' (expected replace, delete, or move)", i, action
                )),
            };

            resolved.push(ResolvedEdit {
                action: action.to_string(),
                start,
                end,
                content: content.to_string(),
                dest,
            });
        }

        // Phase 2: assemble result — mark deletions, queue insertions, build output
        let original_lines: Vec<&str> = source.lines().collect();
        let mut deleted: Vec<bool> = vec![false; original_lines.len()];

        for edit in &resolved {
            match edit.action.as_str() {
                "delete" | "replace" | "move" => {
                    for i in (edit.start - 1)..edit.end {
                        deleted[i] = true;
                    }
                }
                _ => {}
            }
        }

        // Build insertions map: before which original line index do we insert what?
        let mut insert_before: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();

        for edit in &resolved {
            match edit.action.as_str() {
                "replace" => {
                    insert_before
                        .entry(edit.start - 1)
                        .or_default()
                        .push(edit.content.clone());
                }
                "move" => {
                    let moved: String = (edit.start - 1..edit.end)
                        .map(|i| original_lines[i])
                        .collect::<Vec<_>>()
                        .join("\n");
                    insert_before
                        .entry(edit.dest - 1)
                        .or_default()
                        .push(moved);
                }
                _ => {}
            }
        }

        // Assemble output
        let mut result = String::new();
        for (i, line) in original_lines.iter().enumerate() {
            // Insert anything queued before this line
            if let Some(chunks) = insert_before.get(&i) {
                for chunk in chunks {
                    result.push_str(chunk);
                    if !chunk.ends_with('\n') {
                        result.push('\n');
                    }
                }
            }
            // Keep this line if not deleted
            if !deleted[i] {
                result.push_str(line);
                result.push('\n');
            }
        }
        // Handle insertions after the last line
        if let Some(chunks) = insert_before.get(&original_lines.len()) {
            for chunk in chunks {
                result.push_str(chunk);
                if !chunk.ends_with('\n') {
                    result.push('\n');
                }
            }
        }

        let metadata = diffstat_metadata(path, "edit", &source, &result);

        if self.dry_run {
            return ToolResult::success_with_metadata(
                format!("(dry run) Would apply {} edits to {}", edits.len(), path),
                metadata,
            );
        }

        match std::fs::write(&full_path, &result) {
            Ok(()) => {
                // Mark each resolved edit's section as dirty
                use super::read_tracker::LineRange;
                for edit in &resolved {
                    let new_lines = if edit.action == "delete" {
                        0
                    } else {
                        edit.content.lines().count().max(1)
                    };
                    self.read_tracker.mark_section_written(
                        &full_path,
                        LineRange::new(edit.start, edit.end),
                        new_lines,
                    );
                }
                // Auto-summary from resolved edits
                let parts: Vec<String> = resolved.iter().map(|e| {
                    if e.start == e.end {
                        format!("{} line {}", e.action, e.start)
                    } else {
                        format!("{} lines {}-{}", e.action, e.start, e.end)
                    }
                }).collect();
                let auto_summary = if parts.len() == 1 {
                    parts[0].clone()
                } else {
                    parts.join("; ")
                };

                // Notify coordination stream for each sub-edit
                if let Some(ref hook) = self.coordination {
                    for edit in &resolved {
                        hook.notify_write(
                            &self.agent_id, path,
                            edit.start, edit.end,
                            &format!("{} lines {}-{}", edit.action, edit.start, edit.end),
                        );
                    }
                }

                self.success_with_summary(path, &auto_summary, reason, metadata)
            }
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }

    /// Write arbitrary content to a file, creating parent directories as needed.
    /// Note: path restrictions are the host's responsibility (e.g. SandboxedWriteTool).
    fn handle_write(&self, path: &str, content: &str, reason: Option<&str>) -> ToolResult {
        let full_path = self.resolve_path(path);

        let previous = std::fs::read_to_string(&full_path).unwrap_or_default();
        let metadata = diffstat_metadata(path, "write", &previous, content);

        if self.dry_run {
            return ToolResult::success_with_metadata(
                format!("(dry run) Would write {} bytes to {}", content.len(), path),
                metadata,
            );
        }

        // Create parent directories
        if let Some(parent) = full_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("Failed to create directories: {}", e));
            }
        }

        match std::fs::write(&full_path, content) {
            Ok(()) => {
                self.read_tracker.mark_written(&full_path);
                let lines = content.lines().count();
                let auto_summary = format!("{lines} lines");

                if let Some(ref hook) = self.coordination {
                    hook.notify_write(&self.agent_id, path, 1, lines.max(1), &auto_summary);
                }

                self.success_with_summary(path, &auto_summary, reason, metadata)
            }
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }

    /// Handle a batch of writes.
    async fn handle_batch(&self, writes: &[serde_json::Value]) -> ToolResult {
        let mut results = Vec::new();

        for (i, w) in writes.iter().enumerate() {
            let path = w.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = w.get("content").and_then(|v| v.as_str()).unwrap_or("");

            if path.is_empty() {
                results.push(format!("[{}] error: path is required", i));
                continue;
            }

            let result = self.handle_write(path, content, None);
            if result.is_error() {
                results.push(format!("[{}] error writing {}: {}", i, path, result.to_content_string()));
            } else {
                results.push(format!("[{}] {}", i, result.to_content_string()));
            }
        }

        ToolResult::success(results.join("\n"))
    }

    /// Preview what an edit would produce without writing.
    pub fn preview_edit(&self, path: &str, edit: &StructuralEdit) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path).ok_or_else(|| "Unsupported language".to_string())?;

        self.apply_structural_edit(&content, lang, edit)
    }
}

#[async_trait]
impl Tool for SmartWriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "write",
            InputSchema::object()
                .required_string("path", "File path to write or edit")
                .optional_string("content", "File content (for write/overwrite) or replacement content (for replace/insert_after)")
                .optional_string("operation", "Edit operation: 'write' (default, creates new file — fails if exists), 'overwrite' (replace entire file), 'replace' (replace a matched node/region), 'delete' (remove a matched node/region), 'insert_after' (insert after a matched node), 'edit' (compound atomic edits), 'replace_lines' (anchor-based line replacement for non-code files)")
                .optional_string("target", "What to find — a natural expression like 'fn handle_event', 'impl AppState', 'struct Config', 'match KeyCode::Enter', 'use std::collections', or just 'handle_event'. Used by replace, delete, insert_after.")
                .optional_string("anchor", "For replace_lines: text to match near the edit region start.")
                .property("line", PropertySchema::integer().with_description("Line number hint — disambiguates when target/anchor text appears multiple times. Also used alone for blank-line regions."), false)
                .optional_string("end_anchor", "For replace_lines: text marking the end of the region.")
                .property("end_line", PropertySchema::integer().with_description("Line number hint for end_anchor."), false)
                .property("edits", PropertySchema::array(PropertySchema::object()).with_description(
                    "For 'edit' operation: array of sub-edits applied atomically. Each: {action: replace|delete|move, target: 'fn foo' (or anchor+line for non-code), content: '...' (for replace), dest: 'fn bar' (for move — insert before this target), line/end_line: number hints}"
                ), false)
                .optional_string("reason", "Optional: why you're making this change. Shown alongside the auto-generated edit summary.")
                .property("lint", PropertySchema::boolean().with_description("If true, the write only succeeds if the file passes lint/compile check after the edit. Atomic: file is reverted on failure."), false),
        )
        .with_description(
            "Write or edit files with tree-sitter-powered targeting. 'write' creates new files. 'overwrite' replaces entire files. 'replace'/'delete'/'insert_after' target AST nodes by natural expressions (e.g. 'fn foo', 'impl Bar', 'struct Cfg'). 'edit' does compound atomic edits (replace, delete, move in one call). 'replace_lines' uses text anchors for non-code files.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        // Batch mode: array of writes
        if let Some(writes) = input.get("writes").and_then(|v| v.as_array()) {
            return self.handle_batch(writes).await;
        }

        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        // Enforce read-before-write
        let full_path = self.resolve_path(path);
        if let Err(e) = self.read_tracker.check_write(&full_path) {
            return ToolResult::error(e);
        }

        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("write");

        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let reason = input.get("reason").and_then(|v| v.as_str());
        let lint = input.get("lint").and_then(|v| v.as_bool()).unwrap_or(false);

        // Save original content for lint rollback
        let original = if lint {
            std::fs::read_to_string(&full_path).ok()
        } else {
            None
        };

        // Plain write: create new file only (fails if exists)
        if operation == "write" {
            let full = self.resolve_path(path);
            if full.exists() {
                return ToolResult::error(format!(
                    "File '{}' already exists. Use operation 'overwrite' to replace it, \
                     or a structural operation (replace, delete, insert_after, edit) for surgical edits.",
                    path
                ));
            }
            let result = self.handle_write(path, content, reason);
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        // Explicit full-file overwrite
        if operation == "overwrite" {
            let result = self.handle_write(path, content, reason);
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        // Compound edit: multiple sub-edits applied atomically
        if operation == "edit" {
            let edits = match input.get("edits").and_then(|v| v.as_array()) {
                Some(e) if !e.is_empty() => e,
                _ => return ToolResult::error("'edit' operation requires a non-empty 'edits' array."),
            };
            let result = self.handle_compound_edit(path, edits, reason);
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        let target = input.get("target").and_then(|v| v.as_str()).unwrap_or("");
        let line_hint = input.get("line").and_then(|v| v.as_u64()).map(|n| n as usize);

        // Tree-sitter powered: replace, delete, insert_after
        if matches!(operation, "replace" | "delete" | "insert_after") {
            if target.is_empty() {
                return ToolResult::error(format!(
                    "'{}' requires 'target' — e.g. 'fn handle_event', 'impl AppState', 'struct Config'",
                    operation
                ));
            }
            let result = self.handle_node_edit(path, operation, target, line_hint, content, reason);
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        // Anchor-based line replacement (for non-code files or raw line regions)
        if operation == "replace_lines" {
            let anchor_text = input.get("anchor").and_then(|v| v.as_str());
            let anchor_line = input.get("line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let end_anchor_text = input.get("end_anchor").and_then(|v| v.as_str());
            let end_anchor_line = input.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

            if anchor_text.is_none() && anchor_line.is_none() {
                return ToolResult::error(
                    "replace_lines requires 'anchor' (text) and/or 'line' (number)."
                );
            }

            let full = self.resolve_path(path);
            let source = match std::fs::read_to_string(&full) {
                Ok(s) => s,
                Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
            };
            let file_lines: Vec<&str> = source.lines().collect();

            let start = match Self::resolve_anchor(anchor_text, anchor_line, &file_lines, 0) {
                Ok(n) => n,
                Err(e) => return ToolResult::error(format!("Anchor: {}", e)),
            };

            let end = if end_anchor_text.is_some() || end_anchor_line.is_some() {
                match Self::resolve_anchor(end_anchor_text, end_anchor_line, &file_lines, start.saturating_sub(1)) {
                    Ok(n) => n,
                    Err(e) => return ToolResult::error(format!("End anchor: {}", e)),
                }
            } else {
                file_lines.len()
            };

            // Coordination check
            if let Some(ref hook) = self.coordination {
                if let Err(e) = hook.check_write(&self.agent_id, path, start, end) {
                    return ToolResult::error(e);
                }
            }

            let edit = StructuralEdit::replace_lines(start, end, content);
            let before = std::fs::read_to_string(&full_path).unwrap_or_default();
            let result = match self.apply_edit(path, &edit) {
                Ok(result) => {
                    let metadata = diffstat_metadata(path, operation, &before, &result);
                    let auto_summary = if start == end {
                        format!("replaced line {start}")
                    } else {
                        format!("replaced lines {start}-{end}")
                    };

                    if let Some(ref hook) = self.coordination {
                        hook.notify_write(&self.agent_id, path, start, end, &auto_summary);
                    }

                    self.success_with_summary(path, &auto_summary, reason, metadata)
                }
                Err(e) => ToolResult::error(e),
            };
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        // Legacy aliases (backwards compat)
        if matches!(operation, "replace_function" | "replace_symbol") {
            let result = self.handle_node_edit(path, "replace", target, line_hint, content, reason);
            return self.maybe_lint(path, lint, original.as_deref(), result);
        }

        ToolResult::error(format!("Unknown operation: '{}'. Valid: write, overwrite, replace, delete, insert_after, edit, replace_lines", operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_replace_function() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn hello() {
    println!("Hello");
}

fn world() {
    println!("World");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit =
            StructuralEdit::replace_function("hello", "fn hello() {\n    println!(\"Hi!\");\n}");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("Hi!"));
        assert!(result.contains("fn world()"));
        assert!(!result.contains("Hello"));
    }

    #[test]
    fn test_insert_after() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn first() {
    println!("First");
}

fn third() {
    println!("Third");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit =
            StructuralEdit::insert_after("first", "fn second() {\n    println!(\"Second\");\n}");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("fn first()"));
        assert!(result.contains("fn second()"));
        assert!(result.contains("fn third()"));

        // Verify order
        let first_pos = result.find("fn first()").unwrap();
        let second_pos = result.find("fn second()").unwrap();
        let third_pos = result.find("fn third()").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_delete_symbol() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn keep_me() {
    println!("Keep");
}

fn delete_me() {
    println!("Delete");
}

fn also_keep() {
    println!("Also keep");
}
"#;

        fs::write(dir.path().join("test.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());
        let edit = StructuralEdit::delete_symbol("delete_me");

        let result = tool.apply_edit("test.rs", &edit).unwrap();

        assert!(result.contains("fn keep_me()"));
        assert!(result.contains("fn also_keep()"));
        assert!(!result.contains("fn delete_me()"));
        assert!(!result.contains("Delete"));
    }

    #[tokio::test]
    async fn test_smart_write_tool() {
        let dir = TempDir::new().unwrap();

        let source = r#"pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
"#;

        fs::write(dir.path().join("calc.rs"), source).unwrap();

        let tool = SmartWriteTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert(
            "operation".to_string(),
            serde_json::json!("replace_function"),
        );
        input.insert("target".to_string(), serde_json::json!("add"));
        input.insert(
            "content".to_string(),
            serde_json::json!(
                "pub fn add(a: i32, b: i32) -> i32 {\n    // Optimized\n    a + b\n}"
            ),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(dir.path().join("calc.rs")).unwrap();
        assert!(content.contains("// Optimized"));
        assert!(content.contains("fn subtract"));
    }
}
