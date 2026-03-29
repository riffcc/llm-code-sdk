//! SmartRead tool - reads code using the layercake approach.
//!
//! Instead of dumping raw code, presents structured representations
//! at the appropriate layer for token efficiency.
//!
//! Supports tree-style batch reads for efficient multi-file context gathering:
//! ```json
//! {
//!   "reads": [
//!     {"path": "src/main.rs", "layer": "ast"},
//!     {"path": "src/lib.rs", "layer": "call_graph"},
//!     {"path": "src/utils.rs", "symbol": "helper", "layer": "raw"}
//!   ]
//! }
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::ast::{AstParser, Lang};
use super::layers::{CodeLayer, LayerAnalyzer, LayerView};
use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// Build read metadata for the host UI.
fn read_metadata(path: &str, layer: &str, symbol: Option<&str>, line_count: usize) -> serde_json::Value {
    let mut meta = serde_json::json!({
        "path": path,
        "layer": layer,
        "lines": line_count,
    });

    let read_kind = if layer == "codebase" {
        "codebase"
    } else if layer.starts_with("directory:") {
        "directory"
    } else if symbol.is_some() {
        "symbol"
    } else if layer == "raw" || layer == "Ast" || layer == "CallGraph" || layer == "Cfg" || layer == "Dfg" || layer == "Pdg" || layer == "TheoryGraph" {
        "file"
    } else {
        "read"
    };
    meta["kind"] = serde_json::json!(read_kind);

    if let Some(sym) = symbol {
        meta["symbol"] = serde_json::json!(sym);
    }
    meta
}

/// Build batch read metadata.
fn batch_read_metadata(items: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({
        "batch": true,
        "kind": "batch",
        "reads": items,
    })
}

/// A single read request in a batch.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub path: String,
    pub layer: CodeLayer,
    pub symbol: Option<String>, // Optional: read only this symbol at raw level
}

impl ReadRequest {
    pub fn new(path: &str, layer: CodeLayer) -> Self {
        Self {
            path: path.to_string(),
            layer,
            symbol: None,
        }
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryLayer {
    Summary,
    Structure,
    Hotspots,
    Semantic,
    Scalpel,
}

impl DirectoryLayer {
    fn label(&self) -> &'static str {
        match self {
            DirectoryLayer::Summary => "summary",
            DirectoryLayer::Structure => "structure",
            DirectoryLayer::Hotspots => "hotspots",
            DirectoryLayer::Semantic => "semantic",
            DirectoryLayer::Scalpel => "scalpel",
        }
    }
}

#[derive(Debug, Clone)]
struct DirectoryReadOptions {
    recursive: bool,
    max_depth: Option<usize>,
    include: Vec<String>,
    exclude: Vec<String>,
    layer: DirectoryLayer,
}

impl DirectoryReadOptions {
    fn summary(recursive: bool) -> Self {
        Self {
            recursive,
            max_depth: None,
            include: Vec::new(),
            exclude: Vec::new(),
            layer: DirectoryLayer::Summary,
        }
    }

    fn allows_path(&self, relative: &str) -> bool {
        let normalized = relative.replace('\\', "/");

        if !self.include.is_empty()
            && !self.include.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(&normalized))
                    .unwrap_or(false)
            })
        {
            return false;
        }

        if self.exclude.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .map(|p| p.matches(&normalized))
                .unwrap_or(false)
        }) {
            return false;
        }

        true
    }
}

#[derive(Debug, Clone)]
struct DirectoryEntryInfo {
    relative: String,
    depth: usize,
    is_dir: bool,
    code: bool,
}

impl DirectoryEntryInfo {
    fn file_name(&self) -> &str {
        self.relative.rsplit('/').next().unwrap_or(&self.relative)
    }
}

/// SmartRead tool for token-efficient code reading.
pub struct SmartReadTool {
    project_root: PathBuf,
    analyzer: Arc<RwLock<LayerAnalyzer>>,
    read_tracker: super::ReadTracker,
}

impl SmartReadTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            read_tracker: super::ReadTracker::new(),
        }
    }

    /// Create with a shared read tracker (for pairing with SmartWrite).
    pub fn with_tracker(project_root: impl Into<PathBuf>, tracker: super::ReadTracker) -> Self {
        Self {
            project_root: project_root.into(),
            analyzer: Arc::new(RwLock::new(LayerAnalyzer::new())),
            read_tracker: tracker,
        }
    }

    /// Get a reference to the read tracker for sharing with SmartWrite.
    pub fn tracker(&self) -> &super::ReadTracker {
        &self.read_tracker
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }

    fn validate_read_path(&self, full_path: &Path, original_path: &str) -> Result<(), String> {
        if Path::new(original_path).is_absolute() {
            return Ok(());
        }

        if let Ok(canonical) = full_path.canonicalize() {
            if !canonical.starts_with(&self.project_root) {
                return Err("Relative paths must stay within project root".to_string());
            }
        }

        Ok(())
    }

    /// Search git history for commits matching a query, related to a path.
    /// Returns detailed info for matching commits.
    fn git_search(&self, path: &str, query: &str, limit: usize) -> String {
        let full_path = self.resolve_path(path);

        let repo_root = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(&self.project_root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        let repo_root = match repo_root {
            Some(r) => PathBuf::from(r),
            None => return String::new(),
        };

        let relative_path = full_path
            .strip_prefix(&repo_root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        // Search for commits matching query that touch this path
        let output = Command::new("git")
            .args([
                "log",
                "--all",
                "-i", // case insensitive
                &format!("--grep={}", query),
                &format!("-{}", limit * 2), // get extra, filter later
                "--format=%H %s",
                "--",
                &relative_path,
            ])
            .current_dir(&repo_root)
            .output();

        let mut commits: Vec<(String, String)> = match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|line| {
                    let mut parts = line.splitn(2, ' ');
                    let hash = parts.next()?.to_string();
                    let subject = parts.next().unwrap_or("").to_string();
                    Some((hash, subject))
                })
                .collect(),
            _ => vec![],
        };

        // Also search by path only if query didn't find enough
        if commits.len() < limit {
            let path_output = Command::new("git")
                .args([
                    "log",
                    "--all",
                    &format!("-{}", limit),
                    "--format=%H %s",
                    "--",
                    &relative_path,
                ])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = path_output {
                if o.status.success() {
                    let existing: std::collections::HashSet<_> =
                        commits.iter().map(|(h, _)| h.clone()).collect();
                    for line in String::from_utf8_lossy(&o.stdout).lines() {
                        let mut parts = line.splitn(2, ' ');
                        if let Some(hash) = parts.next() {
                            if !existing.contains(hash) {
                                let subject = parts.next().unwrap_or("").to_string();
                                // Only include if subject contains query terms
                                if query
                                    .split_whitespace()
                                    .any(|q| subject.to_lowercase().contains(&q.to_lowercase()))
                                {
                                    commits.push((hash.to_string(), subject));
                                }
                            }
                        }
                    }
                }
            }
        }

        if commits.is_empty() {
            return format!("No commits found matching '{}' for {}\n", query, path);
        }

        commits.truncate(limit);

        let mut result = format!("## Git History: '{}' ({})\n\n", query, path);

        for (hash, subject) in &commits {
            result.push_str(&format!("### {} {}\n", &hash[..7.min(hash.len())], subject));

            // Get commit body
            let detail = Command::new("git")
                .args(["show", "--no-patch", "--format=%b", hash])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = detail {
                if o.status.success() {
                    let body = String::from_utf8_lossy(&o.stdout);
                    let body_lines: Vec<&str> = body
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .take(3)
                        .collect();
                    for line in body_lines {
                        result.push_str(&format!("{}\n", line));
                    }
                }
            }

            // Get diff stats
            let stats = Command::new("git")
                .args(["show", "--numstat", "--format=", hash, "--", &relative_path])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = stats {
                if o.status.success() {
                    let stat_str = String::from_utf8_lossy(&o.stdout);
                    for line in stat_str.lines().take(1) {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            result.push_str(&format!("Changes: +{} -{}\n", parts[0], parts[1]));
                        }
                    }
                }
            }

            result.push('\n');
        }

        result
    }

    /// Get brief git history for a file (one-liners).
    fn git_history_brief(&self, path: &str, limit: usize) -> String {
        let full_path = self.resolve_path(path);

        let repo_root = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(&self.project_root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        let repo_root = match repo_root {
            Some(r) => PathBuf::from(r),
            None => return String::new(),
        };

        let relative_path = full_path
            .strip_prefix(&repo_root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        let output = Command::new("git")
            .args([
                "log",
                "--oneline",
                "--follow",
                &format!("-{}", limit),
                "--format=%h %s",
                "--",
                &relative_path,
            ])
            .current_dir(&repo_root)
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let commits = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if commits.is_empty() {
                    String::new()
                } else {
                    format!(
                        "History: {}\n\n",
                        commits.lines().collect::<Vec<_>>().join(" | ")
                    )
                }
            }
            _ => String::new(),
        }
    }

    /// Get recent git history for a file with detailed change summaries.
    /// Returns a formatted string with recent commits and what they changed.
    fn git_history(&self, path: &str, limit: usize) -> String {
        let full_path = self.resolve_path(path);

        // Try to get the repo root
        let repo_root = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(&self.project_root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        let repo_root = match repo_root {
            Some(r) => PathBuf::from(r),
            None => return String::new(), // Not a git repo
        };

        // Get path relative to repo root for git commands
        let relative_path = full_path
            .strip_prefix(&repo_root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        // Get recent commit hashes for this file
        let output = Command::new("git")
            .args([
                "log",
                "--follow",
                &format!("-{}", limit),
                "--format=%H",
                "--",
                &relative_path,
            ])
            .current_dir(&repo_root)
            .output();

        let hashes: Vec<String> = match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            _ => return String::new(),
        };

        if hashes.is_empty() {
            return String::new();
        }

        let mut result = String::from("## Git History (this file)\n\n");

        for hash in hashes.iter().take(limit) {
            // Get commit details
            let detail = Command::new("git")
                .args(["show", "--no-patch", "--format=%h %s%n%b", hash])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = detail {
                if o.status.success() {
                    let info = String::from_utf8_lossy(&o.stdout);
                    let lines: Vec<&str> = info.lines().collect();

                    if let Some(subject_line) = lines.first() {
                        result.push_str(&format!("### {}\n", subject_line));

                        // Include commit body (first few non-empty lines)
                        let body_lines: Vec<&str> = lines
                            .iter()
                            .skip(1)
                            .filter(|l| !l.trim().is_empty())
                            .take(3)
                            .copied()
                            .collect();

                        if !body_lines.is_empty() {
                            for line in body_lines {
                                result.push_str(&format!("{}\n", line));
                            }
                        }
                    }
                }
            }

            // Get the diff for this file in this commit (abbreviated)
            let diff = Command::new("git")
                .args([
                    "show",
                    "--no-color",
                    "--stat",
                    &format!("{}:{}", hash, relative_path),
                ])
                .current_dir(&repo_root)
                .output();

            // Get actual changes (added/removed lines summary)
            let changes = Command::new("git")
                .args(["show", "--numstat", "--format=", hash, "--", &relative_path])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = changes {
                if o.status.success() {
                    let stats = String::from_utf8_lossy(&o.stdout);
                    for line in stats.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            result.push_str(&format!("  +{} -{} lines\n", parts[0], parts[1]));
                        }
                    }
                }
            }

            result.push('\n');
        }

        result
    }

    /// Get git history for a directory with detailed commit info.
    fn git_history_dir(&self, path: &str, limit: usize) -> String {
        let full_path = self.resolve_path(path);

        let repo_root = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(&self.project_root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        let repo_root = match repo_root {
            Some(r) => PathBuf::from(r),
            None => return String::new(),
        };

        let relative_path = full_path
            .strip_prefix(&repo_root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .to_string();

        // Get recent commit hashes touching this directory
        let output = Command::new("git")
            .args([
                "log",
                &format!("-{}", limit),
                "--format=%H",
                "--",
                &relative_path,
            ])
            .current_dir(&repo_root)
            .output();

        let hashes: Vec<String> = match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            _ => return String::new(),
        };

        if hashes.is_empty() {
            return String::new();
        }

        let mut result = String::from("## Git History (this directory)\n\n");

        for hash in hashes.iter().take(limit) {
            // Get commit subject and body
            let detail = Command::new("git")
                .args(["show", "--no-patch", "--format=%h %s%n%b", hash])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = detail {
                if o.status.success() {
                    let info = String::from_utf8_lossy(&o.stdout);
                    let lines: Vec<&str> = info.lines().collect();

                    if let Some(subject_line) = lines.first() {
                        result.push_str(&format!("### {}\n", subject_line));

                        // Include first few lines of commit body
                        let body_lines: Vec<&str> = lines
                            .iter()
                            .skip(1)
                            .filter(|l| !l.trim().is_empty())
                            .take(2)
                            .copied()
                            .collect();

                        if !body_lines.is_empty() {
                            for line in body_lines {
                                result.push_str(&format!("{}\n", line));
                            }
                        }
                    }
                }
            }

            // Get files changed in this commit within the directory
            let files = Command::new("git")
                .args([
                    "show",
                    "--name-only",
                    "--format=",
                    hash,
                    "--",
                    &relative_path,
                ])
                .current_dir(&repo_root)
                .output();

            if let Ok(o) = files {
                if o.status.success() {
                    let file_list = String::from_utf8_lossy(&o.stdout);
                    let files: Vec<&str> = file_list
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .take(5)
                        .collect();
                    if !files.is_empty() {
                        result.push_str("Files: ");
                        result.push_str(&files.join(", "));
                        result.push('\n');
                    }
                }
            }

            result.push('\n');
        }

        result
    }

    /// Read a file at multiple layers and combine results.
    pub fn read_multi_layer(&self, path: &str, layers: &[CodeLayer]) -> ToolResult {
        let mut results = Vec::new();

        for layer in layers {
            let layer_name = match layer {
                CodeLayer::Raw => "raw",
                CodeLayer::Ast => "ast",
                CodeLayer::CallGraph => "call_graph",
                CodeLayer::Cfg => "cfg",
                CodeLayer::Dfg => "dfg",
                CodeLayer::Pdg => "pdg",
                CodeLayer::TheoryGraph => "theory_graph",
            };

            match self.read_at_layer(path, *layer) {
                Ok(view) => {
                    results.push(format!(
                        "═══ {} ═══\n{}",
                        layer_name.to_uppercase(),
                        view.to_context()
                    ));
                }
                Err(e) => {
                    results.push(format!(
                        "═══ {} ═══\n[Error: {}]",
                        layer_name.to_uppercase(),
                        e
                    ));
                }
            }
        }

        ToolResult::success(results.join("\n\n"))
    }

    /// Read a file at the specified layer.
    pub fn read_at_layer(&self, path: &str, layer: CodeLayer) -> Result<LayerView, String> {
        let full_path = self.resolve_path(path);

        self.validate_read_path(&full_path, path)?;

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut analyzer = self.analyzer.write().unwrap();
        Ok(analyzer.analyze(path, &content, layer))
    }

    /// Read a file with automatic layer selection.
    pub fn read_smart(&self, path: &str) -> Result<LayerView, String> {
        let full_path = self.resolve_path(path);

        self.validate_read_path(&full_path, path)?;

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut analyzer = self.analyzer.write().unwrap();
        Ok(analyzer.analyze_efficient(path, &content))
    }

    /// Read an entire folder, returning compressed AST view of all code files.
    ///
    /// If `recursive` is true, includes subdirectories.
    /// Returns a tree-structured view showing files and their symbols.
    pub fn read_folder(&self, path: &str, recursive: bool) -> Result<String, String> {
        self.read_directory(path, &DirectoryReadOptions::summary(recursive))
    }


    fn read_directory(&self, path: &str, options: &DirectoryReadOptions) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        self.validate_read_path(&full_path, path)?;

        if !full_path.is_dir() {
            return Err(format!("{} is not a directory", path));
        }

        let visible_entries = self.collect_directory_entries(path, &full_path, options)?;
        let scope_entries = if options.recursive && options.max_depth.is_some() {
            self.collect_directory_entries_full_scope(path, &full_path, options)?
        } else {
            visible_entries.clone()
        };

        let visible_code_files: Vec<&DirectoryEntryInfo> =
            visible_entries.iter().filter(|entry| entry.code).collect();
        let visible_dirs: Vec<&DirectoryEntryInfo> =
            visible_entries.iter().filter(|entry| entry.is_dir).collect();
        let scope_code_files: Vec<&DirectoryEntryInfo> =
            scope_entries.iter().filter(|entry| entry.code).collect();
        let scope_dirs: Vec<&DirectoryEntryInfo> =
            scope_entries.iter().filter(|entry| entry.is_dir).collect();
        let detected_languages = self.detect_languages(&scope_code_files);
        let likely_files = self.likely_entry_targets(&scope_entries, 8);

        let mut output = String::new();
        output.push_str(&format!("📂 {}\n", path));
        output.push_str(&format!("Mode: directory:{}\n", options.layer.label()));

        if scope_entries.len() != visible_entries.len() {
            output.push_str(&format!(
                "Visible entries: {} ({} directories, {} code files)\n",
                visible_entries.len(),
                visible_dirs.len(),
                visible_code_files.len()
            ));
            output.push_str(&format!(
                "Analyzed scope: {} total ({} directories, {} code files)\n",
                scope_entries.len(),
                scope_dirs.len(),
                scope_code_files.len()
            ));
        } else {
            output.push_str(&format!(
                "Entries: {} total ({} directories, {} code files)\n",
                scope_entries.len(),
                scope_dirs.len(),
                scope_code_files.len()
            ));
        }

        if let Some(max_depth) = options.max_depth {
            output.push_str(&format!("Depth limit: {}\n", max_depth));
        } else if options.recursive {
            output.push_str("Depth limit: recursive\n");
        } else {
            output.push_str("Depth limit: 1\n");
        }

        if !detected_languages.is_empty() {
            output.push_str(&format!("Languages: {}\n", detected_languages.join(", ")));
        }

        if !options.include.is_empty() {
            output.push_str(&format!("Include: {}\n", options.include.join(", ")));
        }

        if !options.exclude.is_empty() {
            output.push_str(&format!("Exclude: {}\n", options.exclude.join(", ")));
        }

        output.push('\n');

        match options.layer {
            DirectoryLayer::Summary => {
                if likely_files.is_empty() {
                    output.push_str("No matching files found.\n");
                } else {
                    output.push_str("## Likely important files\n\n");
                    for target in likely_files {
                        output.push_str(&format!("- `{}`\n", target));
                    }
                }
            }
            DirectoryLayer::Structure => {
                output.push_str("## Directory structure\n\n");
                output.push_str(&self.render_directory_tree(&visible_entries));
            }
            DirectoryLayer::Hotspots => {
                output.push_str("## Hotspots\n\n");
                if likely_files.is_empty() {
                    output.push_str("No matching files found.\n");
                } else {
                    for target in likely_files {
                        let reason = self.hotspot_reason(&target);
                        output.push_str(&format!("- `{}` — {}\n", target, reason));
                    }
                }
            }
            DirectoryLayer::Semantic => {
                output.push_str("## Semantic map\n\n");
                if scope_code_files.is_empty() {
                    output.push_str("No code files found.\n");
                } else {
                    for entry in scope_code_files.iter().take(12) {
                        output.push_str(&self.semantic_summary_for(&entry.relative));
                    }
                    if scope_code_files.len() > 12 {
                        output.push_str(&format!("- ... and {} more code files\n", scope_code_files.len() - 12));
                    }
                }
            }
            DirectoryLayer::Scalpel => {
                output.push_str("## Next inspection targets\n\n");
                if likely_files.is_empty() {
                    output.push_str("No matching files found.\n");
                } else {
                    for target in likely_files {
                        output.push_str(&format!("- read(path=\"{}\", layer=\"ast\")\n", target));
                    }
                }
            }
        }

        Ok(output)
    }

    fn collect_directory_entries(
        &self,
        requested_path: &str,
        dir: &Path,
        options: &DirectoryReadOptions,
    ) -> Result<Vec<DirectoryEntryInfo>, String> {
        let mut entries = Vec::new();
        self.collect_directory_entries_recursive(requested_path, dir, options, 0, &mut entries)?;
        entries.sort_by(|a, b| a.relative.cmp(&b.relative));
        Ok(entries)
    }

    fn collect_directory_entries_full_scope(
        &self,
        requested_path: &str,
        dir: &Path,
        options: &DirectoryReadOptions,
    ) -> Result<Vec<DirectoryEntryInfo>, String> {
        let mut full_options = options.clone();
        full_options.max_depth = None;
        self.collect_directory_entries(requested_path, dir, &full_options)
    }

    fn collect_directory_entries_recursive(
        &self,
        requested_path: &str,
        dir: &Path,
        options: &DirectoryReadOptions,
        depth: usize,
        entries: &mut Vec<DirectoryEntryInfo>,
    ) -> Result<(), String> {
        let read_dir = std::fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in read_dir {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if name.starts_with('.') || matches!(name, "target" | "node_modules" | "dist" | "build" | "__pycache__" | ".git") {
                continue;
            }

            let relative = self.relative_to_request(requested_path, &path);
            if !options.allows_path(&relative) {
                continue;
            }

            let is_dir = path.is_dir();
            let is_code = !is_dir && Self::is_code_file(&path);
            let entry_depth = depth + 1;

            entries.push(DirectoryEntryInfo {
                relative: relative.clone(),
                depth: entry_depth,
                is_dir,
                code: is_code,
            });

            let can_descend = is_dir
                && options.recursive
                && options.max_depth.map(|max| entry_depth < max).unwrap_or(true);

            if can_descend {
                self.collect_directory_entries_recursive(requested_path, &path, options, entry_depth, entries)?;
            }
        }

        Ok(())
    }

    fn relative_to_request(&self, requested_path: &str, path: &Path) -> String {
        let base = self.resolve_path(requested_path);
        path.strip_prefix(&base)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn render_directory_tree(&self, entries: &[DirectoryEntryInfo]) -> String {
        let mut output = String::new();
        for entry in entries {
            let indent = "  ".repeat(entry.depth.saturating_sub(1));
            let marker = if entry.is_dir { "📁" } else if entry.code { "📄" } else { "•" };
            output.push_str(&format!("{}{} {}\n", indent, marker, entry.relative));
        }
        output
    }

    fn detect_languages(&self, entries: &[&DirectoryEntryInfo]) -> Vec<String> {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for entry in entries {
            let language = Self::language_for_path(&entry.relative);
            *counts.entry(language).or_insert(0) += 1;
        }

        let mut counts: Vec<_> = counts.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        counts
            .into_iter()
            .map(|(language, count)| format!("{} ({})", language, count))
            .collect()
    }

    fn language_for_path(path: &str) -> &'static str {
        Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext {
                "rs" => "Rust",
                "py" => "Python",
                "js" | "jsx" => "JavaScript",
                "ts" | "tsx" => "TypeScript",
                "go" => "Go",
                "java" => "Java",
                "c" | "h" => "C",
                "cpp" | "hpp" => "C++",
                "rb" => "Ruby",
                "php" => "PHP",
                "swift" => "Swift",
                "kt" => "Kotlin",
                "scala" => "Scala",
                "zig" => "Zig",
                "lean" => "Lean",
                "pl" | "pm" | "t" | "cgi" => "Perl",
                "nim" | "nims" | "nimble" => "Nim",
                _ => "Other",
            })
            .unwrap_or("Other")
    }

    fn likely_entry_targets(&self, entries: &[DirectoryEntryInfo], limit: usize) -> Vec<String> {
        let mut candidates: Vec<(i32, String)> = entries
            .iter()
            .filter(|entry| !entry.is_dir)
            .map(|entry| (self.importance_score(&entry.relative), entry.relative.clone()))
            .collect();

        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        candidates.into_iter().take(limit).map(|(_, path)| path).collect()
    }

    fn importance_score(&self, relative: &str) -> i32 {
        let path = Path::new(relative);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(relative);
        let parent = path
            .parent()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        let mut score = 0;

        if matches!(file_name, "Cargo.toml" | "package.json" | "README.md" | "CLAUDE.md") {
            score += 120;
        }
        if matches!(file_name, "lib.rs" | "main.rs" | "mod.rs") {
            score += 90;
        }
        if relative.starts_with("src/") || relative.starts_with("server/src/") {
            score += 40;
        }
        if relative.starts_with("lean/") {
            score += 38;
        }
        if relative.starts_with("crates/") {
            score += 36;
        }
        if relative.contains("/tests") || relative.contains("test") {
            score += 10;
        }
        if relative.contains("examples") || relative.contains("benches") {
            score += 5;
        }
        if parent.ends_with("/src") || parent == "src" {
            score += 18;
        }
        if parent.ends_with("/lean") || relative.contains("/Gutoe/") {
            score += 16;
        }
        if Self::is_code_file(path) {
            score += 20;
        }

        score - (relative.matches('/').count() as i32 * 3)
    }

    fn hotspot_reason(&self, relative: &str) -> &'static str {
        let file_name = Path::new(relative)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(relative);

        if matches!(file_name, "Cargo.toml" | "package.json") {
            "build/package manifest"
        } else if matches!(file_name, "README.md" | "CLAUDE.md") {
            "project/docs entrypoint"
        } else if matches!(file_name, "lib.rs" | "main.rs" | "mod.rs") {
            "module entrypoint"
        } else if relative.starts_with("src/") {
            "primary source file"
        } else {
            "likely relevant file"
        }
    }


    fn semantic_role_hint(&self, relative: &str, symbols: &[super::ast::Symbol]) -> String {
        let file_name = Path::new(relative)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(relative);

        if file_name == "lib.rs" {
            return "crate root / public API boundary".to_string();
        }
        if file_name == "main.rs" {
            return "CLI/application entrypoint".to_string();
        }
        if file_name == "mod.rs" {
            return "module root / fanout point".to_string();
        }
        if file_name.eq_ignore_ascii_case("Cargo.toml") {
            return "build/package manifest".to_string();
        }
        if file_name.contains("error") {
            return "shared error surface".to_string();
        }
        if file_name.contains("config") || file_name.contains("settings") {
            return "configuration surface".to_string();
        }
        if file_name.contains("schema") {
            return "config/schema surface".to_string();
        }
        if file_name.contains("runner") || file_name.contains("dispatch") {
            return "execution/orchestration seam".to_string();
        }
        if file_name.contains("tool") || file_name.contains("registry") {
            return "tooling/registry surface".to_string();
        }
        if file_name.contains("cli") || file_name.contains("command") {
            return "CLI command surface".to_string();
        }
        if relative.contains("/tests") || file_name.contains("test") {
            return "test surface".to_string();
        }

        let trait_count = symbols
            .iter()
            .filter(|symbol| matches!(symbol.kind, super::ast::SymbolKind::Trait | super::ast::SymbolKind::Interface))
            .count();
        let type_count = symbols
            .iter()
            .filter(|symbol| matches!(symbol.kind, super::ast::SymbolKind::Struct | super::ast::SymbolKind::Enum | super::ast::SymbolKind::Class))
            .count();
        let function_count = symbols
            .iter()
            .filter(|symbol| matches!(symbol.kind, super::ast::SymbolKind::Function | super::ast::SymbolKind::Method))
            .count();
        let module_count = symbols
            .iter()
            .filter(|symbol| matches!(symbol.kind, super::ast::SymbolKind::Module))
            .count();

        if trait_count > 0 && trait_count >= type_count.max(function_count) {
            "trait-heavy interface surface".to_string()
        } else if type_count > function_count && type_count > 0 {
            "type-heavy domain surface".to_string()
        } else if function_count >= 4 && (file_name.contains("registry") || file_name.contains("dispatch")) {
            "registry/dispatch surface".to_string()
        } else if function_count >= 4 {
            "implementation-heavy function surface".to_string()
        } else if module_count > 0 {
            "module aggregation surface".to_string()
        } else {
            "semantic entrypoint".to_string()
        }
    }

    fn semantic_symbol_mix(&self, symbols: &[super::ast::Symbol]) -> String {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();

        for symbol in symbols {
            let label = match symbol.kind {
                super::ast::SymbolKind::Function => "fn",
                super::ast::SymbolKind::Method => "method",
                super::ast::SymbolKind::Struct => "struct",
                super::ast::SymbolKind::Enum => "enum",
                super::ast::SymbolKind::Trait => "trait",
                super::ast::SymbolKind::Interface => "interface",
                super::ast::SymbolKind::Class => "class",
                super::ast::SymbolKind::Module => "module",
                super::ast::SymbolKind::Constant => "const",
                super::ast::SymbolKind::Variable => "var",
                super::ast::SymbolKind::Import => continue,
            };
            *counts.entry(label).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .map(|(label, count)| format!("{} {}", count, label))
            .take(3)
            .collect::<Vec<_>>()
            .join(", ")
    }


    fn dedupe_notable_symbols(&self, symbols: &[super::ast::Symbol]) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut names = Vec::new();

        for symbol in symbols {
            if matches!(symbol.kind, super::ast::SymbolKind::Import) {
                continue;
            }
            if seen.insert(symbol.name.clone()) {
                names.push(symbol.name.clone());
            }
            if names.len() >= 4 {
                break;
            }
        }

        names
    }

    fn semantic_notable_symbols(&self, symbols: &[super::ast::Symbol]) -> String {
        self.dedupe_notable_symbols(symbols).join(", ")
    }

    fn semantic_summary_for(&self, relative: &str) -> String {
        match self.read_at_layer(relative, CodeLayer::Ast) {
            Ok(view) => {
                let symbols = &view.symbols;
                let role = self.semantic_role_hint(relative, symbols);
                let mix = self.semantic_symbol_mix(symbols);
                let notable = self.semantic_notable_symbols(symbols);

                let mut line = format!("- `{}` — {}", relative, role);

                if !mix.is_empty() {
                    line.push_str(&format!("; {}", mix));
                }

                if !notable.is_empty() {
                    line.push_str(&format!("; notable: {}", notable));
                }

                line.push('\n');
                line
            }
            Err(_) => format!("- `{}` — semantic summary unavailable\n", relative),
        }
    }

    /// Collect code files from a directory.
    fn collect_code_files(
        &self,
        dir: &Path,
        recursive: bool,
        files: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        let entries =
            std::fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            // Skip hidden files/dirs
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false)
            {
                continue;
            }

            if path.is_dir() {
                // Skip common non-code directories
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(
                    name,
                    "target" | "node_modules" | "dist" | "build" | "__pycache__" | ".git"
                ) {
                    continue;
                }
                if recursive {
                    self.collect_code_files(&path, recursive, files)?;
                }
            } else if Self::is_code_file(&path) {
                files.push(path);
            }
        }

        Ok(())
    }

    /// Check if a file is a recognized code file.
    fn is_code_file(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| {
                matches!(
                    ext,
                    "rs" | "py" | "js" | "ts" | "tsx" | "jsx" |
                "go" | "java" | "c" | "cpp" | "h" | "hpp" |
                "rb" | "php" | "swift" | "kt" | "scala" | "zig" |
                "pl" | "pm" | "cgi" | "t" |  // Perl files
                "nim" | "nims" | "nimble" |  // Nim files
                "lean" // Lean files
                )
            })
            .unwrap_or(false)
    }

    /// Read a specific symbol from a file (returns raw code for just that symbol).
    pub fn read_symbol(&self, path: &str, symbol_name: &str) -> Result<String, String> {
        let full_path = self.resolve_path(path);

        self.validate_read_path(&full_path, path)?;

        let content = std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let lang = Lang::from_path(&full_path).ok_or_else(|| "Unsupported language".to_string())?;

        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(&content, lang);

        let symbol = symbols
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(symbol_name))
            .ok_or_else(|| format!("Symbol '{}' not found in {}", symbol_name, path))?;

        let lines: Vec<&str> = content.lines().collect();
        let symbol_lines: Vec<&str> = lines[symbol.start_line - 1..symbol.end_line].to_vec();

        Ok(format!(
            "## {} (lines {}-{})\n```\n{}\n```",
            symbol_name,
            symbol.start_line,
            symbol.end_line,
            symbol_lines.join("\n")
        ))
    }

    /// Batch read multiple files/symbols with individual granularity.
    /// This is the key efficiency feature - gather diverse context in one call.
    pub fn read_batch(&self, requests: &[ReadRequest]) -> Vec<Result<String, String>> {
        requests
            .iter()
            .map(|req| {
                if let Some(symbol) = &req.symbol {
                    // Symbol-specific read (always raw)
                    self.read_symbol(&req.path, symbol)
                } else {
                    // Full file at specified layer
                    self.read_at_layer(&req.path, req.layer.clone())
                        .map(|view| view.to_context())
                }
            })
            .collect()
    }

    /// Batch read and combine into single context string.
    pub fn read_tree(&self, requests: &[ReadRequest]) -> String {
        // Track all batch reads
        for req in requests {
            let full_path = self.resolve_path(&req.path);
            self.read_tracker.mark_read(&full_path);
        }

        // Read and build output incrementally — each file is read, formatted,
        // and the intermediate data dropped before the next file loads.
        let mut output = String::new();

        for req in requests {
            crate::trace_rss(&format!("read_tree: {:?} layer={:?}", req.path, req.layer));

            let result = if let Some(symbol) = &req.symbol {
                self.read_symbol(&req.path, symbol)
            } else {
                self.read_at_layer(&req.path, req.layer.clone())
                    .map(|view| view.to_context())
            };

            output.push_str(&format!("\n─── {} ", req.path));
            if let Some(sym) = &req.symbol {
                output.push_str(&format!("({}) ", sym));
            }
            output.push_str(&format!("[{:?}] ───\n", req.layer));

            match result {
                Ok(content) => output.push_str(&content),
                Err(e) => output.push_str(&format!("Error: {}", e)),
            }
            output.push('\n');

            crate::trace_rss(&format!("read_tree done: {:?} output={}KB", req.path, output.len()/1024));
        }

        output
    }

    /// Read the entire codebase structure in a single call.
    ///
    /// Returns a comprehensive view of:
    /// - Workspace/crate structure (from Cargo.toml)
    /// - Crate purposes (from CLAUDE.md, README, or lib.rs docs)
    /// - Key public types and functions per crate
    /// - File tree with symbol counts
    ///
    /// Respects .gitignore patterns.
    ///
    /// This is the most token-efficient way to understand an entire codebase.
    pub fn read_codebase(&self) -> Result<String, String> {
        let mut output = String::new();

        // Parse .gitignore
        let gitignore = self.parse_gitignore();

        // Try to find workspace Cargo.toml
        let cargo_path = self.project_root.join("Cargo.toml");
        let workspace_info = if cargo_path.exists() {
            self.parse_cargo_workspace(&cargo_path)?
        } else {
            WorkspaceInfo::default()
        };

        // Header
        output.push_str(&format!("# {} Codebase Structure\n\n", workspace_info.name));

        // Workspace overview
        if !workspace_info.members.is_empty() {
            output.push_str("## Workspace Members\n\n");
            for member in &workspace_info.members {
                output.push_str(&format!("- `{}`\n", member));
            }
            output.push_str("\n");
        }

        // Analyze each crate
        output.push_str("## Crates\n\n");

        let crates_dir = self.project_root.join("crates");
        let src_dir = self.project_root.join("src");

        if crates_dir.is_dir() {
            // Multi-crate workspace
            let mut crates: Vec<_> = std::fs::read_dir(&crates_dir)
                .map_err(|e| format!("Failed to read crates dir: {}", e))?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter(|e| !self.is_ignored(&e.path(), &gitignore))
                .collect();
            crates.sort_by_key(|e| e.file_name());

            for entry in crates {
                let crate_name = entry.file_name().to_string_lossy().to_string();
                output.push_str(&self.analyze_crate(&entry.path(), &crate_name)?);
            }
        } else if src_dir.is_dir() {
            // Single crate
            output.push_str(&self.analyze_crate(&self.project_root, &workspace_info.name)?);
        }

        // File tree summary
        output.push_str("## File Tree\n\n");
        output.push_str(&self.generate_file_tree_with_gitignore(&gitignore)?);

        Ok(output)
    }

    /// Parse .gitignore file and return list of patterns.
    fn parse_gitignore(&self) -> Vec<String> {
        let gitignore_path = self.project_root.join(".gitignore");
        if !gitignore_path.exists() {
            return vec![];
        }

        std::fs::read_to_string(&gitignore_path)
            .unwrap_or_default()
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#')
            })
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .collect()
    }

    /// Check if a path matches any gitignore pattern.
    fn is_ignored(&self, path: &Path, patterns: &[String]) -> bool {
        let relative = path.strip_prefix(&self.project_root).unwrap_or(path);
        let relative_str = relative.to_string_lossy();

        for pattern in patterns {
            // Simple pattern matching - handle common cases
            if pattern.starts_with('*') {
                // Wildcard at start (e.g., *.log)
                let suffix = &pattern[1..];
                if relative_str.ends_with(suffix) {
                    return true;
                }
            } else if pattern.ends_with('*') {
                // Wildcard at end
                let prefix = &pattern[..pattern.len() - 1];
                if relative_str.starts_with(prefix) {
                    return true;
                }
            } else {
                // Exact match or directory match
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                // Match directory/file name
                if name == pattern.as_str() {
                    return true;
                }
                // Match path component
                if relative_str.as_ref() == pattern.as_str()
                    || relative_str.starts_with(&format!("{}/", pattern))
                {
                    return true;
                }
            }
        }

        false
    }

    /// Parse Cargo.toml for workspace info.
    fn parse_cargo_workspace(&self, path: &Path) -> Result<WorkspaceInfo, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read Cargo.toml: {}", e))?;

        let mut info = WorkspaceInfo::default();

        // Extract package name
        for line in content.lines() {
            if line.starts_with("name = ") {
                info.name = line
                    .trim_start_matches("name = ")
                    .trim_matches('"')
                    .to_string();
                break;
            }
        }

        // Extract workspace members
        let mut in_members = false;
        for line in content.lines() {
            if line.contains("[workspace]") {
                continue;
            }
            if line.contains("members = [") || line.starts_with("members = [") {
                in_members = true;
                // Handle single-line members
                if let Some(start) = line.find('[') {
                    let members_part = &line[start + 1..];
                    if let Some(end) = members_part.find(']') {
                        let members_str = &members_part[..end];
                        for m in members_str.split(',') {
                            let m = m.trim().trim_matches('"').trim();
                            if !m.is_empty() {
                                info.members.push(m.to_string());
                            }
                        }
                        in_members = false;
                    }
                }
                continue;
            }
            if in_members {
                if line.contains(']') {
                    in_members = false;
                    continue;
                }
                let member = line.trim().trim_matches('"').trim_matches(',').trim();
                if !member.is_empty() {
                    info.members.push(member.to_string());
                }
            }
        }

        Ok(info)
    }

    /// Analyze a single crate.
    fn analyze_crate(&self, crate_path: &Path, name: &str) -> Result<String, String> {
        let mut output = format!("### {}\n\n", name);

        // Try to get purpose from CLAUDE.md, README.md, or lib.rs
        let purpose = self.get_crate_purpose(crate_path);
        if !purpose.is_empty() {
            output.push_str(&format!("**Purpose:** {}\n\n", purpose));
        }

        // Find src directory
        let src_path = crate_path.join("src");
        if !src_path.is_dir() {
            output.push_str("_No src directory_\n\n");
            return Ok(output);
        }

        // Analyze lib.rs or main.rs for public API
        let lib_path = src_path.join("lib.rs");
        let main_path = src_path.join("main.rs");

        let entry_file = if lib_path.exists() {
            Some(lib_path)
        } else if main_path.exists() {
            Some(main_path)
        } else {
            None
        };

        if let Some(entry) = entry_file {
            let relative = entry
                .strip_prefix(&self.project_root)
                .unwrap_or(&entry)
                .to_string_lossy();

            match self.read_at_layer(&relative, CodeLayer::Ast) {
                Ok(view) => {
                    // Show key symbols (functions, structs, traits, etc.)
                    let key_symbols: Vec<_> = view
                        .symbols
                        .iter()
                        .filter(|s| {
                            matches!(
                                s.kind,
                                super::ast::SymbolKind::Function
                                    | super::ast::SymbolKind::Struct
                                    | super::ast::SymbolKind::Trait
                                    | super::ast::SymbolKind::Enum
                                    | super::ast::SymbolKind::Class
                            )
                        })
                        .collect();

                    if !key_symbols.is_empty() {
                        output.push_str("**Key Types/Functions:**\n");
                        for sym in key_symbols.iter().take(15) {
                            output.push_str(&format!("- `{}` ({:?})\n", sym.name, sym.kind));
                        }
                        if key_symbols.len() > 15 {
                            output.push_str(&format!("- ... +{} more\n", key_symbols.len() - 15));
                        }
                        output.push_str("\n");
                    }
                }
                Err(_) => {}
            }
        }

        // Count files
        let mut file_count = 0;
        let _ = self.count_code_files(&src_path, &mut file_count);
        output.push_str(&format!("_Files: {}_\n\n", file_count));

        Ok(output)
    }

    /// Get crate purpose from documentation files.
    fn get_crate_purpose(&self, crate_path: &Path) -> String {
        // Try CLAUDE.md first
        let claude_path = crate_path.join("CLAUDE.md");
        if claude_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&claude_path) {
                // Look for first paragraph or "Purpose:" section
                for line in content.lines() {
                    if line.starts_with("**Purpose:**") || line.starts_with("Purpose:") {
                        return line
                            .trim_start_matches("**Purpose:**")
                            .trim_start_matches("Purpose:")
                            .trim()
                            .to_string();
                    }
                }
                // Or first non-header, non-empty line
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty()
                        && !trimmed.starts_with('#')
                        && !trimmed.starts_with("```")
                    {
                        return trimmed.chars().take(150).collect::<String>();
                    }
                }
            }
        }

        // Try README.md
        let readme_path = crate_path.join("README.md");
        if readme_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&readme_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        return trimmed.chars().take(150).collect::<String>();
                    }
                }
            }
        }

        // Try lib.rs doc comment
        let lib_path = crate_path.join("src/lib.rs");
        if lib_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&lib_path) {
                // Look for //! doc comments
                let mut doc = String::new();
                for line in content.lines() {
                    if line.starts_with("//!") {
                        let text = line.trim_start_matches("//!").trim();
                        if !text.is_empty() {
                            doc.push_str(text);
                            doc.push(' ');
                            if doc.len() > 100 {
                                break;
                            }
                        }
                    } else if !line.trim().is_empty() && !doc.is_empty() {
                        break;
                    }
                }
                if !doc.is_empty() {
                    return doc.trim().to_string();
                }
            }
        }

        String::new()
    }

    /// Count code files recursively.
    fn count_code_files(&self, dir: &Path, count: &mut usize) -> Result<(), String> {
        self.count_code_files_with_gitignore(dir, count, &[])
    }

    /// Count code files recursively, respecting gitignore.
    fn count_code_files_with_gitignore(
        &self,
        dir: &Path,
        count: &mut usize,
        gitignore: &[String],
    ) -> Result<(), String> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || matches!(name, "target" | "node_modules") {
                continue;
            }

            // Check gitignore
            if self.is_ignored(&path, gitignore) {
                continue;
            }

            if path.is_dir() {
                self.count_code_files_with_gitignore(&path, count, gitignore)?;
            } else if Self::is_code_file(&path) {
                *count += 1;
            }
        }

        Ok(())
    }

    /// Generate a compact file tree.
    fn generate_file_tree(&self) -> Result<String, String> {
        let mut output = String::new();
        self.build_file_tree(&self.project_root, "", &mut output, 0, &[])?;
        Ok(output)
    }

    /// Generate a compact file tree respecting gitignore.
    fn generate_file_tree_with_gitignore(&self, gitignore: &[String]) -> Result<String, String> {
        let mut output = String::new();
        self.build_file_tree(&self.project_root, "", &mut output, 0, gitignore)?;
        Ok(output)
    }

    /// Recursively build file tree.
    fn build_file_tree(
        &self,
        dir: &Path,
        prefix: &str,
        output: &mut String,
        depth: usize,
        gitignore: &[String],
    ) -> Result<(), String> {
        if depth > 4 {
            // Limit depth to keep output manageable
            return Ok(());
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read dir: {}", e))?
            .filter_map(|e| e.ok())
            .collect();

        // Skip hidden and build directories
        entries.retain(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            !name.starts_with('.')
                && !matches!(name.as_str(), "target" | "node_modules" | "dist" | "build")
        });

        // Skip gitignored entries
        entries.retain(|e| !self.is_ignored(&e.path(), gitignore));

        entries.sort_by_key(|e| {
            let is_dir = e.path().is_dir();
            let name = e.file_name().to_string_lossy().to_string();
            (!is_dir, name) // Dirs first, then alphabetical
        });

        for (i, entry) in entries.iter().enumerate() {
            let is_last = i == entries.len() - 1;
            let connector = if is_last { "└── " } else { "├── " };
            let name = entry.file_name().to_string_lossy().to_string();
            let path = entry.path();

            if path.is_dir() {
                // Count contents
                let mut file_count = 0;
                let _ = self.count_code_files_with_gitignore(&path, &mut file_count, gitignore);
                if file_count > 0 {
                    output.push_str(&format!(
                        "{}{}{}/  ({} files)\n",
                        prefix, connector, name, file_count
                    ));
                } else {
                    output.push_str(&format!("{}{}{}/\n", prefix, connector, name));
                }

                let new_prefix = format!("{}{}   ", prefix, if is_last { " " } else { "│" });
                self.build_file_tree(&path, &new_prefix, output, depth + 1, gitignore)?;
            } else if Self::is_code_file(&path) || name.ends_with(".toml") || name.ends_with(".md")
            {
                output.push_str(&format!("{}{}{}\n", prefix, connector, name));
            }
        }

        Ok(())
    }
}

/// Workspace metadata parsed from Cargo.toml.
#[derive(Default)]
struct WorkspaceInfo {
    name: String,
    members: Vec<String>,
}

#[async_trait]
impl Tool for SmartReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn to_param(&self) -> ToolParam {
        use crate::types::PropertySchema;

        let read_item = PropertySchema::object()
            .property(
                "path",
                PropertySchema::string().with_description("File path (relative to project root, or absolute anywhere on the machine)"),
                true,
            )
            .property(
                "layer",
                PropertySchema::string()
                    .with_description("Layer: raw, ast, call_graph, cfg, dfg, pdg, theory_graph"),
                false,
            )
            .property(
                "layers",
                PropertySchema::array(PropertySchema::string())
                    .with_description("Multiple layers at once"),
                false,
            )
            .property(
                "symbol",
                PropertySchema::string().with_description("Specific symbol to extract"),
                false,
            );

        ToolParam::new(
            "read",
            InputSchema::object()
                .optional_string("path", "File or folder path (relative to project root, or absolute anywhere on the machine)")
                .optional_string("layer", "Single layer: 'raw', 'ast', 'call_graph', 'cfg', 'dfg', 'pdg', 'theory_graph' (default: 'ast')")
                .property("layers", PropertySchema::array(PropertySchema::string()).with_description("Multiple layers: ['ast', 'call_graph', 'dfg'] - returns all in one call"), false)
                .optional_string("symbol", "Specific symbol to extract (returns raw)")
                .optional_string("query", "Search git history for commits matching this query (e.g. 'permission fix', '#30351')")
                .property("recursive", PropertySchema::boolean().with_description("For folders: include subdirectories (default: true)"), false)
                .property("max_depth", PropertySchema::number().with_description("For folders: maximum tree depth to include"), false)
                .property("include", PropertySchema::array(PropertySchema::string()).with_description("For folders: glob filters to include (e.g. ['src/**', '*.rs'])"), false)
                .property("exclude", PropertySchema::array(PropertySchema::string()).with_description("For folders: glob filters to exclude"), false)
                .optional_string("directory_layer", "For folders/codebases: summary, structure, hotspots, semantic, scalpel")
                .property("reads", PropertySchema::array(read_item).with_description("Batch reads with individual granularity"), false)
                .property("codebase", PropertySchema::boolean().with_description("Read entire codebase structure (ignores path)"), false),
        )
        .with_description(
            "Read code with layered analysis and git history. Prefer this over ad hoc shell file readers when inspecting code or text. Paths may be relative to the project root or absolute anywhere on the machine. Add 'query' to search git for relevant commits (e.g. query='permission fix'). Layers: raw, ast, call_graph, cfg, dfg, pdg, theory_graph. File: {path, layer?, query?}. Directory: {path, recursive?, max_depth?, include?, exclude?, directory_layer?}. Batch: {reads: [...]}.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        if input
            .get("codebase")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let layer = parse_directory_layer(input.get("directory_layer").and_then(|v| v.as_str()));
            return match layer {
                DirectoryLayer::Summary => match self.read_codebase() {
                    Ok(content) => {
                        let meta = read_metadata(".", "directory:summary", None, content.lines().count());
                        ToolResult::success_with_metadata(content, meta)
                    }
                    Err(e) => ToolResult::error(e),
                },
                _ => {
                    let options = DirectoryReadOptions {
                        recursive: true,
                        max_depth: input
                            .get("max_depth")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize),
                        include: input
                            .get("include")
                            .and_then(|v| v.as_array())
                            .map(|items| items.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default(),
                        exclude: input
                            .get("exclude")
                            .and_then(|v| v.as_array())
                            .map(|items| items.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default(),
                        layer,
                    };
                    match self.read_directory(".", &options) {
                        Ok(content) => {
                            let meta = read_metadata(".", &format!("directory:{}", options.layer.label()), None, content.lines().count());
                            ToolResult::success_with_metadata(content, meta)
                        }
                        Err(e) => ToolResult::error(e),
                    }
                }
            };
        }

        if let Some(reads) = input.get("reads").and_then(|v| v.as_array()) {
            let requests: Vec<ReadRequest> = reads
                .iter()
                .filter_map(|r| {
                    let path = r.get("path")?.as_str()?;
                    let layer = parse_layer(r.get("layer").and_then(|v| v.as_str()));
                    let symbol = r
                        .get("symbol")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    Some(ReadRequest {
                        path: path.to_string(),
                        layer,
                        symbol,
                    })
                })
                .collect();

            if requests.is_empty() {
                return ToolResult::error("reads array is empty or invalid");
            }

            crate::trace_rss(&format!("batch read start: {} files", requests.len()));
            let content = self.read_tree(&requests);
            crate::trace_rss(&format!("batch read done: {}KB output", content.len()/1024));
            let items: Vec<serde_json::Value> = requests
                .iter()
                .map(|r| read_metadata(&r.path, &format!("{:?}", r.layer), r.symbol.as_deref(), 0))
                .collect();
            let meta = batch_read_metadata(&items);
            return ToolResult::success_with_metadata(content, meta);
        }

        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path or reads is required");
        }

        let full_path = self.resolve_path(path);
        let query = input.get("query").and_then(|v| v.as_str());

        if full_path.is_dir() {
            let options = DirectoryReadOptions {
                recursive: input
                    .get("recursive")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                max_depth: input
                    .get("max_depth")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize),
                include: input
                    .get("include")
                    .and_then(|v| v.as_array())
                    .map(|items| items.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default(),
                exclude: input
                    .get("exclude")
                    .and_then(|v| v.as_array())
                    .map(|items| items.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default(),
                layer: parse_directory_layer(input.get("directory_layer").and_then(|v| v.as_str())),
            };

            return match self.read_directory(path, &options) {
                Ok(content) => {
                    let full = if let Some(q) = query {
                        let history = self.git_search(path, q, 5);
                        format!("{}{}", history, content)
                    } else {
                        content
                    };
                    let meta = read_metadata(path, &format!("directory:{}", options.layer.label()), None, full.lines().count());
                    ToolResult::success_with_metadata(full, meta)
                }
                Err(e) => ToolResult::error(e),
            };
        }

        self.read_tracker.mark_read(&full_path);

        if let Some(symbol) = input.get("symbol").and_then(|v| v.as_str()) {
            return match self.read_symbol(path, symbol) {
                Ok(content) => {
                    let full = if let Some(q) = query {
                        let history = self.git_search(path, q, 5);
                        format!("{}{}", history, content)
                    } else {
                        content
                    };
                    let meta = read_metadata(path, "raw", Some(symbol), full.lines().count());
                    ToolResult::success_with_metadata(full, meta)
                }
                Err(e) => ToolResult::error(e),
            };
        }

        if let Some(layers_array) = input.get("layers").and_then(|v| v.as_array()) {
            let layers: Vec<CodeLayer> = layers_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| parse_layer(Some(s)))
                .collect();

            if layers.is_empty() {
                return ToolResult::error("layers array is empty");
            }

            return self.read_multi_layer(path, &layers);
        }

        let layer = parse_layer(input.get("layer").and_then(|v| v.as_str()));

        match self.read_at_layer(path, layer) {
            Ok(view) => {
                let layer_name = format!("{:?}", view.layer);
                let content = view.to_context();
                let full = if let Some(q) = query {
                    let history = self.git_search(path, q, 5);
                    format!("{}{}", history, content)
                } else {
                    let history = self.git_history_brief(path, 5);
                    format!("{}{}", history, content)
                };
                let meta = read_metadata(path, &layer_name, None, full.lines().count());
                ToolResult::success_with_metadata(full, meta)
            }
            Err(e) => ToolResult::error(e),
        }
    }
}

fn parse_layer(s: Option<&str>) -> CodeLayer {
    match s.unwrap_or("ast") {
        "raw" => CodeLayer::Raw,
        "ast" => CodeLayer::Ast,
        "call_graph" | "callgraph" => CodeLayer::CallGraph,
        "cfg" => CodeLayer::Cfg,
        "dfg" => CodeLayer::Dfg,
        "pdg" => CodeLayer::Pdg,
        "theory_graph" | "theorygraph" | "decl_graph" | "declgraph" => CodeLayer::TheoryGraph,
        _ => CodeLayer::Ast,
    }
}



fn parse_directory_layer(s: Option<&str>) -> DirectoryLayer {
    match s.unwrap_or("summary") {
        "summary" => DirectoryLayer::Summary,
        "structure" | "tree" => DirectoryLayer::Structure,
        "hotspots" => DirectoryLayer::Hotspots,
        "semantic" => DirectoryLayer::Semantic,
        "scalpel" => DirectoryLayer::Scalpel,
        _ => DirectoryLayer::Summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_smart_read_tool() {
        let dir = TempDir::new().unwrap();

        let source = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

struct Calculator {
    value: i32,
}
"#;

        fs::write(dir.path().join("calc.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Test AST layer
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert("layer".to_string(), serde_json::json!("ast"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("add"));
        assert!(content.contains("Calculator"));

        // Test raw layer
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("calc.rs"));
        input.insert("layer".to_string(), serde_json::json!("raw"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("pub fn add(a: i32, b: i32)"));
    }

    #[tokio::test]
    async fn test_symbol_read() {
        let dir = TempDir::new().unwrap();

        let source = r#"pub fn first() {
    println!("First function");
}

pub fn second() {
    println!("Second function");
}

pub fn third() {
    println!("Third function");
}
"#;

        fs::write(dir.path().join("funcs.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Read just one function
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("funcs.rs"));
        input.insert("symbol".to_string(), serde_json::json!("second"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should have the second function
        assert!(content.contains("Second function"));
        // Should NOT have the other functions' content
        assert!(!content.contains("First function"));
        assert!(!content.contains("Third function"));
    }

    #[tokio::test]
    async fn test_symbol_read_case_insensitive() {
        let dir = TempDir::new().unwrap();

        let source = r#"pub fn _calculate_separability_matrix() {
    println!("lowercase function");
}
"#;

        fs::write(dir.path().join("funcs.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Request with different case (capital C)
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("funcs.rs"));
        input.insert(
            "symbol".to_string(),
            serde_json::json!("_Calculate_separability_matrix"),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error(), "Should find symbol with different case");
        let content = result.to_content_string();
        assert!(content.contains("lowercase function"));
    }

    #[tokio::test]
    async fn test_batch_read() {
        let dir = TempDir::new().unwrap();

        let math_source = r#"pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"#;

        let utils_source = r#"pub fn helper() {
    println!("Helper");
}

pub fn format_output(s: &str) -> String {
    format!("[{}]", s)
}
"#;

        fs::write(dir.path().join("math.rs"), math_source).unwrap();
        fs::write(dir.path().join("utils.rs"), utils_source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Batch read with different granularities
        let mut input = HashMap::new();
        input.insert(
            "reads".to_string(),
            serde_json::json!([
                {"path": "math.rs", "layer": "ast"},
                {"path": "utils.rs", "layer": "ast"},
                {"path": "math.rs", "symbol": "multiply"}
            ]),
        );

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should have AST view of math.rs
        assert!(content.contains("math.rs"));
        // Should have AST view of utils.rs
        assert!(content.contains("utils.rs"));
        // Should have raw code for multiply specifically
        assert!(content.contains("a * b"));
        // Should have format_output from utils
        assert!(content.contains("format_output"));
    }

    #[tokio::test]
    async fn test_lean_theory_graph_layer() {
        let dir = TempDir::new().unwrap();

        let source = r#"
import Mathlib.Data.Nat.Basic

namespace Demo

def helper (n : Nat) : Nat := Nat.succ n

theorem helper_gt (n : Nat) : helper n > n := by
  exact Nat.lt_succ_self n

theorem use_helper (n : Nat) : helper n > n := by
  exact helper_gt n
"#;

        fs::write(dir.path().join("Demo.lean"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("Demo.lean"));
        input.insert("layer".to_string(), serde_json::json!("theory_graph"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("Lean Theory Graph"));
        assert!(content.contains("helper_gt"));
        assert!(content.contains("use_helper"));
        assert!(content.contains("Local Dependency Edges"));
    }

    #[test]
    fn test_read_tree() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn main() {
    println!("Main");
}

fn helper() {
    println!("Helper");
}
"#;

        fs::write(dir.path().join("main.rs"), source).unwrap();

        let tool = SmartReadTool::new(dir.path());

        let requests = vec![
            ReadRequest::new("main.rs", CodeLayer::Ast),
            ReadRequest::new("main.rs", CodeLayer::Raw),
        ];

        let result = tool.read_tree(&requests);

        // Should have both views
        assert!(result.contains("[Ast]"));
        assert!(result.contains("[Raw]"));
        assert!(result.contains("main.rs"));
    }

    #[tokio::test]
    async fn test_read_folder() {
        let dir = TempDir::new().unwrap();

        let src = dir.path().join("src");
        fs::create_dir(&src).unwrap();

        let main_source = r#"fn main() {
    lib::greet();
}
"#;
        let lib_source = r#"pub fn greet() {
    println!("Hello");
}

pub fn farewell() {
    println!("Goodbye");
}
"#;

        fs::write(src.join("main.rs"), main_source).unwrap();
        fs::write(src.join("lib.rs"), lib_source).unwrap();
        fs::write(src.join("README.md"), "hello").unwrap();

        let tool = SmartReadTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("src"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        assert!(content.contains("📂 src"));
        assert!(content.contains("Mode: directory:summary"));
        assert!(content.contains("Entries:"));
        assert!(content.contains("Languages: Rust (2)"));
        assert!(content.contains("Likely important files"));
        assert!(content.contains("main.rs"));
        assert!(content.contains("lib.rs"));
    }

    #[test]
    fn test_read_folder_non_recursive() {
        let dir = TempDir::new().unwrap();

        let src = dir.path().join("src");
        let nested = src.join("nested");
        fs::create_dir_all(&nested).unwrap();

        fs::write(src.join("top.rs"), "fn top() {}\n").unwrap();
        fs::write(nested.join("deep.rs"), "fn deep() {}\n").unwrap();

        let tool = SmartReadTool::new(dir.path());
        let result = tool.read_folder("src", false).unwrap();

        assert!(result.contains("top.rs"));
        assert!(!result.contains("deep.rs"));
        assert!(result.contains("Depth limit: 1"));
    }


    #[tokio::test]
    async fn test_directory_layers_and_filters() {
        let dir = TempDir::new().unwrap();

        let src = dir.path().join("src");
        let docs = dir.path().join("docs");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&docs).unwrap();

        fs::write(src.join("lib.rs"), "pub struct Config {}\npub fn greet() {}\n").unwrap();
        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(docs.join("guide.md"), "hi\n").unwrap();

        let tool = SmartReadTool::new(dir.path());

        let mut structure = HashMap::new();
        structure.insert("path".to_string(), serde_json::json!("."));
        structure.insert("directory_layer".to_string(), serde_json::json!("structure"));
        structure.insert("max_depth".to_string(), serde_json::json!(1));

        let structure_result = tool.call(structure).await;
        assert!(!structure_result.is_error());
        let structure_content = structure_result.to_content_string();
        assert!(structure_content.contains("Directory structure"));
        assert!(structure_content.contains("src"));

        let mut semantic = HashMap::new();
        semantic.insert("path".to_string(), serde_json::json!("."));
        semantic.insert("directory_layer".to_string(), serde_json::json!("semantic"));
        semantic.insert("include".to_string(), serde_json::json!(["src/**"]));

        let semantic_result = tool.call(semantic).await;
        assert!(!semantic_result.is_error());
        let semantic_content = semantic_result.to_content_string();
        assert!(semantic_content.contains("Semantic map"));
        assert!(semantic_content.contains("crate root / public API boundary"));
        assert!(semantic_content.contains("notable: Config, greet") || semantic_content.contains("notable: greet, Config"));
        assert!(!semantic_content.contains("guide.md"));
    }

    #[tokio::test]
    async fn test_codebase_directory_layer_orientation() {
        let dir = TempDir::new().unwrap();

        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"orient\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/lib.rs"), "pub fn wave() {}\n").unwrap();

        let tool = SmartReadTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("codebase".to_string(), serde_json::json!(true));
        input.insert("directory_layer".to_string(), serde_json::json!("scalpel"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("Next inspection targets"));
        assert!(content.contains("read(path="));
    }

    #[tokio::test]
    async fn test_summary_uses_full_recursive_scope_for_language_mix_and_hotspots() {
        let dir = TempDir::new().unwrap();

        let src = dir.path().join("src");
        let lean = dir.path().join("lean/Gutoe");
        let crates_core = dir.path().join("crates/core/src");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&lean).unwrap();
        fs::create_dir_all(&crates_core).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();
        fs::write(crates_core.join("lib.rs"), "pub fn wave() {}\n").unwrap();
        fs::write(lean.join("Basic.lean"), "def wave : Nat := 1\n").unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/core\"]\n",
        )
        .unwrap();

        let tool = SmartReadTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("."));
        input.insert("directory_layer".to_string(), serde_json::json!("summary"));
        input.insert("max_depth".to_string(), serde_json::json!(1));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        assert!(content.contains("Visible entries:"));
        assert!(content.contains("Analyzed scope:"));
        assert!(content.contains("Languages: Rust (2), Lean (1)") || content.contains("Languages: Lean (1), Rust (2)"));
        assert!(content.contains("Cargo.toml"));
        assert!(content.contains("src/main.rs") || content.contains("main.rs"));
        assert!(content.contains("lean/Gutoe/Basic.lean"));
    }

    #[tokio::test]
    async fn test_read_codebase() {
        let dir = TempDir::new().unwrap();

        // Create a mock Rust workspace
        let cargo_toml = r#"[package]
name = "test-project"
version = "0.1.0"

[workspace]
members = [
    "crates/core",
    "crates/utils",
]
"#;
        fs::write(dir.path().join("Cargo.toml"), cargo_toml).unwrap();

        // Create crates
        let core_dir = dir.path().join("crates/core/src");
        let utils_dir = dir.path().join("crates/utils/src");
        fs::create_dir_all(&core_dir).unwrap();
        fs::create_dir_all(&utils_dir).unwrap();

        // Core lib.rs with doc comment
        let core_lib = r#"//! Core functionality for the project.
//! Provides the main types and logic.

pub struct Config {
    pub name: String,
}

pub fn init() -> Config {
    Config { name: "default".to_string() }
}
"#;
        fs::write(core_dir.join("lib.rs"), core_lib).unwrap();

        // Utils lib.rs
        let utils_lib = r#"//! Utility functions.

pub fn helper() {}
pub fn format_output(s: &str) -> String {
    format!("[{}]", s)
}
"#;
        fs::write(utils_dir.join("lib.rs"), utils_lib).unwrap();

        // Create CLAUDE.md for core
        let claude_md = r#"# Core Crate

**Purpose:** Main business logic and configuration.
"#;
        fs::create_dir_all(dir.path().join("crates/core")).unwrap();
        fs::write(dir.path().join("crates/core/CLAUDE.md"), claude_md).unwrap();

        let tool = SmartReadTool::new(dir.path());

        // Test codebase reading via tool interface
        let mut input = HashMap::new();
        input.insert("codebase".to_string(), serde_json::json!(true));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();

        // Should have project name
        assert!(content.contains("test-project"));

        // Should have workspace members
        assert!(content.contains("crates/core"));
        assert!(content.contains("crates/utils"));

        // Should have crate sections
        assert!(content.contains("### core"));
        assert!(content.contains("### utils"));

        // Should extract purpose from CLAUDE.md
        assert!(content.contains("Main business logic"));

        // Should have file tree
        assert!(content.contains("File Tree"));
    }

    #[test]
    fn test_read_codebase_direct() {
        let dir = TempDir::new().unwrap();

        // Simple single-crate project
        let cargo_toml = r#"[package]
name = "simple"
version = "0.1.0"
"#;
        fs::write(dir.path().join("Cargo.toml"), cargo_toml).unwrap();

        let src = dir.path().join("src");
        fs::create_dir(&src).unwrap();

        let lib_rs = r#"//! A simple library.

pub fn greet() -> &'static str {
    "Hello!"
}
"#;
        fs::write(src.join("lib.rs"), lib_rs).unwrap();

        let tool = SmartReadTool::new(dir.path());
        let result = tool.read_codebase().unwrap();

        // Should work for single-crate projects
        assert!(result.contains("simple") || result.contains("Codebase Structure"));
        assert!(result.contains("lib.rs") || result.contains("greet"));
    }

    /// Test on real Palace codebase - run with --ignored to execute.
    #[test]
    #[ignore]
    fn test_real_palace_codebase() {
        // Find the workspace root
        let mut path = std::env::current_dir().unwrap();
        while !path.join("Cargo.toml").exists() || !path.join("crates").is_dir() {
            path = path.parent().unwrap().to_path_buf();
        }

        let tool = SmartReadTool::new(&path);
        let result = tool.read_codebase().unwrap();

        println!("\n=== PALACE CODEBASE STRUCTURE ===\n");
        println!("{}", result);

        // Basic sanity checks
        assert!(result.contains("palace"));
        assert!(result.contains("director"));
        assert!(result.contains("llm-code-sdk"));
    }
}
