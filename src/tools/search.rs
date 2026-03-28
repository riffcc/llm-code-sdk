//! Full-text search tool using MiniRust Search.
//!
//! Builds the index once on first search, keeps it for the session.
//! Files are incrementally re-indexed when their mtime changes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use async_trait::async_trait;
use minirust_search::{Document, MiniSearch, MiniSearchOptions, SearchOptions};

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// A document representing a code file.
#[derive(Debug, Clone)]
pub struct CodeDocument {
    pub id: String,
    pub path: String,
    pub content: String,
    pub kind: String,
}

impl Document for CodeDocument {
    fn id(&self) -> &str {
        &self.id
    }

    fn field(&self, name: &str) -> Option<&str> {
        match name {
            "id" => Some(&self.id),
            "path" => Some(&self.path),
            "content" => Some(&self.content),
            "kind" => Some(&self.kind),
            _ => None,
        }
    }
}

/// Tracked file state for incremental re-indexing.
struct TrackedFile {
    mtime: SystemTime,
}

/// Search tool with a persistent, incrementally-updated index.
pub struct SearchTool {
    index: Arc<RwLock<MiniSearch>>,
    project_root: PathBuf,
    /// mtime tracking for incremental updates.
    tracked: Arc<RwLock<HashMap<String, TrackedFile>>>,
    /// Whether the initial index has been built.
    initialized: Arc<RwLock<bool>>,
}

impl SearchTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        let options = MiniSearchOptions::new(&["path", "content", "kind"]);
        Self {
            index: Arc::new(RwLock::new(MiniSearch::new(options))),
            project_root: project_root.into(),
            tracked: Arc::new(RwLock::new(HashMap::new())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Create with a pre-built index.
    pub fn with_index(project_root: impl Into<PathBuf>, index: MiniSearch) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            project_root: project_root.into(),
            tracked: Arc::new(RwLock::new(HashMap::new())),
            initialized: Arc::new(RwLock::new(true)),
        }
    }

    /// Build or refresh the index. On first call, indexes the whole repo.
    /// On subsequent calls, only re-indexes files whose mtime changed.
    fn ensure_index(&self) {
        let mut init = self.initialized.write().unwrap();
        if !*init {
            self.build_full_index();
            *init = true;
        } else {
            self.refresh_changed();
        }
    }

    fn build_full_index(&self) {
        let mut index = self.index.write().unwrap();
        let mut tracked = self.tracked.write().unwrap();
        Self::index_dir_recursive(&self.project_root, &self.project_root, &mut index, &mut tracked);
    }

    fn refresh_changed(&self) {
        let mut index = self.index.write().unwrap();
        let mut tracked = self.tracked.write().unwrap();

        // Check tracked files for mtime changes
        let stale: Vec<String> = tracked.iter()
            .filter(|(rel_path, state)| {
                let full = self.project_root.join(rel_path);
                match std::fs::metadata(&full).and_then(|m| m.modified()) {
                    Ok(mtime) => mtime != state.mtime,
                    Err(_) => true, // file deleted
                }
            })
            .map(|(path, _)| path.clone())
            .collect();

        for rel_path in &stale {
            let full = self.project_root.join(rel_path);
            // Discard old version (clean removal via reverse index)
            if index.has(rel_path) {
                index.discard(rel_path);
            }
            tracked.remove(rel_path);

            // Re-index if file still exists
            if let Ok(content) = std::fs::read_to_string(&full) {
                if let Ok(meta) = std::fs::metadata(&full) {
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    let doc = CodeDocument {
                        id: rel_path.clone(),
                        path: rel_path.clone(),
                        content,
                        kind: "file".to_string(),
                    };
                    index.add(doc);
                    tracked.insert(rel_path.clone(), TrackedFile { mtime });
                }
            }
        }
    }

    fn index_dir_recursive(
        root: &Path,
        dir: &Path,
        index: &mut MiniSearch,
        tracked: &mut HashMap<String, TrackedFile>,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with('.') || name_str == "node_modules" || name_str == "target" {
                continue;
            }

            if path.is_dir() {
                Self::index_dir_recursive(root, &path, index, tracked);
            } else if is_text_file(&path) {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let rel = path.strip_prefix(root).unwrap_or(&path)
                        .to_string_lossy().to_string();
                    let mtime = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH);

                    let doc = CodeDocument {
                        id: rel.clone(),
                        path: rel.clone(),
                        content,
                        kind: "file".to_string(),
                    };
                    index.add(doc);
                    tracked.insert(rel, TrackedFile { mtime });
                }
            }
        }
    }

    /// Manually index a file (for callers that know a file changed).
    pub fn index_file(&self, path: &Path, content: &str) {
        let rel = path.strip_prefix(&self.project_root).unwrap_or(path)
            .to_string_lossy().to_string();

        let mut index = self.index.write().unwrap();
        if index.has(&rel) {
            index.discard(&rel);
        }
        let doc = CodeDocument {
            id: rel.clone(),
            path: rel.clone(),
            content: content.to_string(),
            kind: "file".to_string(),
        };
        index.add(doc);

        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.tracked.write().unwrap().insert(rel, TrackedFile { mtime });
    }

    /// Search the index.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        self.ensure_index();
        let index = self.index.read().unwrap();
        let mut opts = SearchOptions::new();
        opts.prefix = true;
        index.search(query, Some(opts))
            .into_iter()
            .take(limit)
            .map(|r| SearchResult {
                path: r.id,
                score: r.score,
                matched_terms: r.terms,
            })
            .collect()
    }

    pub fn document_count(&self) -> usize {
        self.index.read().unwrap().document_count()
    }

    /// Index all files in a directory (for explicit callers).
    pub fn index_directory(&self, dir: &Path) -> usize {
        let mut index = self.index.write().unwrap();
        let mut tracked = self.tracked.write().unwrap();
        let count_before = index.document_count();
        Self::index_dir_recursive(&self.project_root, dir, &mut index, &mut tracked);
        index.document_count() - count_before
    }
}

/// A search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub path: String,
    pub score: f64,
    pub matched_terms: Vec<String>,
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "search",
            InputSchema::object()
                .required_string("query", "Search query (supports prefix matching)")
                .optional_string("limit", "Max results to return (default: 10)"),
        )
        .with_description(
            "Search the codebase for files matching a query. Returns paths ranked by relevance.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            return ToolResult::error("query is required");
        }

        let limit = input
            .get("limit")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(10usize);

        let results = self.search(query, limit);

        if results.is_empty() {
            ToolResult::success("No matches found")
        } else {
            let output: Vec<String> = results
                .iter()
                .map(|r| format!("{} (score: {:.2})", r.path, r.score))
                .collect();
            ToolResult::success(output.join("\n"))
        }
    }
}

fn is_text_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "rs" | "py"
            | "js" | "ts" | "tsx" | "jsx"
            | "go" | "c" | "cpp" | "h" | "hpp"
            | "java" | "rb" | "sh"
            | "md" | "txt" | "toml" | "yaml" | "yml"
            | "json" | "html" | "css" | "scss"
            | "vue" | "svelte"
            | "sql" | "graphql" | "proto" | "xml"
            | "env" | "conf" | "cfg" | "ini"
            | "lean" | "nim" | "pl" | "pm"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_search_tool() {
        let dir = TempDir::new().unwrap();

        fs::write(dir.path().join("auth.rs"), "fn authenticate(user: &str) {}").unwrap();
        fs::write(dir.path().join("user.rs"), "struct User { name: String }").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() { authenticate(\"test\"); }").unwrap();

        let tool = SearchTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("query".to_string(), serde_json::json!("authenticate"));
        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("auth.rs") || content.contains("main.rs"));
    }

    #[test]
    fn test_incremental_update() {
        let dir = TempDir::new().unwrap();

        fs::write(dir.path().join("a.rs"), "fn original() {}").unwrap();

        let tool = SearchTool::new(dir.path());
        tool.ensure_index();
        assert_eq!(tool.document_count(), 1);

        // Modify the file
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(dir.path().join("a.rs"), "fn replacement() {}").unwrap();

        // Add a new file
        fs::write(dir.path().join("b.rs"), "fn new_file() {}").unwrap();

        // Refresh picks up the change
        tool.ensure_index();

        let results = tool.search("replacement", 10);
        assert!(!results.is_empty());

        let results = tool.search("original", 10);
        assert!(results.is_empty());
    }
}
