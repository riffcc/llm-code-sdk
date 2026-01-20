//! MRS-based semantic search tool.
//!
//! Provides full-text search over indexed content using MiniRust Search.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use minirust_search::{MiniSearch, MiniSearchOptions, SearchOptions, Document};

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// A document representing a code file or chunk.
#[derive(Debug, Clone)]
pub struct CodeDocument {
    pub id: String,
    pub path: String,
    pub content: String,
    pub kind: String, // "file", "function", "class", etc.
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

/// Tool for searching indexed code using MRS.
pub struct SearchTool {
    index: Arc<RwLock<MiniSearch>>,
    project_root: PathBuf,
}

impl SearchTool {
    /// Create a new search tool with an empty index.
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        let options = MiniSearchOptions::new(&["path", "content", "kind"]);
        Self {
            index: Arc::new(RwLock::new(MiniSearch::new(options))),
            project_root: project_root.into(),
        }
    }

    /// Create a search tool with a pre-built index.
    pub fn with_index(project_root: impl Into<PathBuf>, index: MiniSearch) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            project_root: project_root.into(),
        }
    }

    /// Index a file.
    pub fn index_file(&self, path: &Path, content: &str) {
        let relative = path
            .strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let doc = CodeDocument {
            id: relative.clone(),
            path: relative,
            content: content.to_string(),
            kind: "file".to_string(),
        };

        let mut index = self.index.write().unwrap();
        if index.has(&doc.id) {
            index.replace(doc);
        } else {
            index.add(doc);
        }
    }

    /// Index all text files in a directory.
    pub fn index_directory(&self, dir: &Path) -> usize {
        let mut count = 0;
        self.index_directory_recursive(dir, &mut count);
        count
    }

    fn index_directory_recursive(&self, dir: &Path, count: &mut usize) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            // Skip hidden, node_modules, target
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }

            if path.is_dir() {
                self.index_directory_recursive(&path, count);
            } else if is_text_file(&path) {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    self.index_file(&path, &content);
                    *count += 1;
                }
            }
        }
    }

    /// Search the index.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let index = self.index.read().unwrap();
        let mut opts = SearchOptions::new();
        opts.prefix = true;

        index
            .search(query, Some(opts))
            .into_iter()
            .take(limit)
            .map(|r| SearchResult {
                path: r.id,
                score: r.score,
                matched_terms: r.terms,
            })
            .collect()
    }

    /// Get document count.
    pub fn document_count(&self) -> usize {
        self.index.read().unwrap().document_count()
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
        .with_description("Search the codebase for files matching a query. Returns paths ranked by relevance.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if query.is_empty() {
            return ToolResult::error("query is required");
        }

        let limit = input
            .get("limit")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or(10usize);

        // Index on demand if empty
        if self.document_count() == 0 {
            let count = self.index_directory(&self.project_root);
            if count == 0 {
                return ToolResult::error("No files to search");
            }
        }

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
        "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp"
            | "java" | "rb" | "sh" | "md" | "txt" | "toml" | "yaml" | "yml" | "json"
            | "html" | "css" | "scss" | "vue" | "svelte" | "sql" | "graphql" | "proto"
            | "xml" | "env" | "conf" | "cfg" | "ini"
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

        // Create some test files
        fs::write(dir.path().join("auth.rs"), "fn authenticate(user: &str) {}").unwrap();
        fs::write(dir.path().join("user.rs"), "struct User { name: String }").unwrap();
        fs::write(dir.path().join("main.rs"), "fn main() { authenticate(\"test\"); }").unwrap();

        let tool = SearchTool::new(dir.path());

        // Index
        let count = tool.index_directory(dir.path());
        assert_eq!(count, 3);

        // Search
        let results = tool.search("authenticate", 10);
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.path.contains("auth")));
    }
}
