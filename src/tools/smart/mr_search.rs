//! MRSearch - Merge Request / Git History Search Tool
//!
//! Provides intelligent git history exploration:
//! - Zero-param mode: analyzes recent activity
//! - Hotphrase search: finds related commits and their relationships
//! - Shows commit clusters, affected files, and change context

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Command;

use async_trait::async_trait;

use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// A commit with its metadata.
#[derive(Debug, Clone)]
pub struct GitCommit {
    pub hash: String,
    pub short_hash: String,
    pub subject: String,
    pub author: String,
    pub date: String,
    pub files: Vec<String>,
    pub parent_hashes: Vec<String>,
}

/// A cluster of related commits.
#[derive(Debug, Clone)]
pub struct CommitCluster {
    pub commits: Vec<GitCommit>,
    pub files_affected: HashSet<String>,
    pub issue_refs: HashSet<String>,
}

/// MRSearch tool for intelligent git history exploration.
pub struct MRSearchTool {
    project_root: PathBuf,
}

impl MRSearchTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    /// Run a git command and return stdout.
    fn git(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| format!("Failed to run git: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    /// Parse a commit from git log output.
    fn parse_commit(&self, hash: &str) -> Option<GitCommit> {
        // Get commit info
        let format = "%H%n%h%n%s%n%an%n%ai%n%P";
        let info = self.git(&["log", "-1", &format!("--format={}", format), hash]).ok()?;
        let lines: Vec<&str> = info.trim().lines().collect();

        if lines.len() < 5 {
            return None;
        }

        // Get files changed
        let files_output = self.git(&["diff-tree", "--no-commit-id", "--name-only", "-r", hash]).ok()?;
        let files: Vec<String> = files_output.lines().map(|s| s.to_string()).collect();

        let parent_hashes: Vec<String> = if lines.len() > 5 {
            lines[5].split_whitespace().map(|s| s.to_string()).collect()
        } else {
            vec![]
        };

        Some(GitCommit {
            hash: lines[0].to_string(),
            short_hash: lines[1].to_string(),
            subject: lines[2].to_string(),
            author: lines[3].to_string(),
            date: lines[4].to_string(),
            files,
            parent_hashes,
        })
    }

    /// Search commits by query (message, issue number, etc).
    fn search_commits(&self, query: &str, limit: usize) -> Vec<GitCommit> {
        // Try multiple search strategies
        let mut hashes = HashSet::new();

        // 1. Search commit messages
        if let Ok(output) = self.git(&[
            "log", "--all", "--oneline",
            &format!("--grep={}", query),
            "-i", // case insensitive
            &format!("-{}", limit * 2)
        ]) {
            for line in output.lines() {
                if let Some(hash) = line.split_whitespace().next() {
                    hashes.insert(hash.to_string());
                }
            }
        }

        // 2. Search for issue numbers (e.g., #12345)
        if query.starts_with('#') || query.chars().all(|c| c.is_numeric()) {
            let issue_query = if query.starts_with('#') { query.to_string() } else { format!("#{}", query) };
            if let Ok(output) = self.git(&[
                "log", "--all", "--oneline",
                &format!("--grep={}", issue_query),
                &format!("-{}", limit * 2)
            ]) {
                for line in output.lines() {
                    if let Some(hash) = line.split_whitespace().next() {
                        hashes.insert(hash.to_string());
                    }
                }
            }
        }

        // Parse commits
        let mut commits: Vec<GitCommit> = hashes
            .iter()
            .filter_map(|h| self.parse_commit(h))
            .collect();

        // Sort by date (newest first)
        commits.sort_by(|a, b| b.date.cmp(&a.date));
        commits.truncate(limit);
        commits
    }

    /// Find commits that touch specific files.
    fn commits_touching_files(&self, files: &[String], limit: usize) -> Vec<GitCommit> {
        let mut hashes = HashSet::new();

        for file in files {
            if let Ok(output) = self.git(&[
                "log", "--all", "--oneline",
                &format!("-{}", limit),
                "--", file
            ]) {
                for line in output.lines() {
                    if let Some(hash) = line.split_whitespace().next() {
                        hashes.insert(hash.to_string());
                    }
                }
            }
        }

        hashes
            .iter()
            .filter_map(|h| self.parse_commit(h))
            .collect()
    }

    /// Find related commits (follow-ups, reverts, etc).
    fn find_related_commits(&self, commit: &GitCommit, all_commits: &[GitCommit]) -> Vec<String> {
        let mut related = Vec::new();

        // Find commits that mention this commit's hash
        for other in all_commits {
            if other.hash == commit.hash {
                continue;
            }

            // Check if other commit references this one
            let refs_this = other.subject.contains(&commit.short_hash)
                || other.subject.to_lowercase().contains("revert")
                    && other.files.iter().any(|f| commit.files.contains(f));

            if refs_this {
                related.push(other.short_hash.clone());
            }
        }

        // Find commits touching same files around same time
        for other in all_commits {
            if other.hash == commit.hash || related.contains(&other.short_hash) {
                continue;
            }

            let shared_files: Vec<_> = other.files.iter()
                .filter(|f| commit.files.contains(f))
                .collect();

            if shared_files.len() >= commit.files.len() / 2 && !commit.files.is_empty() {
                related.push(other.short_hash.clone());
            }
        }

        related
    }

    /// Build a cluster of related commits from a search.
    fn build_cluster(&self, query: &str, limit: usize) -> CommitCluster {
        let commits = self.search_commits(query, limit);

        let mut files_affected = HashSet::new();
        let mut issue_refs = HashSet::new();

        for commit in &commits {
            for file in &commit.files {
                files_affected.insert(file.clone());
            }

            // Extract issue references from subject
            for word in commit.subject.split_whitespace() {
                if word.starts_with('#') && word.len() > 1 {
                    issue_refs.insert(word.to_string());
                }
                // Also catch "Fixed #NNN" patterns
                if word.chars().all(|c| c.is_numeric()) && word.len() >= 4 {
                    issue_refs.insert(format!("#{}", word));
                }
            }
        }

        // Expand with commits touching the same files
        if !files_affected.is_empty() && commits.len() < limit {
            let file_list: Vec<String> = files_affected.iter().cloned().collect();
            let file_commits = self.commits_touching_files(&file_list, limit - commits.len());

            let mut all_commits = commits.clone();
            let existing_hashes: HashSet<_> = commits.iter().map(|c| &c.hash).collect();

            for fc in file_commits {
                if !existing_hashes.contains(&fc.hash) {
                    for file in &fc.files {
                        files_affected.insert(file.clone());
                    }
                    all_commits.push(fc);
                }
            }

            return CommitCluster {
                commits: all_commits,
                files_affected,
                issue_refs,
            };
        }

        CommitCluster {
            commits,
            files_affected,
            issue_refs,
        }
    }

    /// Get recent activity summary (zero-param mode).
    fn recent_activity(&self, days: usize) -> String {
        let mut output = String::new();

        // Recent commits
        let since = format!("--since={} days ago", days);
        if let Ok(log) = self.git(&["log", "--all", "--oneline", &since, "-30"]) {
            let lines: Vec<&str> = log.lines().collect();
            output.push_str(&format!("## Recent Commits ({} in last {} days)\n\n", lines.len(), days));
            for line in lines.iter().take(15) {
                output.push_str(&format!("- {}\n", line));
            }
            if lines.len() > 15 {
                output.push_str(&format!("... and {} more\n", lines.len() - 15));
            }
            output.push('\n');
        }

        // Hot files (most changed recently)
        if let Ok(shortlog) = self.git(&["log", "--all", &since, "--name-only", "--format="]) {
            let mut file_counts: HashMap<String, usize> = HashMap::new();
            for line in shortlog.lines() {
                let line = line.trim();
                if !line.is_empty() {
                    *file_counts.entry(line.to_string()).or_insert(0) += 1;
                }
            }

            let mut hot_files: Vec<_> = file_counts.into_iter().collect();
            hot_files.sort_by(|a, b| b.1.cmp(&a.1));

            if !hot_files.is_empty() {
                output.push_str("## Hot Files (most changed)\n\n");
                for (file, count) in hot_files.iter().take(10) {
                    output.push_str(&format!("- {} ({} commits)\n", file, count));
                }
                output.push('\n');
            }
        }

        // Active branches
        if let Ok(branches) = self.git(&["branch", "-a", "--sort=-committerdate"]) {
            let lines: Vec<&str> = branches.lines().take(10).collect();
            if !lines.is_empty() {
                output.push_str("## Active Branches\n\n");
                for line in lines {
                    output.push_str(&format!("{}\n", line.trim()));
                }
            }
        }

        output
    }

    /// Format cluster for output.
    fn format_cluster(&self, cluster: &CommitCluster, query: &str) -> String {
        let mut output = String::new();

        output.push_str(&format!("# MRSearch: \"{}\"\n\n", query));

        // Commits
        output.push_str(&format!("## Commits ({} found)\n\n", cluster.commits.len()));
        for commit in &cluster.commits {
            let related = self.find_related_commits(commit, &cluster.commits);
            output.push_str(&format!("### {} {}\n", commit.short_hash, commit.subject));
            output.push_str(&format!("Author: {} | Date: {}\n", commit.author, &commit.date[..10]));

            if !commit.files.is_empty() {
                output.push_str(&format!("Files: {}\n", commit.files.join(", ")));
            }

            if !related.is_empty() {
                output.push_str(&format!("Related: {}\n", related.join(", ")));
            }
            output.push('\n');
        }

        // Files affected
        if !cluster.files_affected.is_empty() {
            output.push_str("## Files Affected\n\n");
            let mut files: Vec<_> = cluster.files_affected.iter().collect();
            files.sort();
            for file in files.iter().take(20) {
                output.push_str(&format!("- {}\n", file));
            }
            if files.len() > 20 {
                output.push_str(&format!("... and {} more\n", files.len() - 20));
            }
            output.push('\n');
        }

        // Issue references
        if !cluster.issue_refs.is_empty() {
            output.push_str("## Related Issues\n\n");
            for issue in &cluster.issue_refs {
                output.push_str(&format!("- {}\n", issue));
            }
        }

        output
    }
}

#[async_trait]
impl Tool for MRSearchTool {
    fn name(&self) -> &str {
        "mr_search"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "mr_search",
            InputSchema::object()
                .optional_string("query", "Search query: issue number (#12345), phrase, or commit message. Empty = recent activity")
                .optional_string("limit", "Max commits to return (default: 20)")
                .optional_string("days", "For recent activity, how many days back (default: 7)"),
        )
        .with_description(
            "Search git history intelligently. Shows related commits, affected files, and change context. \
             Call with no params for recent activity summary, or search by issue number/phrase.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        let limit = input
            .get("limit")
            .and_then(|v| v.as_str().or_else(|| v.as_u64().map(|_| "")))
            .and_then(|s| if s.is_empty() { input.get("limit").and_then(|v| v.as_u64()) } else { s.parse().ok() })
            .unwrap_or(20) as usize;

        let days = input
            .get("days")
            .and_then(|v| v.as_str().or_else(|| v.as_u64().map(|_| "")))
            .and_then(|s| if s.is_empty() { input.get("days").and_then(|v| v.as_u64()) } else { s.parse().ok() })
            .unwrap_or(7) as usize;

        // Verify we're in a git repo
        if self.git(&["rev-parse", "--git-dir"]).is_err() {
            return ToolResult::error("Not a git repository");
        }

        if query.is_empty() {
            // Zero-param mode: show recent activity
            let output = self.recent_activity(days);
            if output.is_empty() {
                ToolResult::success("No recent git activity found")
            } else {
                ToolResult::success(output)
            }
        } else {
            // Search mode
            let cluster = self.build_cluster(query, limit);

            if cluster.commits.is_empty() {
                ToolResult::success(format!("No commits found matching \"{}\"", query))
            } else {
                ToolResult::success(self.format_cluster(&cluster, query))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_git_repo() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Init repo
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Configure git
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Create initial commit
        fs::write(dir.path().join("README.md"), "# Test").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Add another commit with issue reference
        fs::write(dir.path().join("fix.rs"), "fn fix() {}").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Fixed #12345 -- Handle edge case"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        dir
    }

    #[tokio::test]
    async fn test_mr_search_recent_activity() {
        let dir = setup_git_repo();
        let tool = MRSearchTool::new(dir.path());

        let input = HashMap::new();
        let result = tool.call(input).await;

        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("Recent Commits"));
    }

    #[tokio::test]
    async fn test_mr_search_by_issue() {
        let dir = setup_git_repo();
        let tool = MRSearchTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("query".to_string(), serde_json::json!("#12345"));

        let result = tool.call(input).await;

        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("12345"));
        assert!(content.contains("fix.rs"));
    }

    #[tokio::test]
    async fn test_mr_search_by_phrase() {
        let dir = setup_git_repo();
        let tool = MRSearchTool::new(dir.path());

        let mut input = HashMap::new();
        input.insert("query".to_string(), serde_json::json!("edge case"));

        let result = tool.call(input).await;

        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("edge case") || content.contains("Handle"));
    }
}
