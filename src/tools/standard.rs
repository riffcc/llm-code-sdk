//! Standard tools for agentic exploration and code editing.
//!
//! These tools mirror the capabilities of Anthropic's builtin tools:
//! - [`BashTool`] - Execute shell commands
//! - [`ReadFileTool`] - Read file contents
//! - [`WriteFileTool`] - Write/create files
//! - [`EditFileTool`] - Edit files with str_replace
//! - [`GlobTool`] - Find files by glob pattern
//! - [`GrepTool`] - Search file contents
//! - [`ListDirectoryTool`] - List directory contents

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex as TokioMutex;

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

fn canonical_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn is_within_project_root(path: &Path, project_root: &Path) -> bool {
    let root = canonical_or_original(project_root);

    if let Ok(canonical) = path.canonicalize() {
        return canonical.starts_with(&root);
    }

    if let Some(parent) = path.parent() {
        let canonical_parent = canonical_or_original(parent);
        return canonical_parent.starts_with(&root);
    }

    false
}

#[cfg(feature = "smart")]
use super::smart::{AskCodeTool, MRSearchTool, SmartReadTool, SmartWriteTool};

#[cfg(feature = "search")]
use super::SearchTool;

/// A persistent interactive shell session.
///
/// Instead of spawning a new process per command, this keeps a single bash
/// process alive across tool calls. The model can:
/// - Run commands via `command`
/// - Send stdin input via `stdin` (for interactive prompts, passwords, etc.)
/// - Set a per-command timeout via `timeout`
///
/// The shell is started in a new session (`setsid`) so child processes cannot
/// open `/dev/tty` and hang waiting for terminal input.
pub struct BashTool {
    working_dir: PathBuf,
    /// Default timeout in seconds per command.
    default_timeout_secs: u64,
    /// The persistent shell session (lazily initialized).
    session: Arc<TokioMutex<Option<ShellSession>>>,
}

/// Sentinel that marks the end of a command's output.
const SENTINEL: &str = "__BASH_TOOL_SENTINEL_a7f3b2e1__";

/// Internal state for the persistent shell process.
struct ShellSession {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
}

impl ShellSession {
    async fn spawn(working_dir: &Path) -> Result<Self, String> {
        // Spawn bash with merged stdout+stderr, detached from controlling TTY.
        // We use `setsid` (via pre_exec) so that child processes cannot open /dev/tty.
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("--noediting")
            .arg("--noprofile")
            .arg("--norc")
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "")
            .env("SSH_ASKPASS", "")
            .env("DEBIAN_FRONTEND", "noninteractive");

        // Detach from controlling terminal on Unix
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    // Create a new session so children can't access /dev/tty
                    libc::setsid();
                    Ok(())
                });
            }
        }

        // Redirect stderr to stdout so we get interleaved output
        cmd.arg("-c")
            .args([] as [&str; 0]); // clear previous args

        // Actually, we want an interactive-ish shell that reads commands from stdin.
        // Reset: don't use -c, just launch bash that reads from stdin.
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("--noediting")
            .arg("--noprofile")
            .arg("--norc")
            .current_dir(working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "")
            .env("SSH_ASKPASS", "")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .env("PS1", "")
            .env("PS2", "");

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }
        }

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn bash: {e}"))?;

        let stdin = child.stdin.take().ok_or("Failed to capture stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
        let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;

        // Merge stderr into the output stream by spawning a reader that
        // forwards stderr lines to a shared buffer. We'll handle this by
        // redirecting stderr to stdout at the shell level instead.
        // For now, just drop stderr and use shell-level redirection (2>&1).
        drop(stderr);

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// Run a command, collect output until the sentinel, with timeout.
    async fn execute(&mut self, command: &str, timeout_secs: u64) -> Result<(String, i32), String> {
        // Wrap in a subshell so set -e, exit, etc. can't kill our sentinel.
        // The subshell captures the exit code; the sentinel always runs.
        let full_cmd = format!(
            "( {command} ) 2>&1\necho \"{SENTINEL}$?\"\n",
        );
        self.stdin
            .write_all(full_cmd.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to shell stdin: {e}"))?;
        self.stdin.flush().await.ok();

        // Read lines until we see the sentinel
        let mut output = String::new();
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout_secs);

        loop {
            let mut line = String::new();
            let read_result = tokio::time::timeout_at(
                deadline,
                self.stdout.read_line(&mut line),
            ).await;

            match read_result {
                Ok(Ok(0)) => {
                    // EOF — shell exited
                    return Ok((output, 1));
                }
                Ok(Ok(_)) => {
                    if line.starts_with(SENTINEL) {
                        // Parse exit code from sentinel
                        let code_str = line.trim().strip_prefix(SENTINEL).unwrap_or("1");
                        let code = code_str.parse::<i32>().unwrap_or(1);
                        return Ok((output, code));
                    }
                    output.push_str(&line);
                }
                Ok(Err(e)) => {
                    return Err(format!("Error reading shell output: {e}"));
                }
                Err(_) => {
                    // Timeout — return what we have
                    output.push_str("\n(command timed out)");
                    return Ok((output, 124)); // 124 = timeout exit code
                }
            }
        }
    }

    /// Send raw input to the shell's stdin (for interactive prompts).
    async fn send_input(&mut self, input: &str) -> Result<(), String> {
        self.stdin
            .write_all(input.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to shell stdin: {e}"))?;
        // Add newline if not present
        if !input.ends_with('\n') {
            self.stdin.write_all(b"\n").await.ok();
        }
        self.stdin.flush().await.ok();
        Ok(())
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl BashTool {
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
            default_timeout_secs: 120,
            session: Arc::new(TokioMutex::new(None)),
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    /// Get or create the shell session.
    async fn get_session(&self) -> Result<(), String> {
        let mut guard = self.session.lock().await;
        if guard.as_mut().map(|s| !s.is_alive()).unwrap_or(true) {
            *guard = Some(ShellSession::spawn(&self.working_dir).await?);
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "bash",
            InputSchema::object()
                .required_string("command", "The bash command to execute")
                .optional_string("stdin", "Input to send to the command's stdin (for interactive prompts)")
                .optional_string("timeout", "Timeout in seconds (default: 120)"),
        )
        .with_description(
            "Execute a bash command in a persistent shell session. \
             The shell persists across calls so cd, env vars, etc. carry over. \
             If a command prompts for input, use the stdin parameter to respond. \
             Commands that would open an interactive editor (vim, nano) will hang — avoid them."
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let stdin_input = input.get("stdin").and_then(|v| v.as_str());
        let timeout = input
            .get("timeout")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| input.get("timeout").and_then(|v| v.as_u64()))
            .unwrap_or(self.default_timeout_secs);

        if command.is_empty() && stdin_input.is_none() {
            return ToolResult::error("command is required");
        }

        // Ensure session is alive
        if let Err(e) = self.get_session().await {
            return ToolResult::error(e);
        }

        let mut guard = self.session.lock().await;
        let session = guard.as_mut().unwrap();

        // If there's stdin to send before/with the command
        if let Some(input_text) = stdin_input {
            if let Err(e) = session.send_input(input_text).await {
                return ToolResult::error(e);
            }
            // If no command, just sending stdin input
            if command.is_empty() {
                return ToolResult::success("Input sent");
            }
        }

        // Execute the command
        match session.execute(command, timeout).await {
            Ok((output, exit_code)) => {
                let trimmed = output.trim_end();
                if exit_code == 0 {
                    ToolResult::success(trimmed.to_string())
                } else {
                    ToolResult::error(format!(
                        "Command failed (exit code {exit_code})\n{trimmed}"
                    ))
                }
            }
            Err(e) => {
                // Session is likely dead — clear it so next call respawns
                *guard = None;
                ToolResult::error(e)
            }
        }
    }
}

/// Tool for reading file contents.
pub struct ReadFileTool {
    project_root: PathBuf,
}

impl ReadFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "read_file",
            InputSchema::object().required_string("path", "File path to read"),
        )
        .with_description("Read the contents of a file.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        // Reject requests to read from .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        match std::fs::read_to_string(&full_path) {
            Ok(content) => {
                // Truncate very large files
                if content.len() > 100_000 {
                    let truncated: String = content.chars().take(100_000).collect();
                    ToolResult::success(format!(
                        "{}\\n\\n... (truncated, {} bytes total)",
                        truncated,
                        content.len()
                    ))
                } else {
                    ToolResult::success(content)
                }
            }
            Err(e) => ToolResult::error(format!("Failed to read file: {}", e)),
        }
    }
}

/// Tool for writing/creating files.
pub struct WriteFileTool {
    project_root: PathBuf,
}

impl WriteFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "write_file",
            InputSchema::object()
                .required_string("path", "File path to write")
                .required_string("content", "Content to write to the file"),
        )
        .with_description("Write content to a file. Creates the file if it doesn't exist.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("Failed to create directories: {}", e));
            }
        }

        match std::fs::write(&full_path, content) {
            Ok(_) => ToolResult::success(format!("Wrote {} bytes to {}", content.len(), path)),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}

/// Tool for editing files with str_replace.
pub struct EditFileTool {
    project_root: PathBuf,
}

impl EditFileTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "edit_file",
            InputSchema::object()
                .required_string("path", "File path to edit")
                .required_string("old_string", "The exact string to replace")
                .required_string("new_string", "The replacement string"),
        )
        .with_description(
            "Edit a file by replacing an exact string match. The old_string must match exactly.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if path.is_empty() {
            return ToolResult::error("path is required");
        }
        if old_string.is_empty() {
            return ToolResult::error("old_string is required");
        }

        let full_path = self.resolve_path(path);

        // Security: ensure path is within project root
        if !is_within_project_root(&full_path, &self.project_root) {
            return ToolResult::error("Path must be within project root");
        }

        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        // Count occurrences
        let count = content.matches(old_string).count();
        if count == 0 {
            return ToolResult::error("old_string not found in file");
        }
        if count > 1 {
            return ToolResult::error(format!(
                "old_string found {} times - must be unique. Provide more context.",
                count
            ));
        }

        let new_content = content.replace(old_string, new_string);

        match std::fs::write(&full_path, &new_content) {
            Ok(_) => ToolResult::success(format!("Successfully edited {}", path)),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}

/// Tool for finding files by glob pattern.
pub struct GlobTool {
    project_root: PathBuf,
}

impl GlobTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "glob",
            InputSchema::object()
                .required_string("pattern", "Glob pattern (e.g., '**/*.rs', 'src/**/*.ts')"),
        )
        .with_description("Find files matching a glob pattern.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");

        if pattern.is_empty() {
            return ToolResult::error("pattern is required");
        }

        let full_pattern = self.project_root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy();

        match glob::glob(&pattern_str) {
            Ok(paths) => {
                let mut results: Vec<String> = paths
                    .flatten()
                    .filter_map(|p| {
                        p.strip_prefix(&self.project_root)
                            .ok()
                            .map(|rel| rel.to_string_lossy().to_string())
                    })
                    .take(100)
                    .collect();
                results.sort();

                if results.is_empty() {
                    ToolResult::success("No files matched the pattern")
                } else {
                    ToolResult::success(results.join("\n"))
                }
            }
            Err(e) => ToolResult::error(format!("Invalid glob pattern: {}", e)),
        }
    }
}

/// Tool for searching file contents.
pub struct GrepTool {
    project_root: PathBuf,
}

impl GrepTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn search_file(
        &self,
        path: &Path,
        pattern: &str,
        results: &mut Vec<String>,
        max_results: usize,
    ) {
        if results.len() >= max_results {
            return;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };

        let relative = path
            .strip_prefix(&self.project_root)
            .unwrap_or(path)
            .to_string_lossy();

        for (i, line) in content.lines().enumerate() {
            if results.len() >= max_results {
                break;
            }
            if line.to_lowercase().contains(&pattern.to_lowercase()) {
                results.push(format!("{}:{}: {}", relative, i + 1, line.trim()));
            }
        }
    }

    fn search_dir(&self, dir: &Path, pattern: &str, results: &mut Vec<String>, max_results: usize) {
        if results.len() >= max_results {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            if results.len() >= max_results {
                break;
            }

            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            // Skip hidden, node_modules, target
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }

            if path.is_dir() {
                self.search_dir(&path, pattern, results, max_results);
            } else if is_text_file(&path) {
                self.search_file(&path, pattern, results, max_results);
            }
        }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "grep",
            InputSchema::object()
                .required_string("pattern", "Search pattern (case-insensitive)")
                .optional_string(
                    "path",
                    "Directory or file to search (defaults to project root)",
                ),
        )
        .with_description(
            "Search for text in files. Returns matching lines with file:line: prefix.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");

        if pattern.is_empty() {
            return ToolResult::error("pattern is required");
        }

        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        // Reject requests to search in .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let search_path = if path == "." {
            self.project_root.clone()
        } else {
            self.project_root.join(path)
        };

        let mut results = Vec::new();
        let max_results = 50;

        if search_path.is_file() {
            self.search_file(&search_path, pattern, &mut results, max_results);
        } else if search_path.is_dir() {
            self.search_dir(&search_path, pattern, &mut results, max_results);
        } else {
            return ToolResult::error(format!("Path not found: {}", path));
        }

        if results.is_empty() {
            ToolResult::success("No matches found")
        } else {
            let truncated = if results.len() >= max_results {
                "\n... (results truncated)"
            } else {
                ""
            };
            ToolResult::success(format!("{}{}", results.join("\n"), truncated))
        }
    }
}

/// Tool for listing directory contents.
pub struct ListDirectoryTool {
    project_root: PathBuf,
}

impl ListDirectoryTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "list_directory",
            InputSchema::object()
                .optional_string("path", "Directory path (defaults to project root)"),
        )
        .with_description("List files and directories. Directories end with /")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        // Reject requests to list .palace directory
        if path.contains(".palace") {
            return ToolResult::error("Cannot access .palace directory");
        }

        let full_path = self.resolve_path(path);

        match std::fs::read_dir(&full_path) {
            Ok(entries) => {
                let mut items: Vec<String> = entries
                    .flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        // Skip hidden files and common noise
                        if name.starts_with('.') || name == "node_modules" || name == "target" {
                            return None;
                        }
                        if e.path().is_dir() {
                            Some(format!("{}/", name))
                        } else {
                            Some(name)
                        }
                    })
                    .collect();
                items.sort();
                ToolResult::success(items.join("\n"))
            }
            Err(e) => ToolResult::error(format!("Failed to list directory: {}", e)),
        }
    }
}

/// Check if a file is likely a text file based on extension.
fn is_text_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "java"
            | "rb"
            | "sh"
            | "md"
            | "txt"
            | "toml"
            | "yaml"
            | "yml"
            | "json"
            | "html"
            | "css"
            | "scss"
            | "vue"
            | "svelte"
            | "sql"
            | "graphql"
            | "proto"
            | "xml"
            | "env"
            | "conf"
            | "cfg"
            | "ini"
            | "lock"
            | "sum"
            | "lean"
    )
}

/// Create standard exploration tools for a project.
pub fn create_exploration_tools(project_root: &Path) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(ListDirectoryTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GlobTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GrepTool::new(project_root)) as Arc<dyn Tool>,
    ];

    #[cfg(feature = "search")]
    tools.push(Arc::new(SearchTool::new(project_root)) as Arc<dyn Tool>);

    #[cfg(feature = "smart")]
    {
        tools.push(Arc::new(SmartReadTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(MRSearchTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(AskCodeTool::new(project_root)) as Arc<dyn Tool>);
    }

    tools
}

/// Create standard editing tools for a project.
pub fn create_editing_tools(project_root: &Path) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(WriteFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(EditFileTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(ListDirectoryTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GlobTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(GrepTool::new(project_root)) as Arc<dyn Tool>,
        Arc::new(BashTool::new(project_root)) as Arc<dyn Tool>,
    ];

    #[cfg(feature = "smart")]
    {
        tools.push(Arc::new(SmartReadTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(SmartWriteTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(MRSearchTool::new(project_root)) as Arc<dyn Tool>);
        tools.push(Arc::new(AskCodeTool::new(project_root)) as Arc<dyn Tool>);
    }

    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file_tool() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let tool = ReadFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("test.txt"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        assert_eq!(result.to_content_string(), "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_file_tool() {
        let dir = TempDir::new().unwrap();

        let tool = WriteFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("new_file.txt"));
        input.insert("content".to_string(), serde_json::json!("New content!"));

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(dir.path().join("new_file.txt")).unwrap();
        assert_eq!(content, "New content!");
    }

    #[tokio::test]
    async fn test_edit_file_tool() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "Hello, World!").unwrap();

        let tool = EditFileTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("path".to_string(), serde_json::json!("test.txt"));
        input.insert("old_string".to_string(), serde_json::json!("World"));
        input.insert("new_string".to_string(), serde_json::json!("Rust"));

        let result = tool.call(input).await;
        assert!(!result.is_error());

        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_list_directory_tool() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListDirectoryTool::new(dir.path());
        let input = HashMap::new();

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("a.txt"));
        assert!(content.contains("b.txt"));
        assert!(content.contains("subdir/"));
    }

    #[tokio::test]
    async fn test_grep_tool() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("test.rs"),
            "fn main() {\n    println!(\"PLACEHOLDER: fix this\");\n}",
        )
        .unwrap();

        let tool = GrepTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("pattern".to_string(), serde_json::json!("PLACEHOLDER"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("PLACEHOLDER"));
        assert!(content.contains("test.rs:2:"));
    }

    #[tokio::test]
    async fn test_bash_tool() {
        let dir = TempDir::new().unwrap();

        let tool = BashTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert("command".to_string(), serde_json::json!("echo hello"));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        assert!(result.to_content_string().contains("hello"));
    }
}
