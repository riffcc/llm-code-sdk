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

fn diffstat_metadata(path: &str, operation: &str, before: &str, after: &str) -> serde_json::Value {
    let (added_lines, removed_lines) = diff_line_counts(before, after);
    serde_json::json!({
        "path": path,
        "operation": operation,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
    })
}

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

/// Execute bash commands — fresh process per command, with background process support.
///
/// Three modes:
/// 1. Normal: `bash(command: "cargo build")` — runs, waits, returns output
/// 2. Background: `bash(command: "server", background: true)` — spawns via PTY, returns process_id
/// 3. Follow-up: `bash(process_id: "1", action: "read|write|status|kill")` — interact with background
///
/// Working directory tracked across calls. Output capped at 1 MiB.
pub struct BashTool {
    initial_dir: PathBuf,
    cwd: Arc<TokioMutex<PathBuf>>,
    default_timeout_secs: u64,
    /// Background processes — shared with the host for UI/interaction.
    bg: Arc<TokioMutex<BgProcessRegistry>>,
}

/// Registry of background processes. Shared between BashTool and the host.
pub struct BgProcessRegistry {
    next_id: u32,
    procs: HashMap<u32, BgProcess>,
}

/// A background process entry.
pub struct BgProcess {
    pub command: String,
    handle: lcs_pty::ProcessHandle,
    /// Accumulated output buffer.
    output: String,
    /// Stdout channel — read directly on demand.
    stdout_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    /// Stderr channel.
    stderr_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    /// Screen parser for PTY processes (provides snapshots and diffs).
    screen: Option<super::screen::Screen>,
    pub exited: bool,
    pub exit_code: Option<i32>,
}

/// Snapshot of a background process for display.
#[derive(Debug, Clone)]
pub struct BgProcessInfo {
    pub id: u32,
    pub command: String,
    pub exited: bool,
    pub exit_code: Option<i32>,
}

impl BgProcessRegistry {
    fn new() -> Self {
        Self { next_id: 1, procs: HashMap::new() }
    }

    /// List all processes.
    pub fn list(&self) -> Vec<BgProcessInfo> {
        self.procs.iter().map(|(&id, p)| BgProcessInfo {
            id,
            command: p.command.clone(),
            exited: p.exited,
            exit_code: p.exit_code,
        }).collect()
    }

    /// Get a process's writer channel (for forwarding user keystrokes).
    pub fn writer(&self, id: u32) -> Option<tokio::sync::mpsc::Sender<Vec<u8>>> {
        self.procs.get(&id).map(|p| p.handle.writer_sender())
    }

    /// Read buffered output from a process.
    pub fn read_output(&mut self, id: u32) -> Option<String> {
        let proc = self.procs.get_mut(&id)?;
        // Drain pending channel data
        while let Ok(data) = proc.stdout_rx.try_recv() {
            proc.output.push_str(&String::from_utf8_lossy(&data));
        }
        while let Ok(data) = proc.stderr_rx.try_recv() {
            proc.output.push_str(&String::from_utf8_lossy(&data));
        }
        if proc.output.is_empty() { None } else { Some(proc.output.clone()) }
    }

    /// Terminate a process.
    pub fn terminate(&mut self, id: u32) {
        if let Some(proc) = self.procs.get_mut(&id) {
            proc.handle.terminate();
            proc.exited = true;
            proc.exit_code = Some(-9);
        }
    }

    /// Remove completed processes.
    pub fn clean(&mut self) -> usize {
        let dead: Vec<u32> = self.procs.iter()
            .filter(|(_, p)| p.exited)
            .map(|(id, _)| *id)
            .collect();
        let count = dead.len();
        for id in dead {
            self.procs.remove(&id);
        }
        count
    }

    /// Number of running processes.
    pub fn running_count(&self) -> usize {
        self.procs.values().filter(|p| !p.exited).count()
    }
}

/// Max output size in bytes.
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;

impl BashTool {
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        let dir = working_dir.into();
        Self {
            initial_dir: dir.clone(),
            cwd: Arc::new(TokioMutex::new(dir)),
            default_timeout_secs: 30,
            bg: Arc::new(TokioMutex::new(BgProcessRegistry::new())),
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    /// Get a handle to the background process registry.
    /// The host uses this to display /jobs, /attach, /clean.
    pub fn process_registry(&self) -> Arc<TokioMutex<BgProcessRegistry>> {
        Arc::clone(&self.bg)
    }

    /// Spawn a background process via lcs-pty. Returns the process ID.
    async fn spawn_background(&self, command: &str, tty: bool) -> Result<(u32, String), String> {
        let cwd = self.cwd.lock().await.clone();
        let args = vec!["-c".to_string(), command.to_string()];
        let mut env: HashMap<String, String> = [
            ("GIT_TERMINAL_PROMPT", "0"),
            ("PAGER", "cat"),
            ("GIT_PAGER", "cat"),
            ("TERM", "xterm-256color"),
            ("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"),
            ("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        ].into_iter().map(|(k,v)| (k.to_string(), v.to_string())).collect();
        // Inherit HOME if available
        if let Ok(home) = std::env::var("HOME") {
            env.insert("HOME".to_string(), home);
        }
        // Inherit real PATH if available (overrides default)
        if let Ok(path) = std::env::var("PATH") {
            env.insert("PATH".to_string(), path);
        }
        let arg0: Option<String> = None;

        let spawned = if tty {
            let size = lcs_pty::TerminalSize { rows: 24, cols: 120 };
            lcs_pty::spawn_pty_process("/bin/bash", &args, &cwd, &env, &arg0, size)
                .await.map_err(|e| format!("PTY spawn failed: {e}"))?
        } else {
            lcs_pty::spawn_pipe_process("/bin/bash", &args, &cwd, &env, &arg0)
                .await.map_err(|e| format!("Pipe spawn failed: {e}"))?
        };

        // Store channels directly — read on demand, no background collector
        let exit_rx = spawned.exit_rx;
        let bg_ref = Arc::clone(&self.bg);
        let mut bg = self.bg.lock().await;
        let id = bg.next_id;
        bg.next_id += 1;

        let screen = if tty {
            Some(super::screen::Screen::new(120, 24))
        } else {
            None
        };

        bg.procs.insert(id, BgProcess {
            command: command.to_string(),
            handle: spawned.session,
            output: String::new(),
            stdout_rx: spawned.stdout_rx,
            stderr_rx: spawned.stderr_rx,
            screen,
            exited: false,
            exit_code: None,
        });

        // Watch for exit
        let exit_id = id;
        tokio::spawn(async move {
            if let Ok(code) = exit_rx.await {
                let mut bg = bg_ref.lock().await;
                if let Some(proc) = bg.procs.get_mut(&exit_id) {
                    proc.exited = true;
                    proc.exit_code = Some(code);
                }
            }
        });

        let mode = if tty { "tty" } else { "interactive" };
        Ok((id, format!("Background process #{id} started ({mode}). Use process_id: \"{id}\" with action: \"read\"/\"write\"/\"status\"/\"kill\" to interact.")))
    }

    /// Drain pending channel data into output buffer and screen.
    fn drain_process(proc: &mut BgProcess) {
        while let Ok(data) = proc.stdout_rx.try_recv() {
            proc.output.push_str(&String::from_utf8_lossy(&data));
            if let Some(screen) = &mut proc.screen {
                screen.feed(&data);
            }
        }
        while let Ok(data) = proc.stderr_rx.try_recv() {
            proc.output.push_str(&String::from_utf8_lossy(&data));
            if let Some(screen) = &mut proc.screen {
                screen.feed(&data);
            }
        }
    }

    /// Handle follow-up actions on background processes.
    async fn process_action(&self, pid: u32, action: &str, input: &str, orig_input: &HashMap<String, serde_json::Value>) -> Result<String, String> {
        let mut bg = self.bg.lock().await;
        let proc = bg.procs.get_mut(&pid).ok_or_else(|| format!("No background process #{pid}"))?;

        match action {
            "read" => {
                Self::drain_process(proc);

                if proc.output.is_empty() && !proc.exited {
                    // No data yet — release lock, yield, try once more
                    let is_exited = proc.exited;
                    drop(bg);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    let mut bg = self.bg.lock().await;
                    if let Some(proc) = bg.procs.get_mut(&pid) {
                        Self::drain_process(proc);
                        if proc.output.is_empty() {
                            Ok(format!("(no output yet from #{pid})"))
                        } else {
                            if proc.output.len() > MAX_OUTPUT_BYTES { proc.output.truncate(MAX_OUTPUT_BYTES); }
                            Ok(proc.output.clone())
                        }
                    } else {
                        Ok(format!("(process #{pid} not found)"))
                    }
                } else {
                    if proc.output.len() > MAX_OUTPUT_BYTES { proc.output.truncate(MAX_OUTPUT_BYTES); }
                    if proc.output.is_empty() {
                        Ok(format!("(process #{pid} exited with code {})", proc.exit_code.unwrap_or(-1)))
                    } else {
                        Ok(proc.output.clone())
                    }
                }
            }
            "write" => {
                if input.is_empty() {
                    return Err("'input' is required for write action".to_string());
                }
                let sender = proc.handle.writer_sender();
                let mut data = input.as_bytes().to_vec();
                if !input.ends_with('\n') {
                    data.push(b'\n');
                }
                sender.send(data).await.map_err(|e| format!("Write failed: {e}"))?;
                Ok(format!("Sent {} bytes to #{pid}", input.len()))
            }
            "status" => {
                if proc.exited {
                    let code = proc.exit_code.unwrap_or(-1);
                    Ok(format!("Process #{pid} exited with code {code}"))
                } else {
                    Ok(format!("Process #{pid} is running"))
                }
            }
            "kill" => {
                proc.handle.terminate();
                proc.exited = true;
                proc.exit_code = Some(-9);
                Ok(format!("Process #{pid} terminated"))
            }
            "snapshot" => {
                Self::drain_process(proc);
                if let Some(screen) = &mut proc.screen {
                    let snap = screen.snapshot();
                    // Return just the visible text lines (compact)
                    let text: Vec<&str> = snap.text.iter()
                        .map(|s| s.as_str())
                        .collect();
                    Ok(serde_json::json!({
                        "width": snap.width,
                        "height": snap.height,
                        "cursor": [snap.cursor_x, snap.cursor_y],
                        "cursor_visible": snap.cursor_visible,
                        "alternate_screen": snap.alternate_screen,
                        "text": text,
                    }).to_string())
                } else {
                    Err(format!("Process #{pid} has no screen (not a PTY session)"))
                }
            }
            "diff" => {
                Self::drain_process(proc);
                if let Some(screen) = &mut proc.screen {
                    let d = screen.diff();
                    Ok(serde_json::json!({
                        "cursor": d.cursor,
                        "changed_lines": d.changed_lines.iter()
                            .map(|(line, text)| serde_json::json!({"line": line, "text": text}))
                            .collect::<Vec<_>>(),
                    }).to_string())
                } else {
                    Err(format!("Process #{pid} has no screen (not a PTY session)"))
                }
            }
            "key" => {
                // Read key from top-level fields (not nested JSON)
                let key_name = orig_input.get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if key_name.is_empty() {
                    return Err("'key' field is required for key action (e.g. key: 'Enter')".to_string());
                }
                let modifiers_str = orig_input.get("modifiers")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let modifiers: Vec<String> = if modifiers_str.is_empty() {
                    vec![]
                } else {
                    modifiers_str.split(',').map(|s| s.trim().to_string()).collect()
                };
                let event_type = orig_input.get("key_event")
                    .and_then(|v| v.as_str())
                    .unwrap_or("press")
                    .to_string();

                let event = super::terminal::KeyEvent {
                    event_type,
                    key: key_name.to_string(),
                    modifiers,
                    repeat: 1,
                };
                let bytes = super::terminal::encode_key_event(&event);
                let sender = proc.handle.writer_sender();
                sender.try_send(bytes).map_err(|e| format!("Send failed: {e}"))?;
                Ok(format!("Key '{}' sent to #{pid}", key_name))
            }
            "paste" => {
                if input.is_empty() {
                    return Err("'input' is required for paste action".to_string());
                }
                let bytes = super::terminal::encode_paste(input);
                let sender = proc.handle.writer_sender();
                sender.try_send(bytes).map_err(|e| format!("Send failed: {e}"))?;
                Ok(format!("Pasted {} bytes to #{pid}", input.len()))
            }
            "resize" => {
                if input.is_empty() {
                    return Err("'input' is required for resize (e.g. '80x24')".to_string());
                }
                let parts: Vec<&str> = input.split('x').collect();
                if parts.len() != 2 {
                    return Err("Resize format: 'COLSxROWS' (e.g. '80x24')".to_string());
                }
                let cols: u16 = parts[0].parse().map_err(|_| "Invalid cols")?;
                let rows: u16 = parts[1].parse().map_err(|_| "Invalid rows")?;
                proc.handle.resize(lcs_pty::TerminalSize { rows, cols })
                    .map_err(|e| format!("Resize failed: {e}"))?;
                if let Some(screen) = &mut proc.screen {
                    screen.resize(cols, rows);
                }
                Ok(format!("Resized #{pid} to {cols}x{rows}"))
            }
            _ => Err(format!("Unknown action '{action}'. Use: read, write, status, kill, snapshot, diff, key, paste, resize")),
        }
    }

    /// Execute a command in a fresh process. Returns (stdout+stderr, exit_code).
    async fn exec(&self, command: &str, timeout_secs: u64) -> Result<(String, i32), String> {
        let cwd = self.cwd.lock().await.clone();

        // Wrap command: run in subshell, merge stderr, print cwd at end for tracking.
        let wrapped = format!(
            "( {command} ) 2>&1; __exit=$?; echo; echo \"__CWD__$(pwd)\"; exit $__exit"
        );

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c")
            .arg(&wrapped)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "")
            .env("SSH_ASKPASS", "")
            .env("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new")
            .env("DEBIAN_FRONTEND", "noninteractive")
            .env("PAGER", "cat")
            .env("GIT_PAGER", "cat");

        // Process group isolation on Unix
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    // New session + process group
                    libc::setsid();
                    Ok(())
                });
            }
        }

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn bash: {e}"))?;

        let stdout = child.stdout.take().ok_or("No stdout")?;
        let stderr = child.stderr.take().ok_or("No stderr")?;

        // Read stdout and stderr concurrently, capped
        let stdout_handle = tokio::spawn(read_capped(stdout, MAX_OUTPUT_BYTES));
        let stderr_handle = tokio::spawn(read_capped(stderr, MAX_OUTPUT_BYTES));

        // Wait with timeout
        let timeout_duration = tokio::time::Duration::from_secs(timeout_secs);
        let result = tokio::time::timeout(timeout_duration, child.wait()).await;

        let (exit_code, timed_out) = match result {
            Ok(Ok(status)) => (status.code().unwrap_or(1), false),
            Ok(Err(e)) => return Err(format!("Process error: {e}")),
            Err(_) => {
                // Timeout — kill process group
                #[cfg(unix)]
                if let Some(pid) = child.id() {
                    unsafe { libc::killpg(pid as libc::pid_t, libc::SIGKILL); }
                }
                let _ = child.start_kill();
                let _ = child.wait().await;
                (124, true)
            }
        };

        // Collect output (with short drain timeout)
        let stdout_text = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            stdout_handle,
        ).await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default();

        let stderr_text = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            stderr_handle,
        ).await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default();

        // Extract cwd from output and update tracking
        let (clean_output, new_cwd) = extract_cwd(&stdout_text);

        if let Some(dir) = new_cwd {
            let path = PathBuf::from(&dir);
            if path.is_dir() {
                *self.cwd.lock().await = path;
            }
        }

        // Combine stdout + stderr
        let mut output = clean_output;
        if !stderr_text.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&stderr_text);
        }

        if timed_out {
            output.push_str("\n(command timed out after ");
            output.push_str(&timeout_secs.to_string());
            output.push_str("s)");
        }

        Ok((output, exit_code))
    }
}

/// Read from an async reader up to max_bytes.
async fn read_capped(reader: impl tokio::io::AsyncRead + Unpin, max_bytes: usize) -> String {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::with_capacity(8192);
    let mut reader = reader;
    let mut chunk = [0u8; 8192];

    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                let remaining = max_bytes.saturating_sub(buf.len());
                let take = n.min(remaining);
                buf.extend_from_slice(&chunk[..take]);
                if buf.len() >= max_bytes {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    String::from_utf8_lossy(&buf).to_string()
}

/// Extract __CWD__ marker from output and return (clean_output, optional_cwd).
fn extract_cwd(output: &str) -> (String, Option<String>) {
    let mut lines: Vec<&str> = output.lines().collect();
    let mut cwd = None;

    // Look for __CWD__ in the last few lines
    if let Some(pos) = lines.iter().rposition(|l| l.starts_with("__CWD__")) {
        cwd = Some(lines[pos].strip_prefix("__CWD__").unwrap_or("").to_string());
        lines.remove(pos);
        // Remove trailing empty line before the marker
        while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
            lines.pop();
        }
    }

    (lines.join("\n"), cwd)
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
                .optional_string("command", "The bash command to execute")
                .optional_string("timeout", "Timeout in seconds (default: 30)")
                .optional_string("tty", "'true' for full PTY terminal (interactive commands). Runs in background, returns process_id.")
                .optional_string("interactive", "'true' to keep stdin open. Runs in background, returns process_id.")
                .optional_string("process_id", "Interact with a background process by ID")
                .optional_string("action", "Action on background process: read, write, status, kill, snapshot, diff, key, paste, resize")
                .optional_string("input", "Data for write (stdin text), paste (text to paste), or resize ('COLSxROWS')")
                .optional_string("key", "Key name for action: 'key'. E.g. Enter, Backspace, Tab, Escape, ArrowUp, ArrowDown, F1-F12, or a single character")
                .optional_string("modifiers", "Comma-separated modifiers for key action: Ctrl, Alt, Shift, Meta")
                .optional_string("key_event", "Key event type: 'press' (default), 'down', or 'up'"),
        )
        .with_description(
            "Execute bash commands or manage background processes. \
             Normal: bash(command: 'cargo build') — runs, waits, returns output. \
             Background: bash(command: 'htop', tty: true) — spawns PTY terminal, returns process_id. \
             Follow-up: bash(process_id: '1', action: 'read') — read output from background process. \
             Actions: read (get output), write (send stdin), status (check alive), kill (terminate), \
             snapshot (parsed screen grid), diff (changed lines since last snapshot), \
             key (send key press — set 'key' field to key name like 'Enter', 'ArrowUp', 'a'; set 'modifiers' to 'Ctrl,Alt' etc), \
             paste (bracketed paste via input), resize (e.g. input: '120x24'). \
             Key names: Enter, Backspace, Tab, Escape, ArrowUp/Down/Left/Right, Home, End, PageUp, PageDown, Delete, Insert, F1-F12, or any single character. \
             Key event types via key_event: 'press' (default), 'down', 'up'."
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let process_id = input.get("process_id")
            .and_then(|v| v.as_str().or_else(|| v.as_u64().map(|_| "")).and_then(|s| if s.is_empty() { v.as_u64().map(|n| n.to_string()).as_deref().map(|_| s) } else { Some(s) }))
            .unwrap_or("");

        // Simpler process_id extraction
        let pid: Option<u32> = input.get("process_id")
            .and_then(|v| {
                v.as_str().and_then(|s| s.parse().ok())
                    .or_else(|| v.as_u64().map(|n| n as u32))
            });

        // Follow-up mode: interact with existing background process
        if let Some(pid) = pid {
            let action = input.get("action").and_then(|v| v.as_str()).unwrap_or("read");
            let stdin_input = input.get("input").and_then(|v| v.as_str()).unwrap_or("");

            return match self.process_action(pid, action, stdin_input, &input).await {
                Ok(result) => ToolResult::success(result),
                Err(e) => ToolResult::error(e),
            };
        }

        if command.is_empty() {
            return ToolResult::error("'command' is required for new processes, or 'process_id' for follow-up");
        }

        let timeout = input.get("timeout")
            .and_then(|v| v.as_str().and_then(|s| s.parse::<u64>().ok()).or_else(|| v.as_u64()))
            .unwrap_or(self.default_timeout_secs);

        let tty = input.get("tty")
            .and_then(|v| v.as_str().map(|s| s == "true").or_else(|| v.as_bool()))
            .unwrap_or(false);

        let interactive = input.get("interactive")
            .and_then(|v| v.as_str().map(|s| s == "true").or_else(|| v.as_bool()))
            .unwrap_or(false);

        // Background mode: spawn via lcs-pty, return process_id
        if tty || interactive {
            return match self.spawn_background(command, tty).await {
                Ok((_id, msg)) => ToolResult::success(msg),
                Err(e) => ToolResult::error(e),
            };
        }

        // Normal mode: run and wait
        match self.exec(command, timeout).await {
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
            Err(e) => ToolResult::error(e),
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

        let previous = std::fs::read_to_string(&full_path).unwrap_or_default();
        let metadata = diffstat_metadata(path, "write", &previous, content);

        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return ToolResult::error(format!("Failed to create directories: {}", e));
            }
        }

        match std::fs::write(&full_path, content) {
            Ok(_) => ToolResult::success_with_metadata(
                format!("Wrote {} bytes to {}", content.len(), path),
                metadata,
            ),
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
