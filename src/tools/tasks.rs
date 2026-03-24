//! Tasks tool — native task/issue management for agents.
//!
//! Wraps `bd` (beads) for now. Will be replaced with Rust-native implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use async_trait::async_trait;

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// Status icon + ANSI color for a task status.
fn status_display(status: &str) -> (&'static str, &'static str) {
    match status {
        "open" => ("◌", "\x1b[37m"),           // white
        "in_progress" => ("◐", "\x1b[33m"),     // yellow
        "deferred" => ("❄", "\x1b[36m"),        // cyan
        "blocked" => ("●", "\x1b[31m"),         // red
        "tests_written" => ("◍", "\x1b[35m"),   // magenta
        "passing_tests" => ("✪", "\x1b[34m"),   // blue
        "accepted" => ("⦿", "\x1b[32m"),        // green
        "closed" => ("✔", "\x1b[32m"),          // green
        _ => ("?", "\x1b[90m"),                 // gray
    }
}

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

/// Format a list of tasks as markdown.
fn format_task_list(json: &str) -> String {
    let tasks: Vec<serde_json::Value> = match serde_json::from_str(json) {
        Ok(t) => t,
        Err(_) => return json.to_string(),
    };

    if tasks.is_empty() {
        return "No tasks found.".to_string();
    }

    let mut out = String::new();
    for task in &tasks {
        let id = task["id"].as_str().unwrap_or("?");
        let title = task["title"].as_str().unwrap_or("?");
        let status = task["status"].as_str().unwrap_or("open");
        let priority = task["priority"].as_u64().unwrap_or(0);
        let (icon, color) = status_display(status);

        out.push_str(&format!("{color}{icon}{RESET} {BOLD}{title}{RESET}\n"));
        out.push_str(&format!("  {DIM}{id} · P{priority}{RESET}\n"));
    }
    out
}

/// Format a single task as markdown.
fn format_task(json: &str) -> String {
    let task: serde_json::Value = match serde_json::from_str(json) {
        Ok(t) => t,
        Err(_) => return json.to_string(),
    };

    let id = task["id"].as_str().unwrap_or("?");
    let title = task["title"].as_str().unwrap_or("?");
    let status = task["status"].as_str().unwrap_or("open");
    let priority = task["priority"].as_u64().unwrap_or(0);
    let desc = task["description"].as_str().unwrap_or("");
    let issue_type = task["issue_type"].as_str().unwrap_or("");
    let owner = task["owner"].as_str().unwrap_or("");
    let (icon, color) = status_display(status);

    let mut out = format!("{color}{icon}{RESET} {BOLD}{title}{RESET}\n");
    out.push_str(&format!("  {DIM}{id} · P{priority} · {issue_type}{RESET}\n"));
    if !owner.is_empty() {
        out.push_str(&format!("  {DIM}Owner: {owner}{RESET}\n"));
    }
    if !desc.is_empty() {
        out.push_str(&format!("\n{desc}\n"));
    }
    out
}

/// Native task management tool.
pub struct TasksTool {
    project_root: PathBuf,
}

impl TasksTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
        }
    }

    fn bd(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new("bd")
            .args(args)
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| format!("failed to run bd: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("bd failed: {stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn handle_op(&self, op: &str, input: &HashMap<String, serde_json::Value>) -> ToolResult {
        let s = |key: &str| input.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();

        match op {
            "list" => match self.bd(&["list", "--json"]) {
                Ok(out) => ToolResult::success(format_task_list(&out)),
                Err(e) => ToolResult::error(e),
            },
            "ready" => match self.bd(&["ready", "--json"]) {
                Ok(out) => ToolResult::success(format_task_list(&out)),
                Err(e) => ToolResult::error(e),
            },
            "show" => {
                let id = s("id");
                if id.is_empty() {
                    return ToolResult::error("'id' is required for show");
                }
                match self.bd(&["show", &id, "--json"]) {
                    Ok(out) => ToolResult::success(format_task(&out)),
                    Err(e) => ToolResult::error(e),
                }
            }
            "create" => {
                let title = s("title");
                if title.is_empty() {
                    return ToolResult::error("'title' is required for create");
                }
                let mut args = vec!["create", "--json"];
                let title_owned = title;
                args.extend(["--title", &title_owned]);

                let desc = s("description");
                if !desc.is_empty() {
                    args.extend(["--description", &desc]);
                }

                let task_type = s("type");
                if !task_type.is_empty() {
                    args.extend(["--type", &task_type]);
                }

                match self.bd(&args) {
                    Ok(out) => ToolResult::success(format_task(&out)),
                    Err(e) => ToolResult::error(e),
                }
            }
            "claim" => {
                let id = s("id");
                if id.is_empty() {
                    return ToolResult::error("'id' is required for claim");
                }
                match self.bd(&["update", &id, "--claim"]) {
                    Ok(out) => ToolResult::success(format!("Claimed {id}\n{out}")),
                    Err(e) => ToolResult::error(e),
                }
            }
            "close" => {
                let id = s("id");
                if id.is_empty() {
                    return ToolResult::error("'id' is required for close");
                }
                let reason = s("reason");
                let mut args = vec!["close", &id];
                if !reason.is_empty() {
                    args.extend(["--reason", &reason]);
                }
                match self.bd(&args) {
                    Ok(out) => ToolResult::success(format!("Closed {id}\n{out}")),
                    Err(e) => ToolResult::error(e),
                }
            }
            "init" => match self.bd(&["init"]) {
                Ok(out) => ToolResult::success(out),
                Err(e) => ToolResult::error(e),
            },
            _ => ToolResult::error(format!("Unknown operation: {op}. Use: list, ready, show, create, claim, close, init")),
        }
    }
}

#[async_trait]
impl Tool for TasksTool {
    fn name(&self) -> &str {
        "tasks"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "tasks",
            InputSchema::object()
                .required_string("operation", "Operation: 'list', 'ready', 'show', 'create', 'claim', 'close', 'init'")
                .optional_string("id", "Task ID (for show, claim, close)")
                .optional_string("title", "Task title (for create)")
                .optional_string("description", "Task description (for create)")
                .optional_string("type", "Task type: 'task', 'bug', 'feature' (for create)")
                .optional_string("reason", "Reason (for close)"),
        )
        .with_description(
            "Manage tasks and issues. Use 'list' to see all, 'ready' for available work, 'create' to plan, 'claim' to start, 'close' to finish.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let op = input
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if op.is_empty() {
            return ToolResult::error("'operation' is required");
        }

        self.handle_op(op, &input)
    }
}
