//! Integration test: BashTool background process spawn + read

use std::collections::HashMap;
use llm_code_sdk::tools::{BashTool, Tool};

#[tokio::test]
async fn bash_background_pipe_produces_output() {
    let tool = BashTool::new("/tmp");

    // Spawn background process
    let mut spawn_input = HashMap::new();
    spawn_input.insert("command".to_string(), serde_json::json!("echo BG_PIPE_TEST && sleep 0.5"));
    spawn_input.insert("interactive".to_string(), serde_json::json!("true"));

    let result = tool.call(spawn_input).await;
    let text = result.to_content_string();
    println!("Spawn result: {text}");
    assert!(text.contains("process #"), "Should contain process ID: {text}");

    // Wait a bit for output to arrive
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // Read output
    let mut read_input = HashMap::new();
    read_input.insert("process_id".to_string(), serde_json::json!("1"));
    read_input.insert("action".to_string(), serde_json::json!("read"));

    let result = tool.call(read_input).await;
    let text = result.to_content_string();
    println!("Read result: {text}");
    assert!(text.contains("BG_PIPE_TEST"), "Should contain output: {text}");
}

#[tokio::test]
async fn bash_background_pty_produces_output() {
    let tool = BashTool::new("/tmp");

    // Spawn PTY background process
    let mut spawn_input = HashMap::new();
    spawn_input.insert("command".to_string(), serde_json::json!("echo PTY_BG_TEST && sleep 0.5"));
    spawn_input.insert("tty".to_string(), serde_json::json!("true"));

    let result = tool.call(spawn_input).await;
    let text = result.to_content_string();
    println!("PTY Spawn result: {text}");
    assert!(text.contains("process #"), "Should contain process ID: {text}");

    // Wait a bit for output to arrive
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // Read output
    let mut read_input = HashMap::new();
    read_input.insert("process_id".to_string(), serde_json::json!("1"));
    read_input.insert("action".to_string(), serde_json::json!("read"));

    let result = tool.call(read_input).await;
    let text = result.to_content_string();
    println!("PTY Read result: {text}");
    assert!(text.contains("PTY_BG_TEST"), "Should contain output: {text}");
}

#[tokio::test]
async fn bash_background_status_and_kill() {
    let tool = BashTool::new("/tmp");

    // Spawn a long-running process
    let mut spawn_input = HashMap::new();
    spawn_input.insert("command".to_string(), serde_json::json!("sleep 60"));
    spawn_input.insert("interactive".to_string(), serde_json::json!("true"));

    let result = tool.call(spawn_input).await;
    let text = result.to_content_string();
    println!("Spawn: {text}");

    // Check status
    let mut status_input = HashMap::new();
    status_input.insert("process_id".to_string(), serde_json::json!("1"));
    status_input.insert("action".to_string(), serde_json::json!("status"));

    let result = tool.call(status_input).await;
    let text = result.to_content_string();
    println!("Status: {text}");
    assert!(text.contains("running"), "Should be running: {text}");

    // Kill it
    let mut kill_input = HashMap::new();
    kill_input.insert("process_id".to_string(), serde_json::json!("1"));
    kill_input.insert("action".to_string(), serde_json::json!("kill"));

    let result = tool.call(kill_input).await;
    let text = result.to_content_string();
    println!("Kill: {text}");
    assert!(text.contains("terminated"), "Should be terminated: {text}");
}
