//! Smoke test: does PTY output actually flow?

use std::collections::HashMap;
use lcs_pty::{TerminalSize, spawn_pty_process, spawn_pipe_process};

#[tokio::test]
async fn pty_echo_produces_output() {
    let env: HashMap<String, String> = [
        ("PATH", "/usr/local/bin:/usr/bin:/bin"),
        ("TERM", "xterm-256color"),
    ].into_iter().map(|(k,v)| (k.to_string(), v.to_string())).collect();

    let args = vec!["-c".to_string(), "echo HELLO_PTY && sleep 0.1".to_string()];
    let arg0: Option<String> = None;
    let size = TerminalSize { rows: 24, cols: 80 };

    let spawned = spawn_pty_process("/bin/bash", &args, std::path::Path::new("/tmp"), &env, &arg0, size)
        .await
        .expect("PTY spawn failed");

    let mut stdout_rx = spawned.stdout_rx;
    let exit_rx = spawned.exit_rx;

    // Collect output with timeout
    let mut output = String::new();
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);

    loop {
        tokio::select! {
            chunk = stdout_rx.recv() => {
                match chunk {
                    Some(data) => {
                        output.push_str(&String::from_utf8_lossy(&data));
                        if output.contains("HELLO_PTY") {
                            break;
                        }
                    }
                    None => break,
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                break;
            }
        }
    }

    let code = exit_rx.await.unwrap_or(-1);

    println!("PTY output ({} bytes): {:?}", output.len(), &output[..output.len().min(200)]);
    println!("Exit code: {code}");

    assert!(output.contains("HELLO_PTY"), "PTY output should contain HELLO_PTY, got: {output:?}");
}

#[tokio::test]
async fn pipe_echo_produces_output() {
    let env: HashMap<String, String> = [
        ("PATH", "/usr/local/bin:/usr/bin:/bin"),
    ].into_iter().map(|(k,v)| (k.to_string(), v.to_string())).collect();

    let args = vec!["-c".to_string(), "echo HELLO_PIPE && sleep 0.1".to_string()];
    let arg0: Option<String> = None;

    let spawned = spawn_pipe_process("/bin/bash", &args, std::path::Path::new("/tmp"), &env, &arg0)
        .await
        .expect("Pipe spawn failed");

    let mut stdout_rx = spawned.stdout_rx;
    let exit_rx = spawned.exit_rx;

    let mut output = String::new();
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);

    loop {
        tokio::select! {
            chunk = stdout_rx.recv() => {
                match chunk {
                    Some(data) => {
                        output.push_str(&String::from_utf8_lossy(&data));
                        if output.contains("HELLO_PIPE") {
                            break;
                        }
                    }
                    None => break,
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                break;
            }
        }
    }

    let code = exit_rx.await.unwrap_or(-1);

    println!("Pipe output ({} bytes): {:?}", output.len(), &output[..output.len().min(200)]);
    println!("Exit code: {code}");

    assert!(output.contains("HELLO_PIPE"), "Pipe output should contain HELLO_PIPE, got: {output:?}");
}
