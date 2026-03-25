//! Tests for the terminal session API.

use llm_code_sdk::tools::terminal::*;

#[test]
fn encode_simple_key() {
    let event = KeyEvent {
        event_type: "key_press".to_string(),
        key: "a".to_string(),
        modifiers: vec![],
        repeat: 1,
    };
    assert_eq!(encode_key_event(&event), b"a");
}

#[test]
fn encode_ctrl_c() {
    let event = KeyEvent {
        event_type: "key_press".to_string(),
        key: "c".to_string(),
        modifiers: vec!["Ctrl".to_string()],
        repeat: 1,
    };
    assert_eq!(encode_key_event(&event), vec![3]); // ETX
}

#[test]
fn encode_enter() {
    let event = KeyEvent {
        event_type: "key_press".to_string(),
        key: "Enter".to_string(),
        modifiers: vec![],
        repeat: 1,
    };
    assert_eq!(encode_key_event(&event), vec![b'\r']);
}

#[test]
fn encode_alt_enter() {
    let event = KeyEvent {
        event_type: "key_press".to_string(),
        key: "Enter".to_string(),
        modifiers: vec!["Alt".to_string()],
        repeat: 1,
    };
    assert_eq!(encode_key_event(&event), vec![0x1b, b'\r']);
}

#[test]
fn encode_arrow_keys() {
    let up = KeyEvent { event_type: "key_press".into(), key: "Up".into(), modifiers: vec![], repeat: 1 };
    let down = KeyEvent { event_type: "key_press".into(), key: "Down".into(), modifiers: vec![], repeat: 1 };
    assert_eq!(encode_key_event(&up), b"\x1b[A");
    assert_eq!(encode_key_event(&down), b"\x1b[B");
}

#[test]
fn encode_f_keys() {
    let f1 = KeyEvent { event_type: "key_press".into(), key: "F1".into(), modifiers: vec![], repeat: 1 };
    let f5 = KeyEvent { event_type: "key_press".into(), key: "F5".into(), modifiers: vec![], repeat: 1 };
    assert_eq!(encode_key_event(&f1), b"\x1bOP");
    assert_eq!(encode_key_event(&f5), b"\x1b[15~");
}

#[test]
fn encode_repeat() {
    let event = KeyEvent {
        event_type: "key_press".to_string(),
        key: "x".to_string(),
        modifiers: vec![],
        repeat: 3,
    };
    assert_eq!(encode_key_event(&event), b"xxx");
}

#[test]
fn encode_bracketed_paste() {
    let bytes = encode_paste("hello world");
    assert!(bytes.starts_with(b"\x1b[200~"));
    assert!(bytes.ends_with(b"\x1b[201~"));
    assert!(bytes.windows(11).any(|w| w == b"hello world"));
}

#[test]
fn encode_mouse_sgr() {
    let event = MouseEvent {
        event_type: "mouse_down".to_string(),
        x: 10,
        y: 5,
        button: "left".to_string(),
        modifiers: vec![],
    };
    let bytes = encode_mouse_event(&event);
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(text, "\x1b[<0;11;6M"); // SGR: 0=left, 1-indexed coords
}

#[test]
fn encode_mouse_up_sgr() {
    let event = MouseEvent {
        event_type: "mouse_up".to_string(),
        x: 10,
        y: 5,
        button: "left".to_string(),
        modifiers: vec![],
    };
    let bytes = encode_mouse_event(&event);
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.ends_with('m')); // lowercase m for release
}

#[test]
fn strip_ansi_basic() {
    let input = "\x1b[32mhello\x1b[0m world";
    assert_eq!(strip_ansi(input), "hello world");
}

#[tokio::test]
async fn session_spawn_and_read() {
    let mut registry = SessionRegistry::new();
    let config = SessionConfig {
        command: Some("echo TERMINAL_TEST && sleep 0.2".to_string()),
        ..Default::default()
    };

    let id = registry.spawn(config).await.expect("spawn failed");

    // Wait for output
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let session = registry.get_mut(id).unwrap();
    let text = session.read_text().await;
    println!("Terminal output: {text:?}");
    assert!(text.contains("TERMINAL_TEST"), "Expected TERMINAL_TEST in output, got: {text:?}");
}

#[tokio::test]
async fn session_send_key_and_read() {
    let mut registry = SessionRegistry::new();
    let config = SessionConfig::default(); // interactive shell

    let id = registry.spawn(config).await.expect("spawn failed");

    // Wait for shell prompt
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    // Type a command
    let session = registry.get_mut(id).unwrap();
    session.send_text("echo KEY_TEST\r").unwrap();

    // Wait for execution
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

    let text = session.read_text().await;
    println!("Shell output: {text:?}");
    assert!(text.contains("KEY_TEST"), "Expected KEY_TEST in output, got: {text:?}");
}
