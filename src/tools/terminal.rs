//! Terminal session — structured interactive terminal control.
//!
//! Provides a high-fidelity API for controlling PTY processes:
//! - Structured key events (not just raw bytes)
//! - Mouse events with terminal protocol translation
//! - Screen snapshots (rendered cells, not raw bytes)
//! - Lifecycle management (spawn, resize, terminate)
//! - Bracketed paste vs typed input distinction
//!
//! This is the substrate for agents operating full-screen TUIs,
//! nested tmux sessions, editors, debuggers, and games.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex as TokioMutex};

// ─── Input Events ────────────────────────────────────────────────

/// A structured keyboard event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEvent {
    /// Event type: "key_down", "key_up", or "key_press" (down + up).
    #[serde(rename = "type")]
    pub event_type: String,
    /// Key name. Printable chars as themselves, specials as names.
    pub key: String,
    /// Active modifiers.
    #[serde(default)]
    pub modifiers: Vec<String>,
    /// Repeat count (default 1).
    #[serde(default = "default_repeat")]
    pub repeat: u32,
}

fn default_repeat() -> u32 { 1 }

/// A structured mouse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouseEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "mouse_down", "mouse_up", "mouse_move", "wheel_up", "wheel_down"
    pub x: u16,
    pub y: u16,
    #[serde(default)]
    pub button: String, // "left", "right", "middle", ""
    #[serde(default)]
    pub modifiers: Vec<String>,
}

/// A gamepad event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamepadEvent {
    #[serde(rename = "type")]
    pub event_type: String, // "button_down", "button_up", "axis"
    pub button: Option<String>, // "a", "b", "x", "y", "start", "select", "dpad_up", etc.
    pub axis: Option<String>,   // "left_x", "left_y", "right_x", "right_y", "left_trigger", etc.
    pub value: Option<f32>,     // axis value -1.0..1.0 or trigger 0.0..1.0
}

// ─── Screen Snapshot ─────────────────────────────────────────────

/// A single cell in the terminal grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub ch: char,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bg: Option<String>,
    #[serde(default)]
    pub bold: bool,
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub underline: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: None,
            bg: None,
            bold: false,
            italic: false,
            underline: false,
        }
    }
}

/// A full screen snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenSnapshot {
    pub width: u16,
    pub height: u16,
    pub cursor_x: u16,
    pub cursor_y: u16,
    pub cursor_visible: bool,
    /// Grid of cells, row-major. rows[y][x].
    pub cells: Vec<Vec<Cell>>,
    /// Plain text lines (visible text only, no styling).
    pub text: Vec<String>,
    /// Whether alternate screen is active.
    pub alternate_screen: bool,
    /// Title if set by the application.
    pub title: Option<String>,
}

impl ScreenSnapshot {
    pub fn empty(width: u16, height: u16) -> Self {
        let cells = vec![vec![Cell::default(); width as usize]; height as usize];
        let text = vec![String::new(); height as usize];
        Self {
            width,
            height,
            cursor_x: 0,
            cursor_y: 0,
            cursor_visible: true,
            cells,
            text,
            alternate_screen: false,
            title: None,
        }
    }
}

/// Incremental diff between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenDiff {
    /// Changed cells: (x, y, new_cell).
    pub changed_cells: Vec<(u16, u16, Cell)>,
    /// New cursor position if changed.
    pub cursor: Option<(u16, u16)>,
    /// Changed text lines: (line_number, new_text).
    pub changed_lines: Vec<(u16, String)>,
}

// ─── Session Configuration ───────────────────────────────────────

/// Configuration for spawning a terminal session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Terminal dimensions.
    #[serde(default = "default_rows")]
    pub rows: u16,
    #[serde(default = "default_cols")]
    pub cols: u16,
    /// TERM environment variable.
    #[serde(default = "default_term")]
    pub term: String,
    /// Working directory.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Shell to use (default: /bin/bash).
    #[serde(default = "default_shell")]
    pub shell: String,
    /// Additional environment variables.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Whether to use login shell (-l flag).
    #[serde(default)]
    pub login_shell: bool,
    /// Direct command to execute (instead of interactive shell).
    #[serde(default)]
    pub command: Option<String>,
}

fn default_rows() -> u16 { 24 }
fn default_cols() -> u16 { 120 }
fn default_term() -> String { "xterm-256color".to_string() }
fn default_shell() -> String { "/bin/bash".to_string() }

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            rows: 24,
            cols: 120,
            term: "xterm-256color".to_string(),
            cwd: None,
            shell: "/bin/bash".to_string(),
            env: HashMap::new(),
            login_shell: false,
            command: None,
        }
    }
}

// ─── Key-to-escape translation ───────────────────────────────────

/// Encode a structured key event into terminal escape bytes.
pub fn encode_key_event(event: &KeyEvent) -> Vec<u8> {
    let has_ctrl = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("ctrl"));
    let has_alt = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("alt"));
    let has_shift = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("shift"));

    let base = match event.key.as_str() {
        // Special keys
        "Enter" | "Return" => vec![b'\r'],
        "Backspace" => vec![0x7f],
        "Tab" => if has_shift { b"\x1b[Z".to_vec() } else { vec![b'\t'] },
        "Escape" | "Esc" => vec![0x1b],
        "Space" => vec![b' '],
        "Delete" => b"\x1b[3~".to_vec(),
        "Insert" => b"\x1b[2~".to_vec(),

        // Arrow keys
        "Up" | "ArrowUp" => b"\x1b[A".to_vec(),
        "Down" | "ArrowDown" => b"\x1b[B".to_vec(),
        "Right" | "ArrowRight" => b"\x1b[C".to_vec(),
        "Left" | "ArrowLeft" => b"\x1b[D".to_vec(),

        // Navigation
        "Home" => b"\x1b[H".to_vec(),
        "End" => b"\x1b[F".to_vec(),
        "PageUp" => b"\x1b[5~".to_vec(),
        "PageDown" => b"\x1b[6~".to_vec(),

        // Function keys
        "F1" => b"\x1bOP".to_vec(),
        "F2" => b"\x1bOQ".to_vec(),
        "F3" => b"\x1bOR".to_vec(),
        "F4" => b"\x1bOS".to_vec(),
        "F5" => b"\x1b[15~".to_vec(),
        "F6" => b"\x1b[17~".to_vec(),
        "F7" => b"\x1b[18~".to_vec(),
        "F8" => b"\x1b[19~".to_vec(),
        "F9" => b"\x1b[20~".to_vec(),
        "F10" => b"\x1b[21~".to_vec(),
        "F11" => b"\x1b[23~".to_vec(),
        "F12" => b"\x1b[24~".to_vec(),

        // Single character
        key if key.len() == 1 => {
            let ch = key.chars().next().unwrap();
            if has_ctrl && ch.is_ascii_alphabetic() {
                // Ctrl+letter = ASCII 1-26
                vec![(ch.to_ascii_lowercase() as u8) - b'a' + 1]
            } else if has_ctrl && ch == '[' {
                vec![0x1b] // Ctrl+[ = Escape
            } else if has_ctrl && ch == ']' {
                vec![0x1d] // Ctrl+]
            } else if has_ctrl && ch == '\\' {
                vec![0x1c] // Ctrl+backslash
            } else {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            }
        }

        // Unknown key — try as literal
        other => other.as_bytes().to_vec(),
    };

    let mut result = Vec::new();
    for _ in 0..event.repeat.max(1) {
        if has_alt && !base.is_empty() {
            // Alt wraps in ESC prefix
            result.push(0x1b);
        }
        result.extend_from_slice(&base);
    }

    result
}

/// Encode text for bracketed paste.
pub fn encode_paste(text: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x1b[200~"); // begin bracketed paste
    bytes.extend_from_slice(text.as_bytes());
    bytes.extend_from_slice(b"\x1b[201~"); // end bracketed paste
    bytes
}

/// Encode text as typed characters (no bracketed paste).
pub fn encode_type(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

/// Encode a mouse event into terminal escape bytes (SGR mode).
pub fn encode_mouse_event(event: &MouseEvent) -> Vec<u8> {
    let button_code = match (event.event_type.as_str(), event.button.as_str()) {
        ("mouse_down", "left") => 0,
        ("mouse_down", "middle") => 1,
        ("mouse_down", "right") => 2,
        ("mouse_up", _) => 3,
        ("mouse_move", _) => 35,
        ("wheel_up", _) => 64,
        ("wheel_down", _) => 65,
        _ => return vec![],
    };

    let has_shift = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("shift"));
    let has_alt = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("alt"));
    let has_ctrl = event.modifiers.iter().any(|m| m.eq_ignore_ascii_case("ctrl"));

    let mut code = button_code;
    if has_shift { code |= 4; }
    if has_alt { code |= 8; }
    if has_ctrl { code |= 16; }

    let action = if event.event_type == "mouse_up" { 'm' } else { 'M' };

    // SGR encoding: \x1b[<code;x;yM or \x1b[<code;x;ym
    format!("\x1b[<{};{};{}{}", code, event.x + 1, event.y + 1, action)
        .into_bytes()
}

// ─── Terminal Session ────────────────────────────────────────────

/// Metadata about a terminal session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: u32,
    pub config: SessionConfig,
    pub running: bool,
    pub exit_code: Option<i32>,
}

/// A managed terminal session.
pub struct TerminalSession {
    pub id: u32,
    pub config: SessionConfig,
    handle: lcs_pty::ProcessHandle,
    /// Raw output bytes (append-only log).
    raw_output: Arc<TokioMutex<Vec<u8>>>,
    /// Channel for reading new output.
    stdout_rx: Arc<TokioMutex<mpsc::Receiver<Vec<u8>>>>,
    pub exited: bool,
    pub exit_code: Option<i32>,
}

impl TerminalSession {
    /// Send a structured key event.
    pub fn send_key(&self, event: &KeyEvent) -> Result<(), String> {
        let bytes = encode_key_event(event);
        self.send_raw(&bytes)
    }

    /// Send a mouse event.
    pub fn send_mouse(&self, event: &MouseEvent) -> Result<(), String> {
        let bytes = encode_mouse_event(event);
        if bytes.is_empty() {
            return Err("Unknown mouse event type".to_string());
        }
        self.send_raw(&bytes)
    }

    /// Type text (character by character, no bracketed paste).
    pub fn send_text(&self, text: &str) -> Result<(), String> {
        self.send_raw(&encode_type(text))
    }

    /// Paste text (with bracketed paste escape sequences).
    pub fn send_paste(&self, text: &str) -> Result<(), String> {
        self.send_raw(&encode_paste(text))
    }

    /// Send raw bytes to the PTY.
    pub fn send_raw(&self, data: &[u8]) -> Result<(), String> {
        let sender = self.handle.writer_sender();
        sender.try_send(data.to_vec())
            .map_err(|e| format!("Failed to send to terminal: {e}"))
    }

    /// Read new output bytes since last read. Non-blocking.
    pub async fn read_output(&self) -> Vec<u8> {
        let mut rx = self.stdout_rx.lock().await;
        let mut output = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            output.extend(chunk);
        }
        // Also append to raw log
        if !output.is_empty() {
            self.raw_output.lock().await.extend(&output);
        }
        output
    }

    /// Read output as lossy UTF-8 text.
    pub async fn read_text(&self) -> String {
        let bytes = self.read_output().await;
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Get all raw output bytes since session start.
    pub async fn full_output(&self) -> Vec<u8> {
        // Drain any pending first
        let _ = self.read_output().await;
        self.raw_output.lock().await.clone()
    }

    /// Get visible text from raw output (strip ANSI, extract last screen).
    pub async fn visible_text(&self) -> String {
        let text = self.read_text().await;
        // Simple ANSI strip for now — a proper terminal emulator would parse this
        strip_ansi(&text)
    }

    /// Resize the terminal.
    pub fn resize(&self, rows: u16, cols: u16) -> Result<(), String> {
        self.handle.resize(lcs_pty::TerminalSize { rows, cols })
            .map_err(|e| format!("Resize failed: {e}"))
    }

    /// Close stdin (signal EOF to the process).
    pub fn close_stdin(&self) {
        self.handle.close_stdin();
    }

    /// Request graceful termination.
    pub fn terminate(&self) {
        self.handle.request_terminate();
    }

    /// Force kill.
    pub fn kill(&self) {
        self.handle.terminate();
    }

    /// Check if the process has exited.
    pub fn has_exited(&self) -> bool {
        self.handle.has_exited() || self.exited
    }

    /// Get exit code if available.
    pub fn get_exit_code(&self) -> Option<i32> {
        self.handle.exit_code().or(self.exit_code)
    }

    /// Get session info.
    pub fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id,
            config: self.config.clone(),
            running: !self.has_exited(),
            exit_code: self.get_exit_code(),
        }
    }
}

/// Strip ANSI escape sequences from text.
pub fn strip_ansi(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                // CSI sequence — consume until letter
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c.is_ascii_alphabetic() || c == '~' {
                        break;
                    }
                }
            } else if chars.peek() == Some(&']') {
                // OSC sequence — consume until BEL or ST
                chars.next();
                while let Some(&c) = chars.peek() {
                    chars.next();
                    if c == '\x07' { break; }
                    if c == '\x1b' {
                        if chars.peek() == Some(&'\\') { chars.next(); }
                        break;
                    }
                }
            } else {
                // Other escape — skip next char
                chars.next();
            }
        } else if ch == '\r' {
            // Skip carriage returns
        } else {
            result.push(ch);
        }
    }

    result
}

// ─── Session Registry ────────────────────────────────────────────

/// Registry of all terminal sessions.
pub struct SessionRegistry {
    sessions: HashMap<u32, TerminalSession>,
    next_id: u32,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 1,
        }
    }

    /// Spawn a new terminal session.
    pub async fn spawn(&mut self, config: SessionConfig) -> Result<u32, String> {
        let id = self.next_id;
        self.next_id += 1;

        let cwd = config.cwd.as_deref().unwrap_or("/tmp");
        let shell = &config.shell;

        let args = if let Some(cmd) = &config.command {
            vec!["-c".to_string(), cmd.clone()]
        } else if config.login_shell {
            vec!["-l".to_string()]
        } else {
            vec![]
        };

        let mut env: HashMap<String, String> = HashMap::new();
        env.insert("TERM".to_string(), config.term.clone());
        // Inherit PATH and HOME
        if let Ok(path) = std::env::var("PATH") {
            env.insert("PATH".to_string(), path);
        }
        if let Ok(home) = std::env::var("HOME") {
            env.insert("HOME".to_string(), home);
        }
        // Add user env
        env.extend(config.env.clone());

        let arg0: Option<String> = None;
        let size = lcs_pty::TerminalSize { rows: config.rows, cols: config.cols };

        let spawned = lcs_pty::spawn_pty_process(shell, &args, Path::new(cwd), &env, &arg0, size)
            .await
            .map_err(|e| format!("Failed to spawn terminal: {e}"))?;

        let raw_output = Arc::new(TokioMutex::new(Vec::new()));
        let stdout_rx = Arc::new(TokioMutex::new(spawned.stdout_rx));

        // Watch for exit
        let exit_rx = spawned.exit_rx;

        let session = TerminalSession {
            id,
            config,
            handle: spawned.session,
            raw_output,
            stdout_rx,
            exited: false,
            exit_code: None,
        };

        self.sessions.insert(id, session);

        // Spawn exit watcher
        // Note: we can't easily update the session from a spawned task without Arc<Mutex>
        // The caller should check has_exited() which reads from the ProcessHandle directly.
        tokio::spawn(async move {
            let _ = exit_rx.await;
        });

        Ok(id)
    }

    pub fn get(&self, id: u32) -> Option<&TerminalSession> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut TerminalSession> {
        self.sessions.get_mut(&id)
    }

    pub fn list(&self) -> Vec<SessionInfo> {
        self.sessions.values().map(|s| s.info()).collect()
    }

    pub fn remove(&mut self, id: u32) -> Option<TerminalSession> {
        self.sessions.remove(&id)
    }

    pub fn clean(&mut self) -> usize {
        let dead: Vec<u32> = self.sessions.iter()
            .filter(|(_, s)| s.has_exited())
            .map(|(id, _)| *id)
            .collect();
        let count = dead.len();
        for id in dead {
            self.sessions.remove(&id);
        }
        count
    }
}
