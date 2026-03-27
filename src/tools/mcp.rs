//! MCP (Model Context Protocol) client — connects to external tool servers.
//!
//! Supports two transports:
//! - **Stdio**: spawns a child process, JSON-RPC over stdin/stdout
//! - **HTTP**: POST JSON-RPC to an endpoint (streamable-http / simple HTTP)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// Transport for communicating with an MCP server.
enum Transport {
    Stdio {
        child: Mutex<Child>,
    },
    Http {
        url: String,
        client: reqwest::Client,
        session_id: Mutex<Option<String>>,
    },
}

impl Transport {
    async fn request(&self, id: u64, method: &str, params: Value) -> Result<Value, String> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        match self {
            Transport::Stdio { child } => {
                let mut child = child.lock().map_err(|e| format!("Lock error: {e}"))?;
                let stdin = child.stdin.as_mut().ok_or("No stdin")?;
                let msg = serde_json::to_string(&body).map_err(|e| format!("Serialize: {e}"))?;
                writeln!(stdin, "{msg}").map_err(|e| format!("Write: {e}"))?;
                stdin.flush().map_err(|e| format!("Flush: {e}"))?;

                let stdout = child.stdout.as_mut().ok_or("No stdout")?;
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                reader
                    .read_line(&mut line)
                    .map_err(|e| format!("Read: {e}"))?;

                parse_response(&line)
            }
            Transport::Http {
                url,
                client,
                session_id,
            } => {
                let mut req = client
                    .post(url)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream");

                if let Ok(sid) = session_id.lock() {
                    if let Some(sid) = sid.as_ref() {
                        req = req.header("Mcp-Session-Id", sid.as_str());
                    }
                }

                let resp = req
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {e}"))?;

                if let Some(sid) = resp.headers().get("mcp-session-id") {
                    if let Ok(sid) = sid.to_str() {
                        if let Ok(mut stored) = session_id.lock() {
                            *stored = Some(sid.to_string());
                        }
                    }
                }

                let text = resp.text().await.map_err(|e| format!("HTTP read: {e}"))?;

                if text.starts_with("data:") || text.contains("\ndata:") {
                    for line in text.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                                if parsed.get("id").and_then(|i| i.as_u64()) == Some(id) {
                                    return parse_json_response(&parsed);
                                }
                            }
                        }
                    }
                    Err("No matching response in SSE stream".to_string())
                } else {
                    parse_response(&text)
                }
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        match self {
            Transport::Stdio { child } => {
                let mut child = child.lock().map_err(|e| format!("Lock: {e}"))?;
                let stdin = child.stdin.as_mut().ok_or("No stdin")?;
                let msg = serde_json::to_string(&body).map_err(|e| format!("Serialize: {e}"))?;
                writeln!(stdin, "{msg}").map_err(|e| format!("Write: {e}"))?;
                stdin.flush().map_err(|e| format!("Flush: {e}"))?;
                Ok(())
            }
            Transport::Http {
                url,
                client,
                session_id,
            } => {
                let mut req = client.post(url).header("Content-Type", "application/json");

                if let Ok(sid) = session_id.lock() {
                    if let Some(sid) = sid.as_ref() {
                        req = req.header("Mcp-Session-Id", sid.as_str());
                    }
                }

                let _ = req.json(&body).send().await;
                Ok(())
            }
        }
    }
}

fn parse_response(text: &str) -> Result<Value, String> {
    let response: Value = serde_json::from_str(text).map_err(|e| format!("Parse error: {e}"))?;
    parse_json_response(&response)
}

fn parse_json_response(response: &Value) -> Result<Value, String> {
    if let Some(error) = response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("MCP error: {msg}"));
    }
    response
        .get("result")
        .cloned()
        .ok_or("No result in response".to_string())
}

/// An MCP server connection.
pub struct McpServer {
    name: String,
    transport: Transport,
    request_id: AtomicU64,
}

/// A tool discovered from an MCP server.
pub struct McpTool {
    server: std::sync::Arc<McpServer>,
    tool_name: String,
    description: String,
    input_schema: Value,
}

/// Transport type for config.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a child process, communicate over stdin/stdout.
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        cwd: Option<PathBuf>,
    },
    /// POST JSON-RPC to an HTTP endpoint.
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
}

/// Configuration for connecting to an MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
}

impl McpServer {
    pub async fn connect(config: &McpServerConfig) -> Result<std::sync::Arc<Self>, String> {
        let transport = match &config.transport {
            McpTransport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let mut cmd = Command::new(command);
                cmd.args(args)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null());
                for (k, v) in env {
                    cmd.env(k, v);
                }
                if let Some(cwd) = cwd {
                    cmd.current_dir(cwd);
                }
                let child = cmd
                    .spawn()
                    .map_err(|e| format!("Failed to spawn '{}': {e}", config.name))?;
                Transport::Stdio {
                    child: Mutex::new(child),
                }
            }
            McpTransport::Http { url, headers } => {
                let mut builder = reqwest::ClientBuilder::new();
                let mut header_map = reqwest::header::HeaderMap::new();
                for (k, v) in headers {
                    if let (Ok(name), Ok(val)) = (
                        reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                        reqwest::header::HeaderValue::from_str(v),
                    ) {
                        header_map.insert(name, val);
                    }
                }
                builder = builder.default_headers(header_map);
                let client = builder.build().map_err(|e| format!("HTTP client: {e}"))?;
                Transport::Http {
                    url: url.clone(),
                    client,
                    session_id: Mutex::new(None),
                }
            }
        };

        let server = std::sync::Arc::new(Self {
            name: config.name.clone(),
            transport,
            request_id: AtomicU64::new(1),
        });

        let _init = server
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "replay", "version": "0.1.0" }
                }),
            )
            .await?;

        server
            .transport
            .notify("notifications/initialized", json!({}))
            .await?;

        Ok(server)
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        self.transport.request(id, method, params).await
    }

    pub async fn list_tools(self: &std::sync::Arc<Self>) -> Result<Vec<McpTool>, String> {
        let result = self.request("tools/list", json!({})).await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or("MCP server returned no tools")?;

        let mut out = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = tool.get("inputSchema").cloned().unwrap_or(json!({}));

            if !name.is_empty() {
                out.push(McpTool {
                    server: std::sync::Arc::clone(self),
                    tool_name: name,
                    description,
                    input_schema,
                });
            }
        }
        Ok(out)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, String> {
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;

        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(texts.join("\n"))
        } else {
            Ok(serde_json::to_string_pretty(&result).unwrap_or_default())
        }
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        if let Transport::Stdio { child } = &self.transport {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
        }
        let _ = &self.name;
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn to_param(&self) -> ToolParam {
        let schema = if let Some(props) = self.input_schema.get("properties") {
            let mut input = InputSchema::object();
            let required: Vec<String> = self
                .input_schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if let Some(obj) = props.as_object() {
                for (key, val) in obj {
                    let desc = val
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    if required.contains(key) {
                        input = input.required_string(key, desc);
                    } else {
                        input = input.optional_string(key, desc);
                    }
                }
            }
            input
        } else {
            InputSchema::object()
        };

        ToolParam::new(&self.tool_name, schema).with_description(&self.description)
    }

    async fn call(&self, input: HashMap<String, Value>) -> ToolResult {
        let args = Value::Object(input.into_iter().collect());
        match self.server.call_tool(&self.tool_name, args).await {
            Ok(text) => ToolResult::success(text),
            Err(e) => ToolResult::error(e),
        }
    }
}

/// Connect to MCP servers and collect all their tools.
pub async fn connect_servers(configs: &[McpServerConfig]) -> Vec<std::sync::Arc<dyn Tool>> {
    let mut tools: Vec<std::sync::Arc<dyn Tool>> = Vec::new();

    for config in configs {
        match McpServer::connect(config).await {
            Ok(server) => match server.list_tools().await {
                Ok(server_tools) => {
                    for tool in server_tools {
                        tools.push(std::sync::Arc::new(tool));
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to list tools from '{}': {e}", config.name);
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to connect to '{}': {e}", config.name);
            }
        }
    }

    tools
}

/// Built-in MCP server configurations.
pub fn builtin_servers() -> Vec<McpServerConfig> {
    vec![McpServerConfig {
        name: "deepwiki".to_string(),
        transport: McpTransport::Http {
            url: "https://mcp.deepwiki.com/mcp".to_string(),
            headers: HashMap::new(),
        },
    }]
}
