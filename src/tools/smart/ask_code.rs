//! ask_code - evidence-grounded codebase Q&A.
//!
//! The goal is closer to DeepWiki's query tool than a raw search primitive,
//! but with a lighter contract: gather strong local evidence, answer only from
//! that evidence, and stay precise about uncertainty.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;

use super::{CodeLayer, SmartReadTool};
use crate::client::{AdaptiveConfig, ApiFormat, Client};
use crate::lcs::ipfs_ha::{
    discover_ipfs_ha_snapshot, load_ipfs_ha_snapshot, query_ipfs_ha_snapshot,
};
use crate::tools::standard::{GlobTool, GrepTool, ListDirectoryTool, ReadFileTool};
use crate::tools::{Tool, ToolResult, ToolRunner, ToolRunnerConfig};
use crate::types::{
    InputSchema, MessageCreateParams, MessageParam, PropertySchema, ToolChoice, ToolParam,
};

#[derive(Debug, Clone)]
struct SearchHit {
    path: String,
    line: usize,
    text: String,
    score: usize,
}

#[derive(Debug, Clone)]
struct EvidenceBlock {
    heading: String,
    path: Option<String>,
    start: Option<usize>,
    end: Option<usize>,
    body: String,
}

#[derive(Debug, Clone)]
struct AskCodeConfig {
    model: String,
    use_model: bool,
    agentic: bool,
    max_context_files: usize,
    answer_language: Option<String>,
    glass: String,
    snapshot_path: Option<PathBuf>,
}

/// Query-oriented codebase answer tool.
pub struct AskCodeTool {
    project_root: PathBuf,
    smart_read: SmartReadTool,
}

impl AskCodeTool {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        let project_root = project_root.into();
        Self {
            smart_read: SmartReadTool::new(&project_root),
            project_root,
        }
    }

    fn extract_terms(query: &str) -> Vec<String> {
        const STOPWORDS: &[&str] = &[
            "the", "and", "that", "with", "from", "this", "what", "when", "where", "which",
            "does", "work", "works", "need", "into", "about", "their", "there", "like",
            "just", "have", "then", "than", "your", "please", "show", "explain", "tell",
            "give", "make", "using", "used", "user", "codebase", "repo", "project",
        ];

        let mut raw = Vec::new();
        let mut current = String::new();
        let mut quoted = false;

        for ch in query.chars() {
            if ch == '"' {
                if !current.is_empty() {
                    raw.push(current.clone());
                    current.clear();
                }
                quoted = !quoted;
                continue;
            }

            let keep = ch.is_ascii_alphanumeric()
                || matches!(ch, '_' | ':' | '/' | '.' | '-' | '?');
            if keep || quoted {
                current.push(ch);
            } else if !current.is_empty() {
                raw.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            raw.push(current);
        }

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for token in raw {
            let normalized = token.trim().trim_matches('`').to_string();
            if normalized.len() < 3
                && !normalized.contains("::")
                && !normalized.contains('/')
                && !normalized.contains('?')
            {
                continue;
            }
            let lowered = normalized.to_ascii_lowercase();
            if STOPWORDS.contains(&lowered.as_str()) {
                continue;
            }
            if seen.insert(lowered.clone()) {
                out.push(normalized);
            }
        }

        out.truncate(8);
        out
    }

    fn infer_glass(query: &str, explicit: Option<&str>) -> String {
        if let Some(explicit) = explicit {
            return explicit.to_string();
        }

        let q = query.to_ascii_lowercase();
        if q.contains("deploy") || q.contains("ops") || q.contains("systemd") {
            "administrator".to_string()
        } else if q.contains("api") || q.contains("endpoint") || q.contains("/api/") {
            "api".to_string()
        } else if q.contains("proof") || q.contains("lean") || q.contains("theorem") {
            "lean".to_string()
        } else if q.contains("pin") || q.contains("flatfs") || q.contains("kubo")
            || q.contains("cluster") || q.contains("ipfs") || q.contains("storage")
        {
            "storage".to_string()
        } else {
            "developer".to_string()
        }
    }

    fn resolve_paths_input(input: &HashMap<String, serde_json::Value>) -> Vec<String> {
        input
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn config_from_input(&self, input: &HashMap<String, serde_json::Value>) -> AskCodeConfig {
        let glass = Self::infer_glass(
            input.get("query").and_then(|v| v.as_str()).unwrap_or_default(),
            input.get("glass").and_then(|v| v.as_str()),
        );
        let use_model = input
            .get("use_model")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let agentic = input
            .get("agentic")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let max_context_files = input
            .get("max_context_files")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(6)
            .clamp(1, 12);
        let model = input
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::env::var("ANTHROPIC_MODEL").ok())
            .or_else(|| std::env::var("LM_STUDIO_MODEL").ok())
            .or_else(|| std::env::var("OPENAI_MODEL").ok())
            .unwrap_or_else(|| "MiniMax-M2.5-highspeed".to_string());
        let snapshot_path = input
            .get("snapshot_path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);

        AskCodeConfig {
            model,
            use_model,
            agentic,
            max_context_files,
            answer_language: input
                .get("language")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            glass,
            snapshot_path,
        }
    }

    fn grep_term(&self, term: &str, limit: usize) -> Vec<SearchHit> {
        let output = Command::new("rg")
            .args([
                "-n",
                "--no-heading",
                "--color",
                "never",
                "-i",
                "-m",
                "3",
                "-g",
                "!target",
                "-g",
                "!node_modules",
                "-g",
                "!.git",
                term,
                ".",
            ])
            .current_dir(&self.project_root)
            .output();

        let Ok(output) = output else {
            return Vec::new();
        };
        if !output.status.success() {
            return Vec::new();
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut hits = Vec::new();
        for line in stdout.lines().take(limit) {
            let mut parts = line.splitn(3, ':');
            let Some(path) = parts.next() else { continue };
            let Some(line_no) = parts.next() else { continue };
            let Some(text) = parts.next() else { continue };
            let Ok(line) = line_no.parse::<usize>() else {
                continue;
            };

            let score = if text.to_ascii_lowercase().contains(&term.to_ascii_lowercase()) {
                term.len()
            } else {
                1
            };
            hits.push(SearchHit {
                path: path.trim_start_matches("./").to_string(),
                line,
                text: text.to_string(),
                score,
            });
        }

        hits
    }

    fn gather_hits(&self, query: &str, max_context_files: usize) -> Vec<SearchHit> {
        let terms = Self::extract_terms(query);
        let mut hits = Vec::new();
        for term in &terms {
            hits.extend(self.grep_term(term, 16));
        }

        hits.sort_by_key(|hit| Reverse((hit.score, usize::MAX.saturating_sub(hit.line))));
        let mut seen_paths = HashSet::new();
        let mut filtered = Vec::new();
        for hit in hits {
            if seen_paths.insert(hit.path.clone()) {
                filtered.push(hit);
            }
            if filtered.len() >= max_context_files {
                break;
            }
        }
        filtered
    }

    fn snippet_around(&self, relative_path: &str, line: usize, radius: usize) -> Option<String> {
        let full_path = self.project_root.join(relative_path);
        let content = fs::read_to_string(full_path).ok()?;
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return None;
        }

        let start = line.saturating_sub(radius + 1);
        let end = usize::min(lines.len(), line + radius);
        Some(lines[start..end].join("\n"))
    }

    fn build_file_evidence(&self, hit: &SearchHit) -> Vec<EvidenceBlock> {
        let mut blocks = Vec::new();
        let layer = if hit.path.ends_with(".lean") {
            CodeLayer::TheoryGraph
        } else {
            CodeLayer::Ast
        };

        if let Ok(view) = self.smart_read.read_at_layer(&hit.path, layer) {
            blocks.push(EvidenceBlock {
                heading: format!("Layer view for {}", hit.path),
                path: Some(hit.path.clone()),
                start: None,
                end: None,
                body: view.to_context(),
            });
        }

        if let Some(snippet) = self.snippet_around(&hit.path, hit.line, 3) {
            blocks.push(EvidenceBlock {
                heading: format!("Matched snippet for {}", hit.path),
                path: Some(hit.path.clone()),
                start: Some(hit.line.saturating_sub(3).max(1)),
                end: Some(hit.line + 3),
                body: format!("```\n{}\n```", snippet),
            });
        } else {
            blocks.push(EvidenceBlock {
                heading: format!("Matched line for {}", hit.path),
                path: Some(hit.path.clone()),
                start: Some(hit.line),
                end: Some(hit.line),
                body: hit.text.clone(),
            });
        }

        blocks
    }

    fn build_snapshot_evidence(
        &self,
        query: &str,
        snapshot_path: Option<&Path>,
    ) -> Vec<EvidenceBlock> {
        let snapshot_path = snapshot_path
            .map(PathBuf::from)
            .or_else(|| discover_ipfs_ha_snapshot(&self.project_root));

        let Some(snapshot_path) = snapshot_path else {
            return Vec::new();
        };

        let Ok(snapshot) = load_ipfs_ha_snapshot(&snapshot_path) else {
            return Vec::new();
        };

        query_ipfs_ha_snapshot(&snapshot, query)
            .into_iter()
            .map(|body| EvidenceBlock {
                heading: "IPFS HA stitched snapshot".to_string(),
                path: Some(snapshot_path.display().to_string()),
                start: None,
                end: None,
                body,
            })
            .collect()
    }

    fn build_evidence(
        &self,
        query: &str,
        explicit_paths: &[String],
        config: &AskCodeConfig,
    ) -> Vec<EvidenceBlock> {
        let mut blocks = Vec::new();

        if explicit_paths.is_empty() {
            for hit in self.gather_hits(query, config.max_context_files) {
                blocks.extend(self.build_file_evidence(&hit));
            }
        } else {
            for path in explicit_paths.iter().take(config.max_context_files) {
                let synthetic = SearchHit {
                    path: path.clone(),
                    line: 1,
                    text: String::new(),
                    score: 1,
                };
                blocks.extend(self.build_file_evidence(&synthetic));
            }
        }

        blocks.extend(self.build_snapshot_evidence(
            query,
            config.snapshot_path.as_deref(),
        ));

        if blocks.is_empty() {
            if let Ok(summary) = self.smart_read.read_codebase() {
                blocks.push(EvidenceBlock {
                    heading: "Codebase summary".to_string(),
                    path: None,
                    start: None,
                    end: None,
                    body: summary,
                });
            }
        }

        blocks.truncate(config.max_context_files * 3);
        blocks
    }

    fn format_evidence_for_prompt(&self, evidence: &[EvidenceBlock]) -> String {
        let mut out = String::new();
        for block in evidence {
            out.push_str("\n<context>\n");
            out.push_str(&format!("heading: {}\n", block.heading));
            if let Some(path) = &block.path {
                out.push_str(&format!("path: {}\n", path));
            }
            if let (Some(start), Some(end)) = (block.start, block.end) {
                out.push_str(&format!("lines: {}-{}\n", start, end));
            }
            out.push('\n');
            out.push_str(&block.body);
            out.push_str("\n</context>\n");
        }
        out
    }

    fn offline_answer(
        &self,
        query: &str,
        config: &AskCodeConfig,
        evidence: &[EvidenceBlock],
    ) -> String {
        let mut out = String::new();
        out.push_str("Answer\n\n");
        out.push_str(&format!(
            "No model backend was used for this `ask_code` query, so I’m returning the gathered evidence bundle directly. Inferred glass: `{}`.\n\n",
            config.glass
        ));
        out.push_str(&format!("Query: `{}`\n\n", query));
        for block in evidence {
            out.push_str(&format!("### {}\n\n", block.heading));
            if let Some(path) = &block.path {
                match (block.start, block.end) {
                    (Some(start), Some(end)) => {
                        out.push_str(&format!("Source: `{}` lines `{}-{}`\n\n", path, start, end));
                    }
                    _ => {
                        out.push_str(&format!("Source: `{}`\n\n", path));
                    }
                }
            }
            out.push_str(&block.body);
            out.push_str("\n\n");
        }
        out.push_str("Notes\n\n");
        out.push_str("- Set an LLM backend to get a synthesized answer instead of the raw evidence bundle.\n");
        out.push_str("- Supported paths here are Anthropic-compatible envs (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`) or local OpenAI-compatible envs (`LM_STUDIO_URL`, `LM_STUDIO_MODEL`).\n");
        out
    }

    fn model_client(&self, config: &AskCodeConfig) -> Option<Client> {
        let _ = config;
        if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
        {
            let base = std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| crate::client::DEFAULT_ANTHROPIC_BASE_URL.to_string());
            return Client::builder(token)
                .base_url(base)
                .format(ApiFormat::Anthropic)
                .build()
                .ok();
        }

        if let Ok(base) = std::env::var("LM_STUDIO_URL")
            .or_else(|_| std::env::var("LM_STUDIO_BASE_URL"))
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
        {
            return Client::openai_compatible(base).ok();
        }

        None
    }

    fn system_prompt(&self, config: &AskCodeConfig) -> String {
        let mut out = String::from(
            "You answer questions about the current codebase using only the supplied evidence.\n\
Prefer precise code references in backticks.\n\
Do not guess. If the evidence is insufficient, say so plainly.\n\
Explain repo-specific terms when they differ from generic usage.\n\
When a claim is backed by a specific snippet with path and lines, cite it inline using <cite repo=\"current\" path=\"FILE\" start=\"LINE\" end=\"LINE\" /> immediately after the sentence.\n\
Use CommonMark.\n\
Keep the response lean and concrete.\n\
Output exactly two top-level sections:\n\
Answer\n\
Notes\n",
        );

        out.push_str(&format!("\nActive glass: `{}`.\n", config.glass));
        if let Some(language) = &config.answer_language {
            out.push_str(&format!("Reply in `{}`.\n", language));
        }
        out
    }

    fn agentic_system_prompt(&self, config: &AskCodeConfig) -> String {
        let mut out = String::from(
            "You are an experienced software engineer answering questions about the current codebase.\n\
Use the available tools aggressively before answering. Prefer `smart_read` for structure, `mr_search` for history, and raw file/search tools when needed.\n\
Treat this like a repo query tool: gather evidence, then answer only from that evidence.\n\
Do not guess. If the codebase evidence is insufficient, say so clearly.\n\
Refer to code entities with precise backticked names and file paths.\n\
Keep the answer concise but specific.\n\
Use CommonMark.\n\
Output exactly two top-level sections:\n\
Answer\n\
Notes\n",
        );

        out.push_str(&format!("\nActive glass: `{}`.\n", config.glass));
        if let Some(language) = &config.answer_language {
            out.push_str(&format!("Reply in `{}`.\n", language));
        }
        out
    }

    fn agentic_tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(SmartReadTool::new(&self.project_root)) as Arc<dyn Tool>,
            Arc::new(super::MRSearchTool::new(&self.project_root)) as Arc<dyn Tool>,
            Arc::new(ReadFileTool::new(&self.project_root)) as Arc<dyn Tool>,
            Arc::new(ListDirectoryTool::new(&self.project_root)) as Arc<dyn Tool>,
            Arc::new(GlobTool::new(&self.project_root)) as Arc<dyn Tool>,
            Arc::new(GrepTool::new(&self.project_root)) as Arc<dyn Tool>,
        ]
    }

    async fn answer_with_model(
        &self,
        query: &str,
        config: &AskCodeConfig,
        evidence: &[EvidenceBlock],
    ) -> Option<String> {
        let client = self.model_client(config)?;
        let prompt = format!(
            "Repository root: `{}`\n\nUser query:\n{}\n\nEvidence:\n{}",
            self.project_root.display(),
            query,
            self.format_evidence_for_prompt(evidence)
        );

        let ask_params = MessageCreateParams {
            model: config.model.clone(),
            max_tokens: 2200,
            temperature: Some(0.1),
            system: Some(self.system_prompt(config).into()),
            messages: vec![MessageParam::user(prompt)],
            ..Default::default()
        };
        let message = client
            .messages()
            .create_adaptive(&ask_params, AdaptiveConfig::default())
            .await
            .ok()?;

        Some(message.all_text())
    }

    async fn answer_with_agentic_runner(
        &self,
        query: &str,
        config: &AskCodeConfig,
        evidence: &[EvidenceBlock],
    ) -> Option<String> {
        let client = self.model_client(config)?;
        let runner = ToolRunner::with_config(
            client,
            self.agentic_tools(),
            ToolRunnerConfig {
                max_iterations: Some(12),
                verbose: false,
                on_event: None,
                adaptive_config: AdaptiveConfig::default(),
                cancel: None,
            },
        );

        let prompt = format!(
            "Repository root: `{}`\n\nQuestion:\n{}\n\nInitial context already gathered:\n{}\n\nUse tools before answering unless the answer is already fully determined by the supplied evidence.\n",
            self.project_root.display(),
            query,
            self.format_evidence_for_prompt(evidence)
        );

        let message = runner
            .run(MessageCreateParams {
                model: config.model.clone(),
                max_tokens: 2600,
                temperature: Some(0.1),
                system: Some(self.agentic_system_prompt(config).into()),
                messages: vec![MessageParam::user(prompt)],
                tool_choice: Some(ToolChoice::Auto),
                ..Default::default()
            })
            .await
            .ok()?;

        Some(message.all_text())
    }
}

#[async_trait]
impl Tool for AskCodeTool {
    fn name(&self) -> &str {
        "ask_code"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "ask_code",
            InputSchema::object()
                .required_string("query", "Natural-language question about the current codebase")
                .property(
                    "paths",
                    PropertySchema::array(PropertySchema::string())
                        .with_description("Optional file paths to pin the query to"),
                    false,
                )
                .optional_string(
                    "glass",
                    "Optional lens override, e.g. developer, api, administrator, storage, security, lean",
                )
                .optional_string(
                    "language",
                    "Optional answer language override; otherwise match the query language",
                )
                .optional_string(
                    "model",
                    "Optional model override; otherwise use environment defaults",
                )
                .property(
                    "max_context_files",
                    PropertySchema::integer()
                        .with_description("Limit how many files are pulled into evidence (default: 6)"),
                    false,
                )
                .property(
                    "use_model",
                    PropertySchema::boolean()
                        .with_description("Whether to synthesize an answer with an LLM (default: true)"),
                    false,
                )
                .property(
                    "agentic",
                    PropertySchema::boolean()
                        .with_description("Whether to run ask_code through the full agentic tool loop (default: true)"),
                    false,
                )
                .optional_string(
                    "snapshot_path",
                    "Optional LCS snapshot JSON to fold into the answer, e.g. ipfs_ha_add_path.json",
                ),
        )
        .with_description(
            "Ask a real codebase question. By default this runs an agentic investigation loop over SmartRead and exploration tools, seeded with local evidence and optional stitched LCS snapshots. Use `smart_read` instead when you want a fast grounded structural read rather than a roaming query.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("").trim();
        if query.is_empty() {
            return ToolResult::error("query is required");
        }

        let config = self.config_from_input(&input);
        let explicit_paths = Self::resolve_paths_input(&input);
        let evidence = self.build_evidence(query, &explicit_paths, &config);

        if evidence.is_empty() {
            return ToolResult::error("No relevant local evidence found for ask_code query");
        }

        if config.use_model {
            let answer = if config.agentic {
                self.answer_with_agentic_runner(query, &config, &evidence)
                    .await
            } else {
                self.answer_with_model(query, &config, &evidence).await
            };

            if let Some(answer) = answer {
                return ToolResult::success(answer);
            }
        }

        ToolResult::success(self.offline_answer(query, &config, &evidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_ask_code_offline_with_rust_and_lean() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("main.rs"),
            r#"
pub fn shared_repo_mode() -> bool {
    true
}
"#,
        )
        .unwrap();

        fs::write(
            dir.path().join("Proof.lean"),
            r#"
theorem upstream_final_unchanged : True := by
  trivial
"#,
        )
        .unwrap();

        let tool = AskCodeTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert(
            "query".to_string(),
            serde_json::json!("How does upstream_final_unchanged relate to shared_repo_mode?"),
        );
        input.insert("use_model".to_string(), serde_json::json!(false));
        input.insert("agentic".to_string(), serde_json::json!(false));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("Answer"));
        assert!(content.contains("Proof.lean"));
        assert!(content.contains("theorem"));
        assert!(content.contains("main.rs"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_ask_code_live_minimax_if_configured() {
        if std::env::var("ANTHROPIC_AUTH_TOKEN").is_err() {
            return;
        }

        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("main.rs"),
            r#"
pub fn shared_repo_mode() -> bool {
    true
}
"#,
        )
        .unwrap();

        let tool = AskCodeTool::new(dir.path());
        let mut input = HashMap::new();
        input.insert(
            "query".to_string(),
            serde_json::json!("What does shared_repo_mode do?"),
        );
        input.insert("use_model".to_string(), serde_json::json!(true));
        input.insert("agentic".to_string(), serde_json::json!(true));

        let result = tool.call(input).await;
        assert!(!result.is_error());
        let content = result.to_content_string();
        assert!(content.contains("Answer"));
    }
}
