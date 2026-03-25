//! Survey tool — structured choice capture for agents.
//!
//! Presents options to the user and waits for selection.
//! Designed for couch-mode / low-input sessions.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::{Tool, ToolResult};
use crate::types::{InputSchema, PropertySchema, ToolParam};

/// A single survey option.
#[derive(Debug, Clone)]
pub struct SurveyOption {
    /// Short label shown in the list.
    pub label: String,
    /// Optional description shown alongside.
    pub description: Option<String>,
}

/// A survey request from the agent.
#[derive(Debug, Clone)]
pub struct SurveyRequest {
    /// The question being asked.
    pub prompt: String,
    /// Available options.
    pub options: Vec<SurveyOption>,
    /// Whether multiple selections are allowed.
    pub multi: bool,
}

/// The user's response to a survey.
#[derive(Debug, Clone)]
pub struct SurveyResponse {
    /// Selected option indices (0-based).
    pub selected: Vec<usize>,
}

impl SurveyResponse {
    /// Get selected option labels from the original request.
    pub fn labels<'a>(&self, options: &'a [SurveyOption]) -> Vec<&'a str> {
        self.selected
            .iter()
            .filter_map(|&i| options.get(i).map(|o| o.label.as_str()))
            .collect()
    }
}

/// Callback that presents a survey to the user and returns their selection.
/// The host (Replay) provides this implementation.
pub type SurveyCallback = Arc<dyn Fn(SurveyRequest) -> SurveyResponse + Send + Sync>;

/// Tool that lets the agent present structured choices to the user.
pub struct SurveyTool {
    callback: Arc<Mutex<Option<SurveyCallback>>>,
}

impl SurveyTool {
    /// Create a new survey tool. Must call `set_callback` before use.
    pub fn new() -> Self {
        Self {
            callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a survey tool with a callback.
    pub fn with_callback(callback: SurveyCallback) -> Self {
        Self {
            callback: Arc::new(Mutex::new(Some(callback))),
        }
    }

    /// Set or replace the callback.
    pub async fn set_callback(&self, callback: SurveyCallback) {
        let mut cb = self.callback.lock().await;
        *cb = Some(callback);
    }
}

#[async_trait]
impl Tool for SurveyTool {
    fn name(&self) -> &str {
        "survey"
    }

    fn to_param(&self) -> ToolParam {
        let option_schema = PropertySchema::object()
            .property("label", PropertySchema::string().with_description("Short label for the option"), true)
            .property("description", PropertySchema::string().with_description("Description shown alongside the label"), false);

        ToolParam::new(
            "survey",
            InputSchema::object()
                .required_string("prompt", "The question to ask the user")
                .property(
                    "options",
                    PropertySchema::array(option_schema)
                        .with_description("List of options, each with a label and optional description"),
                    true,
                )
                .property(
                    "multi",
                    PropertySchema::boolean()
                        .with_description("Allow multiple selections (default: false)"),
                    false,
                ),
        )
        .with_description(
            "Present structured choices to the user. Use for decisions, confirmations, and routing. Prefer this over open-ended questions when options are clear.",
        )
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if prompt.is_empty() {
            return ToolResult::error("'prompt' is required");
        }

        let options: Vec<SurveyOption> = input
            .get("options")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            // Plain string option
                            SurveyOption { label: s.to_string(), description: None }
                        } else {
                            // Object with label + description
                            let label = v.get("label").and_then(|l| l.as_str()).unwrap_or("").to_string();
                            let description = v.get("description").and_then(|d| d.as_str()).map(|s| s.to_string());
                            SurveyOption { label, description }
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        if options.is_empty() {
            return ToolResult::error("'options' must be a non-empty array");
        }

        let multi = input
            .get("multi")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let request = SurveyRequest {
            prompt,
            options: options.clone(),
            multi,
        };

        let cb = self.callback.lock().await;
        match &*cb {
            Some(callback) => {
                let response = callback(request);
                let labels = response.labels(&options);
                if labels.is_empty() {
                    ToolResult::success("(no selection)")
                } else {
                    ToolResult::success(labels.join(", "))
                }
            }
            None => ToolResult::error("Survey not available — no input handler configured"),
        }
    }
}
