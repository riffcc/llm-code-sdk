//! E2E testing tools for LLM use.
//!
//! This module provides tool definitions that LLMs can use to interact with
//! browsers for end-to-end testing.

use super::browser::{BrowserSession, BrowserType, E2EResult};
use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, PropertySchema, ToolParam};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// E2E tool types for browser automation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum E2EAction {
    /// Navigate to a URL.
    Navigate { url: String },
    /// Click an element.
    Click { selector: String },
    /// Double-click an element.
    DoubleClick { selector: String },
    /// Fill a form field.
    Fill { selector: String, text: String },
    /// Type text character by character.
    Type { selector: String, text: String },
    /// Press a keyboard key.
    Press { selector: String, key: String },
    /// Get text content from an element.
    GetText { selector: String },
    /// Get input value from a form field.
    GetInputValue { selector: String },
    /// Get an attribute from an element.
    GetAttribute { selector: String, attribute: String },
    /// Check if an element is visible.
    IsVisible { selector: String },
    /// Check if an element is enabled.
    IsEnabled { selector: String },
    /// Check if a checkbox is checked.
    IsChecked { selector: String },
    /// Set checkbox checked state.
    SetChecked { selector: String, checked: bool },
    /// Select an option from a dropdown.
    SelectOption { selector: String, value: String },
    /// Hover over an element.
    Hover { selector: String },
    /// Focus an element.
    Focus { selector: String },
    /// Wait for an element to appear.
    WaitFor { selector: String },
    /// Count elements matching a selector.
    Count { selector: String },
    /// Take a screenshot.
    Screenshot { full_page: Option<bool> },
    /// Take a screenshot of an element.
    ScreenshotElement { selector: String },
    /// Get the current URL.
    GetUrl,
    /// Get the page title.
    GetTitle,
    /// Get the page HTML.
    GetHtml,
    /// Evaluate JavaScript and return result as string.
    Evaluate { expression: String },
    /// Reload the page.
    Reload,
}

/// Result from an E2E action.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum E2EActionResult {
    /// Operation succeeded with no return value.
    Success { success: bool },
    /// String result.
    Text { text: String },
    /// Optional string result.
    OptionalText { text: Option<String> },
    /// Boolean result.
    Boolean { value: bool },
    /// Number result.
    Number { value: usize },
    /// Binary data (base64 encoded).
    Binary { data: String, mime_type: String },
    /// Error result.
    Error { error: String },
}

/// E2E testing tool for LLM use.
pub struct E2ETool {
    session: Arc<Mutex<Option<BrowserSession>>>,
    browser_type: BrowserType,
}

impl E2ETool {
    /// Create a new E2E tool.
    pub fn new(browser_type: BrowserType) -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            browser_type,
        }
    }

    /// Create with Chromium browser.
    pub fn chromium() -> Self {
        Self::new(BrowserType::Chromium)
    }

    /// Create with Firefox browser.
    pub fn firefox() -> Self {
        Self::new(BrowserType::Firefox)
    }

    /// Create with WebKit browser.
    pub fn webkit() -> Self {
        Self::new(BrowserType::Webkit)
    }

    /// Ensure browser session is initialized.
    async fn ensure_session(&self) -> E2EResult<()> {
        let mut session = self.session.lock().await;
        if session.is_none() {
            *session = Some(BrowserSession::new(self.browser_type).await?);
        }
        Ok(())
    }

    /// Execute an E2E action.
    pub async fn execute(&self, action: E2EAction) -> E2EActionResult {
        if let Err(e) = self.ensure_session().await {
            return E2EActionResult::Error {
                error: e.to_string(),
            };
        }

        let session_guard = self.session.lock().await;
        let session = match session_guard.as_ref() {
            Some(s) => s,
            None => {
                return E2EActionResult::Error {
                    error: "Browser session not initialized".to_string(),
                }
            }
        };

        match action {
            E2EAction::Navigate { url } => match session.navigate(&url).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Click { selector } => match session.click(&selector).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::DoubleClick { selector } => match session.double_click(&selector).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Fill { selector, text } => match session.fill(&selector, &text).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Type { selector, text } => match session.type_text(&selector, &text).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Press { selector, key } => match session.press(&selector, &key).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::GetText { selector } => match session.get_text(&selector).await {
                Ok(text) => E2EActionResult::OptionalText { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::GetInputValue { selector } => match session.get_input_value(&selector).await
            {
                Ok(text) => E2EActionResult::Text { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::GetAttribute {
                selector,
                attribute,
            } => match session.get_attribute(&selector, &attribute).await {
                Ok(text) => E2EActionResult::OptionalText { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::IsVisible { selector } => match session.is_visible(&selector).await {
                Ok(value) => E2EActionResult::Boolean { value },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::IsEnabled { selector } => match session.is_enabled(&selector).await {
                Ok(value) => E2EActionResult::Boolean { value },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::IsChecked { selector } => match session.is_checked(&selector).await {
                Ok(value) => E2EActionResult::Boolean { value },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::SetChecked { selector, checked } => {
                match session.set_checked(&selector, checked).await {
                    Ok(()) => E2EActionResult::Success { success: true },
                    Err(e) => E2EActionResult::Error {
                        error: e.to_string(),
                    },
                }
            }

            E2EAction::SelectOption { selector, value } => {
                match session.select_option(&selector, &value).await {
                    Ok(()) => E2EActionResult::Success { success: true },
                    Err(e) => E2EActionResult::Error {
                        error: e.to_string(),
                    },
                }
            }

            E2EAction::Hover { selector } => match session.hover(&selector).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Focus { selector } => match session.focus(&selector).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::WaitFor { selector } => match session.wait_for_selector(&selector).await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Count { selector } => match session.count(&selector).await {
                Ok(value) => E2EActionResult::Number { value },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Screenshot { full_page } => {
                let result = if full_page.unwrap_or(false) {
                    session.screenshot_full_page().await
                } else {
                    session.screenshot().await
                };
                match result {
                    Ok(data) => E2EActionResult::Binary {
                        data: STANDARD.encode(&data),
                        mime_type: "image/png".to_string(),
                    },
                    Err(e) => E2EActionResult::Error {
                        error: e.to_string(),
                    },
                }
            }

            E2EAction::ScreenshotElement { selector } => {
                match session.screenshot_element(&selector).await {
                    Ok(data) => E2EActionResult::Binary {
                        data: STANDARD.encode(&data),
                        mime_type: "image/png".to_string(),
                    },
                    Err(e) => E2EActionResult::Error {
                        error: e.to_string(),
                    },
                }
            }

            E2EAction::GetUrl => match session.get_url().await {
                Ok(text) => E2EActionResult::Text { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::GetTitle => match session.get_title().await {
                Ok(text) => E2EActionResult::Text { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::GetHtml => match session.get_html().await {
                Ok(text) => E2EActionResult::Text { text },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },

            E2EAction::Evaluate { expression } => {
                match session.evaluate_value(&expression).await {
                    Ok(text) => E2EActionResult::Text { text },
                    Err(e) => E2EActionResult::Error {
                        error: e.to_string(),
                    },
                }
            }

            E2EAction::Reload => match session.reload().await {
                Ok(()) => E2EActionResult::Success { success: true },
                Err(e) => E2EActionResult::Error {
                    error: e.to_string(),
                },
            },
        }
    }

    /// Close the browser session.
    pub async fn close(&self) -> E2EResult<()> {
        let mut session = self.session.lock().await;
        if let Some(s) = session.take() {
            s.close().await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for E2ETool {
    fn name(&self) -> &str {
        "browser"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "browser",
            InputSchema::object()
                .property(
                    "action",
                    PropertySchema::string()
                        .with_description("The action to perform")
                        .with_enum(vec![
                            "navigate".to_string(),
                            "click".to_string(),
                            "double_click".to_string(),
                            "fill".to_string(),
                            "type".to_string(),
                            "press".to_string(),
                            "get_text".to_string(),
                            "get_input_value".to_string(),
                            "get_attribute".to_string(),
                            "is_visible".to_string(),
                            "is_enabled".to_string(),
                            "is_checked".to_string(),
                            "set_checked".to_string(),
                            "select_option".to_string(),
                            "hover".to_string(),
                            "focus".to_string(),
                            "wait_for".to_string(),
                            "count".to_string(),
                            "screenshot".to_string(),
                            "screenshot_element".to_string(),
                            "get_url".to_string(),
                            "get_title".to_string(),
                            "get_html".to_string(),
                            "evaluate".to_string(),
                            "reload".to_string(),
                        ]),
                    true,
                )
                .property(
                    "url",
                    PropertySchema::string().with_description("URL to navigate to"),
                    false,
                )
                .property(
                    "selector",
                    PropertySchema::string().with_description("CSS selector for element"),
                    false,
                )
                .property(
                    "text",
                    PropertySchema::string().with_description("Text to fill or type"),
                    false,
                )
                .property(
                    "key",
                    PropertySchema::string().with_description("Keyboard key to press"),
                    false,
                )
                .property(
                    "value",
                    PropertySchema::string().with_description("Value for select option"),
                    false,
                )
                .property(
                    "attribute",
                    PropertySchema::string().with_description("Attribute name to get"),
                    false,
                )
                .property(
                    "checked",
                    PropertySchema::boolean().with_description("Checked state for checkboxes"),
                    false,
                )
                .property(
                    "full_page",
                    PropertySchema::boolean().with_description("Take full page screenshot"),
                    false,
                )
                .property(
                    "expression",
                    PropertySchema::string().with_description("JavaScript expression to evaluate"),
                    false,
                ),
        )
        .with_description("Browser automation tool for E2E testing. Supports navigation, clicking, filling forms, taking screenshots, and more.")
    }

    async fn call(&self, input: HashMap<String, Value>) -> ToolResult {
        // Parse the action from input
        let action_str = input
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("navigate");

        let action = match action_str {
            "navigate" => {
                let url = input
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Navigate { url }
            }
            "click" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Click { selector }
            }
            "double_click" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::DoubleClick { selector }
            }
            "fill" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = input
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Fill { selector, text }
            }
            "type" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = input
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Type { selector, text }
            }
            "press" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let key = input
                    .get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Press { selector, key }
            }
            "get_text" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::GetText { selector }
            }
            "get_input_value" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::GetInputValue { selector }
            }
            "get_attribute" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let attribute = input
                    .get("attribute")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::GetAttribute {
                    selector,
                    attribute,
                }
            }
            "is_visible" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::IsVisible { selector }
            }
            "is_enabled" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::IsEnabled { selector }
            }
            "is_checked" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::IsChecked { selector }
            }
            "set_checked" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let checked = input
                    .get("checked")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                E2EAction::SetChecked { selector, checked }
            }
            "select_option" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let value = input
                    .get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::SelectOption { selector, value }
            }
            "hover" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Hover { selector }
            }
            "focus" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Focus { selector }
            }
            "wait_for" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::WaitFor { selector }
            }
            "count" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Count { selector }
            }
            "screenshot" => {
                let full_page = input.get("full_page").and_then(|v| v.as_bool());
                E2EAction::Screenshot { full_page }
            }
            "screenshot_element" => {
                let selector = input
                    .get("selector")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::ScreenshotElement { selector }
            }
            "get_url" => E2EAction::GetUrl,
            "get_title" => E2EAction::GetTitle,
            "get_html" => E2EAction::GetHtml,
            "evaluate" => {
                let expression = input
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                E2EAction::Evaluate { expression }
            }
            "reload" => E2EAction::Reload,
            _ => {
                return ToolResult::Error(format!("Unknown action: {}", action_str));
            }
        };

        let result = self.execute(action).await;
        match serde_json::to_string(&result) {
            Ok(json) => ToolResult::Success(json),
            Err(e) => ToolResult::Error(format!("Failed to serialize result: {}", e)),
        }
    }
}

/// E2E tool runner for managing browser sessions.
pub struct E2EToolRunner {
    tool: E2ETool,
}

impl E2EToolRunner {
    /// Create a new E2E tool runner with Chromium.
    pub fn new() -> Self {
        Self {
            tool: E2ETool::chromium(),
        }
    }

    /// Create with a specific browser type.
    pub fn with_browser(browser_type: BrowserType) -> Self {
        Self {
            tool: E2ETool::new(browser_type),
        }
    }

    /// Get the tool for registration with ToolRunner.
    pub fn tool(&self) -> &E2ETool {
        &self.tool
    }

    /// Close the browser session.
    pub async fn close(&self) -> E2EResult<()> {
        self.tool.close().await
    }
}

impl Default for E2EToolRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_action_serialize() {
        let action = E2EAction::Navigate {
            url: "https://example.com".to_string(),
        };
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("\"action\":\"navigate\""));
        assert!(json.contains("\"url\":\"https://example.com\""));
    }

    #[test]
    fn test_e2e_tool_to_param() {
        let tool = E2ETool::chromium();
        let params = tool.to_param();
        assert_eq!(params.name, "browser");
        assert!(params.description.is_some());
    }
}
