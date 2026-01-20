//! Browser session management for E2E testing.

use playwright_rs::{
    Browser, BrowserContext, Page, Playwright, ScreenshotOptions, ScreenshotType,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Error type for E2E operations.
#[derive(Debug, thiserror::Error)]
pub enum E2EError {
    #[error("Playwright error: {0}")]
    Playwright(#[from] playwright_rs::Error),

    #[error("Browser not initialized")]
    NotInitialized,

    #[error("Page not available")]
    NoPage,

    #[error("Element not found: {0}")]
    ElementNotFound(String),

    #[error("Navigation failed: {0}")]
    NavigationFailed(String),

    #[error("Timeout waiting for: {0}")]
    Timeout(String),

    #[error("Screenshot failed: {0}")]
    ScreenshotFailed(String),

    #[error("Operation not supported: {0}")]
    NotSupported(String),
}

/// Result type for E2E operations.
pub type E2EResult<T> = std::result::Result<T, E2EError>;

/// Browser type selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BrowserType {
    #[default]
    Chromium,
    Firefox,
    Webkit,
}

/// Options for launching a browser session.
#[derive(Debug, Clone, Default)]
pub struct BrowserOptions {
    /// Whether to run in headless mode (default: true).
    /// Note: Currently not configurable in playwright-rs 0.8.
    pub headless: bool,
    /// Viewport width.
    pub viewport_width: Option<u32>,
    /// Viewport height.
    pub viewport_height: Option<u32>,
    /// User agent string.
    pub user_agent: Option<String>,
}

impl BrowserOptions {
    /// Create new browser options with defaults.
    pub fn new() -> Self {
        Self {
            headless: true,
            ..Default::default()
        }
    }

    /// Run with visible browser window.
    pub fn headful(mut self) -> Self {
        self.headless = false;
        self
    }

    /// Set viewport size.
    pub fn viewport(mut self, width: u32, height: u32) -> Self {
        self.viewport_width = Some(width);
        self.viewport_height = Some(height);
        self
    }
}

/// A browser session for E2E testing.
///
/// This provides a high-level API for browser automation that LLMs can use.
pub struct BrowserSession {
    #[allow(dead_code)]
    playwright: Playwright,
    browser: Browser,
    #[allow(dead_code)]
    context: BrowserContext,
    page: Arc<Mutex<Page>>,
}

impl BrowserSession {
    /// Create a new browser session with default options.
    pub async fn new(browser_type: BrowserType) -> E2EResult<Self> {
        Self::with_options(browser_type, BrowserOptions::new()).await
    }

    /// Create a new browser session with custom options.
    pub async fn with_options(
        browser_type: BrowserType,
        _options: BrowserOptions,
    ) -> E2EResult<Self> {
        let playwright = Playwright::launch().await?;

        let browser_type_obj = match browser_type {
            BrowserType::Chromium => playwright.chromium(),
            BrowserType::Firefox => playwright.firefox(),
            BrowserType::Webkit => playwright.webkit(),
        };

        // playwright-rs 0.8 uses simplified API without options
        let browser = browser_type_obj.launch().await?;
        let context = browser.new_context().await?;
        let page = context.new_page().await?;

        Ok(Self {
            playwright,
            browser,
            context,
            page: Arc::new(Mutex::new(page)),
        })
    }

    /// Navigate to a URL.
    pub async fn navigate(&self, url: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        page.goto(url, None)
            .await
            .map_err(|e| E2EError::NavigationFailed(e.to_string()))?;
        Ok(())
    }

    /// Get the current URL.
    pub async fn get_url(&self) -> E2EResult<String> {
        let page = self.page.lock().await;
        Ok(page.url())
    }

    /// Get the page title.
    pub async fn get_title(&self) -> E2EResult<String> {
        let page = self.page.lock().await;
        Ok(page.title().await?)
    }

    /// Click an element by selector.
    pub async fn click(&self, selector: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .click(None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Double-click an element by selector.
    pub async fn double_click(&self, selector: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .dblclick(None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Fill a form field with text.
    pub async fn fill(&self, selector: &str, text: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .fill(text, None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Type text character by character.
    /// Note: In playwright-rs 0.8, this is implemented using individual key presses.
    pub async fn type_text(&self, selector: &str, text: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        // Type each character individually using press
        for c in text.chars() {
            locator
                .press(&c.to_string(), None)
                .await
                .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        }
        Ok(())
    }

    /// Press a keyboard key.
    pub async fn press(&self, selector: &str, key: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .press(key, None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Get text content from an element.
    pub async fn get_text(&self, selector: &str) -> E2EResult<Option<String>> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .text_content()
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))
    }

    /// Get inner text from an element.
    pub async fn get_inner_text(&self, selector: &str) -> E2EResult<String> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .inner_text()
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))
    }

    /// Get input value from a form field.
    pub async fn get_input_value(&self, selector: &str) -> E2EResult<String> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .input_value(None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))
    }

    /// Check if an element is visible.
    pub async fn is_visible(&self, selector: &str) -> E2EResult<bool> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        Ok(locator.is_visible().await?)
    }

    /// Check if an element is enabled.
    pub async fn is_enabled(&self, selector: &str) -> E2EResult<bool> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        Ok(locator.is_enabled().await?)
    }

    /// Check if a checkbox/radio is checked.
    pub async fn is_checked(&self, selector: &str) -> E2EResult<bool> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        Ok(locator.is_checked().await?)
    }

    /// Set checkbox/radio checked state.
    pub async fn set_checked(&self, selector: &str, checked: bool) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .set_checked(checked, None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Select an option from a dropdown.
    pub async fn select_option(&self, selector: &str, value: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .select_option(value, None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Hover over an element.
    pub async fn hover(&self, selector: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .hover(None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))?;
        Ok(())
    }

    /// Focus an element by clicking it.
    /// Note: playwright-rs 0.8 doesn't have a direct focus method.
    pub async fn focus(&self, selector: &str) -> E2EResult<()> {
        // Focus by clicking the element
        self.click(selector).await
    }

    /// Wait for an element to be visible.
    /// Note: This is a simple polling implementation since wait_for isn't available.
    pub async fn wait_for_selector(&self, selector: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;

        // Poll for visibility with a timeout
        for _ in 0..60 {
            // ~30 seconds at 500ms intervals
            if locator.is_visible().await.unwrap_or(false) {
                return Ok(());
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Err(E2EError::Timeout(selector.to_string()))
    }

    /// Count elements matching a selector.
    pub async fn count(&self, selector: &str) -> E2EResult<usize> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        Ok(locator.count().await?)
    }

    /// Get an attribute value from an element.
    pub async fn get_attribute(
        &self,
        selector: &str,
        attribute: &str,
    ) -> E2EResult<Option<String>> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .get_attribute(attribute)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))
    }

    /// Take a screenshot of the page.
    pub async fn screenshot(&self) -> E2EResult<Vec<u8>> {
        let page = self.page.lock().await;
        let options = ScreenshotOptions {
            full_page: Some(false),
            screenshot_type: Some(ScreenshotType::Png),
            ..Default::default()
        };
        page.screenshot(Some(options))
            .await
            .map_err(|e| E2EError::ScreenshotFailed(e.to_string()))
    }

    /// Take a full-page screenshot.
    pub async fn screenshot_full_page(&self) -> E2EResult<Vec<u8>> {
        let page = self.page.lock().await;
        let options = ScreenshotOptions {
            full_page: Some(true),
            screenshot_type: Some(ScreenshotType::Png),
            ..Default::default()
        };
        page.screenshot(Some(options))
            .await
            .map_err(|e| E2EError::ScreenshotFailed(e.to_string()))
    }

    /// Take a screenshot of a specific element.
    pub async fn screenshot_element(&self, selector: &str) -> E2EResult<Vec<u8>> {
        let page = self.page.lock().await;
        let locator = page.locator(selector).await;
        locator
            .screenshot(None)
            .await
            .map_err(|_| E2EError::ElementNotFound(selector.to_string()))
    }

    /// Get the page HTML content.
    pub async fn get_html(&self) -> E2EResult<String> {
        let page = self.page.lock().await;
        let locator = page.locator("html").await;
        locator.inner_html().await.map_err(E2EError::Playwright)
    }

    /// Evaluate JavaScript on the page.
    ///
    /// Note: This evaluates a simple expression and returns the result as a string.
    pub async fn evaluate_expression(&self, expression: &str) -> E2EResult<()> {
        let page = self.page.lock().await;
        page.evaluate_expression(expression)
            .await
            .map_err(E2EError::Playwright)
    }

    /// Evaluate JavaScript and get the result as a string.
    pub async fn evaluate_value(&self, expression: &str) -> E2EResult<String> {
        let page = self.page.lock().await;
        page.evaluate_value(expression)
            .await
            .map_err(E2EError::Playwright)
    }

    /// Reload the page.
    pub async fn reload(&self) -> E2EResult<()> {
        let page = self.page.lock().await;
        page.reload(None).await?;
        Ok(())
    }

    /// Close the browser session.
    pub async fn close(self) -> E2EResult<()> {
        self.browser.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_options() {
        let options = BrowserOptions::new().headful().viewport(1920, 1080);

        assert!(!options.headless);
        assert_eq!(options.viewport_width, Some(1920));
        assert_eq!(options.viewport_height, Some(1080));
    }

    #[test]
    fn test_browser_type_serialize() {
        let chromium = BrowserType::Chromium;
        let json = serde_json::to_string(&chromium).unwrap();
        assert_eq!(json, "\"chromium\"");

        let firefox = BrowserType::Firefox;
        let json = serde_json::to_string(&firefox).unwrap();
        assert_eq!(json, "\"firefox\"");
    }
}
