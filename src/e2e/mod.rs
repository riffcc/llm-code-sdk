//! End-to-end testing module with Playwright integration.
//!
//! This module provides browser automation tools that can be used by LLMs
//! for end-to-end testing tasks.
//!
//! # Feature Flag
//!
//! This module requires the `e2e` feature flag to be enabled:
//!
//! ```toml
//! [dependencies]
//! llm-code-sdk = { version = "0.1", features = ["e2e"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use llm_code_sdk::e2e::{BrowserSession, BrowserType};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a browser session
//!     let session = BrowserSession::new(BrowserType::Chromium).await?;
//!
//!     // Navigate to a page
//!     session.navigate("https://example.com").await?;
//!
//!     // Get page title
//!     let title = session.get_title().await?;
//!     println!("Page title: {}", title);
//!
//!     // Click a button
//!     session.click("button#submit").await?;
//!
//!     // Take a screenshot
//!     let screenshot = session.screenshot().await?;
//!
//!     Ok(())
//! }
//! ```

mod browser;
mod tools;

pub use browser::{BrowserSession, BrowserType};
pub use tools::{E2ETool, E2EToolRunner};
