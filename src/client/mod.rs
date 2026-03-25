//! HTTP client for LLM APIs.

mod messages;
mod models;
mod throughput;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use tokio::sync::Semaphore;

pub use messages::{AdaptiveConfig, AdaptiveStreamConfig, MessagesClient};
pub use models::{ModelInfo, ModelListParams, ModelListResponse, ModelsClient};
pub use throughput::{ThroughputConfig, ThroughputStats, ThroughputTracker, global_throughput};

/// Default base URL for the Anthropic API.
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

/// Default base URL for Z.ai's Anthropic-compatible API.
pub const DEFAULT_ZAI_BASE_URL: &str = "https://api.z.ai/api/anthropic";

/// Default base URL for OpenAI API.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";

/// API version header value.
pub const API_VERSION: &str = "2023-06-01";

/// API format to use for requests/responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApiFormat {
    /// Anthropic Messages API format (default).
    #[default]
    Anthropic,
    /// OpenAI Chat Completions API format (compatible with LM Studio, Ollama, etc.).
    OpenAI,
    /// OpenAI Responses API format (/v1/responses) — used with ChatGPT OAuth tokens.
    OpenAIResponses,
}

/// Default timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Default max retries.
pub const DEFAULT_MAX_RETRIES: u32 = 5;

/// Default max concurrent requests (global rate limiter).
pub const DEFAULT_MAX_CONCURRENT: usize = 4;

/// Global rate limiter - limits concurrent API requests across all clients.
static GLOBAL_SEMAPHORE: std::sync::OnceLock<Semaphore> = std::sync::OnceLock::new();

/// Global backoff timestamp - when we're rate limited, all requests wait.
static RATE_LIMIT_UNTIL: AtomicU64 = AtomicU64::new(0);

fn get_global_semaphore() -> &'static Semaphore {
    GLOBAL_SEMAPHORE.get_or_init(|| Semaphore::new(DEFAULT_MAX_CONCURRENT))
}

/// Check if we're in a global rate limit backoff period.
fn check_rate_limit_backoff() -> Option<std::time::Duration> {
    let until = RATE_LIMIT_UNTIL.load(Ordering::Relaxed);
    if until == 0 {
        return None;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if now < until {
        Some(std::time::Duration::from_secs(until - now))
    } else {
        RATE_LIMIT_UNTIL.store(0, Ordering::Relaxed);
        None
    }
}

/// Set global rate limit backoff.
fn set_rate_limit_backoff(seconds: u64) {
    let until = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + seconds;
    RATE_LIMIT_UNTIL.store(until, Ordering::Relaxed);
}

/// Errors that can occur when using the client.
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Rate limited: retry after {retry_after:?} seconds")]
    RateLimited { retry_after: Option<u64> },

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Operation cancelled")]
    Cancelled,
}

/// Result type for client operations.
pub type Result<T> = std::result::Result<T, ClientError>;

/// The main client for interacting with LLM APIs.
#[derive(Debug, Clone)]
pub struct Client {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) max_retries: u32,
    pub(crate) format: ApiFormat,
    /// Optional account ID for ChatGPT OAuth (OpenAI Responses API).
    pub(crate) account_id: Option<String>,
}

impl Client {
    /// Get the API format.
    pub fn format(&self) -> ApiFormat {
        self.format
    }
}

impl Client {
    /// Create a new client with the given API key.
    ///
    /// Uses the default Anthropic API base URL.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        ClientBuilder::new(api_key).build()
    }

    /// Create a new client builder.
    pub fn builder(api_key: impl Into<String>) -> ClientBuilder {
        ClientBuilder::new(api_key)
    }

    /// Create a client configured for Z.ai's Anthropic-compatible API.
    pub fn zai(api_key: impl Into<String>) -> Result<Self> {
        ClientBuilder::new(api_key)
            .base_url(DEFAULT_ZAI_BASE_URL)
            .build()
    }

    /// Create a client configured for OpenAI-compatible APIs (LM Studio, Ollama, etc.).
    pub fn openai_compatible(base_url: impl Into<String>) -> Result<Self> {
        ClientBuilder::new("not-required")
            .base_url(base_url)
            .format(ApiFormat::OpenAI)
            .build()
    }

    /// Get the messages resource for creating messages.
    pub fn messages(&self) -> MessagesClient<'_> {
        MessagesClient::new(self)
    }

    /// Get the models resource for listing and retrieving models.
    pub fn models(&self) -> ModelsClient<'_> {
        ModelsClient::new(self)
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the API key (for internal use).
    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Make a POST request to the API with global rate limiting.
    pub(crate) async fn post<T, R>(&self, path: &str, body: &T) -> Result<R>
    where
        T: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        if let Ok(body_json) = serde_json::to_string(body) {
            tracing::debug!(
                "POST {} - Request: {}",
                url,
                &body_json[..body_json.len().min(500)]
            );
        }

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Check global rate limit backoff
            if let Some(wait) = check_rate_limit_backoff() {
                tracing::info!("Global rate limit backoff: waiting {}s", wait.as_secs());
                tokio::time::sleep(wait).await;
            }

            if attempt > 0 {
                // Exponential backoff: 2s, 4s, 8s, 16s, 32s
                let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                tracing::debug!(
                    "Retry attempt {} after {}s backoff",
                    attempt,
                    delay.as_secs()
                );
                tokio::time::sleep(delay).await;
            }

            // Acquire global semaphore permit
            let _permit = get_global_semaphore().acquire().await.unwrap();

            let response = match self
                .http
                .post(&url)
                .headers(self.default_headers())
                .json(body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(ClientError::Request(e));
                    continue;
                }
            };

            let status = response.status();
            tracing::debug!("API Response status: {}", status);

            if status.is_success() {
                let text = response.text().await.map_err(ClientError::from)?;
                tracing::debug!(target: "llm_code_sdk", "API response: {}", &text[..text.len().min(500)]);
                return serde_json::from_str(&text).map_err(ClientError::from);
            }

            // Parse Retry-After header before consuming response
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let error_text = response.text().await.unwrap_or_default();
            tracing::debug!(
                "API Error response: {}",
                &error_text[..error_text.len().min(500)]
            );

            // Don't retry client errors (4xx) except rate limits
            if status.as_u16() == 401 {
                tracing::warn!("401 Unauthorized: {}", &error_text[..error_text.len().min(200)]);
                return Err(ClientError::Api {
                    status: 401,
                    message: if error_text.is_empty() { "Invalid API key".into() } else { error_text },
                });
            }

            if status.as_u16() == 429 {
                let wait_secs = retry_after.unwrap_or(30);
                tracing::warn!("Rate limited, setting global backoff for {}s", wait_secs);
                set_rate_limit_backoff(wait_secs);
                last_error = Some(ClientError::RateLimited { retry_after });
                continue;
            }

            // Retry server errors (5xx)
            if status.is_server_error() {
                tracing::warn!("Server error {}, will retry", status.as_u16());
                last_error = Some(ClientError::Api {
                    status: status.as_u16(),
                    message: error_text,
                });
                continue;
            }

            // Other client errors are not retryable
            return Err(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Err(last_error
            .unwrap_or_else(|| ClientError::Configuration("Unknown error occurred".into())))
    }

    /// Make a streaming POST request to the API with global rate limiting and retries.
    pub(crate) async fn post_stream<T>(&self, path: &str, body: &T) -> Result<reqwest::Response>
    where
        T: serde::Serialize,
    {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Check global rate limit backoff
            if let Some(wait) = check_rate_limit_backoff() {
                tracing::info!("Global rate limit backoff: waiting {}s", wait.as_secs());
                tokio::time::sleep(wait).await;
            }

            if attempt > 0 {
                // Exponential backoff: 2s, 4s, 8s, 16s, 32s
                let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                tracing::info!(
                    "Stream retry attempt {} after {}s backoff",
                    attempt,
                    delay.as_secs()
                );
                tokio::time::sleep(delay).await;
            }

            // Acquire global semaphore permit
            let _permit = get_global_semaphore().acquire().await.unwrap();

            let response = match self
                .http
                .post(&url)
                .headers(self.default_headers())
                .json(body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(ClientError::Request(e));
                    continue;
                }
            };

            let status = response.status();

            if status.is_success() {
                return Ok(response);
            }

            // Parse Retry-After header before consuming response
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let error_text = response.text().await.unwrap_or_default();

            if status.as_u16() == 401 {
                return Err(ClientError::InvalidApiKey);
            }

            if status.as_u16() == 429 {
                let wait_secs = retry_after.unwrap_or(30);
                tracing::warn!(
                    "Rate limited on stream, setting global backoff for {}s",
                    wait_secs
                );
                set_rate_limit_backoff(wait_secs);
                last_error = Some(ClientError::RateLimited { retry_after });
                continue;
            }

            // Retry server errors (5xx)
            if status.is_server_error() {
                tracing::warn!("Server error {} on stream, will retry", status.as_u16());
                last_error = Some(ClientError::Api {
                    status: status.as_u16(),
                    message: error_text,
                });
                continue;
            }

            // Other client errors are not retryable
            return Err(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Err(last_error
            .unwrap_or_else(|| ClientError::Configuration("Unknown error occurred".into())))
    }

    /// Make a GET request to the API with global rate limiting.
    pub(crate) async fn get<R>(&self, path: &str) -> Result<R>
    where
        R: serde::de::DeserializeOwned,
    {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Check global rate limit backoff
            if let Some(wait) = check_rate_limit_backoff() {
                tracing::info!("Global rate limit backoff: waiting {}s", wait.as_secs());
                tokio::time::sleep(wait).await;
            }

            if attempt > 0 {
                // Exponential backoff: 2s, 4s, 8s, 16s, 32s
                let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                tracing::debug!(
                    "Retry attempt {} after {}s backoff",
                    attempt,
                    delay.as_secs()
                );
                tokio::time::sleep(delay).await;
            }

            // Acquire global semaphore permit
            let _permit = get_global_semaphore().acquire().await.unwrap();

            let response = match self
                .http
                .get(&url)
                .headers(self.default_headers())
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(ClientError::Request(e));
                    continue;
                }
            };

            let status = response.status();

            if status.is_success() {
                let text = response.text().await.map_err(ClientError::from)?;
                tracing::debug!("API Response: {}", &text[..text.len().min(1000)]);
                return serde_json::from_str(&text).map_err(ClientError::from);
            }

            // Parse Retry-After header before consuming response
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let error_text = response.text().await.unwrap_or_default();

            if status.as_u16() == 401 {
                return Err(ClientError::InvalidApiKey);
            }

            if status.as_u16() == 429 {
                let wait_secs = retry_after.unwrap_or(30);
                tracing::warn!("Rate limited, setting global backoff for {}s", wait_secs);
                set_rate_limit_backoff(wait_secs);
                last_error = Some(ClientError::RateLimited { retry_after });
                continue;
            }

            // Retry server errors (5xx)
            if status.is_server_error() {
                tracing::warn!("Server error {}, will retry", status.as_u16());
                last_error = Some(ClientError::Api {
                    status: status.as_u16(),
                    message: error_text,
                });
                continue;
            }

            // Other client errors are not retryable
            return Err(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Err(last_error
            .unwrap_or_else(|| ClientError::Configuration("Unknown error occurred".into())))
    }

    pub(crate) fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if !self.api_key.trim().is_empty() {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
            );

            if self.format == ApiFormat::Anthropic {
                headers.insert(
                    "x-api-key",
                    HeaderValue::from_str(&self.api_key).unwrap(),
                );
            }
        }

        if self.format == ApiFormat::Anthropic {
            headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        }

        if self.format == ApiFormat::OpenAIResponses {
            if let Some(ref acc) = self.account_id {
                if let Ok(hv) = HeaderValue::from_str(acc) {
                    headers.insert("ChatGPT-Account-Id", hv);
                }
            }
        }

        headers
    }
}

/// Builder for creating a configured client.
#[derive(Debug)]
pub struct ClientBuilder {
    api_key: String,
    base_url: Option<String>,
    timeout: Option<std::time::Duration>,
    max_retries: Option<u32>,
    format: ApiFormat,
    account_id: Option<String>,
}

impl ClientBuilder {
    /// Create a new client builder with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            timeout: None,
            max_retries: None,
            format: ApiFormat::default(),
            account_id: None,
        }
    }

    /// Set the base URL for the API.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the maximum number of retries for failed requests.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set the API format (Anthropic, OpenAI, or OpenAIResponses).
    pub fn format(mut self, format: ApiFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the account ID for ChatGPT OAuth (OpenAI Responses API).
    pub fn account_id(mut self, id: impl Into<String>) -> Self {
        self.account_id = Some(id.into());
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<Client> {
        let timeout = self
            .timeout
            .unwrap_or(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| ClientError::Configuration(e.to_string()))?;

        Ok(Client {
            http,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_ANTHROPIC_BASE_URL.to_string()),
            api_key: self.api_key,
            max_retries: self.max_retries.unwrap_or(DEFAULT_MAX_RETRIES),
            format: self.format,
            account_id: self.account_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = Client::builder("test-key")
            .base_url("https://custom.api.com")
            .build()
            .unwrap();

        assert_eq!(client.base_url(), "https://custom.api.com");
    }

    #[test]
    fn test_client_builder_empty_key() {
        let result = Client::builder("").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_zai_client() {
        let client = Client::zai("test-key").unwrap();
        assert_eq!(client.base_url(), DEFAULT_ZAI_BASE_URL);
    }

    #[test]
    fn test_default_client() {
        let client = Client::new("test-key").unwrap();
        assert_eq!(client.base_url(), DEFAULT_ANTHROPIC_BASE_URL);
    }

    #[test]
    fn test_client_builder_with_retries() {
        let client = Client::builder("test-key").max_retries(5).build().unwrap();

        assert_eq!(client.max_retries, 5);
    }
}
