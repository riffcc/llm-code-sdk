//! HTTP client for LLM APIs.

mod messages;
mod models;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use thiserror::Error;

pub use messages::MessagesClient;
pub use models::{ModelInfo, ModelListParams, ModelListResponse, ModelsClient};

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
}

/// Default timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Default max retries.
pub const DEFAULT_MAX_RETRIES: u32 = 2;

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

    /// Make a POST request to the API.
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
            tracing::debug!("POST {} - Request: {}", url, &body_json[..body_json.len().min(500)]);
        }

        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Exponential backoff
                let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tokio::time::sleep(delay).await;
            }

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

            let error_text = response.text().await.unwrap_or_default();
            tracing::debug!("API Error response: {}", &error_text[..error_text.len().min(500)]);

            // Don't retry client errors (4xx) except rate limits
            if status.as_u16() == 401 {
                return Err(ClientError::InvalidApiKey);
            }

            if status.as_u16() == 429 {
                last_error = Some(ClientError::RateLimited { retry_after: None });
                continue; // Retry rate limits
            }

            if status.is_client_error() {
                return Err(ClientError::Api {
                    status: status.as_u16(),
                    message: error_text,
                });
            }

            // Server errors are retryable
            last_error = Some(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Err(last_error.unwrap_or_else(|| {
            ClientError::Configuration("Unknown error occurred".into())
        }))
    }

    /// Make a streaming POST request to the API.
    pub(crate) async fn post_stream<T>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<reqwest::Response>
    where
        T: serde::Serialize,
    {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );

        let response = self
            .http
            .post(&url)
            .headers(self.default_headers())
            .json(body)
            .send()
            .await?;

        let status = response.status();

        if status.is_success() {
            Ok(response)
        } else {
            let error_text = response.text().await.unwrap_or_default();

            if status.as_u16() == 401 {
                return Err(ClientError::InvalidApiKey);
            }

            if status.as_u16() == 429 {
                return Err(ClientError::RateLimited { retry_after: None });
            }

            Err(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            })
        }
    }

    /// Make a GET request to the API.
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
            if attempt > 0 {
                // Exponential backoff
                let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tokio::time::sleep(delay).await;
            }

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

            let error_text = response.text().await.unwrap_or_default();

            if status.as_u16() == 401 {
                return Err(ClientError::InvalidApiKey);
            }

            if status.as_u16() == 429 {
                last_error = Some(ClientError::RateLimited { retry_after: None });
                continue;
            }

            if status.is_client_error() {
                return Err(ClientError::Api {
                    status: status.as_u16(),
                    message: error_text,
                });
            }

            last_error = Some(ClientError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        Err(last_error.unwrap_or_else(|| {
            ClientError::Configuration("Unknown error occurred".into())
        }))
    }

    pub(crate) fn default_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).unwrap(),
        );
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

    /// Set the API format (Anthropic or OpenAI).
    pub fn format(mut self, format: ApiFormat) -> Self {
        self.format = format;
        self
    }

    /// Build the client.
    pub fn build(self) -> Result<Client> {
        // Allow empty API key for OpenAI-compatible local servers
        if self.api_key.is_empty() && self.format == ApiFormat::Anthropic {
            return Err(ClientError::Configuration("API key cannot be empty".into()));
        }

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
        assert!(result.is_err());
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
        let client = Client::builder("test-key")
            .max_retries(5)
            .build()
            .unwrap();

        assert_eq!(client.max_retries, 5);
    }
}
