//! Messages API client.

use futures::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use super::{ApiFormat, Client, ClientError, Result};
use crate::streaming::{MessageStream, RawStreamEvent};
use crate::types::openai::{OpenAIChatRequest, OpenAIChatResponse};
use crate::types::{Message, MessageCreateParams};

/// Configuration for adaptive streaming with stall detection.
#[derive(Debug, Clone)]
pub struct AdaptiveStreamConfig {
    /// Initial stall timeout - abort if no tokens for this long (default: 30s)
    pub initial_stall_timeout: Duration,
    /// Multiplier for stall timeout on each retry (default: 1.5)
    pub timeout_multiplier: f64,
    /// Maximum stall timeout (default: 120s)
    pub max_stall_timeout: Duration,
    /// Maximum number of retries (default: 5)
    pub max_retries: u32,
}

impl Default for AdaptiveStreamConfig {
    fn default() -> Self {
        Self {
            initial_stall_timeout: Duration::from_secs(30),
            timeout_multiplier: 1.5,
            max_stall_timeout: Duration::from_secs(120),
            max_retries: 5,
        }
    }
}

/// Configuration for adaptive (non-streaming) requests with timeout and retry.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Whether to use token-based timeout calculation (default: true).
    /// When true, timeout is calculated based on max_tokens and recent throughput.
    /// When false, uses fixed initial_timeout.
    pub use_token_based_timeout: bool,
    /// Fallback timeout when token-based calculation is disabled or no data (default: 60s)
    pub fallback_timeout: Duration,
    /// Multiplier for timeout on each retry (default: 1.5)
    pub timeout_multiplier: f64,
    /// Maximum timeout regardless of calculation (default: 600s)
    pub max_timeout: Duration,
    /// Minimum timeout regardless of calculation (default: 30s)
    pub min_timeout: Duration,
    /// Maximum number of retries (default: 5)
    pub max_retries: u32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            use_token_based_timeout: true,
            fallback_timeout: Duration::from_secs(60),
            timeout_multiplier: 1.5,
            max_timeout: Duration::from_secs(600),
            min_timeout: Duration::from_secs(30),
            max_retries: 5,
        }
    }
}

/// Client for the messages endpoint.
#[derive(Debug)]
pub struct MessagesClient<'a> {
    client: &'a Client,
}

impl<'a> MessagesClient<'a> {
    pub(crate) fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Create a new message.
    ///
    /// This sends a request to the API and returns the complete response.
    /// Automatically handles Anthropic vs OpenAI format based on client configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let message = client.messages().create(MessageCreateParams {
    ///     model: "glm-4-plus".into(),
    ///     max_tokens: 1024,
    ///     messages: vec![MessageParam::user("Hello!")],
    ///     ..Default::default()
    /// }).await?;
    ///
    /// println!("Response: {:?}", message.text());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create(&self, params: MessageCreateParams) -> Result<Message> {
        match self.client.format {
            ApiFormat::Anthropic => self.client.post("/v1/messages", &params).await,
            ApiFormat::OpenAI => {
                let openai_request = OpenAIChatRequest::from(&params);
                tracing::debug!(target: "llm_code_sdk", "OpenAI request: {:?}", openai_request);
                let response: OpenAIChatResponse = self
                    .client
                    .post("/chat/completions", &openai_request)
                    .await?;
                Ok(Message::from(response))
            }
            ApiFormat::OpenAIResponses => {
                use crate::types::openai_responses::{ResponsesRequest, ResponsesResponse, OutputItem, OutputContent, ResponsesUsage};

                let request = ResponsesRequest::from(&params);
                tracing::debug!(target: "llm_code_sdk", "OpenAI Responses SSE request model={}", request.model);

                let url = format!(
                    "{}/{}",
                    self.client.base_url.trim_end_matches('/'),
                    "codex/responses"
                );

                let resp = self.client.http
                    .post(&url)
                    .headers(self.client.default_headers())
                    .json(&request)
                    .send()
                    .await
                    .map_err(ClientError::from)?;

                let status = resp.status();
                if !status.is_success() {
                    let error_text = resp.text().await.unwrap_or_default();
                    return Err(ClientError::Api {
                        status: status.as_u16(),
                        message: error_text,
                    });
                }

                // Parse SSE stream, accumulate into a ResponsesResponse
                let mut output: Vec<OutputItem> = Vec::new();
                let mut usage: Option<ResponsesUsage> = None;
                let mut response_id = String::new();
                let mut model_name = None;
                let mut final_status = "completed".to_string();

                let body = resp.text().await.map_err(ClientError::from)?;
                tracing::debug!(target: "llm_code_sdk", "Responses SSE body length: {}", body.len());

                let mut current_event_type = String::new();
                for line in body.lines() {
                    let line = line.trim();

                    // SSE event type line
                    if let Some(et) = line.strip_prefix("event: ") {
                        current_event_type = et.to_string();
                        continue;
                    }

                    if !line.starts_with("data: ") {
                        if line.is_empty() {
                            current_event_type.clear();
                        }
                        continue;
                    }
                    let data = &line[6..];
                    if data == "[DONE]" {
                        break;
                    }

                    let event: serde_json::Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    // Event type from JSON `type` field, falling back to SSE `event:` line
                    let event_type = event.get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&current_event_type);

                    match event_type {
                        "response.output_item.done" => {
                            if let Some(item) = event.get("item") {
                                if let Ok(output_item) = serde_json::from_value::<OutputItem>(item.clone()) {
                                    output.push(output_item);
                                }
                            }
                        }
                        "response.completed" => {
                            if let Some(resp_obj) = event.get("response") {
                                tracing::debug!(target: "llm_code_sdk",
                                    "response.completed usage: {:?}",
                                    resp_obj.get("usage"));
                                response_id = resp_obj.get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                model_name = resp_obj.get("model")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                final_status = resp_obj.get("status")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("completed")
                                    .to_string();
                                if let Some(u) = resp_obj.get("usage") {
                                    usage = serde_json::from_value(u.clone()).ok();
                                }
                            }
                        }
                        "response.failed" => {
                            let msg = event.get("response")
                                .and_then(|r| r.get("error"))
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown error");
                            return Err(ClientError::Api {
                                status: 400,
                                message: msg.to_string(),
                            });
                        }
                        _ => {}
                    }
                }

                let responses_resp = ResponsesResponse {
                    id: response_id,
                    status: final_status,
                    output,
                    usage,
                    model: model_name,
                    error: None,
                    incomplete_details: None,
                };

                Ok(Message::from(responses_resp))
            }
        }
    }

    /// Create a message with adaptive timeout and retry.
    ///
    /// Wraps `create()` with timeout detection and half-exponential backoff retry.
    /// If a request times out, it's retried with an increased timeout.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
    /// use llm_code_sdk::client::AdaptiveConfig;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let message = client.messages().create_adaptive(
    ///     MessageCreateParams {
    ///         model: "glm-4-plus".into(),
    ///         max_tokens: 1024,
    ///         messages: vec![MessageParam::user("Hello!")],
    ///         ..Default::default()
    ///     },
    ///     AdaptiveConfig::default(),
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create_adaptive(
        &self,
        params: MessageCreateParams,
        config: AdaptiveConfig,
    ) -> Result<Message> {
        use super::throughput::global_throughput;

        let tracker = global_throughput();
        let max_tokens = params.max_tokens;

        // Calculate initial timeout based on token count and recent throughput
        let base_timeout = if config.use_token_based_timeout {
            let calculated = tracker.expected_timeout(max_tokens);
            tracing::debug!(
                "Token-based timeout: {} tokens at {:.1} tok/s = {:.1}s (clamped to {:?})",
                max_tokens,
                tracker.tokens_per_second(),
                max_tokens as f64 / tracker.tokens_per_second(),
                calculated
            );
            calculated
        } else {
            config.fallback_timeout
        };

        let mut current_timeout = base_timeout.clamp(config.min_timeout, config.max_timeout);
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            let start = Instant::now();

            tracing::debug!(
                "Adaptive create attempt {} with {:.1}s timeout (max_tokens={})",
                attempt,
                current_timeout.as_secs_f64(),
                max_tokens
            );

            match tokio::time::timeout(current_timeout, self.create(params.clone())).await {
                Ok(Ok(message)) => {
                    // Record successful inference to throughput tracker
                    let duration = start.elapsed();
                    let output_tokens = message.usage.output_tokens as u32;

                    if output_tokens > 0 {
                        tracker.record(output_tokens, duration);
                        tracing::debug!(
                            "Inference complete: {} tokens in {:.1}s ({:.1} tok/s)",
                            output_tokens,
                            duration.as_secs_f64(),
                            output_tokens as f64 / duration.as_secs_f64()
                        );
                    }

                    return Ok(message);
                }
                Ok(Err(e)) => {
                    // API error - check if retryable
                    let is_retryable = match &e {
                        ClientError::RateLimited { .. } => true,
                        ClientError::Api { status, .. } => *status >= 500,
                        _ => false,
                    };

                    if !is_retryable || attempt >= config.max_retries {
                        return Err(e);
                    }

                    tracing::info!("Retryable error on attempt {}: {}", attempt, e);
                }
                Err(_) => {
                    // Timeout - likely a stall
                    if attempt >= config.max_retries {
                        let stats = tracker.stats();
                        tracing::warn!(
                            "Request timed out after {} attempts. Throughput stats: {:.1} tok/s ({} samples)",
                            attempt,
                            stats.tokens_per_second,
                            stats.sample_count
                        );
                        return Err(ClientError::Stream(format!(
                            "Request timed out after {} attempts (last timeout: {:.1}s, expected throughput: {:.1} tok/s)",
                            attempt,
                            current_timeout.as_secs_f64(),
                            stats.tokens_per_second
                        )));
                    }

                    tracing::info!(
                        "Request timeout after {:.1}s (attempt {}), will retry with increased timeout",
                        current_timeout.as_secs_f64(),
                        attempt
                    );
                }
            }

            // Increase timeout for next attempt (half-exponential backoff)
            current_timeout = Duration::from_secs_f64(
                (current_timeout.as_secs_f64() * config.timeout_multiplier)
                    .min(config.max_timeout.as_secs_f64()),
            );
        }
    }

    /// Create a message with streaming response.
    ///
    /// Returns a `MessageStream` that yields events as they arrive.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::{Client, MessageCreateParams, MessageParam, StreamEvent};
    /// use tokio_stream::StreamExt;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let mut stream = client.messages().stream(MessageCreateParams {
    ///     model: "glm-4-plus".into(),
    ///     max_tokens: 1024,
    ///     messages: vec![MessageParam::user("Tell me a story")],
    ///     ..Default::default()
    /// }).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     if let llm_code_sdk::StreamEvent::Text { text, .. } = event {
    ///         print!("{}", text);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream(&self, mut params: MessageCreateParams) -> Result<MessageStream> {
        // Force streaming mode
        params.stream = Some(true);

        let response = self.client.post_stream("/v1/messages", &params).await?;

        let (tx, rx) = mpsc::channel(100);

        // Spawn task to process SSE events
        let bytes_stream = response.bytes_stream();
        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut stream = bytes_stream;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx
                            .send(RawStreamEvent::Error {
                                error: crate::streaming::StreamError {
                                    error_type: "stream_error".to_string(),
                                    message: e.to_string(),
                                },
                            })
                            .await;
                        break;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events
                while let Some(pos) = buffer.find("\n\n") {
                    let event_data = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(event) = parse_sse_event(&event_data) {
                        if tx.send(event).await.is_err() {
                            return; // Receiver dropped
                        }
                    }
                }
            }
        });

        Ok(MessageStream::new(rx))
    }

    /// Create a message with streaming and adaptive stall detection.
    ///
    /// This wraps `stream()` with automatic retry on stalls. If no tokens arrive
    /// within the stall timeout, the request is aborted and retried with an
    /// increased timeout (half-exponential backoff).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::{Client, MessageCreateParams, MessageParam};
    /// use llm_code_sdk::client::AdaptiveStreamConfig;
    /// use tokio_stream::StreamExt;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let mut stream = client.messages().stream_adaptive(
    ///     MessageCreateParams {
    ///         model: "glm-4-plus".into(),
    ///         max_tokens: 1024,
    ///         messages: vec![MessageParam::user("Tell me a story")],
    ///         ..Default::default()
    ///     },
    ///     AdaptiveStreamConfig::default(),
    /// ).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     // Process events...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream_adaptive(
        &self,
        params: MessageCreateParams,
        config: AdaptiveStreamConfig,
    ) -> Result<MessageStream> {
        let mut current_timeout = config.initial_stall_timeout;
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            tracing::debug!(
                "Adaptive stream attempt {} with {}s stall timeout",
                attempt,
                current_timeout.as_secs()
            );

            match self
                .stream_with_stall_detection(params.clone(), current_timeout)
                .await
            {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    // Check if it was a stall timeout
                    let is_stall = matches!(&e, ClientError::Stream(msg) if msg.contains("stall"));

                    if !is_stall || attempt >= config.max_retries {
                        tracing::warn!("Adaptive stream failed after {} attempts: {}", attempt, e);
                        return Err(e);
                    }

                    // Increase timeout for next attempt (half-exponential)
                    current_timeout = Duration::from_secs_f64(
                        (current_timeout.as_secs_f64() * config.timeout_multiplier)
                            .min(config.max_stall_timeout.as_secs_f64()),
                    );
                    tracing::info!(
                        "Stream stalled, retrying with {}s timeout",
                        current_timeout.as_secs()
                    );
                }
            }
        }
    }

    /// Stream with stall detection - aborts if no events for too long.
    async fn stream_with_stall_detection(
        &self,
        mut params: MessageCreateParams,
        stall_timeout: Duration,
    ) -> Result<MessageStream> {
        params.stream = Some(true);

        let response = self.client.post_stream("/v1/messages", &params).await?;

        let (tx, rx) = mpsc::channel(100);

        // Shared state for watchdog
        let last_event_time = Arc::new(AtomicU64::new(Instant::now().elapsed().as_millis() as u64));
        let start = Instant::now();
        let aborted = Arc::new(AtomicBool::new(false));

        let last_event_clone = last_event_time.clone();
        let aborted_clone = aborted.clone();
        let tx_clone = tx.clone();

        // Spawn watchdog task
        let watchdog_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;

                if aborted_clone.load(Ordering::Relaxed) {
                    break;
                }

                let last = last_event_clone.load(Ordering::Relaxed);
                let now = start.elapsed().as_millis() as u64;
                let elapsed_since_event = Duration::from_millis(now.saturating_sub(last));

                if elapsed_since_event > stall_timeout {
                    tracing::warn!(
                        "Stream stall detected: {}s since last event",
                        elapsed_since_event.as_secs()
                    );
                    aborted_clone.store(true, Ordering::Relaxed);
                    let _ = tx_clone
                        .send(RawStreamEvent::Error {
                            error: crate::streaming::StreamError {
                                error_type: "stall_timeout".to_string(),
                                message: format!(
                                    "No events for {}s (stall detected)",
                                    elapsed_since_event.as_secs()
                                ),
                            },
                        })
                        .await;
                    break;
                }
            }
        });

        // Spawn task to process SSE events
        let bytes_stream = response.bytes_stream();
        let last_event_for_stream = last_event_time.clone();
        let aborted_for_stream = aborted.clone();
        let start_for_stream = start;

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut stream = bytes_stream;

            while let Some(chunk_result) = stream.next().await {
                // Check if aborted
                if aborted_for_stream.load(Ordering::Relaxed) {
                    break;
                }

                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx
                            .send(RawStreamEvent::Error {
                                error: crate::streaming::StreamError {
                                    error_type: "stream_error".to_string(),
                                    message: e.to_string(),
                                },
                            })
                            .await;
                        break;
                    }
                };

                // Update last event time
                last_event_for_stream.store(
                    start_for_stream.elapsed().as_millis() as u64,
                    Ordering::Relaxed,
                );

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events
                while let Some(pos) = buffer.find("\n\n") {
                    let event_data = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(event) = parse_sse_event(&event_data) {
                        // Update time on each parsed event too
                        last_event_for_stream.store(
                            start_for_stream.elapsed().as_millis() as u64,
                            Ordering::Relaxed,
                        );

                        if tx.send(event).await.is_err() {
                            aborted_for_stream.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            }

            // Signal completion
            aborted_for_stream.store(true, Ordering::Relaxed);
            watchdog_handle.abort();
        });

        // Check if we were aborted before returning
        if aborted.load(Ordering::Relaxed) {
            return Err(ClientError::Stream("Stream stall detected".to_string()));
        }

        Ok(MessageStream::new(rx))
    }

    /// Count the number of tokens in a message.
    ///
    /// This can be used to count tokens before sending a request,
    /// including tools, images, and documents.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::{Client, CountTokensParams, MessageParam};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let count = client.messages().count_tokens(CountTokensParams {
    ///     model: "glm-4-plus".into(),
    ///     messages: vec![MessageParam::user("Hello, world!")],
    ///     ..Default::default()
    /// }).await?;
    ///
    /// println!("Input tokens: {}", count.input_tokens);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn count_tokens(
        &self,
        params: crate::types::CountTokensParams,
    ) -> Result<crate::types::MessageTokensCount> {
        self.client.post("/v1/messages/count_tokens", &params).await
    }
}

/// Parse a single SSE event from raw text.
fn parse_sse_event(data: &str) -> Option<RawStreamEvent> {
    let mut event_type = None;
    let mut event_data = None;

    for line in data.lines() {
        if line.starts_with("event: ") {
            event_type = Some(line[7..].to_string());
        } else if line.starts_with("data: ") {
            event_data = Some(line[6..].to_string());
        }
    }

    let data = event_data?;
    let event = event_type.as_deref().unwrap_or("message");

    // Handle different event types
    match event {
        "message_start" => serde_json::from_str(&data).ok(),
        "content_block_start" => serde_json::from_str(&data).ok(),
        "content_block_delta" => serde_json::from_str(&data).ok(),
        "content_block_stop" => serde_json::from_str(&data).ok(),
        "message_delta" => serde_json::from_str(&data).ok(),
        "message_stop" => Some(RawStreamEvent::MessageStop),
        "ping" => Some(RawStreamEvent::Ping),
        "error" => serde_json::from_str(&data).ok(),
        _ => {
            // Try to parse as generic event
            serde_json::from_str(&data).ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageParam;

    #[test]
    fn test_message_create_params_default() {
        let params = MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 1024,
            messages: vec![MessageParam::user("Hello")],
            ..Default::default()
        };

        assert_eq!(params.model, "glm-4-plus");
        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.messages.len(), 1);
        assert!(params.system.is_none());
        assert!(params.tools.is_empty());
    }

    #[test]
    fn test_message_param_serialization() {
        let params = MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 1024,
            messages: vec![
                MessageParam::user("What is 2+2?"),
                MessageParam::assistant("2+2 equals 4."),
                MessageParam::user("Thanks!"),
            ],
            system: Some("You are a helpful assistant.".into()),
            ..Default::default()
        };

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"role\":\"assistant\""));
        assert!(json.contains("\"system\":\"You are a helpful assistant.\""));
    }
}
