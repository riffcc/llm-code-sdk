//! Messages API client.

use futures::StreamExt;
use tokio::sync::mpsc;

use super::{ApiFormat, Client, Result};
use crate::streaming::{MessageStream, RawStreamEvent};
use crate::types::openai::{OpenAIChatRequest, OpenAIChatResponse};
use crate::types::{Message, MessageCreateParams};

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
            ApiFormat::Anthropic => {
                self.client.post("/v1/messages", &params).await
            }
            ApiFormat::OpenAI => {
                let openai_request = OpenAIChatRequest::from(&params);
                tracing::debug!(target: "llm_code_sdk", "OpenAI request: {:?}", openai_request);
                let response: OpenAIChatResponse = self.client
                    .post("/chat/completions", &openai_request)
                    .await?;
                Ok(Message::from(response))
            }
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
