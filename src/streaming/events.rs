//! Streaming event types.

use serde::{Deserialize, Serialize};

use crate::types::{ContentBlock, Message, StopReason, Usage};

/// Raw streaming events from the SSE API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawStreamEvent {
    /// Message has started.
    MessageStart { message: MessageStart },

    /// A content block has started.
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStart,
    },

    /// Content block delta (partial content).
    ContentBlockDelta { index: usize, delta: ContentDelta },

    /// A content block has completed.
    ContentBlockStop { index: usize },

    /// Message delta (usage, stop_reason).
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<MessageDeltaUsage>,
    },

    /// Message has completed.
    MessageStop,

    /// Ping event (keep-alive).
    Ping,

    /// Error event.
    Error { error: StreamError },
}

/// Message start data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Content block start data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Thinking {
        thinking: String,
    },
}

/// Content delta data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text delta.
    TextDelta { text: String },

    /// Tool use input JSON delta.
    InputJsonDelta { partial_json: String },

    /// Thinking delta.
    ThinkingDelta { thinking: String },

    /// Signature delta.
    SignatureDelta { signature: String },
}

impl ContentDelta {
    /// Get text content if this is a text delta.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentDelta::TextDelta { text } => Some(text),
            _ => None,
        }
    }
}

/// Message delta data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
}

/// Usage information in message delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaUsage {
    pub output_tokens: u64,
}

/// Stream error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// High-level stream events for easier consumption.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text content was received.
    Text { text: String, snapshot: String },

    /// Thinking content was received.
    Thinking { thinking: String, snapshot: String },

    /// Tool use input JSON was received.
    InputJson {
        partial_json: String,
        snapshot: serde_json::Value,
    },

    /// A content block completed.
    ContentBlockStop {
        index: usize,
        content_block: ContentBlock,
    },

    /// The message completed.
    MessageStop { message: Message },

    /// An error occurred.
    Error { error: StreamError },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_event_deserialization_message_start() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "glm-4-plus",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }"#;

        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        if let RawStreamEvent::MessageStart { message } = event {
            assert_eq!(message.id, "msg_123");
        } else {
            panic!("Expected MessageStart");
        }
    }

    #[test]
    fn test_raw_event_deserialization_text_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }"#;

        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        if let RawStreamEvent::ContentBlockDelta { index, delta } = event {
            assert_eq!(index, 0);
            assert_eq!(delta.as_text(), Some("Hello"));
        } else {
            panic!("Expected ContentBlockDelta");
        }
    }

    #[test]
    fn test_raw_event_deserialization_message_stop() {
        let json = r#"{"type": "message_stop"}"#;
        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RawStreamEvent::MessageStop));
    }

    #[test]
    fn test_raw_event_deserialization_ping() {
        let json = r#"{"type": "ping"}"#;
        let event: RawStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RawStreamEvent::Ping));
    }
}
