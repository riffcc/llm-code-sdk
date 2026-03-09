//! Message stream implementation.

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::events::{ContentBlockStart, ContentDelta, RawStreamEvent, StreamEvent};
use crate::types::{ContentBlock, Message, TextBlock, ThinkingBlock, ToolUseBlock};

/// A stream of message events.
///
/// This wraps the raw SSE stream and provides both raw and high-level event access.
pub struct MessageStream {
    inner: ReceiverStream<RawStreamEvent>,
    message_snapshot: Option<Message>,
    text_snapshots: Vec<String>,
    thinking_snapshots: Vec<String>,
    input_json_snapshots: Vec<String>,
}

impl MessageStream {
    /// Create a new message stream from a receiver.
    pub fn new(receiver: mpsc::Receiver<RawStreamEvent>) -> Self {
        Self {
            inner: ReceiverStream::new(receiver),
            message_snapshot: None,
            text_snapshots: Vec::new(),
            thinking_snapshots: Vec::new(),
            input_json_snapshots: Vec::new(),
        }
    }

    /// Get the current message snapshot.
    ///
    /// This is updated as events are received.
    pub fn current_message(&self) -> Option<&Message> {
        self.message_snapshot.as_ref()
    }

    /// Consume the stream and return the final message.
    pub async fn get_final_message(mut self) -> Option<Message> {
        use tokio_stream::StreamExt;

        while let Some(_) = self.next().await {}
        self.message_snapshot
    }

    /// Get all accumulated text so far.
    pub fn get_current_text(&self) -> String {
        self.text_snapshots.join("")
    }

    fn accumulate_event(&mut self, event: &RawStreamEvent) {
        match event {
            RawStreamEvent::MessageStart { message } => {
                self.message_snapshot = Some(Message {
                    id: message.id.clone(),
                    message_type: message.message_type.clone(),
                    role: message.role.clone(),
                    content: message.content.clone(),
                    model: message.model.clone(),
                    stop_reason: message.stop_reason.clone(),
                    stop_sequence: message.stop_sequence.clone(),
                    usage: message.usage.clone(),
                });
                // Initialize snapshot vectors based on content count
                self.text_snapshots.clear();
                self.thinking_snapshots.clear();
                self.input_json_snapshots.clear();
            }

            RawStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if let Some(ref mut msg) = self.message_snapshot {
                    // Ensure we have enough content blocks
                    while msg.content.len() <= *index {
                        msg.content.push(ContentBlock::Text(TextBlock::new("")));
                    }

                    // Ensure we have enough snapshot vectors
                    while self.text_snapshots.len() <= *index {
                        self.text_snapshots.push(String::new());
                    }
                    while self.thinking_snapshots.len() <= *index {
                        self.thinking_snapshots.push(String::new());
                    }
                    while self.input_json_snapshots.len() <= *index {
                        self.input_json_snapshots.push(String::new());
                    }

                    msg.content[*index] = match content_block {
                        ContentBlockStart::Text { text } => {
                            self.text_snapshots[*index] = text.clone();
                            ContentBlock::Text(TextBlock::new(text.clone()))
                        }
                        ContentBlockStart::ToolUse { id, name, input } => {
                            let input_map = if let Some(obj) = input.as_object() {
                                obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                            } else {
                                std::collections::HashMap::new()
                            };
                            ContentBlock::ToolUse(ToolUseBlock {
                                id: id.clone(),
                                name: name.clone(),
                                input: input_map,
                            })
                        }
                        ContentBlockStart::Thinking { thinking } => {
                            self.thinking_snapshots[*index] = thinking.clone();
                            ContentBlock::Thinking(ThinkingBlock {
                                thinking: thinking.clone(),
                                signature: None,
                            })
                        }
                    };
                }
            }

            RawStreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(ref mut msg) = self.message_snapshot {
                    if *index < msg.content.len() {
                        match delta {
                            ContentDelta::TextDelta { text } => {
                                if *index < self.text_snapshots.len() {
                                    self.text_snapshots[*index].push_str(text);
                                }
                                if let ContentBlock::Text(ref mut block) = msg.content[*index] {
                                    block.text.push_str(text);
                                }
                            }
                            ContentDelta::ThinkingDelta { thinking } => {
                                if *index < self.thinking_snapshots.len() {
                                    self.thinking_snapshots[*index].push_str(thinking);
                                }
                                if let ContentBlock::Thinking(ref mut block) = msg.content[*index] {
                                    block.thinking.push_str(thinking);
                                }
                            }
                            ContentDelta::InputJsonDelta { partial_json } => {
                                if *index < self.input_json_snapshots.len() {
                                    self.input_json_snapshots[*index].push_str(partial_json);
                                }
                                // Tool use input is accumulated but parsed at the end
                            }
                            ContentDelta::SignatureDelta { signature } => {
                                if let ContentBlock::Thinking(ref mut block) = msg.content[*index] {
                                    block.signature = Some(signature.clone());
                                }
                            }
                        }
                    }
                }
            }

            RawStreamEvent::ContentBlockStop { index } => {
                // Parse accumulated JSON for tool use blocks
                if let Some(ref mut msg) = self.message_snapshot {
                    if *index < msg.content.len() && *index < self.input_json_snapshots.len() {
                        if let ContentBlock::ToolUse(ref mut block) = msg.content[*index] {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(
                                &self.input_json_snapshots[*index],
                            ) {
                                if let Some(obj) = parsed.as_object() {
                                    block.input =
                                        obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                                }
                            }
                        }
                    }
                }
            }

            RawStreamEvent::MessageDelta { delta, usage } => {
                if let Some(ref mut msg) = self.message_snapshot {
                    if let Some(stop_reason) = &delta.stop_reason {
                        msg.stop_reason = Some(stop_reason.clone());
                    }
                    if let Some(stop_sequence) = &delta.stop_sequence {
                        msg.stop_sequence = Some(stop_sequence.clone());
                    }
                    if let Some(usage_delta) = usage {
                        msg.usage.output_tokens = usage_delta.output_tokens;
                    }
                }
            }

            RawStreamEvent::MessageStop => {
                // Message is complete
            }

            RawStreamEvent::Ping | RawStreamEvent::Error { .. } => {
                // No accumulation needed
            }
        }
    }

    fn build_high_level_event(&self, raw: &RawStreamEvent) -> Option<StreamEvent> {
        match raw {
            RawStreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => Some(StreamEvent::Text {
                    text: text.clone(),
                    snapshot: self.text_snapshots.get(*index).cloned().unwrap_or_default(),
                }),
                ContentDelta::ThinkingDelta { thinking } => Some(StreamEvent::Thinking {
                    thinking: thinking.clone(),
                    snapshot: self
                        .thinking_snapshots
                        .get(*index)
                        .cloned()
                        .unwrap_or_default(),
                }),
                ContentDelta::InputJsonDelta { partial_json } => {
                    let snapshot = self
                        .input_json_snapshots
                        .get(*index)
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or(serde_json::Value::Null);
                    Some(StreamEvent::InputJson {
                        partial_json: partial_json.clone(),
                        snapshot,
                    })
                }
                ContentDelta::SignatureDelta { .. } => None,
            },

            RawStreamEvent::ContentBlockStop { index } => {
                self.message_snapshot.as_ref().and_then(|msg| {
                    msg.content
                        .get(*index)
                        .map(|block| StreamEvent::ContentBlockStop {
                            index: *index,
                            content_block: block.clone(),
                        })
                })
            }

            RawStreamEvent::MessageStop => {
                self.message_snapshot
                    .as_ref()
                    .map(|msg| StreamEvent::MessageStop {
                        message: msg.clone(),
                    })
            }

            RawStreamEvent::Error { error } => Some(StreamEvent::Error {
                error: error.clone(),
            }),

            _ => None,
        }
    }
}

impl Stream for MessageStream {
    type Item = StreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        use tokio_stream::StreamExt;

        loop {
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(raw_event)) => {
                    self.accumulate_event(&raw_event);

                    if let Some(event) = self.build_high_level_event(&raw_event) {
                        return Poll::Ready(Some(event));
                    }
                    // Skip events that don't produce high-level events
                    continue;
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Iterator over just the text deltas in a stream.
pub struct TextStream {
    stream: MessageStream,
}

impl TextStream {
    pub fn new(stream: MessageStream) -> Self {
        Self { stream }
    }
}

impl Stream for TextStream {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(StreamEvent::Text { text, .. })) => {
                    return Poll::Ready(Some(text));
                }
                Poll::Ready(Some(_)) => continue, // Skip non-text events
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::events::MessageStart;
    use crate::types::Usage;

    #[tokio::test]
    async fn test_message_stream_accumulation() {
        use tokio_stream::StreamExt;

        let (tx, rx) = mpsc::channel(10);
        let mut stream = MessageStream::new(rx);

        // Send events
        tx.send(RawStreamEvent::MessageStart {
            message: MessageStart {
                id: "msg_123".to_string(),
                message_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: "glm-4-plus".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage::default(),
            },
        })
        .await
        .unwrap();

        tx.send(RawStreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlockStart::Text {
                text: "".to_string(),
            },
        })
        .await
        .unwrap();

        tx.send(RawStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "Hello".to_string(),
            },
        })
        .await
        .unwrap();

        tx.send(RawStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: " World".to_string(),
            },
        })
        .await
        .unwrap();

        tx.send(RawStreamEvent::ContentBlockStop { index: 0 })
            .await
            .unwrap();

        tx.send(RawStreamEvent::MessageStop).await.unwrap();

        drop(tx);

        // Consume stream
        let mut texts = Vec::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::Text { text, .. } = event {
                texts.push(text);
            }
        }

        assert_eq!(texts, vec!["Hello", " World"]);
        assert_eq!(stream.get_current_text(), "Hello World");
    }
}
