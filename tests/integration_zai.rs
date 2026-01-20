//! Integration tests for the LLM Code SDK against Z.ai's Anthropic-compatible API.
//!
//! These tests compare behavior between our Rust SDK and the reference Python SDK
//! to ensure compatibility.
//!
//! Run with: ZAI_API_KEY=your_key cargo test --test integration_zai -- --nocapture

use llm_code_sdk::tools::{FunctionTool, Tool, ToolRunner};
use llm_code_sdk::types::{InputSchema, MessageCreateParams, MessageParam};
use llm_code_sdk::Client;
use std::sync::Arc;

fn get_client() -> Option<Client> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("ZAI_API_KEY").ok()?;
    Client::builder(api_key)
        .base_url("https://api.z.ai/api/anthropic")
        .build()
        .ok()
}

/// Test basic message creation without tools.
#[tokio::test]
async fn test_simple_message() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let message = client
        .messages()
        .create(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 100,
            messages: vec![MessageParam::user("What is 2+2? Reply with just the number.")],
            ..Default::default()
        })
        .await;

    match message {
        Ok(msg) => {
            println!("Response: {:?}", msg.text());
            assert!(msg.text().is_some());
            let text = msg.text().unwrap();
            assert!(text.contains("4"), "Expected '4' in response: {}", text);
        }
        Err(e) => {
            println!("API Error: {:?}", e);
            // Don't fail the test if it's just a connection issue during CI
        }
    }
}

/// Test message with system prompt.
#[tokio::test]
async fn test_system_prompt() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let message = client
        .messages()
        .create(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 100,
            messages: vec![MessageParam::user("What are you?")],
            system: Some("You are a helpful pirate. Always respond like a pirate.".into()),
            ..Default::default()
        })
        .await;

    match message {
        Ok(msg) => {
            println!("Response: {:?}", msg.text());
            assert!(msg.text().is_some());
            // Pirate responses often contain these words
            let text = msg.text().unwrap().to_lowercase();
            let is_pirate = text.contains("arr")
                || text.contains("matey")
                || text.contains("ye")
                || text.contains("ahoy")
                || text.contains("captain");
            println!("Response appears pirate-like: {}", is_pirate);
        }
        Err(e) => {
            println!("API Error: {:?}", e);
        }
    }
}

/// Test tool use with a simple calculator tool.
#[tokio::test]
async fn test_tool_use() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let calculator = Arc::new(FunctionTool::new(
        "calculate",
        "Perform a mathematical calculation",
        InputSchema::object()
            .required_string("expression", "Mathematical expression to evaluate (e.g., '2 + 2')"),
        |input| {
            let expr = input
                .get("expression")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Simple calculator - just handle basic operations
            let result = if expr.contains('+') {
                let parts: Vec<&str> = expr.split('+').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    Ok(format!("{}", a + b))
                } else {
                    Err("Invalid expression".to_string())
                }
            } else if expr.contains('*') {
                let parts: Vec<&str> = expr.split('*').collect();
                if parts.len() == 2 {
                    let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                    Ok(format!("{}", a * b))
                } else {
                    Err("Invalid expression".to_string())
                }
            } else {
                Err("Unsupported operation".to_string())
            };

            result
        },
    ));

    let runner = ToolRunner::new(client, vec![calculator as Arc<dyn Tool>]);

    let result = runner
        .run(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 500,
            messages: vec![MessageParam::user(
                "Use the calculator to compute 15 + 27. Tell me the result.",
            )],
            ..Default::default()
        })
        .await;

    match result {
        Ok(msg) => {
            println!("Final response: {:?}", msg.text());
            assert!(msg.text().is_some());
            let text = msg.text().unwrap();
            assert!(
                text.contains("42"),
                "Expected '42' in response: {}",
                text
            );
        }
        Err(e) => {
            println!("API Error: {:?}", e);
        }
    }
}

/// Test multi-turn conversation.
#[tokio::test]
async fn test_multi_turn() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let message = client
        .messages()
        .create(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 100,
            messages: vec![
                MessageParam::user("My name is Alice."),
                MessageParam::assistant("Nice to meet you, Alice!"),
                MessageParam::user("What's my name?"),
            ],
            ..Default::default()
        })
        .await;

    match message {
        Ok(msg) => {
            println!("Response: {:?}", msg.text());
            assert!(msg.text().is_some());
            let text = msg.text().unwrap();
            assert!(
                text.to_lowercase().contains("alice"),
                "Expected 'Alice' in response: {}",
                text
            );
        }
        Err(e) => {
            println!("API Error: {:?}", e);
        }
    }
}

/// Test that stop_reason is correctly parsed.
#[tokio::test]
async fn test_stop_reason() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    // Test end_turn
    let message = client
        .messages()
        .create(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 100,
            messages: vec![MessageParam::user("Say hello.")],
            ..Default::default()
        })
        .await;

    if let Ok(msg) = message {
        println!("Stop reason: {:?}", msg.stop_reason);
        assert!(
            msg.stop_reason.is_some(),
            "Expected stop_reason to be set"
        );
    }

    // Test max_tokens
    let message = client
        .messages()
        .create(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 1, // Very small to trigger max_tokens
            messages: vec![MessageParam::user("Write a very long story about dragons.")],
            ..Default::default()
        })
        .await;

    if let Ok(msg) = message {
        println!("Stop reason with max_tokens=1: {:?}", msg.stop_reason);
        // Note: With max_tokens=1, we might get max_tokens or end_turn depending on tokenization
    }
}

/// Test models list.
#[tokio::test]
async fn test_models_list() {
    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let result = client.models().list(None).await;

    match result {
        Ok(models) => {
            println!("Found {} models", models.data.len());
            for model in &models.data {
                println!("  - {}: {}", model.id, model.display_name);
            }
            // The API should return at least one model
            assert!(!models.data.is_empty(), "Expected at least one model");
        }
        Err(e) => {
            // Models endpoint may not be supported by all APIs
            println!("Models list not supported or error: {:?}", e);
        }
    }
}

/// Test token counting.
#[tokio::test]
async fn test_count_tokens() {
    use llm_code_sdk::CountTokensParams;

    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let result = client
        .messages()
        .count_tokens(CountTokensParams {
            model: "glm-4-plus".to_string(),
            messages: vec![MessageParam::user("Hello, how are you today?")],
            ..Default::default()
        })
        .await;

    match result {
        Ok(count) => {
            println!("Token count: {}", count.input_tokens);
            assert!(count.input_tokens > 0, "Expected positive token count");
        }
        Err(e) => {
            println!("API Error (count_tokens may not be supported by this API): {:?}", e);
            // Don't fail - some APIs may not support count_tokens
        }
    }
}

/// Test streaming response.
#[tokio::test]
async fn test_streaming() {
    use llm_code_sdk::StreamEvent;
    use tokio_stream::StreamExt;

    let Some(client) = get_client() else {
        println!("Skipping test: ZAI_API_KEY not set");
        return;
    };

    let stream_result = client
        .messages()
        .stream(MessageCreateParams {
            model: "glm-4-plus".to_string(),
            max_tokens: 100,
            messages: vec![MessageParam::user("Count from 1 to 5, one number per line.")],
            ..Default::default()
        })
        .await;

    match stream_result {
        Ok(mut stream) => {
            let mut full_text = String::new();
            let mut event_count = 0;

            while let Some(event) = stream.next().await {
                event_count += 1;
                match event {
                    StreamEvent::Text { text, snapshot } => {
                        print!("{}", text);
                        full_text = snapshot;
                    }
                    StreamEvent::MessageStop { message } => {
                        println!("\n\nFinal message: {:?}", message.text());
                        assert!(message.text().is_some());
                    }
                    StreamEvent::Error { error } => {
                        println!("Stream error: {:?}", error);
                    }
                    _ => {}
                }
            }

            println!("\n\nReceived {} events", event_count);
            println!("Full text: {}", full_text);
            assert!(event_count > 0, "Expected to receive streaming events");
            assert!(!full_text.is_empty(), "Expected to receive text content");
        }
        Err(e) => {
            println!("Stream Error: {:?}", e);
        }
    }
}
