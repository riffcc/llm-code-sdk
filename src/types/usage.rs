//! Usage information types.

use serde::{Deserialize, Serialize};

/// Token count result from the count_tokens API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MessageTokensCount {
    /// The total number of tokens across the provided list of messages,
    /// system prompt, and tools.
    pub input_tokens: u64,
}

/// Token usage information from the API response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Usage {
    /// Number of input tokens used.
    pub input_tokens: u64,

    /// Number of output tokens generated.
    pub output_tokens: u64,

    /// Number of input tokens used to create the cache entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,

    /// Number of input tokens read from the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

impl Usage {
    /// Returns the total number of input tokens (including cache operations).
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens
            + self.cache_creation_input_tokens.unwrap_or(0)
            + self.cache_read_input_tokens.unwrap_or(0)
    }

    /// Returns the total number of tokens used.
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens() + self.output_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_deserialization() {
        let json = r#"{"input_tokens":100,"output_tokens":50}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn test_usage_with_cache() {
        let json = r#"{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":20,"cache_read_input_tokens":30}"#;
        let usage: Usage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.total_input_tokens(), 150);
        assert_eq!(usage.total_tokens(), 200);
    }

    #[test]
    fn test_usage_serialization_skips_none() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(!json.contains("cache_creation_input_tokens"));
        assert!(!json.contains("cache_read_input_tokens"));
    }
}
