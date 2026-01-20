//! Error types for the SDK.
//!
//! This module provides a comprehensive error hierarchy matching the Python SDK.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main error type for the SDK.
#[derive(Debug, Error)]
pub enum Error {
    /// API returned an error response.
    #[error("API error: {0}")]
    Api(#[from] ApiError),

    /// HTTP request failed.
    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Connection to the API failed.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Request timed out.
    #[error("Request timed out")]
    Timeout,

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Result type alias for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// API error response.
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
#[error("{error_type}: {message}")]
pub struct ApiError {
    /// The type of error.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Human-readable error message.
    pub message: String,

    /// HTTP status code.
    #[serde(skip)]
    pub status: u16,
}

impl ApiError {
    /// Create a new API error.
    pub fn new(error_type: impl Into<String>, message: impl Into<String>, status: u16) -> Self {
        Self {
            error_type: error_type.into(),
            message: message.into(),
            status,
        }
    }

    /// Returns true if this is an authentication error.
    pub fn is_auth_error(&self) -> bool {
        self.status == 401 || self.error_type == "authentication_error"
    }

    /// Returns true if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        self.status == 429 || self.error_type == "rate_limit_error"
    }

    /// Returns true if this is a bad request error.
    pub fn is_bad_request(&self) -> bool {
        self.status == 400 || self.error_type == "invalid_request_error"
    }

    /// Returns true if this is a not found error.
    pub fn is_not_found(&self) -> bool {
        self.status == 404 || self.error_type == "not_found_error"
    }

    /// Returns true if this is an overloaded error.
    pub fn is_overloaded(&self) -> bool {
        self.status == 529 || self.error_type == "overloaded_error"
    }

    /// Returns true if this is a server error (5xx).
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }

    /// Returns true if this error should be retried.
    pub fn is_retryable(&self) -> bool {
        self.is_rate_limit() || self.is_overloaded() || self.is_server_error()
    }
}

/// Specific error types matching the API.
#[derive(Debug, Error)]
pub enum ApiErrorType {
    /// Authentication failed (401).
    #[error("Authentication failed: {message}")]
    AuthenticationError { message: String },

    /// Bad request (400).
    #[error("Bad request: {message}")]
    BadRequestError { message: String },

    /// Permission denied (403).
    #[error("Permission denied: {message}")]
    PermissionDeniedError { message: String },

    /// Resource not found (404).
    #[error("Not found: {message}")]
    NotFoundError { message: String },

    /// Conflict (409).
    #[error("Conflict: {message}")]
    ConflictError { message: String },

    /// Unprocessable entity (422).
    #[error("Unprocessable entity: {message}")]
    UnprocessableEntityError { message: String },

    /// Rate limited (429).
    #[error("Rate limited: {message}")]
    RateLimitError { message: String, retry_after: Option<u64> },

    /// Internal server error (500).
    #[error("Internal server error: {message}")]
    InternalServerError { message: String },

    /// API overloaded (529).
    #[error("API overloaded: {message}")]
    OverloadedError { message: String },

    /// Gateway timeout (502/504).
    #[error("Gateway timeout: {message}")]
    GatewayTimeoutError { message: String },

    /// Billing error.
    #[error("Billing error: {message}")]
    BillingError { message: String },
}

impl From<ApiError> for ApiErrorType {
    fn from(err: ApiError) -> Self {
        match err.status {
            401 => ApiErrorType::AuthenticationError { message: err.message },
            400 => ApiErrorType::BadRequestError { message: err.message },
            403 => ApiErrorType::PermissionDeniedError { message: err.message },
            404 => ApiErrorType::NotFoundError { message: err.message },
            409 => ApiErrorType::ConflictError { message: err.message },
            422 => ApiErrorType::UnprocessableEntityError { message: err.message },
            429 => ApiErrorType::RateLimitError {
                message: err.message,
                retry_after: None,
            },
            500 => ApiErrorType::InternalServerError { message: err.message },
            502 | 504 => ApiErrorType::GatewayTimeoutError { message: err.message },
            529 => ApiErrorType::OverloadedError { message: err.message },
            _ => ApiErrorType::InternalServerError { message: err.message },
        }
    }
}

/// Error response from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// The error object.
    pub error: ApiError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_classification() {
        let auth_error = ApiError::new("authentication_error", "Invalid API key", 401);
        assert!(auth_error.is_auth_error());
        assert!(!auth_error.is_retryable());

        let rate_limit = ApiError::new("rate_limit_error", "Too many requests", 429);
        assert!(rate_limit.is_rate_limit());
        assert!(rate_limit.is_retryable());

        let server_error = ApiError::new("internal_error", "Something went wrong", 500);
        assert!(server_error.is_server_error());
        assert!(server_error.is_retryable());

        let bad_request = ApiError::new("invalid_request_error", "Bad input", 400);
        assert!(bad_request.is_bad_request());
        assert!(!bad_request.is_retryable());
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"error": {"type": "invalid_request_error", "message": "Invalid model"}}"#;
        let response: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(response.error.message, "Invalid model");
    }

    #[test]
    fn test_api_error_type_conversion() {
        let api_error = ApiError::new("rate_limit_error", "Too many requests", 429);
        let typed: ApiErrorType = api_error.into();
        assert!(matches!(typed, ApiErrorType::RateLimitError { .. }));
    }
}
