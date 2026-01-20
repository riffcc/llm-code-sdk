//! Models API client.

use super::{Client, Result};
use serde::{Deserialize, Serialize};

/// Client for the models endpoint.
#[derive(Debug)]
pub struct ModelsClient<'a> {
    client: &'a Client,
}

impl<'a> ModelsClient<'a> {
    pub(crate) fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// List available models.
    ///
    /// Returns a paginated list of available models, with most recently
    /// released models listed first.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::Client;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let models = client.models().list(None).await?;
    /// for model in &models.data {
    ///     println!("{}: {}", model.id, model.display_name);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list(&self, params: Option<ModelListParams>) -> Result<ModelListResponse> {
        let query = params.map(|p| {
            let mut q = Vec::new();
            if let Some(limit) = p.limit {
                q.push(format!("limit={}", limit));
            }
            if let Some(after_id) = p.after_id {
                q.push(format!("after_id={}", after_id));
            }
            if let Some(before_id) = p.before_id {
                q.push(format!("before_id={}", before_id));
            }
            q.join("&")
        });

        let path = match query {
            Some(q) if !q.is_empty() => format!("/v1/models?{}", q),
            _ => "/v1/models".to_string(),
        };

        self.client.get(&path).await
    }

    /// Get a specific model by ID.
    ///
    /// This can be used to get information about a specific model or
    /// resolve a model alias to a model ID.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use llm_code_sdk::Client;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let client = Client::new("your-api-key")?;
    ///
    /// let model = client.models().retrieve("claude-sonnet-4-20250514").await?;
    /// println!("Model: {} ({})", model.display_name, model.id);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn retrieve(&self, model_id: &str) -> Result<ModelInfo> {
        let path = format!("/v1/models/{}", model_id);
        self.client.get(&path).await
    }
}

/// Parameters for listing models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelListParams {
    /// Number of items to return per page (1-1000, default 20).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// ID of the object to use as a cursor for pagination.
    /// Returns the page of results immediately after this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_id: Option<String>,

    /// ID of the object to use as a cursor for pagination.
    /// Returns the page of results immediately before this object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_id: Option<String>,
}

/// Response from the list models endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// List of model information.
    pub data: Vec<ModelInfo>,

    /// Whether there are more results.
    #[serde(default)]
    pub has_more: bool,

    /// ID of the first object in the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// ID of the last object in the list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

/// Information about a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier.
    pub id: String,

    /// Object type (always "model").
    #[serde(rename = "type")]
    pub object_type: String,

    /// A human-readable name for the model.
    pub display_name: String,

    /// RFC 3339 datetime string representing when the model was released.
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_deserialization() {
        let json = r#"{
            "id": "claude-sonnet-4-20250514",
            "type": "model",
            "display_name": "Claude Sonnet 4",
            "created_at": "2025-05-14T00:00:00Z"
        }"#;

        let model: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(model.id, "claude-sonnet-4-20250514");
        assert_eq!(model.display_name, "Claude Sonnet 4");
        assert_eq!(model.object_type, "model");
    }

    #[test]
    fn test_model_list_response_deserialization() {
        let json = r#"{
            "data": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "type": "model",
                    "display_name": "Claude Sonnet 4",
                    "created_at": "2025-05-14T00:00:00Z"
                }
            ],
            "has_more": false,
            "first_id": "claude-sonnet-4-20250514",
            "last_id": "claude-sonnet-4-20250514"
        }"#;

        let response: ModelListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 1);
        assert!(!response.has_more);
    }
}
