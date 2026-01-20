//! Tool definition types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// A tool definition parameter for the API request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParam {
    /// The name of the tool.
    pub name: String,

    /// Description of what the tool does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the tool's input parameters.
    pub input_schema: InputSchema,

    /// Cache control settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<super::CacheControl>,
}

impl ToolParam {
    /// Create a new tool parameter with the given name and schema.
    pub fn new(name: impl Into<String>, input_schema: InputSchema) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
            cache_control: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set cache control to ephemeral.
    pub fn with_cache_control(mut self) -> Self {
        self.cache_control = Some(super::CacheControl::ephemeral());
        self
    }
}

/// JSON Schema for tool input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputSchema {
    /// Schema type (usually "object").
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Property definitions.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, PropertySchema>,

    /// Required property names.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,

    /// Additional properties allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_properties: Option<bool>,
}

impl InputSchema {
    /// Create a new object schema.
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            additional_properties: None,
        }
    }

    /// Add a property to the schema.
    pub fn property(
        mut self,
        name: impl Into<String>,
        schema: PropertySchema,
        required: bool,
    ) -> Self {
        let name = name.into();
        if required {
            self.required.push(name.clone());
        }
        self.properties.insert(name, schema);
        self
    }

    /// Add a required string property.
    pub fn required_string(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.property(
            name,
            PropertySchema::string().with_description(description),
            true,
        )
    }

    /// Add an optional string property.
    pub fn optional_string(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.property(
            name,
            PropertySchema::string().with_description(description),
            false,
        )
    }
}

impl Default for InputSchema {
    fn default() -> Self {
        Self::object()
    }
}

/// JSON Schema for a property.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertySchema {
    /// Property type.
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Description of the property.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Enum values (for string enums).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(rename = "enum")]
    pub enum_values: Option<Vec<String>>,

    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,

    /// Items schema (for arrays).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<PropertySchema>>,

    /// Properties (for nested objects).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, PropertySchema>>,

    /// Required properties (for nested objects).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

impl PropertySchema {
    /// Create a string property schema.
    pub fn string() -> Self {
        Self {
            schema_type: "string".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an integer property schema.
    pub fn integer() -> Self {
        Self {
            schema_type: "integer".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a number property schema.
    pub fn number() -> Self {
        Self {
            schema_type: "number".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create a boolean property schema.
    pub fn boolean() -> Self {
        Self {
            schema_type: "boolean".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: None,
            properties: None,
            required: None,
        }
    }

    /// Create an array property schema.
    pub fn array(items: PropertySchema) -> Self {
        Self {
            schema_type: "array".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: Some(Box::new(items)),
            properties: None,
            required: None,
        }
    }

    /// Create an object property schema.
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            description: None,
            enum_values: None,
            default: None,
            items: None,
            properties: Some(HashMap::new()),
            required: Some(Vec::new()),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set enum values (for string type).
    pub fn with_enum(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Add a property to an object schema.
    pub fn property(
        mut self,
        name: impl Into<String>,
        schema: PropertySchema,
        required: bool,
    ) -> Self {
        let name = name.into();
        let props = self.properties.get_or_insert_with(HashMap::new);
        props.insert(name.clone(), schema);
        if required {
            let req = self.required.get_or_insert_with(Vec::new);
            req.push(name);
        }
        self
    }
}

/// Server-side tool types provided by Anthropic.
///
/// These tools are executed server-side and don't require client implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ServerTool {
    /// Bash command execution tool.
    #[serde(rename = "bash_20250124")]
    Bash {
        /// Tool name (always "bash").
        name: BashToolName,
        /// Cache control settings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<super::CacheControl>,
    },

    /// Text editor tool for viewing and editing files.
    #[serde(rename = "text_editor_20250124")]
    TextEditor20250124 {
        /// Tool name (always "str_replace_editor").
        name: TextEditorToolName,
        /// Cache control settings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<super::CacheControl>,
    },

    /// Text editor tool (newer version 20250429).
    #[serde(rename = "text_editor_20250429")]
    TextEditor20250429 {
        /// Tool name (always "str_replace_editor").
        name: TextEditorToolName,
        /// Cache control settings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<super::CacheControl>,
    },

    /// Text editor tool (newest version 20250728).
    #[serde(rename = "text_editor_20250728")]
    TextEditor20250728 {
        /// Tool name (always "str_replace_editor").
        name: TextEditorToolName,
        /// Cache control settings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<super::CacheControl>,
    },

    /// Web search tool for searching the internet.
    #[serde(rename = "web_search_20250305")]
    WebSearch {
        /// Tool name (always "web_search").
        name: WebSearchToolName,
        /// Only include results from these domains.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_domains: Option<Vec<String>>,
        /// Exclude results from these domains.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        blocked_domains: Option<Vec<String>>,
        /// Maximum number of times the tool can be used.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        /// User location for more relevant results.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
        /// Cache control settings.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<super::CacheControl>,
    },
}

/// Bash tool name (always "bash").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BashToolName {
    #[serde(rename = "bash")]
    Bash,
}

impl Default for BashToolName {
    fn default() -> Self {
        BashToolName::Bash
    }
}

/// Text editor tool name (always "str_replace_editor").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TextEditorToolName {
    #[serde(rename = "str_replace_editor")]
    StrReplaceEditor,
}

impl Default for TextEditorToolName {
    fn default() -> Self {
        TextEditorToolName::StrReplaceEditor
    }
}

/// Web search tool name (always "web_search").
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WebSearchToolName {
    #[serde(rename = "web_search")]
    WebSearch,
}

impl Default for WebSearchToolName {
    fn default() -> Self {
        WebSearchToolName::WebSearch
    }
}

/// User location for web search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserLocation {
    /// Location type (always "approximate").
    #[serde(rename = "type")]
    pub location_type: String,
    /// City name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Two-letter ISO country code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Region/state name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// IANA timezone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl UserLocation {
    /// Create a new approximate user location.
    pub fn approximate() -> Self {
        Self {
            location_type: "approximate".to_string(),
            city: None,
            country: None,
            region: None,
            timezone: None,
        }
    }

    /// Set the city.
    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Set the country (two-letter ISO code).
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the timezone (IANA format).
    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

impl ServerTool {
    /// Create a bash tool.
    pub fn bash() -> Self {
        ServerTool::Bash {
            name: BashToolName::Bash,
            cache_control: None,
        }
    }

    /// Create a text editor tool (latest version).
    pub fn text_editor() -> Self {
        ServerTool::TextEditor20250728 {
            name: TextEditorToolName::StrReplaceEditor,
            cache_control: None,
        }
    }

    /// Create a web search tool.
    pub fn web_search() -> Self {
        ServerTool::WebSearch {
            name: WebSearchToolName::WebSearch,
            allowed_domains: None,
            blocked_domains: None,
            max_uses: None,
            user_location: None,
            cache_control: None,
        }
    }

    /// Create a web search tool with allowed domains.
    pub fn web_search_with_allowed_domains(domains: Vec<String>) -> Self {
        ServerTool::WebSearch {
            name: WebSearchToolName::WebSearch,
            allowed_domains: Some(domains),
            blocked_domains: None,
            max_uses: None,
            user_location: None,
            cache_control: None,
        }
    }

    /// Create a web search tool with blocked domains.
    pub fn web_search_with_blocked_domains(domains: Vec<String>) -> Self {
        ServerTool::WebSearch {
            name: WebSearchToolName::WebSearch,
            allowed_domains: None,
            blocked_domains: Some(domains),
            max_uses: None,
            user_location: None,
            cache_control: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_param_serialization() {
        let tool = ToolParam::new(
            "read_file",
            InputSchema::object().required_string("path", "The file path to read"),
        )
        .with_description("Read a file from disk");

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"read_file\""));
        assert!(json.contains("\"description\":\"Read a file from disk\""));
        assert!(json.contains("\"type\":\"object\""));
        assert!(json.contains("\"required\":[\"path\"]"));
    }

    #[test]
    fn test_input_schema_builder() {
        let schema = InputSchema::object()
            .required_string("command", "The command to run")
            .optional_string("cwd", "Working directory");

        assert_eq!(schema.required, vec!["command"]);
        assert!(schema.properties.contains_key("command"));
        assert!(schema.properties.contains_key("cwd"));
    }

    #[test]
    fn test_property_schema_types() {
        let string_prop = PropertySchema::string().with_description("A string");
        assert_eq!(string_prop.schema_type, "string");
        assert_eq!(string_prop.description, Some("A string".to_string()));

        let array_prop = PropertySchema::array(PropertySchema::string());
        assert_eq!(array_prop.schema_type, "array");
        assert!(array_prop.items.is_some());

        let enum_prop = PropertySchema::string()
            .with_enum(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(enum_prop.enum_values, Some(vec!["a".to_string(), "b".to_string()]));
    }

    #[test]
    fn test_complex_tool_schema() {
        let tool = ToolParam::new(
            "search",
            InputSchema::object()
                .property(
                    "query",
                    PropertySchema::string().with_description("Search query"),
                    true,
                )
                .property(
                    "limit",
                    PropertySchema::integer()
                        .with_description("Max results")
                        .with_default(serde_json::json!(10)),
                    false,
                )
                .property(
                    "filters",
                    PropertySchema::array(PropertySchema::string()),
                    false,
                ),
        );

        let json = serde_json::to_string_pretty(&tool).unwrap();
        println!("{}", json);

        // Verify it deserializes back correctly
        let parsed: ToolParam = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "search");
        assert_eq!(parsed.input_schema.required, vec!["query"]);
    }

    #[test]
    fn test_server_tool_bash() {
        let tool = ServerTool::bash();
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"bash_20250124\""));
        assert!(json.contains("\"name\":\"bash\""));
    }

    #[test]
    fn test_server_tool_text_editor() {
        let tool = ServerTool::text_editor();
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"text_editor_20250728\""));
        assert!(json.contains("\"name\":\"str_replace_editor\""));
    }

    #[test]
    fn test_server_tool_web_search() {
        let tool = ServerTool::web_search();
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"web_search_20250305\""));
        assert!(json.contains("\"name\":\"web_search\""));
    }

    #[test]
    fn test_server_tool_web_search_with_domains() {
        let tool = ServerTool::web_search_with_allowed_domains(vec![
            "example.com".to_string(),
            "docs.example.com".to_string(),
        ]);
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"allowed_domains\""));
        assert!(json.contains("example.com"));
    }

    #[test]
    fn test_user_location() {
        let location = UserLocation::approximate()
            .with_city("San Francisco")
            .with_country("US")
            .with_region("California")
            .with_timezone("America/Los_Angeles");

        let json = serde_json::to_string(&location).unwrap();
        assert!(json.contains("\"type\":\"approximate\""));
        assert!(json.contains("\"city\":\"San Francisco\""));
        assert!(json.contains("\"country\":\"US\""));
    }
}
