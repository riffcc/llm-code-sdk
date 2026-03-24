//! Skills API support for Claude agents.
//!
//! Skills are reusable capabilities that can be applied to sessions,
//! enabling specialization (TypeScript expert, Vue.js expert, etc.).
//!
//! # Anthropic Skills Standard
//!
//! Skills are managed through the Anthropic API:
//! - Skills have names, descriptions, and versions
//! - Each version has a definition with capabilities
//! - Skills are passed via the `container` parameter in messages
//!
//! # Example
//!
//! ```rust,no_run
//! use llm_code_sdk::skills::{Container, SkillRef, SkillType};
//!
//! // Reference a skill in a message
//! let container = Container {
//!     id: "session-123".to_string(),
//!     skills: vec![
//!         SkillRef {
//!             skill_id: "typescript-expert".to_string(),
//!             skill_type: SkillType::Anthropic,
//!             version: "1.0.0".to_string(),
//!         }
//!     ],
//! };
//! ```

pub mod agent_skills;

pub use agent_skills::{
    ActivateSkillTool, AgentSkill, SkillMeta, SkillRegistry, SkillResourceTool,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A skill definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique skill identifier.
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// Description of what the skill does.
    pub description: String,

    /// When the skill was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// When the skill was last updated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// A skill version with its definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillVersion {
    /// The skill this version belongs to.
    pub skill_id: String,

    /// Version string (e.g., "1.0.0").
    pub version: String,

    /// The skill definition for this version.
    pub definition: SkillDefinition,

    /// When this version was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// The definition of a skill version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    /// Capabilities this skill provides.
    #[serde(default)]
    pub capabilities: Vec<String>,

    /// System prompt additions for this skill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Instructions for the skill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Additional metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for SkillDefinition {
    fn default() -> Self {
        Self {
            capabilities: Vec::new(),
            system_prompt: None,
            instructions: None,
            metadata: HashMap::new(),
        }
    }
}

/// Type of skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SkillType {
    /// Anthropic-managed skill.
    Anthropic,
    /// Custom skill.
    Custom,
}

impl Default for SkillType {
    fn default() -> Self {
        SkillType::Custom
    }
}

/// Reference to a skill for use in messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRef {
    /// The skill ID.
    pub skill_id: String,

    /// Type of skill.
    #[serde(rename = "type")]
    pub skill_type: SkillType,

    /// Version to use.
    pub version: String,
}

impl SkillRef {
    /// Create a new skill reference.
    pub fn new(skill_id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            skill_id: skill_id.into(),
            skill_type: SkillType::Custom,
            version: version.into(),
        }
    }

    /// Create an Anthropic skill reference.
    pub fn anthropic(skill_id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            skill_id: skill_id.into(),
            skill_type: SkillType::Anthropic,
            version: version.into(),
        }
    }
}

/// Container for skills and context in a message.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Container {
    /// Container identifier (e.g., session ID).
    pub id: String,

    /// Skills to apply to this container.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillRef>,
}

impl Container {
    /// Create a new container with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            skills: Vec::new(),
        }
    }

    /// Add a skill to this container.
    pub fn with_skill(mut self, skill: SkillRef) -> Self {
        self.skills.push(skill);
        self
    }

    /// Add multiple skills to this container.
    pub fn with_skills(mut self, skills: impl IntoIterator<Item = SkillRef>) -> Self {
        self.skills.extend(skills);
        self
    }
}

/// Beta features that can be enabled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BetaFeature {
    /// Skills API (2025-01-01 version).
    Skills,
    /// Context management.
    ContextManagement,
    /// Custom beta feature.
    Custom(String),
}

impl BetaFeature {
    /// Get the beta header value for this feature.
    pub fn header_value(&self) -> &str {
        match self {
            BetaFeature::Skills => "skills-2025-01-01",
            BetaFeature::ContextManagement => "context-management-2025-01-01",
            BetaFeature::Custom(s) => s,
        }
    }
}

impl Serialize for BetaFeature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.header_value())
    }
}

impl<'de> Deserialize<'de> for BetaFeature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "skills-2025-01-01" => BetaFeature::Skills,
            "context-management-2025-01-01" => BetaFeature::ContextManagement,
            _ => BetaFeature::Custom(s),
        })
    }
}

/// A local skill file (markdown format like Claude Code uses).
#[derive(Debug, Clone)]
pub struct LocalSkill {
    /// Skill name (derived from filename).
    pub name: String,

    /// Skill content (instructions).
    pub content: String,

    /// Source path.
    pub path: Option<std::path::PathBuf>,
}

impl LocalSkill {
    /// Load a skill from a markdown file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();

        Ok(Self {
            name,
            content,
            path: Some(path.to_path_buf()),
        })
    }

    /// Create a skill from content directly.
    pub fn from_content(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            path: None,
        }
    }

    /// Convert to a system prompt addition.
    pub fn to_system_prompt(&self) -> String {
        format!("## Skill: {}\n\n{}", self.name, self.content)
    }
}

/// Manager for loading and stacking skills.
#[derive(Debug, Default)]
pub struct SkillStack {
    /// Skills in order (base to specialized).
    skills: Vec<LocalSkill>,
}

impl SkillStack {
    /// Create a new empty skill stack.
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Push a skill onto the stack.
    pub fn push(&mut self, skill: LocalSkill) {
        self.skills.push(skill);
    }

    /// Load and push a skill from a file.
    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let skill = LocalSkill::from_file(path)?;
        self.push(skill);
        Ok(())
    }

    /// Load skills from a directory.
    pub fn load_dir(&mut self, dir: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let dir = dir.as_ref();
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "md").unwrap_or(false) {
                    self.load(&path)?;
                }
            }
        }
        Ok(())
    }

    /// Get all skills in the stack.
    pub fn skills(&self) -> &[LocalSkill] {
        &self.skills
    }

    /// Generate a combined system prompt from all skills.
    pub fn to_system_prompt(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("# Active Skills\n\n");
        for skill in &self.skills {
            prompt.push_str(&skill.to_system_prompt());
            prompt.push_str("\n\n---\n\n");
        }
        prompt
    }

    /// Check if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Get the number of skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_ref_creation() {
        let skill = SkillRef::new("typescript-expert", "1.0.0");
        assert_eq!(skill.skill_id, "typescript-expert");
        assert_eq!(skill.version, "1.0.0");
        assert_eq!(skill.skill_type, SkillType::Custom);
    }

    #[test]
    fn test_container_with_skills() {
        let container = Container::new("session-123")
            .with_skill(SkillRef::new("base", "1.0"))
            .with_skill(SkillRef::new("typescript", "2.0"));

        assert_eq!(container.id, "session-123");
        assert_eq!(container.skills.len(), 2);
    }

    #[test]
    fn test_beta_feature_serialization() {
        let feature = BetaFeature::Skills;
        let json = serde_json::to_string(&feature).unwrap();
        assert_eq!(json, "\"skills-2025-01-01\"");
    }

    #[test]
    fn test_local_skill_from_content() {
        let skill = LocalSkill::from_content("test", "Do something");
        assert_eq!(skill.name, "test");
        assert_eq!(skill.content, "Do something");
    }

    #[test]
    fn test_skill_stack() {
        let mut stack = SkillStack::new();
        stack.push(LocalSkill::from_content("base", "Base instructions"));
        stack.push(LocalSkill::from_content("typescript", "TS instructions"));

        assert_eq!(stack.len(), 2);

        let prompt = stack.to_system_prompt();
        assert!(prompt.contains("Base instructions"));
        assert!(prompt.contains("TS instructions"));
    }
}
