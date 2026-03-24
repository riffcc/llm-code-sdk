//! Agent Skills implementation — the agentskills.io spec.
//!
//! Progressive disclosure:
//! 1. Startup: name + description for all skills (~100 tokens each)
//! 2. Activation: full SKILL.md body loaded on demand
//! 3. Resources: scripts/references/assets loaded when needed

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::tools::{Tool, ToolResult};
use crate::types::{InputSchema, ToolParam};

/// Parsed SKILL.md frontmatter.
#[derive(Debug, Clone)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub metadata: HashMap<String, String>,
    pub allowed_tools: Vec<String>,
}

/// A discovered agent skill.
#[derive(Debug, Clone)]
pub struct AgentSkill {
    /// Parsed frontmatter.
    pub meta: SkillMeta,
    /// Full SKILL.md body (instructions).
    pub body: String,
    /// Root directory of the skill.
    pub path: PathBuf,
}

/// Registry of discovered skills.
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, AgentSkill>,
    activated: HashMap<String, bool>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Discover skills from a directory. Each subdirectory with a SKILL.md is a skill.
    pub fn discover(&mut self, dir: &Path) {
        if !dir.is_dir() {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_md = path.join("SKILL.md");
                if skill_md.exists() {
                    if let Ok(skill) = parse_skill(&path) {
                        self.skills.insert(skill.meta.name.clone(), skill);
                    }
                }
            }
        }
    }

    /// Get the startup catalog — name + description only, for the system prompt.
    pub fn catalog_prompt(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }

        let mut lines = vec!["# Available Skills".to_string()];
        lines.push(String::new());
        lines.push("Use the `activate_skill` tool to load a skill's full instructions.".to_string());
        lines.push(String::new());

        for skill in self.skills.values() {
            lines.push(format!("- **{}**: {}", skill.meta.name, skill.meta.description));
        }

        lines.join("\n")
    }

    /// Activate a skill — returns its full body.
    pub fn activate(&mut self, name: &str) -> Option<&AgentSkill> {
        if self.skills.contains_key(name) {
            self.activated.insert(name.to_string(), true);
            self.skills.get(name)
        } else {
            None
        }
    }

    /// Check if a skill is activated.
    pub fn is_activated(&self, name: &str) -> bool {
        self.activated.get(name).copied().unwrap_or(false)
    }

    /// Read a resource file from a skill.
    pub fn read_resource(&self, skill_name: &str, resource_path: &str) -> Result<String, String> {
        let skill = self.skills.get(skill_name)
            .ok_or_else(|| format!("Unknown skill: {}", skill_name))?;

        if !self.is_activated(skill_name) {
            return Err(format!("Skill '{}' is not activated. Activate it first.", skill_name));
        }

        let full_path = skill.path.join(resource_path);

        // Security: don't escape the skill directory
        let canonical = full_path.canonicalize()
            .map_err(|e| format!("Invalid path: {}", e))?;
        let skill_canonical = skill.path.canonicalize()
            .map_err(|e| format!("Skill path error: {}", e))?;
        if !canonical.starts_with(&skill_canonical) {
            return Err("Path must be within skill directory".to_string());
        }

        std::fs::read_to_string(&full_path)
            .map_err(|e| format!("Failed to read resource: {}", e))
    }

    /// List all skill names.
    pub fn list(&self) -> Vec<&str> {
        self.skills.keys().map(|s| s.as_str()).collect()
    }

    /// Get a skill by name.
    pub fn get(&self, name: &str) -> Option<&AgentSkill> {
        self.skills.get(name)
    }

    /// Number of discovered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

/// Parse a SKILL.md from a skill directory.
fn parse_skill(skill_dir: &Path) -> Result<AgentSkill, String> {
    let skill_md = skill_dir.join("SKILL.md");
    let raw = std::fs::read_to_string(&skill_md)
        .map_err(|e| format!("Failed to read SKILL.md: {}", e))?;

    let (meta, body) = parse_frontmatter(&raw)?;

    // Validate name matches directory
    let dir_name = skill_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if meta.name != dir_name {
        return Err(format!(
            "Skill name '{}' doesn't match directory name '{}'",
            meta.name, dir_name
        ));
    }

    Ok(AgentSkill {
        meta,
        body,
        path: skill_dir.to_path_buf(),
    })
}

/// Parse YAML frontmatter from a SKILL.md file.
fn parse_frontmatter(raw: &str) -> Result<(SkillMeta, String), String> {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return Err("SKILL.md must start with YAML frontmatter (---)".to_string());
    }

    let after_first = &trimmed[3..];
    let end = after_first.find("\n---")
        .ok_or("SKILL.md frontmatter must be closed with ---")?;

    let yaml_str = &after_first[..end];
    let body = after_first[end + 4..].trim().to_string();

    // Parse YAML manually (avoid adding a yaml dependency)
    let mut name = None;
    let mut description = None;
    let mut license = None;
    let mut compatibility = None;
    let mut metadata = HashMap::new();
    let mut allowed_tools = Vec::new();
    let mut in_metadata = false;

    for line in yaml_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if we're in a nested block
        if !line.starts_with(' ') && !line.starts_with('\t') {
            in_metadata = false;
        }

        if in_metadata {
            if let Some((k, v)) = trimmed.split_once(':') {
                let k = k.trim().trim_matches('"');
                let v = v.trim().trim_matches('"');
                metadata.insert(k.to_string(), v.to_string());
            }
            continue;
        }

        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim();
            let value = value.trim();

            match key {
                "name" => name = Some(value.trim_matches('"').to_string()),
                "description" => description = Some(value.trim_matches('"').to_string()),
                "license" => license = Some(value.trim_matches('"').to_string()),
                "compatibility" => compatibility = Some(value.trim_matches('"').to_string()),
                "allowed-tools" => {
                    allowed_tools = value.split_whitespace().map(|s| s.to_string()).collect();
                }
                "metadata" => {
                    in_metadata = true;
                }
                _ => {}
            }
        }
    }

    let name = name.ok_or("SKILL.md frontmatter must include 'name'")?;
    let description = description.ok_or("SKILL.md frontmatter must include 'description'")?;

    Ok((
        SkillMeta {
            name,
            description,
            license,
            compatibility,
            metadata,
            allowed_tools,
        },
        body,
    ))
}

// --- Tools ---

use std::sync::{Arc, RwLock};

/// Tool for activating a skill — loads its full instructions.
pub struct ActivateSkillTool {
    registry: Arc<RwLock<SkillRegistry>>,
}

impl ActivateSkillTool {
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ActivateSkillTool {
    fn name(&self) -> &str {
        "activate_skill"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "activate_skill",
            InputSchema::object()
                .required_string("name", "Name of the skill to activate"),
        )
        .with_description("Activate a skill to load its full instructions. See the Available Skills list in the system prompt.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let name = input.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if name.is_empty() {
            return ToolResult::error("name is required");
        }

        let mut registry = self.registry.write().unwrap();
        match registry.activate(name) {
            Some(skill) => {
                let mut output = format!("# Skill: {}\n\n", skill.meta.name);
                output.push_str(&skill.body);

                // List available resources
                let resources = list_resources(&skill.path);
                if !resources.is_empty() {
                    output.push_str("\n\n## Available Resources\n\n");
                    output.push_str("Use `skill_resource` tool to read these:\n\n");
                    for r in &resources {
                        output.push_str(&format!("- {}\n", r));
                    }
                }

                ToolResult::success(output)
            }
            None => ToolResult::error(format!("Unknown skill: '{}'. Check the Available Skills list.", name)),
        }
    }
}

/// Tool for reading a resource file from an activated skill.
pub struct SkillResourceTool {
    registry: Arc<RwLock<SkillRegistry>>,
}

impl SkillResourceTool {
    pub fn new(registry: Arc<RwLock<SkillRegistry>>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for SkillResourceTool {
    fn name(&self) -> &str {
        "skill_resource"
    }

    fn to_param(&self) -> ToolParam {
        ToolParam::new(
            "skill_resource",
            InputSchema::object()
                .required_string("skill", "Name of the activated skill")
                .required_string("path", "Relative path to the resource file"),
        )
        .with_description("Read a resource file (script, reference, asset) from an activated skill.")
    }

    async fn call(&self, input: HashMap<String, serde_json::Value>) -> ToolResult {
        let skill = input.get("skill").and_then(|v| v.as_str()).unwrap_or("");
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        if skill.is_empty() || path.is_empty() {
            return ToolResult::error("Both 'skill' and 'path' are required");
        }

        let registry = self.registry.read().unwrap();
        match registry.read_resource(skill, path) {
            Ok(content) => ToolResult::success(content),
            Err(e) => ToolResult::error(e),
        }
    }
}

/// List resource files in a skill directory (excluding SKILL.md).
fn list_resources(skill_dir: &Path) -> Vec<String> {
    let mut resources = Vec::new();
    list_resources_recursive(skill_dir, skill_dir, &mut resources);
    resources
}

fn list_resources_recursive(root: &Path, dir: &Path, resources: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let name = relative.to_string_lossy().to_string();

        if path.is_dir() {
            list_resources_recursive(root, &path, resources);
        } else if name != "SKILL.md" {
            resources.push(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_skill(dir: &Path, name: &str, description: &str, body: &str) {
        let skill_dir = dir.join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {description}\n---\n{body}"),
        )
        .unwrap();
    }

    #[test]
    fn test_discover_skills() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "code-review", "Review code for quality", "## Steps\n1. Read the code\n2. Check for issues");
        create_test_skill(dir.path(), "testing", "Write and run tests", "## Steps\n1. Write tests");

        let mut registry = SkillRegistry::new();
        registry.discover(dir.path());

        assert_eq!(registry.len(), 2);
        assert!(registry.get("code-review").is_some());
        assert!(registry.get("testing").is_some());
    }

    #[test]
    fn test_catalog_prompt() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "code-review", "Review code for quality", "body");

        let mut registry = SkillRegistry::new();
        registry.discover(dir.path());

        let prompt = registry.catalog_prompt();
        assert!(prompt.contains("code-review"));
        assert!(prompt.contains("Review code for quality"));
        assert!(!prompt.contains("body")); // Body should NOT be in catalog
    }

    #[test]
    fn test_activate_skill() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "code-review", "Review code", "Do the review.");

        let mut registry = SkillRegistry::new();
        registry.discover(dir.path());

        assert!(!registry.is_activated("code-review"));

        let skill = registry.activate("code-review").unwrap();
        assert_eq!(skill.body, "Do the review.");
        assert!(registry.is_activated("code-review"));
    }

    #[test]
    fn test_read_resource() {
        let dir = TempDir::new().unwrap();
        create_test_skill(dir.path(), "my-skill", "A skill", "body");

        let refs_dir = dir.path().join("my-skill").join("references");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("REFERENCE.md"), "# Reference\nSome docs").unwrap();

        let mut registry = SkillRegistry::new();
        registry.discover(dir.path());

        // Can't read before activation
        assert!(registry.read_resource("my-skill", "references/REFERENCE.md").is_err());

        registry.activate("my-skill");
        let content = registry.read_resource("my-skill", "references/REFERENCE.md").unwrap();
        assert!(content.contains("Some docs"));
    }

    #[test]
    fn test_frontmatter_parsing() {
        let raw = "---\nname: test-skill\ndescription: A test skill\nlicense: MIT\nallowed-tools: Bash(git:*) Read\nmetadata:\n  author: test\n  version: \"1.0\"\n---\n\n## Instructions\nDo stuff.";
        let (meta, body) = parse_frontmatter(raw).unwrap();

        assert_eq!(meta.name, "test-skill");
        assert_eq!(meta.description, "A test skill");
        assert_eq!(meta.license.as_deref(), Some("MIT"));
        assert_eq!(meta.allowed_tools, vec!["Bash(git:*)", "Read"]);
        assert_eq!(meta.metadata.get("author").map(|s| s.as_str()), Some("test"));
        assert!(body.contains("Do stuff."));
    }
}
