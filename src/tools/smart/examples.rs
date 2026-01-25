//! JECJIT Code Examples - Just Enough Context, Just In Time
//!
//! Provides valid, idiomatic code examples to improve model adherence.
//! Examples are sourced from:
//! 1. Built-in language idioms (Rust, Nim, Python, etc.)
//! 2. Project-specific patterns (from `.claude/examples/` or codebase analysis)
//! 3. Dynamic extraction from the current codebase
//!
//! Usage:
//! ```rust,ignore
//! let examples = CodeExamples::new(project_root);
//! let ctx = examples.for_task(Lang::Nim, TaskKind::ErrorHandling);
//! // Returns idiomatic Nim error handling examples from this project
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::ast::{AstParser, Lang, Symbol, SymbolKind};

/// Categories of code patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternKind {
    /// Error handling (Result, Option, try/except, etc.)
    ErrorHandling,
    /// Test structure and assertions
    Testing,
    /// API/function signatures and usage
    ApiUsage,
    /// Language-specific idioms
    Idiom,
    /// Async/concurrent patterns
    Async,
    /// Data structures and types
    DataStructure,
    /// Imports and module organization
    Imports,
}

/// A code example with metadata.
#[derive(Debug, Clone)]
pub struct CodeExample {
    /// The actual code
    pub code: String,
    /// What pattern this demonstrates
    pub kind: PatternKind,
    /// Language of the example
    pub lang: Lang,
    /// Short description
    pub description: String,
    /// Source file (if from codebase)
    pub source_file: Option<String>,
    /// Line range in source
    pub line_range: Option<(usize, usize)>,
}

/// JECJIT code examples provider.
pub struct CodeExamples {
    project_root: PathBuf,
    /// Cached project-specific examples
    project_examples: HashMap<(Lang, PatternKind), Vec<CodeExample>>,
    /// Whether we've scanned the project
    scanned: bool,
}

impl CodeExamples {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            project_examples: HashMap::new(),
            scanned: false,
        }
    }

    /// Get examples for a specific task.
    /// Combines built-in idioms with project-specific patterns.
    pub fn for_task(&mut self, lang: Lang, kind: PatternKind) -> Vec<CodeExample> {
        let mut examples = Vec::new();

        // Add built-in idioms first
        examples.extend(self.builtin_examples(lang, kind));

        // Scan project if not done
        if !self.scanned {
            self.scan_project();
        }

        // Add project-specific examples
        if let Some(project) = self.project_examples.get(&(lang, kind)) {
            examples.extend(project.iter().cloned());
        }

        examples
    }

    /// Get all examples for a language.
    pub fn for_language(&mut self, lang: Lang) -> Vec<CodeExample> {
        let kinds = [
            PatternKind::ErrorHandling,
            PatternKind::Testing,
            PatternKind::ApiUsage,
            PatternKind::Idiom,
            PatternKind::Async,
            PatternKind::DataStructure,
            PatternKind::Imports,
        ];

        kinds.iter()
            .flat_map(|k| self.for_task(lang, *k))
            .collect()
    }

    /// Format examples for LLM context.
    pub fn to_context(&mut self, lang: Lang, kinds: &[PatternKind]) -> String {
        let mut output = format!("## {} Code Examples\n\n", lang_name(lang));

        for kind in kinds {
            let examples = self.for_task(lang, *kind);
            if examples.is_empty() {
                continue;
            }

            output.push_str(&format!("### {}\n\n", pattern_name(*kind)));

            for ex in examples.iter().take(2) {
                output.push_str(&format!("**{}**\n", ex.description));
                if let Some(ref file) = ex.source_file {
                    output.push_str(&format!("_From: {}_\n", file));
                }
                output.push_str(&format!("```{}\n{}\n```\n\n", lang_ext(lang), ex.code));
            }
        }

        output
    }

    /// Built-in language idioms.
    fn builtin_examples(&self, lang: Lang, kind: PatternKind) -> Vec<CodeExample> {
        match lang {
            Lang::Nim => self.nim_examples(kind),
            Lang::Rust => self.rust_examples(kind),
            Lang::Python => self.python_examples(kind),
            Lang::Go => self.go_examples(kind),
            _ => vec![],
        }
    }

    fn nim_examples(&self, kind: PatternKind) -> Vec<CodeExample> {
        match kind {
            PatternKind::ErrorHandling => vec![
                CodeExample {
                    code: r#"proc readFile(path: string): Result[string, IOError] =
  try:
    ok(readFile(path))
  except IOError as e:
    err(e)"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Nim,
                    description: "Result type for error handling".to_string(),
                    source_file: None,
                    line_range: None,
                },
                CodeExample {
                    code: r#"proc divide(a, b: int): Option[int] =
  if b == 0:
    none(int)
  else:
    some(a div b)"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Nim,
                    description: "Option type for nullable values".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Testing => vec![
                CodeExample {
                    code: r#"import unittest

suite "Calculator":
  test "addition":
    check add(2, 3) == 5

  test "division by zero":
    expect DivByZeroError:
      discard divide(1, 0)"#.to_string(),
                    kind: PatternKind::Testing,
                    lang: Lang::Nim,
                    description: "unittest suite structure".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Idiom => vec![
                CodeExample {
                    code: r#"# Use 'result' implicit variable instead of explicit return
proc greet(name: string): string =
  result = "Hello, " & name & "!""#.to_string(),
                    kind: PatternKind::Idiom,
                    lang: Lang::Nim,
                    description: "Implicit result variable".to_string(),
                    source_file: None,
                    line_range: None,
                },
                CodeExample {
                    code: r#"# Uniform Function Call Syntax (UFCS)
let nums = @[1, 2, 3, 4, 5]
let doubled = nums.map(x => x * 2).filter(x => x > 4)"#.to_string(),
                    kind: PatternKind::Idiom,
                    lang: Lang::Nim,
                    description: "UFCS method chaining".to_string(),
                    source_file: None,
                    line_range: None,
                },
                CodeExample {
                    code: r#"# Named arguments for clarity
proc connect(host: string, port: int, timeout: int = 30) =
  discard

connect(host = "localhost", port = 8080, timeout = 60)"#.to_string(),
                    kind: PatternKind::Idiom,
                    lang: Lang::Nim,
                    description: "Named arguments".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Async => vec![
                CodeExample {
                    code: r#"import asyncdispatch

proc fetchData(url: string): Future[string] {.async.} =
  let client = newAsyncHttpClient()
  result = await client.getContent(url)

proc main() {.async.} =
  let data = await fetchData("https://example.com")
  echo data

waitFor main()"#.to_string(),
                    kind: PatternKind::Async,
                    lang: Lang::Nim,
                    description: "Async/await pattern".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::DataStructure => vec![
                CodeExample {
                    code: r#"type
  User* = object
    name*: string
    age*: int
    email: string  # private field

proc newUser*(name: string, age: int): User =
  User(name: name, age: age, email: "")"#.to_string(),
                    kind: PatternKind::DataStructure,
                    lang: Lang::Nim,
                    description: "Object type with constructor".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Imports => vec![
                CodeExample {
                    code: r#"import std/[strutils, sequtils, tables, json]
from std/os import fileExists, dirExists
import ./mymodule except internalProc"#.to_string(),
                    kind: PatternKind::Imports,
                    lang: Lang::Nim,
                    description: "Import patterns".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            _ => vec![],
        }
    }

    fn rust_examples(&self, kind: PatternKind) -> Vec<CodeExample> {
        match kind {
            PatternKind::ErrorHandling => vec![
                CodeExample {
                    code: r#"fn read_config(path: &str) -> Result<Config, ConfigError> {
    let content = std::fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&content)?;
    Ok(config)
}"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Rust,
                    description: "? operator for error propagation".to_string(),
                    source_file: None,
                    line_range: None,
                },
                CodeExample {
                    code: r#"fn find_user(id: u64) -> Option<User> {
    users.iter().find(|u| u.id == id).cloned()
}

// Usage with combinators
let name = find_user(42)
    .map(|u| u.name.clone())
    .unwrap_or_else(|| "Unknown".to_string());"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Rust,
                    description: "Option combinators".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Testing => vec![
                CodeExample {
                    code: r#"#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_divide_by_zero() {
        divide(1, 0);
    }
}"#.to_string(),
                    kind: PatternKind::Testing,
                    lang: Lang::Rust,
                    description: "Test module structure".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Idiom => vec![
                CodeExample {
                    code: r#"// Builder pattern
let config = ConfigBuilder::new()
    .host("localhost")
    .port(8080)
    .timeout(Duration::from_secs(30))
    .build()?;"#.to_string(),
                    kind: PatternKind::Idiom,
                    lang: Lang::Rust,
                    description: "Builder pattern".to_string(),
                    source_file: None,
                    line_range: None,
                },
                CodeExample {
                    code: r#"// Iterator chains
let sum: i32 = items
    .iter()
    .filter(|x| x.is_valid())
    .map(|x| x.value)
    .sum();"#.to_string(),
                    kind: PatternKind::Idiom,
                    lang: Lang::Rust,
                    description: "Iterator chains".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Async => vec![
                CodeExample {
                    code: r#"async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

#[tokio::main]
async fn main() {
    let data = fetch_data("https://example.com").await.unwrap();
    println!("{}", data);
}"#.to_string(),
                    kind: PatternKind::Async,
                    lang: Lang::Rust,
                    description: "Async/await with tokio".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            _ => vec![],
        }
    }

    fn python_examples(&self, kind: PatternKind) -> Vec<CodeExample> {
        match kind {
            PatternKind::ErrorHandling => vec![
                CodeExample {
                    code: r#"def read_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Config not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON: {e}")"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Python,
                    description: "Exception handling with context".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Testing => vec![
                CodeExample {
                    code: r#"import pytest

class TestCalculator:
    def test_addition(self):
        assert add(2, 3) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            divide(1, 0)"#.to_string(),
                    kind: PatternKind::Testing,
                    lang: Lang::Python,
                    description: "pytest class structure".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Async => vec![
                CodeExample {
                    code: r#"import asyncio
import aiohttp

async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    data = await fetch_data("https://example.com")
    print(data)

asyncio.run(main())"#.to_string(),
                    kind: PatternKind::Async,
                    lang: Lang::Python,
                    description: "Async/await with aiohttp".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            _ => vec![],
        }
    }

    fn go_examples(&self, kind: PatternKind) -> Vec<CodeExample> {
        match kind {
            PatternKind::ErrorHandling => vec![
                CodeExample {
                    code: r#"func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("reading config: %w", err)
    }

    var config Config
    if err := json.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parsing config: %w", err)
    }

    return &config, nil
}"#.to_string(),
                    kind: PatternKind::ErrorHandling,
                    lang: Lang::Go,
                    description: "Error wrapping pattern".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            PatternKind::Testing => vec![
                CodeExample {
                    code: r#"func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive", 2, 3, 5},
        {"negative", -1, 1, 0},
        {"zero", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Add(tt.a, tt.b)
            if got != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.expected)
            }
        })
    }
}"#.to_string(),
                    kind: PatternKind::Testing,
                    lang: Lang::Go,
                    description: "Table-driven tests".to_string(),
                    source_file: None,
                    line_range: None,
                },
            ],
            _ => vec![],
        }
    }

    /// Scan project for patterns.
    fn scan_project(&mut self) {
        self.scanned = true;

        // Check for .claude/examples/ directory
        let examples_dir = self.project_root.join(".claude/examples");
        if examples_dir.exists() {
            self.load_examples_dir(&examples_dir);
        }

        // Extract patterns from codebase
        self.extract_project_patterns();
    }

    /// Load examples from .claude/examples/ directory.
    fn load_examples_dir(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
            let Some(lang) = Lang::from_extension(ext) else { continue };

            let Ok(content) = std::fs::read_to_string(&path) else { continue };

            // Parse file for examples (marked with comments)
            self.parse_example_file(&content, lang, &path);
        }
    }

    /// Parse an example file with markers.
    /// Format: `# EXAMPLE: PatternKind - Description`
    fn parse_example_file(&mut self, content: &str, lang: Lang, path: &Path) {
        let comment_prefix = match lang {
            Lang::Nim | Lang::Python | Lang::Perl => "#",
            Lang::Rust | Lang::Go | Lang::JavaScript | Lang::TypeScript => "//",
        };

        let mut current_kind: Option<PatternKind> = None;
        let mut current_desc = String::new();
        let mut current_code = String::new();
        let mut start_line = 0;

        for (i, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Check for example marker
            if trimmed.starts_with(comment_prefix) {
                let comment = trimmed.trim_start_matches(comment_prefix).trim();
                if comment.starts_with("EXAMPLE:") {
                    // Save previous example
                    if let Some(kind) = current_kind {
                        if !current_code.trim().is_empty() {
                            self.add_project_example(CodeExample {
                                code: current_code.trim().to_string(),
                                kind,
                                lang,
                                description: current_desc.clone(),
                                source_file: Some(path.to_string_lossy().to_string()),
                                line_range: Some((start_line, i)),
                            });
                        }
                    }

                    // Parse new example marker
                    let rest = comment.trim_start_matches("EXAMPLE:").trim();
                    let parts: Vec<&str> = rest.splitn(2, '-').collect();

                    current_kind = parse_pattern_kind(parts.first().unwrap_or(&"").trim());
                    current_desc = parts.get(1).unwrap_or(&"").trim().to_string();
                    current_code = String::new();
                    start_line = i + 1;
                    continue;
                }
            }

            if current_kind.is_some() {
                current_code.push_str(line);
                current_code.push('\n');
            }
        }

        // Save last example
        if let Some(kind) = current_kind {
            if !current_code.trim().is_empty() {
                self.add_project_example(CodeExample {
                    code: current_code.trim().to_string(),
                    kind,
                    lang,
                    description: current_desc,
                    source_file: Some(path.to_string_lossy().to_string()),
                    line_range: Some((start_line, content.lines().count())),
                });
            }
        }
    }

    /// Extract patterns from the codebase itself.
    fn extract_project_patterns(&mut self) {
        // Look for test files
        self.extract_test_patterns();

        // Look for error handling patterns
        self.extract_error_patterns();
    }

    fn extract_test_patterns(&mut self) {
        // Find test files
        let patterns = ["**/test*.rs", "**/test*.nim", "**/test*.py", "**/*_test.go"];

        for pattern in patterns {
            let glob = glob::glob(&self.project_root.join(pattern).to_string_lossy());
            let Ok(paths) = glob else { continue };

            for path in paths.flatten().take(2) {
                let Ok(content) = std::fs::read_to_string(&path) else { continue };
                let Some(lang) = Lang::from_path(&path) else { continue };

                // Extract first test function as example
                if let Some(example) = self.extract_first_test(&content, lang, &path) {
                    self.add_project_example(example);
                }
            }
        }
    }

    fn extract_first_test(&self, content: &str, lang: Lang, path: &Path) -> Option<CodeExample> {
        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(content, lang);

        // Find first test function
        let test_fn = symbols.iter().find(|s| {
            matches!(s.kind, SymbolKind::Function) &&
            (s.name.starts_with("test") || s.name.starts_with("Test"))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let code: String = lines[test_fn.start_line.saturating_sub(1)..test_fn.end_line]
            .join("\n");

        Some(CodeExample {
            code,
            kind: PatternKind::Testing,
            lang,
            description: format!("Test pattern from {}", path.file_name()?.to_string_lossy()),
            source_file: Some(path.to_string_lossy().to_string()),
            line_range: Some((test_fn.start_line, test_fn.end_line)),
        })
    }

    fn extract_error_patterns(&mut self) {
        // This could be expanded to detect error handling patterns
        // For now, we rely on built-in examples
    }

    fn add_project_example(&mut self, example: CodeExample) {
        let key = (example.lang, example.kind);
        self.project_examples
            .entry(key)
            .or_default()
            .push(example);
    }
}

fn lang_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "Rust",
        Lang::Nim => "Nim",
        Lang::Python => "Python",
        Lang::Go => "Go",
        Lang::JavaScript => "JavaScript",
        Lang::TypeScript => "TypeScript",
        Lang::Perl => "Perl",
    }
}

fn lang_ext(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "rust",
        Lang::Nim => "nim",
        Lang::Python => "python",
        Lang::Go => "go",
        Lang::JavaScript => "javascript",
        Lang::TypeScript => "typescript",
        Lang::Perl => "perl",
    }
}

fn pattern_name(kind: PatternKind) -> &'static str {
    match kind {
        PatternKind::ErrorHandling => "Error Handling",
        PatternKind::Testing => "Testing",
        PatternKind::ApiUsage => "API Usage",
        PatternKind::Idiom => "Idioms",
        PatternKind::Async => "Async Patterns",
        PatternKind::DataStructure => "Data Structures",
        PatternKind::Imports => "Imports",
    }
}

fn parse_pattern_kind(s: &str) -> Option<PatternKind> {
    match s.to_lowercase().as_str() {
        "error" | "errorhandling" | "error_handling" => Some(PatternKind::ErrorHandling),
        "test" | "testing" => Some(PatternKind::Testing),
        "api" | "apiusage" | "api_usage" => Some(PatternKind::ApiUsage),
        "idiom" | "idioms" => Some(PatternKind::Idiom),
        "async" | "concurrent" => Some(PatternKind::Async),
        "data" | "datastructure" | "data_structure" => Some(PatternKind::DataStructure),
        "import" | "imports" => Some(PatternKind::Imports),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_nim_examples() {
        let examples = CodeExamples::new("/tmp");
        let nim_error = examples.builtin_examples(Lang::Nim, PatternKind::ErrorHandling);

        assert!(!nim_error.is_empty());
        assert!(nim_error[0].code.contains("Result"));
    }

    #[test]
    fn test_builtin_rust_examples() {
        let examples = CodeExamples::new("/tmp");
        let rust_error = examples.builtin_examples(Lang::Rust, PatternKind::ErrorHandling);

        assert!(!rust_error.is_empty());
        assert!(rust_error[0].code.contains("Result"));
    }

    #[test]
    fn test_context_output() {
        let mut examples = CodeExamples::new("/tmp");
        let ctx = examples.to_context(Lang::Nim, &[PatternKind::Idiom]);

        assert!(ctx.contains("## Nim Code Examples"));
        assert!(ctx.contains("### Idioms"));
        assert!(ctx.contains("```nim"));
    }

    #[test]
    fn test_for_task() {
        let mut examples = CodeExamples::new("/tmp");
        let testing = examples.for_task(Lang::Rust, PatternKind::Testing);

        assert!(!testing.is_empty());
        assert!(testing[0].code.contains("#[test]"));
    }
}
