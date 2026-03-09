//! Staged transaction system for code changes.
//!
//! Allows building up arbitrarily large changesets (even entire codebase refactors)
//! with shadow edits that only commit when preconditions are met.
//!
//! ## Stages
//!
//! 1. **Setup**: Define preconditions (tests, compilation, custom checks)
//! 2. **Build**: Accumulate shadow edits across many files
//! 3. **Validate**: Check preconditions against shadow state
//! 4. **Commit**: Apply atomically when all preconditions pass
//!
//! ## Example Flow
//!
//! ```ignore
//! let mut tx = ChangeTransaction::new("rename-api-v2");
//!
//! // Setup: define what success looks like
//! tx.require_compiles();
//! tx.require_tests(&["tests/api_v2.rs"]);
//! tx.add_test_file("tests/api_v2.rs", test_content);  // test is part of changeset!
//!
//! // Build: accumulate shadow edits
//! tx.shadow_edit("src/api.rs", Edit::replace_symbol("ApiV1", new_api_content));
//! tx.shadow_edit("src/client.rs", Edit::replace_lines(10, 20, new_client_code));
//! tx.shadow_edit("src/server.rs", Edit::insert_after("handle_request", new_handler));
//!
//! // Validate: check preconditions
//! let result = tx.validate().await;
//! if result.is_ok() {
//!     tx.commit();  // Atomic apply
//! } else {
//!     // Iterate: model can see what failed and add more edits
//!     tx.shadow_edit("src/api.rs", Edit::fix_error(result.errors[0]));
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::smart_write::StructuralEdit;

/// A precondition that must be met before committing.
#[derive(Debug, Clone)]
pub enum Precondition {
    /// Code must compile (optionally just specific crates).
    Compiles { crates: Option<Vec<String>> },
    /// Specific tests must pass.
    TestsPass { patterns: Vec<String> },
    /// All tests must pass.
    AllTestsPass,
    /// Custom check command.
    CustomCheck {
        command: String,
        description: String,
    },
    /// A file must contain specific content.
    FileContains { path: String, content: String },
    /// A file must NOT contain specific content.
    FileNotContains { path: String, content: String },
}

/// Result of validating a precondition.
#[derive(Debug, Clone)]
pub struct PreconditionResult {
    pub precondition: Precondition,
    pub passed: bool,
    pub message: String,
    pub output: Option<String>,
}

/// A shadow edit waiting to be applied.
#[derive(Debug, Clone)]
pub struct ShadowEdit {
    pub path: PathBuf,
    pub edit: StructuralEdit,
    pub original_content: Option<String>,
    pub shadowed_content: Option<String>,
}

/// Validation result for the entire transaction.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub precondition_results: Vec<PreconditionResult>,
    pub compilation_errors: Vec<String>,
    pub test_failures: Vec<String>,
    pub shadow_file_count: usize,
}

impl ValidationResult {
    pub fn success(shadow_count: usize) -> Self {
        Self {
            passed: true,
            precondition_results: vec![],
            compilation_errors: vec![],
            test_failures: vec![],
            shadow_file_count: shadow_count,
        }
    }

    pub fn failed(reason: &str) -> Self {
        Self {
            passed: false,
            precondition_results: vec![],
            compilation_errors: vec![reason.to_string()],
            test_failures: vec![],
            shadow_file_count: 0,
        }
    }

    /// Get a summary of what failed.
    pub fn failure_summary(&self) -> String {
        let mut parts = vec![];

        if !self.compilation_errors.is_empty() {
            parts.push(format!(
                "Compilation: {} errors",
                self.compilation_errors.len()
            ));
        }

        if !self.test_failures.is_empty() {
            parts.push(format!("Tests: {} failures", self.test_failures.len()));
        }

        let failed_preconditions: Vec<_> = self
            .precondition_results
            .iter()
            .filter(|r| !r.passed)
            .collect();

        if !failed_preconditions.is_empty() {
            parts.push(format!(
                "Preconditions: {} failed",
                failed_preconditions.len()
            ));
        }

        if parts.is_empty() {
            "Unknown failure".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Transaction state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Building up edits.
    Building,
    /// Validation in progress.
    Validating,
    /// Validated and ready to commit.
    Ready,
    /// Committed successfully.
    Committed,
    /// Rolled back.
    RolledBack,
}

/// A staged transaction for code changes.
pub struct ChangeTransaction {
    /// Transaction name/description.
    pub name: String,
    /// Project root for resolving paths.
    project_root: PathBuf,
    /// Current state.
    state: TransactionState,
    /// Preconditions to check before commit.
    preconditions: Vec<Precondition>,
    /// Shadow edits accumulated.
    shadow_edits: Vec<ShadowEdit>,
    /// Shadow file contents (path -> content).
    shadow_files: HashMap<PathBuf, String>,
    /// New files to create (not edits to existing).
    new_files: HashMap<PathBuf, String>,
    /// Files to delete.
    delete_files: Vec<PathBuf>,
    /// Temporary directory for shadow compilation.
    shadow_dir: Option<PathBuf>,
    /// Last validation result.
    last_validation: Option<ValidationResult>,
    /// Whether to skip compilation check.
    skip_compile: bool,
}

impl ChangeTransaction {
    /// Create a new transaction.
    pub fn new(name: &str, project_root: impl Into<PathBuf>) -> Self {
        Self {
            name: name.to_string(),
            project_root: project_root.into(),
            state: TransactionState::Building,
            preconditions: vec![],
            shadow_edits: vec![],
            shadow_files: HashMap::new(),
            new_files: HashMap::new(),
            delete_files: vec![],
            shadow_dir: None,
            last_validation: None,
            skip_compile: false,
        }
    }

    /// Skip compilation check (for when refactoring and expecting breakage).
    pub fn skip_compilation(&mut self) -> &mut Self {
        self.skip_compile = true;
        self
    }

    /// Require that the code compiles.
    pub fn require_compiles(&mut self) -> &mut Self {
        self.preconditions
            .push(Precondition::Compiles { crates: None });
        self
    }

    /// Require that specific crates compile.
    pub fn require_crates_compile(&mut self, crates: Vec<String>) -> &mut Self {
        self.preconditions.push(Precondition::Compiles {
            crates: Some(crates),
        });
        self
    }

    /// Require that specific tests pass.
    pub fn require_tests(&mut self, patterns: Vec<String>) -> &mut Self {
        self.preconditions
            .push(Precondition::TestsPass { patterns });
        self
    }

    /// Require all tests pass.
    pub fn require_all_tests(&mut self) -> &mut Self {
        self.preconditions.push(Precondition::AllTestsPass);
        self
    }

    /// Add a custom check.
    pub fn require_custom(&mut self, command: &str, description: &str) -> &mut Self {
        self.preconditions.push(Precondition::CustomCheck {
            command: command.to_string(),
            description: description.to_string(),
        });
        self
    }

    /// Require a file contains specific content.
    pub fn require_file_contains(&mut self, path: &str, content: &str) -> &mut Self {
        self.preconditions.push(Precondition::FileContains {
            path: path.to_string(),
            content: content.to_string(),
        });
        self
    }

    /// Add a shadow edit to an existing file.
    pub fn shadow_edit(&mut self, path: &str, edit: StructuralEdit) -> Result<(), String> {
        if self.state != TransactionState::Building {
            return Err("Transaction not in building state".to_string());
        }

        let full_path = self.resolve_path(path);

        // Read original content if not already shadowed
        if !self.shadow_files.contains_key(&full_path) {
            let content = std::fs::read_to_string(&full_path)
                .map_err(|e| format!("Failed to read {}: {}", path, e))?;
            self.shadow_files.insert(full_path.clone(), content.clone());
        }

        self.shadow_edits.push(ShadowEdit {
            path: full_path,
            edit,
            original_content: None,
            shadowed_content: None,
        });

        Ok(())
    }

    /// Add a new file (not an edit).
    pub fn add_file(&mut self, path: &str, content: &str) -> Result<(), String> {
        if self.state != TransactionState::Building {
            return Err("Transaction not in building state".to_string());
        }

        let full_path = self.resolve_path(path);
        self.new_files.insert(full_path, content.to_string());
        Ok(())
    }

    /// Add a test file as part of the changeset.
    pub fn add_test_file(&mut self, path: &str, content: &str) -> Result<(), String> {
        self.add_file(path, content)
    }

    /// Mark a file for deletion.
    pub fn delete_file(&mut self, path: &str) -> Result<(), String> {
        if self.state != TransactionState::Building {
            return Err("Transaction not in building state".to_string());
        }

        let full_path = self.resolve_path(path);
        self.delete_files.push(full_path);
        Ok(())
    }

    /// Get the current shadow content of a file.
    pub fn get_shadow_content(&self, path: &str) -> Option<&str> {
        let full_path = self.resolve_path(path);
        self.shadow_files.get(&full_path).map(|s| s.as_str())
    }

    /// Get count of shadow edits.
    pub fn edit_count(&self) -> usize {
        self.shadow_edits.len() + self.new_files.len() + self.delete_files.len()
    }

    /// Get list of affected files.
    pub fn affected_files(&self) -> Vec<&Path> {
        let mut files: Vec<&Path> = self.shadow_files.keys().map(|p| p.as_path()).collect();
        files.extend(self.new_files.keys().map(|p| p.as_path()));
        files.extend(self.delete_files.iter().map(|p| p.as_path()));
        files.sort();
        files.dedup();
        files
    }

    /// Apply shadow edits to get final content.
    fn apply_shadow_edits(&mut self) -> Result<(), String> {
        // For each file with edits, apply them in order
        let mut files_to_edit: HashMap<PathBuf, Vec<&StructuralEdit>> = HashMap::new();

        for shadow_edit in &self.shadow_edits {
            files_to_edit
                .entry(shadow_edit.path.clone())
                .or_default()
                .push(&shadow_edit.edit);
        }

        for (path, edits) in files_to_edit {
            let mut content = self
                .shadow_files
                .get(&path)
                .cloned()
                .ok_or_else(|| format!("No shadow content for {:?}", path))?;

            // Apply edits in order (note: this is simplified, real impl would need smarter merging)
            for edit in edits {
                content = self.apply_single_edit(&content, &path, edit)?;
            }

            self.shadow_files.insert(path, content);
        }

        Ok(())
    }

    /// Apply a single edit to content.
    fn apply_single_edit(
        &self,
        content: &str,
        path: &Path,
        edit: &StructuralEdit,
    ) -> Result<String, String> {
        use super::ast::{AstParser, Lang};
        use super::smart_write::EditGranularity;

        let lang = Lang::from_path(path).ok_or_else(|| "Unsupported language".to_string())?;

        let mut parser = AstParser::new();
        let symbols = parser.extract_symbols(content, lang);

        match edit.granularity {
            EditGranularity::Function | EditGranularity::Symbol => {
                let symbol = symbols
                    .iter()
                    .find(|s| s.name == edit.target)
                    .ok_or_else(|| format!("Symbol '{}' not found", edit.target))?;

                let lines: Vec<&str> = content.lines().collect();
                let mut result = String::new();

                for line in &lines[..symbol.start_line - 1] {
                    result.push_str(line);
                    result.push('\n');
                }
                result.push_str(&edit.new_content);
                if !edit.new_content.ends_with('\n') {
                    result.push('\n');
                }
                for line in &lines[symbol.end_line..] {
                    result.push_str(line);
                    result.push('\n');
                }

                Ok(result)
            }
            EditGranularity::Insert => {
                let after = edit
                    .after
                    .as_ref()
                    .ok_or("Insert requires 'after' target")?;
                let symbol = symbols
                    .iter()
                    .find(|s| s.name == *after)
                    .ok_or_else(|| format!("Symbol '{}' not found", after))?;

                let lines: Vec<&str> = content.lines().collect();
                let mut result = String::new();

                for line in &lines[..symbol.end_line] {
                    result.push_str(line);
                    result.push('\n');
                }
                result.push('\n');
                result.push_str(&edit.new_content);
                if !edit.new_content.ends_with('\n') {
                    result.push('\n');
                }
                for line in &lines[symbol.end_line..] {
                    result.push_str(line);
                    result.push('\n');
                }

                Ok(result)
            }
            EditGranularity::Delete => {
                let symbol = symbols
                    .iter()
                    .find(|s| s.name == edit.target)
                    .ok_or_else(|| format!("Symbol '{}' not found", edit.target))?;

                let lines: Vec<&str> = content.lines().collect();
                let mut result = String::new();

                for line in &lines[..symbol.start_line - 1] {
                    result.push_str(line);
                    result.push('\n');
                }
                for line in &lines[symbol.end_line..] {
                    result.push_str(line);
                    result.push('\n');
                }

                Ok(result)
            }
            EditGranularity::Lines => {
                let parts: Vec<&str> = edit.target.split(':').collect();
                if parts.len() != 2 {
                    return Err("Line range must be 'start:end'".to_string());
                }

                let start: usize = parts[0].parse().map_err(|_| "Invalid start line")?;
                let end: usize = parts[1].parse().map_err(|_| "Invalid end line")?;

                let lines: Vec<&str> = content.lines().collect();
                let mut result = String::new();

                for line in &lines[..start - 1] {
                    result.push_str(line);
                    result.push('\n');
                }
                result.push_str(&edit.new_content);
                if !edit.new_content.ends_with('\n') {
                    result.push('\n');
                }
                for line in &lines[end..] {
                    result.push_str(line);
                    result.push('\n');
                }

                Ok(result)
            }
        }
    }

    /// Validate the transaction against preconditions.
    pub async fn validate(&mut self) -> ValidationResult {
        self.state = TransactionState::Validating;

        // Apply shadow edits
        if let Err(e) = self.apply_shadow_edits() {
            return ValidationResult::failed(&e);
        }

        let mut result = ValidationResult::success(self.shadow_files.len() + self.new_files.len());

        // Create shadow directory for compilation
        let shadow_dir = match self.create_shadow_directory() {
            Ok(dir) => dir,
            Err(e) => {
                return ValidationResult::failed(&format!("Failed to create shadow dir: {}", e));
            }
        };

        self.shadow_dir = Some(shadow_dir.clone());

        // Check compilation if not skipped
        if !self.skip_compile {
            let compile_result = self.check_compilation(&shadow_dir);
            if !compile_result.0 {
                result.passed = false;
                result.compilation_errors = compile_result.1;
            }
        }

        // Check other preconditions
        for precondition in &self.preconditions {
            let precond_result = self.check_precondition(precondition, &shadow_dir);
            if !precond_result.passed {
                result.passed = false;
            }
            result.precondition_results.push(precond_result);
        }

        if result.passed {
            self.state = TransactionState::Ready;
        } else {
            self.state = TransactionState::Building; // Allow more edits
        }

        self.last_validation = Some(result.clone());
        result
    }

    /// Create a shadow directory with the modified files.
    fn create_shadow_directory(&self) -> Result<PathBuf, String> {
        // Create temp directory
        let shadow_dir = std::env::temp_dir().join(format!("palace-shadow-{}", self.name));

        if shadow_dir.exists() {
            std::fs::remove_dir_all(&shadow_dir)
                .map_err(|e| format!("Failed to clean shadow dir: {}", e))?;
        }

        // Copy the project structure
        self.copy_project_to_shadow(&shadow_dir)?;

        // Apply shadow edits
        for (path, content) in &self.shadow_files {
            let shadow_path = self.shadow_path(path, &shadow_dir)?;
            if let Some(parent) = shadow_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create dir: {}", e))?;
            }
            std::fs::write(&shadow_path, content)
                .map_err(|e| format!("Failed to write shadow file: {}", e))?;
        }

        // Add new files
        for (path, content) in &self.new_files {
            let shadow_path = self.shadow_path(path, &shadow_dir)?;
            if let Some(parent) = shadow_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create dir: {}", e))?;
            }
            std::fs::write(&shadow_path, content)
                .map_err(|e| format!("Failed to write new file: {}", e))?;
        }

        // Delete files
        for path in &self.delete_files {
            let shadow_path = self.shadow_path(path, &shadow_dir)?;
            if shadow_path.exists() {
                std::fs::remove_file(&shadow_path)
                    .map_err(|e| format!("Failed to delete file: {}", e))?;
            }
        }

        Ok(shadow_dir)
    }

    /// Copy project to shadow directory (minimal copy for affected crates).
    fn copy_project_to_shadow(&self, shadow_dir: &Path) -> Result<(), String> {
        // For Rust projects, copy Cargo.toml and affected source files
        // This is a simplified version - real impl would be smarter

        let copy_files = ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"];

        for file in copy_files {
            let src = self.project_root.join(file);
            if src.exists() {
                let dest = shadow_dir.join(file);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::copy(&src, &dest).ok();
            }
        }

        // Copy src directories for affected files
        for path in self.shadow_files.keys() {
            if let Ok(rel) = path.strip_prefix(&self.project_root) {
                let dest = shadow_dir.join(rel);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                // Original file will be overwritten by shadow content
            }
        }

        // Copy entire src and crates directories (for now)
        for dir in ["src", "crates"] {
            let src_dir = self.project_root.join(dir);
            if src_dir.exists() {
                self.copy_dir_recursive(&src_dir, &shadow_dir.join(dir))?;
            }
        }

        Ok(())
    }

    /// Recursively copy a directory.
    fn copy_dir_recursive(&self, src: &Path, dest: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dest).map_err(|e| format!("Failed to create {:?}: {}", dest, e))?;

        for entry in
            std::fs::read_dir(src).map_err(|e| format!("Failed to read {:?}: {}", src, e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            let dest_path = dest.join(entry.file_name());

            if path.is_dir() {
                // Skip target directory
                if path.file_name().map(|n| n == "target").unwrap_or(false) {
                    continue;
                }
                self.copy_dir_recursive(&path, &dest_path)?;
            } else {
                std::fs::copy(&path, &dest_path)
                    .map_err(|e| format!("Failed to copy {:?}: {}", path, e))?;
            }
        }

        Ok(())
    }

    /// Get the shadow path for a file.
    fn shadow_path(&self, path: &Path, shadow_dir: &Path) -> Result<PathBuf, String> {
        let rel = path
            .strip_prefix(&self.project_root)
            .map_err(|_| format!("Path {:?} not in project root", path))?;
        Ok(shadow_dir.join(rel))
    }

    /// Check if the shadow project compiles.
    fn check_compilation(&self, shadow_dir: &Path) -> (bool, Vec<String>) {
        let output = Command::new("cargo")
            .arg("check")
            .arg("--message-format=short")
            .current_dir(shadow_dir)
            .output();

        match output {
            Ok(output) => {
                let success = output.status.success();
                let stderr = String::from_utf8_lossy(&output.stderr);

                if success {
                    (true, vec![])
                } else {
                    let errors: Vec<String> = stderr
                        .lines()
                        .filter(|l| l.contains("error"))
                        .map(|l| l.to_string())
                        .collect();
                    (false, errors)
                }
            }
            Err(e) => (false, vec![format!("Failed to run cargo check: {}", e)]),
        }
    }

    /// Check a single precondition.
    fn check_precondition(
        &self,
        precondition: &Precondition,
        shadow_dir: &Path,
    ) -> PreconditionResult {
        match precondition {
            Precondition::Compiles { crates } => {
                let args = if let Some(crates) = crates {
                    crates
                        .iter()
                        .flat_map(|c| vec!["-p".to_string(), c.clone()])
                        .collect()
                } else {
                    vec![]
                };

                let mut cmd = Command::new("cargo");
                cmd.arg("check").current_dir(shadow_dir);
                for arg in args {
                    cmd.arg(arg);
                }

                match cmd.output() {
                    Ok(output) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: output.status.success(),
                        message: if output.status.success() {
                            "Compilation succeeded".to_string()
                        } else {
                            "Compilation failed".to_string()
                        },
                        output: Some(String::from_utf8_lossy(&output.stderr).to_string()),
                    },
                    Err(e) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: false,
                        message: format!("Failed to run cargo: {}", e),
                        output: None,
                    },
                }
            }

            Precondition::TestsPass { patterns } => {
                let mut all_passed = true;
                let mut output_lines = vec![];

                for pattern in patterns {
                    let output = Command::new("cargo")
                        .arg("test")
                        .arg(pattern)
                        .arg("--")
                        .arg("--test-threads=1")
                        .current_dir(shadow_dir)
                        .output();

                    match output {
                        Ok(output) => {
                            if !output.status.success() {
                                all_passed = false;
                            }
                            output_lines.push(String::from_utf8_lossy(&output.stdout).to_string());
                        }
                        Err(e) => {
                            all_passed = false;
                            output_lines.push(format!("Failed to run tests: {}", e));
                        }
                    }
                }

                PreconditionResult {
                    precondition: precondition.clone(),
                    passed: all_passed,
                    message: if all_passed {
                        "All specified tests passed".to_string()
                    } else {
                        "Some tests failed".to_string()
                    },
                    output: Some(output_lines.join("\n")),
                }
            }

            Precondition::AllTestsPass => {
                let output = Command::new("cargo")
                    .arg("test")
                    .current_dir(shadow_dir)
                    .output();

                match output {
                    Ok(output) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: output.status.success(),
                        message: if output.status.success() {
                            "All tests passed".to_string()
                        } else {
                            "Some tests failed".to_string()
                        },
                        output: Some(String::from_utf8_lossy(&output.stdout).to_string()),
                    },
                    Err(e) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: false,
                        message: format!("Failed to run tests: {}", e),
                        output: None,
                    },
                }
            }

            Precondition::CustomCheck {
                command,
                description,
            } => {
                let output = Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .current_dir(shadow_dir)
                    .output();

                match output {
                    Ok(output) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: output.status.success(),
                        message: description.clone(),
                        output: Some(format!(
                            "{}\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        )),
                    },
                    Err(e) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: false,
                        message: format!("Failed to run custom check: {}", e),
                        output: None,
                    },
                }
            }

            Precondition::FileContains { path, content } => {
                let full_path = shadow_dir.join(path);
                match std::fs::read_to_string(&full_path) {
                    Ok(file_content) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: file_content.contains(content),
                        message: format!("File {} contains check", path),
                        output: None,
                    },
                    Err(e) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: false,
                        message: format!("Failed to read {}: {}", path, e),
                        output: None,
                    },
                }
            }

            Precondition::FileNotContains { path, content } => {
                let full_path = shadow_dir.join(path);
                match std::fs::read_to_string(&full_path) {
                    Ok(file_content) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: !file_content.contains(content),
                        message: format!("File {} not-contains check", path),
                        output: None,
                    },
                    Err(_) => PreconditionResult {
                        precondition: precondition.clone(),
                        passed: true, // File doesn't exist, so it doesn't contain the content
                        message: format!("File {} not found (passes not-contains)", path),
                        output: None,
                    },
                }
            }
        }
    }

    /// Commit the transaction (apply all changes).
    pub fn commit(&mut self) -> Result<(), String> {
        if self.state != TransactionState::Ready {
            return Err("Transaction not ready to commit. Run validate() first.".to_string());
        }

        // Apply shadow file changes
        for (path, content) in &self.shadow_files {
            std::fs::write(path, content)
                .map_err(|e| format!("Failed to write {:?}: {}", path, e))?;
        }

        // Create new files
        for (path, content) in &self.new_files {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            }
            std::fs::write(path, content)
                .map_err(|e| format!("Failed to write {:?}: {}", path, e))?;
        }

        // Delete files
        for path in &self.delete_files {
            if path.exists() {
                std::fs::remove_file(path)
                    .map_err(|e| format!("Failed to delete {:?}: {}", path, e))?;
            }
        }

        // Cleanup shadow directory
        if let Some(shadow_dir) = &self.shadow_dir {
            std::fs::remove_dir_all(shadow_dir).ok();
        }

        self.state = TransactionState::Committed;
        Ok(())
    }

    /// Rollback (discard all changes).
    pub fn rollback(&mut self) {
        // Cleanup shadow directory
        if let Some(shadow_dir) = &self.shadow_dir {
            std::fs::remove_dir_all(shadow_dir).ok();
        }

        self.shadow_edits.clear();
        self.shadow_files.clear();
        self.new_files.clear();
        self.delete_files.clear();
        self.state = TransactionState::RolledBack;
    }

    /// Get the last validation result.
    pub fn last_validation(&self) -> Option<&ValidationResult> {
        self.last_validation.as_ref()
    }

    /// Get current state.
    pub fn state(&self) -> TransactionState {
        self.state
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_root.join(path)
        }
    }
}

impl Drop for ChangeTransaction {
    fn drop(&mut self) {
        // Cleanup shadow directory on drop
        if let Some(shadow_dir) = &self.shadow_dir {
            std::fs::remove_dir_all(shadow_dir).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_transaction_building() {
        let dir = TempDir::new().unwrap();

        let source = r#"fn hello() {
    println!("Hello");
}
"#;
        fs::write(dir.path().join("test.rs"), source).unwrap();

        let mut tx = ChangeTransaction::new("test-tx", dir.path());

        let edit =
            StructuralEdit::replace_function("hello", "fn hello() {\n    println!(\"Hi!\");\n}");
        tx.shadow_edit("test.rs", edit).unwrap();

        assert_eq!(tx.edit_count(), 1);
        assert_eq!(tx.state(), TransactionState::Building);
    }

    #[test]
    fn test_add_new_file() {
        let dir = TempDir::new().unwrap();

        let mut tx = ChangeTransaction::new("test-tx", dir.path());

        tx.add_file("new_file.rs", "fn new() {}").unwrap();

        assert_eq!(tx.edit_count(), 1);
        assert!(
            tx.affected_files()
                .iter()
                .any(|p| p.ends_with("new_file.rs"))
        );
    }

    #[test]
    fn test_precondition_setup() {
        let dir = TempDir::new().unwrap();

        let mut tx = ChangeTransaction::new("test-tx", dir.path());

        tx.require_compiles()
            .require_tests(vec!["test_foo".to_string()])
            .require_file_contains("src/lib.rs", "pub fn");

        assert_eq!(tx.preconditions.len(), 3);
    }
}
