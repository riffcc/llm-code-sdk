//! Benchmark system for measuring SmartRead "understanding".
//!
//! Tests whether SmartRead provides enough context to answer questions
//! about the codebase. Each benchmark has:
//! - A question about the code
//! - Known answer(s)
//! - Test whether SmartRead output contains the information to answer
//!
//! Question types:
//! 1. Signature: "What are the parameters of function X?"
//! 2. Purpose: "What does function X do?"
//! 3. Callers: "What functions call X?"
//! 4. Callees: "What does X call?"
//! 5. Complexity: "How complex is X?"
//! 6. Data flow: "What variables does X use/modify?"
//! 7. Impact: "What is affected if I change line Y?"

use std::path::Path;

use super::adaptive::{AdaptiveReader, AdaptiveResult, Granularity};

/// A benchmark question with known answer.
#[derive(Debug, Clone)]
pub struct BenchmarkQuestion {
    /// The question type
    pub kind: QuestionKind,
    /// Human-readable question
    pub question: String,
    /// Target function/symbol
    pub target: String,
    /// Keywords that MUST appear in useful context
    pub required_keywords: Vec<String>,
    /// Keywords that SHOULD appear (nice to have)
    pub optional_keywords: Vec<String>,
    /// The known answer (for reference)
    pub answer: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    /// What are the parameters/return type?
    Signature,
    /// What does this function do?
    Purpose,
    /// What functions call this?
    Callers,
    /// What does this function call?
    Callees,
    /// How complex is this function?
    Complexity,
    /// What data flows through this?
    DataFlow,
    /// What is affected by changes here?
    Impact,
}

impl QuestionKind {
    /// Minimum granularity typically needed for this question type.
    pub fn minimum_granularity(&self) -> Granularity {
        match self {
            QuestionKind::Signature => Granularity::Minimal,
            QuestionKind::Purpose => Granularity::WithDocs,
            QuestionKind::Callers | QuestionKind::Callees => Granularity::WithCalls,
            QuestionKind::Complexity => Granularity::WithComplexity,
            QuestionKind::DataFlow => Granularity::WithPreview,
            QuestionKind::Impact => Granularity::WithBodies,
        }
    }
}

/// Result of evaluating a benchmark question.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub question: BenchmarkQuestion,
    pub granularity_used: Granularity,
    pub tokens_used: usize,
    pub required_found: usize,
    pub required_total: usize,
    pub optional_found: usize,
    pub optional_total: usize,
    /// Score from 0.0 to 1.0
    pub understanding_score: f64,
    /// Efficiency = understanding / tokens
    pub efficiency_score: f64,
}

impl BenchmarkResult {
    pub fn passed(&self) -> bool {
        // Pass if all required keywords found
        self.required_found == self.required_total
    }
}

/// Benchmark runner.
pub struct Benchmarker {
    reader: AdaptiveReader,
}

impl Benchmarker {
    pub fn new() -> Self {
        Self {
            reader: AdaptiveReader::with_defaults(),
        }
    }

    /// Run a single benchmark question against source code.
    pub fn evaluate(
        &self,
        source: &str,
        path: &Path,
        question: &BenchmarkQuestion,
    ) -> BenchmarkResult {
        // Read at adaptive granularity
        let result = self.reader.read_adaptive(path, source);
        self.evaluate_with_result(question, &result)
    }

    /// Evaluate at a specific granularity.
    pub fn evaluate_at_granularity(
        &self,
        source: &str,
        path: &Path,
        question: &BenchmarkQuestion,
        granularity: Granularity,
    ) -> BenchmarkResult {
        let result = self.reader.read_at_granularity(path, source, granularity);
        self.evaluate_with_result(question, &result)
    }

    fn evaluate_with_result(
        &self,
        question: &BenchmarkQuestion,
        result: &AdaptiveResult,
    ) -> BenchmarkResult {
        let content_lower = result.content.to_lowercase();

        // Check required keywords
        let required_found = question
            .required_keywords
            .iter()
            .filter(|kw| content_lower.contains(&kw.to_lowercase()))
            .count();

        // Check optional keywords
        let optional_found = question
            .optional_keywords
            .iter()
            .filter(|kw| content_lower.contains(&kw.to_lowercase()))
            .count();

        // Calculate understanding score
        // Required keywords are weighted 2x
        let required_weight = 2.0;
        let optional_weight = 1.0;

        let max_score = (question.required_keywords.len() as f64 * required_weight)
            + (question.optional_keywords.len() as f64 * optional_weight);

        let actual_score =
            (required_found as f64 * required_weight) + (optional_found as f64 * optional_weight);

        let understanding_score = if max_score > 0.0 {
            actual_score / max_score
        } else {
            1.0 // No keywords to check = assume understood
        };

        // Efficiency = understanding per 100 tokens
        let efficiency_score = if result.tokens > 0 {
            understanding_score * 100.0 / result.tokens as f64
        } else {
            0.0
        };

        BenchmarkResult {
            question: question.clone(),
            granularity_used: result.granularity,
            tokens_used: result.tokens,
            required_found,
            required_total: question.required_keywords.len(),
            optional_found,
            optional_total: question.optional_keywords.len(),
            understanding_score,
            efficiency_score,
        }
    }

    /// Find optimal granularity for a question (best understanding/tokens ratio).
    pub fn find_optimal_granularity(
        &self,
        source: &str,
        path: &Path,
        question: &BenchmarkQuestion,
    ) -> (Granularity, BenchmarkResult) {
        let granularities = [
            Granularity::Minimal,
            Granularity::WithDocs,
            Granularity::WithCalls,
            Granularity::WithComplexity,
            Granularity::WithPreview,
            Granularity::WithBodies,
        ];

        let mut best: Option<(Granularity, BenchmarkResult)> = None;

        for g in granularities {
            let result = self.evaluate_at_granularity(source, path, question, g);

            // Must pass (all required keywords found)
            if !result.passed() {
                continue;
            }

            // First passing result, or better efficiency
            match &best {
                None => best = Some((g, result)),
                Some((_, best_result)) => {
                    if result.efficiency_score > best_result.efficiency_score {
                        best = Some((g, result));
                    }
                }
            }
        }

        // If nothing passed, return highest granularity
        best.unwrap_or_else(|| {
            let result =
                self.evaluate_at_granularity(source, path, question, Granularity::WithBodies);
            (Granularity::WithBodies, result)
        })
    }

    /// Run a suite of benchmarks and return aggregate stats.
    pub fn run_suite(
        &self,
        source: &str,
        path: &Path,
        questions: &[BenchmarkQuestion],
    ) -> BenchmarkSuite {
        let results: Vec<_> = questions
            .iter()
            .map(|q| self.evaluate(source, path, q))
            .collect();

        let passed = results.iter().filter(|r| r.passed()).count();
        let total_tokens: usize = results.iter().map(|r| r.tokens_used).sum();
        let avg_understanding: f64 =
            results.iter().map(|r| r.understanding_score).sum::<f64>() / results.len() as f64;
        let avg_efficiency: f64 =
            results.iter().map(|r| r.efficiency_score).sum::<f64>() / results.len() as f64;

        BenchmarkSuite {
            results,
            passed,
            total: questions.len(),
            total_tokens,
            avg_understanding,
            avg_efficiency,
        }
    }
}

impl Default for Benchmarker {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from running a benchmark suite.
#[derive(Debug)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
    pub passed: usize,
    pub total: usize,
    pub total_tokens: usize,
    pub avg_understanding: f64,
    pub avg_efficiency: f64,
}

impl BenchmarkSuite {
    pub fn pass_rate(&self) -> f64 {
        self.passed as f64 / self.total as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "Benchmark: {}/{} passed ({:.0}%), {:.0}% avg understanding, {} tokens, {:.2} efficiency",
            self.passed,
            self.total,
            self.pass_rate() * 100.0,
            self.avg_understanding * 100.0,
            self.total_tokens,
            self.avg_efficiency
        )
    }
}

/// Create standard benchmark questions for a codebase.
pub fn create_standard_benchmarks(functions: &[&str]) -> Vec<BenchmarkQuestion> {
    let mut questions = Vec::new();

    for func in functions {
        // Signature question
        questions.push(BenchmarkQuestion {
            kind: QuestionKind::Signature,
            question: format!("What are the parameters of {}?", func),
            target: func.to_string(),
            required_keywords: vec![func.to_string(), "fn".to_string()],
            optional_keywords: vec!["->".to_string(), "pub".to_string()],
            answer: String::new(), // Fill in per-codebase
        });

        // Purpose question
        questions.push(BenchmarkQuestion {
            kind: QuestionKind::Purpose,
            question: format!("What does {} do?", func),
            target: func.to_string(),
            required_keywords: vec![func.to_string()],
            optional_keywords: vec![], // Docstring content varies
            answer: String::new(),
        });

        // Complexity question
        questions.push(BenchmarkQuestion {
            kind: QuestionKind::Complexity,
            question: format!("How complex is {}?", func),
            target: func.to_string(),
            required_keywords: vec![func.to_string()],
            optional_keywords: vec![
                "complex".to_string(),
                "simple".to_string(),
                "moderate".to_string(),
            ],
            answer: String::new(),
        });
    }

    questions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_evaluation_forced() {
        // Test with forced granularity to ensure predictable results
        let source = r#"
/// Validates user input and returns sanitized output.
/// Checks for XSS and SQL injection patterns.
pub fn validate_input(input: &str, config: &ValidationConfig) -> Result<String, ValidationError> {
    if input.is_empty() {
        return Err(ValidationError::Empty);
    }

    let sanitized = sanitize(input);
    let checked = check_patterns(sanitized, &config.patterns)?;

    Ok(checked)
}

fn sanitize(input: &str) -> String {
    input.trim().to_string()
}

fn check_patterns(input: String, patterns: &[Pattern]) -> Result<String, ValidationError> {
    for pattern in patterns {
        if pattern.matches(&input) {
            return Err(ValidationError::PatternMatch);
        }
    }
    Ok(input)
}
"#;

        let question = BenchmarkQuestion {
            kind: QuestionKind::Purpose,
            question: "What does validate_input do?".to_string(),
            target: "validate_input".to_string(),
            required_keywords: vec!["validate_input".to_string()],
            optional_keywords: vec!["sanitize".to_string(), "input".to_string()],
            answer: "Validates and sanitizes user input".to_string(),
        };

        let benchmarker = Benchmarker::new();

        // Test at forced WithDocs granularity (should include docstrings)
        let result = benchmarker.evaluate_at_granularity(
            source,
            Path::new("validate.rs"),
            &question,
            Granularity::WithDocs,
        );

        assert!(
            result.passed(),
            "Should pass with required keywords at WithDocs"
        );
        assert!(
            result.understanding_score > 0.3,
            "Should have understanding score > 0.3, got {}",
            result.understanding_score
        );
    }

    #[test]
    fn test_benchmark_adaptive() {
        // Test adaptive evaluation - granularity may vary
        let source = r#"
/// Simple addition function.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;

        let question = BenchmarkQuestion {
            kind: QuestionKind::Signature,
            question: "What are the parameters of add?".to_string(),
            target: "add".to_string(),
            required_keywords: vec!["add".to_string()],
            optional_keywords: vec!["i32".to_string(), "pub".to_string()],
            answer: "a: i32, b: i32".to_string(),
        };

        let benchmarker = Benchmarker::new();
        let result = benchmarker.evaluate(source, Path::new("math.rs"), &question);

        // Should pass regardless of which granularity was auto-selected
        assert!(result.passed(), "Should find 'add' in output");

        // Understanding should be reasonable
        assert!(
            result.understanding_score > 0.0,
            "Should have positive understanding score"
        );
    }

    #[test]
    fn test_find_optimal_granularity() {
        let source = r#"
/// Simple addition function that adds two integers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtracts b from a.
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
"#;

        let question = BenchmarkQuestion {
            kind: QuestionKind::Signature,
            question: "What are the parameters of add?".to_string(),
            target: "add".to_string(),
            required_keywords: vec!["add".to_string()],
            optional_keywords: vec!["i32".to_string()],
            answer: "a: i32, b: i32".to_string(),
        };

        let benchmarker = Benchmarker::new();
        let (optimal, result) =
            benchmarker.find_optimal_granularity(source, Path::new("math.rs"), &question);

        assert!(result.passed(), "Optimal granularity should pass");

        // For a simple question, we expect a lower granularity to be optimal
        // (higher efficiency = less tokens for same understanding)
        assert!(
            optimal <= Granularity::WithComplexity,
            "Expected lower granularity for simple question, got {:?}",
            optimal
        );
    }

    #[test]
    fn test_benchmark_suite() {
        let source = r#"
/// Processes data by doubling each element.
pub fn process(data: Vec<i32>) -> Vec<i32> {
    data.iter().map(|x| x * 2).collect()
}

/// Filters data keeping only elements above threshold.
pub fn filter(data: Vec<i32>, threshold: i32) -> Vec<i32> {
    data.into_iter().filter(|x| *x > threshold).collect()
}
"#;

        // Create simpler questions that are more likely to pass
        let questions = vec![
            BenchmarkQuestion {
                kind: QuestionKind::Signature,
                question: "What functions are defined?".to_string(),
                target: "process".to_string(),
                required_keywords: vec!["process".to_string()],
                optional_keywords: vec!["filter".to_string()],
                answer: "process and filter".to_string(),
            },
            BenchmarkQuestion {
                kind: QuestionKind::Signature,
                question: "What does filter take?".to_string(),
                target: "filter".to_string(),
                required_keywords: vec!["filter".to_string()],
                optional_keywords: vec!["threshold".to_string()],
                answer: "data and threshold".to_string(),
            },
        ];

        let benchmarker = Benchmarker::new();
        let suite = benchmarker.run_suite(source, Path::new("data.rs"), &questions);

        println!("{}", suite.summary());
        assert!(suite.passed > 0, "Should pass at least one benchmark");
        assert!(
            suite.avg_understanding > 0.0,
            "Should have positive understanding"
        );
    }

    #[test]
    fn test_question_kind_granularity() {
        // Verify minimum granularity requirements make sense
        assert_eq!(
            QuestionKind::Signature.minimum_granularity(),
            Granularity::Minimal
        );
        assert_eq!(
            QuestionKind::Purpose.minimum_granularity(),
            Granularity::WithDocs
        );
        assert_eq!(
            QuestionKind::Complexity.minimum_granularity(),
            Granularity::WithComplexity
        );
    }
}
