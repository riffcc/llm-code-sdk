//! SmartRead token savings demo - analyzing ALL of Palace with all 5 layers
//!
//! Run with: cargo run -p llm-code-sdk --example smart_demo --features smart

use llm_code_sdk::tools::smart::{
    CodeLayer, ContextQuery, DfgAnalyzer, Lang, LayerAnalyzer, PdgBuilder, ReadRequest,
    SmartReadTool,
};
use llm_code_sdk::tools::{GlobTool, Tool};
use std::collections::HashMap;
use std::path::PathBuf;

fn count_tokens(s: &str) -> usize {
    s.split_whitespace().count()
}

#[tokio::main]
async fn main() {
    let palace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    println!(
        "\n\x1b[1;36m╔═══════════════════════════════════════════════════════════════════╗\x1b[0m"
    );
    println!(
        "\x1b[1;36m║         SmartRead: Analyzing ALL of Palace                        ║\x1b[0m"
    );
    println!(
        "\x1b[1;36m╚═══════════════════════════════════════════════════════════════════╝\x1b[0m\n"
    );

    // Use our own GlobTool to find Rust files (dogfooding!)
    let glob_tool = GlobTool::new(&palace_root);
    let mut input: HashMap<String, serde_json::Value> = HashMap::new();
    input.insert("pattern".to_string(), serde_json::json!("**/*.rs"));

    let result = glob_tool.call(input).await;
    let content = result.to_content_string();
    let rust_files: Vec<PathBuf> = content
        .lines()
        .filter(|l: &&str| !l.contains("target/") && !l.starts_with('.'))
        .map(|l: &str| palace_root.join(l))
        .collect();

    println!("  Found {} Rust files in Palace\n", rust_files.len());

    // Calculate raw totals
    let mut total_raw_chars = 0usize;
    let mut total_raw_tokens = 0usize;
    let mut total_raw_lines = 0usize;

    let mut total_ast_chars = 0usize;
    let mut total_ast_tokens = 0usize;

    let mut analyzer = LayerAnalyzer::new();
    let mut file_stats = Vec::new();

    for path in &rust_files {
        if let Ok(content) = std::fs::read_to_string(path) {
            let raw_chars = content.len();
            let raw_tokens = count_tokens(&content);
            let raw_lines = content.lines().count();

            total_raw_chars += raw_chars;
            total_raw_tokens += raw_tokens;
            total_raw_lines += raw_lines;

            // Get relative path for display
            let rel_path = path.strip_prefix(&palace_root).unwrap_or(path);
            let rel_str = rel_path.to_string_lossy().to_string();

            // Analyze at AST layer
            let ast_view = analyzer.analyze(&rel_str, &content, CodeLayer::Ast);
            let ast_content = ast_view.to_context();
            let ast_chars = ast_content.len();
            let ast_tokens = count_tokens(&ast_content);

            total_ast_chars += ast_chars;
            total_ast_tokens += ast_tokens;

            file_stats.push((rel_str, raw_chars, raw_tokens, ast_chars, ast_tokens));
        }
    }

    // Overall stats
    let char_savings =
        ((total_raw_chars - total_ast_chars) as f64 / total_raw_chars as f64) * 100.0;
    let token_savings =
        ((total_raw_tokens - total_ast_tokens) as f64 / total_raw_tokens as f64) * 100.0;

    println!("  \x1b[1;33m═══ PALACE CODEBASE TOTALS ═══\x1b[0m\n");
    println!("  ┌────────────────────────────────────────────────────────┐");
    println!("  │                    Raw          AST         Savings    │");
    println!("  ├────────────────────────────────────────────────────────┤");
    println!(
        "  │ Characters:   {:>8}     {:>8}       {:>5.1}%     │",
        total_raw_chars, total_ast_chars, char_savings
    );
    println!(
        "  │ Tokens:       {:>8}     {:>8}       {:>5.1}%     │",
        total_raw_tokens, total_ast_tokens, token_savings
    );
    println!(
        "  │ Lines:        {:>8}          -            -       │",
        total_raw_lines
    );
    println!(
        "  │ Files:        {:>8}          -            -       │",
        rust_files.len()
    );
    println!("  └────────────────────────────────────────────────────────┘");

    // Top 10 largest files
    file_stats.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n  \x1b[1;33m═══ TOP 10 LARGEST FILES ═══\x1b[0m\n");
    println!("  ┌──────────────────────────────────────────────────────────────────────┐");
    println!("  │ File                                   Raw Tok  AST Tok  Savings     │");
    println!("  ├──────────────────────────────────────────────────────────────────────┤");

    for (path, _raw_chars, raw_tok, _ast_chars, ast_tok) in file_stats.iter().take(10) {
        let short_path = if path.len() > 38 {
            format!("...{}", &path[path.len() - 35..])
        } else {
            path.clone()
        };
        let savings = ((raw_tok - ast_tok) as f64 / *raw_tok as f64) * 100.0;
        println!(
            "  │ {:<38} {:>7}  {:>7}   {:>5.1}%     │",
            short_path, raw_tok, ast_tok, savings
        );
    }
    println!("  └──────────────────────────────────────────────────────────────────────┘");

    // Batch read demo - read entire smart module with tree read
    println!("\n  \x1b[1;33m═══ BATCH READ: Smart Module (7 files) ═══\x1b[0m\n");

    let smart_dir = palace_root.join("crates/llm-code-sdk/src/tools/smart");
    let tool = SmartReadTool::new(&smart_dir);

    let requests = vec![
        ReadRequest::new("mod.rs", CodeLayer::Ast),
        ReadRequest::new("ast.rs", CodeLayer::Ast),
        ReadRequest::new("call_graph.rs", CodeLayer::Ast),
        ReadRequest::new("cfg.rs", CodeLayer::Ast),
        ReadRequest::new("context.rs", CodeLayer::Ast),
        ReadRequest::new("layers.rs", CodeLayer::Ast),
        ReadRequest::new("smart_read.rs", CodeLayer::Ast),
    ];

    let batch_result = tool.read_tree(&requests);
    let batch_tokens = count_tokens(&batch_result);

    // Calculate raw for these files
    let smart_files = [
        "mod.rs",
        "ast.rs",
        "call_graph.rs",
        "cfg.rs",
        "context.rs",
        "layers.rs",
        "smart_read.rs",
    ];
    let mut smart_raw_tokens = 0;
    for f in &smart_files {
        if let Ok(content) = std::fs::read_to_string(smart_dir.join(f)) {
            smart_raw_tokens += count_tokens(&content);
        }
    }

    let smart_savings =
        ((smart_raw_tokens - batch_tokens) as f64 / smart_raw_tokens as f64) * 100.0;

    println!("  Single tool call reading 7 files:");
    println!("  \x1b[32m┌─────────────────────────────────────────────┐\x1b[0m");
    println!(
        "  \x1b[32m│ Raw (7 files):     {:>6} tokens            │\x1b[0m",
        smart_raw_tokens
    );
    println!(
        "  \x1b[32m│ Batch AST:         {:>6} tokens            │\x1b[0m",
        batch_tokens
    );
    println!(
        "  \x1b[32m│ \x1b[1mSavings:            {:>5.1}%\x1b[0m\x1b[32m               │\x1b[0m",
        smart_savings
    );
    println!("  \x1b[32m└─────────────────────────────────────────────┘\x1b[0m");

    // Context query on Palace main
    println!("\n  \x1b[1;33m═══ CONTEXT QUERY: palace/main.rs ═══\x1b[0m\n");

    let palace_crate = palace_root.join("crates/palace/src");
    let mut ctx_query = ContextQuery::new(&palace_crate);

    // Check if main.rs exists
    if palace_crate.join("main.rs").exists() {
        if let Ok(main_content) = std::fs::read_to_string(palace_crate.join("main.rs")) {
            let main_raw_tokens = count_tokens(&main_content);

            let ctx = ctx_query.query("main", "main.rs", 2);
            let ctx_output = ctx.to_llm_string();
            let ctx_tokens = ctx.estimate_tokens();

            println!("  Query: main() at depth 2");
            println!("  \x1b[32m┌─────────────────────────────────────────────┐\x1b[0m");
            println!(
                "  \x1b[32m│ main.rs raw:       {:>6} tokens            │\x1b[0m",
                main_raw_tokens
            );
            println!(
                "  \x1b[32m│ Context query:     {:>6} tokens            │\x1b[0m",
                ctx_tokens
            );
            println!(
                "  \x1b[32m│ Functions found:   {:>6}                   │\x1b[0m",
                ctx.functions.len()
            );
            println!("  \x1b[32m└─────────────────────────────────────────────┘\x1b[0m");

            println!("\n  Context output preview:");
            for line in ctx_output.lines().take(25) {
                println!("  {}", line);
            }
            if ctx_output.lines().count() > 25 {
                println!("  ... ({} more lines)", ctx_output.lines().count() - 25);
            }
        }
    }

    // 5-Layer demo on a single file
    println!("\n  \x1b[1;33m═══ ALL 5 LAYERS: Single File Analysis ═══\x1b[0m\n");

    // Use a sample function for demo
    let sample_source = r#"
/// Processes user input and validates it.
pub fn process_input(input: &str, config: &Config) -> Result<Output, Error> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(Error::Empty);
    }
    let validated = validate(trimmed, config)?;
    let normalized = normalize(validated);
    transform(normalized)
}

fn validate(s: &str, cfg: &Config) -> Result<&str, Error> {
    if cfg.strict && s.len() < 3 {
        return Err(Error::TooShort);
    }
    Ok(s)
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
}

fn transform(s: String) -> Result<Output, Error> {
    Ok(Output { value: s })
}
"#;

    let layers = [
        ("Raw", CodeLayer::Raw),
        ("AST", CodeLayer::Ast),
        ("CFG", CodeLayer::Cfg),
        ("DFG", CodeLayer::Dfg),
        ("PDG", CodeLayer::Pdg),
    ];

    let raw_tokens = count_tokens(sample_source);
    println!("  Source: {} tokens (raw)\n", raw_tokens);

    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │ Layer   Tokens  Savings  Key Info                   │");
    println!("  ├─────────────────────────────────────────────────────┤");

    for (name, layer) in &layers {
        let view = analyzer.analyze("sample.rs", sample_source, *layer);
        let ctx = view.to_context();
        let tokens = count_tokens(&ctx);
        let savings = if *layer == CodeLayer::Raw {
            0.0
        } else {
            ((raw_tokens as f64 - tokens as f64) / raw_tokens as f64) * 100.0
        };

        let key_info = match layer {
            CodeLayer::Raw => "Full source code".to_string(),
            CodeLayer::Ast => "4 functions, types".to_string(),
            CodeLayer::CallGraph => "Who calls whom".to_string(),
            CodeLayer::Cfg => "Complexity metrics".to_string(),
            CodeLayer::Dfg => "Var defs/uses".to_string(),
            CodeLayer::Pdg => "Control+data deps".to_string(),
        };

        println!(
            "  │ {:6} {:>6}   {:>5.1}%  {:25} │",
            name,
            tokens,
            savings.max(0.0),
            key_info
        );
    }
    println!("  └─────────────────────────────────────────────────────┘");

    // Program slicing demo
    println!("\n  \x1b[1;33m═══ PROGRAM SLICING DEMO ═══\x1b[0m\n");

    let pdgs = PdgBuilder::analyze(sample_source, Lang::Rust);
    if let Some(pdg) = pdgs.first() {
        println!("  Function: {}", pdg.function_name);
        println!("  Question: \"What affects line 8 (validated = validate...)?\"");

        let backward = pdg.backward_slice(8, None);
        println!("\n  Backward slice (lines that affect line 8):");
        let sliced = pdg.slice_source(sample_source, &backward);
        for line in sliced.lines().take(10) {
            println!("  {}", line);
        }

        println!(
            "\n  \x1b[32m→ Only shows {} lines instead of full source!\x1b[0m",
            backward.len()
        );
        println!("  \x1b[32m→ Perfect for debugging: \"why is this value wrong?\"\x1b[0m");
    }

    // DFG demo
    println!("\n  \x1b[1;33m═══ DATA FLOW TRACKING ═══\x1b[0m\n");

    let dfgs = DfgAnalyzer::analyze(sample_source, Lang::Rust);
    if let Some(dfg) = dfgs.first() {
        println!("  Function: {}", dfg.function_name);
        println!("  Variables tracked: {}", dfg.variables.join(", "));
        println!("  Def-use chains: {}", dfg.edges.len());
        println!("\n  Data flows:");
        for edge in dfg.edges.iter().take(5) {
            println!(
                "    {} defined@L{} → used@L{}",
                edge.variable, edge.def_line, edge.use_line
            );
        }
    }

    // Summary
    println!("\n  \x1b[1;36m═══ SUMMARY ═══\x1b[0m\n");
    println!(
        "  SmartRead achieves \x1b[1;32m{:.1}%\x1b[0m token savings across {} Palace files.",
        token_savings,
        rust_files.len()
    );
    println!(
        "  That's {:} tokens reduced to {:} tokens.",
        total_raw_tokens, total_ast_tokens
    );
    println!(
        "\n  The LLM gets structural understanding with ~{:.0}x less context.",
        total_raw_tokens as f64 / total_ast_tokens as f64
    );

    println!("\n  \x1b[1;36m5 Layers Available:\x1b[0m");
    println!("    1. AST       - Structure (functions, types)");
    println!("    2. Call Graph - Who calls whom");
    println!("    3. CFG       - Cyclomatic complexity");
    println!("    4. DFG       - Variable definitions & uses");
    println!("    5. PDG       - Program slicing (backward/forward)\n");
}
