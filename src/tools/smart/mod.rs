//! Smart tools using the five-layer analysis approach.
//!
//! Layers (from compiler theory):
//! 1. AST - Abstract Syntax Tree (tree-sitter)
//! 2. Call Graph - function call relationships
//! 3. CFG - Control Flow Graph (cyclomatic complexity)
//! 4. DFG - Data Flow Graph (variable definitions, uses, def-use chains)
//! 5. PDG - Program Dependence Graph (control + data deps, program slicing)
//!
//! Key features:
//! - `ContextQuery` - BFS traversal of call graph for 95-99% token savings
//! - `AdaptiveReader` - Auto-adjusts granularity to meet minimum understanding
//! - `Benchmarker` - Tests whether output provides enough context to answer questions
//! - `PdgBuilder` - Program slicing: backward_slice (what affects X) and forward_slice (what X affects)

mod read_tracker;
mod adaptive;
mod ask_code;
pub(crate) mod ast;
mod benchmark;
pub(crate) mod call_graph;
mod cfg;
mod complexity;
mod context;
mod dfg;
mod examples;
mod examples_tool;
mod lean_graph;
pub(crate) mod layers;
mod mr_search;
mod pdg;
mod query;
mod smart_read;
mod smart_write;
mod transaction;

pub use adaptive::{AdaptiveConfig, AdaptiveReader, AdaptiveResult, Granularity};
pub use ask_code::AskCodeTool;
pub use ast::{AstNode, AstParser, FunctionSignature, Lang, NodeMatch, Symbol, SymbolKind};
pub use benchmark::{
    BenchmarkQuestion, BenchmarkResult, BenchmarkSuite, Benchmarker, QuestionKind,
    create_standard_benchmarks,
};
pub use call_graph::{CallGraph, CallSite};
pub use cfg::{BranchKind, CfgAnalyzer, CfgInfo};
pub use complexity::{
    ComplexityAnalysis, EditComplexity, EditComplexityAnalyzer, EditSplit, SubEdit,
};
pub use context::{ContextQuery, FunctionContext, RelevantContext};
pub use dfg::{DataflowEdge, DfgAnalyzer, DfgInfo, RefType, VarRef};
pub use examples::{CodeExample, CodeExamples, PatternKind};
pub use examples_tool::ExamplesTool;
pub use lean_graph::{LeanDecl, LeanDeclGraph, LeanDeclKind, LeanDepEdge, LeanGraphAnalyzer};
pub use layers::{CodeLayer, LayerAnalyzer, LayerView};
pub use mr_search::MRSearchTool;
pub use pdg::{DependenceType, PdgBuilder, PdgEdge, PdgInfo, PdgNode, PdgNodeType};
pub use query::{CodeQuery, QueryMetadata, QueryResult};
pub use read_tracker::ReadTracker;
pub use smart_read::{ReadRequest, SmartReadTool};
pub use smart_write::{CoordinationHook, EditGranularity, SmartWriteTool, StructuralEdit};
pub use transaction::{
    ChangeTransaction, Precondition, PreconditionResult, ShadowEdit, TransactionState,
    ValidationResult,
};
