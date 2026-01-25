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

mod adaptive;
mod ast;
mod benchmark;
mod call_graph;
mod cfg;
mod complexity;
mod context;
mod dfg;
mod examples;
mod examples_tool;
mod layers;
mod pdg;
mod query;
mod smart_read;
mod smart_write;
mod transaction;

pub use adaptive::{AdaptiveConfig, AdaptiveReader, AdaptiveResult, Granularity};
pub use ast::{AstNode, AstParser, FunctionSignature, Lang, Symbol, SymbolKind};
pub use benchmark::{
    create_standard_benchmarks, BenchmarkQuestion, BenchmarkResult, BenchmarkSuite,
    Benchmarker, QuestionKind,
};
pub use call_graph::{CallGraph, CallSite};
pub use cfg::{BranchKind, CfgAnalyzer, CfgInfo};
pub use context::{ContextQuery, FunctionContext, RelevantContext};
pub use dfg::{DataflowEdge, DfgAnalyzer, DfgInfo, RefType, VarRef};
pub use layers::{CodeLayer, LayerAnalyzer, LayerView};
pub use pdg::{DependenceType, PdgBuilder, PdgEdge, PdgInfo, PdgNode, PdgNodeType};
pub use query::{CodeQuery, QueryMetadata, QueryResult};
pub use smart_read::{ReadRequest, SmartReadTool};
pub use complexity::{ComplexityAnalysis, EditComplexity, EditComplexityAnalyzer, EditSplit, SubEdit};
pub use smart_write::{EditGranularity, SmartWriteTool, StructuralEdit};
pub use transaction::{
    ChangeTransaction, Precondition, PreconditionResult, ShadowEdit,
    TransactionState, ValidationResult,
};
pub use examples::{CodeExample, CodeExamples, PatternKind};
pub use examples_tool::ExamplesTool;
