//! Tool system for agentic interactions.
//!
//! This module provides:
//! - [`Tool`] trait for defining tools
//! - [`ToolRegistry`] for central tool registration (THE one place for tools)
//! - [`FunctionTool`] for wrapping functions as tools
//! - [`ToolRunner`] for running agentic loops with automatic tool execution
//! - Standard tools for file operations, search, and shell execution

mod function_tool;
mod registry;
mod runner;
mod standard;
mod traits;

#[cfg(feature = "search")]
mod search;

#[cfg(feature = "smart")]
pub mod smart;

pub use function_tool::FunctionTool;
pub use registry::{ToolRegistry, create_editing_registry, create_exploration_registry};
pub use runner::{ToolEvent, ToolEventCallback, ToolRunner, ToolRunnerConfig};
pub use standard::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, WriteFileTool,
    create_editing_tools, create_exploration_tools,
};
pub use traits::{Tool, ToolResult, ToolResultContent};

#[cfg(feature = "search")]
pub use search::{CodeDocument, SearchResult, SearchTool};

#[cfg(feature = "smart")]
pub use smart::{
    AskCodeTool, AstNode, AstParser, CallGraph, CodeLayer, FunctionSignature, LayerView,
    LeanDecl, LeanDeclGraph, LeanDeclKind, LeanDepEdge, LeanGraphAnalyzer, MRSearchTool,
    SmartReadTool, Symbol, SymbolKind,
};
