# llm-code-sdk

See [docs/architecture/LLM-CODE-SDK.md](../../docs/architecture/LLM-CODE-SDK.md) for full documentation.

## Quick Reference

This is the autonomous coding agent SDK.
It provides everything needed to build agents that can explore, edit, and work on code.

### Key Exports

```rust
use llm_code_sdk::{
    Client,                    // Multi-backend LLM client
    ToolRunner,                // Agentic loop executor
    create_editing_tools,      // Full tool set (read, write, edit, glob, grep, bash)
    create_exploration_tools,  // Read-only tool set
    Tool,                      // Trait for custom tools
    FunctionTool,              // Quick tool from closure
};
```

### Standard Tools Location

All standard tools are in `src/tools/standard.rs`:
- BashTool
- ReadFileTool
- WriteFileTool
- EditFileTool
- GlobTool
- GrepTool
- ListDirectoryTool

**DO NOT** duplicate these elsewhere.

### Adding Tools

1. Add to `src/tools/` in this crate
2. Export from `src/tools/mod.rs`
3. Document in `docs/tools/TOOLS.md`
