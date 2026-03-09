//! Call Graph layer - function call relationships.
//!
//! Tracks which functions call which other functions.

use std::collections::{HashMap, HashSet};

/// A call site in the code.
#[derive(Debug, Clone)]
pub struct CallSite {
    pub caller: String,
    pub callee: String,
    pub line: usize,
}

/// Call graph representing function relationships.
#[derive(Debug, Clone, Default)]
pub struct CallGraph {
    /// Functions that each function calls.
    pub calls: HashMap<String, HashSet<String>>,
    /// Functions that call each function.
    pub called_by: HashMap<String, HashSet<String>>,
    /// All call sites.
    pub sites: Vec<CallSite>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a call relationship.
    pub fn add_call(&mut self, caller: &str, callee: &str, line: usize) {
        self.calls
            .entry(caller.to_string())
            .or_default()
            .insert(callee.to_string());

        self.called_by
            .entry(callee.to_string())
            .or_default()
            .insert(caller.to_string());

        self.sites.push(CallSite {
            caller: caller.to_string(),
            callee: callee.to_string(),
            line,
        });
    }

    /// Get functions called by a function.
    pub fn get_calls(&self, func: &str) -> Vec<&str> {
        self.calls
            .get(func)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get functions that call a function.
    pub fn get_callers(&self, func: &str) -> Vec<&str> {
        self.called_by
            .get(func)
            .map(|s| s.iter().map(|x| x.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get entry points (functions not called by anything).
    pub fn entry_points(&self) -> Vec<&str> {
        self.calls
            .keys()
            .filter(|f| !self.called_by.contains_key(*f) || self.called_by[*f].is_empty())
            .map(|s| s.as_str())
            .collect()
    }

    /// Get leaf functions (functions that don't call anything).
    pub fn leaf_functions(&self) -> Vec<&str> {
        self.called_by
            .keys()
            .filter(|f| !self.calls.contains_key(*f) || self.calls[*f].is_empty())
            .map(|s| s.as_str())
            .collect()
    }

    /// Get the call chain from a function (BFS).
    pub fn call_chain(&self, func: &str, max_depth: usize) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![(func.to_string(), 0usize)];

        while let Some((current, depth)) = queue.pop() {
            if depth > max_depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if depth > 0 {
                result.push((current.clone(), depth));
            }

            if let Some(callees) = self.calls.get(&current) {
                for callee in callees {
                    queue.push((callee.clone(), depth + 1));
                }
            }
        }

        result.sort_by_key(|(_, d)| *d);
        result
    }

    /// Format as a readable string.
    pub fn to_summary(&self) -> String {
        let mut output = String::new();

        let entry_points = self.entry_points();
        if !entry_points.is_empty() {
            output.push_str("### Entry Points\n");
            for ep in entry_points {
                output.push_str(&format!("- `{}`\n", ep));
            }
            output.push('\n');
        }

        output.push_str("### Call Relationships\n");
        for (caller, callees) in &self.calls {
            if !callees.is_empty() {
                let callees_str: Vec<_> = callees.iter().map(|s| format!("`{}`", s)).collect();
                output.push_str(&format!(
                    "- `{}` calls: {}\n",
                    caller,
                    callees_str.join(", ")
                ));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_graph() {
        let mut cg = CallGraph::new();
        cg.add_call("main", "init", 10);
        cg.add_call("main", "run", 11);
        cg.add_call("run", "process", 20);
        cg.add_call("run", "cleanup", 25);

        assert_eq!(cg.get_calls("main").len(), 2);
        assert!(cg.get_callers("run").contains(&"main"));
        assert!(cg.entry_points().contains(&"main"));
    }
}
