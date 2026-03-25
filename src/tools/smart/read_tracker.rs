//! Read tracker — enforces read-before-write.
//!
//! Shared between SmartRead and SmartWrite.
//! SmartRead records every file it reads.
//! SmartWrite rejects writes to files not yet read.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Tracks which files have been read in this session.
#[derive(Debug, Clone, Default)]
pub struct ReadTracker {
    read_files: Arc<RwLock<HashSet<PathBuf>>>,
}

impl ReadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a file has been read.
    pub fn mark_read(&self, path: &PathBuf) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        self.read_files.write().unwrap().insert(canonical);
    }

    /// Check if a file has been read. New files (that don't exist yet) are allowed.
    pub fn check_write(&self, path: &PathBuf) -> Result<(), String> {
        // New files are always OK
        if !path.exists() {
            return Ok(());
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if self.read_files.read().unwrap().contains(&canonical) {
            Ok(())
        } else {
            Err(format!(
                "You must read '{}' before writing to it. Use the read tool first.",
                path.display()
            ))
        }
    }
}
