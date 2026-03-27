//! Read tracker — enforces read-before-write with filesystem change detection.
//!
//! Shared between SmartRead and SmartWrite.
//! SmartRead records every file it reads.
//! SmartWrite rejects writes to files not yet read, or files that
//! changed since the last read (detected via filesystem notifications).
//!
//! Uses `notify` (kqueue/inotify/ReadDirectoryChanges) for reliable
//! change detection, with mtime as a weak fallback.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// Per-file tracking state.
#[derive(Debug, Clone)]
struct FileState {
    /// mtime when we last read or wrote the file (weak fallback).
    mtime: SystemTime,
    /// Set to true by the fs watcher when the file changes on disk.
    /// Cleared when we read or write the file ourselves.
    dirty: bool,
}

/// Tracks which files have been read in this session and whether they've
/// changed since. Filesystem events mark files dirty; read/write clear them.
#[derive(Clone)]
pub struct ReadTracker {
    files: Arc<RwLock<HashMap<PathBuf, FileState>>>,
    /// Kept alive so the watcher thread continues running.
    _watcher: Arc<RwLock<Option<RecommendedWatcher>>>,
}

impl std::fmt::Debug for ReadTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadTracker")
            .field("files", &self.files)
            .finish()
    }
}

impl Default for ReadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadTracker {
    pub fn new() -> Self {
        let files: Arc<RwLock<HashMap<PathBuf, FileState>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let watcher_files = Arc::clone(&files);
        let watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                // Only care about content modifications and renames
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {}
                    _ => return,
                }
                let mut map = watcher_files.write().unwrap();
                for path in &event.paths {
                    let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                    if let Some(state) = map.get_mut(&canonical) {
                        state.dirty = true;
                    }
                }
            }
        });

        let watcher = match watcher {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!("Failed to create file watcher, falling back to mtime only: {e}");
                None
            }
        };

        Self {
            files,
            _watcher: Arc::new(RwLock::new(watcher)),
        }
    }

    /// Record that a file has been read. Starts watching it for changes.
    pub fn mark_read(&self, path: &PathBuf) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mtime = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        // Start watching this file
        if let Ok(mut watcher_guard) = self._watcher.write() {
            if let Some(ref mut w) = *watcher_guard {
                let _ = w.watch(&canonical, RecursiveMode::NonRecursive);
            }
        }

        self.files.write().unwrap().insert(canonical, FileState {
            mtime,
            dirty: false,
        });
    }

    /// Record that we just wrote a file. Clears dirty state since we
    /// know the current content.
    pub fn mark_written(&self, path: &PathBuf) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mtime = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::now());

        self.files.write().unwrap().insert(canonical, FileState {
            mtime,
            dirty: false,
        });
    }

    /// Check if a file can be written to.
    ///
    /// Rules:
    /// - New files (don't exist yet) are always OK.
    /// - Existing files must have been read.
    /// - If the file was modified since our last read (detected by fs
    ///   notification or mtime fallback), require a re-read.
    pub fn check_write(&self, path: &PathBuf) -> Result<(), String> {
        // New files are always OK
        if !path.exists() {
            return Ok(());
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let files = self.files.read().unwrap();

        match files.get(&canonical) {
            None => Err(format!(
                "You must read '{}' before writing to it. Use the read tool first.",
                path.display()
            )),
            Some(state) => {
                // Primary: fs watcher flagged it dirty
                if state.dirty {
                    return Err(format!(
                        "File '{}' was modified since your last read. Read it again before writing.",
                        path.display()
                    ));
                }

                // Fallback: mtime check (catches changes the watcher might miss,
                // e.g. if the watcher failed to start)
                let current_mtime = std::fs::metadata(&canonical)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                if current_mtime > state.mtime {
                    Err(format!(
                        "File '{}' was modified since your last read. Read it again before writing.",
                        path.display()
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }
}
