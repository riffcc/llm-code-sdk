//! Read tracker — enforces read-before-write with section-level dirty tracking.
//!
//! Shared between SmartRead and SmartWrite.
//! SmartRead records every file it reads.
//! SmartWrite rejects writes to files not yet read, or to sections that
//! overlap with previously modified sections (unless re-read since).
//!
//! Section tracking allows non-overlapping edits to the same file without
//! requiring a full re-read between each write. This directly enables the
//! "range lock" pattern from the Board coordination design.
//!
//! Uses `notify` (kqueue/inotify/ReadDirectoryChanges) for external change
//! detection, with mtime as a weak fallback.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// A line range within a file (1-indexed, inclusive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
}

impl LineRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Check if two ranges overlap.
    pub fn overlaps(&self, other: &LineRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// Per-file tracking state.
#[derive(Debug, Clone)]
struct FileState {
    /// mtime when we last read or wrote the file (weak fallback).
    mtime: SystemTime,
    /// Set to true by the fs watcher when external changes are detected.
    /// External dirty = full re-read required (something outside us changed it).
    externally_dirty: bool,
    /// Sections we've written to since the last read.
    /// Non-overlapping writes are allowed; overlapping writes require re-read.
    dirty_sections: Vec<LineRange>,
    /// How many lines the file had when last read. Used to detect whether
    /// a write shifted lines enough that section tracking is unreliable.
    lines_at_read: usize,
    /// Running delta: how many lines have been added/removed by our writes
    /// since the last read. Used to shift dirty sections for subsequent checks.
    line_delta: isize,
}

/// Tracks which files have been read in this session and which sections
/// have been modified. Allows non-overlapping writes without re-read.
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
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {}
                    _ => return,
                }
                let mut map = watcher_files.write().unwrap();
                for path in &event.paths {
                    let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                    if let Some(state) = map.get_mut(&canonical) {
                        state.externally_dirty = true;
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

    /// Record that a file has been read. Clears all dirty sections.
    pub fn mark_read(&self, path: &PathBuf) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mtime = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        // Count lines for section tracking
        let lines_at_read = std::fs::read_to_string(&canonical)
            .map(|s| s.lines().count())
            .unwrap_or(0);

        // Start watching this file
        if let Ok(mut watcher_guard) = self._watcher.write() {
            if let Some(ref mut w) = *watcher_guard {
                let _ = w.watch(&canonical, RecursiveMode::NonRecursive);
            }
        }

        self.files.write().unwrap().insert(canonical, FileState {
            mtime,
            externally_dirty: false,
            dirty_sections: Vec::new(),
            lines_at_read,
            line_delta: 0,
        });
    }

    /// Record that we wrote to a specific section of a file.
    /// Tracks which lines were affected so non-overlapping writes can proceed.
    pub fn mark_section_written(&self, path: &PathBuf, range: LineRange, new_line_count: usize) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mtime = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::now());

        let mut files = self.files.write().unwrap();
        if let Some(state) = files.get_mut(&canonical) {
            state.mtime = mtime;
            // Track line delta from this edit
            let old_lines = (range.end - range.start + 1) as isize;
            let new_lines = new_line_count as isize;
            state.line_delta += new_lines - old_lines;
            state.dirty_sections.push(range);
        }
    }

    /// Record a full-file write (overwrite). Marks entire file dirty.
    pub fn mark_written(&self, path: &PathBuf) {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        let mtime = std::fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::now());

        let mut files = self.files.write().unwrap();
        if let Some(state) = files.get_mut(&canonical) {
            state.mtime = mtime;
            // Full-file write: dirty the entire range
            state.dirty_sections.push(LineRange::new(1, usize::MAX));
        } else {
            // File wasn't read before (new file) — just track mtime
            files.insert(canonical, FileState {
                mtime,
                externally_dirty: false,
                dirty_sections: vec![LineRange::new(1, usize::MAX)],
                lines_at_read: 0,
                line_delta: 0,
            });
        }
    }

    /// Check if a file can be written to, optionally at a specific section.
    ///
    /// Rules:
    /// - New files (don't exist yet) are always OK.
    /// - Existing files must have been read at least once.
    /// - External modifications (fs watcher or mtime) require full re-read.
    /// - If writing to a specific section: only blocked if it overlaps a
    ///   previously modified section. Non-overlapping sections are allowed.
    /// - If no section specified (full-file write): blocked if any section is dirty.
    pub fn check_write(&self, path: &PathBuf) -> Result<(), String> {
        self.check_write_section(path, None)
    }

    /// Check if a specific section of a file can be written to.
    pub fn check_write_section(
        &self,
        path: &PathBuf,
        section: Option<&LineRange>,
    ) -> Result<(), String> {
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
                // External changes always require full re-read
                if state.externally_dirty {
                    return Err(format!(
                        "File '{}' was modified externally. Read it again before writing.",
                        path.display()
                    ));
                }

                // Fallback: mtime check
                let current_mtime = std::fs::metadata(&canonical)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                if current_mtime > state.mtime {
                    return Err(format!(
                        "File '{}' was modified since your last read. Read it again before writing.",
                        path.display()
                    ));
                }

                // No dirty sections → always OK
                if state.dirty_sections.is_empty() {
                    return Ok(());
                }

                // If no section specified, any dirty section blocks
                let target = match section {
                    Some(s) => s,
                    None => {
                        return Err(format!(
                            "File '{}' has been modified ({}). Read it again or target a non-overlapping section.",
                            path.display(),
                            Self::describe_dirty(&state.dirty_sections)
                        ));
                    }
                };

                // Check for overlap with any dirty section
                // Apply line_delta to shift the target for comparison:
                // if previous edits added/removed lines above us, our target
                // line numbers might have shifted. We compensate by expanding
                // the overlap check with a tolerance.
                let tolerance = state.line_delta.unsigned_abs();
                let expanded_target = LineRange::new(
                    target.start.saturating_sub(tolerance),
                    target.end.saturating_add(tolerance),
                );

                for dirty in &state.dirty_sections {
                    if dirty.overlaps(&expanded_target) {
                        return Err(format!(
                            "Section lines {}-{} in '{}' overlaps with a previously modified section (lines {}-{}). Read the file again before editing this region.",
                            target.start, target.end, path.display(),
                            dirty.start, dirty.end,
                        ));
                    }
                }

                // No overlap — this section is safe to write
                Ok(())
            }
        }
    }

    fn describe_dirty(sections: &[LineRange]) -> String {
        if sections.len() == 1 && sections[0].end == usize::MAX {
            "full file overwrite".to_string()
        } else {
            let descs: Vec<String> = sections.iter().map(|s| {
                if s.end == usize::MAX {
                    format!("lines {}-EOF", s.start)
                } else {
                    format!("lines {}-{}", s.start, s.end)
                }
            }).collect();
            descs.join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    fn make_temp_file(content: &str) -> NamedTempFile {
        let f = NamedTempFile::new().unwrap();
        fs::write(f.path(), content).unwrap();
        f
    }

    #[test]
    fn non_overlapping_writes_allowed() {
        let tracker = ReadTracker::new();
        let f = make_temp_file("line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n");
        let path = f.path().to_path_buf();

        tracker.mark_read(&path);

        // Write to lines 1-3
        tracker.mark_section_written(&path, LineRange::new(1, 3), 3);

        // Writing to lines 7-9 should succeed (no overlap)
        assert!(tracker.check_write_section(&path, Some(&LineRange::new(7, 9))).is_ok());

        // Writing to lines 2-4 should fail (overlaps with 1-3)
        assert!(tracker.check_write_section(&path, Some(&LineRange::new(2, 4))).is_err());
    }

    #[test]
    fn full_file_write_blocks_all_sections() {
        let tracker = ReadTracker::new();
        let f = make_temp_file("a\nb\nc\n");
        let path = f.path().to_path_buf();

        tracker.mark_read(&path);
        tracker.mark_written(&path);

        // Any section write should be blocked
        assert!(tracker.check_write_section(&path, Some(&LineRange::new(1, 1))).is_err());
        // Full-file write should also be blocked
        assert!(tracker.check_write(&path).is_err());
    }

    #[test]
    fn read_clears_dirty_sections() {
        let tracker = ReadTracker::new();
        let f = make_temp_file("a\nb\nc\n");
        let path = f.path().to_path_buf();

        tracker.mark_read(&path);
        tracker.mark_section_written(&path, LineRange::new(1, 2), 2);
        assert!(tracker.check_write_section(&path, Some(&LineRange::new(1, 2))).is_err());

        // Re-read clears everything
        tracker.mark_read(&path);
        assert!(tracker.check_write_section(&path, Some(&LineRange::new(1, 2))).is_ok());
    }

    #[test]
    fn new_file_always_allowed() {
        let tracker = ReadTracker::new();
        let path = PathBuf::from("/tmp/nonexistent_test_file_12345.rs");
        assert!(tracker.check_write(&path).is_ok());
    }

    #[test]
    fn unread_file_blocked() {
        let tracker = ReadTracker::new();
        let f = make_temp_file("content\n");
        let path = f.path().to_path_buf();
        assert!(tracker.check_write(&path).is_err());
    }
}
