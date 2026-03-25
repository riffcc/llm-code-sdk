//! Screen parser — feeds PTY output into alacritty_terminal for
//! full grid state, cursor tracking, and snapshot extraction.

use alacritty_terminal::event::{Event as AlacEvent, EventListener};
use alacritty_terminal::term::Config as TermConfig;
use alacritty_terminal::term::{Term, TermMode};
use alacritty_terminal::term::cell::Flags as CellFlags;
use alacritty_terminal::grid::Dimensions;
use alacritty_terminal::vte::ansi;

use super::terminal::{Cell, ScreenDiff, ScreenSnapshot};

/// Dummy event listener — alacritty_terminal requires one but we don't need events.
struct NoopListener;
impl EventListener for NoopListener {
    fn send_event(&self, _event: AlacEvent) {}
}

/// A screen buffer backed by alacritty_terminal.
pub struct Screen {
    term: Term<NoopListener>,
    width: u16,
    height: u16,
    /// Previous snapshot for diff computation.
    prev_text: Option<Vec<String>>,
    prev_cursor: Option<(u16, u16)>,
}

impl Screen {
    /// Create a new screen buffer with given dimensions.
    pub fn new(width: u16, height: u16) -> Self {
        let size = alacritty_terminal::term::test::TermSize::new(width as usize, height as usize);
        let config = TermConfig::default();
        let term = Term::new(config, &size, NoopListener);

        Self {
            term,
            width,
            height,
            prev_text: None,
            prev_cursor: None,
        }
    }

    /// Feed raw PTY output bytes into the terminal emulator.
    pub fn feed(&mut self, bytes: &[u8]) {
        let mut parser: ansi::Processor<ansi::StdSyncHandler> = ansi::Processor::new();
        parser.advance(&mut self.term, bytes);
    }

    /// Take a full screen snapshot.
    pub fn snapshot(&mut self) -> ScreenSnapshot {
        let grid = self.term.grid();
        let cols = self.width as usize;
        let rows = self.height as usize;

        let mut cells = Vec::with_capacity(rows);
        let mut text_lines = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let line = row_idx as i32;
            let mut row_cells = Vec::with_capacity(cols);
            let mut row_text = String::with_capacity(cols);

            for col_idx in 0..cols {
                let point = alacritty_terminal::index::Point::new(
                    alacritty_terminal::index::Line(line),
                    alacritty_terminal::index::Column(col_idx),
                );

                let term_cell = &grid[point];
                let ch = term_cell.c;

                let cell = Cell {
                    ch,
                    fg: color_to_string(term_cell.fg),
                    bg: color_to_string(term_cell.bg),
                    bold: term_cell.flags.contains(CellFlags::BOLD),
                    italic: term_cell.flags.contains(CellFlags::ITALIC),
                    underline: term_cell.flags.intersects(
                        CellFlags::UNDERLINE | CellFlags::DOUBLE_UNDERLINE
                        | CellFlags::UNDERCURL | CellFlags::DOTTED_UNDERLINE
                        | CellFlags::DASHED_UNDERLINE
                    ),
                };

                row_text.push(ch);
                row_cells.push(cell);
            }

            // Trim trailing whitespace from text
            let trimmed = row_text.trim_end().to_string();
            text_lines.push(trimmed);
            cells.push(row_cells);
        }

        let cursor = self.term.grid().cursor.point;
        let cursor_x = cursor.column.0 as u16;
        let cursor_y = cursor.line.0 as u16;

        let snapshot = ScreenSnapshot {
            width: self.width,
            height: self.height,
            cursor_x,
            cursor_y,
            cursor_visible: self.term.mode().contains(TermMode::SHOW_CURSOR),
            cells,
            text: text_lines.clone(),
            alternate_screen: self.term.mode().contains(TermMode::ALT_SCREEN),
            title: None,
        };

        // Store for diffing
        self.prev_text = Some(text_lines);
        self.prev_cursor = Some((cursor_x, cursor_y));

        snapshot
    }

    /// Compute a diff against the last snapshot.
    /// Only returns changed lines (whitespace-only lines are nulled out).
    pub fn diff(&mut self) -> ScreenDiff {
        let old_text = self.prev_text.clone();
        let old_cursor = self.prev_cursor;
        let current = self.snapshot(); // this updates prev_text/prev_cursor

        let changed_lines: Vec<(u16, String)> = if let Some(prev) = &old_text {
            current.text.iter().enumerate()
                .filter(|(i, line)| {
                    prev.get(*i).map(|p| p != *line).unwrap_or(true)
                })
                .filter(|(_, line)| !line.trim().is_empty()) // null out whitespace
                .map(|(i, line)| (i as u16, line.clone()))
                .collect()
        } else {
            // No previous — all non-empty lines are "changed"
            current.text.iter().enumerate()
                .filter(|(_, line)| !line.trim().is_empty())
                .map(|(i, line)| (i as u16, line.clone()))
                .collect()
        };

        let cursor = if old_cursor != Some((current.cursor_x, current.cursor_y)) {
            Some((current.cursor_x, current.cursor_y))
        } else {
            None
        };

        ScreenDiff {
            changed_cells: vec![], // Cell-level diff is expensive, skip for now
            cursor,
            changed_lines,
        }
    }

    /// Get just the visible text lines (non-empty, trimmed).
    pub fn visible_text(&mut self) -> Vec<String> {
        let snap = self.snapshot();
        snap.text.into_iter().filter(|l| !l.trim().is_empty()).collect()
    }

    /// Resize the screen buffer.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
        let size = alacritty_terminal::term::test::TermSize::new(width as usize, height as usize);
        self.term.resize(size);
        self.prev_text = None;
        self.prev_cursor = None;
    }
}

/// Convert an alacritty color to a CSS-like string.
fn color_to_string(color: alacritty_terminal::vte::ansi::Color) -> Option<String> {
    match color {
        alacritty_terminal::vte::ansi::Color::Named(named) => {
            Some(format!("{:?}", named).to_lowercase())
        }
        alacritty_terminal::vte::ansi::Color::Spec(rgb) => {
            Some(format!("#{:02x}{:02x}{:02x}", rgb.r, rgb.g, rgb.b))
        }
        alacritty_terminal::vte::ansi::Color::Indexed(idx) => {
            Some(format!("color{idx}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_feed_and_snapshot() {
        let mut screen = Screen::new(80, 24);
        screen.feed(b"Hello, World!\r\n");
        let snap = screen.snapshot();

        assert_eq!(snap.width, 80);
        assert_eq!(snap.height, 24);
        assert!(snap.text[0].contains("Hello, World!"));
    }

    #[test]
    fn ansi_colors_in_snapshot() {
        let mut screen = Screen::new(80, 24);
        screen.feed(b"\x1b[31mRED\x1b[0m NORMAL");
        let snap = screen.snapshot();

        // First 3 cells should have red foreground
        assert!(snap.cells[0][0].fg.is_some());
        assert_eq!(snap.cells[0][0].ch, 'R');
    }

    #[test]
    fn diff_only_changed_lines() {
        let mut screen = Screen::new(80, 24);
        screen.feed(b"Line 1\r\nLine 2\r\n");
        let _ = screen.snapshot(); // establish baseline

        // Feed more content that changes the screen
        screen.feed(b"Line 3 NEW CONTENT\r\n");
        let diff = screen.diff();

        // Should have at least one changed line containing the new content
        let has_new = diff.changed_lines.iter().any(|(_, t)| t.contains("Line 3"));
        assert!(has_new, "Expected new content in diff, got: {:?}", diff.changed_lines);
    }

    #[test]
    fn whitespace_nulled_in_diff() {
        let mut screen = Screen::new(80, 24);
        screen.feed(b"content\r\n");
        let _ = screen.snapshot();

        // Feed empty — no visual change
        let diff = screen.diff();
        // Empty/whitespace lines should be filtered out
        for (_, text) in &diff.changed_lines {
            assert!(!text.trim().is_empty());
        }
    }
}
