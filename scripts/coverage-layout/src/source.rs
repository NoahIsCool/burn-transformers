//! Exact source-buffer coordinates and conservative physical-line ownership.
//!
//! Byte ranges are zero-based and half-open. Lines and columns are one-based;
//! `column` counts UTF-8 bytes (matching LLVM source columns), while
//! `unicode_column` counts Unicode scalar values. SHA-256 covers the original
//! bytes, including BOM, CRLF, comments, and any shebang removed by Syn.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// Versioned supplementary source-layout report, never a coverage decision.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Schema identifier independent of runtime library releases.
    pub schema_version: u32,
    /// Explicitly identifies the absence of gate integration.
    pub purpose: &'static str,
    /// Always false: parsed candidates alone do not prove compiler reachability.
    pub qualified_for_coverage_exclusions: bool,
    /// Always false: macros may introduce unobserved source references.
    pub macro_expansion_resolved: bool,
    /// Canonical containment root used for every source read.
    pub root: String,
    /// Entry files relative to the containment root.
    pub entries: Vec<String>,
    /// Deterministically ordered physical files and all observed contexts.
    pub files: Vec<FileReport>,
    /// Unresolved, ambiguous, duplicate, or opaque constructs.
    pub diagnostics: Vec<Diagnostic>,
}

/// Source ownership of a complete physical line.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    /// Some production or unresolved source owns the line.
    Retained,
    /// Every observed parsed context marks the line test-only; a candidate only.
    TestOnly,
    /// Test-only and retained fragments or reference contexts share this line.
    Mixed,
    /// A blank line has no non-whitespace source to classify.
    Blank,
}

/// An exact byte offset and its human-readable source coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct Position {
    /// Zero-based byte offset in the original source buffer.
    pub byte: usize,
    /// One-based physical line number.
    pub line: usize,
    /// One-based UTF-8 byte column, not a Unicode scalar index.
    pub column: usize,
    /// One-based Unicode scalar column for human inspection.
    pub unicode_column: usize,
}

/// A half-open source span with both byte and line/column coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct SourceSpan {
    /// Inclusive starting position.
    pub start: Position,
    /// Exclusive ending position, possibly at end-of-file.
    pub end: Position,
}

/// One syntactic item, or the whole-file fallback, in a reference context.
#[derive(Clone, Debug, Serialize)]
pub struct Region {
    /// Source coordinates including parsed outer attributes.
    pub span: SourceSpan,
    /// Nesting depth; deeper syntax owns its bytes over enclosing syntax.
    pub depth: usize,
    /// AST node kind; no macro expansion or generated-node assumptions.
    pub kind: String,
    /// Whether this region alone has a cfg-based test-only proof.
    pub test_only: bool,
    /// Human-readable proof or reason for retaining source.
    pub reason: String,
}

/// One module-tree reference to a physical file.
#[derive(Debug, Serialize)]
pub struct Context {
    /// Logical module path reached from the supplied entry.
    pub module: String,
    /// Whether an equivalent physical-path/reachability context was seen before.
    pub duplicate_context: bool,
    /// Per-context syntax, later merged with every other context for this file.
    pub regions: Vec<Region>,
}

/// Inclusive physical-line run sharing one classification.
#[derive(Debug, Serialize)]
pub struct LineRun {
    /// First one-based physical line in the run.
    pub start_line: usize,
    /// Last one-based physical line in the run.
    pub end_line: usize,
    /// Conservative classification of each line in this run.
    pub classification: Classification,
}

/// One authenticated physical source buffer and its merged ownership.
#[derive(Debug, Serialize)]
pub struct FileReport {
    /// Canonical file path relative to the containment root.
    pub path: String,
    /// SHA-256 of the exact bytes parsed, not normalized text.
    pub sha256: String,
    /// Size in bytes of the original source snapshot.
    pub byte_length: usize,
    /// Physical source lines, excluding an empty virtual line after final LF.
    pub line_count: usize,
    /// All observed module references, not merely the first visit.
    pub contexts: Vec<Context>,
    /// Complete physical-line classification; no lines are silently dropped.
    pub lines: Vec<LineRun>,
    /// Versioned structural test audit; opaque macro expansion is not inspected.
    pub test_audit: crate::TestAudit,
}

/// A source issue that must remain visible to a human coverage auditor.
#[derive(Debug, Serialize)]
pub struct Diagnostic {
    /// Stable machine-readable diagnostic category.
    pub code: String,
    /// Canonical source path relative to the root when available.
    pub file: String,
    /// Exact source coordinates when the construct was parsed successfully.
    pub span: Option<SourceSpan>,
    /// Actionable explanation without claiming a gate result.
    pub reason: String,
}

/// Immutable text plus line starts for lossless byte-coordinate conversion.
pub(crate) struct Source {
    /// Exact source text read once for all physical-file contexts.
    pub(crate) text: String,
    /// Byte offset to the beginning of each line, including a virtual EOF line.
    starts: Vec<usize>,
}

impl Source {
    /// Creates a coordinate index without changing the source buffer.
    pub(crate) fn new(text: String) -> Self {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(index, _)| index + 1));
        Self { text, starts }
    }

    /// Hashes exactly the original source bytes.
    pub(crate) fn sha256(&self) -> String {
        format!("{:x}", Sha256::digest(self.text.as_bytes()))
    }

    /// Counts real physical lines, including a non-LF-terminated final line.
    pub(crate) fn line_count(&self) -> usize {
        if self.text.is_empty() {
            0
        } else {
            self.starts.len() - usize::from(self.text.ends_with('\n'))
        }
    }

    /// Converts an already-validated byte boundary into both column units.
    fn position(&self, byte: usize) -> Position {
        let line_index = self.starts.partition_point(|start| *start <= byte) - 1;
        let start = self.starts[line_index];
        Position {
            byte,
            line: line_index + 1,
            column: byte - start + 1,
            unicode_column: self.text[start..byte].chars().count() + 1,
        }
    }

    /// Creates a span only if parser offsets address valid original boundaries.
    pub(crate) fn span(&self, start: usize, end: usize) -> Option<SourceSpan> {
        (start <= end
            && end <= self.text.len()
            && self.text.is_char_boundary(start)
            && self.text.is_char_boundary(end))
        .then(|| SourceSpan {
            start: self.position(start),
            end: self.position(end),
        })
    }

    /// Gets whole-buffer coordinates, which are valid even for empty files.
    pub(crate) fn whole_span(&self) -> SourceSpan {
        SourceSpan {
            start: self.position(0),
            end: self.position(self.text.len()),
        }
    }

    /// Merges deepest syntax ownership across every reference context.
    pub(crate) fn classify_lines(&self, contexts: &[Context]) -> Vec<LineRun> {
        let mut runs: Vec<LineRun> = Vec::new();
        for index in 0..self.line_count() {
            let start = self.starts[index];
            let end = self
                .starts
                .get(index + 1)
                .copied()
                .unwrap_or(self.text.len());
            let mut retained = false;
            let mut test = false;
            for context in contexts {
                let mut boundaries = vec![start, end];
                for region in &context.regions {
                    if region.span.start.byte < end && region.span.end.byte > start {
                        boundaries.push(region.span.start.byte.max(start));
                        boundaries.push(region.span.end.byte.min(end));
                    }
                }
                boundaries.sort_unstable();
                boundaries.dedup();
                for pair in boundaries.windows(2) {
                    if self.text[pair[0]..pair[1]].chars().all(char::is_whitespace) {
                        continue;
                    }
                    let owner = context
                        .regions
                        .iter()
                        .filter(|region| {
                            region.span.start.byte <= pair[0] && region.span.end.byte >= pair[1]
                        })
                        .max_by_key(|region| region.depth);
                    if owner.is_some_and(|region| region.test_only) {
                        test = true;
                    } else {
                        retained = true;
                    }
                }
            }
            // A parse failure still has a retained whole-file fallback context.
            let classification = match (retained, test) {
                (true, true) => Classification::Mixed,
                (true, false) => Classification::Retained,
                (false, true) => Classification::TestOnly,
                (false, false) => Classification::Blank,
            };
            if let Some(previous) = runs.last_mut()
                && previous.classification == classification
            {
                previous.end_line = index + 1;
            } else {
                runs.push(LineRun {
                    start_line: index + 1,
                    end_line: index + 1,
                    classification,
                });
            }
        }
        runs
    }
}
