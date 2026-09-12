//! Source-only command line with independently testable arguments and output.

use std::{ffi::OsString, io::Write, path::PathBuf, process::ExitCode};

/// Contextual command-line failures; the library uses the same textual error contract.
type Result<T> = std::result::Result<T, String>;

/// Supported command syntax, printed verbatim for help and invocation failures.
const USAGE: &str = "usage: transformers-coverage-layout [--documentation | --standards] [--] ROOT ENTRY [ENTRY ...]";

/// The versioned source-only report requested by the caller.
#[derive(Debug, Eq, PartialEq)]
enum Mode {
    /// Conservative production/test source classification.
    Layout,
    /// Legacy production-function and module documentation policy.
    Documentation,
    /// Comprehensive authored declaration and test documentation policy.
    Standards,
}

/// Validated explicit source graph; no filesystem reads occur during argument parsing.
#[derive(Debug, Eq, PartialEq)]
struct Invocation {
    /// Report semantics selected by the leading optional flag.
    mode: Mode,
    /// Source containment root, retaining native operating-system path bytes.
    root: PathBuf,
    /// At least one explicitly supplied crate entry, in caller order.
    entries: Vec<PathBuf>,
}

/// Parses one mode and an explicit source graph, or returns `None` for help.
fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<Option<Invocation>> {
    let mut arguments = arguments.into_iter();
    let first = arguments.next().ok_or(USAGE)?;
    if first == "--help" || first == "-h" {
        if arguments.next().is_some() {
            return Err(format!("help takes no source arguments; {USAGE}"));
        }
        return Ok(None);
    }
    let (mode, root) = if first == "--documentation" || first == "--standards" {
        let mode = if first == "--documentation" {
            Mode::Documentation
        } else {
            Mode::Standards
        };
        (mode, arguments.next().ok_or(USAGE)?)
    } else {
        (Mode::Layout, first)
    };
    let root = if root == "--" {
        arguments.next().ok_or(USAGE)?
    } else if root.to_str().is_some_and(|value| value.starts_with('-')) {
        return Err(format!(
            "unknown option {}; {USAGE}",
            root.to_string_lossy()
        ));
    } else {
        root
    };
    let entries: Vec<PathBuf> = arguments.map(PathBuf::from).collect();
    if entries.is_empty() {
        return Err(format!("at least one crate entry is required; {USAGE}"));
    }
    Ok(Some(Invocation {
        mode,
        root: root.into(),
        entries,
    }))
}

/// Writes exactly one pretty JSON value followed by a newline, preserving I/O context.
fn write_report(output: &mut impl Write, report: &impl serde::Serialize) -> Result<()> {
    serde_json::to_writer_pretty(&mut *output, report)
        .map_err(|error| format!("write JSON: {error}"))?;
    writeln!(output).map_err(|error| format!("write final newline: {error}"))
}

/// Produces a report or help text through injected arguments and an output writer.
fn run(arguments: impl IntoIterator<Item = OsString>, output: &mut impl Write) -> Result<()> {
    let Some(invocation) = parse_arguments(arguments)? else {
        return writeln!(output, "{USAGE}").map_err(|error| format!("write help: {error}"));
    };
    match invocation.mode {
        Mode::Layout => write_report(
            output,
            &transformers_coverage_layout::classify(&invocation.root, &invocation.entries)?,
        ),
        Mode::Documentation => write_report(
            output,
            &transformers_coverage_layout::audit_documentation(
                &invocation.root,
                &invocation.entries,
            )?,
        ),
        Mode::Standards => write_report(
            output,
            &transformers_coverage_layout::audit_standards(&invocation.root, &invocation.entries)?,
        ),
    }
}

/// Returns a nonzero exit for invocation/read/write errors, never a coverage verdict.
fn main() -> ExitCode {
    let stdout = std::io::stdout();
    match run(std::env::args_os().skip(1), &mut stdout.lock()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("coverage-layout: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
