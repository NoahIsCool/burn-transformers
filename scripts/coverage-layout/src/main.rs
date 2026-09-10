//! Minimal source-only command line for the supplementary classifier.

use std::{io::Write, path::PathBuf, process::ExitCode};

/// Runs the classifier and writes one versioned JSON report to standard output.
fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let first = args
        .next()
        .ok_or("usage: transformers-coverage-layout ROOT ENTRY [ENTRY ...]")?;
    let documentation = first == "--documentation";
    let root = if documentation {
        args.next()
            .ok_or("usage: transformers-coverage-layout --documentation ROOT ENTRY [ENTRY ...]")?
    } else {
        first
    };
    let entries: Vec<PathBuf> = args.map(PathBuf::from).collect();
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    if documentation {
        let report =
            transformers_coverage_layout::audit_documentation(&PathBuf::from(root), &entries)?;
        serde_json::to_writer_pretty(&mut output, &report)
            .map_err(|error| format!("write JSON: {error}"))?;
    } else {
        let report = transformers_coverage_layout::classify(&PathBuf::from(root), &entries)?;
        serde_json::to_writer_pretty(&mut output, &report)
            .map_err(|error| format!("write JSON: {error}"))?;
    }
    writeln!(output).map_err(|error| format!("write final newline: {error}"))?;
    Ok(())
}

/// Returns a nonzero exit for invocation/read errors, never a coverage verdict.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("coverage-layout: {error}");
            ExitCode::from(2)
        }
    }
}
