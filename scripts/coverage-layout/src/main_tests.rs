//! Argument and output boundary tests for the standalone source-audit command.

use std::io;

use super::*;

/// Converts portable test arguments without changing production path handling.
fn arguments(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

/// Writes only a bounded number of bytes before reporting a deterministic I/O error.
struct LimitedWriter {
    /// Remaining successful bytes, allowing tests to target JSON or its final newline.
    remaining: usize,
}

impl Write for LimitedWriter {
    /// Accepts bytes up to the budget, then models a closed downstream consumer.
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test consumer closed",
            ));
        }
        let written = self.remaining.min(bytes.len());
        self.remaining -= written;
        Ok(written)
    }

    /// No buffering is present in the deterministic failing writer.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Every report mode preserves the requested root and ordered crate entries.
#[test]
fn report_modes_preserve_explicit_source_arguments() {
    for (prefix, mode) in [
        (None, Mode::Layout),
        (Some("--documentation"), Mode::Documentation),
        (Some("--standards"), Mode::Standards),
    ] {
        let mut values = Vec::new();
        values.extend(prefix);
        values.extend(["root", "src/lib.rs", "tests/checks.rs"]);
        let invocation = parse_arguments(arguments(&values))
            .expect("valid invocation")
            .expect("audit request");
        assert_eq!(
            invocation,
            Invocation {
                mode,
                root: PathBuf::from("root"),
                entries: vec![
                    PathBuf::from("src/lib.rs"),
                    PathBuf::from("tests/checks.rs")
                ],
            }
        );
    }
}

/// Invalid invocations fail before source reads and retain an actionable usage string.
#[test]
fn missing_and_unknown_arguments_fail_before_output() {
    for values in [
        &[][..],
        &["--standards"],
        &["--documentation"],
        &["root"],
        &["--standards", "root"],
        &["--unknown", "root", "lib.rs"],
        &["--"],
        &["--help", "root"],
    ] {
        let mut output = Vec::new();
        let error = run(arguments(values), &mut output).expect_err("invalid invocation");
        assert!(error.contains(USAGE), "{values:?}: {error}");
        assert!(output.is_empty(), "{values:?}");
    }
}

/// An explicit terminator permits dash-prefixed roots without confusing them with flags.
#[test]
fn option_terminator_preserves_literal_dash_prefixed_roots() {
    for values in [
        &["--", "--literal-root", "lib.rs"][..],
        &["--standards", "--", "--literal-root", "lib.rs"],
    ] {
        let invocation = parse_arguments(arguments(values))
            .expect("literal root")
            .expect("audit request");
        assert_eq!(invocation.root, PathBuf::from("--literal-root"));
    }
}

/// Help is a successful text response and reports write failures with help-specific context.
#[test]
fn help_is_available_without_a_source_graph() {
    for flag in ["--help", "-h"] {
        let mut output = Vec::new();
        run(arguments(&[flag]), &mut output).expect("help output");
        assert_eq!(output, format!("{USAGE}\n").as_bytes());
    }
    let error = run(arguments(&["--help"]), &mut LimitedWriter { remaining: 0 })
        .expect_err("closed consumer");
    assert!(error.starts_with("write help:"));
}

/// JSON and trailing-newline failures remain distinguishable without losing I/O details.
#[test]
fn report_output_preserves_json_newline_and_failure_context() {
    let mut output = Vec::new();
    write_report(&mut output, &42).expect("numeric report");
    assert_eq!(output, b"42\n");
    for (remaining, context) in [(0, "write JSON:"), (2, "write final newline:")] {
        let error =
            write_report(&mut LimitedWriter { remaining }, &42).expect_err("closed consumer");
        assert!(error.starts_with(context), "{error}");
        assert!(error.contains("test consumer closed"));
    }
}

/// Native non-Unicode path bytes survive parsing on Unix hosts.
#[cfg(unix)]
#[test]
fn native_path_bytes_are_not_lossily_rewritten() {
    use std::os::unix::ffi::OsStringExt;

    let root = OsString::from_vec(vec![b'r', 0xff]);
    let invocation = parse_arguments([root.clone(), OsString::from("lib.rs")])
        .expect("native path")
        .expect("audit request");
    assert_eq!(invocation.root, PathBuf::from(root));
}
