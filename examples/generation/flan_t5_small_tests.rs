//! Command-line parsing checks for the FLAN-T5-small example.
//!
//! The parser receives the arguments after the program name, exactly as
//! `std::env::args_os().skip(1)` yields them, so no process is spawned and no
//! checkpoint is read.

use super::*;

/// Converts string literals into the operating-system strings the parser receives.
fn args(values: &[&str]) -> impl Iterator<Item = OsString> {
    values
        .iter()
        .map(OsString::from)
        .collect::<Vec<_>>()
        .into_iter()
}

/// A model directory alone selects the documented translation prompt.
#[test]
fn directory_alone_uses_the_default_prompt() {
    let (root, prompt) = arguments(args(&["/models/flan"])).unwrap();
    assert_eq!(root, PathBuf::from("/models/flan"));
    assert_eq!(prompt, DEFAULT_PROMPT);
}

/// An explicit prompt replaces the default verbatim.
#[test]
fn explicit_prompt_is_kept_verbatim() {
    let (_, prompt) = arguments(args(&["dir", "summarize: two  spaces"])).unwrap();
    assert_eq!(prompt, "summarize: two  spaces");
}

/// Missing or surplus arguments report the usage line.
#[test]
fn wrong_argument_counts_report_usage() {
    for values in [&[][..], &["dir", "prompt", "extra"][..]] {
        let error = arguments(args(values)).unwrap_err();
        assert_eq!(error.to_string(), "usage: flan_t5_small MODEL_DIR [PROMPT]");
    }
}

/// A directory need not be UTF-8, but a prompt must be, and the error says which argument failed.
#[cfg(unix)]
#[test]
fn only_the_prompt_must_be_utf8() {
    use std::os::unix::ffi::OsStringExt;

    let invalid = || OsString::from_vec(vec![b'm', 0xff]);
    let (root, _) = arguments([invalid()].into_iter()).unwrap();
    assert_eq!(root, PathBuf::from(invalid()));
    let error = arguments([OsString::from("dir"), invalid()].into_iter()).unwrap_err();
    assert_eq!(error.to_string(), "PROMPT must be UTF-8 text");
}
