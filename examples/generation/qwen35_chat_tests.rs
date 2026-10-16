//! Command-line parsing checks for the Qwen3.5 chat example.
//!
//! The parser receives the arguments after the program name, exactly as
//! `std::env::args_os().skip(1)` yields them. These tests only parse text: no
//! checkpoint is read and no Metal device is created.

use super::*;

/// Converts string literals into the operating-system strings the parser receives.
fn args(values: &[&str]) -> impl Iterator<Item = OsString> {
    values
        .iter()
        .map(OsString::from)
        .collect::<Vec<_>>()
        .into_iter()
}

/// A model directory alone selects the default prompt and token budget.
#[test]
fn directory_alone_uses_the_defaults() {
    let (root, prompt, max_new_tokens) = arguments(args(&["/models/qwen"])).unwrap();
    assert_eq!(root, PathBuf::from("/models/qwen"));
    assert_eq!(prompt, DEFAULT_PROMPT);
    assert_eq!(max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
}

/// An explicit prompt and budget are both kept.
#[test]
fn explicit_prompt_and_budget_are_kept() {
    let parsed = arguments(args(&["dir", "Hello", "1"])).unwrap();
    assert_eq!(parsed, (PathBuf::from("dir"), "Hello".into(), 1));
}

/// A budget that is not a non-negative integer names the argument and repeats the bad value.
#[test]
fn malformed_budgets_name_the_argument_and_value() {
    for (value, reason) in [
        ("abc", "invalid digit found in string"),
        ("-1", "invalid digit found in string"),
        ("", "cannot parse integer from empty string"),
        (
            "99999999999999999999999",
            "number too large to fit in target type",
        ),
    ] {
        let error = arguments(args(&["dir", "Hello", value])).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("invalid MAX_NEW_TOKENS {value:?}: {reason}")
        );
    }
}

/// A zero budget is rejected with its own message instead of the usage line.
#[test]
fn zero_budget_is_rejected_explicitly() {
    let error = arguments(args(&["dir", "Hello", "0"])).unwrap_err();
    assert_eq!(
        error.to_string(),
        "MAX_NEW_TOKENS must be greater than zero"
    );
}

/// Missing or surplus arguments report the usage line.
#[test]
fn wrong_argument_counts_report_usage() {
    for values in [&[][..], &["dir", "Hello", "8", "extra"][..]] {
        let error = arguments(args(values)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "usage: qwen35_chat MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]"
        );
    }
}

/// A directory need not be UTF-8, but the prompt and budget must be, and each error names its argument.
#[cfg(unix)]
#[test]
fn only_the_prompt_and_budget_must_be_utf8() {
    use std::os::unix::ffi::OsStringExt;

    let invalid = || OsString::from_vec(vec![b'q', 0xff]);
    let (root, _, _) = arguments([invalid()].into_iter()).unwrap();
    assert_eq!(root, PathBuf::from(invalid()));
    let error = arguments([OsString::from("dir"), invalid()].into_iter()).unwrap_err();
    assert_eq!(error.to_string(), "PROMPT must be UTF-8 text");
    let error = arguments([OsString::from("dir"), OsString::from("Hello"), invalid()].into_iter())
        .unwrap_err();
    assert_eq!(error.to_string(), "MAX_NEW_TOKENS must be UTF-8 text");
}
