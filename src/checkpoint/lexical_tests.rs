//! Exhaustive and property checks of the byte-level path, name, and number rules.

use std::path::{Component, Path};

use proptest::prelude::*;

use super::*;

/// Every string of up to `length` bytes over `alphabet`, in length order.
fn all_strings(alphabet: &[u8], length: usize) -> Vec<Vec<u8>> {
    let mut strings = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..length {
        let mut next = Vec::new();
        for prefix in &frontier {
            for &byte in alphabet {
                let mut string = prefix.clone();
                string.push(byte);
                next.push(string);
            }
        }
        strings.extend(next.iter().cloned());
        frontier = next;
    }
    strings
}

/// The original string-based path rules, as an independent oracle.
fn reference_violation(path: &str) -> Option<PathViolation> {
    if path.is_empty() {
        Some(PathViolation::Empty)
    } else if path.contains('\0') {
        Some(PathViolation::Nul)
    } else if path.contains('\\') {
        Some(PathViolation::Backslash)
    } else if path.len() >= 2
        && path.as_bytes()[0].is_ascii_alphabetic()
        && path.as_bytes()[1] == b':'
    {
        Some(PathViolation::DrivePrefix)
    } else {
        None
    }
}

/// Lexical path verdicts match the original rules on every short string.
#[test]
fn lexical_path_violations_match_the_original_rules() {
    for bytes in all_strings(b"a:\\\0/Z.", 4) {
        let path = std::str::from_utf8(&bytes).expect("ASCII");
        assert_eq!(
            lexical_path_violation(&bytes),
            reference_violation(path),
            "{path:?}"
        );
    }
}

/// On Unix, the byte rule agrees with `Path::components` on every short path.
#[cfg(unix)]
#[test]
fn normal_components_match_std_path_components_on_unix() {
    for bytes in all_strings(b"a./", 7) {
        let path = std::str::from_utf8(&bytes).expect("ASCII");
        let candidate = Path::new(path);
        let std_normal = !candidate.is_absolute()
            && candidate
                .components()
                .all(|component| matches!(component, Component::Normal(_)));
        assert_eq!(normal_components(&bytes), std_normal, "{path:?}");
    }
}

/// Representative portable and escaping paths.
#[test]
fn normal_components_reject_roots_parents_and_leading_dots() {
    for accepted in [
        "a", "a/b", "a//b", "a/", "a/./b", "a/.", ".a", "..a", "a/..b", "a..",
    ] {
        assert!(normal_components(accepted.as_bytes()), "{accepted}");
    }
    for rejected in [
        "/a", ".", "./a", "..", "../a", "a/..", "a/../b", "a//../b", "/",
    ] {
        assert!(!normal_components(rejected.as_bytes()), "{rejected}");
    }
}

/// The original strip-and-check selector, as an independent oracle.
fn reference_layer_match(name: &str, prefix: &str, suffix: &str) -> bool {
    let Some(layer) = name
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
    else {
        return false;
    };
    !layer.is_empty()
        && layer.bytes().all(|byte| byte.is_ascii_digit())
        && (layer == "0" || !layer.starts_with('0'))
}

/// Layer selection matches the original string rules on every short name.
#[test]
fn decimal_layers_match_the_original_selector() {
    let anchors = ["", "a", "a0", "0a", "aa"];
    for bytes in all_strings(b"0a1", 5) {
        let name = std::str::from_utf8(&bytes).expect("ASCII");
        for prefix in anchors {
            for suffix in anchors {
                assert_eq!(
                    decimal_layer_between(&bytes, prefix.as_bytes(), suffix.as_bytes()),
                    reference_layer_match(name, prefix, suffix),
                    "{name:?} between {prefix:?} and {suffix:?}"
                );
            }
        }
    }
    let prefix = b"backbone.layers.";
    let suffix = b".mixer.gate.weight";
    assert!(decimal_layer_between(
        b"backbone.layers.17.mixer.gate.weight",
        prefix,
        suffix
    ));
    assert!(decimal_layer_between(
        b"backbone.layers.0.mixer.gate.weight",
        prefix,
        suffix
    ));
    assert!(!decimal_layer_between(
        b"backbone.layers.07.mixer.gate.weight",
        prefix,
        suffix
    ));
    assert!(!decimal_layer_between(
        b"mtp.layers.1.mixer.gate.weight",
        prefix,
        suffix
    ));
}

/// The original split-and-check lexeme rule, returning the integer digits' length.
fn reference_integral_digits(lexeme: &str) -> Option<usize> {
    let (integer, fraction) = match lexeme.split_once('.') {
        Some((integer, fraction)) => (integer, Some(fraction)),
        None => (lexeme, None),
    };
    let canonical_digits = !integer.is_empty() && integer.bytes().all(|byte| byte.is_ascii_digit());
    let exact_zero_fraction = fraction
        .map(|fraction| !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0'))
        .unwrap_or(true);
    (canonical_digits && exact_zero_fraction).then_some(integer.len())
}

/// Integral lexemes match the original split rule on every short string.
#[test]
fn integral_lexemes_match_the_original_rule() {
    for bytes in all_strings(b"01.-e", 5) {
        let lexeme = std::str::from_utf8(&bytes).expect("ASCII");
        assert_eq!(
            integral_lexeme_digits(&bytes),
            reference_integral_digits(lexeme),
            "{lexeme:?}"
        );
    }
    assert_eq!(integral_lexeme_digits(b"55562855904.0"), Some(11));
    assert_eq!(integral_lexeme_digits(b"12."), None);
    assert_eq!(integral_lexeme_digits(b".0"), None);
    assert_eq!(integral_lexeme_digits(b"1.00.0"), None);
}

/// The original split-based canonical shard key rule, as an independent oracle.
fn reference_canonical(path: &str) -> bool {
    !path
        .split('/')
        .any(|component| matches!(component, "" | "." | ".."))
}

/// Canonical components match the original split rule on every short path.
#[test]
fn canonical_components_match_the_split_rule() {
    for bytes in all_strings(b"a./", 7) {
        let path = std::str::from_utf8(&bytes).expect("ASCII");
        assert_eq!(
            canonical_components(&bytes),
            reference_canonical(path),
            "{path:?}"
        );
    }
}

/// Digest spelling matches the original lowercase-hex rule.
#[test]
fn lowercase_digests_match_the_original_rule() {
    let reference = |digest: &str| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    let valid = "0123456789abcdef".repeat(4);
    assert!(is_lowercase_sha256(valid.as_bytes()));
    for byte in 0..=u8::MAX {
        let mut digest = valid.clone().into_bytes();
        digest[17] = byte;
        let text = String::from_utf8_lossy(&digest).into_owned();
        assert_eq!(
            is_lowercase_sha256(&digest),
            digest.len() == 64 && reference(&text),
            "byte {byte}"
        );
    }
    assert!(!is_lowercase_sha256(&valid.as_bytes()[1..]));
    assert!(!is_lowercase_sha256(format!("{valid}0").as_bytes()));
}

proptest! {
    /// Arbitrary byte strings get the same path verdict as the original rules.
    #[test]
    fn arbitrary_paths_match_the_original_rules(path in "[a-zA-Z0-9:./\\\\\\x00]{0,12}") {
        prop_assert_eq!(lexical_path_violation(path.as_bytes()), reference_violation(&path));
    }

    /// Arbitrary paths agree with `Path::components` on Unix.
    #[cfg(unix)]
    #[test]
    fn arbitrary_paths_match_std_components(path in "[a./]{0,16}") {
        let candidate = Path::new(&path);
        let std_normal = !candidate.is_absolute()
            && candidate
                .components()
                .all(|component| matches!(component, Component::Normal(_)));
        prop_assert_eq!(normal_components(path.as_bytes()), std_normal);
    }
}
