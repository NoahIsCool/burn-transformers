//! Byte-level lexical checks on untrusted checkpoint paths, names, and numbers.
//!
//! Shard indexes and SafeTensors headers are untrusted input, so their path
//! and name components are checked as raw UTF-8 bytes before they reach the
//! filesystem or an accounting policy:
//!
//! - a portable artifact path is nonempty, has no NUL byte or backslash, and
//!   no Windows drive prefix `X:` ([`lexical_path_violation`]); its
//!   `/`-separated components are all *normal*: the path is not rooted, no
//!   component is `..`, and it does not begin with a `.` component
//!   ([`normal_components`]). Empty and interior `.` components are ignored,
//!   exactly as `std::path::Path::components` ignores them on Unix;
//! - a layered tensor name is `prefix ++ layer ++ suffix` with a *canonical*
//!   unsigned decimal layer: nonempty ASCII digits without a leading zero,
//!   except `0` itself ([`decimal_layer_between`]);
//! - an index byte total is an unsigned decimal integer, optionally followed by
//!   `.` and one or more zeros ([`integral_lexeme_digits`]);
//! - a provenance shard key is canonical: every component is nonempty and
//!   neither `.` nor `..` ([`canonical_components`]), and a digest is exactly 64
//!   lowercase hexadecimal characters ([`is_lowercase_sha256`]).
//!
//! The module has no crate, Burn, or allocator dependencies so that Verus can
//! check it in place through `verus/checkpoint.rs`. Callers keep the
//! platform's own path parsing as a second check, so a platform-specific
//! prefix the byte rules do not know is still rejected.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: `byte` is an ASCII decimal digit.
pub open spec fn is_digit(byte: u8) -> bool {
    48 <= byte <= 57
}

/// Specification: `byte` is an ASCII letter.
pub open spec fn is_letter(byte: u8) -> bool {
    (65 <= byte <= 90) || (97 <= byte <= 122)
}

/// Specification: some byte of `path` equals `value`.
pub open spec fn contains_byte(path: Seq<u8>, value: u8) -> bool {
    exists|i: int| 0 <= i < path.len() && path[i] == value
}

/// Specification: `path` begins with a Windows drive prefix such as `C:`.
pub open spec fn drive_prefix(path: Seq<u8>) -> bool {
    path.len() >= 2 && is_letter(path[0]) && path[1] == 58
}

/// Specification: a `..` component begins at byte `i` of `path`.
pub open spec fn parent_component_at(path: Seq<u8>, i: int) -> bool {
    &&& 0 <= i
    &&& i + 1 < path.len()
    &&& path[i] == 46 && path[i + 1] == 46
    &&& (i == 0 || path[i - 1] == 47)
    &&& (i + 2 == path.len() || path[i + 2] == 47)
}

/// Specification: every component of `path` is normal on Unix.
///
/// The path is not rooted, no component is `..`, and the first component is
/// not `.`; empty and later `.` components are ignored, as Rust's
/// `Path::components` ignores them.
pub open spec fn all_normal(path: Seq<u8>) -> bool {
    &&& !(path.len() > 0 && path[0] == 47)
    &&& !(path.len() > 0 && path[0] == 46 && (path.len() == 1 || path[1] == 47))
    &&& forall|i: int| !#[trigger] parent_component_at(path, i)
}

/// Specification: a `.` component begins at byte `i` of `path`.
pub open spec fn dot_component_at(path: Seq<u8>, i: int) -> bool {
    &&& 0 <= i < path.len()
    &&& path[i] == 46
    &&& (i == 0 || path[i - 1] == 47)
    &&& (i + 1 == path.len() || path[i + 1] == 47)
}

/// Specification: an empty component lies between bytes `i` and `i + 1` (`//`).
pub open spec fn empty_component_at(path: Seq<u8>, i: int) -> bool {
    0 <= i && i + 1 < path.len() && path[i] == 47 && path[i + 1] == 47
}

/// Specification: every `/`-separated component of `path` is nonempty and
/// neither `.` nor `..`, so `path` is the one canonical spelling of itself.
pub open spec fn canonical_segments(path: Seq<u8>) -> bool {
    &&& path.len() > 0
    &&& path[0] != 47
    &&& path[path.len() - 1] != 47
    &&& forall|i: int| !#[trigger] empty_component_at(path, i)
    &&& forall|i: int| !#[trigger] dot_component_at(path, i)
    &&& forall|i: int| !#[trigger] parent_component_at(path, i)
}

/// Specification: `byte` is a lowercase hexadecimal digit.
pub open spec fn is_lower_hex(byte: u8) -> bool {
    (48 <= byte <= 57) || (97 <= byte <= 102)
}

/// Specification: `digits` is a canonical unsigned decimal: nonempty ASCII
/// digits with no leading zero unless it is exactly `0`.
pub open spec fn canonical_decimal(digits: Seq<u8>) -> bool {
    &&& digits.len() > 0
    &&& forall|i: int| 0 <= i < digits.len() ==> is_digit(#[trigger] digits[i])
    &&& (digits.len() == 1 || digits[0] != 48)
}

/// Specification: every byte of `bytes[start..end]` is `value`.
pub open spec fn all_equal(bytes: Seq<u8>, start: int, end: int, value: u8) -> bool {
    forall|i: int| start <= i < end ==> #[trigger] bytes[i] == value
}

/// Specification: every byte of `bytes[start..end]` is an ASCII digit.
pub open spec fn all_digits(bytes: Seq<u8>, start: int, end: int) -> bool {
    forall|i: int| start <= i < end ==> is_digit(#[trigger] bytes[i])
}

} // verus!

/// First lexical reason an untrusted relative path is not portable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathViolation {
    /// The path is empty.
    Empty,
    /// The path contains a NUL byte.
    Nul,
    /// The path contains a backslash.
    Backslash,
    /// The path begins with a Windows drive prefix such as `C:`.
    DrivePrefix,
}

/// Returns the first lexical violation of a portable relative path, if any.
///
/// The rules apply in order: empty, NUL anywhere, backslash anywhere, then a
/// leading ASCII letter followed by `:`.
///
/// Verified: each verdict holds exactly when its rule is the first one broken,
/// and `None` exactly when no rule is broken.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(violation =>
    ensures
        violation == Some(PathViolation::Empty) <==> path@.len() == 0,
        violation == Some(PathViolation::Nul) <==> path@.len() > 0 && contains_byte(path@, 0),
        violation == Some(PathViolation::Backslash) <==> path@.len() > 0
            && !contains_byte(path@, 0) && contains_byte(path@, 92),
        violation == Some(PathViolation::DrivePrefix) <==> path@.len() > 0
            && !contains_byte(path@, 0) && !contains_byte(path@, 92) && drive_prefix(path@),
        violation is None <==> path@.len() > 0 && !contains_byte(path@, 0)
            && !contains_byte(path@, 92) && !drive_prefix(path@),
))]
pub(crate) fn lexical_path_violation(path: &[u8]) -> Option<PathViolation> {
    if path.is_empty() {
        return Some(PathViolation::Empty);
    }
    if contains(path, 0) {
        return Some(PathViolation::Nul);
    }
    if contains(path, b'\\') {
        return Some(PathViolation::Backslash);
    }
    if path.len() >= 2 && is_ascii_letter(path[0]) && path[1] == b':' {
        return Some(PathViolation::DrivePrefix);
    }
    None
}

/// Returns whether `bytes` contains `value`.
///
/// Verified: true exactly when some byte equals `value`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    ensures
        found == contains_byte(bytes@, value),
))]
fn contains(bytes: &[u8], value: u8) -> bool {
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= bytes@.len(),
            forall|i: int| 0 <= i < index ==> bytes@[i] != value,
        decreases bytes@.len() - index,
    ))]
    while index < bytes.len() {
        if bytes[index] == value {
            return true;
        }
        index += 1;
    }
    false
}

/// Returns whether `byte` is an ASCII letter, like `u8::is_ascii_alphabetic`.
///
/// Verified: true exactly for `A..=Z` and `a..=z`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(letter =>
    ensures
        letter == is_letter(byte),
))]
const fn is_ascii_letter(byte: u8) -> bool {
    (byte >= b'A' && byte <= b'Z') || (byte >= b'a' && byte <= b'z')
}

/// Returns whether every `/`-separated component of `path` is normal on Unix.
///
/// A path is rejected when it is rooted (`/…`), begins with a `.` component
/// (`.` or `./…`), or contains a `..` component anywhere. Empty components
/// (`a//b`, a trailing `/`) and later `.` components are ignored, as
/// `Path::components` ignores them.
///
/// Verified: true exactly when [`all_normal`] holds, so a `..` component can
/// never pass and a passing path cannot climb above the directory it is
/// joined to.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(normal =>
    ensures
        normal == all_normal(path@),
))]
pub(crate) fn normal_components(path: &[u8]) -> bool {
    let length = path.len();
    if length > 0 && path[0] == b'/' {
        return false;
    }
    if length > 0 && path[0] == b'.' && (length == 1 || path[1] == b'/') {
        return false;
    }
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= length,
            length == path@.len(),
            forall|i: int| 0 <= i < index ==> !#[trigger] parent_component_at(path@, i),
        decreases length - index,
    ))]
    while index < length {
        let starts_component = index == 0 || path[index - 1] == b'/';
        if starts_component
            && index + 1 < length
            && path[index] == b'.'
            && path[index + 1] == b'.'
            && (index + 2 == length || path[index + 2] == b'/')
        {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(parent_component_at(path@, index as int));
            }
            return false;
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|i: int| !#[trigger] parent_component_at(path@, i) by {
            if 0 <= i < length {
            } else {
                assert(!parent_component_at(path@, i));
            }
        }
    }
    true
}

/// Returns whether every `/`-separated component of `path` is nonempty and neither `.` nor `..`.
///
/// This is stricter than [`normal_components`]: it also rejects the empty and
/// `.` components that `Path::components` silently drops, so two spellings of
/// one file (`a/b` and `a//./b`) cannot both pass.
///
/// Verified: true exactly when [`canonical_segments`] holds.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(canonical =>
    ensures
        canonical == canonical_segments(path@),
))]
pub(crate) fn canonical_components(path: &[u8]) -> bool {
    let length = path.len();
    if length == 0 || path[0] == b'/' || path[length - 1] == b'/' {
        return false;
    }
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= length,
            length == path@.len(),
            forall|i: int| 0 <= i < index ==> !#[trigger] empty_component_at(path@, i),
            forall|i: int| 0 <= i < index ==> !#[trigger] dot_component_at(path@, i),
            forall|i: int| 0 <= i < index ==> !#[trigger] parent_component_at(path@, i),
        decreases length - index,
    ))]
    while index < length {
        let starts_component = index == 0 || path[index - 1] == b'/';
        let next_is_separator = index + 1 < length && path[index + 1] == b'/';
        if path[index] == b'/' && next_is_separator {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(empty_component_at(path@, index as int));
            }
            return false;
        }
        if starts_component && path[index] == b'.' && (index + 1 == length || next_is_separator) {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(dot_component_at(path@, index as int));
            }
            return false;
        }
        if starts_component
            && index + 1 < length
            && path[index] == b'.'
            && path[index + 1] == b'.'
            && (index + 2 == length || path[index + 2] == b'/')
        {
            #[cfg(verus_keep_ghost)]
            proof! {
                assert(parent_component_at(path@, index as int));
            }
            return false;
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|i: int| !#[trigger] empty_component_at(path@, i) by {
            if !(0 <= i < length) {
                assert(!empty_component_at(path@, i));
            }
        }
        assert forall|i: int| !#[trigger] dot_component_at(path@, i) by {
            if !(0 <= i < length) {
                assert(!dot_component_at(path@, i));
            }
        }
        assert forall|i: int| !#[trigger] parent_component_at(path@, i) by {
            if !(0 <= i < length) {
                assert(!parent_component_at(path@, i));
            }
        }
    }
    true
}

/// Returns whether `digest` is exactly 64 lowercase hexadecimal characters.
///
/// Verified: true exactly for length 64 with every byte in `0-9` or `a-f`, the
/// one canonical spelling of a SHA-256 digest.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(canonical =>
    ensures
        canonical == (digest@.len() == 64
            && forall|i: int| 0 <= i < digest@.len() ==> is_lower_hex(#[trigger] digest@[i])),
))]
pub(crate) fn is_lowercase_sha256(digest: &[u8]) -> bool {
    if digest.len() != 64 {
        return false;
    }
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= digest@.len(),
            forall|i: int| 0 <= i < index ==> is_lower_hex(#[trigger] digest@[i]),
        decreases digest@.len() - index,
    ))]
    while index < digest.len() {
        if !is_lowercase_hex_digit(digest[index]) {
            return false;
        }
        index += 1;
    }
    true
}

/// Returns whether `byte` is `0-9` or `a-f`.
///
/// Verified: true exactly for the sixteen lowercase hexadecimal digits.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(hex =>
    ensures
        hex == is_lower_hex(byte),
))]
const fn is_lowercase_hex_digit(byte: u8) -> bool {
    (byte >= b'0' && byte <= b'9') || (byte >= b'a' && byte <= b'f')
}

/// Returns whether `bytes[start..end]` contains only ASCII digits.
///
/// Verified: true exactly when every byte in the range is a digit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(digits =>
    requires
        start <= end <= bytes@.len(),
    ensures
        digits == all_digits(bytes@, start as int, end as int),
))]
fn only_digits(bytes: &[u8], start: usize, end: usize) -> bool {
    let mut index = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= end,
            end <= bytes@.len(),
            all_digits(bytes@, start as int, index as int),
        decreases end - index,
    ))]
    while index < end {
        if !(bytes[index] >= b'0' && bytes[index] <= b'9') {
            return false;
        }
        index += 1;
    }
    true
}

/// Returns whether `name` is `prefix`, then a canonical decimal layer, then `suffix`.
///
/// The prefix is stripped from the front and the suffix from what remains,
/// exactly as `str::strip_prefix` followed by `str::strip_suffix` do.
///
/// Verified: true exactly when `name` splits as `prefix ++ layer ++ suffix`
/// with `layer` a canonical decimal; the anchors may not overlap, so a name
/// shorter than both anchors never matches.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(matched =>
    ensures
        matched == ({
            &&& name@.len() >= prefix@.len() + suffix@.len()
            &&& name@.subrange(0, prefix@.len() as int) == prefix@
            &&& name@.subrange(name@.len() - suffix@.len(), name@.len() as int) == suffix@
            &&& canonical_decimal(name@.subrange(
                prefix@.len() as int,
                name@.len() - suffix@.len(),
            ))
        }),
))]
pub(crate) fn decimal_layer_between(name: &[u8], prefix: &[u8], suffix: &[u8]) -> bool {
    let length = name.len();
    let anchors = match prefix.len().checked_add(suffix.len()) {
        Some(anchors) => anchors,
        None => return false,
    };
    if length < anchors {
        return false;
    }
    if !equal_at(name, 0, prefix) || !equal_at(name, name.len() - suffix.len(), suffix) {
        return false;
    }
    let start = prefix.len();
    let end = name.len() - suffix.len();
    #[cfg(verus_keep_ghost)]
    proof! {
        let layer = name@.subrange(start as int, end as int);
        assert forall|i: int| 0 <= i < layer.len() implies layer[i] == name@[start + i] by {}
    }
    if start == end || !only_digits(name, start, end) {
        #[cfg(verus_keep_ghost)]
        proof! {
            let layer = name@.subrange(start as int, end as int);
            if start < end && !all_digits(name@, start as int, end as int) {
                let witness = choose|i: int| start <= i < end && !is_digit(#[trigger] name@[i]);
                assert(!is_digit(layer[witness - start]));
            }
        }
        return false;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let layer = name@.subrange(start as int, end as int);
        assert forall|i: int| 0 <= i < layer.len() implies is_digit(#[trigger] layer[i]) by {
            assert(layer[i] == name@[start + i]);
        }
    }
    end - start == 1 || name[start] != b'0'
}

/// Returns whether `bytes[offset..offset + expected.len()]` equals `expected`.
///
/// Verified: true exactly when the range is in bounds and equals `expected`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(equal =>
    ensures
        equal == (offset + expected@.len() <= bytes@.len()
            && bytes@.subrange(offset as int, offset + expected@.len()) == expected@),
))]
fn equal_at(bytes: &[u8], offset: usize, expected: &[u8]) -> bool {
    let length = bytes.len();
    if offset > length || length - offset < expected.len() {
        return false;
    }
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= expected@.len(),
            length == bytes@.len(),
            offset + expected@.len() <= bytes@.len(),
            forall|i: int| 0 <= i < index ==> bytes@[offset + i] == expected@[i],
        decreases expected@.len() - index,
    ))]
    while index < expected.len() {
        if bytes[offset + index] != expected[index] {
            #[cfg(verus_keep_ghost)]
            proof! {
                let window = bytes@.subrange(offset as int, offset + expected@.len());
                assert(window[index as int] != expected@[index as int]);
            }
            return false;
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        let window = bytes@.subrange(offset as int, offset + expected@.len());
        assert(window =~= expected@);
    }
    true
}

/// Returns the length of the integer digits of an integral JSON number lexeme.
///
/// The lexeme splits at its first `.`. It is accepted when the part before is
/// one or more ASCII digits and the part after, if any, is one or more `0`
/// bytes; the caller then parses only the integer digits.
///
/// Verified: `Some(k)` exactly for accepted lexemes, with `k` the index of the
/// first `.` (or the whole length), so the integer digits are `bytes[..k]` and
/// any fraction is all zeros.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(digits =>
    ensures
        match digits {
            Some(k) => {
                &&& 1 <= k <= lexeme@.len()
                &&& all_digits(lexeme@, 0, k as int)
                &&& (k == lexeme@.len() || {
                    &&& lexeme@[k as int] == 46
                    &&& k + 1 < lexeme@.len()
                    &&& all_equal(lexeme@, k + 1, lexeme@.len() as int, 48)
                })
            },
            None => !exists|k: int| {
                &&& 1 <= k <= lexeme@.len()
                &&& #[trigger] all_digits(lexeme@, 0, k)
                &&& (k == lexeme@.len() || {
                    &&& lexeme@[k] == 46
                    &&& k + 1 < lexeme@.len()
                    &&& all_equal(lexeme@, k + 1, lexeme@.len() as int, 48)
                })
            },
        }
))]
pub(crate) fn integral_lexeme_digits(lexeme: &[u8]) -> Option<usize> {
    let length = lexeme.len();
    let mut point = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            point <= length,
            length == lexeme@.len(),
            forall|i: int| 0 <= i < point ==> lexeme@[i] != 46,
        decreases length - point,
    ))]
    while point < length && lexeme[point] != b'.' {
        point += 1;
    }
    let digits_ok = point > 0 && only_digits(lexeme, 0, point);
    let fraction_ok = point == length || (point + 1 < length && only_zeros(lexeme, point + 1));
    #[cfg(verus_keep_ghost)]
    proof! {
        // Digits never include `.`, so any accepted split is at the first `.`.
        assert forall|k: int| 1 <= k <= length && #[trigger] all_digits(lexeme@, 0, k)
            && k != point implies !(k == lexeme@.len() || lexeme@[k] == 46)
        by {
            if k < point {
                assert(lexeme@[k] != 46);
                assert(k != lexeme@.len());
            } else {
                assert(lexeme@[point as int] == 46);
                assert(!is_digit(lexeme@[point as int]));
            }
        }
    }
    if digits_ok && fraction_ok {
        Some(point)
    } else {
        None
    }
}

/// Returns whether every byte of `bytes[start..]` is `0`.
///
/// Verified: true exactly when the suffix is all ASCII zeros.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(zeros =>
    requires
        start <= bytes@.len(),
    ensures
        zeros == all_equal(bytes@, start as int, bytes@.len() as int, 48),
))]
fn only_zeros(bytes: &[u8], start: usize) -> bool {
    let mut index = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            start <= index <= bytes@.len(),
            all_equal(bytes@, start as int, index as int, 48),
        decreases bytes@.len() - index,
    ))]
    while index < bytes.len() {
        if bytes[index] != b'0' {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
#[path = "lexical_tests.rs"]
mod tests;
