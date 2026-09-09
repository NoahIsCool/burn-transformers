//! Isolated LLVM coverage fixture: this is not a transformer acceptance test.
#![allow(dead_code)]

/// An unused function must contribute mapped, zero-count lines.
fn never_called() -> i32 {
    -1
}

/// Distinct generic instantiations take different branches of the same source.
fn choose<T: Copy>(value: T, take: bool) -> Option<T> {
    // Comments between entries are not independent executable regions.
    if take {
        Some(value)
    } else {
        None
    }
}

/// Nested closures exercise overlapping function and physical-line accounting.
fn nested(flag: bool) -> i32 {
    let text = r###"#[cfg(test)] mod fake { } // λ not Rust syntax"###;
    let outer = |value: i32| {
        let inner = |number: i32| if number > 0 { number + 2 } else { -2 };
        if flag {
            inner(value)
        } else {
            0
        }
    };
    assert!(!text.is_empty());
    outer(3)
}

/// A closing delimiter in column one exercises the wrapped endpoint line.
fn boundary(value: i32) -> i32
{
    value + 1
}

// One physical line can belong to two distinct function coverage summaries.
fn first() -> i32 { 1 } fn second() -> i32 { 2 }

/// Invoke selected branches without executing every function or closure branch.
fn main() {
    assert_eq!(choose(4_u8, true), Some(4));
    assert_eq!(choose(7_u16, false), None);
    assert_eq!(nested(true), 5);
    assert_eq!(boundary(5), 6);
    assert_eq!(first(), 1);
}
