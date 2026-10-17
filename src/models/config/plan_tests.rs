//! Exact verdicts, fault order, and edge cases of the configuration integer cores.

use super::*;

/// Sections are rejected for emptiness or zeros before any width comparison or overflow.
#[test]
fn mrope_sections_check_positivity_before_width() {
    assert_eq!(mrope_section_fault(&[16, 24, 24], 128), None);
    assert_eq!(mrope_section_fault(&[1], 2), None);
    assert_eq!(
        mrope_section_fault(&[], 0),
        Some(MropeSectionFault::EmptyOrZero)
    );
    // A zero after an overflowing prefix is still reported as a zero.
    assert_eq!(
        mrope_section_fault(&[usize::MAX, usize::MAX, 0], 0),
        Some(MropeSectionFault::EmptyOrZero)
    );
    assert_eq!(
        mrope_section_fault(&[16, 24, 23], 128),
        Some(MropeSectionFault::WidthMismatch)
    );
    // Overflowing sums and doublings are mismatches, not wrapped values.
    assert_eq!(
        mrope_section_fault(&[usize::MAX, 1], 0),
        Some(MropeSectionFault::WidthMismatch)
    );
    assert_eq!(
        mrope_section_fault(&[usize::MAX / 2 + 1], 0),
        Some(MropeSectionFault::WidthMismatch)
    );
    assert_eq!(mrope_section_fault(&[usize::MAX / 2], usize::MAX - 1), None);
}

/// Divisibility reports the numerator before the denominator and exposes exact groups.
#[test]
fn divisibility_faults_follow_check_order() {
    assert_eq!(divisibility_fault(32, 8), None);
    assert_eq!(divisibility_fault(7, 7), None);
    assert_eq!(
        divisibility_fault(0, 0),
        Some(DivisibilityFault::ZeroNumerator)
    );
    assert_eq!(
        divisibility_fault(4, 0),
        Some(DivisibilityFault::ZeroDenominator)
    );
    assert_eq!(
        divisibility_fault(4, 3),
        Some(DivisibilityFault::Indivisible)
    );
    assert_eq!(
        divisibility_fault(3, 4),
        Some(DivisibilityFault::Indivisible)
    );
}

/// Token IDs must be present and below the vocabulary size.
#[test]
fn token_ids_must_be_present_and_in_vocabulary() {
    assert_eq!(token_id_fault(&[0, 9], 10), None);
    assert_eq!(token_id_fault(&[], 10), Some(TokenIdFault::Empty));
    assert_eq!(
        token_id_fault(&[0, 10], 10),
        Some(TokenIdFault::OutOfVocabulary)
    );
    assert_eq!(token_id_fault(&[0], 0), Some(TokenIdFault::OutOfVocabulary));
}

/// The trained context must be positive and fit the maximum context.
#[test]
fn original_context_must_be_positive_and_bounded() {
    assert!(original_context_fits(4096, 4096));
    assert!(original_context_fits(1, 2));
    assert!(!original_context_fits(0, 4096));
    assert!(!original_context_fits(4097, 4096));
}

/// Flat and per-layer mappings resolve the trained context with the documented precedence.
#[test]
fn original_context_resolution_follows_precedence() {
    // Flat: nested, then top level, then the fallback; equal sources agree.
    assert_eq!(
        resolve_original_context(Some(4), None, 9, false, true),
        Ok(4)
    );
    assert_eq!(
        resolve_original_context(None, Some(5), 9, false, true),
        Ok(5)
    );
    assert_eq!(resolve_original_context(None, None, 9, false, true), Ok(9));
    assert_eq!(
        resolve_original_context(Some(4), Some(4), 9, false, true),
        Ok(4)
    );
    // Flat conflicts are faults under the reject policy, top level wins otherwise.
    assert_eq!(
        resolve_original_context(Some(4), Some(5), 9, false, true),
        Err(OriginalContextFault::FlatConflict {
            nested: 4,
            top_level: 5
        })
    );
    assert_eq!(
        resolve_original_context(Some(4), Some(5), 9, false, false),
        Ok(5)
    );
    // Per-layer: the top level is never read, only compared.
    assert_eq!(
        resolve_original_context(None, Some(5), 9, true, true),
        Err(OriginalContextFault::LayerConflict {
            top_level: 5,
            resolved: 9
        })
    );
    assert_eq!(
        resolve_original_context(None, Some(5), 9, true, false),
        Ok(9)
    );
    assert_eq!(
        resolve_original_context(Some(5), Some(5), 9, true, true),
        Ok(5)
    );
    // Zeros: a nested zero first, then a resolved zero.
    assert_eq!(
        resolve_original_context(Some(0), Some(5), 9, false, true),
        Err(OriginalContextFault::NestedZero)
    );
    assert_eq!(
        resolve_original_context(None, Some(0), 9, false, true),
        Err(OriginalContextFault::Zero)
    );
    assert_eq!(
        resolve_original_context(Some(4), Some(0), 9, false, false),
        Err(OriginalContextFault::Zero)
    );
    assert_eq!(
        resolve_original_context(None, None, 0, true, true),
        Err(OriginalContextFault::Zero)
    );
}

/// Token starts skip quoted and escaped text and never overlap.
#[test]
fn infinity_tokens_are_found_only_outside_strings() {
    assert_eq!(infinity_token_starts(b"Infinity"), vec![0]);
    assert_eq!(
        infinity_token_starts(b"[Infinity,InfinityInfinity]"),
        vec![1, 10, 18]
    );
    assert_eq!(
        infinity_token_starts(br#"["a \" Infinity", Infinity]"#),
        vec![18]
    );
    assert_eq!(infinity_token_starts(br#"["\\", Infinity]"#), vec![7]);
    assert_eq!(
        infinity_token_starts(br#""unterminated Infinity"#),
        Vec::<usize>::new()
    );
    assert_eq!(infinity_token_starts(b"Infinit"), Vec::<usize>::new());
    assert_eq!(
        infinity_token_starts("\u{7121}Infinity".as_bytes()),
        vec![3]
    );
}

/// One automaton step handles quotes, escapes, and ordinary bytes in each state.
#[test]
fn lexer_step_tracks_strings_and_escapes() {
    // Outside a string: only a quote opens one.
    assert_eq!(lexer_advance(false, false, b'"'), (true, false));
    assert_eq!(lexer_advance(false, false, b'\\'), (false, false));
    assert_eq!(lexer_advance(false, false, b'I'), (false, false));
    // Inside: backslash escapes, an unescaped quote closes, an escaped byte only clears.
    assert_eq!(lexer_advance(true, false, b'\\'), (true, true));
    assert_eq!(lexer_advance(true, false, b'"'), (false, false));
    assert_eq!(lexer_advance(true, false, 0xE6), (true, false));
    assert_eq!(lexer_advance(true, true, b'"'), (true, false));
    assert_eq!(lexer_advance(true, true, b'\\'), (true, false));
}

/// Counts are accepted exactly in `1..=maximum`, a zero reported before an excess.
#[test]
fn count_faults_follow_check_order() {
    assert_eq!(count_fault(1, 1), None);
    assert_eq!(count_fault(7, usize::MAX), None);
    assert_eq!(count_fault(0, 0), Some(CountFault::Zero));
    assert_eq!(count_fault(0, 8), Some(CountFault::Zero));
    assert_eq!(count_fault(9, 8), Some(CountFault::AboveMaximum));
    assert_eq!(
        count_fault(usize::MAX, 1 << 20),
        Some(CountFault::AboveMaximum)
    );
}

/// The inventory total is exact, and any overflow (but only a real one) is `None`.
#[test]
fn moe_inventories_are_exact_or_overflow() {
    let qwen3_30b = MoeInventory {
        fixed: 3,
        layers: 48,
        per_layer: 8,
        sparse_layers: 48,
        per_sparse_layer: 1,
        experts: 128,
        per_expert: 3,
        per_dense_layer: 3,
    };
    assert_eq!(
        moe_inventory(qwen3_30b),
        Some(3 + 48 * 8 + 48 * (1 + 128 * 3))
    );
    let mixed = MoeInventory {
        sparse_layers: 1,
        ..qwen3_30b
    };
    assert_eq!(moe_inventory(mixed), Some(3 + 48 * 8 + (1 + 384) + 47 * 3));
    // Without sparse layers the expert product is never formed.
    let dense = MoeInventory {
        sparse_layers: 0,
        experts: usize::MAX,
        per_expert: usize::MAX,
        ..qwen3_30b
    };
    assert_eq!(moe_inventory(dense), Some(3 + 48 * 8 + 48 * 3));
    for overflowing in [
        MoeInventory {
            experts: usize::MAX,
            ..qwen3_30b
        },
        MoeInventory {
            layers: usize::MAX,
            sparse_layers: 0,
            ..qwen3_30b
        },
        MoeInventory {
            fixed: usize::MAX,
            ..qwen3_30b
        },
        MoeInventory {
            experts: usize::MAX / 3,
            ..mixed
        },
    ] {
        assert_eq!(moe_inventory(overflowing), None, "{overflowing:?}");
    }
    // The largest representable total is still exact.
    let exact = MoeInventory {
        fixed: usize::MAX - 1,
        layers: 1,
        per_layer: 1,
        sparse_layers: 0,
        per_sparse_layer: 0,
        experts: 0,
        per_expert: 0,
        per_dense_layer: 0,
    };
    assert_eq!(moe_inventory(exact), Some(usize::MAX));
}
