//! GPT-2 configuration validation: extent, width, and overflow failures name their field.

use super::*;

/// A valid one-block topology that each case perturbs.
fn tiny() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 7,
        n_embd: 4,
        n_layer: 1,
        n_head: 2,
        n_positions: 8,
        n_inner: Some(6),
        ..Gpt2Config::default()
    }
}

/// Validates `config` and returns the failing `(field, reason)`.
fn failure(config: &Gpt2Config) -> (&'static str, String) {
    let error = Validate::validate(config).unwrap_err();
    assert_eq!(error.architecture, "gpt2");
    (error.field, error.reason)
}

/// Zero or beyond-`i32::MAX` extents are rejected before any other rule, naming the field.
#[test]
fn every_extent_must_be_positive_and_portable() {
    type Edit = fn(&mut Gpt2Config, usize);
    let cases: [(&str, Edit); 5] = [
        ("vocab_size", |config, value| config.vocab_size = value),
        ("n_embd", |config, value| config.n_embd = value),
        ("n_layer", |config, value| config.n_layer = value),
        ("n_head", |config, value| config.n_head = value),
        ("n_positions", |config, value| config.n_positions = value),
    ];
    for (field, edit) in cases {
        for value in [0, i32::MAX as usize + 1] {
            let mut config = tiny();
            edit(&mut config, value);
            assert_eq!(
                failure(&config),
                (
                    field,
                    "must be positive and fit the portable integer index range".into()
                ),
                "{field}={value}"
            );
        }
    }
    Validate::validate(&tiny()).expect("the unperturbed topology is valid");
}

/// The MLP width must be positive, and every `n_embd`-wide parameter product must fit `usize`.
#[test]
fn mlp_width_and_parameter_products_are_bounded() {
    assert_eq!(
        failure(&Gpt2Config {
            n_inner: Some(0),
            ..tiny()
        }),
        ("n_inner", "must be positive and fit usize".into())
    );
    // `n_inner` is not limited to the portable index range, so its product can overflow.
    assert_eq!(
        failure(&Gpt2Config {
            n_inner: Some(usize::MAX / 2),
            ..tiny()
        }),
        ("n_embd", "parameter dimensions overflow usize".into())
    );
    assert_eq!(
        failure(&Gpt2Config {
            add_cross_attention: true,
            ..tiny()
        }),
        (
            "add_cross_attention",
            "cross-attention blocks are not implemented".into()
        )
    );
}
