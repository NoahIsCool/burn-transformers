//! Unit tests of the shared fuzz helpers.

use serde_json::json;

use super::*;

/// Floats and integers must both round-trip exactly.
#[test]
fn json_equal_requires_exact_floats_and_integers() {
    // Floats a few ulps apart (the serde_json round-trip error before
    // `float_roundtrip`) differ; the parser now returns the same bits.
    let float = 3.799_796_971_636_184_6e-262;
    let nearby = 3.799_796_971_636_185e-262;
    assert!(!json_equal(&json!(float), &json!(nearby)));
    let reparsed: serde_json::Value = serde_json::from_str(&json!(float).to_string()).unwrap();
    assert!(json_equal(&json!(float), &reparsed));
    assert!(!json_equal(&json!(1.0), &json!(1.0 + 1e-12)));
    assert!(!json_equal(&json!(0.0), &json!(-0.0)));
    // Integers are exact, even where an f64 view could not tell them apart.
    assert!(json_equal(&json!(u64::MAX), &json!(u64::MAX)));
    assert!(!json_equal(&json!(u64::MAX), &json!(u64::MAX - 1)));
    assert!(!json_equal(
        &json!(9_007_199_254_740_993_u64),
        &json!(9_007_199_254_740_992_u64)
    ));
    assert!(!json_equal(&json!(-5_i64), &json!(-4_i64)));
    // An integer never equals a float, however close.
    assert!(!json_equal(&json!(2), &json!(2.0)));
    assert!(!json_equal(&json!(u64::MAX), &json!(u64::MAX as f64)));
    // Structure must match exactly; numbers nest.
    assert!(json_equal(
        &json!({"a": [1, float], "b": null}),
        &json!({"a": [1, float], "b": null})
    ));
    assert!(!json_equal(
        &json!({"a": [1, float], "b": null}),
        &json!({"a": [1, nearby], "b": null})
    ));
    assert!(!json_equal(&json!({"a": [1]}), &json!({"a": [1, 2]})));
    assert!(!json_equal(&json!({"a": 1}), &json!({"b": 1})));
    assert!(!json_equal(&json!("1"), &json!(1)));
}

/// The total decoder reads zeros after the input and bounds every helper.
#[test]
fn bytes_decoder_is_total() {
    let mut bytes = Bytes::new(&[7, 255, 254, 3]);
    assert_eq!(bytes.below(4), 3);
    assert_eq!(bytes.size(), usize::MAX);
    assert_eq!(bytes.size(), usize::MAX - 1);
    assert!(bytes.flag());
    assert_eq!(bytes.byte(), 0);
    assert_eq!(bytes.below(0), 0);
    assert_eq!(bytes.take(8), &[] as &[u8]);
    let mut floats = Bytes::new(&[0, 255, 0, 0, 0, 0, 0, 0, 0xf0, 0x7f]);
    assert_eq!(floats.interesting_float(), INTERESTING_FLOATS[0]);
    assert_eq!(floats.interesting_float(), f64::INFINITY);
}
