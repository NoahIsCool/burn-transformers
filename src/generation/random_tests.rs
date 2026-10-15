//! SplitMix64 stream tests: equivalence with the classic sequential generator and output ranges.

use proptest::prelude::*;

use super::{SplitMix64, interior_unit_f32, splitmix64, unit_f64};

/// Classic state-incrementing SplitMix64, as previously copied into the speculative decoders.
struct SequentialReference {
    /// Wrapping Weyl-sequence state, advanced before every draw.
    state: u64,
}

impl SequentialReference {
    /// Returns the next mixed value and advances the Weyl state.
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

/// The common SplitMix64 reference vector for seed 1234567 is reproduced exactly.
#[test]
fn splitmix64_matches_reference_vector() {
    // First outputs of the reference `splitmix64.c` algorithm for x = 1234567,
    // cross-checked with an independent arbitrary-precision Python evaluation.
    let expected = [
        6_457_827_717_110_365_317_u64,
        3_203_168_211_198_807_973,
        9_817_491_932_198_370_423,
        4_593_380_528_125_082_431,
        16_408_922_859_458_223_821,
    ];
    let mut stream = SplitMix64::new(1_234_567);
    for (index, value) in expected.into_iter().enumerate() {
        assert_eq!(splitmix64(1_234_567, index as u64), value);
        assert_eq!(stream.next_u64(), value);
    }
    assert_eq!(stream.position(), 5);
}

/// Uniform conversions stay inside their documented half-open or open intervals.
#[test]
fn unit_conversions_respect_interval_boundaries() {
    assert_eq!(interior_unit_f32(0), 2.0_f32.powi(-24));
    assert_eq!(interior_unit_f32(u64::MAX), 1.0 - 2.0_f32.powi(-24));
    assert_eq!(unit_f64(0), 0.0);
    assert_eq!(unit_f64(u64::MAX), 1.0 - 2.0_f64.powi(-53));
    // The largest uniform times a positive total stays strictly below it.
    for total in [1.0_f64, 0.75, 1e-300, f64::MAX] {
        assert!(unit_f64(u64::MAX) * total < total);
    }
}

/// Wrapping at the end of the counter domain matches the sequential generator.
#[test]
fn counter_wraps_like_the_sequential_state() {
    let mut stream = SplitMix64::new(7);
    stream.position = u64::MAX;
    let last = stream.next_u64();
    assert_eq!(stream.position(), 0);
    assert_eq!(last, splitmix64(7, u64::MAX));
    // Draw 2^64 - 1 lands back on the seed's state, whose mixed value is splitmix64(seed - gamma, 0).
    assert_eq!(
        last,
        splitmix64(7_u64.wrapping_sub(0x9E37_79B9_7F4A_7C15), 0)
    );
}

proptest! {
    /// Counter-based draws are bit-identical to the sequential implementation they replace.
    #[test]
    fn counter_stream_equals_sequential_stream(seed in any::<u64>(), draws in 1_usize..64) {
        let mut reference = SequentialReference { state: seed };
        let mut stream = SplitMix64::new(seed);
        for index in 0..draws {
            let expected = reference.next();
            prop_assert_eq!(splitmix64(seed, index as u64), expected);
            prop_assert_eq!(stream.next_u64(), expected);
        }
    }

    /// Every F32 draw is interior and every F64 draw lies in `[0, 1)`.
    #[test]
    fn every_draw_is_a_valid_uniform(bits in any::<u64>()) {
        let interior = interior_unit_f32(bits);
        prop_assert!(interior > 0.0 && interior < 1.0);
        let unit = unit_f64(bits);
        prop_assert!((0.0..1.0).contains(&unit));
    }
}
