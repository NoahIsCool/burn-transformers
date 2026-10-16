//! Normalization construction: both conventions build at any positive width, never at zero.

use super::*;
use crate::models::custom::test_support::{TestBackend, rms};

/// A zero residual width has no channel axis to normalize and is rejected for both conventions.
#[test]
fn zero_width_norms_are_rejected() {
    let device = Default::default();
    for spec in [
        rms(),
        NormalizationSpec::Layer {
            epsilon: 1e-5,
            bias: true,
        },
    ] {
        let error = ModularNorm::<TestBackend>::init(spec, 0, &device).unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, reason }
                if field == "custom.normalization.hidden_size" && reason == "must be non-zero"),
            "{error:?}"
        );
        ModularNorm::<TestBackend>::init(spec, 4, &device).expect("a positive width is valid");
    }
}
