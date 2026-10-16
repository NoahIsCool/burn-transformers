//! DeepSeek-V2 schedule, routing, and latent-geometry tests against reference definitions.

use proptest::prelude::*;

use super::{
    LatentBytes, LatentFault, any_sparse_layer, eligible_experts, latent_cache_bytes, sparse_layer,
};

/// The reference checked-product chain of the latent cache preflight.
///
/// The per-token bytes are the exact `u128` product saturated to `usize`.
fn reference_bytes(
    batch: usize,
    capacity: usize,
    width: usize,
    layers: usize,
    scalar: usize,
) -> Result<LatentBytes, LatentFault> {
    let per_layer = batch
        .checked_mul(capacity)
        .and_then(|value| value.checked_mul(width))
        .filter(|&elements| elements <= i32::MAX as usize)
        .ok_or(LatentFault::LayerTooLarge)?;
    let total = per_layer
        .checked_mul(layers)
        .and_then(|elements| elements.checked_mul(scalar))
        .ok_or(LatentFault::BytesOverflow)?;
    let exact = (width as u128)
        .checked_mul(layers as u128)
        .and_then(|scalars| scalars.checked_mul(scalar as u128));
    let per_token = exact.map_or(usize::MAX, |bytes| {
        usize::try_from(bytes).unwrap_or(usize::MAX)
    });
    Ok(LatentBytes { total, per_token })
}

/// Released DeepSeek-V2-Lite: one dense layer, then every layer sparse.
#[test]
fn sparse_schedule_follows_the_dense_prefix_and_frequency() {
    let schedule = (0..5)
        .map(|index| sparse_layer(index, 1, 1))
        .collect::<Vec<_>>();
    assert_eq!(schedule, [false, true, true, true, true]);
    let schedule = (0..6)
        .map(|index| sparse_layer(index, 1, 2))
        .collect::<Vec<_>>();
    assert_eq!(schedule, [false, false, true, false, true, false]);
    assert!(!sparse_layer(4, 0, 0));
    assert!(any_sparse_layer(2, 1, 1));
    assert!(!any_sparse_layer(1, 1, 1));
    assert!(!any_sparse_layer(3, 1, 3));
    assert!(any_sparse_layer(4, 1, 3));
    assert!(!any_sparse_layer(4, 0, 0));
}

/// Eligible experts and latent bytes match their definitions, including overflow.
#[test]
fn routing_and_latent_arithmetic_report_overflow() {
    assert_eq!(eligible_experts(64, 8, 3), Some(24));
    assert_eq!(eligible_experts(7, 2, 1), Some(3));
    assert_eq!(eligible_experts(usize::MAX, 1, 2), None);
    // DeepSeek-V2-Lite: rank 512 + rope 64, 27 layers of BF16 latents.
    assert_eq!(
        latent_cache_bytes(2, 16, 576, 27, 2),
        Ok(LatentBytes {
            total: 2 * 16 * 576 * 27 * 2,
            per_token: 576 * 27 * 2,
        })
    );
    // An empty batch allocates nothing; its per-token bytes saturate, or are
    // exactly zero for zero-byte scalars.
    assert_eq!(
        latent_cache_bytes(0, 16, usize::MAX, 2, 2),
        Ok(LatentBytes {
            total: 0,
            per_token: usize::MAX,
        })
    );
    assert_eq!(
        latent_cache_bytes(1, 0, usize::MAX, 2, 0),
        Ok(LatentBytes {
            total: 0,
            per_token: 0,
        })
    );
    assert_eq!(
        latent_cache_bytes(0, 1, 1 << 40, 1 << 30, 1),
        Ok(LatentBytes {
            total: 0,
            per_token: usize::MAX,
        })
    );
    assert_eq!(
        latent_cache_bytes(1, 1 << 31, 1, 1, 1),
        Err(LatentFault::LayerTooLarge)
    );
    assert_eq!(
        latent_cache_bytes(usize::MAX, 2, 0, 1, 1),
        Err(LatentFault::LayerTooLarge)
    );
    assert_eq!(
        latent_cache_bytes(1, 1, 1, usize::MAX, 2),
        Err(LatentFault::BytesOverflow)
    );
}

proptest! {
    /// Schedule existence equals the iterator definition over every layer.
    #[test]
    fn schedule_existence_matches_the_iterator_definition(
        layers in 0usize..20,
        first_dense in 0usize..8,
        frequency in 0usize..5,
    ) {
        let reference = (0..layers).any(|index| {
            frequency != 0 && index >= first_dense && index.is_multiple_of(frequency)
        });
        prop_assert_eq!(any_sparse_layer(layers, first_dense, frequency), reference);
    }

    /// The latent preflight equals the reference checked-product chain.
    #[test]
    fn latent_bytes_match_the_checked_chain(
        batch in prop_oneof![0usize..4, Just(usize::MAX)],
        capacity in prop_oneof![0usize..64, Just(1usize << 30)],
        width in prop_oneof![0usize..1_024, Just(usize::MAX / 3)],
        layers in prop_oneof![0usize..64, Just(usize::MAX / 2)],
        scalar in 0usize..9,
    ) {
        prop_assert_eq!(
            latent_cache_bytes(batch, capacity, width, layers, scalar),
            reference_bytes(batch, capacity, width, layers, scalar)
        );
    }
}
