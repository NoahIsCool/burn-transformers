//! The shared dense-decoder interface: rotary schedules, per-call tables, and block points.

use burn::tensor::{Int, Tensor, TensorData};

use super::super::test_support::{B, host};
use super::*;
use crate::{
    intervention::Stack,
    position::{DynamicNtkScaling, LinearRopeScaling, RopeConfig, RopeScaling},
};

/// Returns an F32 default-RoPE set of head width 4 with base `theta`.
fn rope(theta: f64) -> RopeConfig {
    RopeConfig::new(4, 4, theta, RopeScaling::None).expect("valid RoPE")
}

/// `[1, 3]` positions `0, 1, 2`.
fn positions() -> Tensor<B, 2, Int> {
    Tensor::from_data(
        TensorData::new(vec![0_i64, 1, 2], [1, 3]),
        &Default::default(),
    )
}

/// Labels exercised by the field helper.
const LABELS: GraphLabels = GraphLabels {
    family: "Test",
    prefix: "test",
    inputs: InputNames {
        token_ids: "test.input_ids",
        token_mask: "test.token_mask",
        position_ids: "test.position_ids",
    },
    parameters: "test parameters",
    state: "test state",
};

/// Error fields carry the family prefix.
#[test]
fn labels_prefix_their_fields() {
    assert_eq!(LABELS.field("execution.hidden"), "test.execution.hidden");
    let error = missing_rotary_tables(&LABELS);
    assert!(error.to_string().contains("test.rotary"), "{error}");
}

/// Each layer reads its own set; NoPE layers read none; unused sets are never evaluated.
#[test]
fn tables_follow_the_per_layer_schedule() {
    let schedule = RotarySchedule {
        sets: vec![rope(10.0), rope(10_000.0), rope(77.0)],
        layers: vec![Some(0), Some(1), None, Some(0)],
        precision: RotaryPrecision::ActivationDtype,
    };
    assert_eq!(schedule.used_sets(), [true, true, false]);
    let tables = rotary_tables::<B>(schedule, DType::F32, positions(), 3).unwrap();
    assert!(tables.for_layer(2).is_none(), "NoPE layer");
    assert!(tables.for_layer(9).is_none(), "out-of-range layer");
    assert!(tables.tables[2].is_none(), "unused set is not evaluated");
    let local = host(tables.for_layer(0).unwrap().sine.clone());
    let global = host(tables.for_layer(1).unwrap().sine.clone());
    assert_eq!(local, host(tables.for_layer(3).unwrap().sine.clone()));
    // Position 1, pair 1 rotates by theta^(-1/2): a different base gives a different sine.
    assert!((local[5] - (10.0_f32).powf(-0.5).sin()).abs() < 1e-6);
    assert!((global[5] - (10_000.0_f32).powf(-0.5).sin()).abs() < 1e-6);
}

/// Tables are converted once to the rotation dtype.
#[test]
fn tables_use_the_rotation_dtype() {
    let schedule = |precision| RotarySchedule {
        sets: vec![rope(10_000.0)],
        layers: vec![Some(0)],
        precision,
    };
    let activation = rotary_tables::<B>(
        schedule(RotaryPrecision::ActivationDtype),
        DType::F16,
        positions(),
        3,
    )
    .unwrap();
    assert_eq!(activation.for_layer(0).unwrap().cosine.dtype(), DType::F16);
    let float = rotary_tables::<B>(
        schedule(RotaryPrecision::Float32),
        DType::F16,
        positions(),
        3,
    )
    .unwrap();
    assert_eq!(float.for_layer(0).unwrap().cosine.dtype(), DType::F32);
}

/// The tightest context limit of a *used* context-dependent set bounds cached decoding.
#[test]
fn cached_context_limits_come_from_used_sets_only() {
    let dynamic = |original| {
        let scaling = DynamicNtkScaling::new(2.0, original).unwrap().into();
        RopeConfig::new(4, 4, 10_000.0, scaling).unwrap()
    };
    let scaling = LinearRopeScaling::new(8.0).unwrap().into();
    let linear = RopeConfig::new(4, 4, 10_000.0, scaling).unwrap();
    let schedule = RotarySchedule {
        sets: vec![dynamic(16), linear, dynamic(8)],
        layers: vec![Some(0), Some(1)],
        precision: RotaryPrecision::ActivationDtype,
    };
    assert_eq!(
        schedule.cached_context_limit().map(|(limit, _)| limit),
        Some(16)
    );
    let unlimited = RotarySchedule {
        sets: vec![linear, dynamic(8)],
        layers: vec![Some(0), None],
        precision: RotaryPrecision::ActivationDtype,
    };
    assert!(unlimited.cached_context_limit().is_none());
}

/// Block points follow emission order, with score and pattern points only when eager.
#[test]
fn block_points_list_every_emission_in_order() {
    let fused = decoder_block_points(2, false);
    let eager = decoder_block_points(2, true);
    assert_eq!(fused.len(), 12);
    assert_eq!(eager.len(), 14);
    let layer = Layer::decoder(2);
    assert_eq!(fused[0], layer.residual_pre().info());
    assert_eq!(fused[11], layer.residual_post().info());
    assert_eq!(eager[5], layer.self_attention().scores().info());
    assert_eq!(eager[6], layer.self_attention().pattern().info());
    for hidden in HIDDEN_POINTS {
        assert!(fused.contains(&hidden(layer).info()), "{:?}", hidden(layer));
    }
    assert!(!fused.contains(&Stack::Decoder.embedding().info()));
}
