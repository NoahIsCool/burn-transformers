//! Configuration tests for Muse schedules, rotary parameters, and local-window requirements.

use super::*;

/// Default layers repeat three theta-500,000 local blocks followed by a zero-theta global block.
#[test]
fn default_schedule_is_three_rotary_local_layers_then_one_nope_global_layer() {
    let config = MuseGlimmerTextConfig::default();
    let schedule = config.resolved_layer_types();
    let theta = config.resolved_layer_rope_theta();
    assert_eq!(schedule.len(), 52);
    assert_eq!(theta.len(), 52);
    for (index, (layer_type, theta)) in schedule.into_iter().zip(theta).enumerate() {
        if index % 4 == 3 {
            assert_eq!(layer_type, MuseGlimmerLayerType::FullAttention);
            assert_eq!(theta, 0.0);
        } else {
            assert_eq!(layer_type, MuseGlimmerLayerType::SlidingAttention);
            assert_eq!(theta, 500_000.0);
        }
    }
}

/// Explicit head dimensions and full-only schedules remain independent of a window.
#[test]
fn explicit_schedule_and_head_width_allow_architecture_experiments() {
    let config = MuseGlimmerTextConfig {
        hidden_size: 10,
        num_hidden_layers: 4,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        head_dim: 4,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::FullAttention; 4],
        layer_rope_theta: vec![500_000.0; 4],
        ..MuseGlimmerTextConfig::default()
    };
    config.validate().unwrap();
    assert_eq!(config.resolved_sliding_window(), None);

    let unused_window = MuseGlimmerTextConfig {
        sliding_window: Some(0),
        ..config
    };
    unused_window.validate().unwrap();
    assert_eq!(unused_window.resolved_sliding_window(), None);
}

/// A local layer still requires a nonzero cache and mask window.
#[test]
fn sliding_schedule_requires_a_valid_window() {
    let config = MuseGlimmerTextConfig {
        num_hidden_layers: 1,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::SlidingAttention],
        layer_rope_theta: vec![500_000.0],
        ..MuseGlimmerTextConfig::default()
    };
    assert!(config.validate().is_err());

    let config = MuseGlimmerTextConfig {
        sliding_window: Some(0),
        ..config
    };
    assert!(config.validate().is_err());
}

/// Negative per-layer theta is rejected even when the schedule length is valid.
#[test]
fn rejects_a_negative_per_layer_rope_theta() {
    let config = MuseGlimmerTextConfig {
        num_hidden_layers: 4,
        layer_rope_theta: vec![500_000.0, 500_000.0, 500_000.0, -1.0],
        ..MuseGlimmerTextConfig::default()
    };
    assert!(config.validate().is_err());
}
