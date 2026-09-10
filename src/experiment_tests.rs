use super::*;
use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData},
};

type B = Flex<f32, i32>;

fn context() -> ExecutionContext<B> {
    ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &Default::default()),
        None,
    )
    .unwrap()
}

#[test]
fn policy_broadcasts_allowed_links_and_intersects_padding() {
    let device = Default::default();
    // Dynamic Boolean import intentionally exercises backend-native conversion.
    let allowed = Tensor::<B, 4, Bool>::from_data(
        TensorData::new(vec![true, false, true], [1, 1, 1, 3]),
        &device,
    );
    let padding = Tensor::<B, 4, Int>::from_data(
        TensorData::new(vec![0, 0, 1, 0, 1, 0], [1, 1, 2, 3]),
        &device,
    )
    .bool();
    let policy = AttentionPolicy::causal().with_allowed(allowed);
    let mask = policy
        .exclusion_mask(Some(padding), [1, 2, 2, 3], &device)
        .unwrap()
        .unwrap();
    assert_eq!(mask.dims(), [1, 2, 2, 3]);
    assert_eq!(
        mask.int().into_data().to_vec::<i32>().unwrap(),
        vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    );
    assert!(policy.causal);
}

#[test]
fn policy_bias_broadcast_and_invalid_extents_are_explicit() {
    let device = Default::default();
    let policy =
        AttentionPolicy::<B>::bidirectional().with_bias(Tensor::full([1, 1, 1, 1], 0.5, &device));
    let bias = policy
        .expanded_bias([2, 3, 4, 5], &device)
        .unwrap()
        .unwrap();
    assert_eq!(bias.dims(), [2, 3, 4, 5]);
    assert!(
        bias.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|x| *x == 0.5)
    );
    let invalid =
        AttentionPolicy::<B>::custom(Tensor::<B, 4, Int>::zeros([1, 2, 1, 1], &device).bool());
    assert!(
        invalid
            .exclusion_mask(None, [1, 3, 4, 5], &device)
            .unwrap_err()
            .to_string()
            .contains("attention.allowed")
    );
    assert!(policy.validate([0, 1, 1, 1], &device).is_err());
}

#[test]
fn layer_overrides_do_not_change_other_layers_or_cross_attention() {
    let layer = Layer::decoder(1);
    let context = context()
        .with_attention(AttentionPolicy::bidirectional())
        .with_layer_attention(layer, AttentionPolicy::causal())
        .with_cross_attention(AttentionPolicy::bidirectional())
        .with_layer_cross_attention(layer, AttentionPolicy::causal());
    assert!(context.attention_for(layer, false).causal);
    assert!(!context.attention_for(Layer::decoder(0), true).causal);
    assert!(context.cross_attention_for(layer).causal);
    assert!(!context.cross_attention_for(Layer::decoder(0)).causal);
    let context = context.with_layer_attention(layer, AttentionPolicy::bidirectional());
    assert_eq!(context.layer_attention.len(), 1);
    assert!(!context.attention_for(layer, true).causal);
}

#[test]
fn state_rejects_skipped_repeated_and_normalized_layer_ranges() {
    let mut state =
        HiddenState::<B>::new(Tensor::ones([1, 3, 4], &Default::default()), context()).unwrap();
    assert!(state.validate_range(0..0, 3).is_ok());
    assert!(state.validate_range(0..2, 3).is_ok());
    state.next_layer = 2;
    assert!(state.validate_range(2..3, 3).is_ok());
    for range in [0..2, 1..3, 2..4] {
        assert!(state.validate_range(range, 3).is_err());
    }
    state.normalized = true;
    assert!(state.validate_range(2..3, 3).is_err());
}

#[test]
fn state_validation_rejects_mutated_width_positions_and_mask() {
    let device = Default::default();
    let mut state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context()).unwrap();
    assert!(state.validate(5).is_err());
    state.context.positions = Tensor::from_data([[0, 1, 3]], &device);
    assert!(
        state
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("positions")
    );
    state.context.positions = Tensor::from_data([[0, 1, 2]], &device);
    state.context.token_mask = Some(TokenMask::new(
        Tensor::<B, 2, Int>::ones([1, 2], &device).bool(),
    ));
    assert!(
        state
            .validate(4)
            .unwrap_err()
            .to_string()
            .contains("token_mask")
    );
    assert!(ExecutionContext::<B>::new(Tensor::from_data([[-1, 0]], &device), None).is_err());
}

#[test]
fn encoder_export_preserves_positions_validity_and_normalization_boundary() {
    let device = Default::default();
    let context = ExecutionContext::new(
        Tensor::<B, 2, Int>::from_data([[2, 3, 4]], &device),
        Some(TokenMask::new(
            Tensor::<B, 2, Int>::from_data([[1, 0, 1]], &device).bool(),
        )),
    )
    .unwrap();
    let mut state = HiddenState::<B>::new(Tensor::ones([1, 3, 4], &device), context).unwrap();
    assert!(state.clone().into_encoder_context().is_err());
    state.normalized = true;
    let output = state.clone().into_encoder_output().unwrap();
    let source = state.into_encoder_context().unwrap();
    assert_eq!(output.hidden_state.into_data(), source.hidden.into_data());
    assert_eq!(
        source.positions.into_data().to_vec::<i32>().unwrap(),
        vec![2, 3, 4]
    );
    assert_eq!(
        source
            .token_mask
            .unwrap()
            .into_tensor()
            .int()
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        vec![1, 0, 1]
    );
}

#[test]
fn source_positions_are_independent_of_target_extent_but_never_negative() {
    let device = Default::default();
    let mut context = context();
    context.encoder = Some(EncoderContext {
        hidden: Tensor::ones([1, 2, 4], &device),
        positions: Tensor::from_ints([[30, 31]], &device),
        token_mask: None,
    });
    assert!(context.validate([1, 3], &device).is_ok());
    context.encoder.as_mut().unwrap().positions = Tensor::from_ints([[-1, 0]], &device);
    assert!(
        context
            .validate([1, 3], &device)
            .unwrap_err()
            .to_string()
            .contains("execution.encoder.positions")
    );
}
