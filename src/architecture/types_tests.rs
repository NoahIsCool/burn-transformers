use super::{CausalLmOutput, ForwardOptions, LogitsSelection};

#[cfg(feature = "flex")]
use burn::backend::Flex;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(any(feature = "flex", feature = "metal"))]
use burn::tensor::Tensor;
#[cfg(feature = "metal")]
use burn::tensor::{Bool, TensorData};

#[cfg(feature = "flex")]
type TestBackend = Flex<f32, i32>;

/// Boolean host extraction preserves logical values independently of physical storage.
#[cfg(feature = "flex")]
#[test]
fn boolean_metadata_uses_backend_portable_integer_conversion() {
    let device = Default::default();
    let tensor = super::bool_tensor_from_values::<TestBackend, 2>(
        vec![true, false, false, true],
        [2, 2],
        &device,
    );
    assert_eq!(
        super::tensor_bool_values(tensor.clone(), "test mask").unwrap(),
        [true, false, false, true]
    );
    assert_eq!(
        super::tensor_bool_values(tensor.bool_not(), "inverted test mask").unwrap(),
        [false, true, true, false]
    );
}

/// Boolean selection preserves a logical dtype accepted by later Flex Boolean operations.
#[cfg(feature = "flex")]
#[test]
fn boolean_selection_normalizes_backend_storage() {
    let device = Default::default();
    let tensor = super::bool_tensor_from_values::<TestBackend, 2>(
        vec![true, false, false, true],
        [2, 2],
        &device,
    );
    let selected = super::select_bool_tensor(
        tensor,
        0,
        Tensor::<TestBackend, 1, burn::tensor::Int>::from_data([1_i32, 0], &device),
    );
    assert_eq!(
        super::tensor_bool_values(selected.bool_not(), "selected inverted mask").unwrap(),
        [true, false, false, true]
    );
}

/// Metal's native Boolean readback remains portable through the integer conversion path.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a host-visible Metal adapter"]
fn metal_boolean_metadata_does_not_assume_u8_tensor_data() {
    let device = Default::default();
    let tensor = Tensor::<Metal, 2, Bool>::from_data(
        TensorData::new(vec![true, false, false, true], [2, 2]),
        &device,
    );
    assert_eq!(
        super::tensor_bool_values(tensor, "Metal test mask").unwrap(),
        [true, false, false, true]
    );
}

/// Ensures inference avoids a full vocabulary projection unless explicitly requested.
#[test]
fn forward_options_default_to_last_logits_only() {
    let options = ForwardOptions::default();
    assert_eq!(options.logits, LogitsSelection::Last);
    assert!(!options.hidden_states);
    assert!(!options.attentions);
    assert!(!options.router_logits);
    assert!(!options.auxiliary_losses);
}

/// The minimal output constructor initializes every optional diagnostic to absence.
#[cfg(feature = "flex")]
#[test]
fn logits_only_output_drops_no_uninitialized_diagnostics() {
    let device = Default::default();
    let output = CausalLmOutput::<TestBackend>::from_logits(Tensor::zeros([2, 1, 7], &device));

    assert_eq!(output.logits.dims(), [2, 1, 7]);
    assert!(output.final_hidden_state.is_none());
    assert!(output.hidden_states.is_empty());
    assert!(output.attentions.is_empty());
    assert!(output.routers.is_empty());
    assert!(output.load_balancing_loss.is_none());
}

/// Verifies an explicit position list survives serde round trips without ambiguity.
#[test]
fn logits_selection_round_trips_through_json() {
    let selection = LogitsSelection::Positions(vec![0, 3, 8]);
    let encoded = serde_json::to_string(&selection).expect("selection should serialize");
    let decoded: LogitsSelection =
        serde_json::from_str(&encoded).expect("selection should deserialize");
    assert_eq!(decoded, selection);
}
