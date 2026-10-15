//! Consumer-level checks for stateless Qwen intervention and composition.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, backend::Backend},
};

use super::*;
use crate::LogitsSelection;

/// CPU Flex backend with F32 activations and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Trainable additions belong to the experiment's Module tree. A temporary
/// callback borrows that registered module only while running the backbone.
#[cfg(feature = "autodiff")]
#[derive(Module, Debug)]
struct AdaptedQwen<B: Backend> {
    /// Registered hybrid language model whose residual stream receives the intervention.
    backbone: Qwen3_5ForCausalLm<B>,
    /// Registered trainable map applied at the second block residual output.
    adapter: Linear<B>,
}

#[cfg(feature = "autodiff")]
impl<B: Backend> AdaptedQwen<B> {
    /// Registers a tiny hybrid backbone and bias-free 8-by-8 adapter in one Burn module tree.
    fn init(device: &B::Device) -> Self {
        Self {
            backbone: Qwen3_5ForCausalLm::init(tiny_config(), device).expect("hybrid backbone"),
            adapter: operations::init_linear(
                &LinearConfig::new(8, 8)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            ),
        }
    }

    /// Applies the registered adapter to the second block residual through a temporary hook.
    fn forward(&self, input: CausalLmInput<B>) -> Result<CausalLmOutput<B>> {
        let mut hooks = Hooks::new();
        hooks.try_edit(Layer::decoder(1).residual_post(), |hidden| {
            operations::linear(hidden, &self.adapter)
        });
        self.backbone
            .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
    }
}

/// Builds a two-block DeltaNet/full-attention hybrid with no auxiliary MTP graph.
fn tiny_config() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_conv_kernel_dim: 3,
        partial_rotary_factor: Some(1.0),
        rope_theta: Some(10_000.0),
        mtp_num_hidden_layers: 0,
        ..Default::default()
    }
}

/// Builds one unmasked three-token prefix with implicitly derived positions.
fn input<B: Backend>(device: &B::Device) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], device),
        token_mask: None,
        position_ids: None,
    }
}

/// Materializes an F32 tensor for exact output and gradient comparisons.
fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("F32 tensor")
}

/// Embedding, public block calls, and final normalization exactly reproduce the headless backbone.
#[test]
fn public_blocks_reproduce_the_backbone_without_a_vocabulary_head() {
    let device = Default::default();
    TestBackend::seed(&device, 42);
    let model = Qwen3_5TextModel::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny hybrid backbone");
    let input = input(&device);
    let (expected, _) = model.forward(&input, false).expect("backbone");
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device);
    let mut hidden = model.embed_tokens.forward(input.token_ids);
    for block in &model.layers {
        hidden = block
            .forward(hidden, positions.clone(), None, model.rope, 3)
            .expect("public block");
    }
    let actual = model.norm.forward(hidden);
    assert_eq!(values(actual), values(expected));
}

/// An empty hook set leaves every selected logit and retained hidden-state value unchanged.
#[test]
fn no_hooks_preserve_hybrid_logits_and_retained_hidden_states() {
    let device = Default::default();
    TestBackend::seed(&device, 43);
    let model = Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny hybrid language model");
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        hidden_states: true,
        ..Default::default()
    };
    let expected = model.forward(input(&device), &options).expect("ordinary");
    let actual = model
        .forward_with_hooks(input(&device), &options, &mut Hooks::new())
        .expect("hooked");
    assert_eq!(values(actual.logits), values(expected.logits));
    assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
    for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
        assert_eq!(values(actual), values(expected));
    }
}

/// A selected MLP slice changes logits when ablated and restores them when clean values are patched.
#[test]
fn selected_mlp_ablation_changes_logits_and_clean_patching_restores_them() {
    let device = Default::default();
    TestBackend::seed(&device, 45);
    let model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let selection = Layer::decoder(0).mlp_activation().token(1).channels(0..3);
    let mut clean_hooks = Hooks::new();
    let saved = clean_hooks.capture(selection.clone());
    let clean = model
        .forward_with_hooks(input(&device), &options, &mut clean_hooks)
        .expect("capture clean activation");
    let activation = clean_hooks.get(&saved).expect("saved activation");
    assert_eq!(activation.dims(), [1, 1, 3]);

    let mut ablation = Hooks::new();
    ablation.edit(selection.clone(), |tensor| tensor * 0.0);
    let ablated = model
        .forward_with_hooks(input(&device), &options, &mut ablation)
        .expect("ablate selected neurons");
    assert_ne!(values(clean.logits.clone()), values(ablated.logits));

    // Edits are ordered: restore this same slice after the ablation.
    ablation.replace(selection, activation);
    let restored = model
        .forward_with_hooks(input(&device), &options, &mut ablation)
        .expect("patch clean neurons");
    assert_eq!(values(clean.logits), values(restored.logits));
}

/// Zeroing the gated activation removes the MLP contribution while preserving the residual path.
#[test]
fn gated_mlp_activation_is_before_content_multiplication_and_residual_addition() {
    let device = Default::default();
    let model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    let layer = Layer::decoder(0);
    let mut hooks = Hooks::new();
    let residual = hooks.capture(layer.residual_pre());
    let attention = hooks.capture(layer.attention_output());
    hooks.edit(layer.mlp_activation(), |tensor| tensor * 0.0);
    let mlp = hooks.capture(layer.mlp_output());
    let output = hooks.capture(layer.residual_post());
    model
        .forward_with_hooks(input(&device), &ForwardOptions::default(), &mut hooks)
        .expect("gated intervention");
    assert!(
        values(hooks.get(&mlp).expect("MLP"))
            .into_iter()
            .all(|x| x == 0.0)
    );
    assert_eq!(
        values(hooks.get(&output).expect("block output")),
        values(hooks.get(&residual).expect("residual") + hooks.get(&attention).expect("attention"))
    );
}

/// Attention captures preserve logits and expose native GQA head counts before repetition.
#[test]
fn full_attention_capture_preserves_logits_and_native_key_value_heads() {
    let device = Default::default();
    let model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    let options = ForwardOptions::default();
    let expected = model.forward(input(&device), &options).expect("ordinary");
    let points = Layer::decoder(1).self_attention();
    let mut hooks = Hooks::new();
    let query = hooks.capture(points.query());
    let key = hooks.capture(points.key());
    let value = hooks.capture(points.value());
    let context = hooks.capture(points.head_context().head(1));
    let actual = model
        .forward_with_hooks(input(&device), &options, &mut hooks)
        .expect("native attention captures");
    assert_eq!(values(actual.logits), values(expected.logits));
    assert_eq!(hooks.get(&query).expect("query").dims(), [1, 2, 3, 4]);
    assert_eq!(hooks.get(&key).expect("key").dims(), [1, 1, 3, 4]);
    assert_eq!(hooks.get(&value).expect("value").dims(), [1, 1, 3, 4]);
    assert_eq!(
        hooks.get(&context).expect("head context").dims(),
        [1, 1, 3, 4]
    );
}

/// Requesting an unavailable hybrid activation point fails before any edit callback executes.
#[test]
fn unavailable_hybrid_points_fail_before_callbacks_run() {
    use std::cell::Cell;

    let device = Default::default();
    let model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    // DeltaNet has no softmax-head context; native full attention does not
    // advertise materialized probability matrices until eager mode is chosen.
    for unsupported in [
        Layer::decoder(0).self_attention().head_context(),
        Layer::decoder(1).self_attention().pattern(),
        Layer::decoder(5).self_attention().query(),
    ] {
        let visits = Cell::new(0);
        let mut hooks = Hooks::new();
        hooks.edit(Stack::Decoder.embedding(), |tensor| {
            visits.set(visits.get() + 1);
            tensor
        });
        hooks.capture(unsupported.clone());
        let error = model
            .forward_with_hooks(input(&device), &ForwardOptions::default(), &mut hooks)
            .expect_err("unsupported point");
        assert!(error.to_string().contains(&unsupported.info().name));
        assert_eq!(visits.get(), 0);
    }
}

/// Captured eager attention has zero future-token probabilities and preserves ordinary logits.
#[test]
fn eager_pattern_capture_preserves_causal_qwen_attention() {
    let device = Default::default();
    let mut model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    model.model.layers[1]
        .self_attn
        .as_mut()
        .expect("full attention")
        .precision_mode = AttentionPrecisionMode::EagerFp32Softmax;
    let options = ForwardOptions::default();
    let expected = model
        .forward(input(&device), &options)
        .expect("eager model");
    let mut hooks = Hooks::new();
    let pattern = hooks.capture(Layer::decoder(1).self_attention().pattern());
    let actual = model
        .forward_with_hooks(input(&device), &options, &mut hooks)
        .expect("eager attention capture");
    assert_eq!(values(actual.logits), values(expected.logits));
    let pattern = hooks.get(&pattern).expect("causal patterns");
    assert_eq!(pattern.dims(), [1, 2, 3, 3]);
    for head in values(pattern).chunks(9) {
        assert_eq!(head[1], 0.0);
        assert_eq!(head[2], 0.0);
        assert_eq!(head[5], 0.0);
    }
}

/// An out-of-range vocabulary-position request clears partial captures from the failed run.
#[test]
fn failed_vocabulary_selection_discards_partial_captures() {
    let device = Default::default();
    let model =
        Qwen3_5ForCausalLm::<TestBackend>::init(tiny_config(), &device).expect("tiny hybrid model");
    let mut hooks = Hooks::new();
    let capture = hooks.capture(Stack::Decoder.embedding());
    let options = ForwardOptions {
        logits: LogitsSelection::Positions(vec![9]),
        ..Default::default()
    };
    assert!(
        model
            .forward_with_hooks(input(&device), &options, &mut hooks)
            .is_err()
    );
    assert!(hooks.get(&capture).is_err());
}

/// Public blocks report contextual errors for a wrong residual width or position sequence length.
#[test]
fn public_block_rejects_incompatible_hidden_and_position_shapes() {
    let device = Default::default();
    let model = Qwen3_5TextModel::<TestBackend>::init(tiny_config(), &device)
        .expect("tiny hybrid backbone");
    let block = &model.layers[0];
    let positions = Tensor::<TestBackend, 2, Int>::from_ints([[0, 1, 2]], &device);
    let error = block
        .forward(
            Tensor::zeros([1, 3, 7], &device),
            positions,
            None,
            model.rope,
            3,
        )
        .expect_err("incorrect width is recoverable");
    assert!(error.to_string().contains("qwen.block.hidden"));
    let error = block
        .forward(
            Tensor::zeros([1, 3, 8], &device),
            Tensor::<TestBackend, 2, Int>::from_ints([[0, 1]], &device),
            None,
            model.rope,
            3,
        )
        .expect_err("incorrect positions are recoverable");
    assert!(error.to_string().contains("qwen.block.positions"));
}

/// Empty hooks preserve exact embedding and MLP gate gradients through the hybrid graph.
#[cfg(feature = "autodiff")]
#[test]
fn no_hooks_preserve_hybrid_parameter_gradients() {
    use burn::backend::Autodiff;

    /// Autodiff wrapper for comparing gradients of registered model parameters.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    AdBackend::seed(&device, 44);
    let model = Qwen3_5ForCausalLm::<AdBackend>::init(tiny_config(), &device)
        .expect("differentiable hybrid model");
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let ordinary = model
        .forward(input(&device), &options)
        .expect("ordinary")
        .logits
        .square()
        .mean()
        .backward();
    let hooked = model
        .forward_with_hooks(input(&device), &options, &mut Hooks::new())
        .expect("hooked")
        .logits
        .square()
        .mean()
        .backward();
    assert_eq!(
        values(
            model
                .model
                .embed_tokens
                .weight
                .grad(&ordinary)
                .expect("ordinary embedding")
        ),
        values(
            model
                .model
                .embed_tokens
                .weight
                .grad(&hooked)
                .expect("hooked embedding")
        )
    );
    for block in &model.model.layers {
        assert_eq!(
            values(
                block
                    .mlp
                    .gate_proj
                    .weight
                    .grad(&ordinary)
                    .expect("ordinary gate")
            ),
            values(
                block
                    .mlp
                    .gate_proj
                    .weight
                    .grad(&hooked)
                    .expect("hooked gate")
            )
        );
    }
}

/// A registered adapter has finite nonzero gradients and round-trips with the backbone record.
#[cfg(feature = "autodiff")]
#[test]
fn registered_adapter_participates_in_gradients_and_record_roundtrip() {
    use burn::{
        backend::Autodiff,
        optim::GradientsParams,
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };

    /// Autodiff wrapper for comparing gradients of registered model parameters.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    AdBackend::seed(&device, 46);
    let experiment = AdaptedQwen::<AdBackend>::init(&device);
    let output = experiment
        .forward(input(&device))
        .expect("insert registered adapter");
    let gradients = GradientsParams::from_grads(
        output.logits.clone().square().mean().backward(),
        &experiment,
    );
    for (name, id) in [
        ("adapter", experiment.adapter.weight.id),
        (
            "embedding",
            experiment.backbone.model.embed_tokens.weight.id,
        ),
    ] {
        let gradient = values(
            gradients
                .get::<TestBackend, 2>(id)
                .unwrap_or_else(|| panic!("{name} absent from Module gradients")),
        );
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient.iter().any(|value| *value != 0.0));
    }

    // Serialization crosses the tensor-data boundary; records should not carry
    // a live autodiff graph into newly initialized parameters.
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(experiment.clone().into_record(), ())
        .expect("serialize complete experiment");
    let restored = AdaptedQwen::<AdBackend>::init(&device).load_record(
        recorder
            .load(bytes, &device)
            .expect("deserialize complete experiment"),
    );
    assert_eq!(
        values(output.logits.inner()),
        values(
            restored
                .forward(input(&device))
                .expect("restored experiment")
                .logits
                .inner()
        )
    );
    assert_eq!(
        experiment.num_params(),
        experiment.backbone.num_params() + 64
    );
}
