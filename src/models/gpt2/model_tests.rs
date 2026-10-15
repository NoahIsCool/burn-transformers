//! Public GPT-2 graph and import tests; fixtures come from independent PyTorch execution.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde::Deserialize;

use super::*;
use crate::LogitsSelection;
use crate::checkpoint::LoadOptions;
use crate::models::gpt2::{Gpt2Activation, Gpt2Config};

/// Deterministic CPU backend using F32 activations and I32 token indices.
type B = Flex<f32, i32>;

/// Synthetic checkpoint geometry, masked inputs, and independent activation references.
#[derive(Deserialize)]
struct Fixture {
    /// Tiny architecture shared by the synthetic checkpoint and PyTorch reference.
    config: Gpt2Config,
    /// Checkpoint-native tensor shapes keyed by source parameter name.
    parameter_shapes: std::collections::BTreeMap<String, Vec<usize>>,
    /// Two equal-length token-ID rows used by the numerical oracle.
    input_ids: Vec<Vec<i32>>,
    /// Integer validity mask aligned with the input rows.
    token_mask: Vec<Vec<i32>>,
    /// Expected outputs keyed by exact or tanh GELU convention.
    references: std::collections::BTreeMap<String, Reference>,
}

/// Independent PyTorch outputs for one activation convention and input batch.
#[derive(Deserialize)]
struct Reference {
    /// Vocabulary scores shaped `[batch, token, vocabulary]`.
    logits: Vec<Vec<Vec<f32>>>,
    /// Ordered reference snapshots shaped `[snapshot, batch, token, channel]`.
    hidden_states: Vec<Vec<Vec<Vec<f32>>>>,
    /// Block MLP outputs keyed by layer, each shaped `[batch, token, channel]`.
    mlp_outputs: std::collections::BTreeMap<String, Vec<Vec<Vec<f32>>>>,
}

/// Parses the checked-in tiny reference generated independently of the Rust graph.
fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/tiny_reference.json")).unwrap()
}

/// Reconstruct compact synthetic data rather than checking in any weight file.
/// The data generator is independent of the transformer implementation.
fn synthetic_checkpoint() -> Vec<u8> {
    use safetensors::tensor::{Dtype, TensorView, serialize};
    let parameters: Vec<_> = fixture()
        .parameter_shapes
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape))| {
            let mut state = 20_260_910u64 ^ (index as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
            let is_scale =
                name.ends_with(".weight") && (name.contains(".ln_") || name.starts_with("ln_f."));
            let mut bytes = Vec::new();
            for _ in 0..shape.iter().product::<usize>() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let centered = (state >> 40) as i64 - (1 << 23);
                let divisor = if name.ends_with(".bias") {
                    1u64 << 32
                } else if name.ends_with("mlp.c_fc.weight") {
                    1u64 << 23
                } else {
                    1u64 << 29
                };
                let value = centered as f64 / divisor as f64 + if is_scale { 1.0 } else { 0.0 };
                bytes.extend_from_slice(&(value as f32).to_le_bytes());
            }
            (name, shape, bytes)
        })
        .collect();
    let views = parameters.iter().map(|(name, shape, bytes)| {
        (
            name,
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize(views, None).unwrap()
}

/// Flattens batch/token/channel reference arrays into contiguous comparison order.
fn flat(values: &[Vec<Vec<f32>>]) -> Vec<f32> {
    values.iter().flatten().flatten().copied().collect()
}

/// Materializes an F32 CPU tensor for exact or tolerance-based comparison.
fn values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec().unwrap()
}

/// Requires equal lengths, finite actual values, and a bounded maximum absolute error.
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        actual.iter().all(|x| x.is_finite()) && error <= tolerance,
        "maximum absolute error {error} > {tolerance}"
    );
}

/// Builds the reference two-row, four-token input with its semantic padding mask.
fn input(fixture: &Fixture) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                fixture.input_ids.iter().flatten().copied().collect(),
                [2, 4],
            ),
            &Default::default(),
        ),
        token_mask: Some(TokenMask::new(
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    fixture.token_mask.iter().flatten().copied().collect(),
                    [2, 4],
                ),
                &Default::default(),
            )
            .bool(),
        )),
        position_ids: None,
    }
}

/// Imports synthetic Conv1D weights and checks that the tied head has one alias.
fn loaded(config: Gpt2Config) -> Gpt2ForCausalLm<B> {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("model.safetensors"),
        synthetic_checkpoint(),
    )
    .unwrap();
    let mut model = Gpt2ForCausalLm::init(config, &Default::default()).unwrap();
    let outcome = model
        .load_checkpoint(root.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(outcome.stats.aliases_applied, 1);
    model
}

/// Both distinct GELU conventions reproduce PyTorch logits, hidden states, and MLP outputs.
#[test]
fn imported_conv1d_weights_match_pytorch_logits_and_mlp_boundaries_for_both_gelus() {
    let fixture = fixture();
    let exact = flat(&fixture.references["gelu"].logits);
    let approximate = flat(&fixture.references["gelu_new"].logits);
    assert!(
        exact
            .iter()
            .zip(&approximate)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "fixture must distinguish the two activation conventions"
    );
    for (name, activation) in [
        ("gelu_new", Gpt2Activation::GeluNew),
        ("gelu", Gpt2Activation::Gelu),
    ] {
        let model = loaded(Gpt2Config {
            activation_function: activation,
            ..fixture.config.clone()
        });
        let mut hooks = Hooks::new();
        let captures: Vec<_> = (0..2)
            .map(|i| hooks.capture(Layer::decoder(i).mlp_output()))
            .collect();
        let output = model
            .forward_with_hooks(
                input(&fixture),
                &ForwardOptions {
                    logits: LogitsSelection::All,
                    hidden_states: true,
                    ..Default::default()
                },
                &mut hooks,
            )
            .unwrap();
        let expected = &fixture.references[name];
        close(&values(output.logits), &flat(&expected.logits), 2e-6);
        // HF retains embedding, pre-last-block residuals, and final normalized states.
        for (actual, expected) in [
            (&output.hidden_states[0], &expected.hidden_states[0]),
            (&output.hidden_states[1], &expected.hidden_states[1]),
            (&output.hidden_states[3], &expected.hidden_states[2]),
        ] {
            close(&values(actual.clone()), &flat(expected), 3e-6);
        }
        for (i, capture) in captures.iter().enumerate() {
            close(
                &values(hooks.get(capture).unwrap()),
                &flat(&expected.mlp_outputs[&format!("mlp_{i}")]),
                2e-6,
            );
        }
    }
}

/// Captures preserve logits, ablations change them, replacement restores them, and residual readout agrees.
#[test]
fn captures_edits_restoration_and_direct_readout_use_actual_residuals() {
    let fixture = fixture();
    let model = loaded(fixture.config.clone());
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let baseline = model.forward(input(&fixture), &options).unwrap();
    let empty = model
        .forward_with_hooks(input(&fixture), &options, &mut Hooks::new())
        .unwrap();
    assert_eq!(values(baseline.logits.clone()), values(empty.logits));
    let selected = Layer::decoder(0).mlp_activation().token(2).channels(1..5);
    let mut capture = Hooks::new();
    let saved = capture.capture(selected.clone());
    let final_residual = capture.capture(Layer::decoder(1).residual_post());
    let observed = model
        .forward_with_hooks(input(&fixture), &options, &mut capture)
        .unwrap();
    assert_eq!(values(baseline.logits.clone()), values(observed.logits));
    assert_eq!(
        values(
            model
                .readout(capture.get(&final_residual).unwrap())
                .unwrap()
        ),
        values(baseline.logits.clone())
    );

    let mut edit = Hooks::new();
    edit.edit(selected.clone(), |x| x * 0.0);
    let ablated = model
        .forward_with_hooks(input(&fixture), &options, &mut edit)
        .unwrap();
    assert_ne!(values(ablated.logits), values(baseline.logits.clone()));
    edit.replace(selected, capture.get(&saved).unwrap());
    let restored = model
        .forward_with_hooks(input(&fixture), &options, &mut edit)
        .unwrap();
    assert_eq!(values(restored.logits), values(baseline.logits));
}

/// Manual block composition matches the backbone and disabling causality exposes a future key.
#[test]
fn public_blocks_reproduce_backbone_and_visibility_can_change_per_block() {
    let fixture = fixture();
    let mut model = loaded(fixture.config.clone());
    let input = input(&fixture);
    let expected = model.model.forward(&input, false).unwrap();
    let positions =
        Tensor::<B, 2, Int>::from_ints([[0, 1, 2, 3], [0, 1, 2, 3]], &Default::default());
    let mut hidden = model.model.token_embedding.forward(input.token_ids.clone())
        + model.model.position_embedding.forward(positions);
    for block in &model.model.layers {
        hidden = block.forward(hidden, input.token_mask.as_ref()).unwrap();
    }
    assert_eq!(
        values(model.model.normalize(hidden).unwrap()),
        values(expected.last_hidden_state)
    );

    let point = Layer::decoder(0).self_attention().pattern();
    let mut hooks = Hooks::new();
    let capture = hooks.capture(point.query_token(0).key_token(1));
    model
        .forward_with_hooks(input.clone(), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert!(
        values(hooks.get(&capture).unwrap())
            .into_iter()
            .all(|x| x == 0.0)
    );
    model.model.layers[0].causal = false;
    model
        .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert!(
        values(hooks.get(&capture).unwrap())
            .into_iter()
            .all(|x| x > 0.0)
    );
}

/// Last logits follow each row's valid-token mask; explicit positions preserve order and duplicates.
#[test]
fn last_and_ordered_logit_selection_respect_valid_tokens() {
    let fixture = fixture();
    let model = loaded(fixture.config.clone());
    let all = model
        .forward(
            input(&fixture),
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    let last = model
        .forward(input(&fixture), &ForwardOptions::default())
        .unwrap()
        .logits;
    assert_eq!(
        values(last),
        values(Tensor::cat(
            vec![
                all.clone().narrow(0, 0, 1).narrow(1, 3, 1),
                all.clone().narrow(0, 1, 1).narrow(1, 2, 1)
            ],
            0
        ))
    );
    let ordered = model
        .forward(
            input(&fixture),
            &ForwardOptions {
                logits: LogitsSelection::Positions(vec![2, 0, 2]),
                ..Default::default()
            },
        )
        .unwrap()
        .logits;
    assert_eq!(
        values(ordered),
        values(Tensor::cat(
            vec![
                all.clone().narrow(1, 2, 1),
                all.clone().narrow(1, 0, 1),
                all.narrow(1, 2, 1)
            ],
            1
        ))
    );
}

/// Bad selections, hook points, IDs, and readout widths return errors and invalidate stale captures.
#[test]
fn invalid_input_unknown_points_and_readout_failures_are_recoverable_and_clear_captures() {
    let fixture = fixture();
    let model = loaded(fixture.config.clone());
    let mut hooks = Hooks::new();
    let captured = hooks.capture(Stack::Decoder.embedding());
    model
        .forward_with_hooks(input(&fixture), &ForwardOptions::default(), &mut hooks)
        .unwrap();
    assert!(hooks.get(&captured).is_ok());
    let options = ForwardOptions {
        logits: LogitsSelection::Positions(vec![99]),
        ..Default::default()
    };
    assert!(
        model
            .forward_with_hooks(input(&fixture), &options, &mut hooks)
            .is_err()
    );
    assert!(hooks.get(&captured).is_err());
    hooks.capture(Layer::decoder(7).mlp_output());
    assert!(
        model
            .forward_with_hooks(input(&fixture), &ForwardOptions::default(), &mut hooks)
            .is_err()
    );
    for ids in [[[-1, 1, 2, 3], [1, 2, 3, 4]], [[19, 1, 2, 3], [1, 2, 3, 4]]] {
        let mut invalid = input(&fixture);
        invalid.token_ids = Tensor::<B, 2, Int>::from_ints(ids, &Default::default());
        assert!(model.forward(invalid, &ForwardOptions::default()).is_err());
    }
    let mut invalid = input(&fixture);
    invalid.position_ids = Some(Tensor::<B, 2, Int>::from_ints(
        [[0, 1, 2, 16], [0, 1, 2, 3]],
        &Default::default(),
    ));
    assert!(model.forward(invalid, &ForwardOptions::default()).is_err());
    assert!(
        model
            .readout(Tensor::zeros([1, 2, 13], &Default::default()))
            .is_err()
    );
}

/// Matching duplicate head bytes preserve reference logits; mismatches leave fresh parameters uninitialized.
#[test]
fn language_model_namespace_verifies_tied_head_bytes_before_installing_parameters() {
    use safetensors::{
        SafeTensors,
        tensor::{Dtype, TensorView, serialize},
    };
    let fixture = fixture();
    let bytes = synthetic_checkpoint();
    let source = SafeTensors::deserialize(&bytes).unwrap();
    let mut named: Vec<_> = source
        .tensors()
        .into_iter()
        .map(|(name, view)| (format!("transformer.{name}"), view))
        .collect();
    named.push((
        "lm_head.weight".into(),
        source.tensor("wte.weight").unwrap(),
    ));
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("model.safetensors"),
        serialize(named.clone(), None).unwrap(),
    )
    .unwrap();
    let mut model =
        Gpt2ForCausalLm::<B>::init(fixture.config.clone(), &Default::default()).unwrap();
    let result = model
        .load_checkpoint(root.path(), &LoadOptions::default())
        .unwrap();
    assert_eq!(result.stats.equivalences_verified, 1);
    close(
        &values(
            model
                .forward(
                    input(&fixture),
                    &ForwardOptions {
                        logits: LogitsSelection::All,
                        ..Default::default()
                    },
                )
                .unwrap()
                .logits,
        ),
        &flat(&fixture.references["gelu_new"].logits),
        2e-6,
    );

    let bad_data = vec![0u8; 19 * 12 * 4];
    *named.last_mut().unwrap() = (
        "lm_head.weight".into(),
        TensorView::new(Dtype::F32, vec![19, 12], &bad_data).unwrap(),
    );
    std::fs::write(
        root.path().join("model.safetensors"),
        serialize(named, None).unwrap(),
    )
    .unwrap();
    let mut fresh = Gpt2ForCausalLm::<B>::init(fixture.config, &Default::default()).unwrap();
    assert!(
        fresh
            .load_checkpoint(root.path(), &LoadOptions::default())
            .is_err()
    );
    assert!(!fresh.model.token_embedding.weight.is_initialized());
    assert!(
        !fresh.model.layers[0]
            .attention
            .q_proj
            .weight
            .is_initialized()
    );
}

/// Missing, unexpected, or wrong-shaped MLP inventory fails before embedding initialization.
#[test]
fn missing_unknown_and_wrong_shape_checkpoints_fail_before_parameter_initialization() {
    use safetensors::{SafeTensors, tensor::serialize};
    let fixture = fixture();
    let bytes = synthetic_checkpoint();
    let source = SafeTensors::deserialize(&bytes).unwrap();
    for mode in 0..3 {
        let mut named = source.tensors();
        match mode {
            0 => named.retain(|(name, _)| name != "h.0.mlp.c_fc.weight"),
            1 => named.push((
                "unknown.weight".into(),
                source.tensor("ln_f.weight").unwrap(),
            )),
            _ => {
                let entry = named
                    .iter_mut()
                    .find(|(name, _)| name == "h.0.mlp.c_fc.weight")
                    .unwrap();
                entry.1 = source.tensor("h.0.attn.c_proj.weight").unwrap();
            }
        }
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("model.safetensors"),
            serialize(named, None).unwrap(),
        )
        .unwrap();
        let mut model =
            Gpt2ForCausalLm::<B>::init(fixture.config.clone(), &Default::default()).unwrap();
        assert!(
            model
                .load_checkpoint(root.path(), &LoadOptions::default())
                .is_err()
        );
        assert!(!model.model.token_embedding.weight.is_initialized());
    }
}

/// Invalid head divisibility, MLP width, epsilon, attention ordering, and untied readout are rejected.
#[test]
fn invalid_architecture_conventions_are_rejected_explicitly() {
    let config = fixture().config;
    for invalid in [
        Gpt2Config {
            n_head: 5,
            ..config.clone()
        },
        Gpt2Config {
            n_inner: Some(0),
            ..config.clone()
        },
        Gpt2Config {
            layer_norm_epsilon: f64::NAN,
            ..config.clone()
        },
        Gpt2Config {
            reorder_and_upcast_attn: true,
            ..config.clone()
        },
        Gpt2Config {
            tie_word_embeddings: false,
            ..config
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

/// Wrong dtypes or norm-bias widths fail before callbacks and clear pending captures.
#[test]
fn public_normalization_and_blocks_reject_incompatible_parameter_contracts() {
    use burn::{module::Param, tensor::DType};
    let fixture = fixture();
    let mut model = loaded(fixture.config.clone());
    let hidden = Tensor::<B, 3>::ones([1, 2, 12], &Default::default());
    let wrong_dtype = hidden.clone().cast(DType::F64);
    assert!(model.model.normalize(wrong_dtype.clone()).is_err());
    assert!(model.readout(wrong_dtype.clone()).is_err());
    assert!(model.model.layers[0].forward(wrong_dtype, None).is_err());

    model.model.layers[0].mlp_norm.beta = Some(Param::from_tensor(Tensor::<B, 1>::zeros(
        [11],
        &Default::default(),
    )));
    assert!(model.model.layers[0].forward(hidden, None).is_err());
    let callbacks = std::cell::Cell::new(0);
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(Stack::Decoder.embedding());
    hooks.edit(Layer::decoder(0).residual_pre(), |hidden| {
        callbacks.set(callbacks.get() + 1);
        hidden
    });
    assert!(
        model
            .forward_with_hooks(input(&fixture), &ForwardOptions::default(), &mut hooks)
            .is_err()
    );
    assert_eq!(
        callbacks.get(),
        0,
        "block parameters must be checked before callbacks"
    );
    assert!(hooks.get(&embedding).is_err());
}

/// A sharded index may declare the tied head without storing duplicate bytes in either namespace.
#[test]
fn index_only_tied_alias_loads_both_checkpoint_namespaces() {
    use safetensors::{SafeTensors, tensor::serialize};
    let fixture = fixture();
    let bytes = synthetic_checkpoint();
    let source = SafeTensors::deserialize(&bytes).unwrap();
    for prefix in ["", "transformer."] {
        let root = tempfile::tempdir().unwrap();
        let named: Vec<_> = source
            .tensors()
            .into_iter()
            .map(|(name, tensor)| (format!("{prefix}{name}"), tensor))
            .collect();
        let mut weight_map: std::collections::BTreeMap<_, _> = named
            .iter()
            .map(|(name, _)| (name.clone(), "part.safetensors"))
            .collect();
        weight_map.insert("lm_head.weight".into(), "part.safetensors");
        std::fs::write(
            root.path().join("part.safetensors"),
            serialize(named, None).unwrap(),
        )
        .unwrap();
        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        let mut model =
            Gpt2ForCausalLm::<B>::init(fixture.config.clone(), &Default::default()).unwrap();
        let outcome = model
            .load_checkpoint(root.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(outcome.stats.aliases_applied, 1);
        close(
            &values(
                model
                    .forward(
                        input(&fixture),
                        &ForwardOptions {
                            logits: LogitsSelection::All,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .logits,
            ),
            &flat(&fixture.references["gelu_new"].logits),
            2e-6,
        );
    }
}

/// Empty and capture-only hooks preserve nonzero imported embedding, attention, and MLP gradients exactly.
#[cfg(feature = "autodiff")]
#[test]
fn empty_and_capture_only_hooks_preserve_imported_parameter_gradients() {
    use burn::{backend::Autodiff, optim::GradientsParams};
    /// Reverse-mode F32 backend for optimizer-visible parameter-gradient checks.
    type Ad = Autodiff<B>;
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("model.safetensors"),
        synthetic_checkpoint(),
    )
    .unwrap();
    let mut model = Gpt2ForCausalLm::<Ad>::init(fixture().config, &Default::default()).unwrap();
    model
        .load_checkpoint(root.path(), &LoadOptions::default())
        .unwrap();
    let input = CausalLmInput {
        token_ids: Tensor::<Ad, 2, Int>::from_ints([[1, 2, 3, 4]], &Default::default()),
        token_mask: None,
        position_ids: None,
    };
    let gradients = |output: CausalLmOutput<Ad>| {
        let grads = GradientsParams::from_grads(output.logits.square().mean().backward(), &model);
        [
            model.model.token_embedding.weight.id,
            model.model.layers[0].attention.q_proj.weight.id,
            model.model.layers[0].mlp.up_proj.weight.id,
        ]
        .map(|id| values(grads.get::<B, 2>(id).expect("imported parameter gradient")))
    };
    let baseline = gradients(
        model
            .forward(input.clone(), &ForwardOptions::default())
            .unwrap(),
    );
    let mut hooks = Hooks::new();
    let empty = gradients(
        model
            .forward_with_hooks(input.clone(), &ForwardOptions::default(), &mut hooks)
            .unwrap(),
    );
    let captured = hooks.capture(Layer::decoder(0).mlp_output());
    let observed = gradients(
        model
            .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
            .unwrap(),
    );
    assert!(hooks.get(&captured).is_ok());
    for ((baseline, empty), observed) in baseline.iter().zip(&empty).zip(&observed) {
        assert!(baseline.iter().all(|x| x.is_finite()) && baseline.iter().any(|x| *x != 0.0));
        close(empty, baseline, 0.0);
        close(observed, baseline, 0.0);
    }
}

/// A hook-inserted adapter and imported backbone all expose finite nonzero optimizer gradients.
#[cfg(feature = "autodiff")]
#[test]
fn imported_and_inserted_parameters_keep_optimizer_visible_gradients() {
    use burn::{backend::Autodiff, module::Module, nn::LinearConfig, optim::GradientsParams};
    /// Reverse-mode F32 backend for optimizer-visible parameter-gradient checks.
    type Ad = Autodiff<B>;
    /// Composite trainable module registering the imported GPT-2 backbone and inserted adapter.
    #[derive(Module, Debug)]
    struct Experiment<B: Backend> {
        /// Imported language model whose parameter IDs remain optimizer-visible.
        backbone: Gpt2ForCausalLm<B>,
        /// Trainable channel projection inserted at the first block's MLP output.
        adapter: Linear<B>,
    }
    let device = Default::default();
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("model.safetensors"),
        synthetic_checkpoint(),
    )
    .unwrap();
    let mut experiment = Experiment {
        backbone: Gpt2ForCausalLm::<Ad>::init(fixture().config, &device).unwrap(),
        adapter: LinearConfig::new(12, 12)
            .with_bias(false)
            .init::<Ad>(&device),
    };
    experiment
        .backbone
        .load_checkpoint(root.path(), &LoadOptions::default())
        .unwrap();
    let input = CausalLmInput {
        token_ids: Tensor::<Ad, 2, Int>::from_ints([[1, 2, 3, 4]], &device),
        token_mask: None,
        position_ids: None,
    };
    let mut hooks = Hooks::new();
    hooks.try_edit(Layer::decoder(0).mlp_output(), |hidden| {
        operations::linear(hidden, &experiment.adapter)
    });
    let output = experiment
        .backbone
        .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
        .unwrap();
    let grads = GradientsParams::from_grads(output.logits.square().mean().backward(), &experiment);
    for id in [
        experiment.adapter.weight.id,
        experiment.backbone.model.token_embedding.weight.id,
        experiment.backbone.model.layers[0]
            .attention
            .q_proj
            .weight
            .id,
        experiment.backbone.model.layers[0].mlp.up_proj.weight.id,
    ] {
        let grad = values(
            grads
                .get::<B, 2>(id)
                .expect("registered differentiable parameter"),
        );
        assert!(grad.iter().all(|x| x.is_finite()) && grad.iter().any(|x| *x != 0.0));
    }
}

/// Opt-in local checkpoint validation compares final logits, probabilities, and layers 8–11 MLP outputs.
#[test]
#[ignore = "requires pinned local GPT-2 weights and independent PyTorch reference; set GPT2_CHECKPOINT and GPT2_REFERENCE"]
fn pretrained_gpt2_matches_pytorch_last_logits_and_late_mlp_outputs() {
    /// Pinned real-checkpoint configuration and PyTorch outputs for unmasked input sequences.
    #[derive(Deserialize)]
    struct RealFixture {
        /// Architecture corresponding to the pinned local checkpoint.
        config: Gpt2Config,
        /// Unmasked token-ID rows submitted to both implementations.
        input_ids: Vec<Vec<i32>>,
        /// Independent final logits and intermediate MLP outputs.
        reference: Reference,
    }
    let checkpoint = std::env::var("GPT2_CHECKPOINT").expect("GPT2_CHECKPOINT");
    let reference: RealFixture = serde_json::from_slice(
        &std::fs::read(std::env::var("GPT2_REFERENCE").expect("GPT2_REFERENCE")).unwrap(),
    )
    .unwrap();
    let mut model = Gpt2ForCausalLm::<B>::init(reference.config, &Default::default()).unwrap();
    model
        .load_checkpoint(checkpoint, &LoadOptions::default())
        .unwrap();
    let batch = reference.input_ids.len();
    let sequence = reference.input_ids[0].len();
    let input = CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                reference.input_ids.into_iter().flatten().collect(),
                [batch, sequence],
            ),
            &Default::default(),
        ),
        token_mask: None,
        position_ids: None,
    };
    let mut hooks = Hooks::new();
    let captured: Vec<_> = (8..12)
        .map(|layer| hooks.capture(Layer::decoder(layer).mlp_output()))
        .collect();
    let output = model
        .forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)
        .unwrap();
    let actual = values(output.logits);
    let expected = flat(&reference.reference.logits);
    let logit_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("GPT-2 pretrained maximum final-logit absolute error: {logit_error}");
    // Raw logits have a large common offset. Use a conventional mixed F32
    // tolerance, and separately check probabilities so that a scale-relative
    // bound cannot hide a changed output distribution.
    let logits_match = actual
        .iter()
        .zip(&expected)
        .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-4 + 1e-5 * b.abs());
    let probability = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<_> = row.iter().map(|x| ((*x - max) as f64).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|x| x / sum).collect::<Vec<_>>()
    };
    let probability_pairs: Vec<_> = actual
        .chunks(50_257)
        .zip(expected.chunks(50_257))
        .flat_map(|(a, b)| probability(a).into_iter().zip(probability(b)))
        .collect();
    let probability_error = probability_pairs
        .iter()
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let probabilities_match = probability_pairs
        .iter()
        .all(|(a, b)| (a - b).abs() <= 1e-5 + 1e-4 * b.abs());
    eprintln!("GPT-2 pretrained maximum probability absolute error: {probability_error}");
    let mut mlp_errors = Vec::new();
    let mut mlps_match = true;
    for (layer, captured) in (8..12).zip(captured) {
        let actual = values(hooks.get(&captured).unwrap());
        let expected = flat(&reference.reference.mlp_outputs[&format!("mlp_{layer}")]);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("GPT-2 MLP {layer} maximum absolute error: {error}");
        mlps_match &= actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-4 + 1e-5 * b.abs());
        mlp_errors.push(serde_json::json!({"layer": layer, "max_absolute_error": error}));
    }
    if let Ok(path) = std::env::var("GPT2_MEASUREMENTS") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "maximum_logit_absolute_error": logit_error,
                "maximum_probability_absolute_error": probability_error,
                "late_mlp_errors": mlp_errors,
                "logits_and_mlp_absolute_tolerance": 1e-4,
                "logits_and_mlp_relative_tolerance": 1e-5,
                "probability_absolute_tolerance": 1e-5,
                "probability_relative_tolerance": 1e-4,
                "logits_match": logits_match,
                "late_mlps_match": mlps_match,
                "probabilities_match": probabilities_match,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    assert!(logits_match && mlps_match && probabilities_match);
}
