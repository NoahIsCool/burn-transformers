//! Prepared unpadded inputs preserve causal outputs, hooks, gradients, and checked bounds.

use super::*;
use crate::intervention::{Hooks, Layer};

/// Fully materialized RoPE fixture avoids lazy-clone randomness in parity checks.
fn model() -> ModularCausalLm<TestBackend> {
    deterministic_attention_model(
        None,
        PositionEncodingSpec::Rope {
            theta: 10_000.0,
            rotary_dim: 4,
            scaling: RopeScaling::None,
        },
        &Default::default(),
    )
}

/// Ordinary input for the identical complete host rows used by preparation.
fn ordinary(tokens: Vec<i32>, shape: [usize; 2]) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::from_data(TensorData::new(tokens, shape), &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// All/last/selected logits and retained hidden states match at context boundaries.
#[test]
fn prepared_inputs_preserve_logits_and_hidden_states() {
    let model = model();
    for shape in [[1, 1], [2, 7], [2, 32]] {
        let tokens = (0..shape[0] * shape[1])
            .map(|i| (i % 32) as i32)
            .collect::<Vec<_>>();
        let prepared = model.prepare_host_tokens(tokens.clone(), shape).unwrap();
        assert_eq!(prepared.shape(), shape);
        for logits in [
            LogitsSelection::All,
            LogitsSelection::Last,
            LogitsSelection::Positions(vec![shape[1] - 1, 0]),
        ] {
            let options = ForwardOptions {
                logits,
                hidden_states: true,
                ..Default::default()
            };
            let expected = model
                .forward(ordinary(tokens.clone(), shape), &options)
                .unwrap();
            let actual = model.forward_prepared(&prepared, &options).unwrap();
            assert_eq!(actual.logits.into_data(), expected.logits.into_data());
            assert_eq!(actual.hidden_states.len(), expected.hidden_states.len());
            for (actual, expected) in actual.hidden_states.into_iter().zip(expected.hidden_states) {
                assert_eq!(actual.into_data(), expected.into_data());
            }
            assert_eq!(
                actual.final_hidden_state.unwrap().into_data(),
                expected.final_hidden_state.unwrap().into_data()
            );
        }
    }
}

/// Bad host storage, empty/overflowing extents, context overflow, and IDs fail early.
#[test]
fn prepared_inputs_reject_invalid_host_values() {
    let model = model();
    for (tokens, shape) in [
        (vec![], [0, 1]),
        (vec![], [1, 0]),
        (vec![1], [2, 1]),
        (vec![1], [usize::MAX, 2]),
        (vec![1; 33], [1, 33]),
        (vec![-1], [1, 1]),
        (vec![32], [1, 1]),
    ] {
        assert!(
            model.prepare_host_tokens(tokens, shape).is_err(),
            "shape={shape:?}"
        );
    }
}

/// A prepared batch can be reused after cloning and by models with the same bounds.
#[test]
fn prepared_inputs_are_reusable_by_compatible_models() {
    let model = model();
    let prepared = model.prepare_host_tokens(vec![0, 31], [1, 2]).unwrap();
    let expected = model
        .forward_prepared(&prepared, &Default::default())
        .unwrap()
        .logits
        .into_data();
    assert_eq!(
        expected,
        model
            .forward_prepared(&prepared.clone(), &Default::default())
            .unwrap()
            .logits
            .into_data()
    );
    assert_eq!(
        expected,
        model
            .clone()
            .forward_prepared(&prepared, &Default::default())
            .unwrap()
            .logits
            .into_data()
    );
}

/// A capability cannot cross vocabulary/context bounds or bypass output-option checks.
#[test]
fn prepared_inputs_reject_incompatible_models_and_options() {
    let model = model();
    let prepared = model.prepare_host_tokens(vec![1, 2], [1, 2]).unwrap();
    for (vocabulary, capacity) in [(31, 32), (32, 31)] {
        let mut spec = model.spec().clone();
        spec.vocabulary_size = vocabulary;
        spec.max_position_embeddings = capacity;
        let other = ModularCausalLm::<TestBackend>::init(spec, &Default::default()).unwrap();
        assert!(
            other
                .forward_prepared(&prepared, &Default::default())
                .unwrap_err()
                .to_string()
                .contains("compatibility")
        );
    }
    for options in [
        ForwardOptions {
            attentions: true,
            ..Default::default()
        },
        ForwardOptions {
            logits: LogitsSelection::Positions(vec![2]),
            ..Default::default()
        },
    ] {
        assert!(model.forward_prepared(&prepared, &options).is_err());
    }
}

/// Captures and edits take the identical shared backbone path for prepared tokens.
#[test]
fn prepared_inputs_preserve_hook_edits_and_captures() {
    let model = model();
    let prepared = model.prepare_host_tokens(vec![1, 2, 3], [1, 3]).unwrap();
    let point = Layer::decoder(0).residual_post();
    let mut hooks = Hooks::new();
    hooks.edit(&point, |hidden| hidden * 0.5);
    let captured = hooks.capture(&point);
    let expected = model
        .forward_with_hooks(
            ordinary(vec![1, 2, 3], [1, 3]),
            &Default::default(),
            &mut hooks,
        )
        .unwrap();
    let expected_capture = hooks.get(&captured).unwrap().into_data();
    let actual = model
        .forward_prepared_with_hooks(&prepared, &Default::default(), &mut hooks)
        .unwrap();
    assert_eq!(actual.logits.into_data(), expected.logits.into_data());
    assert_eq!(hooks.get(&captured).unwrap().into_data(), expected_capture);
}

/// Prepared inputs retain embedding/head gradients and support repeated fresh graphs.
#[cfg(feature = "autodiff")]
#[test]
fn prepared_inputs_preserve_training_gradients() {
    use burn::backend::Autodiff;
    /// Native reverse-mode CPU backend with unchanged F32 arithmetic.
    type Ad = Autodiff<TestBackend>;
    let device = Default::default();
    let model = ModularCausalLm::<Ad>::init(
        decoder(vec![layer(attention(PositionEncodingSpec::None))]),
        &device,
    )
    .unwrap();
    assert!(model.num_params() > 0);
    let prepared = model.prepare_host_tokens(vec![1, 2, 3, 4], [2, 2]).unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::from_ints([[1, 2], [3, 4]], &device),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .unwrap()
        .logits
        .square()
        .mean()
        .backward();
    let expected_embedding = model
        .embed_tokens
        .weight
        .grad(&expected)
        .unwrap()
        .into_data();
    let expected_head = model
        .lm_head
        .as_ref()
        .unwrap()
        .weight
        .grad(&expected)
        .unwrap()
        .into_data();
    assert!(
        expected_embedding
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .any(|&v| v.abs() > 0.0)
    );
    for _ in 0..2 {
        let actual = model
            .forward_prepared(&prepared, &options)
            .unwrap()
            .logits
            .square()
            .mean()
            .backward();
        model
            .embed_tokens
            .weight
            .grad(&actual)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&expected_embedding, Tolerance::absolute(1e-6));
        model
            .lm_head
            .as_ref()
            .unwrap()
            .weight
            .grad(&actual)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(&expected_head, Tolerance::absolute(1e-6));
    }
}
