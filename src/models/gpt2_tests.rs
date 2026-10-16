//! GPT-2 configuration validation: extent, width, and overflow failures name their field.

use super::*;

/// A valid one-block topology that each case perturbs.
fn tiny() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 7,
        n_embd: 4,
        n_layer: 1,
        n_head: 2,
        n_positions: 8,
        n_inner: Some(6),
        ..Gpt2Config::default()
    }
}

/// Validates `config` and returns the failing `(field, reason)`.
fn failure(config: &Gpt2Config) -> (&'static str, String) {
    let error = Validate::validate(config).unwrap_err();
    assert_eq!(error.architecture, "gpt2");
    (error.field, error.reason)
}

/// Zero or beyond-`i32::MAX` extents are rejected before any other rule, naming the field.
#[test]
fn every_extent_must_be_positive_and_portable() {
    /// Sets one extent field of a configuration to the given value.
    type Edit = fn(&mut Gpt2Config, usize);
    let cases: [(&str, Edit); 5] = [
        ("vocab_size", |config, value| config.vocab_size = value),
        ("n_embd", |config, value| config.n_embd = value),
        ("n_layer", |config, value| config.n_layer = value),
        ("n_head", |config, value| config.n_head = value),
        ("n_positions", |config, value| config.n_positions = value),
    ];
    for (field, edit) in cases {
        for value in [0, i32::MAX as usize + 1] {
            let mut config = tiny();
            edit(&mut config, value);
            assert_eq!(
                failure(&config),
                (
                    field,
                    "must be positive and fit the portable integer index range".into()
                ),
                "{field}={value}"
            );
        }
    }
    Validate::validate(&tiny()).expect("the unperturbed topology is valid");
}

/// The MLP width must be positive, and every `n_embd`-wide parameter product must fit `usize`.
#[test]
fn mlp_width_and_parameter_products_are_bounded() {
    assert_eq!(
        failure(&Gpt2Config {
            n_inner: Some(0),
            ..tiny()
        }),
        ("n_inner", "must be positive and fit usize".into())
    );
    // `n_inner` is not limited to the portable index range, so its product can overflow.
    assert_eq!(
        failure(&Gpt2Config {
            n_inner: Some(usize::MAX / 2),
            ..tiny()
        }),
        ("n_embd", "parameter dimensions overflow usize".into())
    );
    assert_eq!(
        failure(&Gpt2Config {
            add_cross_attention: true,
            ..tiny()
        }),
        (
            "add_cross_attention",
            "cross-attention blocks are not implemented".into()
        )
    );
}

/// The re-initialization policy resets every LayerNorm as Hugging Face's `_init_weights` does.
///
/// Regression for GPT2-NEOX-INIT-DOC: the policy documentation claimed
/// `ParameterInit` could not express ones, and left norm resets to a manual
/// rebuild; per-path `Ones`/`Zeros` overrides express them exactly.
#[test]
fn reinit_policy_resets_every_layer_norm() {
    use crate::initialization::ParameterRole;
    use burn::{backend::Flex, module::Param, tensor::Tensor};

    type B = Flex<f32, i32>;
    let config = Gpt2Config {
        n_layer: 2,
        ..tiny()
    };
    let device = Default::default();
    let mut model = Gpt2ForCausalLm::<B>::init(config.clone(), &device).unwrap();
    let width = config.n_embd;
    let perturb = |norm: &mut burn::nn::LayerNorm<B>| {
        norm.gamma = Param::from_tensor(Tensor::full([width], 2.0, &device));
        norm.beta = Some(Param::from_tensor(Tensor::full([width], 0.5, &device)));
    };
    perturb(&mut model.model.layers[0].attention_norm);
    perturb(&mut model.model.layers[1].mlp_norm);
    perturb(&mut model.model.final_norm);
    let policy = config.reinit_policy();
    policy.validate_for(&model).unwrap();
    let (model, report) = policy.apply(model).unwrap();

    let values = |tensor: Tensor<B, 1>| tensor.into_data().to_vec::<f32>().unwrap();
    let blocks = &model.model.layers;
    for norm in [
        &blocks[0].attention_norm,
        &blocks[0].mlp_norm,
        &blocks[1].attention_norm,
        &blocks[1].mlp_norm,
        &model.model.final_norm,
    ] {
        assert!(values(norm.gamma.val()).iter().all(|&value| value == 1.0));
        assert!(
            values(norm.beta.as_ref().unwrap().val())
                .iter()
                .all(|&value| value == 0.0)
        );
    }
    // Every parameter without a role rule is a norm parameter the policy now resets.
    let others = report
        .parameters
        .iter()
        .filter(|parameter| parameter.role == ParameterRole::Other)
        .collect::<Vec<_>>();
    assert_eq!(others.len(), 2 * (2 * config.n_layer + 1));
    for parameter in others {
        let expected = if parameter.path.ends_with(".gamma") {
            ParameterInit::Ones
        } else {
            assert!(parameter.path.ends_with(".beta"), "{}", parameter.path);
            ParameterInit::Zeros
        };
        assert_eq!(parameter.init, expected, "{}", parameter.path);
    }
    // The base policy still keeps the norms of a fresh model as constructed.
    assert!(
        config
            .init_policy()
            .resolve("model.final_norm.gamma", ParameterRole::Other)
            == ParameterInit::Keep
    );
}
