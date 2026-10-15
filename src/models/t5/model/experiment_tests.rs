//! T5 through the family-generic hooked and staged experiment interfaces.

use burn::{
    backend::Flex,
    tensor::{Int, Tensor},
};

use super::*;
use crate::intervention::DynamicActivation;

/// Deterministic CPU backend; trait and inherent calls must agree bit for bit.
type TestBackend = Flex<f32, i32>;

/// Two encoder and two decoder blocks with a gated feed-forward.
fn config() -> T5Config {
    T5Config {
        vocab_size: 24,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: 2,
        num_decoder_layers: Some(2),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: T5FeedForwardProjection::GatedGelu,
        tie_word_embeddings: false,
        ..T5Config::default()
    }
}

/// Tokens of one batch row without padding or explicit positions.
fn tokens(ids: [i64; 4]) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints([ids], &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// Every captured point of one run, in catalog order.
type ActivationCache<B> = Vec<(PointInfo, DynamicActivation<B>)>;

/// Generic activation cache: captures every listed point of any hooked model.
fn run_with_cache<B: Backend, M: HookedModel<B>>(
    model: &M,
    input: M::Input,
) -> Result<(M::Output, ActivationCache<B>)> {
    let mut hooks = Hooks::new();
    let handles = model
        .activation_points()?
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
    let cache = handles
        .iter()
        .map(|handle| Ok((handle.point().clone(), hooks.get_dynamic(handle)?)))
        .collect::<Result<_>>()?;
    Ok((output, cache))
}

/// Generic patching: replaces the residual stream entering `layer` with `donor`.
fn patch_residual<B: Backend, M: StagedModel<B>>(
    model: &M,
    input: CausalLmInput<B>,
    donor: Tensor<B, 3>,
    layer: usize,
) -> Result<Tensor<B, 3>> {
    let state = model.embed(input)?;
    let mut state = model.run_layers(state, 0..layer)?;
    state.hidden = donor;
    let state = model.run_layers(state, layer..model.num_layers())?;
    Ok(model.normalize(state)?.hidden)
}

/// The hooked trait runs the complete encoder-decoder pass and captures every point.
#[test]
fn hooked_trait_runs_the_inherent_encoder_decoder_pass() {
    let device = Default::default();
    let model = T5ForConditionalGeneration::<TestBackend>::init(config(), &device).unwrap();
    let input = || Seq2SeqInput {
        encoder: tokens([3, 4, 5, 6]),
        decoder: tokens([0, 7, 8, 9]),
    };
    let (output, cache) = run_with_cache(&model, input()).unwrap();
    let expected = model.forward(input(), &ForwardOptions::default()).unwrap();
    assert_eq!(
        output.decoder.logits.into_data(),
        expected.decoder.logits.into_data()
    );
    let points = HookedModel::activation_points(&model).unwrap();
    assert_eq!(
        points,
        T5ForConditionalGeneration::activation_points(&model)
    );
    assert_eq!(cache.len(), points.len());
    assert_eq!(cache[0].0, Stack::Encoder.embedding().info());
    assert_eq!(cache.last().unwrap().0, Stack::Decoder.final_norm().info());
    // The decoder's final-norm capture is the state the logits project.
    let final_norm = cache.last().unwrap().1.clone().into_tensor::<3>().unwrap();
    assert_eq!(final_norm.dims(), [1, 4, 8]);
    assert_eq!(
        model.project_logits(final_norm).unwrap().into_data(),
        model
            .forward(
                input(),
                &ForwardOptions {
                    logits: crate::LogitsSelection::All,
                    ..ForwardOptions::default()
                },
            )
            .unwrap()
            .decoder
            .logits
            .into_data()
    );
}

/// The encoder is hooked and staged through the traits exactly like its inherent methods.
#[test]
fn encoder_traits_stage_patch_and_capture_like_the_inherent_methods() {
    let device = Default::default();
    let encoder = T5ForConditionalGeneration::<TestBackend>::init(config(), &device)
        .unwrap()
        .into_encoder()
        .unwrap();
    assert_eq!(StagedModel::num_layers(&encoder), 2);
    assert_eq!(StagedModel::stack(&encoder), Stack::Encoder);
    assert_eq!(StagedModel::layer(&encoder, 1), Layer::encoder(1));

    let expected = encoder
        .encode(tokens([3, 4, 5, 6]), &ForwardOptions::default())
        .unwrap()
        .hidden_state;
    let staged = encoder.run_stack(tokens([3, 4, 5, 6])).unwrap();
    assert!(staged.normalized);
    assert_eq!(staged.hidden.into_data(), expected.clone().into_data());

    // Patching the whole residual entering layer 1 with another input's
    // residual reproduces that input's output: positions and padding agree.
    let donor = StagedModel::run_layers(
        &encoder,
        StagedModel::embed(&encoder, tokens([9, 8, 7, 6])).unwrap(),
        0..1,
    )
    .unwrap()
    .hidden;
    let patched = patch_residual(&encoder, tokens([3, 4, 5, 6]), donor, 1).unwrap();
    let other = encoder
        .encode(tokens([9, 8, 7, 6]), &ForwardOptions::default())
        .unwrap()
        .hidden_state;
    assert_eq!(patched.into_data(), other.into_data());

    let (output, cache) = run_with_cache(&encoder, tokens([3, 4, 5, 6])).unwrap();
    assert_eq!(output.hidden_state.into_data(), expected.into_data());
    assert_eq!(cache.len(), T5Encoder::activation_points(&encoder).len());
    assert_eq!(cache.last().unwrap().0, Stack::Encoder.final_norm().info());
}

/// Staged trait calls keep the inherent cursor and normalization contracts.
#[test]
fn staged_trait_rejects_skipped_ranges_and_double_normalization() {
    let device = Default::default();
    let encoder = T5ForConditionalGeneration::<TestBackend>::init(config(), &device)
        .unwrap()
        .into_encoder()
        .unwrap();
    let state = StagedModel::embed(&encoder, tokens([3, 4, 5, 6])).unwrap();
    assert!(StagedModel::run_layers(&encoder, state.clone(), 1..2).is_err());
    let normalized = StagedModel::normalize(
        &encoder,
        StagedModel::run_layers(&encoder, state, 0..2).unwrap(),
    )
    .unwrap();
    assert!(StagedModel::normalize(&encoder, normalized.clone()).is_err());
    assert!(StagedModel::run_layers(&encoder, normalized, 2..2).is_err());
}
