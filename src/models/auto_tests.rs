//! Registry dispatch, complete parameter materialization, and decode-state contract regressions.

use burn::{
    backend::Flex,
    module::{ModuleVisitor, Param, ParamId},
    tensor::{Bool, Int, Tensor, TensorData, Tolerance, backend::BackendTypes},
};

use super::*;
use crate::models::{
    muse_glimmer::MuseGlimmerLayerType, nemotron_h::NemotronHBlockType, qwen3_5::Qwen3_5LayerType,
    t5::T5FeedForwardProjection,
};
use crate::{
    CausalGenerationRequest, GenerationConfig, LogitsSelection, TokenMask, generate_causal,
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
type TestBackend = Flex<f32, i32>;

/// Registry-level activation-hook delegation and capability-failure regressions.
#[path = "auto/intervention_tests.rs"]
mod intervention_tests;

/// Constructs a tiny validated hybrid configuration for registry tests.
fn tiny_qwen() -> Qwen3_5TextConfig {
    Qwen3_5TextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        partial_rotary_factor: Some(0.5),
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        linear_conv_kernel_dim: 2,
        linear_key_head_dim: 2,
        linear_value_head_dim: 2,
        linear_num_key_heads: 2,
        linear_num_value_heads: 2,
        mtp_num_hidden_layers: 0,
        ..Qwen3_5TextConfig::default()
    }
}

/// Constructs a tiny validated sparse Qwen configuration for registry tests.
fn tiny_qwen_moe() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 64,
        max_window_layers: 1,
        decoder_sparse_step: 1,
        moe_intermediate_size: 12,
        num_experts: 4,
        num_experts_per_tok: 2,
        ..Qwen3MoeConfig::default()
    }
}

/// Constructs a tiny Muse decoder retaining local and global attention.
fn tiny_muse() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: Some(crate::models::config::TokenIds::One(1)),
        eos_token_id: Some(crate::models::config::TokenIds::One(2)),
        layer_types: vec![
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::SlidingAttention,
            MuseGlimmerLayerType::FullAttention,
        ],
        ..MuseGlimmerTextConfig::default()
    }
}

/// Constructs a tiny Nemotron schedule retaining every executable block kind.
fn tiny_nemotron() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 32,
        hidden_size: 8,
        layers_block_type: vec![
            NemotronHBlockType::LinearAttention,
            NemotronHBlockType::FullAttention,
            NemotronHBlockType::Mlp,
            NemotronHBlockType::Moe,
        ],
        num_hidden_layers: Some(4),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        intermediate_size: 12,
        ssm_state_size: 3,
        mamba_num_heads: 4,
        mamba_head_dim: 2,
        n_groups: 2,
        conv_kernel: 3,
        chunk_size: 2,
        n_routed_experts: 4,
        n_shared_experts: 1,
        moe_intermediate_size: 6,
        moe_shared_expert_intermediate_size: 8,
        num_experts_per_tok: 2,
        n_group: 2,
        topk_group: 1,
        num_nextn_predict_layers: 0,
        ..NemotronHConfig::default()
    }
}

/// Constructs a tiny validated T5 configuration for registry tests.
fn tiny_t5() -> T5Config {
    T5Config {
        vocab_size: 32,
        d_model: 8,
        d_kv: 4,
        d_ff: 16,
        num_layers: 1,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        dropout_rate: 0.0,
        feed_forward_proj: T5FeedForwardProjection::Relu,
        ..T5Config::default()
    }
}

/// Builds a rectangular prompt whose first chunk is inactive for row zero.
///
/// Both rows become active in every later two-column chunk, so the test
/// exercises the coordinator's initial decode fallback before returning to
/// each family's optimized prefill path.
fn staggered_prompt(
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
            [[31, 31, 1, 2, 3, 31], [4, 5, 6, 7, 31, 8]],
            device,
        ),
        token_mask: Some(TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
            [
                [false, false, true, true, true, false],
                [true, true, true, true, false, true],
            ]
            .into(),
            device,
        ))),
        position_ids: None,
    }
}

/// Extracts one physical prompt chunk without changing semantic mask order.
fn prompt_chunk(
    prompt: &CausalLmInput<TestBackend>,
    start: usize,
    end: usize,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: prompt.token_ids.clone().slice([0..2, start..end]),
        token_mask: prompt
            .token_mask
            .as_ref()
            .map(|mask| TokenMask::new(mask.as_tensor().clone().slice([0..2, start..end]))),
        position_ids: prompt
            .position_ids
            .as_ref()
            .map(|positions| positions.clone().slice([0..2, start..end])),
    }
}

/// Proves empty-state decode, chunked prefill, and generation parity for one model.
fn assert_bounded_prefill_contract<M>(
    family: &str,
    model: &M,
    cache: CacheConfig,
    vocabulary_size: usize,
    device: &<TestBackend as burn::tensor::backend::BackendTypes>::Device,
) where
    M: CausalLanguageModel<TestBackend>,
{
    let prompt = staggered_prompt(device);
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };

    let mut one_shot_state = model
        .new_decode_state(cache, 2, device)
        .unwrap_or_else(|error| panic!("{family} one-shot state: {error}"));
    let one_shot = model
        .prefill(prompt.clone(), &mut one_shot_state, &options)
        .unwrap_or_else(|error| panic!("{family} one-shot prefill: {error}"));
    assert_eq!(
        one_shot_state.lengths(),
        [3, 5],
        "{family} one-shot lengths"
    );

    let mut chunked_state = model
        .new_decode_state(cache, 2, device)
        .unwrap_or_else(|error| panic!("{family} chunked state: {error}"));
    let first = prompt_chunk(&prompt, 0, 2);
    model
        .decode(
            DecodeInput {
                token_ids: first.token_ids,
                token_mask: first.token_mask,
                position_ids: first.position_ids,
            },
            &mut chunked_state,
            &options,
        )
        .unwrap_or_else(|error| panic!("{family} fresh-state decode fallback: {error}"));
    assert_eq!(
        chunked_state.lengths(),
        [0, 2],
        "{family} must preserve its initially inactive row"
    );
    model
        .prefill(prompt_chunk(&prompt, 2, 4), &mut chunked_state, &options)
        .unwrap_or_else(|error| panic!("{family} middle prefill chunk: {error}"));
    let chunked = model
        .prefill(prompt_chunk(&prompt, 4, 6), &mut chunked_state, &options)
        .unwrap_or_else(|error| panic!("{family} final prefill chunk: {error}"));
    assert_eq!(chunked_state.lengths(), [3, 5], "{family} chunked lengths");
    chunked.logits.into_data().assert_approx_eq::<f32>(
        &one_shot.logits.into_data(),
        Tolerance::rel_abs(5.0e-4, 5.0e-5),
    );

    let generation = |prefill_chunk_size| CausalGenerationRequest {
        prompt: prompt.clone(),
        cache,
        generation: GenerationConfig {
            max_new_tokens: 2,
            ..Default::default()
        },
        prefill_chunk_size,
        vocabulary_size,
    };
    let one_shot = generate_causal(model, generation(None))
        .unwrap_or_else(|error| panic!("{family} one-shot generation: {error}"));
    let chunked = generate_causal(model, generation(Some(2)))
        .unwrap_or_else(|error| panic!("{family} chunked generation: {error}"));
    assert_eq!(chunked.sequences.len(), one_shot.sequences.len());
    for (row, (chunked, one_shot)) in chunked
        .sequences
        .iter()
        .zip(&one_shot.sequences)
        .enumerate()
    {
        assert_eq!(
            chunked.token_ids, one_shot.token_ids,
            "{family} generated row {row}"
        );
        assert_eq!(chunked.finish_reason, one_shot.finish_reason);
        assert_eq!(chunked.prompt_length, one_shot.prompt_length);
        assert_eq!(
            chunked.log_probabilities.len(),
            one_shot.log_probabilities.len()
        );
        for (&actual, &expected) in chunked
            .log_probabilities
            .iter()
            .zip(&one_shot.log_probabilities)
        {
            assert!(
                (actual - expected).abs() <= 5.0e-4,
                "{family} row {row} log probability {actual} != {expected}"
            );
        }
    }
}

/// Tiny modular decoder: RoPE attention followed by Gated DeltaNet.
fn tiny_modular() -> crate::architecture::DecoderSpec {
    use crate::architecture::{
        Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
        FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
        ResidualTopology, RopeScaling,
    };
    let norm = NormalizationSpec::Rms {
        epsilon: 1e-6,
        zero_centered: false,
    };
    let layer = |mixer| DecoderLayerSpec {
        mixer,
        feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
            intermediate_size: 12,
            activation: Activation::Silu,
            bias: false,
        }),
        mixer_norm: norm,
        feed_forward_norm: norm,
        norm_placement: NormPlacement::Pre,
        residual: ResidualTopology::Add,
    };
    DecoderSpec {
        vocabulary_size: 32,
        hidden_size: 8,
        layers: vec![
            layer(MixerSpec::Attention(AttentionSpec {
                query_heads: 2,
                key_value_heads: 1,
                head_dim: 4,
                query_key_norm: true,
                output_gate: false,
                position: PositionEncodingSpec::Rope {
                    theta: 10_000.0,
                    rotary_dim: 4,
                    scaling: RopeScaling::None,
                },
            })),
            layer(MixerSpec::DeltaNet {
                key_heads: 1,
                value_heads: 2,
                key_head_dim: 4,
                value_head_dim: 4,
                convolution_kernel: 2,
            }),
        ],
        final_norm: norm,
        tie_word_embeddings: false,
        max_position_embeddings: 32,
    }
}

/// Every causal language model supports the bounded-prefill coordinator contract.
///
/// The staggered prompt's first chunk reaches `decode` while row zero has
/// never started, which every implementation must accept as a no-op. The test
/// covers every registered family (each Llama-family alias included) and the
/// modular reference model, and fails if a registered family is left out.
#[test]
fn every_causal_family_matches_bounded_and_one_shot_prefill() {
    let device = Default::default();
    let cache = CacheConfig::new(2, 8).unwrap();
    let mut covered = Vec::new();

    TestBackend::seed(&device, 10_001);
    let qwen = Qwen3_5ForCausalLm::init(tiny_qwen(), &device).expect("tiny Qwen");
    assert_bounded_prefill_contract("Qwen3.5", &qwen, cache, 32, &device);
    covered.push(AutoCausalConfig::Qwen3_5);

    TestBackend::seed(&device, 10_002);
    let qwen_moe = Qwen3MoeForCausalLm::init(tiny_qwen_moe(), &device).expect("tiny Qwen MoE");
    assert_bounded_prefill_contract("Qwen3-MoE", &qwen_moe, cache, 32, &device);
    covered.push(AutoCausalConfig::Qwen3Moe);

    let deepseek = crate::models::deepseek_v2::model::tests_support::initialized_model(
        crate::models::deepseek_v2::model::tests_support::tiny_config(),
        &device,
    );
    assert_bounded_prefill_contract("DeepSeek-V2", &deepseek, cache, 32, &device);
    covered.push(AutoCausalConfig::DeepseekV2);

    TestBackend::seed(&device, 10_003);
    let muse = MuseGlimmerForCausalLm::init(tiny_muse(), &device).expect("tiny Muse");
    assert_bounded_prefill_contract(
        "Muse Glimmer",
        &muse,
        cache.with_sliding_window(3).unwrap(),
        32,
        &device,
    );
    covered.push(AutoCausalConfig::MuseGlimmer);

    TestBackend::seed(&device, 10_004);
    let nemotron = NemotronHForCausalLm::init(tiny_nemotron(), &device).expect("tiny Nemotron-H");
    assert_bounded_prefill_contract("Nemotron-H", &nemotron, cache, 32, &device);
    covered.push(AutoCausalConfig::NemotronH);

    for (name, json) in [
        (
            "Llama 3",
            include_str!("../../tests/fixtures/tiny_goldens/llama3/config.json"),
        ),
        (
            "Llama with bias and tied embeddings",
            include_str!("../../tests/fixtures/tiny_goldens/llama-bias-tied/config.json"),
        ),
        (
            "Mistral sliding",
            include_str!("../../tests/fixtures/tiny_goldens/mistral-sliding/config.json"),
        ),
        (
            "OLMo 2",
            include_str!("../../tests/fixtures/tiny_goldens/olmo2/config.json"),
        ),
        (
            "Qwen2",
            include_str!("../../tests/fixtures/tiny_goldens/qwen2/config.json"),
        ),
        (
            "Qwen3",
            include_str!("../../tests/fixtures/tiny_goldens/qwen3/config.json"),
        ),
        (
            "SmolLM3",
            include_str!("../../tests/fixtures/tiny_goldens/smollm3/config.json"),
        ),
        (
            "Gemma 2",
            include_str!("../../tests/fixtures/tiny_goldens/gemma2/config.json"),
        ),
        (
            "Gemma 3 text",
            include_str!("../../tests/fixtures/tiny_goldens/gemma3-text/config.json"),
        ),
        (
            "gpt-oss",
            include_str!("../../tests/fixtures/tiny_goldens/gpt-oss/config.json"),
        ),
    ] {
        TestBackend::seed(&device, 10_006);
        let model = AutoCausalLm::<TestBackend>::from_config_json(json, &device).unwrap();
        let vocabulary = model.config().vocabulary_size();
        covered.push(model.family());
        // Unwrap to the concrete family so its own implementation is checked.
        match model {
            AutoCausalLm::Llama(model) => {
                assert_bounded_prefill_contract(name, &model, cache, vocabulary, &device);
            }
            AutoCausalLm::Gemma(model) => {
                assert_bounded_prefill_contract(name, &model, cache, vocabulary, &device);
            }
            AutoCausalLm::GptOss(model) => {
                assert_bounded_prefill_contract(name, &model, cache, vocabulary, &device);
            }
            other => panic!("{name} selected the {:?} family", other.family()),
        }
    }

    TestBackend::seed(&device, 10_007);
    let modular =
        crate::models::custom::ModularCausalLm::<TestBackend>::init(tiny_modular(), &device)
            .expect("tiny modular decoder");
    assert_bounded_prefill_contract("ModularCausalLm", &modular, cache, 32, &device);

    for family in AutoCausalConfig::ALL {
        assert!(
            covered.contains(family),
            "registered family {family:?} lacks a bounded-prefill check"
        );
    }
}

/// Every registered causal family as a tiny automatic model, each Llama-family alias included.
///
/// Parameters are seeded per family; the vocabularies are 32 for the hand-built
/// configurations and 96 for the committed tiny-golden ones.
fn tiny_auto_causal_models(
    device: &RegistryDevice,
) -> Vec<(&'static str, AutoCausalLm<TestBackend>)> {
    let mut models = Vec::new();
    TestBackend::seed(device, 10_101);
    models.push((
        "Qwen3.5",
        AutoCausalLm::from(Qwen3_5ForCausalLm::init(tiny_qwen(), device).unwrap()),
    ));
    models.push((
        "Qwen3-MoE",
        AutoCausalLm::from(Qwen3MoeForCausalLm::init(tiny_qwen_moe(), device).unwrap()),
    ));
    models.push((
        "DeepSeek-V2",
        AutoCausalLm::from(
            crate::models::deepseek_v2::model::tests_support::initialized_model(
                crate::models::deepseek_v2::model::tests_support::tiny_config(),
                device,
            ),
        ),
    ));
    models.push((
        "Muse Glimmer",
        AutoCausalLm::from(MuseGlimmerForCausalLm::init(tiny_muse(), device).unwrap()),
    ));
    models.push((
        "Nemotron-H",
        AutoCausalLm::from(NemotronHForCausalLm::init(tiny_nemotron(), device).unwrap()),
    ));
    for (name, json) in [
        (
            "Llama 3",
            include_str!("../../tests/fixtures/tiny_goldens/llama3/config.json"),
        ),
        (
            "Mistral sliding",
            include_str!("../../tests/fixtures/tiny_goldens/mistral-sliding/config.json"),
        ),
        (
            "SmolLM3",
            include_str!("../../tests/fixtures/tiny_goldens/smollm3/config.json"),
        ),
        (
            "Gemma 2",
            include_str!("../../tests/fixtures/tiny_goldens/gemma2/config.json"),
        ),
        (
            "Gemma 3 text",
            include_str!("../../tests/fixtures/tiny_goldens/gemma3-text/config.json"),
        ),
        (
            "gpt-oss",
            include_str!("../../tests/fixtures/tiny_goldens/gpt-oss/config.json"),
        ),
    ] {
        models.push((
            name,
            AutoCausalLm::<TestBackend>::from_config_json(json, device).unwrap(),
        ));
    }
    for family in AutoCausalConfig::ALL {
        assert!(
            models.iter().any(|(_, model)| model.family() == *family),
            "tiny registry fixtures lack {family:?}"
        );
    }
    models
}

/// Every model rejects a batch that disagrees with its cache before allocating.
///
/// `new_decode_state` takes the batch twice, once in `CacheConfig` and once as
/// an argument; `new_decode_state_for` takes it from the configuration, so it
/// cannot disagree. Covers every registered causal family, T5, and the
/// modular causal and seq2seq models.
#[test]
fn decode_state_batch_must_match_the_cache_sequence_count() {
    use crate::{
        cache::TensorCacheError,
        models::custom::{ModularCausalLm, ModularSeq2SeqLm},
    };
    let device = Default::default();
    let cache = CacheConfig::new(2, 8).unwrap();
    /// Family tensor caches report the typed mismatch; the modular reference
    /// state reports the same two counts as a cache error.
    #[track_caller]
    fn assert_mismatch<S: core::fmt::Debug>(name: &str, result: Result<S>, requested: usize) {
        let error = result.expect_err(name);
        let typed = matches!(
            error,
            Error::TensorCache(TensorCacheError::BatchSizeMismatch { configured: 2, requested: found })
                if found == requested
        );
        let modular = matches!(
            &error,
            Error::Cache(reason)
                if *reason == format!("custom decode state expected 2 sequences, received batch size {requested}")
        );
        assert!(
            typed || (modular && name.starts_with("Modular")),
            "{name}: {error:?}"
        );
    }
    for (name, model) in tiny_auto_causal_models(&device) {
        for batch in [1, 3] {
            assert_mismatch(
                name,
                CausalLanguageModel::new_decode_state(&model, cache, batch, &device),
                batch,
            );
        }
        let state = model.new_decode_state_for(cache, &device).unwrap();
        assert_eq!(
            (state.batch_size(), state.lengths()),
            (2, &[0, 0][..]),
            "{name}"
        );
    }
    let t5 = T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap();
    let automatic_t5 = AutoSeq2SeqLm::from(t5.clone());
    assert_mismatch(
        "T5",
        Seq2SeqLanguageModel::new_decode_state(&t5, cache, 1, &device),
        1,
    );
    assert_mismatch(
        "AutoSeq2SeqLm<T5>",
        Seq2SeqLanguageModel::new_decode_state(&automatic_t5, cache, 3, &device),
        3,
    );
    assert_eq!(
        t5.new_decode_state_for(cache, &device)
            .unwrap()
            .batch_size(),
        2
    );
    assert_eq!(
        automatic_t5
            .new_decode_state_for(cache, &device)
            .unwrap()
            .batch_size(),
        2
    );
    let modular = ModularCausalLm::<TestBackend>::init(tiny_modular(), &device).unwrap();
    assert_mismatch(
        "ModularCausalLm",
        CausalLanguageModel::new_decode_state(&modular, cache, 1, &device),
        1,
    );
    assert_eq!(
        modular
            .new_decode_state_for(cache, &device)
            .unwrap()
            .batch_size(),
        2
    );
    // Bidirectional encoders cannot hold the causal DeltaNet scan; keep attention only.
    let mut attention_only = tiny_modular();
    attention_only.layers.truncate(1);
    let modular_seq2seq = ModularSeq2SeqLm::<TestBackend>::init(
        crate::architecture::Seq2SeqSpec {
            encoder: attention_only.clone(),
            decoder: attention_only,
            cross_attention_heads: 2,
            cross_attention_head_dim: 4,
            cross_attention_options: crate::architecture::AttentionOptions::default(),
        },
        &device,
    )
    .unwrap();
    assert_mismatch(
        "ModularSeq2SeqLm",
        Seq2SeqLanguageModel::new_decode_state(&modular_seq2seq, cache, 3, &device),
        3,
    );
    assert_eq!(
        modular_seq2seq
            .new_decode_state_for(cache, &device)
            .unwrap()
            .batch_size(),
        2
    );
}

/// A `CacheConfig` sliding window never changes a model family's retention.
///
/// Families derive per-layer windows from their architecture; the cache-level
/// window only sizes raw cache-bank rings that have no window of their own.
/// Allocating with a window of two and decoding past it must therefore give
/// exactly the logits of the same cache without a window, for every
/// registered causal family, T5, and the modular model.
#[test]
fn cache_sliding_window_never_changes_model_family_retention() {
    let device = Default::default();
    let plain = CacheConfig::new(2, 8).unwrap();
    let windowed = plain.with_sliding_window(2).unwrap();
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let bits = |logits: Tensor<TestBackend, 3>| {
        logits
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    };
    /// Prefills five tokens per row and decodes one more under `cache`.
    fn run_causal<M: CausalLanguageModel<TestBackend>>(
        model: &M,
        cache: CacheConfig,
        options: &ForwardOptions,
        device: &RegistryDevice,
    ) -> Result<Tensor<TestBackend, 3>> {
        let mut state = model.new_decode_state_for(cache, device)?;
        model.prefill(
            registry_tokens([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device),
            &mut state,
            options,
        )?;
        Ok(model
            .decode(registry_decode([[11], [12]], device), &mut state, options)?
            .logits)
    }
    for (name, model) in tiny_auto_causal_models(&device) {
        materialize_registry_parameters(&model);
        let expected = run_causal(&model, plain, &options, &device).unwrap();
        let windowed_run = run_causal(&model, windowed, &options, &device);
        assert_eq!(
            bits(windowed_run.unwrap_or_else(|error| panic!("{name}: {error:?}"))),
            bits(expected),
            "{name} must ignore the cache window"
        );
    }
    let modular =
        crate::models::custom::ModularCausalLm::<TestBackend>::init(tiny_modular(), &device)
            .unwrap();
    assert_eq!(
        bits(run_causal(&modular, windowed, &options, &device).unwrap()),
        bits(run_causal(&modular, plain, &options, &device).unwrap()),
        "the modular model must ignore the cache window"
    );
    let t5 = T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap();
    let encoder = t5
        .encode(
            registry_tokens([[8, 9, 10], [11, 12, 13]], &device),
            &options,
        )
        .unwrap();
    let run_t5 = |cache| {
        let mut state = t5.new_decode_state_for(cache, &device).unwrap();
        t5.prefill_decoder(
            &encoder,
            registry_tokens([[0, 4, 5, 6], [0, 7, 8, 9]], &device),
            &mut state,
            &options,
        )
        .unwrap();
        Seq2SeqLanguageModel::decode(
            &t5,
            &encoder,
            registry_decode([[10], [11]], &device),
            &mut state,
            &options,
        )
        .unwrap()
        .decoder
        .logits
    };
    assert_eq!(
        bits(run_t5(windowed)),
        bits(run_t5(plain)),
        "T5 must ignore the cache window"
    );
}

/// Automatic dispatch preserves the same fresh-state fallback and parity contract.
#[test]
fn auto_causal_dispatch_matches_bounded_and_one_shot_prefill() {
    let device = Default::default();
    TestBackend::seed(&device, 10_005);
    let model = AutoCausalLm::from(
        Qwen3MoeForCausalLm::init(tiny_qwen_moe(), &device).expect("tiny auto Qwen MoE"),
    );
    assert_bounded_prefill_contract(
        "AutoCausalLm<Qwen3-MoE>",
        &model,
        CacheConfig::new(2, 8).unwrap(),
        32,
        &device,
    );
}

/// Wrapped Qwen config detection prefers the language tower and allocates state.
#[test]
fn causal_registry_parses_wrapped_config_and_dispatches_state() {
    let device = Default::default();
    let wrapped = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": tiny_qwen(),
    });
    let model = AutoCausalLm::<TestBackend>::from_config_json(&wrapped.to_string(), &device)
        .expect("registered Qwen config");
    assert_eq!(model.config().model_type(), "qwen3_5_text");
    assert_eq!(model.config().vocabulary_size(), 32);
    let state = model
        .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
        .expect("registered state");
    assert_eq!(state.batch_size(), 1);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.lengths(), [0]);
}

/// Sparse Qwen config selection reaches its executable graph and cache variant.
#[test]
fn causal_registry_dispatches_qwen3_moe() {
    let device = Default::default();
    let json = serde_json::to_string(&tiny_qwen_moe()).expect("serializable config");
    let model = AutoCausalLm::<TestBackend>::from_config_json(&json, &device)
        .expect("registered Qwen3 MoE config");
    assert_eq!(model.config().model_type(), "qwen3_moe");
    assert_eq!(model.config().vocabulary_size(), 32);
    let mut state = model
        .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
        .expect("registered state");
    assert!(matches!(state, AutoCausalDecodeState::Qwen3Moe(_)));
    assert_eq!(state.lengths(), [0]);
    state.reset().expect("empty state reset");
}

/// Remaining v1 causal families select their concrete graphs and state variants.
#[test]
fn causal_registry_dispatches_deepseek_muse_and_nemotron() {
    let device = Default::default();

    let deepseek_config = crate::models::deepseek_v2::model::tests_support::tiny_config();
    let deepseek = AutoCausalLm::<TestBackend>::from_config_json(
        &serde_json::to_string(&deepseek_config).expect("serializable DeepSeek config"),
        &device,
    )
    .expect("registered DeepSeek config");
    assert_eq!(deepseek.config().model_type(), "deepseek_v2");
    assert!(matches!(
        deepseek
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .expect("DeepSeek state"),
        AutoCausalDecodeState::DeepseekV2(_)
    ));

    let muse = AutoCausalLm::<TestBackend>::from_config_json(
        &serde_json::to_string(&tiny_muse()).expect("serializable Muse config"),
        &device,
    )
    .expect("registered Muse config");
    assert_eq!(muse.config().model_type(), "muse_glimmer_text");
    assert!(matches!(
        muse.new_decode_state(
            CacheConfig::new(1, 8)
                .unwrap()
                .with_sliding_window(3)
                .unwrap(),
            1,
            &device,
        )
        .expect("Muse state"),
        AutoCausalDecodeState::MuseGlimmer(_)
    ));

    let nemotron = AutoCausalLm::<TestBackend>::from_config_json(
        &serde_json::to_string(&tiny_nemotron()).expect("serializable Nemotron config"),
        &device,
    )
    .expect("registered Nemotron config");
    assert_eq!(nemotron.config().model_type(), "nemotron_h");
    assert!(matches!(
        nemotron
            .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
            .expect("Nemotron state"),
        AutoCausalDecodeState::NemotronH(_)
    ));
}

/// The automatic causal enum can be passed directly to generic generation.
#[test]
fn causal_registry_implements_generation_contract_end_to_end() {
    let device = Default::default();
    let concrete = crate::models::deepseek_v2::model::tests_support::initialized_model(
        crate::models::deepseek_v2::model::tests_support::tiny_config(),
        &device,
    );
    let model = AutoCausalLm::from(concrete);
    let output = generate_causal(
        &model,
        CausalGenerationRequest {
            prompt: CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            cache: CacheConfig::new(1, 8).unwrap(),
            generation: GenerationConfig {
                max_new_tokens: 2,
                ..Default::default()
            },
            prefill_chunk_size: None,
            vocabulary_size: 32,
        },
    )
    .expect("auto-dispatched cached generation");
    assert_eq!(output.sequences[0].token_ids.len(), 2);
}

/// Cross-family state mismatches fail without advancing the foreign cache.
#[test]
fn causal_registry_rejects_mismatched_state_atomically() {
    let device = Default::default();
    let deepseek = AutoCausalLm::from(
        crate::models::deepseek_v2::model::tests_support::initialized_model(
            crate::models::deepseek_v2::model::tests_support::tiny_config(),
            &device,
        ),
    );
    let qwen = AutoCausalLm::from(
        Qwen3_5ForCausalLm::init(tiny_qwen(), &device).expect("tiny Qwen model"),
    );
    let mut foreign_state = qwen
        .new_decode_state(CacheConfig::new(1, 8).unwrap(), 1, &device)
        .unwrap();
    let error = deepseek
        .prefill(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &mut foreign_state,
            &ForwardOptions::default(),
        )
        .expect_err("foreign state must be rejected");
    assert!(matches!(error, Error::Cache(_)));
    assert_eq!(foreign_state.lengths(), [0]);
}

/// T5 registry initializes the graph and allocates architecture-matched state.
#[test]
fn seq2seq_registry_parses_t5_and_allocates_cached_state() {
    let device = Default::default();
    let json = serde_json::to_string(&tiny_t5()).expect("serializable config");
    let model = AutoSeq2SeqLm::<TestBackend>::from_config_json(&json, &device)
        .expect("registered T5 config");
    assert_eq!(model.config().model_type(), "t5");
    assert_eq!(model.config().vocabulary_size(), 32);
    let state = model
        .new_decode_state(CacheConfig::new(1, 8).expect("valid cache"), 1, &device)
        .expect("registered T5 state");
    assert_eq!(state.batch_size(), 1);
    assert_eq!(state.capacity(), 8);
}

/// Malformed, missing, and non-executable model types fail contextually.
#[test]
fn registry_rejects_unknown_or_malformed_configurations() {
    let device = Default::default();
    for json in ["{", "{}", r#"{"model_type": 7}"#] {
        assert!(AutoCausalLm::<TestBackend>::from_config_json(json, &device).is_err());
    }
    let error = AutoCausalLm::<TestBackend>::from_config_json(r#"{"model_type":"bert"}"#, &device)
        .expect_err("unregistered family must not be advertised as executable");
    assert!(matches!(error, Error::UnsupportedArchitecture(_)));
    assert!(
        AutoCausalLm::<TestBackend>::from_config_json(
            r#"{"model_type":"deepseek_v2","hidden_size":0}"#,
            &device,
        )
        .is_err()
    );
}

/// CPU device used to compare enum dispatch with the same concrete parameters.
type RegistryDevice = <TestBackend as BackendTypes>::Device;

/// Constructs two distinct token rows, preserving batch order explicitly.
fn registry_tokens<const S: usize>(
    ids: [[i32; S]; 2],
    device: &RegistryDevice,
) -> CausalLmInput<TestBackend> {
    CausalLmInput {
        token_ids: Tensor::from_data(
            TensorData::new(ids.into_iter().flatten().collect::<Vec<_>>(), [2, S]),
            device,
        ),
        token_mask: None,
        position_ids: None,
    }
}

/// Converts a continuation without changing any public input semantics.
fn registry_decode<const S: usize>(
    ids: [[i32; S]; 2],
    device: &RegistryDevice,
) -> DecodeInput<TestBackend> {
    let input = registry_tokens(ids, device);
    DecodeInput {
        token_ids: input.token_ids,
        token_mask: input.token_mask,
        position_ids: input.position_ids,
    }
}

/// Requires full-vocabulary parity, not merely agreement of the selected token.
#[track_caller]
fn assert_registry_logits(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(5.0e-4, 5.0e-5));
}

/// Requires the same typed error and complete contextual diagnostic after dispatch.
fn assert_registry_error(actual: Error, expected: Error) {
    assert_eq!(
        core::mem::discriminant(&actual),
        core::mem::discriminant(&expected)
    );
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

/// Materializes every parameter, including experts never selected by a prompt.
///
/// Burn deliberately clones an uninitialized parameter's initializer into an
/// independent cell. Such clones may later draw different random weights. A
/// sparse forward does not initialize skipped experts, so comparing two cloned
/// graphs requires traversing the entire parameter bank before cloning.
#[derive(Default)]
struct RegistryParameterMaterializer {
    /// Number of visited parameters, including those already initialized.
    visited: usize,
}

impl ModuleVisitor<TestBackend> for RegistryParameterMaterializer {
    /// Freezes every floating-point weight before graphs can diverge lazily.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D>>) {
        let _ = param.val();
        assert!(param.is_initialized());
        self.visited += 1;
    }

    /// Preserves the same materialization invariant for integer parameters.
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D, Int>>) {
        let _ = param.val();
        assert!(param.is_initialized());
        self.visited += 1;
    }

    /// Preserves the same materialization invariant for Boolean parameters.
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D, Bool>>) {
        let _ = param.val();
        assert!(param.is_initialized());
        self.visited += 1;
    }
}

/// Makes fixture clones share concrete parameter values, not lazy initializers.
fn materialize_registry_parameters<M: Module<TestBackend>>(model: &M) {
    let mut visitor = RegistryParameterMaterializer::default();
    model.visit(&mut visitor);
    assert!(
        visitor.visited > 0,
        "registry fixture must contain parameters"
    );
}

/// Exercises every causal trait entry point and nonempty cache lifecycle operation.
///
/// Every concrete parameter is materialized before cloning, including skipped
/// experts, so enum and direct calls share weights. Distinct rows expose
/// swapped-cache bugs; a fresh stateless computation additionally detects cache
/// damage after rejected operations.
fn assert_causal_dispatch_contract<M>(
    concrete: M,
    family: &str,
    cache: CacheConfig,
    can_rewind_nonzero: bool,
    device: &RegistryDevice,
) where
    M: CausalLanguageModel<TestBackend> + Clone,
    AutoCausalLm<TestBackend>: From<M>,
{
    materialize_registry_parameters(&concrete);
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let prompt = registry_tokens([[1, 2], [3, 4]], device);
    let initial = concrete.forward(prompt.clone(), &options).unwrap().logits;
    let automatic = AutoCausalLm::from(concrete.clone());
    assert_eq!(automatic.config().model_type(), family);
    assert_eq!(automatic.config().vocabulary_size(), 32);
    assert_registry_logits(
        CausalLanguageModel::forward(&automatic, prompt.clone(), &options)
            .unwrap()
            .logits,
        initial.clone(),
    );

    assert_registry_error(
        CausalLanguageModel::new_decode_state(&automatic, cache, 0, device).unwrap_err(),
        concrete.new_decode_state(cache, 0, device).unwrap_err(),
    );
    let mut state = CausalLanguageModel::new_decode_state(&automatic, cache, 2, device).unwrap();
    let mut direct = concrete.new_decode_state(cache, 2, device).unwrap();
    assert_eq!(state.batch_size(), 2);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.lengths(), [0, 0]);
    assert_eq!(state.device(), device);
    assert_registry_logits(
        CausalLanguageModel::prefill(&automatic, prompt.clone(), &mut state, &options)
            .unwrap()
            .logits,
        concrete
            .prefill(prompt.clone(), &mut direct, &options)
            .unwrap()
            .logits,
    );
    assert_eq!(state.lengths(), [2, 2]);
    assert_registry_logits(
        CausalLanguageModel::decode(
            &automatic,
            registry_decode([[5], [6]], device),
            &mut state,
            &options,
        )
        .unwrap()
        .logits,
        concrete
            .decode(registry_decode([[5], [6]], device), &mut direct, &options)
            .unwrap()
            .logits,
    );
    assert_eq!(state.lengths(), [3, 3]);

    state.reorder(&[1, 0]).unwrap();
    direct.reorder(&[1, 0]).unwrap();
    state.truncate(&[3, 3]).unwrap();
    direct.truncate(&[3, 3]).unwrap();
    for indices in [&[][..], &[2, 0][..]] {
        assert_registry_error(
            state.reorder(indices).unwrap_err(),
            direct.reorder(indices).unwrap_err(),
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
    for lengths in [&[3][..], &[4, 3][..]] {
        assert_registry_error(
            state.truncate(lengths).unwrap_err(),
            direct.truncate(lengths).unwrap_err(),
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
    // Group-preserving reorders reach the concrete state's own validation and
    // update path: a zero or non-dividing group, or an index leaving its group.
    for (indices, group_size) in [(&[1, 0][..], 0), (&[1, 0][..], 3), (&[1, 0][..], 1)] {
        assert_registry_error(
            state
                .reorder_within_groups(indices, group_size)
                .unwrap_err(),
            direct
                .reorder_within_groups(indices, group_size)
                .unwrap_err(),
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
    // Swapping back inside one group of two restores the prompt order.
    state.reorder_within_groups(&[1, 0], 2).unwrap();
    direct.reorder_within_groups(&[1, 0], 2).unwrap();
    let continuation = automatic
        .decode(registry_decode([[9], [10]], device), &mut state, &options)
        .unwrap()
        .logits;
    assert_registry_logits(
        continuation.clone(),
        concrete
            .decode(registry_decode([[9], [10]], device), &mut direct, &options)
            .unwrap()
            .logits,
    );
    assert_registry_logits(
        continuation,
        concrete
            .forward(
                registry_tokens([[1, 2, 5, 9], [3, 4, 6, 10]], device),
                &options,
            )
            .unwrap()
            .logits,
    );

    state.reorder(&[1, 1]).unwrap();
    direct.reorder(&[1, 1]).unwrap();
    let duplicated = automatic
        .decode(registry_decode([[11], [12]], device), &mut state, &options)
        .unwrap()
        .logits;
    assert_registry_logits(
        duplicated.clone(),
        concrete
            .decode(registry_decode([[11], [12]], device), &mut direct, &options)
            .unwrap()
            .logits,
    );
    assert_registry_logits(
        duplicated,
        concrete
            .forward(
                registry_tokens([[3, 4, 6, 10, 11], [3, 4, 6, 10, 12]], device),
                &options,
            )
            .unwrap()
            .logits,
    );
    assert_eq!(state.lengths(), [5, 5]);
    if can_rewind_nonzero {
        state.truncate(&[2, 2]).unwrap();
        direct.truncate(&[2, 2]).unwrap();
        assert_eq!(state.lengths(), [2, 2]);
        assert_registry_logits(
            automatic
                .decode(registry_decode([[7], [8]], device), &mut state, &options)
                .unwrap()
                .logits,
            concrete
                .forward(registry_tokens([[3, 4, 7], [3, 4, 8]], device), &options)
                .unwrap()
                .logits,
        );
    } else {
        assert_registry_error(
            state.truncate(&[2, 2]).unwrap_err(),
            direct.truncate(&[2, 2]).unwrap_err(),
        );
        assert_eq!(state.lengths(), [5, 5]);
    }
    state.reset().unwrap();
    direct.reset().unwrap();
    assert_eq!(state.batch_size(), 2);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.device(), device);
    assert_eq!(state.lengths(), [0, 0]);
    assert_registry_logits(
        automatic
            .prefill(prompt, &mut state, &options)
            .unwrap()
            .logits,
        initial,
    );
}

/// Hybrid Qwen dispatch preserves recurrent-state restrictions and row identity.
#[test]
fn qwen_hybrid_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    assert_causal_dispatch_contract(
        Qwen3_5ForCausalLm::init(tiny_qwen(), &device).unwrap(),
        "qwen3_5_text",
        CacheConfig::new(2, 8).unwrap(),
        false,
        &device,
    );
}

/// Sparse Qwen dispatch supports rollback of full-history attention state.
#[test]
fn qwen_moe_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    assert_causal_dispatch_contract(
        Qwen3MoeForCausalLm::init(tiny_qwen_moe(), &device).unwrap(),
        "qwen3_moe",
        CacheConfig::new(2, 8).unwrap(),
        true,
        &device,
    );
}

/// A deliberately skipped expert proves cloning safety without relying on RNG.
#[test]
fn registry_materialization_includes_experts_skipped_by_the_initial_prompt() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let device = Default::default();
    let mut concrete = Qwen3MoeForCausalLm::<TestBackend>::init(tiny_qwen_moe(), &device).unwrap();
    let initializations = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&initializations);
    let moe = concrete.model.layers[0].sparse_mlp.as_mut().unwrap();
    // Equal router logits deterministically select experts 0 and 1, leaving
    // expert 3 untouched regardless of process-wide random initialization.
    moe.router.weight = Param::from_tensor(Tensor::zeros([8, 4], &device));
    moe.routed[3].gate_proj.weight = Param::uninitialized(
        ParamId::new(),
        move |device, _| {
            let value = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Tensor::full([8, 12], value as f32, device)
        },
        device,
        false,
        [8, 12].into(),
    );
    concrete
        .forward(
            registry_tokens([[1, 2], [3, 4]], &device),
            &ForwardOptions::default(),
        )
        .unwrap();
    assert_eq!(initializations.load(Ordering::SeqCst), 0);
    let skipped = &concrete.model.layers[0].sparse_mlp.as_ref().unwrap().routed[3];
    assert!(!skipped.gate_proj.weight.is_initialized());
    assert!(!skipped.up_proj.weight.is_initialized());
    assert!(!skipped.down_proj.weight.is_initialized());
    let lazy_clone = concrete.clone();

    materialize_registry_parameters(&concrete);
    let prepared_clone = AutoCausalLm::from(concrete.clone());
    let AutoCausalLm::Qwen3Moe(prepared) = prepared_clone else {
        unreachable!("the fixture converts into the Qwen3-MoE variant")
    };
    let copied = &prepared.model.layers[0].sparse_mlp.as_ref().unwrap().routed[3];
    for expert in [skipped, copied] {
        assert!(expert.gate_proj.weight.is_initialized());
        assert!(expert.up_proj.weight.is_initialized());
        assert!(expert.down_proj.weight.is_initialized());
    }
    assert_eq!(initializations.load(Ordering::SeqCst), 1);
    assert_eq!(
        skipped.gate_proj.weight.val().into_data(),
        copied.gate_proj.weight.val().into_data(),
    );
    assert_registry_logits(
        copied.forward(Tensor::ones([2, 1, 8], &device)).unwrap(),
        skipped.forward(Tensor::ones([2, 1, 8], &device)).unwrap(),
    );

    // The pre-materialization clone still owns a separate lazy cell. Its
    // controlled initializer returns 2 instead of 1, demonstrating the old
    // fixture's failure mode even if random weights happen to match by chance.
    let uncopied = &lazy_clone.model.layers[0]
        .sparse_mlp
        .as_ref()
        .unwrap()
        .routed[3];
    assert!(!uncopied.gate_proj.weight.is_initialized());
    assert_eq!(
        skipped.gate_proj.weight.val().into_data(),
        TensorData::new(vec![1.0_f32; 96], [8, 12]),
    );
    assert_eq!(
        uncopied.gate_proj.weight.val().into_data(),
        TensorData::new(vec![2.0_f32; 96], [8, 12]),
    );
    assert_eq!(initializations.load(Ordering::SeqCst), 2);
}

/// MLA dispatch keeps compressed-cache metadata and tensor row order coherent.
#[test]
fn deepseek_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    assert_causal_dispatch_contract(
        crate::models::deepseek_v2::model::tests_support::initialized_model(
            crate::models::deepseek_v2::model::tests_support::tiny_config(),
            &device,
        ),
        "deepseek_v2",
        CacheConfig::new(2, 8).unwrap(),
        true,
        &device,
    );
}

/// Muse dispatch rejects rewinding beyond retained sliding-ring history.
#[test]
fn muse_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    assert_causal_dispatch_contract(
        MuseGlimmerForCausalLm::init(tiny_muse(), &device).unwrap(),
        "muse_glimmer_text",
        CacheConfig::new(2, 8)
            .unwrap()
            .with_sliding_window(3)
            .unwrap(),
        false,
        &device,
    );
}

/// Nemotron dispatch preserves the Mamba recurrent rewind restriction.
#[test]
fn nemotron_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    assert_causal_dispatch_contract(
        NemotronHForCausalLm::init(tiny_nemotron(), &device).unwrap(),
        "nemotron_h",
        CacheConfig::new(2, 8).unwrap(),
        false,
        &device,
    );
}

/// Every cross-family pairing fails before touching the owning model's live cache.
#[test]
fn all_foreign_causal_state_pairs_reject_prefill_and_decode_atomically() {
    let device = Default::default();
    let models: Vec<AutoCausalLm<TestBackend>> = vec![
        Qwen3_5ForCausalLm::init(tiny_qwen(), &device)
            .unwrap()
            .into(),
        Qwen3MoeForCausalLm::init(tiny_qwen_moe(), &device)
            .unwrap()
            .into(),
        crate::models::deepseek_v2::model::tests_support::initialized_model(
            crate::models::deepseek_v2::model::tests_support::tiny_config(),
            &device,
        )
        .into(),
        MuseGlimmerForCausalLm::init(tiny_muse(), &device)
            .unwrap()
            .into(),
        NemotronHForCausalLm::init(tiny_nemotron(), &device)
            .unwrap()
            .into(),
    ];
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    for (owner_index, owner) in models.iter().enumerate() {
        let cache = if owner.config().model_type() == "muse_glimmer_text" {
            CacheConfig::new(2, 8)
                .unwrap()
                .with_sliding_window(3)
                .unwrap()
        } else {
            CacheConfig::new(2, 8).unwrap()
        };
        let mut state = owner.new_decode_state(cache, 2, &device).unwrap();
        owner
            .prefill(
                registry_tokens([[1, 2], [3, 4]], &device),
                &mut state,
                &options,
            )
            .unwrap();
        for (foreign_index, foreign) in models.iter().enumerate() {
            if owner_index == foreign_index {
                continue;
            }
            let expected = || {
                Error::Cache(
                    "automatic causal decode state belongs to a different architecture".into(),
                )
            };
            assert_registry_error(
                foreign
                    .prefill(registry_tokens([[5], [6]], &device), &mut state, &options)
                    .unwrap_err(),
                expected(),
            );
            assert_registry_error(
                foreign
                    .decode(registry_decode([[5], [6]], &device), &mut state, &options)
                    .unwrap_err(),
                expected(),
            );
            assert_eq!(state.lengths(), [2, 2]);
        }
        assert_registry_logits(
            owner
                .decode(registry_decode([[5], [6]], &device), &mut state, &options)
                .unwrap()
                .logits,
            owner
                .forward(registry_tokens([[1, 2, 5], [3, 4, 6]], &device), &options)
                .unwrap()
                .logits,
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
}

/// T5's enum forwards encoder, decoder, state metadata, and cache lifecycle calls.
#[test]
fn seq2seq_auto_trait_dispatch_and_state_lifecycle() {
    let device = Default::default();
    let concrete = T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap();
    materialize_registry_parameters(&concrete);
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let source = registry_tokens([[8, 9, 10], [11, 12, 13]], &device);
    let decoder = registry_tokens([[0, 4], [0, 5]], &device);
    let inputs = Seq2SeqInput {
        encoder: source.clone(),
        decoder: decoder.clone(),
    };
    let expected = concrete.forward(inputs.clone(), &options).unwrap();
    let automatic = AutoSeq2SeqLm::from(concrete.clone());
    assert_eq!(automatic.config().model_type(), "t5");
    assert_eq!(automatic.config().vocabulary_size(), 32);
    let actual = Seq2SeqLanguageModel::forward(&automatic, inputs, &options).unwrap();
    assert_registry_logits(actual.decoder.logits, expected.decoder.logits);
    assert_registry_logits(actual.encoder.hidden_state, expected.encoder.hidden_state);
    let encoder = Seq2SeqLanguageModel::encode(&automatic, source.clone(), &options).unwrap();
    assert_registry_logits(
        encoder.hidden_state.clone(),
        concrete
            .encode(source.clone(), &options)
            .unwrap()
            .hidden_state,
    );

    let cache = CacheConfig::new(2, 8).unwrap();
    assert_registry_error(
        Seq2SeqLanguageModel::new_decode_state(&automatic, cache, 0, &device).unwrap_err(),
        Seq2SeqLanguageModel::new_decode_state(&concrete, cache, 0, &device).unwrap_err(),
    );
    let mut state = Seq2SeqLanguageModel::new_decode_state(&automatic, cache, 2, &device).unwrap();
    let mut direct = Seq2SeqLanguageModel::new_decode_state(&concrete, cache, 2, &device).unwrap();
    assert_eq!(state.batch_size(), 2);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.lengths(), [0, 0]);
    assert_eq!(state.device(), &device);
    assert_registry_logits(
        Seq2SeqLanguageModel::prefill_decoder(
            &automatic,
            &encoder,
            decoder.clone(),
            &mut state,
            &options,
        )
        .unwrap()
        .decoder
        .logits,
        Seq2SeqLanguageModel::prefill_decoder(&concrete, &encoder, decoder, &mut direct, &options)
            .unwrap()
            .decoder
            .logits,
    );
    assert_registry_logits(
        Seq2SeqLanguageModel::decode(
            &automatic,
            &encoder,
            registry_decode([[6], [7]], &device),
            &mut state,
            &options,
        )
        .unwrap()
        .decoder
        .logits,
        Seq2SeqLanguageModel::decode(
            &concrete,
            &encoder,
            registry_decode([[6], [7]], &device),
            &mut direct,
            &options,
        )
        .unwrap()
        .decoder
        .logits,
    );
    assert_eq!(state.lengths(), [3, 3]);
    for indices in [&[][..], &[2, 0][..]] {
        assert_registry_error(
            state.reorder(indices).unwrap_err(),
            DecodeState::reorder(&mut direct, indices).unwrap_err(),
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
    for lengths in [&[3][..], &[4, 3][..]] {
        assert_registry_error(
            state.truncate(lengths).unwrap_err(),
            DecodeState::truncate(&mut direct, lengths).unwrap_err(),
        );
        assert_eq!(state.lengths(), [3, 3]);
    }
    state.reorder(&[1, 0]).unwrap();
    DecodeState::reorder(&mut direct, &[1, 0]).unwrap();
    let selector = Tensor::<TestBackend, 1, Int>::from_ints([1, 0], &device);
    let mut swapped_encoder = encoder.clone();
    swapped_encoder.hidden_state = swapped_encoder.hidden_state.select(0, selector);
    assert_registry_logits(
        automatic
            .decode(
                &swapped_encoder,
                registry_decode([[8], [9]], &device),
                &mut state,
                &options,
            )
            .unwrap()
            .decoder
            .logits,
        Seq2SeqLanguageModel::decode(
            &concrete,
            &swapped_encoder,
            registry_decode([[8], [9]], &device),
            &mut direct,
            &options,
        )
        .unwrap()
        .decoder
        .logits,
    );
    state.truncate(&[1, 1]).unwrap();
    assert_eq!(state.lengths(), [1, 1]);
    assert_registry_logits(
        automatic
            .decode(
                &swapped_encoder,
                registry_decode([[10], [11]], &device),
                &mut state,
                &options,
            )
            .unwrap()
            .decoder
            .logits,
        concrete
            .forward(
                Seq2SeqInput {
                    encoder: registry_tokens([[11, 12, 13], [8, 9, 10]], &device),
                    decoder: registry_tokens([[0, 10], [0, 11]], &device),
                },
                &options,
            )
            .unwrap()
            .decoder
            .logits,
    );
    state.reset().unwrap();
    assert_eq!(state.lengths(), [0, 0]);
    assert_eq!(state.batch_size(), 2);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.device(), &device);
    assert_registry_logits(
        automatic
            .prefill_decoder(
                &encoder,
                registry_tokens([[0], [0]], &device),
                &mut state,
                &options,
            )
            .unwrap()
            .decoder
            .logits,
        concrete
            .forward(
                Seq2SeqInput {
                    encoder: source,
                    decoder: registry_tokens([[0], [0]], &device),
                },
                &options,
            )
            .unwrap()
            .decoder
            .logits,
    );
}

/// Group-preserving reorders reach T5's in-place path through the automatic state.
///
/// T5 moves only decoder self-attention history on
/// [`DecodeState::reorder_within_groups`] and leaves each group's bound encoder
/// output and cross-attention keys and values in place, because beam search
/// promises that the rows of one group share a source. Two *different* sources
/// in one group make the path observable: after swapping the rows, the next
/// step attends to the unmoved binding. The automatic state must give the
/// concrete state's logits and errors; the trait default (a full `reorder`,
/// which also moves the binding) gives visibly different logits.
#[test]
fn seq2seq_auto_state_forwards_group_preserving_reorder() {
    let device = Default::default();
    let concrete = T5ForConditionalGeneration::<TestBackend>::init(tiny_t5(), &device).unwrap();
    materialize_registry_parameters(&concrete);
    let automatic = AutoSeq2SeqLm::from(concrete.clone());
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let encoder = concrete
        .encode(
            registry_tokens([[8, 9, 10], [11, 12, 13]], &device),
            &options,
        )
        .unwrap();
    let cache = CacheConfig::new(2, 8).unwrap();
    let mut state = automatic.new_decode_state(cache, 2, &device).unwrap();
    let mut grouped = Seq2SeqLanguageModel::new_decode_state(&concrete, cache, 2, &device).unwrap();
    let mut full = Seq2SeqLanguageModel::new_decode_state(&concrete, cache, 2, &device).unwrap();
    let decoder = registry_tokens([[0, 4], [0, 5]], &device);
    automatic
        .prefill_decoder(&encoder, decoder.clone(), &mut state, &options)
        .unwrap();
    for direct in [&mut grouped, &mut full] {
        Seq2SeqLanguageModel::prefill_decoder(
            &concrete,
            &encoder,
            decoder.clone(),
            direct,
            &options,
        )
        .unwrap();
    }

    for (indices, group_size) in [(&[1, 0][..], 0), (&[1, 0][..], 3), (&[1, 0][..], 1)] {
        assert_registry_error(
            state
                .reorder_within_groups(indices, group_size)
                .unwrap_err(),
            DecodeState::reorder_within_groups(&mut grouped, indices, group_size).unwrap_err(),
        );
        assert_eq!(state.lengths(), [2, 2]);
    }

    state.reorder_within_groups(&[1, 0], 2).unwrap();
    DecodeState::reorder_within_groups(&mut grouped, &[1, 0], 2).unwrap();
    DecodeState::reorder(&mut full, &[1, 0]).unwrap();
    let step = || registry_decode([[6], [7]], &device);
    let actual = automatic
        .decode(&encoder, step(), &mut state, &options)
        .unwrap()
        .decoder
        .logits;
    let in_place =
        Seq2SeqLanguageModel::decode(&concrete, &encoder, step(), &mut grouped, &options)
            .unwrap()
            .decoder
            .logits;
    let moved = Seq2SeqLanguageModel::decode(&concrete, &encoder, step(), &mut full, &options)
        .unwrap()
        .decoder
        .logits;
    assert_eq!(state.lengths(), [3, 3]);
    let separation = (in_place.clone() - moved).abs().max().into_scalar();
    assert!(
        separation > 1.0e-3,
        "distinct sources must make the in-place and full reorders differ, max |diff| {separation}"
    );
    assert_registry_logits(actual, in_place);
}

/// Release timings of one beam step's reorder through the automatic T5 state (SI-1).
///
/// FLAN-T5-small decoder geometry (8 layers, 6 heads, `d_kv` 64) with one
/// source expanded to four beams over 512 source tokens. Before the
/// automatic state forwarded `reorder_within_groups`, the trait default ran
/// the full `reorder`, which also re-gathers the bound encoder output and
/// every layer's projected cross-attention keys and values; that former path
/// is timed directly. Reproduce with
/// `cargo test --release --lib -- --ignored --nocapture t5_beam_reorder_timings`.
#[test]
#[ignore = "release-mode timing; run explicitly"]
fn t5_beam_reorder_timings() {
    /// Beams per source, as in the README's FLAN-T5 example.
    const BEAMS: usize = 4;
    /// Source tokens, enough for cross-attention gathers to dominate a full reorder.
    const SOURCE: usize = 512;
    let device = Default::default();
    let concrete = T5ForConditionalGeneration::<TestBackend>::init(
        T5Config {
            vocab_size: 32_128,
            d_model: 512,
            d_kv: 64,
            d_ff: 1_024,
            num_layers: 8,
            num_decoder_layers: Some(8),
            num_heads: 6,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.0,
            feed_forward_proj: T5FeedForwardProjection::GatedGelu,
            ..T5Config::default()
        },
        &device,
    )
    .unwrap();
    materialize_registry_parameters(&concrete);
    let automatic = AutoSeq2SeqLm::from(concrete.clone());
    let options = ForwardOptions {
        logits: LogitsSelection::Last,
        ..Default::default()
    };
    let source = CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                (0..SOURCE as i32)
                    .map(|id| 3 + id % 1_000)
                    .collect::<Vec<_>>(),
                [1, SOURCE],
            ),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let mut encoder = concrete.encode(source, &options).unwrap();
    encoder.hidden_state = encoder
        .hidden_state
        .select(0, Tensor::<TestBackend, 1, Int>::zeros([BEAMS], &device));
    let cache = CacheConfig::new(BEAMS, 16).unwrap();
    let start = || CausalLmInput {
        token_ids: Tensor::<TestBackend, 2, Int>::zeros([BEAMS, 1], &device),
        token_mask: None,
        position_ids: None,
    };
    let mut state = automatic.new_decode_state(cache, BEAMS, &device).unwrap();
    automatic
        .prefill_decoder(&encoder, start(), &mut state, &options)
        .unwrap();
    let mut direct =
        Seq2SeqLanguageModel::new_decode_state(&concrete, cache, BEAMS, &device).unwrap();
    Seq2SeqLanguageModel::prefill_decoder(&concrete, &encoder, start(), &mut direct, &options)
        .unwrap();
    let median_ms = |run: &mut dyn FnMut()| {
        run();
        let mut samples = (0..21)
            .map(|_| {
                let begin = std::time::Instant::now();
                run();
                TestBackend::sync(&device).unwrap();
                begin.elapsed().as_secs_f64() * 1e3
            })
            .collect::<Vec<_>>();
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    let parents = [1, 0, 3, 2];
    let former = median_ms(&mut || DecodeState::reorder(&mut state, &parents).unwrap());
    let automatic_grouped =
        median_ms(&mut || state.reorder_within_groups(&parents, BEAMS).unwrap());
    let concrete_grouped = median_ms(&mut || {
        DecodeState::reorder_within_groups(&mut direct, &parents, BEAMS).unwrap();
    });
    let step = median_ms(&mut || {
        let next = DecodeInput {
            token_ids: Tensor::<TestBackend, 2, Int>::ones([BEAMS, 1], &device),
            token_mask: None,
            position_ids: None,
        };
        Seq2SeqLanguageModel::decode(&concrete, &encoder, next, &mut direct, &options)
            .unwrap()
            .decoder
            .logits
            .into_data();
        DecodeState::truncate(&mut direct, &[1; BEAMS]).unwrap();
    });
    println!(
        "SI-1 T5 beam reorder ({BEAMS} beams, {SOURCE} source tokens): former automatic {former:.3} ms, \
         automatic {automatic_grouped:.3} ms, concrete {concrete_grouped:.3} ms; one decode step {step:.3} ms"
    );
}

/// Names of the `fn` items declared directly inside the block opened by `header`.
///
/// A brace-depth scan over the source text: `fn` lines count only at depth one,
/// so default method bodies and nested items are ignored. Format strings in the
/// scanned blocks hold balanced `{}` pairs, which keep the depth exact.
fn block_function_names(source: &str, header: &str) -> Vec<String> {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("source lacks `{header}`"));
    let mut depth = 1_usize;
    let mut names = Vec::new();
    for line in source[start + header.len()..].lines().skip(1) {
        let trimmed = line.trim_start();
        if depth == 1
            && let Some(signature) = trimmed.strip_prefix("fn ")
        {
            let name = signature.split(['(', '<']).next().unwrap_or_default();
            names.push(name.to_owned());
        }
        if trimmed.starts_with("//") {
            continue;
        }
        for character in line.chars() {
            match character {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        if depth == 0 {
            names.sort();
            return names;
        }
    }
    panic!("block opened by `{header}` is not closed");
}

/// Both automatic decode states forward every [`DecodeState`] method, provided ones included.
///
/// A provided method that an enum does not forward silently runs the trait
/// default on the enum instead of the family's override, as
/// `reorder_within_groups` once did for T5. This structural check compares the
/// trait's method list with each dispatch implementation, so adding a trait
/// method fails here until both enums forward it.
#[test]
fn automatic_decode_states_forward_every_trait_method() {
    let trait_methods = block_function_names(
        include_str!("../architecture/traits.rs"),
        "pub trait DecodeState<B: Backend>: Send + core::fmt::Debug {",
    );
    assert!(
        trait_methods.contains(&"reorder_within_groups".to_owned()),
        "{trait_methods:?}"
    );
    let registry = include_str!("auto.rs");
    for state in ["AutoCausalDecodeState", "AutoSeq2SeqDecodeState"] {
        let forwarded = block_function_names(
            registry,
            &format!("impl<B: Backend> DecodeState<B> for {state}<B> {{"),
        );
        assert_eq!(
            forwarded, trait_methods,
            "{state} must forward every method"
        );
    }
}

/// Registry configuration traits enumerate every executable discriminator explicitly.
#[test]
fn automatic_config_discriminators_validate_and_match_runtime_registries() {
    assert_eq!(
        AutoCausalConfig::ALL,
        [
            AutoCausalConfig::DeepseekV2,
            AutoCausalConfig::MuseGlimmer,
            AutoCausalConfig::NemotronH,
            AutoCausalConfig::Qwen3_5,
            AutoCausalConfig::Qwen3Moe,
            AutoCausalConfig::GptOss,
            AutoCausalConfig::Llama,
            AutoCausalConfig::Gemma,
        ]
    );
    for config in AutoCausalConfig::ALL {
        config.validate().unwrap();
        assert_eq!(
            AutoCausalConfig::from_model_type(config.model_type()),
            Some(*config)
        );
        assert!(AutoCausalConfig::MODEL_TYPES.contains(&config.model_type()));
    }
    // Primary types and aliases each resolve to exactly one family.
    for model_type in AutoCausalConfig::MODEL_TYPES {
        assert!(
            AutoCausalConfig::from_model_type(model_type).is_some(),
            "{model_type}"
        );
    }
    for alias in ["mistral", "qwen2", "qwen3", "smollm3", "olmo2"] {
        assert_eq!(
            AutoCausalConfig::from_model_type(alias),
            Some(AutoCausalConfig::Llama)
        );
    }
    for model_type in ["gemma2", "gemma3_text"] {
        assert_eq!(
            AutoCausalConfig::from_model_type(model_type),
            Some(AutoCausalConfig::Gemma)
        );
    }
    // Multimodal wrappers are routed by their text_config, never by their own type.
    for unregistered in ["gemma3", "gemma3n_text"] {
        assert_eq!(AutoCausalConfig::from_model_type(unregistered), None);
    }
    let mut unique = AutoCausalConfig::MODEL_TYPES.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        AutoCausalConfig::MODEL_TYPES.len(),
        "every registered model_type is listed once"
    );
    AutoSeq2SeqConfig::T5.validate().unwrap();
    assert_eq!(
        AutoSeq2SeqConfig::MODEL_TYPES,
        AutoSeq2SeqLm::<TestBackend>::MODEL_TYPES
    );
}

/// Null wrappers fall through, while the first non-null wrapper owns model detection.
#[test]
fn model_detection_wrapper_precedence_and_typed_failures_are_stable() {
    for (json, expected) in [
        (
            r#"{"model_type":"direct","text_config":null,"language_config":{"model_type":"language"},"llm_config":{"model_type":"llm"}}"#,
            "language",
        ),
        (
            r#"{"model_type":"direct","text_config":null,"language_config":null,"llm_config":{"model_type":"llm"}}"#,
            "llm",
        ),
        (
            r#"{"model_type":"direct","text_config":null,"language_config":null,"llm_config":null}"#,
            "direct",
        ),
        (
            r#"{"model_type":"direct","text_config":{"model_type":"text"},"language_config":{"model_type":"language"}}"#,
            "text",
        ),
    ] {
        assert_eq!(detect_model_type(json).unwrap(), expected);
    }
    for json in [
        "null",
        "[]",
        "7",
        r#"{"model_type":false}"#,
        r#"{"model_type":"direct","text_config":{}}"#,
        r#"{"model_type":"direct","text_config":7}"#,
    ] {
        assert!(
            matches!(detect_model_type(json).unwrap_err(), Error::Configuration { field, reason }
            if field == "auto.model_type" && reason == "configuration must contain a string model_type")
        );
    }
    assert!(
        matches!(detect_model_type("{").unwrap_err(), Error::Configuration { field, reason }
        if field == "auto.config_json" && reason.starts_with("invalid JSON:"))
    );
}

/// Each family parser's errors convert through `From<ConfigLoadError>` without allocating weights.
///
/// A malformed field keeps the parser's diagnostic under the field `config`, and a
/// topology violation keeps its structured `architecture.field` path, so the
/// automatic registry reports exactly what the family parser found.
#[test]
fn all_registry_parsers_preserve_configuration_error_context() {
    let device = Default::default();
    for family in AutoCausalLm::<TestBackend>::MODEL_TYPES {
        let json =
            serde_json::json!({"model_type":family,"vocab_size":"not an integer"}).to_string();
        assert!(
            matches!(AutoCausalLm::<TestBackend>::from_config_json(&json, &device).unwrap_err(),
            Error::Configuration { field, reason } if field == "config"
                && reason.starts_with("invalid model config JSON: ") && reason.contains("not an integer")),
            "{family}"
        );
    }
    let invalid_topology = serde_json::to_string(&Qwen3MoeConfig {
        vocab_size: 0,
        ..tiny_qwen_moe()
    })
    .unwrap();
    assert!(
        matches!(AutoCausalLm::<TestBackend>::from_config_json(&invalid_topology, &device).unwrap_err(),
        Error::Configuration { field, reason } if field == "qwen3_moe.vocab_size" && reason == "must be nonzero")
    );
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::from_config_json(r#"{"model_type":"t5","vocab_size":"invalid"}"#, &device).unwrap_err(),
        Error::Configuration { field, reason } if field == "config" && reason.starts_with("invalid model config JSON: "))
    );
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::from_config_json(r#"{"model_type":"qwen3_moe"}"#, &device).unwrap_err(),
        Error::UnsupportedArchitecture(reason) if reason == "seq2seq model_type `qwen3_moe` is not executable; supported values: t5")
    );
}

/// Detection accepts NVIDIA's legacy `Infinity` literal exactly as the family parser does (AUTO-1).
#[test]
fn detection_accepts_the_legacy_nemotron_infinity_literal() {
    let fixture = include_str!("../../tests/fixtures/nemotron_h_8b.json");
    assert!(fixture.contains("Infinity"));
    assert_eq!(detect_model_type(fixture).unwrap(), "nemotron_h");

    let json = serde_json::to_string(&tiny_nemotron()).unwrap();
    let legacy = json.replace(
        "\"time_step_limit\":[0.0,null]",
        "\"time_step_limit\":[0.0,Infinity]",
    );
    assert_ne!(legacy, json, "tiny config must contain the unbounded limit");
    let model = AutoCausalLm::<TestBackend>::from_config_json(&legacy, &Default::default())
        .expect("legacy literal must reach the Nemotron-H parser");
    assert!(matches!(model, AutoCausalLm::NemotronH(_)));

    // Quoted text is never rewritten, and wrapped configs are normalized too.
    assert_eq!(
        detect_model_type(r#"{"model_type":"Infinity","limit":Infinity}"#).unwrap(),
        "Infinity"
    );
    assert_eq!(
        detect_model_type(r#"{"text_config":{"model_type":"nemotron_h","limit":[0,Infinity]}}"#)
            .unwrap(),
        "nemotron_h"
    );
    assert!(
        matches!(detect_model_type("{\"model_type\":-Infinity}").unwrap_err(),
        Error::Configuration { field, .. } if field == "auto.config_json")
    );
}

/// Each family's registry discriminator is the one its own configuration schema accepts,
/// and the model, borrowed config, and family views agree.
#[test]
fn registry_entries_agree_with_family_schemas_and_views() {
    let device = Default::default();
    let entries = [
        (
            AutoCausalConfig::DeepseekV2,
            <DeepseekV2Config as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::MuseGlimmer,
            <MuseGlimmerTextConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::NemotronH,
            <NemotronHConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::Qwen3_5,
            <Qwen3_5TextConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::Qwen3Moe,
            <Qwen3MoeConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::GptOss,
            <crate::models::gpt_oss::GptOssConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::Llama,
            <crate::models::llama::LlamaFamilyConfig as ModelConfig>::MODEL_TYPES,
        ),
        (
            AutoCausalConfig::Gemma,
            <crate::models::gemma::GemmaConfig as ModelConfig>::MODEL_TYPES,
        ),
    ];
    assert_eq!(entries.len(), AutoCausalConfig::ALL.len());
    for (family, schema_types) in entries {
        assert!(schema_types.contains(&family.model_type()), "{family:?}");
        // The registry routes exactly the types the family's schema accepts:
        // an alias missing from the registry would make a variant unreachable.
        let mut routed: Vec<&str> = AutoCausalConfig::MODEL_TYPES
            .iter()
            .copied()
            .filter(|model_type| AutoCausalConfig::from_model_type(model_type) == Some(family))
            .collect();
        let mut accepted = schema_types.to_vec();
        routed.sort_unstable();
        accepted.sort_unstable();
        assert_eq!(routed, accepted, "{family:?}");
    }
    for (json, family) in [
        (
            serde_json::to_string(&tiny_qwen()).unwrap(),
            AutoCausalConfig::Qwen3_5,
        ),
        (
            serde_json::to_string(&tiny_qwen_moe()).unwrap(),
            AutoCausalConfig::Qwen3Moe,
        ),
        (
            serde_json::to_string(&tiny_muse()).unwrap(),
            AutoCausalConfig::MuseGlimmer,
        ),
        (
            serde_json::to_string(&tiny_nemotron()).unwrap(),
            AutoCausalConfig::NemotronH,
        ),
    ] {
        let model = AutoCausalLm::<TestBackend>::from_config_json(&json, &device).unwrap();
        assert_eq!(model.family(), family);
        assert_eq!(model.config().family(), family);
        assert_eq!(model.config().model_type(), family.model_type());
    }
}

/// The dense Llama, Gemma, and gpt-oss families initialize from their committed configs.
///
/// Every Llama-family alias selects the shared Llama graph, and each config
/// selects its own family and decode-state variant.
#[test]
fn dense_family_configs_initialize_through_the_registry() {
    let device = Default::default();
    for (json, family) in [
        (
            include_str!("../../tests/fixtures/tiny_goldens/llama3/config.json"),
            AutoCausalConfig::Llama,
        ),
        (
            include_str!("../../tests/fixtures/tiny_goldens/mistral-sliding/config.json"),
            AutoCausalConfig::Llama,
        ),
        (
            include_str!("../../tests/fixtures/tiny_goldens/olmo2/config.json"),
            AutoCausalConfig::Llama,
        ),
        (
            include_str!("../../tests/fixtures/tiny_goldens/gemma2/config.json"),
            AutoCausalConfig::Gemma,
        ),
        (
            include_str!("../../tests/fixtures/tiny_goldens/gemma3-text/config.json"),
            AutoCausalConfig::Gemma,
        ),
        (
            include_str!("../../tests/fixtures/tiny_goldens/gpt-oss/config.json"),
            AutoCausalConfig::GptOss,
        ),
    ] {
        let model = AutoCausalLm::<TestBackend>::from_config_json(json, &device).unwrap();
        assert_eq!(model.family(), family);
        assert_eq!(model.config().family(), family);
        let declared: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            model.config().model_type(),
            family.model_type(),
            "{declared}"
        );
        assert_eq!(
            AutoCausalConfig::from_model_type(declared["model_type"].as_str().unwrap()),
            Some(family)
        );
        let state = model
            .new_decode_state(CacheConfig::new(1, 4).unwrap(), 1, &device)
            .unwrap();
        let (graph, cache) = match family {
            AutoCausalConfig::Llama => (
                matches!(model, AutoCausalLm::Llama(_)),
                matches!(state, AutoCausalDecodeState::Llama(_)),
            ),
            AutoCausalConfig::Gemma => (
                matches!(model, AutoCausalLm::Gemma(_)),
                matches!(state, AutoCausalDecodeState::Gemma(_)),
            ),
            AutoCausalConfig::GptOss => (
                matches!(model, AutoCausalLm::GptOss(_)),
                matches!(state, AutoCausalDecodeState::GptOss(_)),
            ),
            other => panic!("unexpected dense family {other:?}"),
        };
        assert!(graph && cache, "{declared}: graph {graph}, cache {cache}");
        assert_eq!((state.batch_size(), state.capacity()), (1, 4));
        assert_eq!(state.lengths(), [0]);
    }
}

/// The error a family without experiment support would raise names it and the supported ones.
///
/// Every registered family currently supports experiments, so this pins the
/// contract a `pending` registry entry would inherit: the rejection names the
/// model type, lists the supported families, and clears earlier captures.
#[test]
fn pending_experiment_rejections_name_the_family_and_clear_captures() {
    let device = Default::default();
    let model = AutoCausalLm::<TestBackend>::from_config_json(
        include_str!("../../tests/fixtures/tiny_goldens/llama3/config.json"),
        &device,
    )
    .unwrap();
    let error = model.experiments_unavailable();
    match &error {
        Error::BackendCapability(reason) => {
            assert!(reason.contains("model_type `llama`"), "{reason}");
            for family in AutoCausalConfig::ALL {
                assert!(reason.contains(family.model_type()), "{reason}");
            }
        }
        other => panic!("expected a capability error, found {other:?}"),
    }
    let mut hooks = Hooks::new();
    let embedding = hooks.capture(crate::intervention::Stack::Decoder.embedding());
    HookedModel::forward_with_hooks(
        &model,
        CausalLmInput {
            token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
            token_mask: None,
            position_ids: None,
        },
        &ForwardOptions::default(),
        &mut hooks,
    )
    .unwrap();
    assert!(hooks.get(&embedding).is_ok());
    let message = error.to_string();
    let rejected: Result<()> = reject_hooks(&mut hooks, error);
    assert_eq!(rejected.unwrap_err().to_string(), message);
    assert!(
        hooks.get(&embedding).is_err(),
        "the rejection cleared old captures"
    );
}
