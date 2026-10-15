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
        vocabulary_size: 32,
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

/// Every concrete causal family supports the bounded-prefill coordinator contract.
#[test]
fn every_causal_family_matches_bounded_and_one_shot_prefill() {
    let device = Default::default();

    TestBackend::seed(&device, 10_001);
    let qwen = Qwen3_5ForCausalLm::init(tiny_qwen(), &device).expect("tiny Qwen");
    assert_bounded_prefill_contract("Qwen3.5", &qwen, CacheConfig::new(2, 8).unwrap(), &device);

    TestBackend::seed(&device, 10_002);
    let qwen_moe = Qwen3MoeForCausalLm::init(tiny_qwen_moe(), &device).expect("tiny Qwen MoE");
    assert_bounded_prefill_contract(
        "Qwen3-MoE",
        &qwen_moe,
        CacheConfig::new(2, 8).unwrap(),
        &device,
    );

    let deepseek = crate::models::deepseek_v2::model::tests_support::initialized_model(
        crate::models::deepseek_v2::model::tests_support::tiny_config(),
        &device,
    );
    assert_bounded_prefill_contract(
        "DeepSeek-V2",
        &deepseek,
        CacheConfig::new(2, 8).unwrap(),
        &device,
    );

    TestBackend::seed(&device, 10_003);
    let muse = MuseGlimmerForCausalLm::init(tiny_muse(), &device).expect("tiny Muse");
    assert_bounded_prefill_contract(
        "Muse Glimmer",
        &muse,
        CacheConfig::new(2, 8)
            .unwrap()
            .with_sliding_window(3)
            .unwrap(),
        &device,
    );

    TestBackend::seed(&device, 10_004);
    let nemotron = NemotronHForCausalLm::init(tiny_nemotron(), &device).expect("tiny Nemotron-H");
    assert_bounded_prefill_contract(
        "Nemotron-H",
        &nemotron,
        CacheConfig::new(2, 8).unwrap(),
        &device,
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
                registry_tokens([[3, 4, 6, 9], [1, 2, 5, 10]], device),
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
                registry_tokens([[1, 2, 5, 10, 11], [1, 2, 5, 10, 12]], device),
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
                .forward(registry_tokens([[1, 2, 7], [1, 2, 8]], device), &options)
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
            AutoCausalConfig::Llama,
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
    assert_eq!(AutoCausalConfig::from_model_type("gemma2"), None);
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

/// Each parser maps schema failures into its automatic-model context without allocating weights.
#[test]
fn all_registry_parsers_preserve_configuration_error_context() {
    let device = Default::default();
    for family in AutoCausalLm::<TestBackend>::MODEL_TYPES {
        let json =
            serde_json::json!({"model_type":family,"vocab_size":"not an integer"}).to_string();
        assert!(
            matches!(AutoCausalLm::<TestBackend>::from_config_json(&json, &device).unwrap_err(),
            Error::Configuration { field, reason } if field == "auto_causal.config" && !reason.is_empty())
        );
    }
    assert!(
        matches!(AutoSeq2SeqLm::<TestBackend>::from_config_json(r#"{"model_type":"t5","vocab_size":"invalid"}"#, &device).unwrap_err(),
        Error::Configuration { field, reason } if field == "auto_seq2seq.config" && !reason.is_empty())
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
            AutoCausalConfig::Llama,
            <crate::models::llama::LlamaFamilyConfig as ModelConfig>::MODEL_TYPES,
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
