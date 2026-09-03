//! Opt-in tests for large or externally downloaded model artifacts.
//!
//! Ordinary CI never assumes a model directory exists. Maintainers can point
//! `TRANSFORMERS_QWEN35_DIR` at an immutable local checkout and explicitly run
//! ignored tests when certifying a checkpoint revision.

#![cfg(feature = "tokenizers")]

#[cfg(feature = "metal")]
use std::time::Instant;
use std::{fs, path::PathBuf};

#[cfg(feature = "chat-templates")]
use transformers::{ChatMessage, ChatTemplateOptions};
use transformers::{
    EncodeOptions, LocalTokenizer,
    checkpoint::{
        ArtifactSource, CheckpointManifest, LoadOptions,
        quantization::{
            QuantizedBurnpackArtifact, QuantizedCheckpointOptions, SymmetricQuantization,
            WeightQuantizationConfig, quantize_local_checkpoint_to_burnpack,
        },
    },
    models::{
        config::ModelConfig,
        qwen3_5::{Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5TextConfig},
    },
};

#[cfg(feature = "metal")]
use burn::{
    backend::Metal,
    tensor::{DType, Int, Tensor, TensorData},
};
#[cfg(all(feature = "metal", not(feature = "rocm")))]
use burn::{module::Param, nn::Linear};
#[cfg(feature = "metal")]
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLanguageModel, DecodeInput, DecodeState, ForwardOptions,
    LogitsSelection,
    architecture::CausalLmInput,
    cache::CacheConfig,
    checkpoint::{DTypePolicy, TensorDType, execute_local_load_plan},
    models::qwen3_5::Qwen3_5ForCausalLm,
};
#[cfg(all(feature = "metal", not(feature = "rocm")))]
use transformers::{
    delta_net::{DeltaNetMode, GatedDeltaNetState},
    operations,
};

/// Immutable identity recorded alongside the committed Qwen3.8 header inventory.
#[derive(Debug, serde::Deserialize)]
struct PinnedQwen38Inventory {
    checkpoint: String,
    revision: String,
    tensors: Vec<PinnedTensorInventory>,
}

/// One expected physical tensor from the pinned Qwen3.8 SafeTensors headers.
#[derive(Debug, serde::Deserialize)]
struct PinnedTensorInventory {
    name: String,
    shape: Vec<u64>,
    dtype: String,
}

/// Resolves the opt-in Qwen3.8 artifact root without silently choosing mutable data.
fn pinned_qwen38_root() -> PathBuf {
    PathBuf::from(std::env::var_os("TRANSFORMERS_QWEN38_DIR").expect(
        "set TRANSFORMERS_QWEN38_DIR to Qwen/Qwen3.8-27B revision \
             1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ))
}

/// Parses the checked-in tensor-level inventory for the immutable Qwen3.8 revision.
fn pinned_qwen38_inventory() -> PinnedQwen38Inventory {
    serde_json::from_str(include_str!(
        "fixtures/qwen3_5_checkpoint/qwen3_8_27b_inventory.json"
    ))
    .expect("the committed Qwen3.8 tensor inventory must remain valid JSON")
}

/// Validates every pinned Qwen3.8 configuration field and physical tensor header.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_config_and_header_inventory() {
    let root = pinned_qwen38_root();
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let observed_config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("the pinned nested Qwen3.8 text configuration must validate");
    let mut expected_config =
        Qwen3_5TextConfig::from_json_str(include_str!("fixtures/qwen3_8_27b.json"))
            .expect("the committed Qwen3.8 configuration must validate");
    expected_config.layer_types = expected_config.resolved_layer_types();
    assert_eq!(observed_config, expected_config);
    assert_eq!(observed_config.model_type, "qwen3_5_text");
    assert_eq!(observed_config.num_hidden_layers, 64);
    assert_eq!(observed_config.mtp_num_hidden_layers, 1);

    let expected = pinned_qwen38_inventory();
    assert_eq!(expected.checkpoint, "Qwen/Qwen3.8-27B");
    assert_eq!(
        expected.revision,
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("all 18 pinned SafeTensors shard headers must pass strict discovery");
    assert_eq!(manifest.shards.len(), 18);
    assert_eq!(manifest.tensors.len(), expected.tensors.len());
    assert_eq!(manifest.tensors.len(), 1_199);
    match &manifest.source {
        ArtifactSource::Index {
            relative_path,
            declared_total_size,
        } => {
            assert_eq!(relative_path, "model.safetensors.index.json");
            assert_eq!(*declared_total_size, Some(manifest.tensor_bytes));
        }
        ArtifactSource::Single { relative_path } => {
            panic!("expected a sharded checkpoint index, found {relative_path}")
        }
        _ => panic!("expected the standard sharded checkpoint index"),
    }

    for tensor in &expected.tensors {
        let observed = manifest
            .tensor(&tensor.name)
            .unwrap_or_else(|| panic!("pinned header is missing tensor `{}`", tensor.name));
        assert_eq!(
            observed.shape, tensor.shape,
            "shape changed for `{}`",
            tensor.name
        );
        assert_eq!(
            observed.dtype.as_str(),
            tensor.dtype,
            "dtype changed for `{}`",
            tensor.name
        );
    }

    let language_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("model.language_model."))
        .count();
    let visual_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("model.visual."))
        .count();
    let mtp_tensors = manifest
        .tensors
        .keys()
        .filter(|name| name.starts_with("mtp."))
        .count();
    assert_eq!(language_tensors, 850);
    assert_eq!(usize::from(manifest.tensor("lm_head.weight").is_some()), 1);
    assert_eq!(mtp_tensors, 15);
    assert_eq!(visual_tensors, 333);
    assert_eq!(language_tensors + 1 + mtp_tensors + visual_tensors, 1_199);
}

/// Proves the full 866-tensor text-plus-MTP graph consumes every non-visual key.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn strictly_maps_pinned_qwen38_text_and_mtp() {
    let root = pinned_qwen38_root();
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("the pinned nested Qwen3.8 text configuration must validate");
    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("the pinned SafeTensors inventory must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions {
            source: Qwen3_5CheckpointSource::Multimodal,
            include_mtp: true,
        })
        .expect("the pinned Qwen3.8 topology must produce a text-plus-MTP contract");
    let plan = contract
        .build_load_plan(&manifest, &LoadOptions::default())
        .expect("every text and MTP tensor must map while visual tensors are allowlisted");

    assert_eq!(contract.expected_tensors().len(), 866);
    assert_eq!(plan.tensors.len(), 866);
    assert!(plan.aliases.is_empty());
    assert!(plan.report.is_strictly_valid());
    assert_eq!(plan.report.allowlisted_ignored.len(), 333);
    assert!(
        plan.report
            .allowlisted_ignored
            .iter()
            .all(|name| name.starts_with("model.visual."))
    );
    assert_eq!(
        plan.tensors
            .iter()
            .filter(|tensor| tensor.target.starts_with("mtp."))
            .count(),
        15
    );

    let planned_sources = plan
        .tensors
        .iter()
        .flat_map(|tensor| tensor.sources.iter().map(|source| source.name.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    let expected_sources = manifest
        .tensors
        .keys()
        .filter(|name| !name.starts_with("model.visual."))
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(planned_sources, expected_sources);
    eprintln!("Qwen3.8 pre-allocation memory estimate: {:?}", plan.memory);
}

/// Exercises the exact local Qwen3.8 tokenizer vocabulary without network access.
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_tokenizer() {
    let tokenizer = LocalTokenizer::from_directory(pinned_qwen38_root())
        .expect("the pinned Qwen3.8 tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let encoding = tokenizer
        .encode(
            prompt,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the pinned tokenizer must encode ordinary text");
    let token_ids = &encoding.token_ids()[0];
    assert!(!token_ids.is_empty());
    assert_eq!(encoding.valid_tokens()[0], vec![true; token_ids.len()]);
    assert_eq!(
        tokenizer
            .decode(token_ids, true)
            .expect("the pinned tokenizer must decode its own ordinary-text IDs"),
        prompt
    );
}

/// Validates the pinned Qwen3.8 no-thinking assistant-prefix chat rendering.
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"]
fn validates_pinned_qwen38_chat_template() {
    let tokenizer = LocalTokenizer::from_directory(pinned_qwen38_root())
        .expect("the pinned Qwen3.8 tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let rendered = tokenizer
        .render_chat(
            &[ChatMessage::text("user", prompt)],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..ChatTemplateOptions::default()
            },
        )
        .expect("the pinned Qwen3.8 chat template must render a text-only request");
    assert_eq!(
        rendered,
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
}

/// Loads Qwen3.8 in Metal F16 through the public registry and checks cached parity.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires pinned Qwen3.8 weights, about 60 GiB of free unified memory, and Metal"]
fn loads_pinned_qwen38_text_and_mtp_and_runs_metal() {
    let root = pinned_qwen38_root();
    let device = Default::default();
    let load_started = Instant::now();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: true,
            qwen_mtp: true,
            repository: Some("Qwen/Qwen3.8-27B".into()),
            revision: Some("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("the public auto loader must stream the complete Qwen3.8 text-plus-MTP graph");
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert!(loaded.load_report().is_strictly_valid());
    assert_eq!(loaded.load_report().applied.len(), 866);
    assert_eq!(loaded.load_report().allowlisted_ignored.len(), 333);
    assert_eq!(loaded.source().repository(), Some("Qwen/Qwen3.8-27B"));
    assert_eq!(
        loaded.source().revision(),
        Some("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    );

    let encoding = loaded
        .tokenizer()
        .expect("the public loaded-model bundle must retain the local tokenizer")
        .encode(
            "Hello",
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("the smoke prompt must tokenize");
    let sequence = encoding.token_ids()[0].len();
    let input = encoding
        .to_causal_input::<Metal>(&device)
        .expect("the tokenizer batch must convert to a public causal input");
    let stateless_started = Instant::now();
    let stateless = loaded
        .model()
        .forward(input.clone(), &ForwardOptions::default())
        .expect("the public auto model must execute stateless Qwen3.8 inference");
    assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
    let stateless_logits = metal_tensor_values("Qwen3.8 stateless logits", &stateless.logits);
    let stateless_seconds = stateless_started.elapsed().as_secs_f64();
    assert!(stateless_logits.iter().all(|value| value.is_finite()));

    let mut state = loaded
        .model()
        .new_decode_state(
            CacheConfig::new(1, sequence + 1).expect("the smoke cache geometry must validate"),
            1,
            &device,
        )
        .expect("the public auto model must allocate Qwen3.8 hybrid decode state");
    let prefill_started = Instant::now();
    let cached = loaded
        .model()
        .prefill(input, &mut state, &ForwardOptions::default())
        .expect("the public auto model must execute cached Qwen3.8 prefill");
    let cached_logits = metal_tensor_values("Qwen3.8 cached logits", &cached.logits);
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    assert!(cached_logits.iter().all(|value| value.is_finite()));
    let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
    assert!(
        cached_cosine >= 0.999,
        "Qwen3.8 cached/stateless cosine {cached_cosine} is below the acceptance gate"
    );
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Metal top-1 token IDs must use i32 storage");
    let cached_top = cached
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("cached Metal top-1 token IDs must use i32 storage");
    assert_eq!(cached_top, stateless_top);

    let next_token = i64::from(stateless_top[0]);
    let decode_started = Instant::now();
    let decoded = loaded
        .model()
        .decode(
            DecodeInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![next_token], [1, 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("the public auto model must execute one recurrent decode step");
    assert_eq!(decoded.logits.dims(), [1, 1, 248_320]);
    let decoded_logits = metal_tensor_values("Qwen3.8 decoded logits", &decoded.logits);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    assert!(decoded_logits.iter().all(|value| value.is_finite()));
    assert_eq!(state.lengths(), &[sequence + 1]);
    eprintln!(
        "Qwen3.8 F16: prompt_tokens={sequence}, first_top1={next_token}, \
         cached_cosine={cached_cosine:.9}, load_seconds={load_seconds:.3}, \
         stateless_seconds={stateless_seconds:.3}, prefill_seconds={prefill_seconds:.3}, \
         decode_seconds={decode_seconds:.3}"
    );
}

/// Parses the official nested multimodal config and exercises text-only tokenization.
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory"]
fn validates_pinned_qwen35_local_artifacts() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    assert_eq!(config.model_type, "qwen3_5_text");
    assert_eq!(config.vocab_size, 248_320);

    let manifest = CheckpointManifest::discover(&root, &LoadOptions::default())
        .expect("official SafeTensors index and header must pass strict discovery");
    assert_eq!(manifest.tensors.len(), 488);
    assert_eq!(manifest.tensor_bytes, 1_746_882_752);
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.language_model."))
            .count(),
        320
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("model.visual."))
            .count(),
        153
    );
    assert_eq!(
        manifest
            .tensors
            .keys()
            .filter(|name| name.starts_with("mtp."))
            .count(),
        15
    );

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let prompt = "Explain why cached decoding is faster in one sentence.";
    let encoding = tokenizer
        .encode(
            prompt,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("official tokenizer must encode text");
    assert!(!encoding.token_ids()[0].is_empty());

    #[cfg(feature = "chat-templates")]
    {
        let rendered = tokenizer
            .render_chat(
                &[ChatMessage::text("user", prompt)],
                &ChatTemplateOptions {
                    add_generation_prompt: true,
                    ..ChatTemplateOptions::default()
                },
            )
            .expect("official text-only chat template must render");
        assert!(rendered.contains("<|im_start|>user"));
        assert!(rendered.ends_with("<think>\n\n</think>\n\n"));
    }
}

/// Converts every eligible mapped Qwen tensor to Q8 and Q4 official Burnpacks.
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory and temporary artifact space"]
fn quantizes_pinned_qwen35_weights_to_burnpack() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    let load_options = LoadOptions::default();
    let manifest = CheckpointManifest::discover(&root, &load_options)
        .expect("official SafeTensors headers must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .expect("official topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &load_options)
        .expect("every language tensor must map strictly");
    let requested_output_root =
        std::env::var_os("TRANSFORMERS_QUANTIZED_OUTPUT_DIR").map(PathBuf::from);
    let temporary_outputs = requested_output_root.is_none().then(|| {
        tempfile::tempdir_in(
            root.parent()
                .expect("pinned checkpoint must have a parent directory"),
        )
        .expect("create temporary quantized artifacts beside model storage")
    });
    let output_root = match &requested_output_root {
        Some(path) => {
            fs::create_dir_all(path).unwrap_or_else(|error| {
                panic!(
                    "create persistent quantized output {}: {error}",
                    path.display()
                )
            });
            path.clone()
        }
        None => temporary_outputs
            .as_ref()
            .expect("temporary output exists without an explicit path")
            .path()
            .to_path_buf(),
    };

    let schemes = match std::env::var("TRANSFORMERS_QUANT_SCHEME").as_deref() {
        Ok("q8") => vec![SymmetricQuantization::Q8],
        Ok("q4") => vec![SymmetricQuantization::Q4],
        Ok(other) => panic!("unsupported TRANSFORMERS_QUANT_SCHEME `{other}`; use q8 or q4"),
        Err(std::env::VarError::NotPresent) => {
            vec![SymmetricQuantization::Q8, SymmetricQuantization::Q4]
        }
        Err(error) => panic!("invalid TRANSFORMERS_QUANT_SCHEME: {error}"),
    };

    let mut summaries = Vec::new();
    for scheme in schemes {
        let label = match scheme {
            SymmetricQuantization::Q8 => "q8",
            SymmetricQuantization::Q4 => "q4",
        };
        let output = output_root.join(format!("qwen3.5-0.8b-{label}.bpk"));
        let outcome = quantize_local_checkpoint_to_burnpack(
            &manifest,
            &plan,
            &output,
            QuantizedCheckpointOptions {
                weights: WeightQuantizationConfig {
                    scheme,
                    ..WeightQuantizationConfig::default()
                },
                ..QuantizedCheckpointOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{label} checkpoint conversion failed: {error}"));
        assert!(outcome.report.is_strictly_valid());
        assert!(!outcome.report.quantized.is_empty());
        assert_eq!(
            outcome.report.quantized.len(),
            outcome.artifact.tensor_count
        );
        assert_eq!(outcome.execution.source.tensor_reads, 320);
        assert_eq!(outcome.execution.source.max_open_shards, 1);
        assert_eq!(outcome.staging.tensor_count, outcome.artifact.tensor_count);
        assert!(
            outcome
                .report
                .quantized
                .windows(2)
                .all(|pair| pair[0].target < pair[1].target),
            "quantization report must be deterministic and deduplicated"
        );

        let mut artifact = QuantizedBurnpackArtifact::open(&output)
            .unwrap_or_else(|error| panic!("open {label} Burnpack: {error}"));
        let validation = artifact
            .validate_all()
            .unwrap_or_else(|error| panic!("validate {label} Burnpack: {error}"));
        assert_eq!(validation.tensor_count, outcome.artifact.tensor_count);
        assert_eq!(
            validation.payload_bytes,
            outcome.artifact.quantized_payload_bytes
        );
        assert_eq!(artifact.manifest().tensors.len(), validation.tensor_count);
        assert!(
            artifact.manifest().packed_execution.is_available(),
            "{label} writer must certify the native Burn QFloat layout"
        );

        #[cfg(all(feature = "metal", not(feature = "rocm")))]
        {
            let device = Default::default();
            let loaded = AutoCausalLm::<Metal>::load_local(
                &root,
                &AutoLoadOptions {
                    checkpoint: LoadOptions {
                        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                        ..LoadOptions::default()
                    },
                    load_tokenizer: false,
                    quantized_burnpack: Some(output.clone()),
                    repository: Some("Qwen/Qwen3.5-0.8B".into()),
                    revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
                    ..AutoLoadOptions::default()
                },
                &device,
            )
            .unwrap_or_else(|error| panic!("load mixed-precision {label} model: {error}"));
            assert_eq!(
                loaded.load_report().quantized.len(),
                outcome.artifact.tensor_count
            );
            let AutoCausalLm::Qwen3_5(model) = loaded.model() else {
                panic!("Qwen3.5 config must select the hybrid decoder variant");
            };
            let first_packed_dtype = model.model.layers[0]
                .linear_attn
                .as_ref()
                .expect("official first layer is DeltaNet")
                .in_proj_qkv
                .weight
                .val()
                .dtype();
            assert!(
                matches!(first_packed_dtype, DType::QFloat(_)),
                "{label} parameter must remain QFloat after installation"
            );

            let input = CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(vec![9_419_i64], [1, 1]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            };
            let stateless = loaded
                .model()
                .forward(
                    input.clone(),
                    &ForwardOptions {
                        hidden_states: true,
                        ..ForwardOptions::default()
                    },
                )
                .unwrap_or_else(|error| panic!("run stateless {label} inference: {error}"));
            assert_eq!(stateless.logits.dims(), [1, 1, 248_320]);
            for (index, hidden) in stateless.hidden_states.iter().enumerate() {
                let values = hidden
                    .clone()
                    .cast(DType::F32)
                    .to_data()
                    .to_vec::<f32>()
                    .expect("quantized hidden state must materialize as f32");
                assert!(
                    values.iter().all(|value| value.is_finite()),
                    "{label} hidden-state checkpoint {index} is the first observed non-finite boundary"
                );
            }
            let stateless_logits = stateless
                .logits
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .expect("quantized Metal logits must materialize as f32");
            assert!(stateless_logits.iter().all(|value| value.is_finite()));
            let stateless_top = stateless
                .logits
                .argmax(2)
                .to_data()
                .to_vec::<i32>()
                .expect("quantized top-1 IDs must use i32 storage");

            let mut state = loaded
                .model()
                .new_decode_state(
                    CacheConfig::new(1, 2).expect("quantized smoke cache"),
                    1,
                    &device,
                )
                .expect("quantized model cache must initialize");
            let cached = loaded
                .model()
                .prefill(input, &mut state, &ForwardOptions::default())
                .unwrap_or_else(|error| panic!("run cached {label} prefill: {error}"));
            let cached_logits = cached
                .logits
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .expect("cached quantized Metal logits must materialize as f32");
            let cached_top = cached
                .logits
                .argmax(2)
                .to_data()
                .to_vec::<i32>()
                .expect("cached quantized top-1 IDs must use i32 storage");
            assert_eq!(cached_top, stateless_top);
            let cached_cosine = cosine_similarity(&stateless_logits, &cached_logits);
            assert!(
                cached_cosine >= 0.999,
                "{label} cached/stateless cosine {cached_cosine} is below the gate"
            );
            eprintln!(
                "{label}: packed_dtype={first_packed_dtype:?}, top1={}, cached_cosine={cached_cosine:.9}",
                stateless_top[0]
            );
        }
        eprintln!(
            "{label}: tensors={}, excluded={}, payload_bytes={}, artifact_bytes={}, largest_tensor_bytes={}",
            outcome.artifact.tensor_count,
            outcome.excluded.len(),
            outcome.artifact.quantized_payload_bytes,
            outcome.artifact.artifact_bytes,
            outcome.artifact.largest_materialized_tensor_bytes,
        );
        summaries.push((
            outcome.artifact.quantized_payload_bytes,
            outcome.artifact.artifact_bytes,
            outcome.report.quantized,
            outcome.excluded,
        ));
    }

    if let [q8, q4] = summaries.as_slice() {
        assert!(q4.0 < q8.0);
        assert!(q4.1 < q8.1);
        assert_eq!(
            q4.2.iter()
                .map(|record| record.target.as_str())
                .collect::<Vec<_>>(),
            q8.2.iter()
                .map(|record| record.target.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(q4.3, q8.3);
    } else {
        assert_eq!(summaries.len(), 1, "a filtered run must execute one scheme");
    }
}

/// Replays the first real Qwen DeltaNet layer one packed projection at a time.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3.5 weights, a retained Burnpack, and Metal"]
fn diagnoses_pinned_qwen35_packed_delta_layer_on_metal() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let burnpack = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_BURNPACK")
            .expect("set TRANSFORMERS_QWEN35_BURNPACK to a retained Q4/Q8 sidecar"),
    );
    let mut artifact =
        QuantizedBurnpackArtifact::open(&burnpack).expect("retained diagnostic Burnpack must open");
    let device = Default::default();
    let mut loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack.clone()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("retained packed model must load");
    let AutoCausalLm::Qwen3_5(model) = loaded.model_mut() else {
        panic!("Qwen config must select the hybrid decoder");
    };
    let token_ids =
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(vec![9_419_i64], [1, 1]), &device);
    let embedding = model.model.embed_tokens.forward(token_ids);
    assert_metal_tensor_finite("embedding", &embedding);
    let layer = &mut model.model.layers[0];
    let normalized = layer.input_layernorm.forward(embedding.clone());
    assert_metal_tensor_finite("input norm", &normalized);
    let delta = layer
        .linear_attn
        .as_mut()
        .expect("official first layer is DeltaNet");
    for (name, target, projection) in [
        (
            "in_proj_qkv",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            &delta.in_proj_qkv,
        ),
        (
            "in_proj_z",
            "model.layers.0.linear_attn.in_proj_z.weight",
            &delta.in_proj_z,
        ),
        (
            "in_proj_b",
            "model.layers.0.linear_attn.in_proj_b.weight",
            &delta.in_proj_b,
        ),
        (
            "in_proj_a",
            "model.layers.0.linear_attn.in_proj_a.weight",
            &delta.in_proj_a,
        ),
    ] {
        let output = operations::linear(normalized.clone(), projection)
            .unwrap_or_else(|error| panic!("{name} execution failed: {error}"));
        assert_metal_tensor_finite(name, &output);
        let reference_weight = Tensor::<Metal, 2>::from_data(
            artifact
                .read_weight(target)
                .unwrap_or_else(|error| panic!("read {target}: {error}"))
                .dequantize_f32()
                .unwrap_or_else(|error| panic!("dequantize {target}: {error}")),
            &device,
        )
        .transpose()
        .cast(normalized.dtype());
        let reference_projection = Linear {
            weight: Param::from_tensor(reference_weight),
            bias: None,
        };
        let reference = operations::linear(normalized.clone(), &reference_projection)
            .unwrap_or_else(|error| panic!("{name} dequantized execution failed: {error}"));
        let actual_values = metal_tensor_values(name, &output);
        let reference_values = metal_tensor_values("dequantized projection", &reference);
        let cosine = cosine_similarity(&actual_values, &reference_values);
        let maximum = actual_values
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        eprintln!("{name}: packed/dequantized cosine={cosine:.9}, max_abs={maximum}");
        assert!(
            cosine >= 0.999,
            "{name} packed/dequantized cosine {cosine} is below the dispatch gate"
        );
    }
    let out_probe = operations::linear(
        Tensor::<Metal, 3>::ones([1, 1, delta.value_heads * delta.value_head_dim], &device)
            .cast(DType::F16),
        &delta.out_proj,
    )
    .expect("standalone packed output projection must execute");
    assert_metal_tensor_finite("standalone out_proj", &out_probe);
    let packed_out = delta.out_proj.weight.val();
    assert!(matches!(packed_out.dtype(), DType::QFloat(_)));
    let delta_config = delta.config();
    let mut recurrent_state =
        GatedDeltaNetState::zeros(&delta_config, 1, &device).expect("recurrent diagnostic state");
    let recurrent_mixed = delta
        .forward_with_state(
            normalized.clone(),
            None,
            &mut recurrent_state,
            DeltaNetMode::Recurrent,
        )
        .expect("packed recurrent DeltaNet must execute");
    assert_metal_tensor_finite("recurrent DeltaNet", &recurrent_mixed);
    let mut chunked_state =
        GatedDeltaNetState::zeros(&delta_config, 1, &device).expect("chunked diagnostic state");
    let mixed = delta
        .forward_with_state(
            normalized,
            None,
            &mut chunked_state,
            DeltaNetMode::ChunkedPrefill,
        )
        .expect("packed chunked DeltaNet must execute");
    assert_metal_tensor_finite("chunked DeltaNet", &mixed);
    let residual = embedding + mixed;
    assert_metal_tensor_finite("post-DeltaNet residual", &residual);
    let mlp = layer
        .mlp
        .forward(layer.post_attention_layernorm.forward(residual.clone()))
        .expect("packed layer MLP must execute");
    assert_metal_tensor_finite("layer MLP", &mlp);
    assert_metal_tensor_finite("complete layer", &(residual + mlp));
}

/// Loads a retained Q4 sidecar and verifies complete stateless/cached Metal inference.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
#[test]
#[ignore = "requires pinned Qwen3.5 weights, a retained Q4 Burnpack, and Metal"]
fn loads_pinned_qwen35_q4_burnpack_and_runs_metal() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let burnpack = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_BURNPACK")
            .expect("set TRANSFORMERS_QWEN35_BURNPACK to the retained Q4 sidecar"),
    );
    let artifact =
        QuantizedBurnpackArtifact::open(&burnpack).expect("retained Q4 Burnpack must open");
    let quantized_count = artifact.manifest().tensors.len();
    assert!(
        artifact
            .manifest()
            .tensors
            .values()
            .all(|tensor| tensor.scheme == SymmetricQuantization::Q4),
        "the Q4 acceptance path must not silently consume a different precision"
    );
    drop(artifact);

    let device = Default::default();
    let loaded = AutoCausalLm::<Metal>::load_local(
        &root,
        &AutoLoadOptions {
            checkpoint: LoadOptions {
                dtype_policy: DTypePolicy::Cast(TensorDType::F16),
                ..LoadOptions::default()
            },
            load_tokenizer: false,
            quantized_burnpack: Some(burnpack),
            repository: Some("Qwen/Qwen3.5-0.8B".into()),
            revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
            ..AutoLoadOptions::default()
        },
        &device,
    )
    .expect("mixed F16/Q4 model must load transactionally");
    assert_eq!(loaded.load_report().quantized.len(), quantized_count);

    let input = CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::from_data(
            TensorData::new(vec![9_419_i64], [1, 1]),
            &device,
        ),
        token_mask: None,
        position_ids: None,
    };
    let stateless = loaded
        .model()
        .forward(input.clone(), &ForwardOptions::default())
        .expect("Q4 stateless inference must execute");
    let stateless_values = stateless
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("Q4 logits must materialize as f32");
    assert!(stateless_values.iter().all(|value| value.is_finite()));
    let stateless_top = stateless
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("Q4 top-1 IDs must use i32 storage");

    let mut state = loaded
        .model()
        .new_decode_state(CacheConfig::new(1, 2).expect("Q4 smoke cache"), 1, &device)
        .expect("Q4 decode state must initialize");
    let cached = loaded
        .model()
        .prefill(input, &mut state, &ForwardOptions::default())
        .expect("Q4 cached prefill must execute");
    let cached_values = cached
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("cached Q4 logits must materialize as f32");
    let cached_top = cached
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("cached Q4 top-1 IDs must use i32 storage");
    let cached_cosine = cosine_similarity(&stateless_values, &cached_values);
    assert_eq!(cached_top, stateless_top);
    assert!(cached_cosine >= 0.999);
    eprintln!(
        "q4: tensors={quantized_count}, top1={}, cached_cosine={cached_cosine:.9}",
        stateless_top[0]
    );
}

/// Synchronizes one Metal tensor and rejects the first non-finite component result.
#[cfg(all(feature = "metal", not(feature = "rocm")))]
fn assert_metal_tensor_finite<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) {
    let values = metal_tensor_values(label, tensor);
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} produced non-finite values"
    );
}

/// Returns synchronized F32 values for quantitative Metal diagnostics.
#[cfg(feature = "metal")]
fn metal_tensor_values<const D: usize>(label: &str, tensor: &Tensor<Metal, D>) -> Vec<f32> {
    tensor
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .unwrap_or_else(|error| panic!("materialize {label}: {error}"))
}

/// Streams the pinned language weights into Metal F16 and executes one prompt.
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a pinned external Qwen3.5 artifact directory and Metal GPU"]
fn loads_pinned_qwen35_weights_and_runs_metal_inference() {
    let root = PathBuf::from(
        std::env::var_os("TRANSFORMERS_QWEN35_DIR")
            .expect("set TRANSFORMERS_QWEN35_DIR to the pinned local checkout"),
    );
    let config_path = root.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config_path.display()));
    let config = Qwen3_5TextConfig::from_json_str(&config_json)
        .expect("official nested text configuration must validate");
    let options = LoadOptions {
        dtype_policy: DTypePolicy::Cast(TensorDType::F16),
        ..LoadOptions::default()
    };
    let manifest = CheckpointManifest::discover(&root, &options)
        .expect("official SafeTensors headers must pass strict discovery");
    let contract = config
        .checkpoint_contract(Qwen3_5CheckpointOptions::default())
        .expect("official topology must produce a checkpoint contract");
    let plan = contract
        .build_load_plan(&manifest, &options)
        .expect("every language tensor must map strictly");
    assert_eq!(plan.tensors.len(), 320);
    assert_eq!(plan.aliases.len(), 1);
    assert_eq!(plan.report.allowlisted_ignored.len(), 168);

    let device = Default::default();
    let mut model = Qwen3_5ForCausalLm::<Metal>::init(config, &device)
        .expect("official Qwen graph must initialize lazily");
    let outcome = execute_local_load_plan(&manifest, &plan, &mut model)
        .unwrap_or_else(|error| panic!("streamed model load failed: {error}"));
    assert!(outcome.report.is_strictly_valid());
    assert_eq!(outcome.report.applied.len(), 321);
    assert_eq!(outcome.stats.tensors_applied, 320);
    assert_eq!(outcome.stats.aliases_applied, 1);

    let tokenizer = LocalTokenizer::from_directory(&root)
        .expect("official tokenizer artifacts must load locally");
    let encoding = tokenizer
        .encode(
            "Hello",
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("prompt must tokenize");
    let tokens = encoding.token_ids()[0]
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let sequence = tokens.len();
    let input_ids =
        Tensor::<Metal, 2, Int>::from_data(TensorData::new(tokens.clone(), [1, sequence]), &device);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect("loaded model must execute inference");
    assert_eq!(output.logits.dims(), [1, 1, 248_320]);
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/qwen3_5_checkpoint/qwen3_5_0_8b_reference.json"
    ))
    .expect("committed reference logits must be valid JSON");
    let expected_top_ids = reference["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let (_, top_ids) = output.logits.clone().topk_with_indices(10, 2);
    let top_ids = top_ids
        .to_data()
        .to_vec::<i32>()
        .expect("Metal top-k must return i32 token IDs");
    assert_eq!(top_ids, expected_top_ids);

    let sample_indices = reference["sample_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect::<Vec<_>>();
    let sample_count = sample_indices.len();
    let sample_indices = Tensor::<Metal, 1, Int>::from_data(
        TensorData::new(sample_indices, [sample_count]),
        &device,
    );
    let sample_logits = output
        .logits
        .clone()
        .select(2, sample_indices)
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("selected Metal logits must convert to f32");
    let expected_sample_logits = reference["sample_logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    for (actual, expected) in sample_logits.iter().zip(expected_sample_logits) {
        assert!(
            (actual - expected).abs() <= 0.25,
            "reference logit mismatch: expected {expected}, found {actual}"
        );
    }

    if let Some(reference_path) = std::env::var_os("TRANSFORMERS_QWEN35_REFERENCE_LOGITS") {
        let reference_path = PathBuf::from(reference_path);
        let bytes = fs::read(&reference_path).unwrap_or_else(|error| {
            panic!(
                "failed to read raw little-endian f32 reference logits from {}: {error}",
                reference_path.display()
            )
        });
        let expected_bytes = 248_320_usize
            .checked_mul(size_of::<f32>())
            .expect("the fixed reference vocabulary byte length must fit usize");
        assert_eq!(
            bytes.len(),
            expected_bytes,
            "reference logit file must contain exactly one little-endian f32 vocabulary vector"
        );
        let reference_logits = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let actual_logits = output
            .logits
            .clone()
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .expect("Metal logits must convert to an f32 host vector");
        assert_eq!(actual_logits.len(), reference_logits.len());
        let cosine = cosine_similarity(&actual_logits, &reference_logits);
        eprintln!("full-vocabulary reference-logit cosine similarity: {cosine:.9}");
        assert!(
            cosine >= 0.999,
            "full-vocabulary cosine similarity {cosine} is below the 0.999 acceptance gate"
        );
    }

    let top_token = top_ids
        .first()
        .copied()
        .expect("top-k returns one result for a non-empty vocabulary");
    assert!((0..248_320).contains(&(top_token as usize)));

    let expected_greedy = reference["greedy_token_ids"]
        .as_array()
        .expect("reference must contain a greedy continuation")
        .iter()
        .map(|value| value.as_i64().expect("greedy token IDs must be integers") as i32)
        .collect::<Vec<_>>();
    assert_eq!(
        expected_greedy.len(),
        reference["greedy_max_new_tokens"]
            .as_u64()
            .expect("reference generation length must be an integer") as usize
    );
    let cache_config = CacheConfig::new(1, sequence + expected_greedy.len())
        .expect("reference generation cache geometry must be valid");
    let mut state = model
        .new_decode_state(cache_config, 1, &device)
        .expect("official Qwen cache geometry must initialize");
    let mut cached = model
        .prefill(
            CausalLmInput {
                token_ids: input_ids.clone(),
                token_mask: None,
                position_ids: None,
            },
            &mut state,
            &ForwardOptions::default(),
        )
        .expect("cached prompt prefill must execute");
    let prompt_stateless = output
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("stateless prompt logits must convert to f32");
    let prompt_cached = cached
        .logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .expect("cached prompt logits must convert to f32");
    let prompt_cosine = cosine_similarity(&prompt_stateless, &prompt_cached);
    assert!(
        prompt_cosine >= 0.999,
        "cached prefill cosine similarity {prompt_cosine} is below the 0.999 gate"
    );

    let mut generated = Vec::with_capacity(expected_greedy.len());
    for (step, expected) in expected_greedy.iter().copied().enumerate() {
        let selected = cached
            .logits
            .clone()
            .argmax(2)
            .to_data()
            .to_vec::<i32>()
            .expect("cached greedy IDs must use i32 storage");
        let token = *selected
            .first()
            .expect("one cached logit row must select one token");
        assert_eq!(
            token, expected,
            "cached continuation differs from the official reference at step {step}"
        );
        generated.push(token);
        if step + 1 < expected_greedy.len() {
            cached = model
                .decode(
                    DecodeInput {
                        token_ids: Tensor::<Metal, 2, Int>::from_data(
                            TensorData::new(vec![i64::from(token)], [1, 1]),
                            &device,
                        ),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &ForwardOptions::default(),
                )
                .unwrap_or_else(|error| panic!("cached decode step {step} failed: {error}"));
        }
    }
    assert_eq!(generated, expected_greedy);
    assert_eq!(state.lengths(), &[sequence + expected_greedy.len() - 1]);

    let mut teacher_tokens = tokens;
    teacher_tokens.extend(
        expected_greedy
            .iter()
            .take(expected_greedy.len() - 1)
            .map(|token| i64::from(*token)),
    );
    let teacher_sequence = teacher_tokens.len();
    let teacher_output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<Metal, 2, Int>::from_data(
                    TensorData::new(teacher_tokens, [1, teacher_sequence]),
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                ..ForwardOptions::default()
            },
        )
        .expect("teacher-forced stateless forward must execute");
    let stateless_greedy = teacher_output
        .logits
        .argmax(2)
        .to_data()
        .to_vec::<i32>()
        .expect("stateless greedy IDs must use i32 storage");
    assert_eq!(
        stateless_greedy, expected_greedy,
        "cached and stateless top-1 continuations must match at every step"
    );
}

/// Computes cosine similarity with f64 accumulation for acceptance diagnostics.
#[cfg(feature = "metal")]
fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    assert_eq!(left.len(), right.len());
    let (dot, left_square, right_square) =
        left.iter()
            .zip(right)
            .fold((0.0_f64, 0.0_f64, 0.0_f64), |acc, (left, right)| {
                let left = f64::from(*left);
                let right = f64::from(*right);
                (
                    acc.0 + left * right,
                    acc.1 + left * left,
                    acc.2 + right * right,
                )
            });
    dot / (left_square.sqrt() * right_square.sqrt())
}
