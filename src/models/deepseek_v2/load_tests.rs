//! DeepSeek-V2 streamed parameter installation and malformed-checkpoint rejection.

use std::{fs::File, io::Write, path::Path};

use burn::{
    backend::Flex,
    tensor::{Int, Tensor, TensorData},
};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use super::*;
use crate::{
    CausalLmInput, ForwardOptions,
    checkpoint::{CheckpointError, TensorOperation, TensorSink},
    models::deepseek_v2::model::tests_support::tiny_config,
};

/// CPU backend with F32 tensor values and I32 token indices for deterministic fixtures.
type TestBackend = Flex<f32, i32>;

/// Writes a complete zero-valued F32 SafeTensors artifact.
fn write_checkpoint(
    root: &Path,
    contract: &super::super::DeepseekV2CheckpointContract,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u64>)>),
) {
    let mut tensors = contract
        .mappings()
        .operations()
        .iter()
        .filter_map(|operation| match operation {
            TensorOperation::Map { source, target, .. } => {
                let shape = contract
                    .expected_tensors()
                    .iter()
                    .find(|expected| &expected.name == target)
                    .unwrap()
                    .shape
                    .clone();
                Some((source.clone(), shape))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    mutate(&mut tensors);
    let mut offset = 0_u64;
    let mut header = Map::new();
    for (name, shape) in tensors {
        let elements = shape.iter().copied().product::<u64>();
        let end = offset + elements * 4;
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&vec![0; offset as usize]).unwrap();
}

/// Invalid target syntax is rejected before model indexing.
#[test]
fn target_parser_rejects_unknown_and_malformed_indices() {
    assert!(matches!(
        parse_layer_target("not.a.layer"),
        Err(DeepseekV2LoadError::UnknownTarget { .. })
    ));
    assert!(matches!(
        parse_layer_target("model.layers.x.self_attn.q_proj.weight"),
        Err(DeepseekV2LoadError::InvalidIndex { .. })
    ));
}

/// A strict target cannot install a tensor with the wrong checkpoint shape.
#[test]
fn sink_rejects_wrong_parameter_shape() {
    let device = Default::default();
    let mut config = super::super::model::tests_support::tiny_config();
    config.tie_word_embeddings = false;
    let mut model = DeepseekV2ForCausalLm::<Flex>::init(config, &device).unwrap();
    let error = TensorSink::write_tensor(
        &mut model,
        "model.embed_tokens.weight",
        TensorData::zeros::<f32, _>([1, 1]),
    )
    .unwrap_err();
    assert!(matches!(error, DeepseekV2LoadError::Parameter { .. }));
}

/// Every dense, MLA, routed, and shared target installs and executes.
#[test]
fn complete_contract_installs_and_runs() {
    let config = tiny_config();
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();
    let device = Default::default();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert!(
        output
            .logits
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0)
    );
}

/// Streamed local loading opens one shard and applies every owner once.
#[test]
fn local_loader_streams_tied_and_untied_checkpoints() {
    let device = Default::default();
    for tied in [false, true] {
        let mut config = tiny_config();
        config.tie_word_embeddings = tied;
        let contract = config
            .checkpoint_contract(DeepseekV2CheckpointOptions::default())
            .unwrap();
        let temp = TempDir::new().unwrap();
        write_checkpoint(temp.path(), &contract, |_| {});
        let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
        let outcome = model
            .load_checkpoint(temp.path(), &LoadOptions::default())
            .unwrap();
        assert_eq!(
            outcome.report.applied.len(),
            contract.expected_tensors().len()
        );
        assert_eq!(outcome.stats.source.shard_opens, 1);
        assert_eq!(outcome.stats.source.max_open_shards, 1);
        assert_eq!(outcome.stats.aliases_applied, usize::from(tied));
    }
}

/// Missing, unexpected, and wrong-shaped inventory fails before first write.
#[test]
fn strict_planning_rejects_inventory_mismatches_before_loading() {
    let device = Default::default();
    let config = tiny_config();
    let contract = config
        .checkpoint_contract(DeepseekV2CheckpointOptions::default())
        .unwrap();

    let missing = TempDir::new().unwrap();
    write_checkpoint(missing.path(), &contract, |tensors| {
        tensors.retain(|(name, _)| name != "model.layers.1.mlp.gate.weight");
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(missing.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::MissingMappingSource { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let unexpected = TempDir::new().unwrap();
    write_checkpoint(unexpected.path(), &contract, |tensors| {
        tensors.push(("unexpected.weight".into(), vec![1]));
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    let error = model
        .load_checkpoint(unexpected.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::StrictValidation {
            unexpected: 1,
            ..
        })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());

    let wrong = TempDir::new().unwrap();
    write_checkpoint(wrong.path(), &contract, |tensors| {
        tensors
            .iter_mut()
            .find(|(name, _)| name == "model.embed_tokens.weight")
            .unwrap()
            .1 = vec![31, 8];
    });
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(wrong.path(), &LoadOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(CheckpointError::ShapeMismatch { .. })
    ));
    assert!(!model.model.embed_tokens.weight.is_initialized());
}

/// Query-LoRA targets and expert bounds dispatch to typed optional fields.
#[test]
fn query_lora_and_invalid_expert_targets_are_typed() {
    let device = Default::default();
    let mut config = tiny_config();
    config.q_lora_rank = Some(4);
    let contract = config.checkpoint_contract(Default::default()).unwrap();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    assert!(matches!(
        model.write_tensor(
            "model.layers.1.sparse_mlp.routed.routed.9.gate_proj.weight",
            TensorData::zeros::<f32, _>([6, 8]),
        ),
        Err(DeepseekV2LoadError::InvalidIndex { .. })
    ));
    assert!(matches!(
        model.write_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            TensorData::zeros::<f32, _>([10, 8]),
        ),
        Err(DeepseekV2LoadError::MissingComponent { .. })
    ));
}

/// One `[1]`-shaped F32 payload; every rejection below fails before its shape matters.
fn scalar() -> TensorData {
    TensorData::from([0.0_f32])
}

/// Returns the internal target named by a sink error.
fn error_target(error: &DeepseekV2LoadError) -> String {
    match error {
        DeepseekV2LoadError::UnknownTarget { target }
        | DeepseekV2LoadError::InvalidIndex { target, .. }
        | DeepseekV2LoadError::MissingComponent { target, .. }
        | DeepseekV2LoadError::Parameter { target, .. }
        | DeepseekV2LoadError::InvalidAlias { target, .. } => target.clone(),
    }
}

/// Malformed names, out-of-range indices, and absent components each produce one typed error.
///
/// The tiny model has a dense block 0 and a greedy-routed sparse block 1,
/// a direct query projection, and no attention biases, so each target below
/// names a component the configured topology does not own.
#[test]
fn sink_reports_every_target_rejection_with_its_reason() {
    let device = Default::default();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();
    let invalid = |target: &str, reason: &str| DeepseekV2LoadError::InvalidIndex {
        target: target.into(),
        reason: reason.into(),
    };
    let missing = |target: &str, component: &'static str| DeepseekV2LoadError::MissingComponent {
        target: target.into(),
        component,
    };
    let unknown = |target: &str| DeepseekV2LoadError::UnknownTarget {
        target: target.into(),
    };
    let sparse = "sparse routed/shared expert block";
    let cases = [
        unknown("model.bogus.weight"),
        unknown("model.layers.0.bogus.weight"),
        unknown("model.layers.1.sparse_mlp.shared_experts.bogus.weight"),
        unknown("model.layers.1.sparse_mlp.routed.routed.0.bogus.weight"),
        invalid(
            "model.layers.0",
            "expected `model.layers.<index>.<parameter>`",
        ),
        invalid("model.layers.0.", "parameter suffix must not be empty"),
        invalid(
            "model.layers.2.input_layernorm.weight",
            "layer 2 is outside 0..2",
        ),
        invalid(
            "model.layers.1.sparse_mlp.routed.routed.0",
            "expected an expert index followed by a parameter",
        ),
        invalid(
            "model.layers.1.sparse_mlp.routed.routed.x.gate_proj.weight",
            "`x` is not a decimal expert index: invalid digit found in string",
        ),
        invalid(
            "model.layers.1.sparse_mlp.routed.routed.4.gate_proj.weight",
            "expert 4 is outside 0..4",
        ),
        missing(
            "model.layers.0.self_attn.q_a_proj.weight",
            "query down-projection",
        ),
        missing(
            "model.layers.0.self_attn.q_a_proj.bias",
            "query down-projection bias",
        ),
        missing(
            "model.layers.0.self_attn.q_a_layernorm.weight",
            "query latent RMSNorm",
        ),
        missing(
            "model.layers.0.self_attn.q_b_proj.weight",
            "query up-projection",
        ),
        missing(
            "model.layers.0.self_attn.kv_a_proj_with_mqa.bias",
            "configured linear bias",
        ),
        missing(
            "model.layers.0.self_attn.o_proj.bias",
            "configured linear bias",
        ),
        missing("model.layers.0.sparse_mlp.routed.router.weight", sparse),
        missing("model.layers.0.sparse_mlp.routed.correction_bias", sparse),
        missing(
            "model.layers.0.sparse_mlp.routed.routed.0.gate_proj.weight",
            sparse,
        ),
        missing(
            "model.layers.0.sparse_mlp.shared_experts.up_proj.weight",
            sparse,
        ),
        missing(
            "model.layers.1.sparse_mlp.routed.correction_bias",
            "noaux_tc selection bias",
        ),
        missing(
            "model.layers.1.dense_mlp.gate_proj.weight",
            "dense SwiGLU block",
        ),
    ];
    for expected in cases {
        let target = error_target(&expected);
        assert_eq!(
            model.write_tensor(&target, scalar()),
            Err(expected),
            "{target}"
        );
    }

    let mut tied_config = tiny_config();
    tied_config.tie_word_embeddings = true;
    let mut tied = DeepseekV2ForCausalLm::<TestBackend>::init(tied_config, &device).unwrap();
    assert_eq!(
        tied.write_tensor("lm_head.weight", scalar()),
        Err(missing(
            "lm_head.weight",
            "independent LM head for an untied checkpoint"
        ))
    );
}

/// Only the tied embedding-to-head alias is accepted, and only after its owner is installed.
#[test]
fn alias_requires_the_tied_installed_embedding_owner() {
    let device = Default::default();
    let alias = |target: &str, source: &str, reason: &str| DeepseekV2LoadError::InvalidAlias {
        target: target.into(),
        source_target: source.into(),
        reason: reason.into(),
    };
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let mut tied = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    for (target, source) in [
        ("lm_head.weight", "model.norm.weight"),
        ("model.norm.weight", "model.embed_tokens.weight"),
    ] {
        assert_eq!(
            tied.alias_tensor(target, source),
            Err(alias(
                target,
                source,
                "only the tied embedding-to-LM-head alias is supported"
            ))
        );
    }
    assert_eq!(
        tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(alias(
            "lm_head.weight",
            "model.embed_tokens.weight",
            "the embedding storage owner has not been installed"
        ))
    );
    tied.write_tensor(
        "model.embed_tokens.weight",
        TensorData::zeros::<f32, _>([32, 8]),
    )
    .unwrap();
    tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight")
        .expect("the installed tied owner aliases the head");

    let mut untied = DeepseekV2ForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();
    assert_eq!(
        untied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(alias(
            "lm_head.weight",
            "model.embed_tokens.weight",
            "the configured model declares an independent LM head"
        ))
    );
}

/// Biased low-rank-query attention installs every bias target and runs.
///
/// The contract adds `q_a_proj.bias`, `kv_a_proj_with_mqa.bias`, and
/// `o_proj.bias` to each layer; each dispatches to its configured Burn bias.
#[test]
fn biased_low_rank_query_targets_install_and_run() {
    let device = Default::default();
    let mut config = tiny_config();
    config.q_lora_rank = Some(4);
    config.attention_bias = true;
    let contract = config.checkpoint_contract(Default::default()).unwrap();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let mut biases = 0;
    for expected in contract.expected_tensors() {
        let shape = expected
            .shape
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        biases += usize::from(expected.name.ends_with(".bias"));
        model
            .write_tensor(&expected.name, TensorData::zeros::<f32, _>(shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    assert_eq!(biases, 3 * 2);
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap();
    assert!(
        output
            .logits
            .into_data()
            .iter::<f32>()
            .all(|value| value == 0.0)
    );
}

/// A sink rejection during streaming is an execution failure with partial diagnostics.
///
/// The checkpoint matches the configuration's contract, so planning succeeds;
/// removing block 1's expert bank afterwards makes the sink reject its first
/// sparse tensor. Configuration and discovery failures carry no execution
/// diagnostics.
#[test]
fn loader_distinguishes_configuration_planning_and_execution_failures() {
    let device = Default::default();
    let config = tiny_config();
    let contract = config.checkpoint_contract(Default::default()).unwrap();
    let temp = TempDir::new().unwrap();
    write_checkpoint(temp.path(), &contract, |_| {});

    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    model.model.layers[1].sparse_mlp = None;
    let error = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .expect_err("the sink cannot install the expert bank");
    let failure = error
        .execution_failure()
        .expect("execution reached the sink");
    let target = failure.failed_target.as_deref().expect("failed target");
    assert!(target.starts_with("model.layers.1.sparse_mlp."), "{target}");
    assert!(matches!(
        &failure.cause,
        crate::checkpoint::ExecutionFailureCause::Sink(DeepseekV2LoadError::MissingComponent {
            component: "sparse routed/shared expert block",
            ..
        })
    ));

    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config.clone(), &device).unwrap();
    model.model.config.topk_group = model.model.config.n_group + 1;
    let error = model
        .load_checkpoint(temp.path(), &LoadOptions::default())
        .expect_err("an invalid configuration has no contract");
    assert!(matches!(
        &error,
        DeepseekV2CheckpointLoadError::Config(config) if config.field == "topk_group"
    ));
    assert!(error.execution_failure().is_none());

    let empty = TempDir::new().unwrap();
    let mut model = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let error = model
        .load_checkpoint(empty.path(), &LoadOptions::default())
        .expect_err("an empty directory has no checkpoint");
    assert!(matches!(
        error,
        DeepseekV2CheckpointLoadError::Checkpoint(_)
    ));
    assert!(error.execution_failure().is_none());
}

/// A tied configuration whose model nevertheless owns an independent head rejects the alias.
#[test]
fn tied_alias_rejects_a_model_that_owns_a_head() {
    let device = Default::default();
    let mut config = tiny_config();
    config.tie_word_embeddings = true;
    let mut tied = DeepseekV2ForCausalLm::<TestBackend>::init(config, &device).unwrap();
    let untied = DeepseekV2ForCausalLm::<TestBackend>::init(tiny_config(), &device).unwrap();
    tied.lm_head = untied.lm_head;
    tied.write_tensor(
        "model.embed_tokens.weight",
        TensorData::zeros::<f32, _>([32, 8]),
    )
    .unwrap();
    assert_eq!(
        tied.alias_tensor("lm_head.weight", "model.embed_tokens.weight"),
        Err(DeepseekV2LoadError::InvalidAlias {
            target: "lm_head.weight".into(),
            source_target: "model.embed_tokens.weight".into(),
            reason: "the configured model declares an independent LM head".into(),
        })
    );
}
