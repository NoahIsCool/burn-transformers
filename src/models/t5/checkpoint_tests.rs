use std::{collections::BTreeMap, path::PathBuf};

use crate::{
    checkpoint::{
        ArtifactSource, CheckpointError, CheckpointManifest, KeyMatcher, LoadOptions,
        PlannedTransform, TensorDType, TensorInventory, TensorOperation, TensorTransform,
    },
    models::config::ModelConfig,
};

use super::*;

/// Returns a deliberately asymmetric tiny topology for ownership tests.
fn tiny(projection: super::super::T5FeedForwardProjection, tied: bool) -> T5Config {
    T5Config {
        vocab_size: 11,
        d_model: 4,
        d_kv: 2,
        d_ff: 6,
        num_layers: 2,
        num_decoder_layers: Some(1),
        num_heads: 2,
        relative_attention_num_buckets: 8,
        relative_attention_max_distance: 16,
        feed_forward_proj: projection,
        dense_act_fn: None,
        is_gated_act: None,
        tie_word_embeddings: tied,
        ..T5Config::default()
    }
}

/// Builds a metadata-only manifest in official checkpoint source layout.
fn manifest(contract: &T5CheckpointContract) -> CheckpointManifest {
    let mut tensors = BTreeMap::new();
    let mut offset = 0_u64;
    for operation in contract.mappings().operations() {
        let TensorOperation::Map {
            source,
            target,
            transform,
        } = operation
        else {
            continue;
        };
        let expected = contract
            .expected_tensors()
            .iter()
            .find(|expected| expected.name == *target)
            .unwrap();
        let mut source_shape = expected.shape.clone();
        if *transform == TensorTransform::Transpose2d {
            source_shape.swap(0, 1);
        }
        let elements = source_shape.iter().product();
        let byte_len = TensorDType::F32.storage_bytes(elements).unwrap();
        let end = offset.checked_add(byte_len).unwrap();
        tensors.insert(
            source.clone(),
            TensorInventory {
                name: source.clone(),
                shard: "metadata-only.safetensors".into(),
                shape: source_shape,
                dtype: TensorDType::F32,
                data_offsets: [offset, end],
                byte_len,
            },
        );
        offset = end;
    }
    CheckpointManifest {
        root: PathBuf::from("metadata-only"),
        source: ArtifactSource::Single {
            relative_path: "metadata-only.safetensors".into(),
        },
        shards: Vec::new(),
        tensors,
        index_aliases: Vec::new(),
        index_alias_size_accounting: None,
        indexed_tensor_bytes: offset,
        tensor_bytes: offset,
        artifact_bytes: offset,
    }
}

/// Finds one expected tensor and returns its Burn runtime shape.
fn shape<'a>(contract: &'a T5CheckpointContract, name: &str) -> &'a [u64] {
    &contract
        .expected_tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap()
        .shape
}

#[test]
fn dense_and_gated_variants_have_exact_historical_keys() {
    use super::super::T5FeedForwardProjection::*;

    for projection in [Relu, Gelu] {
        let contract = tiny(projection, false).checkpoint_contract().unwrap();
        let names = contract
            .expected_tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi.weight"));
        assert!(!names.iter().any(|name| name.ends_with("wi_0.weight")));
        assert_eq!(contract.expected_tensors().len(), 35);
    }

    for projection in [GatedGelu, GatedGeluNew, GatedSilu] {
        let contract = tiny(projection, false).checkpoint_contract().unwrap();
        let names = contract
            .expected_tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi_0.weight"));
        assert!(names.contains(&"encoder.block.0.layer.1.DenseReluDense.wi_1.weight"));
        assert!(
            !names
                .iter()
                .any(|name| name.ends_with("DenseReluDense.wi.weight"))
        );
        assert_eq!(contract.expected_tensors().len(), 38);
    }
}

#[test]
fn relative_bias_is_owned_only_by_each_stack_block_zero() {
    let contract = tiny(super::super::T5FeedForwardProjection::GatedGelu, false)
        .checkpoint_contract()
        .unwrap();
    let bias = contract
        .expected_tensors()
        .iter()
        .filter(|tensor| tensor.name.ends_with("relative_attention_bias.weight"))
        .collect::<Vec<_>>();
    assert_eq!(bias.len(), 2);
    assert_eq!(
        bias[0].name,
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    );
    assert_eq!(
        bias[1].name,
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    );
    assert_eq!(bias[0].shape, [8, 2]);
}

#[test]
fn attention_and_feed_forward_shapes_follow_t5_projection_widths() {
    let mut config = tiny(super::super::T5FeedForwardProjection::GatedGelu, false);
    config.d_kv = 3;
    let contract = config.checkpoint_contract().unwrap();
    assert_eq!(
        shape(&contract, "encoder.block.1.layer.0.SelfAttention.q.weight"),
        [4, 6]
    );
    assert_eq!(
        shape(
            &contract,
            "decoder.block.0.layer.1.EncDecAttention.o.weight"
        ),
        [6, 4]
    );
    assert_eq!(
        shape(
            &contract,
            "decoder.block.0.layer.2.DenseReluDense.wi_0.weight"
        ),
        [4, 6]
    );
    assert_eq!(
        shape(
            &contract,
            "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        ),
        [6, 4]
    );
    assert_eq!(shape(&contract, "lm_head.weight"), [4, 11]);
}

#[test]
fn every_independent_linear_is_reported_as_an_element_equivalent_transpose() {
    use super::super::T5FeedForwardProjection::{GatedGelu, Relu};

    for (projection, tied, expected_transposes) in [
        (Relu, false, 23),
        (Relu, true, 22),
        (GatedGelu, false, 26),
        (GatedGelu, true, 25),
    ] {
        let contract = tiny(projection, tied).checkpoint_contract().unwrap();
        for operation in contract.mappings().operations() {
            let TensorOperation::Map {
                target, transform, ..
            } = operation
            else {
                continue;
            };
            let preserves_layout = target == "shared.weight"
                || target.ends_with("layer_norm.weight")
                || target.ends_with("relative_attention_bias.weight");
            assert_eq!(
                *transform,
                if preserves_layout {
                    TensorTransform::Identity
                } else {
                    TensorTransform::Transpose2d
                },
                "unexpected layout policy for {target}"
            );
        }

        let observed = manifest(&contract);
        let plan = contract
            .build_load_plan(&observed, &LoadOptions::default())
            .unwrap();
        assert_eq!(plan.report.transformed.len(), expected_transposes);
        assert!(plan.report.transformed.iter().all(|record| {
            record.operation == "transpose_2d"
                && record.sources.as_slice() == [record.target.as_str()]
        }));

        // A transpose changes axes but neither element count nor dtype. The
        // unique runtime parameter footprint must therefore equal all
        // consumed source payload bytes exactly.
        assert_eq!(plan.memory.target_parameter_bytes, observed.tensor_bytes);
        assert_eq!(plan.memory.largest_target_tensor_bytes, 11 * 4 * 4);
        assert_eq!(
            plan.memory.largest_target_tensor_bytes,
            plan.memory.largest_source_tensor_bytes
        );
        let alias_bytes = if tied { 11 * 4 * 4 } else { 0 };
        assert_eq!(
            plan.memory.logical_model_bytes,
            observed.tensor_bytes + alias_bytes
        );
        assert_eq!(plan.report.memory, Some(plan.memory));
    }
}

#[test]
fn tied_head_is_a_logical_alias_while_untied_head_consumes_storage() {
    let tied = tiny(super::super::T5FeedForwardProjection::Relu, true)
        .checkpoint_contract()
        .unwrap();
    let tied_plan = tied
        .build_load_plan(&manifest(&tied), &LoadOptions::default())
        .unwrap();
    assert_eq!(shape(&tied, "shared.weight"), [11, 4]);
    assert_eq!(shape(&tied, "lm_head.weight"), [11, 4]);
    assert_eq!(tied_plan.parameter_count(), 35);
    assert_eq!(tied_plan.tensors.len(), 34);
    assert!(matches!(
        tied_plan.aliases.as_slice(),
        [alias]
            if alias.target == "lm_head.weight"
                && matches!(
                    &alias.transform,
                    PlannedTransform::Alias { source_target }
                        if source_target == "shared.weight"
                )
    ));

    let untied = tiny(super::super::T5FeedForwardProjection::Relu, false)
        .checkpoint_contract()
        .unwrap();
    let untied_plan = untied
        .build_load_plan(&manifest(&untied), &LoadOptions::default())
        .unwrap();
    assert_eq!(shape(&untied, "lm_head.weight"), [4, 11]);
    assert_eq!(untied_plan.tensors.len(), 35);
    assert!(untied_plan.aliases.is_empty());
}

#[test]
fn known_hugging_face_embedding_aliases_are_exact_and_reported() {
    for tied in [false, true] {
        let contract = tiny(super::super::T5FeedForwardProjection::GatedGelu, tied)
            .checkpoint_contract()
            .unwrap();
        let mut observed = manifest(&contract);
        let shared = observed.tensors["shared.weight"].clone();
        for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"] {
            observed.tensors.insert(
                name.into(),
                TensorInventory {
                    name: name.into(),
                    ..shared.clone()
                },
            );
        }
        if tied {
            observed.tensors.insert(
                "lm_head.weight".into(),
                TensorInventory {
                    name: "lm_head.weight".into(),
                    ..shared
                },
            );
        }

        let plan = contract
            .build_load_plan(&observed, &LoadOptions::default())
            .unwrap();
        assert!(plan.report.allowlisted_ignored.is_empty());
        assert_eq!(
            plan.report.equivalence_planned.len(),
            if tied { 3 } else { 2 }
        );
        assert!(plan.report.equivalence_verified.is_empty());
        assert_eq!(plan.parameter_count(), 38);

        let custom = KeyMatcher::exact("local.optional.weight");
        let mut options = LoadOptions {
            allow_unexpected: vec![
                KeyMatcher::exact("encoder.embed_tokens.weight"),
                custom.clone(),
            ],
            ..LoadOptions::default()
        };
        contract.configure_load_options(&mut options);
        assert_eq!(
            options
                .allow_unexpected
                .iter()
                .filter(|matcher| { **matcher == KeyMatcher::exact("encoder.embed_tokens.weight") })
                .count(),
            1
        );
        assert!(options.allow_unexpected.contains(&custom));
    }
}

#[test]
fn strict_planning_reports_missing_unexpected_and_shape_errors() {
    let contract = tiny(super::super::T5FeedForwardProjection::GatedSilu, false)
        .checkpoint_contract()
        .unwrap();

    let mut missing = manifest(&contract);
    missing.tensors.remove("encoder.final_layer_norm.weight");
    assert!(matches!(
        contract.build_load_plan(&missing, &LoadOptions::default()),
        Err(CheckpointError::MissingMappingSource { tensor })
            if tensor == "encoder.final_layer_norm.weight"
    ));

    let mut unexpected = manifest(&contract);
    let template = unexpected.tensors.values().next().unwrap().clone();
    unexpected.tensors.insert(
        "encoder.block.1.forbidden.weight".into(),
        TensorInventory {
            name: "encoder.block.1.forbidden.weight".into(),
            ..template
        },
    );
    assert!(matches!(
        contract.build_load_plan(&unexpected, &LoadOptions::default()),
        Err(CheckpointError::StrictValidation { unexpected: 1, .. })
    ));

    let mut wrong_shape = manifest(&contract);
    wrong_shape
        .tensors
        .get_mut("decoder.block.0.layer.1.EncDecAttention.k.weight")
        .unwrap()
        .shape = vec![5, 4];
    assert!(matches!(
        contract.build_load_plan(&wrong_shape, &LoadOptions::default()),
        Err(CheckpointError::ShapeMismatch { tensor, .. })
            if tensor == "decoder.block.0.layer.1.EncDecAttention.k.weight"
    ));

    let mut integer = manifest(&contract);
    let shared = integer.tensors.get_mut("shared.weight").unwrap();
    shared.dtype = TensorDType::I32;
    assert!(matches!(
        contract.build_load_plan(&integer, &LoadOptions::default()),
        Err(CheckpointError::InvalidMapping { message })
            if message.contains("shared.weight") && message.contains("I32")
    ));
}

#[test]
fn official_reduced_flan_and_ul2_configs_generate_complete_contracts() {
    let flan =
        T5Config::from_json_str(include_str!("../../../tests/fixtures/flan_t5_xl.json")).unwrap();
    let flan_contract = flan.checkpoint_contract().unwrap();
    assert_eq!(flan_contract.expected_tensors().len(), 558);
    assert!(
        flan_contract
            .mappings()
            .operations()
            .iter()
            .all(|operation| { !matches!(operation, TensorOperation::Alias { .. }) })
    );
    assert_eq!(
        flan_contract
            .mappings()
            .operations()
            .iter()
            .filter(|operation| matches!(
                operation,
                TensorOperation::Map {
                    transform: TensorTransform::Transpose2d,
                    ..
                }
            ))
            .count(),
        433
    );
    assert_eq!(
        shape(
            &flan_contract,
            "decoder.block.23.layer.2.DenseReluDense.wi_0.weight"
        ),
        [2048, 5120]
    );
    assert_eq!(
        flan_contract.index_aliases(),
        [
            IndexTensorAlias::new("encoder.embed_tokens.weight", "shared.weight"),
            IndexTensorAlias::new("decoder.embed_tokens.weight", "shared.weight"),
        ]
    );
    let mut options = LoadOptions::default();
    flan_contract.configure_load_options(&mut options);
    flan_contract.configure_load_options(&mut options);
    assert_eq!(options.index_aliases, flan_contract.index_aliases());

    let ul2 = T5Config::from_json_str(include_str!("../../../tests/fixtures/ul2.json")).unwrap();
    let ul2_contract = ul2.checkpoint_contract().unwrap();
    assert_eq!(ul2_contract.expected_tensors().len(), 742);
    assert!(matches!(
        ul2_contract.mappings().operations().last(),
        Some(TensorOperation::Alias { source_target, target })
            if source_target == "shared.weight" && target == "lm_head.weight"
    ));
    assert_eq!(
        ul2_contract.index_aliases(),
        [
            IndexTensorAlias::new("encoder.embed_tokens.weight", "shared.weight"),
            IndexTensorAlias::new("decoder.embed_tokens.weight", "shared.weight"),
            IndexTensorAlias::new("lm_head.weight", "shared.weight"),
        ]
    );
}

#[test]
fn dimension_overflow_is_reported_before_inventory_allocation() {
    let config = T5Config {
        d_model: 1,
        d_kv: 2,
        d_ff: 1,
        vocab_size: 1,
        num_heads: usize::MAX,
        num_layers: 1,
        num_decoder_layers: Some(1),
        relative_attention_num_buckets: 4,
        relative_attention_max_distance: 2,
        ..T5Config::default()
    };
    let error = config.checkpoint_contract().unwrap_err();
    assert!(error.reason.contains("overflows"));
}
