//! Local-tokenizer tests for artifact validation, batching, padding, and chat-template boundaries.

use std::{collections::HashMap, fs};

use burn::backend::Flex;
use tempfile::TempDir;
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    models::{bpe::BPE, wordlevel::WordLevel},
    pre_tokenizers::{byte_level::ByteLevel, whitespace::Whitespace},
    processors::template::TemplateProcessing,
};

use super::*;

/// Writes a deterministic local tokenizer without downloading fixtures.
fn fixture(config: Value) -> TempDir {
    fixture_with(config, |_| {})
}

/// Writes the word-level fixture after letting a test adjust the tokenizer pipeline.
///
/// Vocabulary: `<unk>`=0, `<pad>`=1, `<bos>`=2, `<eos>`=3, `hello`=4, `world`=5.
fn fixture_with(config: Value, configure: impl FnOnce(&mut Tokenizer)) -> TempDir {
    let directory = TempDir::new().expect("temporary tokenizer directory");
    let vocabulary = HashMap::from([
        ("<unk>".to_owned(), 0),
        ("<pad>".to_owned(), 1),
        ("<bos>".to_owned(), 2),
        ("<eos>".to_owned(), 3),
        ("hello".to_owned(), 4),
        ("world".to_owned(), 5),
    ])
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("<unk>".into())
        .build()
        .expect("word-level fixture model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    configure(&mut tokenizer);
    tokenizer
        .save(directory.path().join("tokenizer.json"), false)
        .expect("save tokenizer fixture");
    fs::write(
        directory.path().join("tokenizer_config.json"),
        serde_json::to_vec(&config).expect("serialize config"),
    )
    .expect("write tokenizer config");
    directory
}

/// Supplies common metadata fields used by all positive fixtures.
fn base_config() -> Value {
    serde_json::json!({
        "tokenizer_class": "FixtureTokenizer",
        "padding_side": "left",
        "truncation_side": "right",
        "model_max_length": 16,
        "unk_token": "<unk>",
        "pad_token": {"content": "<pad>"},
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    })
}

/// Metadata normalization resolves string and AddedToken object spellings.
#[test]
fn loads_only_local_artifacts_and_resolves_special_ids() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    assert_eq!(tokenizer.metadata().vocabulary_size, 6);
    assert_eq!(tokenizer.metadata().padding_side, PaddingSide::Left);
    assert_eq!(
        tokenizer.metadata().special_tokens.pad,
        Some(SpecialToken {
            content: "<pad>".into(),
            id: 1
        })
    );
    assert!(tokenizer.root().is_absolute());
    tokenizer
        .validate_model_vocabulary(6)
        .expect("equal tokenizer and model vocabularies must be compatible");
    tokenizer
        .validate_model_vocabulary(8)
        .expect("an accelerator-padded model vocabulary must be compatible");
    let error = tokenizer
        .validate_model_vocabulary(5)
        .expect_err("a tokenizer ID outside the model embedding must fail");
    assert!(error.to_string().contains("ID 5"));
    assert!(error.to_string().contains("vocabulary size 5"));
    assert!(tokenizer.validate_model_vocabulary(0).is_err());
}

/// Debug output identifies the local artifact without dumping tokenizer internals.
#[test]
fn debug_reports_local_metadata_without_the_vocabulary_model() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let debug = format!("{tokenizer:?}");
    assert!(debug.starts_with("LocalTokenizer {"));
    assert!(debug.contains(&format!("root: {:?}", tokenizer.root())));
    assert!(debug.contains("vocabulary_size: 6"));
    assert!(debug.contains("FixtureTokenizer"));
    assert!(!debug.contains("WordLevel"));
    assert!(!debug.contains("\"hello\":"));
    assert!(!debug.contains("\"world\":"));
}

/// A valid tokenizer model may fail on one input; row context and reuse survive.
#[test]
fn encoding_failure_reports_the_row_and_allows_a_later_valid_batch() {
    let directory = fixture(serde_json::json!({}));
    // WordLevel permits loading a vocabulary without an unknown-token entry.
    // Known words still encode, while an unknown word returns a normal error.
    let model = WordLevel::builder()
        .vocab(
            [("hello".into(), 0), ("world".into(), 1)]
                .into_iter()
                .collect(),
        )
        .unk_token("missing-unknown-token".into())
        .build()
        .unwrap();
    let mut raw = Tokenizer::new(model);
    raw.with_pre_tokenizer(Some(Whitespace {}));
    raw.save(directory.path().join("tokenizer.json"), false)
        .unwrap();
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let options = EncodeOptions {
        add_special_tokens: false,
        ..Default::default()
    };
    let error = tokenizer
        .encode_batch(&["hello", "absent", "world"], &options)
        .expect_err("unknown words cannot be encoded without an unknown-token entry");
    assert!(matches!(error, Error::Tokenizer(ref reason)
        if reason.starts_with("failed to encode batch row 1: ") && reason.len() > 30));
    let recovered = tokenizer
        .encode_batch(&["hello world", "world hello"], &options)
        .unwrap();
    assert_eq!(recovered.token_ids(), &[vec![0, 1], vec![1, 0]]);
    assert_eq!(
        recovered.valid_tokens(),
        &[vec![true, true], vec![true, true]]
    );
}

/// Left padding produces a rectangular `true == valid` tensor batch.
#[test]
fn batches_with_semantic_left_padding_and_backend_conversion() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let encoded = tokenizer
        .encode_batch(
            &["hello", "hello world"],
            &EncodeOptions {
                padding: Padding::Longest,
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("encode batch");
    assert_eq!(encoded.token_ids(), &[vec![1, 4], vec![4, 5]]);
    assert_eq!(
        encoded.valid_tokens(),
        &[vec![false, true], vec![true, true]]
    );
    let input = encoded
        .to_causal_input::<Flex>(&Default::default())
        .expect("rectangular batch converts");
    assert_eq!(input.token_ids.dims(), [2, 2]);
    let invalid = input
        .token_mask
        .expect("batch conversion always supplies a validity mask")
        .into_tensor()
        .bool_not();
    assert_eq!(
        crate::architecture::tensor_bool_values(invalid, "inverted tokenizer mask").unwrap(),
        [true, false, false, false]
    );
}

/// Both truncation directions are explicit and fixed padding never truncates silently.
#[test]
fn truncates_on_selected_side_and_rejects_fixed_overflow() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let left = tokenizer
        .encode(
            "hello world hello",
            &EncodeOptions {
                add_special_tokens: false,
                max_length: Some(2),
                truncation_side: Some(TruncationSide::Left),
                ..EncodeOptions::default()
            },
        )
        .expect("left truncation");
    assert_eq!(left.token_ids(), &[vec![5, 4]]);
    let error = tokenizer
        .encode(
            "hello world",
            &EncodeOptions {
                add_special_tokens: false,
                padding: Padding::Fixed(1),
                ..EncodeOptions::default()
            },
        )
        .expect_err("padding must not imply truncation");
    assert!(error.to_string().contains("exceeding padding width"));
}

/// Ragged unpadded batches remain usable on the host but cannot form one tensor.
#[test]
fn rejects_ragged_backend_conversion() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let encoded = tokenizer
        .encode_batch(
            &["hello", "hello world"],
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("ragged host encoding");
    let error = encoded
        .to_causal_input::<Flex>(&Default::default())
        .expect_err("ragged rows cannot form a tensor");
    assert!(error.to_string().contains("not rectangular"));
}

/// Invalid metadata and missing pad tokens produce contextual errors.
#[test]
fn rejects_invalid_sides_limits_and_missing_padding_token() {
    let mut invalid = base_config();
    invalid["padding_side"] = Value::String("middle".into());
    let directory = fixture(invalid);
    assert!(
        LocalTokenizer::from_directory(directory.path())
            .expect_err("invalid side")
            .to_string()
            .contains("padding_side")
    );

    let directory = fixture(serde_json::json!({"unk_token": "<unk>"}));
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let error = tokenizer
        .encode_batch(
            &["hello", "hello world"],
            &EncodeOptions {
                add_special_tokens: false,
                padding: Padding::Longest,
                ..EncodeOptions::default()
            },
        )
        .expect_err("pad token is required");
    assert!(error.to_string().contains("no pad token"));
}

/// Named template parsing rejects duplicate and malformed declarations.
#[test]
fn validates_chat_template_metadata_shapes() {
    let duplicate = serde_json::json!([
        {"name": "same", "template": "one"},
        {"name": "same", "template": "two"}
    ]);
    assert!(parse_chat_templates(Some(&duplicate)).is_err());
    assert!(parse_chat_templates(Some(&Value::Bool(true))).is_err());
    assert!(parse_chat_templates(Some(&Value::String("  ".into()))).is_err());
}

/// Invalid limits fail before encoding, while explicit right-side controls preserve masks.
#[test]
fn validates_encoding_limits_and_right_padding_boundaries() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    for (options, expected) in [
        (
            EncodeOptions {
                max_length: Some(0),
                ..Default::default()
            },
            "max_length",
        ),
        (
            EncodeOptions {
                padding: Padding::Fixed(0),
                ..Default::default()
            },
            "fixed padding length",
        ),
        (
            EncodeOptions {
                padding: Padding::Fixed(1),
                max_length: Some(2),
                ..Default::default()
            },
            "exceeds fixed padding width",
        ),
    ] {
        let error = tokenizer.encode("hello", &options).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let error = tokenizer
        .encode_batch::<&str>(&[], &EncodeOptions::default())
        .unwrap_err();
    assert!(error.to_string().contains("at least one input"));
    let error = tokenizer
        .encode(
            "",
            &EncodeOptions {
                add_special_tokens: false,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("batch row 0 encoded to no tokens")
    );
    let encoded = tokenizer
        .encode_batch(
            &["hello world hello", "world"],
            &EncodeOptions {
                add_special_tokens: false,
                max_length: Some(2),
                padding: Padding::Fixed(2),
                padding_side: Some(PaddingSide::Right),
                truncation_side: Some(TruncationSide::Right),
            },
        )
        .unwrap();
    assert_eq!(encoded.token_ids(), &[vec![4, 5], vec![5, 1]]);
    assert_eq!(
        encoded.valid_tokens(),
        &[vec![true, true], vec![true, false]]
    );
    assert_eq!(encoded.batch_size(), 2);
    assert_eq!(tokenizer.decode(&[4, 5], false).unwrap(), "hello world");
}

/// Empty or mask-mismatched host encodings fail before tensor allocation and empty vocabularies fail validation.
#[test]
fn rejects_empty_host_batches_and_tokenizer_vocabulary() {
    for token_ids in [vec![], vec![vec![]]] {
        let encoding = BatchEncoding {
            token_ids,
            valid_tokens: vec![],
        };
        assert!(
            encoding
                .to_causal_input::<Flex>(&Default::default())
                .unwrap_err()
                .to_string()
                .contains("empty token batch")
        );
    }
    let missing_masks = BatchEncoding {
        token_ids: vec![vec![1, 2], vec![3, 4]],
        valid_tokens: vec![vec![true, true]],
    };
    assert!(
        missing_masks
            .to_causal_input::<Flex>(&Default::default())
            .unwrap_err()
            .to_string()
            .contains("2 token rows but 1 mask rows")
    );
    let directory = fixture(base_config());
    let mut tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    tokenizer.tokenizer = Tokenizer::new(WordLevel::default());
    assert!(
        tokenizer
            .validate_model_vocabulary(6)
            .unwrap_err()
            .to_string()
            .contains("vocabulary must not be empty")
    );
}

/// Missing, malformed, oversized and non-file artifacts retain their local path.
#[test]
fn local_artifact_errors_identify_the_failing_file() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let missing = LocalTokenizer::from_directory(&root).unwrap_err();
    assert!(matches!(missing, Error::Io { ref path, .. } if path.ends_with("tokenizer.json")));
    let artifact = root.join("tokenizer.json");
    fs::write(&artifact, b"not tokenizer json").unwrap();
    assert!(
        LocalTokenizer::from_directory(&root)
            .unwrap_err()
            .to_string()
            .contains("failed to decode")
    );
    assert!(matches!(
        LocalTokenizer::from_directory(&artifact).unwrap_err(),
        Error::Artifact { .. }
    ));
    assert!(
        matches!(contained_file(&root, "tokenizer.json", true, 3).unwrap_err(), Error::Artifact { reason, .. } if reason.contains("exceeds safety limit"))
    );
    fs::create_dir(root.join("directory.json")).unwrap();
    assert!(
        matches!(contained_file(&root, "directory.json", true, 100).unwrap_err(), Error::Artifact { reason, .. } if reason.contains("regular file"))
    );
    assert!(
        contained_file(&root, "absent.json", false, 100)
            .unwrap()
            .is_none()
    );

    let directory = fixture(base_config());
    fs::write(directory.path().join("tokenizer_config.json"), b"{").unwrap();
    assert!(
        matches!(LocalTokenizer::from_directory(directory.path()).unwrap_err(), Error::Json { path, .. } if path.ends_with("tokenizer_config.json"))
    );
}

/// Symlink containment is enforced for tokenizer and optional metadata files alike.
#[cfg(unix)]
#[test]
fn local_artifacts_reject_escaping_symlinks() {
    use std::os::unix::fs::symlink;

    let outside = fixture(base_config());
    let root = TempDir::new().unwrap();
    symlink(
        outside.path().join("tokenizer.json"),
        root.path().join("tokenizer.json"),
    )
    .unwrap();
    let error = LocalTokenizer::from_directory(root.path()).unwrap_err();
    assert!(matches!(error, Error::Artifact { reason, .. } if reason.contains("symlink escapes")));

    let inside = fixture(base_config());
    symlink(
        outside.path().join("tokenizer_config.json"),
        inside.path().join("special_tokens_map.json"),
    )
    .unwrap();
    let error = LocalTokenizer::from_directory(inside.path()).unwrap_err();
    assert!(
        matches!(error, Error::Artifact { path, reason } if path.ends_with("special_tokens_map.json") && reason.contains("symlink escapes"))
    );
}

/// Metadata objects, sides and standalone template files are validated independently.
#[test]
fn validates_metadata_objects_and_standalone_chat_templates() {
    for value in [Value::Null, serde_json::json!([]), Value::Bool(true)] {
        let directory = fixture(value.clone());
        assert!(
            matches!(LocalTokenizer::from_directory(directory.path()).unwrap_err(), Error::Artifact { path, .. } if path.ends_with("tokenizer_config.json"))
        );
        let directory = fixture(base_config());
        fs::write(
            directory.path().join("special_tokens_map.json"),
            value.to_string(),
        )
        .unwrap();
        assert!(
            matches!(LocalTokenizer::from_directory(directory.path()).unwrap_err(), Error::Artifact { path, .. } if path.ends_with("special_tokens_map.json"))
        );
    }
    let mut config = base_config();
    config["truncation_side"] = serde_json::json!(17);
    let directory = fixture(config);
    assert!(
        LocalTokenizer::from_directory(directory.path())
            .unwrap_err()
            .to_string()
            .contains("truncation_side")
    );
    let mut config = base_config();
    config["truncation_side"] = serde_json::json!("left");
    config["padding_side"] = serde_json::json!("right");
    let directory = fixture(config);
    fs::write(
        directory.path().join("chat_template.jinja"),
        "hello {{ messages[0].content }}",
    )
    .unwrap();
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    assert_eq!(tokenizer.metadata().truncation_side, TruncationSide::Left);
    assert_eq!(tokenizer.metadata().padding_side, PaddingSide::Right);
    assert_eq!(
        tokenizer.metadata().chat_templates["default"],
        "hello {{ messages[0].content }}"
    );
    fs::write(directory.path().join("chat_template.jinja"), " \n ").unwrap();
    assert!(
        matches!(LocalTokenizer::from_directory(directory.path()).unwrap_err(), Error::Artifact { path, reason } if path.ends_with("chat_template.jinja") && reason.contains("must not be empty"))
    );
}

/// Special-token IDs and strings cannot disagree, and null declarations allow fallback.
#[test]
fn special_metadata_resolves_ids_precedence_and_rejects_conflicts() {
    let directory = fixture(base_config());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let special_map = serde_json::json!({"eos_token": {"content": "<eos>"}});
    for config in [
        serde_json::json!({"eos_token_id": 3}),
        serde_json::json!({"eos_token": null}),
    ] {
        assert_eq!(
            resolve_special(
                "eos_token",
                Some(&config),
                Some(&special_map),
                &tokenizer.tokenizer
            )
            .unwrap(),
            Some(SpecialToken {
                content: "<eos>".into(),
                id: 3
            })
        );
    }
    for (config, expected) in [
        (
            serde_json::json!({"eos_token": {"content": 7}}),
            "could not resolve",
        ),
        (serde_json::json!({"eos_token_id": 99}), "could not resolve"),
        (
            serde_json::json!({"eos_token": "unknown-special"}),
            "absent from the vocabulary",
        ),
        (
            serde_json::json!({"eos_token": "<eos>", "eos_token_id": 2}),
            "conflicts with tokenizer vocabulary",
        ),
    ] {
        let error =
            resolve_special("eos_token", Some(&config), None, &tokenizer.tokenizer).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let config = serde_json::json!({"additional_special_tokens": ["<bos>", {"content": "<eos>"}]});
    assert_eq!(
        resolve_additional(None, Some(&config), &tokenizer.tokenizer).unwrap(),
        vec![
            SpecialToken {
                content: "<bos>".into(),
                id: 2
            },
            SpecialToken {
                content: "<eos>".into(),
                id: 3
            },
        ]
    );
    for (tokens, expected) in [
        (serde_json::json!(true), "must be an array"),
        (serde_json::json!([17]), "must be a string or AddedToken"),
        (serde_json::json!(["missing"]), "absent from the vocabulary"),
        (
            serde_json::json!(["<eos>", {"content": "<eos>"}]),
            "duplicate additional special token",
        ),
    ] {
        let config = serde_json::json!({"additional_special_tokens": tokens});
        let error = resolve_additional(Some(&config), None, &tokenizer.tokenizer).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

/// An explicit ID cannot make absent special-token text into a vocabulary member.
#[test]
fn local_special_tokens_reject_unbacked_text_even_with_a_declared_id() {
    for field in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        for id in [1, 99] {
            let mut config = base_config();
            config[field] = Value::String("missing-special".into());
            config[format!("{field}_id")] = Value::from(id);
            let directory = fixture(config);
            let error = LocalTokenizer::from_directory(directory.path())
                .expect_err("an ID must not authorize text absent from the actual vocabulary");
            assert!(matches!(error, Error::Tokenizer(ref reason)
                if reason.contains(field) && reason.contains("absent from the vocabulary")));
        }
    }
}

/// Declared special IDs must be losslessly representable, not silently discarded.
#[test]
fn local_special_tokens_reject_invalid_id_types_and_ranges() {
    for field in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        for id in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("1"),
            serde_json::json!(true),
            serde_json::json!([]),
            serde_json::json!(4_294_967_296_u64),
        ] {
            let mut config = base_config();
            config[format!("{field}_id")] = id;
            let directory = fixture(config);
            let error = LocalTokenizer::from_directory(directory.path())
                .expect_err("present invalid special IDs must not become absent IDs");
            assert!(matches!(error, Error::Tokenizer(ref reason)
                if reason.contains(&format!("{field}_id")) && reason.contains("unsigned 32-bit integer")));
        }
    }
    // A null ID remains an ordinary absent override, and a matching integer
    // remains valid. In either case emitted padding must index the vocabulary.
    for id in [Value::Null, Value::from(1)] {
        let mut config = base_config();
        config["pad_token_id"] = id;
        let directory = fixture(config);
        let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
        tokenizer.validate_model_vocabulary(6).unwrap();
        let encoding = tokenizer
            .encode_batch(
                &["hello", "hello world"],
                &EncodeOptions {
                    padding: Padding::Longest,
                    add_special_tokens: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(encoding.token_ids(), &[vec![1, 4], vec![4, 5]]);
        assert_eq!(
            encoding.valid_tokens(),
            &[vec![false, true], vec![true, true]]
        );
    }
}

/// ID-only metadata is supported, but an explicitly malformed text declaration is not.
#[test]
fn local_special_token_text_errors_cannot_be_hidden_by_valid_ids() {
    for declaration in [
        serde_json::json!(7),
        serde_json::json!(false),
        serde_json::json!([]),
        serde_json::json!({"content": 7}),
    ] {
        let mut config = base_config();
        config["pad_token"] = declaration;
        config["pad_token_id"] = Value::from(1);
        let directory = fixture(config);
        let error = LocalTokenizer::from_directory(directory.path())
            .expect_err("an explicit malformed text field is not an ID-only declaration");
        assert!(matches!(error, Error::Tokenizer(ref reason)
            if reason.contains("could not resolve declared pad_token to token text")));
    }
    let directory = fixture(serde_json::json!({"pad_token_id": 1}));
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    assert_eq!(
        tokenizer.metadata().special_tokens.pad,
        Some(SpecialToken {
            content: "<pad>".into(),
            id: 1
        })
    );
}

/// All supported template metadata shapes normalize identically and malformed entries fail.
#[test]
fn named_template_metadata_validates_each_entry() {
    let expected = BTreeMap::from([("chat".into(), "hello".into())]);
    for value in [
        serde_json::json!({"chat": "hello"}),
        serde_json::json!([{"name": "chat", "template": "hello"}]),
    ] {
        assert_eq!(parse_chat_templates(Some(&value)).unwrap(), expected);
    }
    for (value, expected) in [
        (
            serde_json::json!([{"template": "hello"}]),
            ".name must be a string",
        ),
        (
            serde_json::json!([{"name": "chat"}]),
            ".template must be a string",
        ),
        (serde_json::json!({"chat": 17}), "must be a string"),
        (
            serde_json::json!({"": "hello"}),
            "name and source must not be empty",
        ),
    ] {
        let error = parse_chat_templates(Some(&value)).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

/// Rendering chooses only an explicit, default, or unambiguous sole template.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_rendering_validates_selection_and_reserved_context() {
    let directory = fixture(base_config());
    let mut tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let messages = [ChatMessage::text("user", "hello")];
    assert!(
        tokenizer
            .render_chat(&[], &ChatTemplateOptions::default())
            .unwrap_err()
            .to_string()
            .contains("at least one message")
    );
    assert!(
        tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap_err()
            .to_string()
            .contains("does not provide a chat template")
    );
    tokenizer.metadata.chat_templates = BTreeMap::from([("chat".into(), "hello".into())]);
    assert_eq!(
        tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap(),
        "hello"
    );
    tokenizer
        .metadata
        .chat_templates
        .insert("tools".into(), "world".into());
    assert!(
        tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap_err()
            .to_string()
            .contains("multiple chat templates")
    );
    assert_eq!(
        tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    template_name: Some("tools".into()),
                    ..Default::default()
                }
            )
            .unwrap(),
        "world"
    );
    assert!(
        tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    template_name: Some("missing".into()),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("was not found")
    );
    tokenizer.metadata.chat_templates.insert(
        "default".into(),
        "{{ bos_token }}{{ extra }}{{ eos_token }}".into(),
    );
    assert_eq!(
        tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    extra_context: BTreeMap::from([("extra".into(), serde_json::json!("hello"))]),
                    ..Default::default()
                }
            )
            .unwrap(),
        "<bos>hello<eos>"
    );
    for reserved in [
        "messages",
        "add_generation_prompt",
        "tools",
        "add_vision_id",
        "enable_thinking",
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
    ] {
        let error = tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    extra_context: BTreeMap::from([(reserved.into(), Value::Null)]),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("reserved variable `{reserved}`")),
            "{error}"
        );
    }
}

/// Python string methods validate values and argument arity inside the template.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_python_string_methods_match_supported_semantics() {
    let directory = fixture(base_config());
    let mut tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let messages = [ChatMessage::text("user", "hello")];
    for (expression, expected) in [
        // Booleans print as Python's `str(True)`, as in Hugging Face.
        ("'Hello'.startswith('He')", "True"),
        ("'Hello'.endswith('lo')", "True"),
        ("'a:b::c'.split(':') | join('|')", "a|b||c"),
        ("'  x  '.lstrip()", "x  "),
        ("'  x  '.rstrip()", "  x"),
        ("'  x  '.strip()", "x"),
        ("'xyhelloyx'.lstrip('xy')", "helloyx"),
        ("'xyhelloyx'.rstrip('xy')", "xyhello"),
        ("'xyhelloyx'.strip('xy')", "hello"),
        ("'hello'.strip('')", "hello"),
        ("'HeLLo'.lower()", "hello"),
        ("'HeLLo'.upper()", "HELLO"),
        ("'x y'.split() | join('|')", "x|y"),
    ] {
        tokenizer
            .metadata
            .chat_templates
            .insert("default".into(), format!("{{{{ {expression} }}}}"));
        assert_eq!(
            tokenizer
                .render_chat(&messages, &ChatTemplateOptions::default())
                .unwrap(),
            expected,
            "{expression}"
        );
    }
    for expression in [
        "'x'.startswith()",
        "'x'.endswith('a','b')",
        "'x'.startswith(1)",
        "'x'.strip(1)",
        "'x'.strip('a','b')",
        "'x'.lower(1)",
        "'x'.unsupported()",
        "17.startswith('1')",
    ] {
        tokenizer
            .metadata
            .chat_templates
            .insert("default".into(), format!("{{{{ {expression} }}}}"));
        let error = tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap_err();
        assert!(
            error.to_string().contains("chat template `default` failed"),
            "{expression}: {error}"
        );
    }
}

/// Bounded MiniJinja rendering exposes only intended chat variables.
#[cfg(feature = "chat-templates")]
#[test]
fn renders_chat_and_propagates_template_failures() {
    let mut config = base_config();
    config["chat_template"] = Value::String(
        "{% for message in messages %}{{ message.role }}={{ message.content }};{% endfor %}{% if add_generation_prompt %}assistant={% endif %}"
            .into(),
    );
    let directory = fixture(config);
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let rendered = tokenizer
        .render_chat(
            &[ChatMessage::text("user", "hello")],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                ..ChatTemplateOptions::default()
            },
        )
        .expect("render template");
    assert_eq!(rendered, "user=hello;assistant=");

    let encoded = tokenizer
        .encode_chat(
            &[ChatMessage::text("user", "hello")],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                ..ChatTemplateOptions::default()
            },
            &EncodeOptions {
                add_special_tokens: true,
                ..EncodeOptions::default()
            },
        )
        .expect("rendered chat must tokenize without a second special-token pass");
    let manual = tokenizer
        .encode(
            &rendered,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .expect("manual reference encoding");
    assert_eq!(encoded, manual);

    let mut failing = base_config();
    failing["chat_template"] = Value::String("{{ raise_exception('stop') }}".into());
    let directory = fixture(failing);
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("load fixture");
    let error = tokenizer
        .render_chat(
            &[ChatMessage::text("user", "hello")],
            &ChatTemplateOptions::default(),
        )
        .expect_err("raise_exception must propagate");
    assert!(error.to_string().contains("stop"));
}

/// Installs a single-sequence post-processor template such as `<bos> $A <eos>`.
fn with_template(tokenizer: &mut Tokenizer, template: &str) {
    let processor = TemplateProcessing::builder()
        .try_single(template)
        .expect("valid fixture template")
        .special_tokens(vec![("<bos>", 2), ("<eos>", 3)])
        .build()
        .expect("fixture post-processor");
    tokenizer.with_post_processor(Some(processor));
}

/// Encodes one string with explicit special-token and truncation controls.
fn encode_ids(
    tokenizer: &LocalTokenizer,
    text: &str,
    add_special_tokens: bool,
    max_length: Option<usize>,
    side: TruncationSide,
) -> Result<Vec<u32>> {
    tokenizer
        .encode(
            text,
            &EncodeOptions {
                add_special_tokens,
                max_length,
                truncation_side: Some(side),
                ..EncodeOptions::default()
            },
        )
        .map(|encoding| encoding.token_ids()[0].clone())
}

/// Truncation cuts content, never the BOS/EOS template, on either side (TOK-1).
#[test]
fn truncation_preserves_post_processor_special_tokens() {
    let directory = fixture_with(base_config(), |tokenizer| {
        with_template(tokenizer, "<bos> $A <eos>");
    });
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    assert_eq!(tokenizer.num_special_tokens_to_add(), 2);
    // Content IDs are 4 5 5 4; the template adds 2 before and 3 after.
    let text = "hello world world hello";
    let encode = |max_length, side| encode_ids(&tokenizer, text, true, max_length, side);
    assert_eq!(
        encode(None, TruncationSide::Right).unwrap(),
        [2, 4, 5, 5, 4, 3]
    );
    assert_eq!(
        encode(Some(4), TruncationSide::Right).unwrap(),
        [2, 4, 5, 3]
    );
    assert_eq!(encode(Some(4), TruncationSide::Left).unwrap(), [2, 5, 4, 3]);
    // A limit equal to the template keeps only the template, as Hugging Face does.
    assert_eq!(encode(Some(2), TruncationSide::Right).unwrap(), [2, 3]);
    // Limits at or above the encoded width leave the row unchanged.
    for max_length in [6, 7] {
        assert_eq!(
            encode(Some(max_length), TruncationSide::Left).unwrap(),
            [2, 4, 5, 5, 4, 3]
        );
    }
    let error = encode(Some(1), TruncationSide::Right).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("max_length 1 cannot hold the 2 special tokens"),
        "{error}"
    );
    // Without special tokens the full limit is available to content.
    assert_eq!(
        encode_ids(&tokenizer, text, false, Some(2), TruncationSide::Left).unwrap(),
        [5, 4]
    );
    assert_eq!(
        encode_ids(&tokenizer, text, false, Some(1), TruncationSide::Right).unwrap(),
        [4]
    );
}

/// A T5-style `$A </s>` template keeps its trailing EOS under right truncation.
#[test]
fn t5_style_truncation_ends_with_eos() {
    let directory = fixture_with(base_config(), |tokenizer| {
        with_template(tokenizer, "$A <eos>");
    });
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    assert_eq!(tokenizer.num_special_tokens_to_add(), 1);
    let encoded = tokenizer
        .encode_batch(
            &["hello world hello world", "world"],
            &EncodeOptions {
                max_length: Some(3),
                padding: Padding::Longest,
                padding_side: Some(PaddingSide::Right),
                ..EncodeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(encoded.token_ids(), &[vec![4, 5, 3], vec![5, 3, 1]]);
    assert_eq!(
        encoded.valid_tokens(),
        &[vec![true, true, true], vec![true, true, false]]
    );
    // Only the template survives a limit equal to its width.
    assert_eq!(
        encode_ids(&tokenizer, "hello", true, Some(1), TruncationSide::Right).unwrap(),
        [3]
    );
}

/// Truncation and padding serialized in `tokenizer.json` never apply implicitly.
#[test]
fn serialized_truncation_and_padding_are_cleared_at_load() {
    let directory = fixture_with(base_config(), |tokenizer| {
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 1,
                ..TruncationParams::default()
            }))
            .expect("fixture truncation");
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            pad_id: 1,
            pad_token: "<pad>".into(),
            ..PaddingParams::default()
        }));
    });
    let raw = Tokenizer::from_file(directory.path().join("tokenizer.json")).unwrap();
    assert!(raw.get_truncation().is_some() && raw.get_padding().is_some());
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let encoded = tokenizer
        .encode("hello world hello", &EncodeOptions::default())
        .unwrap();
    assert_eq!(encoded.token_ids(), &[vec![4, 5, 4]]);
    assert_eq!(encoded.valid_tokens(), &[vec![true; 3]]);
}

proptest::proptest! {
    /// For any content and limit, the template survives and the content is the reference window.
    #[test]
    fn truncation_matches_reference_content_window(
        words in proptest::collection::vec(proptest::bool::ANY, 0..12),
        max_length in 2_usize..16,
        left in proptest::bool::ANY,
    ) {
        let directory = fixture_with(base_config(), |tokenizer| {
            with_template(tokenizer, "<bos> $A <eos>");
        });
        let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
        let content = words.iter().map(|word| if *word { 4 } else { 5 }).collect::<Vec<u32>>();
        let text = words
            .iter()
            .map(|word| if *word { "hello" } else { "world" })
            .collect::<Vec<_>>()
            .join(" ");
        let side = if left { TruncationSide::Left } else { TruncationSide::Right };
        let ids = encode_ids(&tokenizer, &text, true, Some(max_length), side).unwrap();

        let budget = max_length - 2;
        let kept = content.len().min(budget);
        let window = if left { &content[content.len() - kept..] } else { &content[..kept] };
        let mut expected = vec![2];
        expected.extend_from_slice(window);
        expected.push(3);
        proptest::prop_assert_eq!(ids, expected);
    }
}

/// Golden IDs from Hugging Face `transformers` 5.2.0 / `tokenizers` 0.22.2 for FLAN-T5.
///
/// Generated with `AutoTokenizer.from_pretrained(<pinned flan-t5-small>)` and
/// `tokenizer(text, truncation=True, max_length=m)` for each truncation side.
#[test]
#[ignore = "requires TRANSFORMERS_FLAN_T5_SMALL_DIR with the pinned FLAN-T5-small tokenizer"]
fn flan_t5_truncation_matches_hugging_face_golden_ids() {
    let root = std::env::var_os("TRANSFORMERS_FLAN_T5_SMALL_DIR")
        .expect("TRANSFORMERS_FLAN_T5_SMALL_DIR must name the pinned FLAN-T5-small snapshot");
    let tokenizer = LocalTokenizer::from_directory(root).unwrap();
    let text = "translate English to German: the quick brown fox jumps over the lazy dog";
    let full = [
        13959, 1566, 12, 2968, 10, 8, 1704, 4216, 3, 20400, 4418, 7, 147, 8, 19743, 1782, 1,
    ];
    assert_eq!(
        encode_ids(&tokenizer, text, true, None, TruncationSide::Right).unwrap(),
        full
    );
    for (max_length, side, expected) in [
        (
            8,
            TruncationSide::Right,
            vec![13959, 1566, 12, 2968, 10, 8, 1704, 1],
        ),
        (
            8,
            TruncationSide::Left,
            vec![20400, 4418, 7, 147, 8, 19743, 1782, 1],
        ),
        (2, TruncationSide::Right, vec![13959, 1]),
        (2, TruncationSide::Left, vec![1782, 1]),
        (1, TruncationSide::Right, vec![1]),
        (1, TruncationSide::Left, vec![1]),
    ] {
        assert_eq!(
            encode_ids(&tokenizer, text, true, Some(max_length), side).unwrap(),
            expected,
            "max_length={max_length} side={side:?}"
        );
    }
    assert_eq!(
        encode_ids(&tokenizer, text, false, Some(4), TruncationSide::Right).unwrap(),
        [13959, 1566, 12, 2968]
    );
}

/// Loads the base fixture with one default chat template.
#[cfg(feature = "chat-templates")]
fn chat_fixture(template: &str) -> (TempDir, LocalTokenizer) {
    let mut config = base_config();
    config["chat_template"] = Value::String(template.into());
    let directory = fixture(config);
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    (directory, tokenizer)
}

/// Unset optional message fields are absent, so `'tool_calls' in message` is false (TOK-3).
#[cfg(feature = "chat-templates")]
#[test]
fn chat_messages_omit_unset_fields_like_python_dicts() {
    // Golden from transformers 5.2.0 render_jinja_template with the same messages.
    let template = "{%- for message in messages %}\n    {%- if 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}{{ raise_exception('one tool call') }}{%- endif %}\n        {%- set call = message.tool_calls[0].function %}\n        {{- '<|call|>' + call.name + ' ' + call.arguments | tojson + '<|end|>' }}\n    {%- elif message.role == 'tool' %}\n        {{- '<|tool ' + message.tool_call_id + '|>' + message.content + '<|end|>' }}\n    {%- else %}\n        {{- '<|' + message.role + '|>' + message.content + '<|end|>' }}\n    {%- endif %}\n{%- endfor %}";
    let (_directory, tokenizer) = chat_fixture(template);
    let messages = [
        ChatMessage::text("user", "weather?"),
        ChatMessage::text("assistant", "").with_tool_calls(serde_json::json!([{
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}}
        }])),
        ChatMessage::text("tool", "21").with_tool_call_id("call_1"),
    ];
    assert_eq!(
        tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap(),
        "<|user|>weather?<|end|><|call|>get_weather {\"city\": \"Zürich\", \"unit\": \"c\"}<|end|><|tool call_1|>21<|end|>"
    );
    let serialized = serde_json::to_value(ChatMessage::text("user", "hi")).unwrap();
    assert_eq!(
        serialized,
        serde_json::json!({"role": "user", "content": "hi"})
    );
    let full = ChatMessage::new(
        "assistant",
        serde_json::json!([{"type": "text", "text": "x"}]),
    )
    .with_reasoning_content("think")
    .with_name("bot")
    .with_tool_calls(serde_json::json!([]))
    .with_tool_call_id("id")
    .with_field("thinking", true);
    let value = serde_json::to_value(&full).unwrap();
    assert_eq!(
        value,
        serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "reasoning_content": "think",
            "name": "bot",
            "tool_calls": [],
            "tool_call_id": "id",
            "thinking": true
        })
    );
    // Deserialization collects unknown keys into `extra` and round-trips.
    let parsed: ChatMessage = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, full);
}

/// Extra message keys cannot silently replace a named field.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_message_extra_keys_cannot_shadow_named_fields() {
    let (_directory, tokenizer) = chat_fixture("{{ messages[0].role }}");
    for field in [
        "role",
        "content",
        "reasoning_content",
        "tool_calls",
        "name",
        "tool_call_id",
    ] {
        let message = ChatMessage::text("user", "x").with_field(field, "shadow");
        let error = tokenizer
            .render_chat(&[message], &ChatTemplateOptions::default())
            .unwrap_err();
        assert!(
            error.to_string().contains(&format!("extra key `{field}`")),
            "{error}"
        );
    }
    let message = ChatMessage::text("user", "x").with_field("custom", "ok");
    assert_eq!(
        tokenizer
            .render_chat(&[message], &ChatTemplateOptions::default())
            .unwrap(),
        "user"
    );
}

/// The context matches `apply_chat_template`: tools, documents, special tokens, optional switches.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_context_matches_hugging_face_variables() {
    // Golden: HF passes tools=None and documents=None and omits undeclared special tokens.
    let template = "{{ bos_token }}|{{ eos_token }}|{{ unk_token }}|{{ pad_token }}|{{ sep_token }}|{{ cls_token }}|{{ mask_token }}|{{ add_generation_prompt }}|{{ tools }}|{{ documents }}|{{ enable_thinking is defined }}|{{ add_vision_id is defined }}";
    let (_directory, mut tokenizer) = chat_fixture(template);
    let messages = [ChatMessage::text("user", "x")];
    assert_eq!(
        tokenizer
            .render_chat(&messages, &ChatTemplateOptions::default())
            .unwrap(),
        "<bos>|<eos>|<unk>|<pad>||||False|None|None|False|False"
    );
    tokenizer.metadata.special_tokens = SpecialTokens::default();
    let options = ChatTemplateOptions {
        add_generation_prompt: true,
        tools: Some(serde_json::json!([{"name": "t"}])),
        documents: Some(serde_json::json!([{"text": "d", "title": "T"}])),
        enable_thinking: Some(false),
        add_vision_id: true,
        ..ChatTemplateOptions::default()
    };
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        "|||||||True|[{'name': 't'}]|[{'text': 'd', 'title': 'T'}]|True|True"
    );
    for documents in [serde_json::json!("text"), serde_json::json!([1])] {
        let error = tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    documents: Some(documents),
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("list of objects"), "{error}");
    }
    for reserved in [
        "documents",
        "sep_token",
        "cls_token",
        "mask_token",
        "raise_exception",
        "strftime_now",
    ] {
        let error = tokenizer
            .render_chat(
                &messages,
                &ChatTemplateOptions {
                    extra_context: BTreeMap::from([(reserved.into(), Value::Null)]),
                    ..ChatTemplateOptions::default()
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("reserved variable"), "{error}");
    }
}

/// Declared `sep`/`cls`/`mask` tokens are resolved and exposed to templates.
#[cfg(feature = "chat-templates")]
#[test]
fn encoder_special_tokens_are_resolved_and_exposed() {
    let mut config = base_config();
    config["sep_token"] = serde_json::json!("hello");
    config["cls_token"] = serde_json::json!({"content": "world"});
    config["mask_token"] = serde_json::json!("<unk>");
    config["chat_template"] = serde_json::json!("{{ sep_token }}{{ cls_token }}{{ mask_token }}");
    let directory = fixture(config);
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let special = &tokenizer.metadata().special_tokens;
    assert_eq!(special.sep.as_ref().map(|token| token.id), Some(4));
    assert_eq!(special.cls.as_ref().map(|token| token.id), Some(5));
    assert_eq!(special.mask.as_ref().map(|token| token.id), Some(0));
    assert_eq!(
        special.named().map(|(name, _)| name).collect::<Vec<_>>(),
        [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token"
        ]
    );
    assert_eq!(
        tokenizer
            .render_chat(
                &[ChatMessage::text("user", "x")],
                &ChatTemplateOptions::default()
            )
            .unwrap(),
        "helloworld<unk>"
    );
}

/// `continue_final_message` cuts the prompt after the final text, as Hugging Face does.
#[cfg(feature = "chat-templates")]
#[test]
fn continue_final_message_leaves_the_final_turn_open() {
    let (_directory, tokenizer) = chat_fixture(
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>\n{% endfor %}",
    );
    let options = ChatTemplateOptions {
        continue_final_message: true,
        ..ChatTemplateOptions::default()
    };
    let messages = [
        ChatMessage::text("user", "hi"),
        ChatMessage::text("assistant", "The answer is"),
    ];
    // Golden from transformers 5.2.0.
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        "<user>hi</user>\n<assistant>The answer is"
    );
    // Structured content continues its last text block.
    let structured = [ChatMessage::new(
        "assistant",
        serde_json::json!([{"type": "text", "text": "first"}, {"type": "image"}, {"type": "text", "text": "second"}]),
    )];
    let (_directory, blocks) = chat_fixture(
        "{% for block in messages[0].content %}{% if block.text %}[{{ block.text }}]{% endif %}{% endfor %}",
    );
    assert_eq!(
        blocks.render_chat(&structured, &options).unwrap(),
        "[first][second"
    );
    // Whitespace trimmed by the template after the marker is trimmed from the prompt too.
    let (_directory, trimming) = chat_fixture("{{ messages[0].content | trim }}  END");
    assert_eq!(
        trimming
            .render_chat(&[ChatMessage::text("assistant", "go")], &options)
            .unwrap(),
        "go"
    );
    for (messages, template, expected) in [
        (
            vec![ChatMessage::new("assistant", Value::Null)],
            "x",
            "contain text",
        ),
        (
            vec![ChatMessage::new(
                "assistant",
                serde_json::json!([{"type": "image"}]),
            )],
            "x",
            "contain text",
        ),
        (
            vec![ChatMessage::text("assistant", "go")],
            "static",
            "dropped the final message",
        ),
        (
            vec![ChatMessage::text("assistant", "go")],
            "{{ messages[0].content | replace('go', 'stop') }}",
            "altered the final message",
        ),
    ] {
        let (_directory, tokenizer) = chat_fixture(template);
        let error = tokenizer.render_chat(&messages, &options).unwrap_err();
        assert!(error.to_string().contains(expected), "{template}: {error}");
    }
    let error = tokenizer
        .render_chat(
            &messages,
            &ChatTemplateOptions {
                continue_final_message: true,
                add_generation_prompt: true,
                ..ChatTemplateOptions::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("mutually exclusive"), "{error}");
}

/// A fixed clock makes `strftime_now` reproducible through the public options.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_options_fix_the_template_clock() {
    let (_directory, tokenizer) = chat_fixture("{{ strftime_now('%d %b %Y') }}");
    let options = ChatTemplateOptions {
        current_time_unix_seconds: Some(1_721_999_109),
        ..ChatTemplateOptions::default()
    };
    assert_eq!(
        tokenizer
            .render_chat(&[ChatMessage::text("user", "x")], &options)
            .unwrap(),
        "26 Jul 2024"
    );
    // Options deserialize from partial JSON, defaulting every omitted switch.
    let parsed: ChatTemplateOptions =
        serde_json::from_value(serde_json::json!({"add_generation_prompt": true})).unwrap();
    assert!(parsed.add_generation_prompt && !parsed.continue_final_message);
    assert_eq!(parsed.current_time_unix_seconds, None);
}

/// Builds the tool-use conversation used by the real-template goldens.
///
/// Dictionaries are key-sorted so the goldens hold with or without
/// order-preserving JSON maps.
#[cfg(feature = "chat-templates")]
fn tool_use_conversation() -> (Vec<ChatMessage>, ChatTemplateOptions) {
    let messages = vec![
        ChatMessage::text("user", "Weather in Paris?"),
        ChatMessage::text("assistant", "").with_tool_calls(serde_json::json!([{
            "function": {"arguments": {"city": "Paris"}, "name": "get_weather"},
            "type": "function"
        }])),
        ChatMessage::text("tool", "18C")
            .with_tool_call_id("call_0")
            .with_name("get_weather"),
    ];
    let options = ChatTemplateOptions {
        add_generation_prompt: true,
        tools: Some(serde_json::json!([{
            "function": {
                "description": "Get the weather for a city.",
                "name": "get_weather",
                "parameters": {
                    "properties": {"city": {"description": "City name", "type": "string"}},
                    "required": ["city"],
                    "type": "object"
                }
            },
            "type": "function"
        }])),
        extra_context: BTreeMap::from([
            ("current_date".into(), serde_json::json!("2026-09-05")),
            ("reasoning_strength".into(), serde_json::json!("low")),
        ]),
        ..ChatTemplateOptions::default()
    };
    (messages, options)
}

/// Loads a pinned local tokenizer named by an environment variable.
#[cfg(feature = "chat-templates")]
fn pinned_tokenizer(variable: &str) -> LocalTokenizer {
    let root = std::env::var_os(variable)
        .unwrap_or_else(|| panic!("{variable} must name the pinned local checkpoint"));
    LocalTokenizer::from_directory(root).unwrap()
}

/// Qwen3.5's official tool prompt matches `apply_chat_template` byte for byte.
///
/// Golden from transformers 5.2.0 with the pinned Qwen3.5-0.8B tokenizer
/// (revision 2fc06364715b967f1860aea9cf38778875588b17).
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires TRANSFORMERS_QWEN35_DIR with the pinned Qwen3.5 tokenizer"]
fn qwen35_tool_prompt_matches_hugging_face() {
    let tokenizer = pinned_tokenizer("TRANSFORMERS_QWEN35_DIR");
    let (messages, options) = tool_use_conversation();
    let expected = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"function\": {\"description\": \"Get the weather for a city.\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"description\": \"City name\", \"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}, \"type\": \"function\"}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        expected
    );
}

/// Nemotron-H's official template renders a plain request (its `tools is iterable` guard).
///
/// Golden from transformers 5.2.0 with the pinned Nemotron 3.5 Lightning
/// tokenizer (revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9).
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires TRANSFORMERS_NEMOTRON_H_DIR with the pinned Nemotron-H tokenizer"]
fn nemotron_plain_prompt_matches_hugging_face() {
    let tokenizer = pinned_tokenizer("TRANSFORMERS_NEMOTRON_H_DIR");
    let rendered = tokenizer
        .render_chat(
            &[ChatMessage::text("user", "Hello")],
            &ChatTemplateOptions {
                add_generation_prompt: true,
                ..ChatTemplateOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        rendered,
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

/// Muse Glimmer's official tool prompt (conditional keyword arguments, `dict.get`) matches.
///
/// Golden from transformers 5.2.0 with the pinned Muse Glimmer tokenizer
/// (revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68).
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires TRANSFORMERS_MUSE_GLIMMER_DIR with the pinned Muse Glimmer tokenizer"]
fn muse_glimmer_tool_prompt_matches_hugging_face() {
    let tokenizer = pinned_tokenizer("TRANSFORMERS_MUSE_GLIMMER_DIR");
    let (messages, options) = tool_use_conversation();
    let expected = "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-09-05.\n\nReasoning strength: low.\n\nIn this environment you have access to a set of tools you can use to answer the user's question.\n\nYou can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\nString and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\nHere are the functions available in JSONSchema format:\n// Tool metadata\n{\"name\": \"get_weather\", \"description\": \"\"}\n// Function schemas\n{\"name\": \"get_weather\", \"description\": \"Get the weather for a city.\", \"parameters\": {\"properties\": {\"city\": {\"description\": \"City name\", \"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}\n\nHere's an example of how to call a function in the tool set:\n(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\nto=example_tool_name.example_function_name\n\n<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n\n# Valid recipients: \"self\", \"get_weather.*\", \"user\".<|eot|><|start|>user<|message|>Weather in Paris?<|eot|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|><|start|>tool get_weather<|message|><tool_output name=\"get_weather\">\n18C\n</tool_output><|eot|><|start|>assistant";
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        expected
    );
}

/// Builds a tool call whose arguments and schema hold booleans, null, a float, and a list.
///
/// These values reach official templates through `| string`, which must print
/// Python's `True`/`None`. Dictionaries are key-sorted (see [`tool_use_conversation`]).
#[cfg(feature = "chat-templates")]
fn typed_tool_conversation() -> (Vec<ChatMessage>, ChatTemplateOptions) {
    let arguments = serde_json::json!({
        "count": 3, "flag": true, "items": [1, 2], "name": "x", "none_val": null, "ratio": 1.5
    });
    let messages = vec![
        ChatMessage::text("user", "do it"),
        ChatMessage::text("assistant", "").with_tool_calls(serde_json::json!([{
            "function": {"arguments": arguments, "name": "f"},
            "type": "function"
        }])),
        ChatMessage::text("tool", "ok"),
    ];
    let options = ChatTemplateOptions {
        add_generation_prompt: true,
        tools: Some(serde_json::json!([{
            "function": {
                "description": "d",
                "name": "f",
                "parameters": {
                    "properties": {"flag": {
                        "default": false, "description": "a flag", "nullable": true, "type": "boolean"
                    }},
                    "required": ["flag"],
                    "type": "object"
                }
            },
            "type": "function"
        }])),
        ..ChatTemplateOptions::default()
    };
    (messages, options)
}

/// Qwen3.5 prints typed tool arguments as Python does (`True`, `None`, `1.5`, `[1, 2]`).
///
/// Golden from transformers 5.2.0 `apply_chat_template` with the pinned
/// Qwen3.5-0.8B tokenizer (revision 2fc06364715b967f1860aea9cf38778875588b17).
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires TRANSFORMERS_QWEN35_DIR with the pinned Qwen3.5 tokenizer"]
fn qwen35_typed_tool_arguments_match_hugging_face() {
    let tokenizer = pinned_tokenizer("TRANSFORMERS_QWEN35_DIR");
    let (messages, options) = typed_tool_conversation();
    let expected = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"function\": {\"description\": \"d\", \"name\": \"f\", \"parameters\": {\"properties\": {\"flag\": {\"default\": false, \"description\": \"a flag\", \"nullable\": true, \"type\": \"boolean\"}}, \"required\": [\"flag\"], \"type\": \"object\"}}, \"type\": \"function\"}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\ndo it<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=f>\n<parameter=count>\n3\n</parameter>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=items>\n[1, 2]\n</parameter>\n<parameter=name>\nx\n</parameter>\n<parameter=none_val>\nNone\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        expected
    );
}

/// Nemotron 3.5 prints typed schema fields and arguments as Python does.
///
/// Its `render_extra_keys` macro emits `<default>False</default>` and
/// `<nullable>True</nullable>`. Golden from transformers 5.2.0
/// `apply_chat_template` with the pinned Nemotron 3.5 Lightning tokenizer
/// (revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9).
#[cfg(feature = "chat-templates")]
#[test]
#[ignore = "requires TRANSFORMERS_NEMOTRON_H_DIR with the pinned Nemotron-H tokenizer"]
fn nemotron_typed_tool_schema_and_arguments_match_hugging_face() {
    let tokenizer = pinned_tokenizer("TRANSFORMERS_NEMOTRON_H_DIR");
    let (messages, options) = typed_tool_conversation();
    let expected = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>f</name>\n<description>d</description>\n<parameters>\n<parameter>\n<name>flag</name>\n<type>boolean</type>\n<description>a flag</description>\n<default>False</default>\n<nullable>True</nullable>\n</parameter>\n<required>[\"flag\"]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\ndo it<|im_end|>\n<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=f>\n<parameter=count>\n3\n</parameter>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=items>\n[1, 2]\n</parameter>\n<parameter=name>\nx\n</parameter>\n<parameter=none_val>\nNone\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n<|im_start|>user\n<tool_response>\nok\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        tokenizer.render_chat(&messages, &options).unwrap(),
        expected
    );
}

/// Map key order is Python's insertion order once both crates preserve it, sorted until then.
///
/// With the crate's current `serde_json` and `minijinja` features, objects are
/// sorted maps, which is the documented difference from Hugging Face. This test
/// pins both states: enabling `preserve_order` on both crates (a Cargo.toml
/// change) switches it to the transformers 5.2.0 golden without editing it.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_map_key_order_is_sorted_until_order_is_preserved() {
    let (_directory, tokenizer) = chat_fixture(
        "{{ messages[0] }}|{% for k, v in messages[0].items() %}{{ k }};{% endfor %}|{{ tools[0] | tojson }}",
    );
    // Parsed from text so an order-preserving `serde_json` keeps "type" first.
    let tools: Value = serde_json::from_str(
        r#"[{"type": "function", "function": {"name": "f", "description": "d"}}]"#,
    )
    .unwrap();
    let options = ChatTemplateOptions {
        tools: Some(tools),
        ..ChatTemplateOptions::default()
    };
    let rendered = tokenizer
        .render_chat(&[ChatMessage::text("user", "hi")], &options)
        .unwrap();
    let probe: Value = serde_json::from_str(r#"{"b": 0, "a": 0}"#).unwrap();
    let json_preserves = probe.as_object().unwrap().keys().next().map(String::as_str) == Some("b");
    let template_preserves = minijinja::Value::from_serialize(&probe)
        .try_iter()
        .unwrap()
        .next()
        .and_then(|key| key.as_str().map(str::to_owned))
        .as_deref()
        == Some("b");
    let expected = if json_preserves && template_preserves {
        // Golden from transformers 5.2.0 `render_jinja_template`.
        "{'role': 'user', 'content': 'hi'}|role;content;|{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"d\"}}"
    } else {
        "{'content': 'hi', 'role': 'user'}|content;role;|{\"function\": {\"description\": \"d\", \"name\": \"f\"}, \"type\": \"function\"}"
    };
    assert_eq!(rendered, expected);
}

/// Writes a byte-level BPE tokenizer with one token per byte and no merges.
///
/// Every multi-byte UTF-8 character therefore spans several tokens, the
/// worst case for streamed decoding.
fn byte_level_fixture() -> (TempDir, LocalTokenizer) {
    let directory = TempDir::new().unwrap();
    let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
    alphabet.sort_unstable();
    let vocabulary = alphabet
        .iter()
        .enumerate()
        .map(|(id, character)| (character.to_string(), u32::try_from(id).unwrap()))
        .collect::<tokenizers::models::bpe::Vocab>();
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .save(directory.path().join("tokenizer.json"), false)
        .unwrap();
    let tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    (directory, tokenizer)
}

/// Encodes text without special tokens and returns its single row of IDs.
fn content_ids(tokenizer: &LocalTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(
            text,
            &EncodeOptions {
                add_special_tokens: false,
                ..EncodeOptions::default()
            },
        )
        .unwrap()
        .token_ids()[0]
        .clone()
}

/// Streams IDs through a decoder and collects the emitted pieces.
fn stream_pieces(tokenizer: &LocalTokenizer, ids: &[u32]) -> Vec<String> {
    let mut stream = tokenizer.decode_stream(false);
    ids.iter()
        .filter_map(|id| stream.step(*id).unwrap())
        .collect()
}

/// Multi-byte characters split across tokens are emitted whole, exactly once (TOK-4).
#[test]
fn decode_stream_emits_whole_characters_for_split_bytes() {
    let (_directory, tokenizer) = byte_level_fixture();
    let text = "héllo 中文 😀!";
    let ids = content_ids(&tokenizer, text);
    assert_eq!(ids.len(), text.len(), "one token per UTF-8 byte");
    // Decoding tokens one at a time corrupts every multi-byte character.
    let naive = ids
        .iter()
        .map(|id| tokenizer.decode(&[*id], false).unwrap())
        .collect::<String>();
    assert!(naive.contains('\u{fffd}'), "{naive}");

    let pieces = stream_pieces(&tokenizer, &ids);
    assert_eq!(pieces.concat(), text);
    assert!(pieces.iter().all(|piece| !piece.contains('\u{fffd}')));
    assert!(pieces.iter().any(|piece| piece == "😀"), "{pieces:?}");
    assert!(pieces.iter().any(|piece| piece == "中"), "{pieces:?}");
    assert!(format!("{:?}", tokenizer.decode_stream(true)).starts_with("TokenStreamDecoder"));
}

/// A stream that stops mid-character withholds the incomplete bytes instead of emitting U+FFFD.
#[test]
fn decode_stream_withholds_incomplete_trailing_characters() {
    let (_directory, tokenizer) = byte_level_fixture();
    let ids = content_ids(&tokenizer, "a😀");
    let pieces = stream_pieces(&tokenizer, &ids[..3]);
    assert_eq!(pieces, ["a"]);
    // Unknown IDs decode to nothing, as with `decode`.
    let mut stream = tokenizer.decode_stream(false);
    assert_eq!(stream.step(u32::MAX).unwrap(), None);
}

proptest::proptest! {
    /// Streaming any text token by token reproduces whole-sequence decoding.
    ///
    /// Text containing a literal U+FFFD is excluded: the stream cannot tell it
    /// from an unfinished character and withholds it until later text arrives.
    #[test]
    fn decode_stream_matches_whole_sequence_decoding(
        text in proptest::strategy::Strategy::prop_filter(
            "\\PC{1,24}",
            "a literal U+FFFD is indistinguishable from an unfinished character",
            |text: &String| !text.contains('\u{fffd}'),
        )
    ) {
        let (_directory, tokenizer) = byte_level_fixture();
        let ids = content_ids(&tokenizer, &text);
        let whole = tokenizer.decode(&ids, false).unwrap();
        proptest::prop_assert_eq!(&whole, &text);
        proptest::prop_assert_eq!(stream_pieces(&tokenizer, &ids).concat(), whole);
    }
}

/// Measures streaming against re-decoding the growing prefix at every step.
///
/// Run with `cargo test --release --features tokenizers --lib
/// decode_stream_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly in release mode"]
fn decode_stream_timing() {
    let (_directory, tokenizer) = byte_level_fixture();
    let text = "Streaming 中文 and emoji 😀 text. ".repeat(64);
    let ids = content_ids(&tokenizer, &text);
    let started = std::time::Instant::now();
    let mut emitted = 0;
    for end in 1..=ids.len() {
        emitted = tokenizer.decode(&ids[..end], false).unwrap().len();
    }
    let prefix = started.elapsed();
    let started = std::time::Instant::now();
    let streamed = stream_pieces(&tokenizer, &ids).concat();
    let stream = started.elapsed();
    assert_eq!(streamed.len(), emitted);
    println!(
        "{} tokens: prefix re-decoding {prefix:?}, streaming {stream:?} ({:.1}x)",
        ids.len(),
        prefix.as_secs_f64() / stream.as_secs_f64()
    );
}
