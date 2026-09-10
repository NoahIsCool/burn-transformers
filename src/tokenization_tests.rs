use std::{collections::HashMap, fs};

use burn::backend::Flex;
use tempfile::TempDir;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

use super::*;

/// Writes a deterministic local tokenizer without downloading fixtures.
fn fixture(config: Value) -> TempDir {
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

/// Empty host encodings fail before tensor allocation and empty vocabularies fail validation.
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

/// The bounded Python-string-method adapter validates values and argument arity.
#[cfg(feature = "chat-templates")]
#[test]
fn chat_python_string_methods_match_supported_semantics() {
    let directory = fixture(base_config());
    let mut tokenizer = LocalTokenizer::from_directory(directory.path()).unwrap();
    let messages = [ChatMessage::text("user", "hello")];
    for (expression, expected) in [
        ("'Hello'.startswith('He')", "true"),
        ("'Hello'.endswith('lo')", "true"),
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
        "'x'.split()",
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
