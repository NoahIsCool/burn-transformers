//! Untrusted chat templates and messages rendered through [`LocalTokenizer::render_chat`].
//!
//! Chat templates ship inside checkpoints and are therefore untrusted
//! programs. The renderer promises a bounded sandbox: no loader (so
//! `include`/`extends`/`import` fail), a fuel budget, a recursion limit, and a
//! bounded printing depth. For every input the harness writes a tiny word-level
//! tokenizer plus a template assembled from Jinja fragments and raw input bytes,
//! and checks:
//!
//! 1. **Loading.** A whitespace-only template is rejected at load time; any other
//!    template loads, embedded in `tokenizer_config.json` as a string, a named
//!    list, or a map, or as a separate `chat_template.jinja` file.
//! 2. **Rendering never panics** and fails only with [`Error::Tokenizer`]. The
//!    documented argument checks fail exactly when expected: no messages, both
//!    `continue_final_message` and `add_generation_prompt`, an unknown template
//!    name, a non-list or non-object `documents`, a reserved `extra_context`
//!    name, and an extra message key that shadows a named field.
//! 3. **Determinism.** With `strftime_now` pinned (`current_time_unix_seconds`,
//!    including `i64::MIN` and `i64::MAX`), rendering twice gives the same result.
//! 4. **Tokenization.** [`LocalTokenizer::encode_chat`] equals encoding the
//!    rendered prompt without added special tokens (both fail for a prompt that
//!    encodes to no tokens) and fails whenever rendering fails.
//! 5. **Resources.** Rendering finishes within [`RENDER_TIME_LIMIT`] and its
//!    output stays below [`OUTPUT_BYTE_LIMIT`]; fuel bounds instructions, but
//!    nothing else bounds the work or memory one instruction performs.
//!
//! Hangs and memory blow-ups beyond these limits surface as libFuzzer timeouts
//! or RSS-limit failures.

use std::{
    collections::BTreeMap,
    fs,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use transformers::{ChatMessage, ChatTemplateOptions, EncodeOptions, Error, LocalTokenizer};

use crate::Bytes;

/// Upper bound on one render's wall time, far above the fuel budget's cost.
pub const RENDER_TIME_LIMIT: Duration = Duration::from_secs(2);
/// Upper bound on one rendered prompt's size.
///
/// The renderer documents no output bound; the harness asserts one far above
/// any template from a real checkpoint so that amplification (string doubling,
/// repetition, padding methods) is reported as a finding.
pub const OUTPUT_BYTE_LIMIT: usize = 64 * 1024 * 1024;

/// Word-level fixture tokenizer: `<unk>`=0, `<pad>`=1, `<bos>`=2, `<eos>`=3, `hello`=4, `world`=5.
///
/// Whitespace pre-tokenization plus an unknown token make every string encodable.
const TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"<pad>":1,"<bos>":2,"<eos>":3,"hello":4,"world":5},"unk_token":"<unk>"}}"#;

/// Jinja fragments the template generator concatenates.
///
/// They cover the Hugging Face idioms the renderer rewrites or emulates
/// (`generation` blocks, conditional keyword arguments, Python string and dict
/// methods, `tojson`, `raise_exception`, `strftime_now`, loop controls) and the
/// features the sandbox must refuse or bound (loaders, deep recursion, large
/// loops, string growth). Growth fragments use small counts on purpose: the
/// open findings CHAT-1..3 (see the sibling tests) make large counts crash or
/// hang, and a table entry that does so would stop every campaign at once;
/// raw-byte fragments can still reach them.
const FRAGMENTS: &[&str] = &[
    "{{ ",
    " }}",
    "{% ",
    " %}",
    "{# ",
    " #}",
    "{%- ",
    " -%}",
    "\n",
    "{% for message in messages %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}<assistant>{% endif %}",
    "{% if message['role'] == 'user' %}",
    "{% elif message.role == 'assistant' %}",
    "{% else %}",
    "{% endif %}",
    "{{ message['role'] }}",
    "{{ message.content }}",
    "{{ message.content | trim }}",
    "{{ message.content is string }}",
    "{{ message.tool_calls | tojson }}",
    "{{ message.reasoning_content }}",
    "{% generation %}{{ message.content }}{% endgeneration %}",
    "{{ bos_token }}",
    "{{ eos_token }}",
    "{{ raise_exception('template refused') }}",
    "{{ strftime_now('%Y-%m-%d %H:%M:%S %A %B %j') }}",
    "{{ tools | tojson }}",
    "{{ messages | tojson(indent=2) }}",
    "{{ documents }}",
    "{% set ns = namespace(text='a', count=0) %}",
    "{% set ns.text = ns.text ~ ns.text %}",
    "{% set ns.count = ns.count + 1 %}",
    "{{ ns.text | length }}",
    "{% raw %}{{ not parsed }}{% endraw %}",
    "{{ 'a b  c'.split() }}",
    "{{ ' x '.strip() }}",
    "{{ 'abc'.startswith(('a', 'z')) }}",
    "{{ 'a,b'.split(',', 1) }}",
    "{{ 'line\\nnext'.splitlines() }}",
    "{{ 'Mixed'.swapcase() }}",
    "{{ 'abcabc'.replace('b', 'bb') }}",
    "{{ 'abcabc'.count('b') }}",
    "{{ 'abc'.find('c') }}",
    "{{ '12'.isdigit() }}",
    "{{ {'k': [1, 2.5, none, true]} }}",
    "{{ {'k': 1}.items() | list }}",
    "{{ {'k': 1}.get('missing', 0) }}",
    "{{ 1.25 | round(1, 'ceil') }}",
    "{{ 2.5 | round }}",
    "{{ [1, 2] | join(', ') }}",
    "{{ none | string }}",
    "{{ true ~ 'x' }}",
    "{{ '%s' % 1 }}",
    "{{ 'ab' * 3 }}",
    "{{ range(5) | list }}",
    "{% for i in range(100) %}{% for j in range(100) %}{{ i * j }}{% endfor %}{% endfor %}",
    "{% for i in range(8) %}{% set ns.text = ns.text ~ ns.text %}{% endfor %}",
    "{% macro m(x) %}{{ m(x) }}{% endmacro %}{{ m(1) }}",
    "{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(key=1 if true else 2) }}",
    "{{ f(key=a if c else b) }}",
    "{% for message in messages %}{% if loop.first %}{% continue %}{% endif %}{{ loop.index }}{% break %}{% endfor %}",
    "{% include 'other' %}",
    "{% extends 'base' %}",
    "{% import 'macros' as macros %}",
    "{{ x is iterable }}",
    "{{ messages is sequence }}",
    "{{ enable_thinking is defined }}",
    "{{ add_vision_id }}",
    "{{ date_string }}",
    "{{ undefined_name.attribute }}",
    "{{ messages[0]['content'][0]['text'] }}",
    "{{ (messages | length) - 1 }}",
    "{{ messages[-1].content }}",
    "{{ 1 / 0 }}",
    "{{ 'x' | tojson(indent='bad') }}",
    "CONTINUE_FINAL_MESSAGE_TAG ",
    "{",
    "}",
    "%",
    "'",
    "\"",
    "é漢🙂",
];

/// Message text fragments; some look like Jinja or contain the continuation marker.
const TEXTS: &[&str] = &[
    "hello world",
    "",
    "  padded  ",
    "{{ not executed }}",
    "{% if %}",
    "CONTINUE_FINAL_MESSAGE_TAG",
    "line\nbreak",
    "é漢🙂",
    "a'b\"c\\d",
    "\u{0}\u{7f}",
];

/// Decodes `count` fragments into a template, mixing table entries with raw input text.
fn template(bytes: &mut Bytes<'_>, count: usize) -> String {
    let mut text = String::new();
    for _ in 0..count {
        let selector = bytes.byte();
        if selector < 224 {
            text.push_str(FRAGMENTS[usize::from(selector) % FRAGMENTS.len()]);
        } else {
            let length = usize::from(selector - 224) + 1;
            text.push_str(&String::from_utf8_lossy(bytes.take(length)));
        }
    }
    text
}

/// Decodes message content: text, a block list, or a non-text JSON value.
fn content(bytes: &mut Bytes<'_>) -> Value {
    match bytes.below(5) {
        0 | 1 => Value::String(bytes.pick(TEXTS).to_owned()),
        2 => json!([
            {"type": "image"},
            {"type": "text", "text": bytes.pick(TEXTS)},
        ]),
        3 => json!([{"type": "image"}]),
        _ => bytes.pick(&[
            json!(null),
            json!(7),
            json!({"nested": [1, {"deep": true}]}),
        ]),
    }
}

/// Decodes one message with optional named fields and extra keys.
fn message(bytes: &mut Bytes<'_>) -> ChatMessage {
    let role = bytes.pick(&["system", "user", "assistant", "tool", "custom role"]);
    let mut message = ChatMessage::new(role, content(bytes));
    let fields = bytes.byte();
    if fields & 1 != 0 {
        message = message.with_reasoning_content(bytes.pick(TEXTS));
    }
    if fields & 2 != 0 {
        message = message.with_tool_calls(json!([
            {"type": "function", "function": {"name": "f", "arguments": {"b": 1, "a": [true, null]}}}
        ]));
    }
    if fields & 4 != 0 {
        message = message.with_name(bytes.pick(TEXTS));
    }
    if fields & 8 != 0 {
        message = message.with_tool_call_id("call-1");
    }
    if fields & 16 != 0 {
        // `role` shadows a named field and must be rejected; `custom` must not.
        let key = bytes.pick(&["custom", "role", "content", "thinking"]);
        message = message.with_field(key, content(bytes));
    }
    message
}

/// How the fixture stores its templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    /// `"chat_template": "<source>"`.
    String,
    /// `"chat_template": [{"name": "default"|"tool_use", "template": ...}, ...]`.
    NamedList,
    /// `"chat_template": {"default": ..., "tool_use": ...}`.
    Map,
    /// A separate `chat_template.jinja` file (the `default` template).
    JinjaFile,
}

/// One decoded rendering problem.
struct Problem {
    /// How templates are stored.
    storage: Storage,
    /// The `default` template source.
    default_template: String,
    /// A second, `tool_use` template (list and map storage only).
    tool_template: String,
    /// Messages to render.
    messages: Vec<ChatMessage>,
    /// Rendering options; the time is always pinned.
    options: ChatTemplateOptions,
}

/// Decodes a problem; see [`exercise`] for the byte layout.
fn decode(bytes: &mut Bytes<'_>) -> Problem {
    let storage = bytes.pick(&[
        Storage::String,
        Storage::NamedList,
        Storage::Map,
        Storage::JinjaFile,
    ]);
    let flags = bytes.byte();
    let default_count = bytes.below(24);
    let default_template = template(bytes, default_count);
    let tool_count = bytes.below(8) + 1;
    let tool_template = template(bytes, tool_count);
    let messages = (0..bytes.below(4)).map(|_| message(bytes)).collect();
    let mut extra_context = BTreeMap::new();
    if flags & 64 != 0 {
        let key = bytes.pick(&["date_string", "custom", "messages", "strftime_now"]);
        extra_context.insert(key.to_owned(), json!("26 Jul 2024"));
    }
    let options = ChatTemplateOptions {
        template_name: match bytes.below(5) {
            0 => Some("default".to_owned()),
            1 => Some("tool_use".to_owned()),
            2 => Some("missing".to_owned()),
            _ => None,
        },
        add_generation_prompt: flags & 1 != 0,
        continue_final_message: flags & 2 != 0,
        tools: (flags & 4 != 0).then(|| {
            json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])
        }),
        documents: (flags & 8 != 0).then(|| {
            bytes.pick(&[
                json!([{"title": "t", "text": "body"}]),
                json!(["not an object"]),
                json!({"title": "t"}),
            ])
        }),
        add_vision_id: flags & 16 != 0,
        enable_thinking: (flags & 32 != 0).then(|| bytes.flag()),
        current_time_unix_seconds: Some(bytes.pick(&[
            0,
            -1,
            1_721_952_000,
            -62_135_596_800,
            253_402_300_799,
            i64::MIN,
            i64::MAX,
        ])),
        extra_context,
    };
    Problem {
        storage,
        default_template,
        tool_template,
        messages,
        options,
    }
}

/// Writes the fixture tokenizer and templates; returns the directory.
fn write_fixture(problem: &Problem) -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("create isolated tokenizer directory");
    fs::write(directory.path().join("tokenizer.json"), TOKENIZER_JSON).expect("write tokenizer");
    let mut config = json!({
        "unk_token": "<unk>",
        "pad_token": {"content": "<pad>"},
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    });
    let templates = match problem.storage {
        Storage::String => Some(json!(problem.default_template)),
        Storage::NamedList => Some(json!([
            {"name": "default", "template": problem.default_template},
            {"name": "tool_use", "template": problem.tool_template},
        ])),
        Storage::Map => Some(json!({
            "default": problem.default_template,
            "tool_use": problem.tool_template,
        })),
        Storage::JinjaFile => {
            fs::write(
                directory.path().join("chat_template.jinja"),
                &problem.default_template,
            )
            .expect("write template file");
            None
        }
    };
    if let Some(templates) = templates {
        config["chat_template"] = templates;
    }
    fs::write(
        directory.path().join("tokenizer_config.json"),
        serde_json::to_vec(&config).expect("serialize tokenizer config"),
    )
    .expect("write tokenizer config");
    directory
}

/// Whether the arguments violate a documented precondition of `render_chat`.
///
/// Such calls must fail regardless of the template; a valid call may still
/// fail because the template itself errs.
fn violates_preconditions(problem: &Problem) -> bool {
    let options = &problem.options;
    let reserved = options
        .extra_context
        .keys()
        .any(|key| key == "messages" || key == "strftime_now");
    let shadowed = problem.messages.iter().any(|message| {
        message
            .extra
            .keys()
            .any(|key| key == "role" || key == "content")
    });
    let has_tool_template = matches!(problem.storage, Storage::NamedList | Storage::Map);
    let missing_template = match options.template_name.as_deref() {
        Some("default") | None => false,
        Some("tool_use") => !has_tool_template,
        Some(_) => true,
    };
    let bad_documents = options.documents.as_ref().is_some_and(|documents| {
        !documents
            .as_array()
            .is_some_and(|documents| documents.iter().all(Value::is_object))
    });
    problem.messages.is_empty()
        || (options.continue_final_message && options.add_generation_prompt)
        || missing_template
        || shadowed
        || reserved
        || bad_documents
}

/// Whether the template selected by the options is empty after trimming.
fn whitespace_only(problem: &Problem) -> bool {
    let tool = matches!(problem.storage, Storage::NamedList | Storage::Map)
        && problem.tool_template.trim().is_empty();
    problem.default_template.trim().is_empty() || tool
}

/// Renders one problem and checks every documented property.
fn check(problem: &Problem) {
    let directory = write_fixture(problem);
    let loaded = LocalTokenizer::from_directory(directory.path());
    if whitespace_only(problem) {
        assert!(loaded.is_err(), "a whitespace-only template must not load");
        return;
    }
    let tokenizer = loaded.expect("non-empty templates load");
    let started = Instant::now();
    let rendered = tokenizer.render_chat(&problem.messages, &problem.options);
    let elapsed = started.elapsed();
    assert!(
        elapsed <= RENDER_TIME_LIMIT,
        "rendering took {elapsed:?}: {:?}",
        problem.default_template
    );
    if violates_preconditions(problem) {
        assert!(rendered.is_err(), "precondition violation was rendered");
    }
    match &rendered {
        Ok(prompt) => assert!(prompt.len() <= OUTPUT_BYTE_LIMIT),
        Err(error) => assert!(matches!(error, Error::Tokenizer(_)), "{error:?}"),
    }
    let again = tokenizer.render_chat(&problem.messages, &problem.options);
    match (&rendered, &again) {
        (Ok(first), Ok(second)) => assert_eq!(first, second),
        (Err(first), Err(second)) => assert_eq!(first.to_string(), second.to_string()),
        _ => panic!("rendering is not deterministic"),
    }
    let encoded = tokenizer.encode_chat(
        &problem.messages,
        &problem.options,
        &EncodeOptions::default(),
    );
    match rendered {
        Ok(prompt) => {
            // `encode_chat` is rendering plus encoding without added special
            // tokens; a prompt that encodes to no tokens fails in both.
            let direct = tokenizer.encode(
                &prompt,
                &EncodeOptions {
                    add_special_tokens: false,
                    ..EncodeOptions::default()
                },
            );
            match (encoded, direct) {
                (Ok(encoded), Ok(direct)) => assert_eq!(encoded, direct),
                (Err(encoded), Err(direct)) => assert_eq!(encoded.to_string(), direct.to_string()),
                (encoded, direct) => panic!("encode_chat {encoded:?} but encode {direct:?}"),
            }
        }
        Err(_) => assert!(encoded.is_err()),
    }
}

/// Decodes and renders one chat-template problem.
///
/// Byte layout: template storage, a flag byte (bit 0 generation prompt, 1
/// continue final message, 2 tools, 3 documents, 4 vision IDs, 5 thinking
/// switch, 6 extra context), the default template's fragment count (0..=23)
/// and fragments, the tool template's count (1..=8) and fragments, up to three
/// messages, then the template name, documents, thinking value, and pinned time.
/// A fragment selector of at least 224 copies the next `selector - 223` raw
/// bytes (lossy UTF-8) into the template instead of a table entry.
pub fn exercise(input: &[u8]) {
    let mut bytes = Bytes::new(input);
    let problem = decode(&mut bytes);
    check(&problem);
}

#[cfg(test)]
#[path = "chat_templates_tests.rs"]
mod tests;
