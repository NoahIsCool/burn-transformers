//! Deterministic regression seeds for the chat-template fuzz harness.

use super::*;

/// A problem with one user message, the given default template, and pinned time.
fn problem(template: &str) -> Problem {
    Problem {
        storage: Storage::String,
        default_template: template.to_owned(),
        tool_template: "{{ tools | tojson }}".to_owned(),
        messages: vec![ChatMessage::text("user", "hello world")],
        options: ChatTemplateOptions {
            current_time_unix_seconds: Some(0),
            ..ChatTemplateOptions::default()
        },
    }
}

/// Renders `template` and returns the result, after the harness checks pass.
fn render(problem: &Problem) -> transformers::Result<String> {
    check(problem);
    let directory = write_fixture(problem);
    LocalTokenizer::from_directory(directory.path())
        .expect("fixture loads")
        .render_chat(&problem.messages, &problem.options)
}

/// The fixture tokenizer loads, renders a conventional template, and encodes it.
#[test]
fn conventional_template_renders_and_encodes() {
    let rendered = render(&problem(
        "{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    ))
    .unwrap();
    assert_eq!(rendered, "<bos>user: hello world\n");
}

/// Every storage form loads both templates and selects by name.
#[test]
fn every_template_storage_selects_by_name() {
    for storage in [Storage::NamedList, Storage::Map, Storage::TemplateFiles] {
        let mut tool = problem("default");
        tool.storage = storage;
        tool.tool_template = "tool".to_owned();
        tool.options.template_name = Some("tool_use".to_owned());
        assert_eq!(render(&tool).unwrap(), "tool");
    }
    let mut file = problem("from file");
    file.storage = Storage::JinjaFile;
    assert_eq!(render(&file).unwrap(), "from file");
    // Whitespace-only templates are refused when loading.
    check(&problem(" \n\t"));
    let mut blank_file = problem("  ");
    blank_file.storage = Storage::JinjaFile;
    check(&blank_file);
    let mut blank_tool_file = problem("default");
    blank_tool_file.storage = Storage::TemplateFiles;
    blank_tool_file.tool_template = "\r\n".to_owned();
    check(&blank_tool_file);
}

/// Template files replace the config template, and every source's line endings render as `\n`.
///
/// Hugging Face 5.2 `_from_pretrained` lets `chat_template.jinja` and
/// `additional_chat_templates/*.jinja` replace `tokenizer_config.json`'s
/// `chat_template` and reads them with Python's universal newlines (the
/// harness checks the loaded text); Jinja2's lexer then turns `\r\n` and
/// `\r` in any template source, config strings included, into `\n`.
#[test]
fn template_files_replace_the_config_template() {
    let mut files = problem("D\r\n{{ messages | length }}\rE");
    files.storage = Storage::TemplateFiles;
    files.tool_template = "T\r\nU".to_owned();
    assert_eq!(render(&files).unwrap(), "D\n1\nE");
    files.options.tools = Some(json!([]));
    assert_eq!(render(&files).unwrap(), "T\nU");
    let mut config = problem("C\r\nX\r");
    config.storage = Storage::Map;
    config.tool_template = "T\rU".to_owned();
    assert_eq!(render(&config).unwrap(), "C\nX");
    config.options.tools = Some(json!([]));
    assert_eq!(render(&config).unwrap(), "T\nU");
}

/// Each documented argument check fails, whatever the template.
#[test]
fn documented_preconditions_are_rejected() {
    let base = problem("{{ messages | length }}");
    let mut cases = Vec::new();
    let mut empty = problem("x");
    empty.messages.clear();
    cases.push(empty);
    let mut both = problem("x");
    both.options.add_generation_prompt = true;
    both.options.continue_final_message = true;
    cases.push(both);
    let mut missing = problem("x");
    missing.options.template_name = Some("missing".to_owned());
    cases.push(missing);
    let mut shadowed = problem("x");
    shadowed.messages[0] = shadowed.messages[0].clone().with_field("role", "system");
    cases.push(shadowed);
    let mut reserved = problem("x");
    reserved
        .options
        .extra_context
        .insert("messages".to_owned(), json!([]));
    cases.push(reserved);
    let mut documents = problem("x");
    documents.options.documents = Some(json!(["not an object"]));
    cases.push(documents);
    for case in &cases {
        assert!(violates_preconditions(case));
        assert!(render(case).is_err());
    }
    assert!(!violates_preconditions(&base));
    assert_eq!(render(&base).unwrap(), "1");
}

/// Every named message field and every reserved variable is refused; others are not.
///
/// The library's lists are private, so the harness restates them. This test
/// fails if the library stops refusing any restated name, or starts refusing
/// the ordinary keys `thinking`, `custom`, and `date_string`. A name the
/// library adds later is not drawn until the restatement is extended.
#[test]
fn every_shadowed_field_and_reserved_variable_is_rejected() {
    for key in MESSAGE_FIELDS {
        let mut shadowed = problem("x");
        shadowed.messages[0] = shadowed.messages[0].clone().with_field(key, "shadow");
        assert!(violates_preconditions(&shadowed), "{key}");
        assert!(
            render(&shadowed).is_err(),
            "extra message key {key} rendered"
        );
    }
    for key in RESERVED_VARIABLES {
        let mut reserved = problem("x");
        reserved
            .options
            .extra_context
            .insert(key.to_owned(), json!("value"));
        assert!(violates_preconditions(&reserved), "{key}");
        assert!(
            render(&reserved).is_err(),
            "reserved variable {key} rendered"
        );
    }
    let mut accepted = problem("{{ messages[0].thinking }}|{{ custom }}|{{ date_string }}");
    accepted.messages[0] = accepted.messages[0]
        .clone()
        .with_field("thinking", "t")
        .with_field("custom", "c");
    for key in ["custom", "date_string"] {
        accepted
            .options
            .extra_context
            .insert(key.to_owned(), json!(key));
    }
    assert!(!violates_preconditions(&accepted));
    assert_eq!(render(&accepted).unwrap(), "t|custom|date_string");
}

/// Hugging Face picks `tool_use` only for named templates with `tools` set.
#[test]
fn hugging_face_selection_follows_get_chat_template() {
    let mut selection = problem("D");
    selection.tool_template = "T".to_owned();
    for storage in [
        Storage::String,
        Storage::NamedList,
        Storage::Map,
        Storage::JinjaFile,
        Storage::TemplateFiles,
    ] {
        selection.storage = storage;
        selection.options.tools = None;
        assert_eq!(hugging_face_selection(&selection), "default");
        assert_eq!(render(&selection).unwrap(), "D");
        selection.options.tools = Some(json!([]));
        let (expected, text) = if storage.has_tool_template() {
            ("tool_use", "T")
        } else {
            ("default", "D")
        };
        assert_eq!(hugging_face_selection(&selection), expected);
        assert_eq!(render(&selection).unwrap(), text, "{storage:?}");
    }
}

/// Seed-style inputs with distinguishable `D:`/`T:` templates select consistently.
#[test]
fn marked_templates_are_selected_by_name() {
    let mut marked = problem("D:{{ messages | length }}");
    marked.storage = Storage::Map;
    marked.tool_template = "T:{{ tools | length }}".to_owned();
    marked.options.template_name = Some("tool_use".to_owned());
    marked.options.tools = Some(json!([{"name": "f"}]));
    assert_eq!(render(&marked).unwrap(), "T:1");
    marked.options.template_name = Some("default".to_owned());
    assert_eq!(render(&marked).unwrap(), "D:1");
    // Flag bit 7 adds the markers to decoded templates.
    let mut bytes = Bytes::new(&[2, 0x80, 0, 0, 0]);
    let decoded = decode(&mut bytes);
    assert_eq!(decoded.default_template, DEFAULT_MARKER);
    assert!(decoded.tool_template.starts_with(TOOL_MARKER));
}

/// CHAT-4 (fixed): with `tools` and a named `tool_use` template, Hugging Face selects `tool_use`.
///
/// Hugging Face 5.2 `get_chat_template` (`tokenization_utils_base.py`) returns
/// `template_dict["tool_use"]` when no template name is given, `tools is not
/// None`, and the checkpoint stores a `tool_use` template. `select_template`
/// returned `default` instead, so tool-use checkpoints with named templates
/// (Command-R, Hermes-2-Pro) rendered a different prompt. It now selects the
/// explicit name, then `tool_use` with tools, then `default`.
#[test]
fn tools_select_the_tool_use_template() {
    for storage in [Storage::NamedList, Storage::Map, Storage::TemplateFiles] {
        let mut tools = problem("D");
        tools.storage = storage;
        tools.tool_template = "T".to_owned();
        tools.options.tools = Some(json!([]));
        let directory = write_fixture(&tools);
        let rendered = LocalTokenizer::from_directory(directory.path())
            .expect("fixture loads")
            .render_chat(&tools.messages, &tools.options);
        assert_eq!(rendered.unwrap(), "T", "{storage:?}");
    }
}

/// Loaders, unbounded recursion, and fuel exhaustion fail contextually.
#[test]
fn sandbox_limits_fail_contextually() {
    for template in [
        "{% include 'other' %}",
        "{% extends 'base' %}",
        "{% import 'macros' as macros %}",
        "{% macro m(x) %}{{ m(x) }}{% endmacro %}{{ m(1) }}",
        "{% for i in range(1000) %}{% for j in range(1000) %}{{ i }}{% endfor %}{% endfor %}",
        "{{ raise_exception('refused') }}",
        "{{ '%s' % 1 }}",
    ] {
        let error = render(&problem(template)).unwrap_err();
        assert!(
            matches!(error, Error::Tokenizer(_)),
            "{template}: {error:?}"
        );
    }
}

/// `strftime_now` formats every pinned instant, including the `i64` extremes.
#[test]
fn strftime_now_accepts_extreme_instants() {
    for seconds in [i64::MIN, -62_135_596_800, -1, 0, 253_402_300_799, i64::MAX] {
        let mut extreme = problem("{{ strftime_now('%Y-%m-%d %H:%M:%S %a %b %j %p %I') }}");
        extreme.options.current_time_unix_seconds = Some(seconds);
        // Either a formatted date or a contextual error, never a panic.
        let _ = render(&extreme);
    }
}

/// A template that renders nothing encodes to no tokens, which `encode_chat` reports.
///
/// Campaign regression (`crash-86c5a5ad`): the harness first assumed every
/// rendered prompt encodes; an empty prompt is a contextual encoding error.
#[test]
fn empty_prompts_fail_to_encode_contextually() {
    let empty = problem("{# nothing #}");
    assert_eq!(render(&empty).unwrap(), "");
    exercise(&[
        2, 20, 1, 152, 36, 2, 37, 20, 67, 181, 198, 198, 125, 27, 26, 67, 2, 0, 0,
    ]);
}

/// Continuing the final message cuts the prompt right after its text.
#[test]
fn continue_final_message_cuts_after_the_text() {
    let mut continued = problem("{% for m in messages %}[{{ m.content }}]{% endfor %}");
    continued.options.continue_final_message = true;
    assert_eq!(render(&continued).unwrap(), "[hello world");
}

/// Seeds in `fuzz/seeds/chat_templates` keep their documented meaning.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/chat_templates/conventional")[..],
        &include_bytes!("../seeds/chat_templates/tool-template")[..],
        &include_bytes!("../seeds/chat_templates/raw-bytes")[..],
    ] {
        exercise(seed);
    }
    exercise(&[]);
    exercise(&[255; 64]);
}

/// Pseudo-random inputs cover many fragment, message, and option mixes.
#[test]
fn pseudo_random_inputs_render_or_fail_contextually() {
    for input in crate::lcg_inputs(0x6368_6174, 160, 96) {
        exercise(&input);
    }
}

/// Renders `template` directly, without the harness's own resource assertions.
fn render_unchecked(template: &str) -> (transformers::Result<String>, Duration) {
    let problem = problem(template);
    let directory = write_fixture(&problem);
    let tokenizer = LocalTokenizer::from_directory(directory.path()).expect("fixture loads");
    let started = Instant::now();
    let rendered = tokenizer.render_chat(&problem.messages, &problem.options);
    (rendered, started.elapsed())
}

/// CHAT-1 (fixed): string repetition by a huge count panicked inside the renderer.
///
/// MiniJinja evaluated `'ab' * n` as `str::repeat(n)` (folding it while
/// compiling when both operands are constants); a byte length beyond
/// `isize::MAX` panicked with "capacity overflow" through `render_chat`, and a
/// smaller but unsatisfiable one could abort the process. The renderer now
/// rewrites `*` into a bounded filter that checks the product before
/// allocating, and renders under `catch_unwind`, so the result is a
/// contextual `Error::Tokenizer`.
#[test]
fn huge_string_repetition_fails_contextually() {
    for template in [
        "{{ 'ab' * 4611686018427387904 }}",
        "{{ ('ab' * 1000000) * 1000000 }}",
    ] {
        let (rendered, _) = render_unchecked(template);
        assert!(
            matches!(&rendered, Err(Error::Tokenizer(message)) if message.contains("resource limit")),
            "{template}: {rendered:?}"
        );
    }
}

/// CHAT-2 (fixed): one cheap expression rendered an arbitrarily large prompt.
///
/// `{{ 'ab' * 100000000 }}` built a 200 MB prompt in a few hundred
/// milliseconds, and string doubling in a loop grew memory exponentially per
/// unit of fuel. Every `+`, `~`, `*`, and print is now checked against a
/// per-value limit (and a cumulative budget) before it allocates, and the
/// prompt streams into a writer bounded by `1 MiB + 4x` the context size.
#[test]
fn rendered_prompts_are_bounded() {
    for template in [
        "{{ 'ab' * 100000000 }}",
        "{% set ns = namespace(text='ab') %}{% for i in range(40) %}{% set ns.text = ns.text ~ ns.text %}{% endfor %}{{ ns.text }}",
        "{% for i in range(100) %}{{ 'x' * 1000000 }}{% endfor %}",
    ] {
        let (rendered, elapsed) = render_unchecked(template);
        assert!(
            matches!(&rendered, Err(Error::Tokenizer(message)) if message.contains("resource limit")),
            "{template}: {:?}",
            rendered.map(|prompt| prompt.len())
        );
        assert!(elapsed <= RENDER_TIME_LIMIT, "{template}: {elapsed:?}");
    }
}

/// CHAT-3 (fixed): lazily concatenated sequences made one instruction arbitrarily slow.
///
/// `seq + seq` built a lazy chained iterable, so doubling a list 30 times cost
/// about 100 instructions of fuel, but `| length` then walked 2^30 elements
/// through 30 chained iterators (about ten minutes on Apple Silicon). The
/// bounded `+` now materializes a list within the render's item limit (2^20)
/// and cumulative budget, so doubling fails contextually after about twenty
/// steps, well within the time limit.
#[test]
fn concatenated_sequences_finish_quickly() {
    let (rendered, elapsed) = render_unchecked(
        "{% set ns = namespace(l=[1]) %}{% for i in range(24) %}{% set ns.l = ns.l + ns.l %}{% endfor %}{{ ns.l | length }}",
    );
    assert!(elapsed <= RENDER_TIME_LIMIT, "rendering took {elapsed:?}");
    assert!(
        matches!(&rendered, Err(Error::Tokenizer(message)) if message.contains("resource limit")),
        "{rendered:?}"
    );
}
