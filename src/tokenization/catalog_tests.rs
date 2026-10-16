//! Chat-template discovery limits, file-name rules, newline translation, and selection precedence.

use tempfile::TempDir;

use super::*;

/// Creates a canonical temporary root holding `files` (relative path, contents).
fn template_root(files: &[(&str, &[u8])]) -> (TempDir, PathBuf) {
    let directory = TempDir::new().expect("temporary template root");
    let root = directory.path().canonicalize().expect("canonical root");
    for (relative, contents) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("files live below the root")).unwrap();
        fs::write(path, contents).unwrap();
    }
    (directory, root)
}

/// Returns the artifact error's file name and reason, failing on any other result.
fn artifact_failure(result: Result<BTreeMap<String, String>>) -> (String, String) {
    match result {
        Err(Error::Artifact { path, reason }) => (
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            reason,
        ),
        other => panic!("expected an artifact error, found {other:?}"),
    }
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

/// An absent or `null` config entry is no template, as Python's `None` is.
#[test]
fn absent_and_null_config_templates_are_empty() {
    assert!(parse_chat_templates(None).unwrap().is_empty());
    assert!(parse_chat_templates(Some(&Value::Null)).unwrap().is_empty());
    let (_directory, root) = template_root(&[]);
    assert!(
        load_chat_templates(&root, Some(&Value::Null))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        load_chat_templates(&root, Some(&Value::String("C".into()))).unwrap(),
        BTreeMap::from([(DEFAULT_TEMPLATE.into(), "C".into())])
    );
}

/// Only case-sensitive `*.jinja` names are templates, hidden ones included.
#[test]
fn template_file_names_follow_the_jinja_glob() {
    for (file_name, expected) in [
        ("tool_use.jinja", TemplateFileName::Template("tool_use")),
        (".hidden.jinja", TemplateFileName::Template(".hidden")),
        (".jinja", TemplateFileName::Template("")),
        ("a.b.jinja", TemplateFileName::Template("a.b")),
        ("tool_use.JINJA", TemplateFileName::Other),
        ("tool_use.jinja.bak", TemplateFileName::Other),
        ("README.md", TemplateFileName::Other),
        ("jinja", TemplateFileName::Other),
    ] {
        assert_eq!(
            classify_file_name(OsStr::new(file_name)),
            expected,
            "{file_name}"
        );
    }
}

/// A `.jinja` name that is not UTF-8 cannot name a template; other such files are ignored.
#[cfg(unix)]
#[test]
fn non_utf8_template_file_names_are_reported() {
    use std::os::unix::ffi::OsStrExt;

    assert_eq!(
        classify_file_name(OsStr::from_bytes(b"\xff.jinja")),
        TemplateFileName::NotUtf8
    );
    assert_eq!(
        classify_file_name(OsStr::from_bytes(b"\xff.txt")),
        TemplateFileName::Other
    );
}

/// A dangling `additional_chat_templates` link holds no templates; a looping one is an error.
///
/// Python's `Path.is_dir()` is false for a dangling link, so Hugging Face
/// reads nothing from it either.
#[cfg(unix)]
#[test]
fn additional_directory_links_resolve_or_fail_with_their_path() {
    use std::os::unix::fs::symlink;

    let (_directory, root) = template_root(&[]);
    let link = root.join(ADDITIONAL_TEMPLATE_DIRECTORY);
    symlink(root.join("missing"), &link).unwrap();
    assert!(
        read_template_files(&root, TemplateFileLimits::CHECKPOINT)
            .unwrap()
            .is_empty()
    );
    fs::remove_file(&link).unwrap();
    symlink(&link, &link).unwrap();
    match read_template_files(&root, TemplateFileLimits::CHECKPOINT) {
        Err(Error::Io { path, source }) => {
            assert!(path.ends_with(ADDITIONAL_TEMPLATE_DIRECTORY));
            assert_ne!(source.kind(), ErrorKind::NotFound);
        }
        other => panic!("expected an I/O error, found {other:?}"),
    }
}

/// A `.jinja` file whose name is not UTF-8 fails with its path, where the filesystem allows one.
///
/// APFS (macOS) refuses such names outright, so there only the refusal is
/// observed; Linux filesystems accept them and the loader must reject them.
#[cfg(unix)]
#[test]
fn non_utf8_template_file_names_fail_with_their_path() {
    use std::os::unix::ffi::OsStrExt;

    let (_directory, root) = template_root(&[("additional_chat_templates/ok.jinja", b"T")]);
    let name = OsStr::from_bytes(b"\xff.jinja");
    if fs::write(root.join(ADDITIONAL_TEMPLATE_DIRECTORY).join(name), b"T").is_err() {
        return;
    }
    let (file, reason) =
        artifact_failure(read_template_files(&root, TemplateFileLimits::CHECKPOINT));
    assert_eq!(file, "\u{fffd}.jinja");
    assert!(reason.contains("must be UTF-8"), "{reason}");
}

/// `\r\n` and a lone `\r` become `\n`, as in Python's text mode and Jinja2's lexer.
///
/// Text without a carriage return is borrowed, not copied.
#[test]
fn universal_newlines_match_python_and_jinja2() {
    for (source, expected) in [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\r\r\nb", "a\n\nb"),
        ("a\n\rb", "a\n\nb"),
        ("\r\n\r\n", "\n\n"),
        ("trailing\r", "trailing\n"),
        ("é\r漢", "é\n漢"),
    ] {
        assert_eq!(universal_newlines(source), expected, "{source:?}");
    }
    for source in ["plain\ntext", "", "no line endings"] {
        assert!(matches!(universal_newlines(source), Cow::Borrowed(text) if text == source));
    }
}

/// Template files are counted and sized against their limits, exactly at each boundary.
#[test]
fn template_file_limits_hold_at_their_boundaries() {
    let limits = TemplateFileLimits {
        file_bytes: 3,
        total_bytes: 6,
        additional_templates: 2,
    };
    let (_directory, root) = template_root(&[
        ("chat_template.jinja", b"abc"),
        ("additional_chat_templates/one.jinja", b"def"),
    ]);
    assert_eq!(
        read_template_files(&root, limits).unwrap(),
        BTreeMap::from([
            (DEFAULT_TEMPLATE.into(), "abc".into()),
            ("one".into(), "def".into()),
        ])
    );
    let tight = TemplateFileLimits {
        total_bytes: 5,
        ..limits
    };
    let (file, reason) = artifact_failure(read_template_files(&root, tight));
    assert_eq!(file, "one.jinja");
    assert!(
        reason.contains("combined safety limit of 5 bytes"),
        "{reason}"
    );

    let (_directory, root) = template_root(&[("additional_chat_templates/one.jinja", b"abcd")]);
    let (file, reason) = artifact_failure(read_template_files(&root, limits));
    assert_eq!(file, "one.jinja");
    assert!(reason.contains("exceeds safety limit 3"), "{reason}");

    let (_directory, root) = template_root(&[
        ("additional_chat_templates/a.jinja", b"a"),
        ("additional_chat_templates/b.jinja", b"b"),
        ("additional_chat_templates/ignored.txt", b"not counted"),
    ]);
    assert_eq!(read_template_files(&root, limits).unwrap().len(), 2);
    fs::write(root.join("additional_chat_templates/c.jinja"), b"c").unwrap();
    let (file, reason) = artifact_failure(read_template_files(&root, limits));
    assert_eq!(file, ADDITIONAL_TEMPLATE_DIRECTORY);
    assert!(reason.contains("more than 2 chat templates"), "{reason}");
}

/// The checkpoint limits keep every published template and bound adversarial sets.
#[test]
fn checkpoint_limits_bound_files_count_and_total() {
    let limits = TemplateFileLimits::CHECKPOINT;
    assert_eq!(limits.file_bytes, MAX_METADATA_BYTES);
    assert_eq!(limits.total_bytes, 4 * MAX_METADATA_BYTES);
    assert_eq!(limits.additional_templates, 64);
    assert!(limits.file_bytes <= limits.total_bytes);
}

/// Errors in template files are reported in sorted name order, whatever the directory order.
#[test]
fn the_first_failing_template_file_is_chosen_by_name() {
    let (_directory, root) = template_root(&[
        ("additional_chat_templates/zeta.jinja", b" "),
        ("additional_chat_templates/alpha.jinja", b" "),
        ("additional_chat_templates/mid.jinja", b"ok"),
    ]);
    let (file, reason) =
        artifact_failure(read_template_files(&root, TemplateFileLimits::CHECKPOINT));
    assert_eq!(file, "alpha.jinja");
    assert!(reason.contains("must not be empty"), "{reason}");
}

/// The template the documented precedence selects, transcribed from Hugging Face 5.2.
///
/// `get_chat_template` for a template dictionary: an explicit name present in
/// the dictionary wins; with no name, `tool_use` when `tools is not None` and
/// present, else `default`, else it raises. The crate's documented
/// deviations replace an absent explicit name (rendered as Jinja source by
/// Hugging Face) with an error and the raise with the sole template, if any.
#[cfg(feature = "chat-templates")]
fn documented_choice<'a>(
    available: &[&'a str],
    name: Option<&'a str>,
    tools_given: bool,
) -> Option<&'a str> {
    let has = |wanted: &str| available.contains(&wanted);
    match name {
        Some(name) => has(name).then_some(name),
        None if tools_given && has("tool_use") => Some("tool_use"),
        None if has("default") => Some("default"),
        None => match available {
            [only] => Some(*only),
            _ => None,
        },
    }
}

/// Selection equals the documented precedence for every template set, name, and tools flag.
///
/// The domain is finite: every subset of `{default, rag, tool_use}` (the
/// three roles the rule distinguishes: the fallback, the tools preference,
/// and any other name), every explicit name including an absent one, and
/// both tools states.
#[cfg(feature = "chat-templates")]
#[test]
fn selection_follows_get_chat_template_exhaustively() {
    let universe = ["default", "rag", "tool_use"];
    for mask in 0..(1_u8 << universe.len()) {
        let available = universe
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, name)| *name)
            .collect::<Vec<_>>();
        let templates = available
            .iter()
            .map(|name| ((*name).to_owned(), format!("source of {name}")))
            .collect::<BTreeMap<_, _>>();
        for name in [
            None,
            Some("default"),
            Some("rag"),
            Some("tool_use"),
            Some("missing"),
        ] {
            for tools_given in [false, true] {
                let selected = select_template(&templates, name, tools_given);
                match documented_choice(&available, name, tools_given) {
                    Some(expected) => {
                        let (chosen, source) = selected.unwrap();
                        assert_eq!(chosen, expected, "{available:?} {name:?} {tools_given}");
                        assert_eq!(source, format!("source of {expected}"));
                    }
                    None => {
                        let error = selected.unwrap_err().to_string();
                        let expected = match (name, available.len()) {
                            (Some(_), _) => "was not found",
                            (None, 0) => "does not provide a chat template",
                            (None, _) => "multiple chat templates",
                        };
                        assert!(error.contains(expected), "{available:?} {name:?}: {error}");
                    }
                }
            }
        }
    }
}

/// Selection errors name the available templates, cutting a long list short.
#[cfg(feature = "chat-templates")]
#[test]
fn selection_errors_list_available_templates() {
    let error = select_template(&BTreeMap::new(), Some("default"), false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("provides no chat templates"), "{error}");
    let many = (0..=LISTED_TEMPLATE_NAMES)
        .map(|index| (format!("t{index}"), "source".to_owned()))
        .collect::<BTreeMap<_, _>>();
    let error = select_template(&many, None, true).unwrap_err().to_string();
    assert!(
        error.contains("`t0`, `t1`")
            && error.ends_with("`t7`, ...) but no `default`; choose one with `template_name`"),
        "{error}"
    );
    let exactly = many
        .into_iter()
        .take(LISTED_TEMPLATE_NAMES)
        .collect::<BTreeMap<_, _>>();
    let error = select_template(&exactly, Some("x"), false)
        .unwrap_err()
        .to_string();
    assert!(error.ends_with("`t6`, `t7`"), "{error}");
}
