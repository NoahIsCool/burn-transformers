//! Chat-template discovery and per-render selection, in Hugging Face 5.2's precedence.
//!
//! A checkpoint can ship several named chat templates, for example a plain
//! `default` and a `tool_use` variant (Hermes-2-Pro, Command-R). This module
//! decides which templates a checkpoint has ([`load_chat_templates`]) and which
//! one a render uses ([`select_template`]), exactly as `transformers` 5.2 does
//! (`PreTrainedTokenizerBase._from_pretrained` and `get_chat_template` in
//! `tokenization_utils_base.py`).
//!
//! # Discovery
//!
//! 1. **Template files win.** `chat_template.jinja` is the template named
//!    `default`, and each `additional_chat_templates/<name>.jinja` is the
//!    template `<name>`; Hugging Face 5.x writes non-default templates there
//!    when it saves a tokenizer. When any of these files exists, they are the
//!    checkpoint's whole template set: a `chat_template` entry in
//!    `tokenizer_config.json` is ignored, even a malformed one.
//! 2. **Otherwise `tokenizer_config.json`.** Its `chat_template` is a string
//!    (the `default` template), a list of `{"name": ..., "template": ...}`
//!    objects, or a name-to-source map; `null` means no template.
//!
//! Template files are read the way Python's text-mode `open` reads them:
//! `\r\n` and a lone `\r` become `\n` (universal newlines), so
//! [`TokenizerMetadata::chat_templates`](super::TokenizerMetadata::chat_templates)
//! holds the text Hugging Face's `tokenizer.chat_template` holds. Strings
//! inside `tokenizer_config.json` are JSON and keep their carriage returns,
//! as in Python.
//!
//! Template files are untrusted checkpoint data. Each must be a regular,
//! UTF-8 file inside the canonical tokenizer root (symlinks are followed but
//! may not escape it), and together they must fit
//! [`TemplateFileLimits::CHECKPOINT`]. A dangling `chat_template.jinja`
//! symlink counts as absent, like every optional artifact, while a dangling
//! `additional_chat_templates/*.jinja` entry is an error, since Python lists
//! it and then fails to open it. The directory listing is sorted so that the
//! first error reported does not depend on the filesystem's order.
//!
//! # Selection
//!
//! Without an explicit name, a render uses `tool_use` when the caller passes
//! tools and the checkpoint has that template, otherwise `default`, otherwise
//! the checkpoint's only template.
//!
//! # Deliberate differences from Hugging Face
//!
//! They matter only for artifacts and calls that Hugging Face rejects or
//! resolves arbitrarily; the public list is in the
//! [`tokenization`](super) module documentation.
//!
//! - An empty or whitespace-only template, an empty template name, and a
//!   duplicate name (in a config list, or `additional_chat_templates/default.jinja`
//!   beside `chat_template.jinja`) are rejected; Hugging Face keeps the last
//!   duplicate.
//! - A name only selects a stored template: an unknown name is an error, and
//!   a template stored without a name is `default`. Hugging Face looks names
//!   up only among several named templates and otherwise renders its
//!   `chat_template` argument as Jinja source.
//! - A checkpoint whose only template is not `default` renders it without a
//!   name; Hugging Face raises "multiple chat templates with no default".

use std::{
    borrow::Cow,
    collections::BTreeMap,
    ffi::OsStr,
    fs,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
};

use serde_json::Value;

use super::{MAX_METADATA_BYTES, contained_file, is_absent};
use crate::{Error, Result};

/// Name of the template a scalar config entry or `chat_template.jinja` defines.
const DEFAULT_TEMPLATE: &str = "default";
/// Name of the template an unnamed render prefers when the caller passes tools.
#[cfg(feature = "chat-templates")]
const TOOL_USE_TEMPLATE: &str = "tool_use";
/// File holding the `default` template (Hugging Face's `CHAT_TEMPLATE_FILE`).
const DEFAULT_TEMPLATE_FILE: &str = "chat_template.jinja";
/// Directory of named templates (Hugging Face's `CHAT_TEMPLATE_DIR`).
const ADDITIONAL_TEMPLATE_DIRECTORY: &str = "additional_chat_templates";
/// Extension that marks a template file; other files in the directory are ignored.
const TEMPLATE_FILE_EXTENSION: &str = ".jinja";
/// Template names quoted in an error message before it is cut short.
#[cfg(feature = "chat-templates")]
const LISTED_TEMPLATE_NAMES: usize = 8;

/// Bounds on the chat-template files that one checkpoint may supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TemplateFileLimits {
    /// Largest accepted template file, in bytes.
    file_bytes: u64,
    /// Largest accepted combined size of every template file, in bytes.
    total_bytes: u64,
    /// Most templates accepted from `additional_chat_templates/`.
    additional_templates: usize,
}

impl TemplateFileLimits {
    /// Limits for every loaded checkpoint: 16 MiB per file, 64 MiB in total,
    /// and 64 additional templates.
    ///
    /// Published templates are tens of kilobytes and checkpoints ship at most
    /// a handful, so only adversarial artifacts reach these bounds.
    const CHECKPOINT: Self = Self {
        file_bytes: MAX_METADATA_BYTES,
        total_bytes: 4 * MAX_METADATA_BYTES,
        additional_templates: 64,
    };
}

/// Loads a checkpoint's named chat templates in Hugging Face's precedence.
///
/// `configured` is the `chat_template` entry of `tokenizer_config.json`, if
/// any; it is used only when the root holds no template files (see the
/// [module documentation](self)).
pub(super) fn load_chat_templates(
    root: &Path,
    configured: Option<&Value>,
) -> Result<BTreeMap<String, String>> {
    let files = read_template_files(root, TemplateFileLimits::CHECKPOINT)?;
    if files.is_empty() {
        parse_chat_templates(configured)
    } else {
        Ok(files)
    }
}

/// Reads `chat_template.jinja` as `default` and each additional `<name>.jinja` as `<name>`.
///
/// Returns an empty map when neither source exists.
fn read_template_files(
    root: &Path,
    limits: TemplateFileLimits,
) -> Result<BTreeMap<String, String>> {
    let mut templates = BTreeMap::new();
    let mut remaining_bytes = limits.total_bytes;
    if let Some(path) = contained_file(root, DEFAULT_TEMPLATE_FILE, false, limits.file_bytes)? {
        let source = read_template_file(&path, &mut remaining_bytes, limits.total_bytes)?;
        insert_template_file(&mut templates, DEFAULT_TEMPLATE, path, source)?;
    }
    for name in additional_template_names(root, limits.additional_templates)? {
        let relative = Path::new(ADDITIONAL_TEMPLATE_DIRECTORY)
            .join(format!("{name}{TEMPLATE_FILE_EXTENSION}"));
        // `contained_file` reports an absent optional file as `None`; a name
        // just listed is absent only when it is a dangling symlink (or was
        // removed meanwhile), which Python's `open` would also fail on.
        let Some(path) = contained_file(root, &relative, false, limits.file_bytes)? else {
            return Err(Error::Artifact {
                path: root.join(relative),
                reason: "chat template file is a dangling symlink or was removed while loading"
                    .into(),
            });
        };
        let source = read_template_file(&path, &mut remaining_bytes, limits.total_bytes)?;
        insert_template_file(&mut templates, &name, path, source)?;
    }
    Ok(templates)
}

/// Lists the template names in `additional_chat_templates/`, sorted.
///
/// An absent directory has no templates. The directory must resolve inside
/// `root`, and at most `maximum` of its entries may be templates.
fn additional_template_names(root: &Path, maximum: usize) -> Result<Vec<String>> {
    let candidate = root.join(ADDITIONAL_TEMPLATE_DIRECTORY);
    if is_absent(&candidate) {
        return Ok(Vec::new());
    }
    let directory = match candidate.canonicalize() {
        Ok(path) => path,
        Err(source) if source.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(Error::io(candidate, source)),
    };
    if !directory.starts_with(root) {
        return Err(Error::Artifact {
            path: candidate,
            reason: "artifact symlink escapes the canonical tokenizer root".into(),
        });
    }
    if !directory.is_dir() {
        return Err(Error::Artifact {
            path: directory,
            reason: "additional chat templates must be a directory of `.jinja` files".into(),
        });
    }
    let entries = fs::read_dir(&directory).map_err(|source| Error::io(&directory, source))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| Error::io(&directory, source))?;
        let file_name = entry.file_name();
        let name = match classify_file_name(&file_name) {
            TemplateFileName::Template(name) => name,
            TemplateFileName::Other => continue,
            TemplateFileName::NotUtf8 => {
                return Err(Error::Artifact {
                    path: entry.path(),
                    reason: "chat template file names must be UTF-8".into(),
                });
            }
        };
        if names.len() == maximum {
            return Err(Error::Artifact {
                path: directory,
                reason: format!("holds more than {maximum} chat templates"),
            });
        }
        names.push(name.to_owned());
    }
    names.sort_unstable();
    Ok(names)
}

/// What one entry of `additional_chat_templates/` contributes.
#[derive(Debug, PartialEq, Eq)]
enum TemplateFileName<'a> {
    /// `<name>.jinja` defines the template `<name>` (possibly empty, which is rejected later).
    Template(
        /// The file name without its `.jinja` extension.
        &'a str,
    ),
    /// Any other file, which Hugging Face's `*.jinja` glob also skips.
    Other,
    /// A `.jinja` file whose name is not UTF-8, so no template name can spell it.
    NotUtf8,
}

/// Classifies a file name the way Hugging Face's case-sensitive `*.jinja` glob does.
///
/// Hidden files match too (Python's `pathlib` globs include them).
fn classify_file_name(file_name: &OsStr) -> TemplateFileName<'_> {
    match file_name.to_str() {
        Some(name) => name
            .strip_suffix(TEMPLATE_FILE_EXTENSION)
            .map_or(TemplateFileName::Other, TemplateFileName::Template),
        None if file_name
            .as_encoded_bytes()
            .ends_with(TEMPLATE_FILE_EXTENSION.as_bytes()) =>
        {
            TemplateFileName::NotUtf8
        }
        None => TemplateFileName::Other,
    }
}

/// Reads one template file within the remaining combined byte budget.
///
/// The read itself is bounded, so a file that grows after its size check
/// cannot exceed the budget. Returns the UTF-8 text with Python's universal
/// newlines applied.
fn read_template_file(path: &Path, remaining_bytes: &mut u64, total_bytes: u64) -> Result<String> {
    let file = fs::File::open(path).map_err(|source| Error::io(path, source))?;
    let mut bytes = Vec::new();
    file.take(remaining_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| Error::io(path, source))?;
    let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    *remaining_bytes = remaining_bytes
        .checked_sub(size)
        .ok_or_else(|| Error::Artifact {
            path: path.to_owned(),
            reason: format!(
                "chat template files exceed the combined safety limit of {total_bytes} bytes"
            ),
        })?;
    let source = String::from_utf8(bytes).map_err(|_| Error::Artifact {
        path: path.to_owned(),
        reason: "chat template must be UTF-8 text".into(),
    })?;
    Ok(match universal_newlines(&source) {
        Cow::Borrowed(_) => source,
        Cow::Owned(normalized) => normalized,
    })
}

/// Turns every `\r\n` and lone `\r` into `\n`, borrowing text that has neither.
///
/// This is Python's universal-newline translation, which its text-mode
/// `open` applies to the template files Hugging Face reads.
fn universal_newlines(source: &str) -> Cow<'_, str> {
    if source.contains('\r') {
        Cow::Owned(source.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(source)
    }
}

/// Inserts one template read from `path`, naming the file in every error.
fn insert_template_file(
    templates: &mut BTreeMap<String, String>,
    name: &str,
    path: PathBuf,
    source: String,
) -> Result<()> {
    let reason = if name.trim().is_empty() {
        "chat template file needs a name before `.jinja`".to_owned()
    } else if source.trim().is_empty() {
        "chat template must not be empty".to_owned()
    } else if templates.contains_key(name) {
        format!("duplicate chat template `{name}`: `{DEFAULT_TEMPLATE_FILE}` already defines it")
    } else {
        templates.insert(name.to_owned(), source);
        return Ok(());
    };
    Err(Error::Artifact { path, reason })
}

/// Accepts the scalar, list, and map `chat_template` encodings of `tokenizer_config.json`.
///
/// An absent or `null` entry means the checkpoint has no configured template.
pub(super) fn parse_chat_templates(value: Option<&Value>) -> Result<BTreeMap<String, String>> {
    let mut templates = BTreeMap::new();
    match value {
        None | Some(Value::Null) => {}
        Some(Value::String(template)) => {
            insert_template(&mut templates, DEFAULT_TEMPLATE, template)?
        }
        Some(Value::Array(entries)) => {
            for (index, entry) in entries.iter().enumerate() {
                let name = entry.get("name").and_then(Value::as_str).ok_or_else(|| {
                    Error::Tokenizer(format!("chat_template[{index}].name must be a string"))
                })?;
                let template = entry
                    .get("template")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        Error::Tokenizer(format!(
                            "chat_template[{index}].template must be a string"
                        ))
                    })?;
                insert_template(&mut templates, name, template)?;
            }
        }
        Some(Value::Object(entries)) => {
            for (name, template) in entries {
                let template = template.as_str().ok_or_else(|| {
                    Error::Tokenizer(format!("chat template `{name}` must be a string"))
                })?;
                insert_template(&mut templates, name, template)?;
            }
        }
        Some(_) => {
            return Err(Error::Tokenizer(
                "chat_template must be a string, named array, or string map".into(),
            ));
        }
    }
    Ok(templates)
}

/// Inserts one named, non-empty chat template without silent replacement.
fn insert_template(
    templates: &mut BTreeMap<String, String>,
    name: &str,
    template: &str,
) -> Result<()> {
    if name.trim().is_empty() || template.trim().is_empty() {
        return Err(Error::Tokenizer(
            "chat template name and source must not be empty".into(),
        ));
    }
    if templates.insert(name.into(), template.into()).is_some() {
        return Err(Error::Tokenizer(format!(
            "duplicate chat template name `{name}`"
        )));
    }
    Ok(())
}

/// Selects the template one render uses, as Hugging Face's `get_chat_template` does.
///
/// The precedence is: the explicit `name`; then `tool_use` when `tools_given`
/// (Python's `tools is not None`) and the checkpoint has it; then `default`;
/// then the checkpoint's only template. Returns the selected `(name, source)`.
///
/// # Errors
///
/// An explicit name the checkpoint lacks, a checkpoint without templates, and
/// several templates of which none applies are errors that list the
/// available names.
#[cfg(feature = "chat-templates")]
pub(super) fn select_template<'a>(
    templates: &'a BTreeMap<String, String>,
    name: Option<&str>,
    tools_given: bool,
) -> Result<(&'a str, &'a str)> {
    let as_strs = |(name, source): (&'a String, &'a String)| (name.as_str(), source.as_str());
    if let Some(name) = name {
        return templates.get_key_value(name).map(as_strs).ok_or_else(|| {
            Error::Tokenizer(format!(
                "chat template `{name}` was not found; the checkpoint provides {}",
                listed_names(templates)
            ))
        });
    }
    let preferred = if tools_given {
        templates.get_key_value(TOOL_USE_TEMPLATE)
    } else {
        None
    };
    if let Some(entry) = preferred.or_else(|| templates.get_key_value(DEFAULT_TEMPLATE)) {
        return Ok(as_strs(entry));
    }
    let mut entries = templates.iter();
    match (entries.next(), entries.next()) {
        (Some(only), None) => Ok(as_strs(only)),
        (None, _) => Err(Error::Tokenizer(
            "checkpoint does not provide a chat template".into(),
        )),
        (Some(_), Some(_)) => Err(Error::Tokenizer(format!(
            "checkpoint provides multiple chat templates ({}) but no `{DEFAULT_TEMPLATE}`; choose one with `template_name`",
            listed_names(templates)
        ))),
    }
}

/// Quotes the available template names for an error message, cut short after a few.
#[cfg(feature = "chat-templates")]
fn listed_names(templates: &BTreeMap<String, String>) -> String {
    if templates.is_empty() {
        return "no chat templates".into();
    }
    let mut listed = templates
        .keys()
        .take(LISTED_TEMPLATE_NAMES)
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    if templates.len() > LISTED_TEMPLATE_NAMES {
        listed.push_str(", ...");
    }
    listed
}

#[cfg(test)]
#[path = "catalog_tests.rs"]
mod tests;
