//! Local Hugging Face tokenizer artifacts and bounded chat-template rendering.
//!
//! Tokenization is intentionally an optional host-side concern. The runtime
//! opens only files below a caller-selected model directory and never invokes
//! Hub APIs. Encoded masks use the crate-wide semantic convention `true ==
//! valid token`; conversion to a backend tensor happens only after rectangular
//! padding has been validated.
//!
//! Chat templates are data supplied by a checkpoint and therefore are treated
//! as untrusted programs. The optional MiniJinja environment has no filesystem
//! loader, rejects undefined values, limits recursion, and meters execution
//! with fuel. These constraints do not prove a template benign, but they bound
//! the resources and capabilities available to ordinary model templates.

use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
};

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::{CausalLmInput, Error, Result, architecture::TokenMask};

const MAX_TOKENIZER_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
#[cfg(feature = "chat-templates")]
const CHAT_TEMPLATE_FUEL: u64 = 250_000;

/// Side on which shorter rows receive padding tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PaddingSide {
    /// Place padding before real tokens, as decoder-only batched prefill often does.
    Left,
    /// Place padding after real tokens, as encoder inputs commonly do.
    #[default]
    Right,
}

/// Side discarded when tokenization exceeds an explicit maximum length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TruncationSide {
    /// Discard the earliest tokens and retain the most recent suffix.
    Left,
    /// Discard the latest tokens and retain the earliest prefix.
    #[default]
    Right,
}

/// Rectangular padding policy for a batch encoding operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    /// Preserve each independently encoded row length.
    #[default]
    None,
    /// Pad every row to the longest encoded row in the call.
    Longest,
    /// Pad every row to exactly the supplied length.
    Fixed(usize),
}

/// Host-side controls for text tokenization and rectangular batching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EncodeOptions {
    /// Whether the tokenizer's post-processor may insert model special tokens.
    pub add_special_tokens: bool,
    /// Optional maximum length applied after special-token insertion.
    pub max_length: Option<usize>,
    /// Batch padding policy.
    pub padding: Padding,
    /// Optional override of the checkpoint padding side.
    pub padding_side: Option<PaddingSide>,
    /// Optional override of the checkpoint truncation side.
    pub truncation_side: Option<TruncationSide>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            max_length: None,
            padding: Padding::None,
            padding_side: None,
            truncation_side: None,
        }
    }
}

impl EncodeOptions {
    /// Rejects empty limits and inconsistent fixed-padding controls.
    fn validate(&self) -> Result<()> {
        if self.max_length == Some(0) {
            return Err(Error::Tokenizer(
                "max_length must be greater than zero when present".into(),
            ));
        }
        if let Padding::Fixed(0) = self.padding {
            return Err(Error::Tokenizer(
                "fixed padding length must be greater than zero".into(),
            ));
        }
        if let (Some(maximum), Padding::Fixed(width)) = (self.max_length, self.padding)
            && maximum > width
        {
            return Err(Error::Tokenizer(format!(
                "max_length {maximum} exceeds fixed padding width {width}"
            )));
        }
        Ok(())
    }
}

/// A special token resolved to both serialized text and vocabulary ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialToken {
    /// String spelling stored by the tokenizer artifacts.
    pub content: String,
    /// Vocabulary row associated with the spelling.
    pub id: u32,
}

/// Common special tokens normalized from Hugging Face metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: Option<SpecialToken>,
    /// End-of-sequence token.
    pub eos: Option<SpecialToken>,
    /// Padding token required by rectangular batching.
    pub pad: Option<SpecialToken>,
    /// Unknown-token fallback.
    pub unk: Option<SpecialToken>,
    /// Additional special tokens in their artifact order.
    pub additional: Vec<SpecialToken>,
}

/// Normalized, serializable tokenizer-side checkpoint metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerMetadata {
    /// Tokenizer implementation name declared by `tokenizer_config.json`.
    pub tokenizer_class: Option<String>,
    /// Finite model maximum that can be represented as `usize`.
    pub model_max_length: Option<usize>,
    /// Default padding side.
    pub padding_side: PaddingSide,
    /// Default truncation side.
    pub truncation_side: TruncationSide,
    /// Vocabulary size including added tokens.
    pub vocabulary_size: usize,
    /// Resolved special-token spellings and IDs.
    pub special_tokens: SpecialTokens,
    /// Named chat templates, with a scalar template normalized as `default`.
    pub chat_templates: BTreeMap<String, String>,
}

/// One role/content pair passed to a checkpoint chat template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Template-visible role such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Template-visible message content; strings and structured multimodal lists are accepted.
    pub content: Value,
}

impl ChatMessage {
    /// Creates a conventional text-only chat message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Value::String(content.into()),
        }
    }
}

/// Optional variables controlling checkpoint chat-template rendering.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatTemplateOptions {
    /// Named template; omission selects `default` or the only available template.
    pub template_name: Option<String>,
    /// Whether the rendered prompt should end at the next assistant turn prefix.
    pub add_generation_prompt: bool,
    /// Optional tool schemas exposed to templates that support tool use.
    pub tools: Option<Value>,
}

/// Token IDs and semantic validity masks retained on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEncoding {
    token_ids: Vec<Vec<u32>>,
    valid_tokens: Vec<Vec<bool>>,
}

impl BatchEncoding {
    /// Returns token rows in input order.
    pub fn token_ids(&self) -> &[Vec<u32>] {
        &self.token_ids
    }

    /// Returns masks whose `true` entries correspond to real, non-padding tokens.
    pub fn valid_tokens(&self) -> &[Vec<bool>] {
        &self.valid_tokens
    }

    /// Returns the number of encoded input strings.
    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }

    /// Converts a rectangular host batch into backend token and mask tensors.
    pub fn to_causal_input<B: Backend>(&self, device: &B::Device) -> Result<CausalLmInput<B>> {
        let batch = self.token_ids.len();
        let sequence = self.token_ids.first().map_or(0, Vec::len);
        if batch == 0 || sequence == 0 {
            return Err(Error::Tokenizer(
                "cannot create model input from an empty token batch".into(),
            ));
        }
        for (row, (tokens, valid)) in self.token_ids.iter().zip(&self.valid_tokens).enumerate() {
            if tokens.len() != sequence || valid.len() != sequence {
                return Err(Error::Tokenizer(format!(
                    "batch row {row} is not rectangular: expected {sequence} tokens, found {} tokens and {} mask entries",
                    tokens.len(),
                    valid.len()
                )));
            }
        }
        let ids = self
            .token_ids
            .iter()
            .flatten()
            .map(|token| i64::from(*token))
            .collect::<Vec<_>>();
        let valid = self
            .valid_tokens
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        Ok(CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids, [batch, sequence]),
                device,
            ),
            token_mask: Some(TokenMask::new(Tensor::<B, 2, Bool>::from_data(
                TensorData::new(valid, [batch, sequence]),
                device,
            ))),
            position_ids: None,
        })
    }
}

/// A tokenizer loaded exclusively from a local model directory.
pub struct LocalTokenizer {
    tokenizer: Tokenizer,
    metadata: TokenizerMetadata,
    root: PathBuf,
}

impl fmt::Debug for LocalTokenizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalTokenizer")
            .field("root", &self.root)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl LocalTokenizer {
    /// Opens `tokenizer.json` and optional metadata from one contained directory.
    pub fn from_directory(root: impl AsRef<Path>) -> Result<Self> {
        let supplied_root = root.as_ref();
        let root = supplied_root
            .canonicalize()
            .map_err(|source| Error::io(supplied_root, source))?;
        if !root.is_dir() {
            return Err(Error::Artifact {
                path: root,
                reason: "tokenizer root must be a directory".into(),
            });
        }
        let tokenizer_path = contained_file(&root, "tokenizer.json", true, MAX_TOKENIZER_BYTES)?
            .ok_or_else(|| Error::Artifact {
                path: root.join("tokenizer.json"),
                reason: "required tokenizer artifact is missing".into(),
            })?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to decode {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        let tokenizer_config = read_optional_json(&root, "tokenizer_config.json")?;
        let special_map = read_optional_json(&root, "special_tokens_map.json")?;
        let metadata = normalize_metadata(
            &root,
            &tokenizer,
            tokenizer_config.as_ref(),
            special_map.as_ref(),
        )?;
        Ok(Self {
            tokenizer,
            metadata,
            root,
        })
    }

    /// Returns the canonical model directory used for every artifact lookup.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns normalized tokenizer metadata and special-token IDs.
    pub fn metadata(&self) -> &TokenizerMetadata {
        &self.metadata
    }

    /// Encodes one input without adding artificial batch padding.
    pub fn encode(&self, text: &str, options: &EncodeOptions) -> Result<BatchEncoding> {
        self.encode_batch(&[text], options)
    }

    /// Encodes, truncates, and optionally pads text entirely from local state.
    pub fn encode_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        options: &EncodeOptions,
    ) -> Result<BatchEncoding> {
        options.validate()?;
        if texts.is_empty() {
            return Err(Error::Tokenizer(
                "encode_batch requires at least one input string".into(),
            ));
        }
        let truncation_side = options
            .truncation_side
            .unwrap_or(self.metadata.truncation_side);
        let mut rows = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let encoding = self
                    .tokenizer
                    .encode(text.as_ref(), options.add_special_tokens)
                    .map_err(|error| {
                        Error::Tokenizer(format!("failed to encode batch row {index}: {error}"))
                    })?;
                let mut ids = encoding.get_ids().to_vec();
                if let Some(maximum) = options.max_length
                    && ids.len() > maximum
                {
                    match truncation_side {
                        TruncationSide::Left => ids = ids.split_off(ids.len() - maximum),
                        TruncationSide::Right => ids.truncate(maximum),
                    }
                }
                if ids.is_empty() {
                    return Err(Error::Tokenizer(format!(
                        "batch row {index} encoded to no tokens"
                    )));
                }
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;

        let target = match options.padding {
            Padding::None => None,
            Padding::Longest => rows.iter().map(Vec::len).max(),
            Padding::Fixed(width) => Some(width),
        };
        let mut masks = rows
            .iter()
            .map(|row| vec![true; row.len()])
            .collect::<Vec<_>>();
        if let Some(target) = target {
            let pad_id = self
                .metadata
                .special_tokens
                .pad
                .as_ref()
                .ok_or_else(|| {
                    Error::Tokenizer("padding requested but no pad token is declared".into())
                })?
                .id;
            let padding_side = options.padding_side.unwrap_or(self.metadata.padding_side);
            for (index, (row, mask)) in rows.iter_mut().zip(&mut masks).enumerate() {
                if row.len() > target {
                    return Err(Error::Tokenizer(format!(
                        "encoded row {index} has length {}, exceeding padding width {target}; set max_length to request truncation",
                        row.len()
                    )));
                }
                let amount = target - row.len();
                if amount == 0 {
                    continue;
                }
                match padding_side {
                    PaddingSide::Left => {
                        let mut padded = vec![pad_id; amount];
                        padded.append(row);
                        *row = padded;
                        let mut padded_mask = vec![false; amount];
                        padded_mask.append(mask);
                        *mask = padded_mask;
                    }
                    PaddingSide::Right => {
                        row.resize(target, pad_id);
                        mask.resize(target, false);
                    }
                }
            }
        }
        Ok(BatchEncoding {
            token_ids: rows,
            valid_tokens: masks,
        })
    }

    /// Decodes vocabulary IDs using the tokenizer's configured decoder.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|error| Error::Tokenizer(format!("failed to decode token IDs: {error}")))
    }

    /// Renders one bounded checkpoint chat template without filesystem access.
    #[cfg(feature = "chat-templates")]
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<String> {
        use minijinja::{Environment, ErrorKind, UndefinedBehavior};

        if messages.is_empty() {
            return Err(Error::Tokenizer(
                "chat rendering requires at least one message".into(),
            ));
        }
        let (name, template) = select_template(&self.metadata.chat_templates, options)?;
        let mut environment = Environment::new();
        environment.set_undefined_behavior(UndefinedBehavior::Strict);
        environment.set_recursion_limit(64);
        environment.set_fuel(Some(CHAT_TEMPLATE_FUEL));
        environment.add_function(
            "raise_exception",
            |message: String| -> core::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
            },
        );

        let mut context = Map::new();
        context.insert(
            "messages".into(),
            serde_json::to_value(messages).map_err(|error| {
                Error::Tokenizer(format!("failed to serialize chat messages: {error}"))
            })?,
        );
        context.insert(
            "add_generation_prompt".into(),
            Value::Bool(options.add_generation_prompt),
        );
        context.insert("tools".into(), options.tools.clone().unwrap_or(Value::Null));
        insert_special_context(&mut context, "bos_token", &self.metadata.special_tokens.bos);
        insert_special_context(&mut context, "eos_token", &self.metadata.special_tokens.eos);
        insert_special_context(&mut context, "pad_token", &self.metadata.special_tokens.pad);
        insert_special_context(&mut context, "unk_token", &self.metadata.special_tokens.unk);

        environment
            .render_str(template, Value::Object(context))
            .map_err(|error| Error::Tokenizer(format!("chat template `{name}` failed: {error}")))
    }
}

/// Resolves a fixed artifact name and ensures symlinks cannot escape the root.
fn contained_file(
    root: &Path,
    name: &str,
    required: bool,
    maximum_bytes: u64,
) -> Result<Option<PathBuf>> {
    let candidate = root.join(name);
    let canonical = match candidate.canonicalize() {
        Ok(path) => path,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound && !required => {
            return Ok(None);
        }
        Err(source) => return Err(Error::io(candidate, source)),
    };
    if !canonical.starts_with(root) {
        return Err(Error::Artifact {
            path: candidate,
            reason: "artifact symlink escapes the canonical tokenizer root".into(),
        });
    }
    let metadata = canonical
        .metadata()
        .map_err(|source| Error::io(&canonical, source))?;
    if !metadata.is_file() {
        return Err(Error::Artifact {
            path: canonical,
            reason: "artifact must be a regular file".into(),
        });
    }
    if metadata.len() > maximum_bytes {
        return Err(Error::Artifact {
            path: canonical,
            reason: format!(
                "artifact size {} exceeds safety limit {maximum_bytes}",
                metadata.len()
            ),
        });
    }
    Ok(Some(canonical))
}

/// Reads a small optional JSON metadata artifact with contextual errors.
fn read_optional_json(root: &Path, name: &str) -> Result<Option<Value>> {
    let Some(path) = contained_file(root, name, false, MAX_METADATA_BYTES)? else {
        return Ok(None);
    };
    let bytes = fs::read(&path).map_err(|source| Error::io(&path, source))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|source| Error::json(path, source))
}

/// Combines permissive Hugging Face JSON shapes into a stable metadata schema.
fn normalize_metadata(
    root: &Path,
    tokenizer: &Tokenizer,
    tokenizer_config: Option<&Value>,
    special_map: Option<&Value>,
) -> Result<TokenizerMetadata> {
    for (name, value) in [
        ("tokenizer_config.json", tokenizer_config),
        ("special_tokens_map.json", special_map),
    ] {
        if value.is_some_and(|value| !value.is_object()) {
            return Err(Error::Artifact {
                path: root.join(name),
                reason: "top-level JSON value must be an object".into(),
            });
        }
    }
    let get = |name: &str| tokenizer_config.and_then(|value| value.get(name));
    let tokenizer_class = get("tokenizer_class")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let model_max_length = get("model_max_length")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let padding_side = parse_side(get("padding_side"), PaddingSide::Right, "padding_side")?;
    let truncation_side = parse_truncation_side(get("truncation_side"))?;
    let special_tokens = SpecialTokens {
        bos: resolve_special("bos_token", tokenizer_config, special_map, tokenizer)?,
        eos: resolve_special("eos_token", tokenizer_config, special_map, tokenizer)?,
        pad: resolve_special("pad_token", tokenizer_config, special_map, tokenizer)?,
        unk: resolve_special("unk_token", tokenizer_config, special_map, tokenizer)?,
        additional: resolve_additional(tokenizer_config, special_map, tokenizer)?,
    };
    let mut chat_templates = parse_chat_templates(get("chat_template"))?;
    if chat_templates.is_empty()
        && let Some(path) = contained_file(root, "chat_template.jinja", false, MAX_METADATA_BYTES)?
    {
        let template = fs::read_to_string(&path).map_err(|source| Error::io(&path, source))?;
        if template.trim().is_empty() {
            return Err(Error::Artifact {
                path,
                reason: "chat template must not be empty".into(),
            });
        }
        chat_templates.insert("default".into(), template);
    }
    Ok(TokenizerMetadata {
        tokenizer_class,
        model_max_length,
        padding_side,
        truncation_side,
        vocabulary_size: tokenizer.get_vocab_size(true),
        special_tokens,
        chat_templates,
    })
}

/// Parses the two accepted string spellings for padding side.
fn parse_side(value: Option<&Value>, default: PaddingSide, field: &str) -> Result<PaddingSide> {
    match value {
        None => Ok(default),
        Some(Value::String(value)) if value == "left" => Ok(PaddingSide::Left),
        Some(Value::String(value)) if value == "right" => Ok(PaddingSide::Right),
        Some(value) => Err(Error::Tokenizer(format!(
            "{field} must be `left` or `right`, found {value}"
        ))),
    }
}

/// Parses checkpoint truncation side independently from padding defaults.
fn parse_truncation_side(value: Option<&Value>) -> Result<TruncationSide> {
    match parse_side(value, PaddingSide::Right, "truncation_side")? {
        PaddingSide::Left => Ok(TruncationSide::Left),
        PaddingSide::Right => Ok(TruncationSide::Right),
    }
}

/// Resolves a declared token from either metadata file and validates its ID.
fn resolve_special(
    field: &str,
    tokenizer_config: Option<&Value>,
    special_map: Option<&Value>,
    tokenizer: &Tokenizer,
) -> Result<Option<SpecialToken>> {
    let declared = tokenizer_config
        .and_then(|value| value.get(field))
        .or_else(|| special_map.and_then(|value| value.get(field)));
    let declared_id = tokenizer_config
        .and_then(|value| value.get(format!("{field}_id")))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let content = declared
        .and_then(token_content)
        .or_else(|| declared_id.and_then(|id| tokenizer.id_to_token(id)));
    let Some(content) = content else {
        if declared.is_some() || declared_id.is_some() {
            return Err(Error::Tokenizer(format!(
                "could not resolve declared {field} to token text"
            )));
        }
        return Ok(None);
    };
    let vocabulary_id = tokenizer.token_to_id(&content);
    let id = match (declared_id, vocabulary_id) {
        (Some(declared), Some(vocabulary)) if declared != vocabulary => {
            return Err(Error::Tokenizer(format!(
                "{field}_id {declared} conflicts with tokenizer vocabulary ID {vocabulary}"
            )));
        }
        (Some(id), _) | (None, Some(id)) => id,
        (None, None) => {
            return Err(Error::Tokenizer(format!(
                "declared {field} token `{content}` is absent from the vocabulary"
            )));
        }
    };
    Ok(Some(SpecialToken { content, id }))
}

/// Extracts the content of a string or serialized Hugging Face AddedToken.
fn token_content(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned).or_else(|| {
        value
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

/// Normalizes the optional additional-special-token array without duplicates.
fn resolve_additional(
    tokenizer_config: Option<&Value>,
    special_map: Option<&Value>,
    tokenizer: &Tokenizer,
) -> Result<Vec<SpecialToken>> {
    let declared = tokenizer_config
        .and_then(|value| value.get("additional_special_tokens"))
        .or_else(|| special_map.and_then(|value| value.get("additional_special_tokens")));
    let Some(declared) = declared else {
        return Ok(Vec::new());
    };
    let array = declared
        .as_array()
        .ok_or_else(|| Error::Tokenizer("additional_special_tokens must be an array".into()))?;
    let mut resolved = Vec::with_capacity(array.len());
    for (index, value) in array.iter().enumerate() {
        let content = token_content(value).ok_or_else(|| {
            Error::Tokenizer(format!(
                "additional_special_tokens[{index}] must be a string or AddedToken object"
            ))
        })?;
        let id = tokenizer.token_to_id(&content).ok_or_else(|| {
            Error::Tokenizer(format!(
                "additional special token `{content}` is absent from the vocabulary"
            ))
        })?;
        if resolved
            .iter()
            .any(|token: &SpecialToken| token.id == id || token.content == content)
        {
            return Err(Error::Tokenizer(format!(
                "duplicate additional special token `{content}`"
            )));
        }
        resolved.push(SpecialToken { content, id });
    }
    Ok(resolved)
}

/// Accepts scalar, list, and map chat-template encodings used by Hub artifacts.
fn parse_chat_templates(value: Option<&Value>) -> Result<BTreeMap<String, String>> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let mut templates = BTreeMap::new();
    match value {
        Value::String(template) => insert_template(&mut templates, "default", template)?,
        Value::Array(entries) => {
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
        Value::Object(entries) => {
            for (name, template) in entries {
                let template = template.as_str().ok_or_else(|| {
                    Error::Tokenizer(format!("chat template `{name}` must be a string"))
                })?;
                insert_template(&mut templates, name, template)?;
            }
        }
        _ => {
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

/// Selects the explicit, default, or only template deterministically.
#[cfg(feature = "chat-templates")]
fn select_template<'a>(
    templates: &'a BTreeMap<String, String>,
    options: &ChatTemplateOptions,
) -> Result<(&'a str, &'a str)> {
    if let Some(name) = options.template_name.as_deref() {
        return templates
            .get_key_value(name)
            .map(|(name, template)| (name.as_str(), template.as_str()))
            .ok_or_else(|| Error::Tokenizer(format!("chat template `{name}` was not found")));
    }
    if let Some(template) = templates.get("default") {
        return Ok(("default", template));
    }
    if templates.len() == 1 {
        let (name, template) = templates
            .first_key_value()
            .expect("length was checked immediately above");
        return Ok((name, template));
    }
    Err(Error::Tokenizer(if templates.is_empty() {
        "checkpoint does not provide a chat template".into()
    } else {
        "checkpoint provides multiple chat templates but no default; choose a name".into()
    }))
}

/// Adds a special token spelling or JSON null to template context.
#[cfg(feature = "chat-templates")]
fn insert_special_context(
    context: &mut Map<String, Value>,
    name: &str,
    token: &Option<SpecialToken>,
) {
    context.insert(
        name.into(),
        token
            .as_ref()
            .map_or(Value::Null, |token| Value::String(token.content.clone())),
    );
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use burn::backend::Flex;
    use tempfile::TempDir;
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

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
}
