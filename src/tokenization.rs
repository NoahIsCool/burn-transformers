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

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
#[cfg(feature = "chat-templates")]
use serde_json::Map;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{
    CausalLmInput, Error, Result,
    architecture::{TokenMask, bool_tensor_from_values},
};

/// Maximum accepted tokenizer JSON size in bytes before parsing.
const MAX_TOKENIZER_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Maximum accepted auxiliary metadata file size in bytes.
const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
/// Instruction budget limiting each local chat-template evaluation.
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
    Fixed(
        /// Required length of each padded token row.
        usize,
    ),
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
    /// Enables checkpoint special tokens with no added padding or truncation limit.
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
    /// Optional hidden reasoning content used by Qwen-style assistant turns.
    pub reasoning_content: Option<String>,
    /// Optional structured tool calls emitted by an assistant turn.
    pub tool_calls: Option<Value>,
    /// Optional participant or tool name.
    pub name: Option<String>,
}

impl ChatMessage {
    /// Creates a conventional text-only chat message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Value::String(content.into()),
            reasoning_content: None,
            tool_calls: None,
            name: None,
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
    /// Whether multimodal templates should number image and video placeholders.
    pub add_vision_id: bool,
    /// Optional reasoning-mode switch used by checkpoints that expose one.
    pub enable_thinking: Option<bool>,
    /// Additional checkpoint-specific variables; reserved built-ins cannot be replaced.
    pub extra_context: BTreeMap<String, Value>,
}

/// Token IDs and semantic validity masks retained on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEncoding {
    /// Encoded token rows, including any explicitly requested padding.
    token_ids: Vec<Vec<u32>>,
    /// Per-token indicators distinguishing real input from added padding.
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
            token_mask: Some(TokenMask::new(bool_tensor_from_values(
                valid,
                [batch, sequence],
                device,
            ))),
            position_ids: None,
        })
    }
}

/// A tokenizer loaded exclusively from a local model directory.
pub struct LocalTokenizer {
    /// Locally deserialized tokenizer and its vocabulary processing rules.
    tokenizer: Tokenizer,
    /// Validated special tokens, padding defaults, and optional chat templates.
    metadata: TokenizerMetadata,
    /// Source model directory used for diagnostics and artifact resolution.
    root: PathBuf,
}

impl fmt::Debug for LocalTokenizer {
    /// Formats metadata without dumping the tokenizer implementation or vocabulary.
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

    /// Verifies that every locally declared token can index a model embedding row.
    ///
    /// A model vocabulary may be larger than the tokenizer vocabulary because
    /// checkpoint matrices are often padded for accelerator alignment. The
    /// reverse relationship is unsafe: tokenization could then produce an ID
    /// outside the embedding table. This method therefore validates the greatest
    /// concrete vocabulary ID rather than requiring the two reported counts to
    /// be equal.
    pub fn validate_model_vocabulary(&self, model_vocabulary_size: usize) -> Result<()> {
        if model_vocabulary_size == 0 {
            return Err(Error::Tokenizer(
                "model vocabulary size must be greater than zero".into(),
            ));
        }
        let maximum_id = self
            .tokenizer
            .get_vocab(true)
            .into_values()
            .max()
            .ok_or_else(|| Error::Tokenizer("tokenizer vocabulary must not be empty".into()))?;
        if maximum_id as usize >= model_vocabulary_size {
            return Err(Error::Tokenizer(format!(
                "tokenizer emits ID {maximum_id}, outside model vocabulary size {model_vocabulary_size}"
            )));
        }
        Ok(())
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
        environment.set_unknown_method_callback(huggingface_string_method);
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
        context.insert("add_vision_id".into(), Value::Bool(options.add_vision_id));
        context.insert(
            "enable_thinking".into(),
            options.enable_thinking.map_or(Value::Null, Value::Bool),
        );
        insert_special_context(&mut context, "bos_token", &self.metadata.special_tokens.bos);
        insert_special_context(&mut context, "eos_token", &self.metadata.special_tokens.eos);
        insert_special_context(&mut context, "pad_token", &self.metadata.special_tokens.pad);
        insert_special_context(&mut context, "unk_token", &self.metadata.special_tokens.unk);
        for (name, value) in &options.extra_context {
            if context.contains_key(name) {
                return Err(Error::Tokenizer(format!(
                    "extra chat context cannot replace reserved variable `{name}`"
                )));
            }
            context.insert(name.clone(), value.clone());
        }

        environment
            .render_str(template, Value::Object(context))
            .map_err(|error| Error::Tokenizer(format!("chat template `{name}` failed: {error}")))
    }

    /// Renders and tokenizes a chat prompt with checkpoint delimiters exactly once.
    ///
    /// Hugging Face chat templates normally emit their own beginning, role, and
    /// end markers. Applying a tokenizer post-processor afterward may duplicate
    /// those special tokens, so this helper always disables
    /// [`EncodeOptions::add_special_tokens`] after rendering. Truncation and
    /// padding controls supplied in `encode_options` remain unchanged.
    #[cfg(feature = "chat-templates")]
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        template_options: &ChatTemplateOptions,
        encode_options: &EncodeOptions,
    ) -> Result<BatchEncoding> {
        let prompt = self.render_chat(messages, template_options)?;
        let mut encode_options = encode_options.clone();
        encode_options.add_special_tokens = false;
        self.encode(&prompt, &encode_options)
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
///
/// Metadata cannot create a vocabulary row: the tokenizer's concrete text-to-ID
/// mapping is authoritative. Every accepted special ID must name that same row,
/// so later padding cannot bypass the model-vocabulary bounds check. Null IDs
/// are absent overrides; malformed present IDs or text are not silently absent.
fn resolve_special(
    field: &str,
    tokenizer_config: Option<&Value>,
    special_map: Option<&Value>,
    tokenizer: &Tokenizer,
) -> Result<Option<SpecialToken>> {
    let declared = tokenizer_config
        .and_then(|value| value.get(field))
        .filter(|value| !value.is_null())
        .or_else(|| {
            special_map
                .and_then(|value| value.get(field))
                .filter(|value| !value.is_null())
        });
    let declared_id = tokenizer_config
        .and_then(|value| value.get(format!("{field}_id")))
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::Tokenizer(format!(
                        "{field}_id must be an unsigned 32-bit integer or null, found {value}"
                    ))
                })
        })
        .transpose()?;
    let content = match declared {
        Some(value) => Some(token_content(value).ok_or_else(|| {
            Error::Tokenizer(format!("could not resolve declared {field} to token text"))
        })?),
        None => declared_id.and_then(|id| tokenizer.id_to_token(id)),
    };
    let Some(content) = content else {
        if declared_id.is_some() {
            return Err(Error::Tokenizer(format!(
                "could not resolve declared {field} to token text"
            )));
        }
        return Ok(None);
    };
    let id = tokenizer.token_to_id(&content).ok_or_else(|| {
        Error::Tokenizer(format!(
            "declared {field} token `{content}` is absent from the vocabulary"
        ))
    })?;
    if let Some(declared) = declared_id
        && declared != id
    {
        return Err(Error::Tokenizer(format!(
            "{field}_id {declared} conflicts with tokenizer vocabulary ID {id}"
        )));
    }
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
        return templates
            .first_key_value()
            .map(|(name, template)| (name.as_str(), template.as_str()))
            .ok_or_else(|| Error::Tokenizer("checkpoint does not provide a chat template".into()));
    }
    Err(Error::Tokenizer(if templates.is_empty() {
        "checkpoint does not provide a chat template".into()
    } else {
        "checkpoint provides multiple chat templates but no default; choose a name".into()
    }))
}

/// Implements the small Python-string-method subset used by official HF templates.
#[cfg(feature = "chat-templates")]
fn huggingface_string_method(
    _state: &minijinja::State<'_, '_>,
    value: &minijinja::Value,
    method: &str,
    arguments: &[minijinja::Value],
) -> core::result::Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error as JinjaError, ErrorKind, Value as JinjaValue};

    let source = value
        .as_str()
        .ok_or_else(|| JinjaError::from(ErrorKind::UnknownMethod))?;
    let one_string = || {
        if arguments.len() != 1 {
            return Err(JinjaError::new(
                ErrorKind::InvalidOperation,
                format!("string.{method} expects exactly one string argument"),
            ));
        }
        arguments[0].as_str().ok_or_else(|| {
            JinjaError::new(
                ErrorKind::InvalidOperation,
                format!("string.{method} expects a string argument"),
            )
        })
    };
    match method {
        "startswith" => Ok(JinjaValue::from(source.starts_with(one_string()?))),
        "endswith" => Ok(JinjaValue::from(source.ends_with(one_string()?))),
        "split" => Ok(JinjaValue::from_serialize(
            source
                .split(one_string()?)
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        )),
        "lstrip" | "rstrip" | "strip" => {
            if arguments.len() > 1
                || arguments
                    .first()
                    .is_some_and(|value| value.as_str().is_none())
            {
                return Err(JinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("string.{method} accepts at most one string argument"),
                ));
            }
            let characters = arguments.first().and_then(JinjaValue::as_str);
            let stripped = match (method, characters) {
                ("lstrip", Some(characters)) => {
                    source.trim_start_matches(|character| characters.contains(character))
                }
                ("rstrip", Some(characters)) => {
                    source.trim_end_matches(|character| characters.contains(character))
                }
                ("strip", Some(characters)) => {
                    source.trim_matches(|character| characters.contains(character))
                }
                ("lstrip", None) => source.trim_start(),
                ("rstrip", None) => source.trim_end(),
                ("strip", None) => source.trim(),
                _ => source,
            };
            Ok(JinjaValue::from(stripped.to_owned()))
        }
        "lower" if arguments.is_empty() => Ok(JinjaValue::from(source.to_lowercase())),
        "upper" if arguments.is_empty() => Ok(JinjaValue::from(source.to_uppercase())),
        _ => Err(JinjaError::from(ErrorKind::UnknownMethod)),
    }
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
#[path = "tokenization_tests.rs"]
mod tests;
