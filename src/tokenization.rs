//! Local Hugging Face tokenizer artifacts and bounded chat-template rendering.
//!
//! Tokenization is intentionally an optional host-side concern. The runtime
//! opens only files below a caller-selected model directory and never invokes
//! Hub APIs. Encoded masks use the crate-wide semantic convention `true ==
//! valid token`; conversion to a backend tensor happens only after rectangular
//! padding has been validated.
//!
//! # Truncation and special tokens
//!
//! Encoding follows Hugging Face `transformers` semantics. A row is encoded as
//! content first; when [`EncodeOptions::max_length`] is set, the content is cut
//! to `max_length - r` IDs, where `r` is the number of template tokens the
//! checkpoint's post-processor adds (`</s>` for T5, `<s>` for Llama-style
//! tokenizers). Only then does the post-processor insert those tokens, so a
//! truncated row still begins and ends exactly as the model was trained to see.
//! Truncation and padding blocks serialized inside `tokenizer.json` are cleared
//! at load time: like `transformers`, this crate applies them only on request.
//!
//! # Chat templates
//!
//! With the `chat-templates` feature, [`LocalTokenizer::render_chat`] renders a
//! checkpoint's Jinja chat template the way Hugging Face's
//! `apply_chat_template` does: the same whitespace rules, lenient undefined
//! variables, Python `str`/`dict` methods and value printing, a `json.dumps`
//! `tojson` filter, `raise_exception`, `strftime_now`, the `tools` and
//! `documents` variables, and every declared special token. Messages serialize
//! like the plain dictionaries Python callers pass, so optional keys are
//! absent rather than `null`.
//!
//! Templates are found and chosen as in Hugging Face 5.2. `chat_template.jinja`
//! (the `default` template) and `additional_chat_templates/<name>.jinja`
//! replace any `chat_template` entry in `tokenizer_config.json`, and, as in
//! Jinja2, every `\r\n` and `\r` line ending of a template's source renders
//! as `\n`. A render without [`ChatTemplateOptions::template_name`] uses the
//! checkpoint's `tool_use` template when [`ChatTemplateOptions::tools`] is
//! set, otherwise `default`, otherwise the only template.
//!
//! Known differences from Hugging Face, each of which changes prompt text:
//!
//! - **Dictionary keys reach templates sorted.** Python preserves insertion
//!   order; this crate's `serde_json` and `minijinja` builds store objects as
//!   sorted maps. A tool schema written `{"type": ..., "function": ...}`
//!   therefore renders as `{"function": ..., "type": ...}`, and a message
//!   prints as `{'content': ..., 'role': ...}`. Prompts match Hugging Face
//!   byte for byte only when every object in `messages`, `tools`, and
//!   `documents` is already key-sorted.
//! - **`strftime_now` formats UTC**, not host local time (see
//!   [`ChatTemplateOptions::current_time_unix_seconds`]).
//! - **`'text' ~ value` prints booleans and null as `true`/`none`**, where
//!   Python prints `True`/`None`. Official templates convert values with
//!   `| string` first, which matches Python exactly.
//! - **`'%s' % value` string formatting is unsupported** and fails rendering.
//! - **A newline right after `{% raw %}` is dropped.** MiniJinja applies
//!   `trim_blocks` to the `raw` tag; Jinja2 does not, so Python keeps that
//!   newline (`{% raw %}\n{{ x }}{% endraw %}` renders `\n{{ x }}` there).
//! - **Resource limits.** A template that builds a value or prompt beyond the
//!   sandbox's limits (below) fails where Python might still render. The
//!   limits scale with the conversation, so only adversarial templates reach
//!   them; no published template does.
//!
//! Deliberate differences in naming and validating templates, which matter
//! only for calls and artifacts Hugging Face rejects or resolves arbitrarily:
//!
//! - **`template_name` only names a stored template.** Hugging Face's
//!   `chat_template` argument is looked up only among several named
//!   templates and is otherwise rendered as Jinja source, so an unknown name,
//!   or any name for a checkpoint with a single unnamed template, renders as
//!   literal text there. Here a template stored without a name (a string
//!   `chat_template` or `chat_template.jinja`) is `default`, and an unknown
//!   name is an error that lists the available names.
//! - **A sole non-`default` template renders without a name**, where Hugging
//!   Face raises "multiple chat templates with no default".
//! - **Ambiguous or malformed template artifacts are rejected**: empty
//!   templates or names, duplicate names (in a config list, or
//!   `additional_chat_templates/default.jinja` beside `chat_template.jinja`),
//!   template files that are not UTF-8, symlinks that leave the model
//!   directory, and template files beyond the loader's size and count limits.
//!   Hugging Face keeps the last duplicate and reads the rest.
//!
//! Chat templates are data supplied by a checkpoint and therefore are treated
//! as untrusted programs. The MiniJinja environment has no filesystem loader,
//! meters execution with fuel, caps recursion and nesting, bounds the rendered
//! prompt (1 MiB plus four times the serialized context), checks every value
//! built by `+`, `~`, `*`, printing, and the size-controlled filters before it
//! is allocated, and turns engine panics into errors. Two costs remain
//! unbounded because MiniJinja offers no hook for them: comparing containers
//! that share one value many times (time), and literal template text replayed
//! into `{% set %}` blocks or macro results (memory, at most fuel times the
//! template size). The private `chat_template` module documents each bound.

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
use tokenizers::{
    DecodeStream, PostProcessor, Tokenizer, TruncationDirection, decoders::DecoderWrapper,
    models::ModelWrapper, normalizers::NormalizerWrapper, pre_tokenizers::PreTokenizerWrapper,
    processors::PostProcessorWrapper,
};

use crate::{
    CausalLmInput, Error, Result,
    architecture::{TokenMask, bool_tensor_from_values},
};

// Submodules document themselves with `//!`; an outer `///` here would make
// rustdoc resolve their intra-doc links from this module's scope.
#[cfg(feature = "chat-templates")]
mod calendar; // Proleptic-Gregorian civil time for `strftime_now` (Verus-verified).
mod catalog; // Chat-template discovery and selection in Hugging Face's precedence.
#[cfg(feature = "chat-templates")]
mod chat_template; // Hugging Face-compatible template environment and `strftime`.
mod layout; // Truncation-budget and padding arithmetic (Verus-verified).
#[cfg(feature = "chat-templates")]
mod pycompat; // Python `str`/`dict` semantics, printing, and filters for templates.
mod vocabulary; // Special-token ID and embedding-row checks (Verus-verified).

use catalog::load_chat_templates;
#[cfg(feature = "chat-templates")]
use catalog::select_template;
use layout::{
    LimitViolation, PaddedBatch, content_token_budget, encoding_limit_violation, first_ragged_row,
    flatten_mask_rows, flatten_token_rows, longest_row, pad_batch, unpadded_masks,
};
use vocabulary::{declared_special_id, id_fits_vocabulary, reconcile_special_id};

/// Maximum accepted tokenizer JSON size in bytes before parsing.
const MAX_TOKENIZER_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Maximum accepted auxiliary metadata file size in bytes.
const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
/// Marker appended to the final message when it is rendered as an open prefix.
///
/// The same text Hugging Face uses; rendering is cut at its last occurrence.
#[cfg(feature = "chat-templates")]
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
/// Template variables supplied by the renderer that `extra_context` cannot replace.
#[cfg(feature = "chat-templates")]
const RESERVED_CHAT_VARIABLES: &[&str] = &[
    "messages",
    "tools",
    "documents",
    "add_generation_prompt",
    "add_vision_id",
    "enable_thinking",
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "raise_exception",
    "strftime_now",
];
/// Named [`ChatMessage`] fields that a flattened extra key must not shadow.
#[cfg(feature = "chat-templates")]
const CHAT_MESSAGE_FIELDS: &[&str] = &[
    "role",
    "content",
    "reasoning_content",
    "tool_calls",
    "name",
    "tool_call_id",
];

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
    /// Optional limit on each complete row, special tokens included.
    ///
    /// Content is truncated on [`Self::truncation_side`] *before* the
    /// post-processor adds its template, exactly as Hugging Face does, so
    /// BOS/EOS and similar template tokens are never cut off. A limit smaller
    /// than the template itself is rejected.
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
    ///
    /// The rules are checked by the verified [`encoding_limit_violation`].
    fn validate(&self) -> Result<()> {
        let fixed_width = match self.padding {
            Padding::Fixed(width) => Some(width),
            Padding::None | Padding::Longest => None,
        };
        match encoding_limit_violation(self.max_length, fixed_width) {
            None => Ok(()),
            Some(LimitViolation::ZeroMaxLength) => Err(Error::Tokenizer(
                "max_length must be greater than zero when present".into(),
            )),
            Some(LimitViolation::ZeroPaddingWidth) => Err(Error::Tokenizer(
                "fixed padding length must be greater than zero".into(),
            )),
            Some(LimitViolation::MaxLengthExceedsWidth {
                max_length: maximum,
                width,
            }) => Err(Error::Tokenizer(format!(
                "max_length {maximum} exceeds fixed padding width {width}"
            ))),
        }
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
///
/// These are Hugging Face's named special-token attributes; each declared one
/// is exposed to chat templates as `bos_token`, `eos_token`, and so on.
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
    /// Segment separator used by encoder tokenizers such as BERT's `[SEP]`.
    #[serde(default)]
    pub sep: Option<SpecialToken>,
    /// Classification token such as BERT's `[CLS]`.
    #[serde(default)]
    pub cls: Option<SpecialToken>,
    /// Masked-language-model token such as `[MASK]`.
    #[serde(default)]
    pub mask: Option<SpecialToken>,
    /// Additional special tokens in their artifact order.
    pub additional: Vec<SpecialToken>,
}

impl SpecialTokens {
    /// Lists declared named tokens with their chat-template variable names.
    ///
    /// The order matches Hugging Face's `SPECIAL_TOKENS_ATTRIBUTES`.
    pub fn named(&self) -> impl Iterator<Item = (&'static str, &SpecialToken)> {
        [
            ("bos_token", &self.bos),
            ("eos_token", &self.eos),
            ("unk_token", &self.unk),
            ("sep_token", &self.sep),
            ("pad_token", &self.pad),
            ("cls_token", &self.cls),
            ("mask_token", &self.mask),
        ]
        .into_iter()
        .filter_map(|(name, token)| token.as_ref().map(|token| (name, token)))
    }
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
    /// Named chat templates, in Hugging Face's precedence.
    ///
    /// When the model directory holds `chat_template.jinja` (stored as
    /// `default`) or `additional_chat_templates/<name>.jinja` files, they are
    /// the whole set; otherwise the entries of `tokenizer_config.json`'s
    /// `chat_template` are, with a scalar template stored as `default`.
    pub chat_templates: BTreeMap<String, String>,
}

/// One message passed to a checkpoint chat template.
///
/// A message reaches the template exactly like the plain dictionary a Python
/// caller would pass to `apply_chat_template`: unset optional fields are
/// *absent*, not `null`, so membership tests such as
/// `{% if 'tool_calls' in message %}` (used by Llama 3.x templates) take the
/// same branch as in Hugging Face. Checkpoint-specific keys go in
/// [`Self::extra`] and appear beside the named fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Template-visible role such as `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// Template-visible message content; strings and structured multimodal lists are accepted.
    pub content: Value,
    /// Optional hidden reasoning content used by Qwen-style assistant turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Optional structured tool calls emitted by an assistant turn.
    ///
    /// Hugging Face's convention is a list of
    /// `{"type": "function", "function": {"name": ..., "arguments": {...}}}`.
    /// Object keys, including argument names, reach the template sorted; see
    /// the [module documentation](self) for this difference from Python.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// Optional participant or tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional ID linking a `tool` result message to the call that produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Additional template-visible keys; they must not repeat a named field.
    ///
    /// Like every message key, they reach the template in sorted order.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ChatMessage {
    /// Creates a message with arbitrary JSON content, such as a multimodal block list.
    pub fn new(role: impl Into<String>, content: impl Into<Value>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            name: None,
            tool_call_id: None,
            extra: BTreeMap::new(),
        }
    }

    /// Creates a conventional text-only chat message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(role, Value::String(content.into()))
    }

    /// Attaches hidden reasoning content to an assistant turn.
    #[must_use]
    pub fn with_reasoning_content(mut self, reasoning_content: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning_content.into());
        self
    }

    /// Attaches the tool calls an assistant turn emitted.
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Value) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Attaches a participant or tool name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Links a `tool` result message to the call that produced it.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Adds one checkpoint-specific key visible to the template.
    #[must_use]
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Optional variables controlling checkpoint chat-template rendering.
///
/// Fields mirror the arguments of Hugging Face's `apply_chat_template`.
/// Optional switches that are unset are *absent* from the template context,
/// so `{% if enable_thinking is defined %}` behaves as in Python.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatTemplateOptions {
    /// Name of the checkpoint template to render, such as `default` or `tool_use`.
    ///
    /// Omission chooses as Hugging Face's `get_chat_template` does: `tool_use`
    /// when [`Self::tools`] is set and the checkpoint has that template,
    /// otherwise `default`, otherwise the checkpoint's only template. A name
    /// the checkpoint lacks is an error; unlike Hugging Face, it is never
    /// rendered as template source.
    pub template_name: Option<String>,
    /// Whether the rendered prompt should end at the next assistant turn prefix.
    pub add_generation_prompt: bool,
    /// Render the final message as an open prefix that the model continues.
    ///
    /// Equivalent to Hugging Face's `continue_final_message`: the prompt is cut
    /// right after the final message's text, dropping the template's
    /// end-of-turn markup. Incompatible with [`Self::add_generation_prompt`].
    pub continue_final_message: bool,
    /// Optional tool schemas exposed as `tools` to templates that support tool use.
    ///
    /// Any value but JSON `null` (Python's `None`), even an empty list, also
    /// selects the checkpoint's `tool_use` template when
    /// [`Self::template_name`] is unset, as in Hugging Face.
    ///
    /// Schema keys reach the template sorted, so a template that prints a
    /// schema with `tojson` emits `{"function": ..., "type": "function"}`
    /// where Python keeps the caller's order; see the
    /// [module documentation](self).
    pub tools: Option<Value>,
    /// Optional retrieval documents exposed as `documents`: a list of objects,
    /// conventionally with `title` and `text` keys (which also arrive sorted).
    pub documents: Option<Value>,
    /// Whether multimodal templates should number image and video placeholders.
    pub add_vision_id: bool,
    /// Optional reasoning-mode switch used by checkpoints that expose one.
    pub enable_thinking: Option<bool>,
    /// Fixed instant, in Unix seconds, that `strftime_now` formats (as UTC).
    ///
    /// `None` reads the host clock. Hugging Face formats host *local* time;
    /// this crate has no timezone database and always uses UTC. Fix the
    /// instant (or pass a template's own date variable, such as Llama's
    /// `date_string`, in [`Self::extra_context`]) for reproducible prompts.
    pub current_time_unix_seconds: Option<i64>,
    /// Additional checkpoint-specific variables; reserved built-ins cannot be replaced.
    pub extra_context: BTreeMap<String, Value>,
}

#[cfg(feature = "chat-templates")]
impl ChatTemplateOptions {
    /// Whether tools were supplied, as Python's `tools is not None`.
    ///
    /// JSON `null` counts as absent: Python callers cannot pass it, and the
    /// template sees `tools` as `none` either way.
    fn tools_given(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_null())
    }
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
        if self.valid_tokens.len() != batch {
            return Err(Error::Tokenizer(format!(
                "batch has {batch} token rows but {} mask rows",
                self.valid_tokens.len()
            )));
        }
        if let Some(row) = first_ragged_row(&self.token_ids, &self.valid_tokens, sequence) {
            return Err(Error::Tokenizer(format!(
                "batch row {row} is not rectangular: expected {sequence} tokens, found {} tokens and {} mask entries",
                self.token_ids[row].len(),
                self.valid_tokens[row].len()
            )));
        }
        // Equal row counts plus `first_ragged_row`'s verified `None` (every one
        // of the `batch` pairs has width `sequence`) establish the flatteners'
        // preconditions; `batch * sequence` fits `usize` because those rows
        // already hold that many entries. Verified row-major layouts: entry
        // `r * sequence + c` is row `r`, column `c`.
        let ids = flatten_token_rows(&self.token_ids, sequence);
        let valid = flatten_mask_rows(&self.valid_tokens, sequence);
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

/// Incremental detokenizer for text streamed one generated token at a time.
///
/// Byte-level BPE vocabularies (GPT-2, Qwen, Llama 3) split multi-byte UTF-8
/// characters such as CJK text and emoji across several tokens. Decoding each
/// token alone therefore yields U+FFFD fragments, while re-decoding the whole
/// prefix after every step is correct but costs `O(T^2)` for `T` tokens. This
/// decoder keeps only the short window of IDs needed to finish the current
/// character (the `tokenizers` crate's `DecodeStream`) and returns each piece
/// of text exactly once, as soon as it is complete. Concatenating the pieces
/// reproduces [`LocalTokenizer::decode`] of the whole sequence whenever that
/// sequence ends on a character boundary.
///
/// A decoded piece ending in a literal U+FFFD is indistinguishable from an
/// unfinished character, so it is withheld until a later token completes more
/// text; such text is rare outside corrupted inputs.
pub struct TokenStreamDecoder<'a> {
    /// Pending-ID window and the text already emitted from it.
    stream: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl fmt::Debug for TokenStreamDecoder<'_> {
    /// Identifies the decoder without dumping the borrowed tokenizer.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenStreamDecoder")
            .finish_non_exhaustive()
    }
}

impl TokenStreamDecoder<'_> {
    /// Feeds one generated ID and returns the text it completes, if any.
    ///
    /// `None` means the ID only extended an unfinished character (or decoded
    /// to nothing, like a skipped special token); its text arrives with a
    /// later step.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.stream.step(id).map_err(|error| {
            Error::Tokenizer(format!("failed to decode streamed token {id}: {error}"))
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
    ///
    /// The optional artifacts are `tokenizer_config.json`,
    /// `special_tokens_map.json`, and the chat-template files
    /// `chat_template.jinja` and `additional_chat_templates/*.jinja` (see
    /// [`TokenizerMetadata::chat_templates`]). Every file must resolve inside
    /// the canonical directory and stay within the loader's size limits.
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
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            Error::Tokenizer(format!(
                "failed to decode {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        clear_serialized_batch_policy(&mut tokenizer)?;
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
        // Verified: the largest ID fitting means every emitted ID fits.
        if !id_fits_vocabulary(maximum_id, model_vocabulary_size) {
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

    /// Returns how many template tokens the post-processor adds to one sequence.
    ///
    /// This mirrors Hugging Face's `num_special_tokens_to_add(pair=False)`: T5
    /// adds one (`</s>`), BERT adds two (`[CLS]`, `[SEP]`), and byte-level
    /// tokenizers such as GPT-2 and Qwen add none.
    pub fn num_special_tokens_to_add(&self) -> usize {
        self.tokenizer
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(false))
    }

    /// Encodes, truncates, and optionally pads text entirely from local state.
    ///
    /// Each row is tokenized as content, truncated to the content budget left
    /// by [`EncodeOptions::max_length`], and then passed through the
    /// checkpoint post-processor, which adds BOS/EOS-style template tokens
    /// when [`EncodeOptions::add_special_tokens`] is set. Padding is applied
    /// last and never truncates: a row wider than a fixed padding width fails.
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
        let content_budget = self.content_budget(options)?;
        let direction = match options
            .truncation_side
            .unwrap_or(self.metadata.truncation_side)
        {
            TruncationSide::Left => TruncationDirection::Left,
            TruncationSide::Right => TruncationDirection::Right,
        };
        let rows = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                self.encode_row(
                    index,
                    text.as_ref(),
                    options.add_special_tokens,
                    content_budget,
                    direction,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.pad_rows(rows, options)
    }

    /// Returns the per-row content budget implied by `max_length`, if any.
    fn content_budget(&self, options: &EncodeOptions) -> Result<Option<usize>> {
        let Some(max_length) = options.max_length else {
            return Ok(None);
        };
        let reserved = if options.add_special_tokens {
            self.num_special_tokens_to_add()
        } else {
            0
        };
        content_token_budget(max_length, reserved)
            .map(Some)
            .ok_or_else(|| {
                Error::Tokenizer(format!(
                    "max_length {max_length} cannot hold the {reserved} special tokens added by the tokenizer post-processor"
                ))
            })
    }

    /// Encodes one row: content, then optional truncation, then the special-token template.
    fn encode_row(
        &self,
        index: usize,
        text: &str,
        add_special_tokens: bool,
        content_budget: Option<usize>,
        direction: TruncationDirection,
    ) -> Result<Vec<u32>> {
        let row_error = |error: tokenizers::Error| {
            Error::Tokenizer(format!("failed to encode batch row {index}: {error}"))
        };
        let mut content = self.tokenizer.encode(text, false).map_err(row_error)?;
        if let Some(budget) = content_budget
            && content.len() > budget
        {
            content.truncate(budget, 0, direction);
            // The discarded windows are never returned; dropping them here
            // keeps the post-processor from templating each one.
            drop(content.take_overflowing());
        }
        let encoding = self
            .tokenizer
            .post_process(content, None, add_special_tokens)
            .map_err(row_error)?;
        if encoding.is_empty() {
            return Err(Error::Tokenizer(format!(
                "batch row {index} encoded to no tokens"
            )));
        }
        Ok(encoding.get_ids().to_vec())
    }

    /// Pads encoded rows to a rectangle and builds their `true == valid` masks.
    fn pad_rows(&self, rows: Vec<Vec<u32>>, options: &EncodeOptions) -> Result<BatchEncoding> {
        let target = match options.padding {
            Padding::None => None,
            Padding::Longest => longest_row(&rows),
            Padding::Fixed(width) => Some(width),
        };
        let Some(target) = target else {
            let valid_tokens = unpadded_masks(&rows);
            return Ok(BatchEncoding {
                token_ids: rows,
                valid_tokens,
            });
        };
        let pad_id = self
            .metadata
            .special_tokens
            .pad
            .as_ref()
            .ok_or_else(|| {
                Error::Tokenizer("padding requested but no pad token is declared".into())
            })?
            .id;
        let pad_left =
            options.padding_side.unwrap_or(self.metadata.padding_side) == PaddingSide::Left;
        // Verified: every row is padded exactly, or the first overlong row is named.
        let PaddedBatch {
            token_ids,
            valid_tokens,
        } = pad_batch(&rows, target, pad_id, pad_left).map_err(|index| {
            Error::Tokenizer(format!(
                "encoded row {index} has length {}, exceeding padding width {target}; set max_length to request truncation",
                rows[index].len()
            ))
        })?;
        Ok(BatchEncoding {
            token_ids,
            valid_tokens,
        })
    }

    /// Decodes vocabulary IDs using the tokenizer's configured decoder.
    ///
    /// For text produced one token at a time, prefer [`Self::decode_stream`].
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|error| Error::Tokenizer(format!("failed to decode token IDs: {error}")))
    }

    /// Starts an incremental decoder for streamed generation.
    ///
    /// Feed each generated ID to [`TokenStreamDecoder::step`] and print the
    /// returned pieces; multi-byte characters split across tokens are emitted
    /// whole, without re-decoding the full prefix at every step.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> TokenStreamDecoder<'_> {
        TokenStreamDecoder {
            stream: self.tokenizer.decode_stream(skip_special_tokens),
        }
    }

    /// Renders one bounded checkpoint chat template without filesystem access.
    ///
    /// The template sees `messages`, `tools`, `documents`,
    /// `add_generation_prompt`, every declared special token (`bos_token`,
    /// `eos_token`, ...), the optional switches in `options`, the globals
    /// `raise_exception` and `strftime_now`, and `options.extra_context`.
    ///
    /// The output equals Hugging Face's `apply_chat_template(...,
    /// tokenize=False)` except for the differences listed in the
    /// [module documentation](self); most importantly, dictionary keys reach
    /// the template sorted rather than in insertion order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] for invalid arguments, a template that
    /// fails to compile or render (including `raise_exception`), and a
    /// template that exceeds a resource limit of the sandboxed renderer (see
    /// the module documentation); a panic inside the template engine is
    /// reported the same way instead of unwinding.
    #[cfg(feature = "chat-templates")]
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(Error::Tokenizer(
                "chat rendering requires at least one message".into(),
            ));
        }
        if options.continue_final_message && options.add_generation_prompt {
            return Err(Error::Tokenizer(
                "continue_final_message and add_generation_prompt are mutually exclusive: the first extends the final message, the second starts a new assistant turn".into(),
            ));
        }
        let (name, template) = select_template(
            &self.metadata.chat_templates,
            options.template_name.as_deref(),
            options.tools_given(),
        )?;
        let mut messages = chat_messages_value(messages)?;
        let continued_text = if options.continue_final_message {
            Some(mark_final_message(&mut messages)?)
        } else {
            None
        };
        let context = self.chat_context(messages, options)?;
        let rendered = chat_template::render_template(
            template,
            &Value::Object(context),
            options.current_time_unix_seconds,
        )
        .map_err(|error| Error::Tokenizer(format!("chat template `{name}` failed: {error}")))?;
        match continued_text {
            Some(final_text) => cut_after_final_message(rendered, &final_text),
            None => Ok(rendered),
        }
    }

    /// Assembles the template variables Hugging Face passes to a chat template.
    #[cfg(feature = "chat-templates")]
    fn chat_context(
        &self,
        messages: Value,
        options: &ChatTemplateOptions,
    ) -> Result<Map<String, Value>> {
        if let Some(documents) = &options.documents
            && !documents
                .as_array()
                .is_some_and(|documents| documents.iter().all(Value::is_object))
        {
            return Err(Error::Tokenizer(
                "chat documents must be a list of objects, conventionally with `title` and `text`"
                    .into(),
            ));
        }
        let mut context = Map::new();
        context.insert("messages".into(), messages);
        context.insert("tools".into(), options.tools.clone().unwrap_or(Value::Null));
        context.insert(
            "documents".into(),
            options.documents.clone().unwrap_or(Value::Null),
        );
        context.insert(
            "add_generation_prompt".into(),
            Value::Bool(options.add_generation_prompt),
        );
        if let Some(enable_thinking) = options.enable_thinking {
            context.insert("enable_thinking".into(), Value::Bool(enable_thinking));
        }
        if options.add_vision_id {
            context.insert("add_vision_id".into(), Value::Bool(true));
        }
        for (name, token) in self.metadata.special_tokens.named() {
            context.insert(name.into(), Value::String(token.content.clone()));
        }
        for (name, value) in &options.extra_context {
            if RESERVED_CHAT_VARIABLES.contains(&name.as_str()) {
                return Err(Error::Tokenizer(format!(
                    "extra chat context cannot replace reserved variable `{name}`"
                )));
            }
            context.insert(name.clone(), value.clone());
        }
        Ok(context)
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

/// Clears truncation and padding serialized inside `tokenizer.json`.
///
/// Hugging Face `transformers` resets backend truncation and padding on every
/// call that does not request them, so a serialized `"truncation"` or
/// `"padding"` block never applies implicitly there. Leaving them active here
/// would truncate rows silently and append pad IDs that [`BatchEncoding`]
/// masks would report as real tokens.
fn clear_serialized_batch_policy(tokenizer: &mut Tokenizer) -> Result<()> {
    tokenizer.with_truncation(None).map_err(|error| {
        Error::Tokenizer(format!("failed to clear serialized truncation: {error}"))
    })?;
    tokenizer.with_padding(None);
    Ok(())
}

/// Whether nothing, not even a dangling symlink, exists at `path`.
///
/// One `lstat` settles the common case of an absent optional artifact, which
/// `canonicalize` would decide only after resolving every path component.
/// Dangling and looping symlinks exist here; resolving them reports them.
fn is_absent(path: &Path) -> bool {
    fs::symlink_metadata(path).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}

/// Resolves an artifact path relative to `root` and ensures symlinks cannot escape it.
///
/// Returns `None` only for an absent optional artifact; the resolved file
/// must be a regular file of at most `maximum_bytes`.
fn contained_file(
    root: &Path,
    name: &str,
    required: bool,
    maximum_bytes: u64,
) -> Result<Option<PathBuf>> {
    let candidate = root.join(name);
    if !required && is_absent(&candidate) {
        return Ok(None);
    }
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
        sep: resolve_special("sep_token", tokenizer_config, special_map, tokenizer)?,
        cls: resolve_special("cls_token", tokenizer_config, special_map, tokenizer)?,
        mask: resolve_special("mask_token", tokenizer_config, special_map, tokenizer)?,
        additional: resolve_additional(tokenizer_config, special_map, tokenizer)?,
    };
    let chat_templates = load_chat_templates(root, get("chat_template"))?;
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
            value.as_u64().and_then(declared_special_id).ok_or_else(|| {
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
    // Verified: an accepted ID is always the vocabulary's own.
    let id = reconcile_special_id(declared_id, id).map_err(|declared| {
        Error::Tokenizer(format!(
            "{field}_id {declared} conflicts with tokenizer vocabulary ID {id}"
        ))
    })?;
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

/// Serializes messages as template dictionaries, rejecting shadowed field names.
#[cfg(feature = "chat-templates")]
fn chat_messages_value(messages: &[ChatMessage]) -> Result<Value> {
    for (index, message) in messages.iter().enumerate() {
        if let Some(key) = message
            .extra
            .keys()
            .find(|key| CHAT_MESSAGE_FIELDS.contains(&key.as_str()))
        {
            return Err(Error::Tokenizer(format!(
                "chat message {index} extra key `{key}` would replace the named field of the same name"
            )));
        }
    }
    serde_json::to_value(messages)
        .map_err(|error| Error::Tokenizer(format!("failed to serialize chat messages: {error}")))
}

/// Appends [`CONTINUE_FINAL_MESSAGE_TAG`] to the final message's text and returns that text.
///
/// For structured content, the last block with a string `text` field is
/// marked, as in Hugging Face.
#[cfg(feature = "chat-templates")]
fn mark_final_message(messages: &mut Value) -> Result<String> {
    let missing_text = || {
        Error::Tokenizer("continue_final_message requires the final message to contain text".into())
    };
    let content = messages
        .as_array_mut()
        .and_then(|messages| messages.last_mut())
        .and_then(|message| message.get_mut("content"))
        .ok_or_else(missing_text)?;
    let text = match content {
        Value::String(text) => text,
        Value::Array(blocks) => blocks
            .iter_mut()
            .rev()
            .find_map(|block| match block.get_mut("text") {
                Some(Value::String(text)) => Some(text),
                _ => None,
            })
            .ok_or_else(missing_text)?,
        _ => return Err(missing_text()),
    };
    let original = text.clone();
    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
    Ok(original)
}

/// Cuts a rendered prompt right after the final message, as Hugging Face does.
///
/// If the template trimmed the whitespace following the marker, trailing
/// whitespace is trimmed from the prompt too.
#[cfg(feature = "chat-templates")]
fn cut_after_final_message(mut rendered: String, final_text: &str) -> Result<String> {
    let marker = CONTINUE_FINAL_MESSAGE_TAG.trim_end();
    let Some(location) = rendered.rfind(marker) else {
        return Err(Error::Tokenizer(
            "continue_final_message is set but the template dropped the final message".into(),
        ));
    };
    if !rendered.contains(final_text.trim()) {
        return Err(Error::Tokenizer(
            "continue_final_message is set but the template altered the final message".into(),
        ));
    }
    let preserved_spacing = rendered[location..].starts_with(CONTINUE_FINAL_MESSAGE_TAG);
    rendered.truncate(location);
    if !preserved_spacing {
        rendered.truncate(rendered.trim_end().len());
    }
    Ok(rendered)
}

#[cfg(test)]
#[path = "tokenization_tests.rs"]
mod tests;
