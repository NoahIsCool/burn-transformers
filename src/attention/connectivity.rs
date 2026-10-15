//! Per-layer key visibility: global, sliding-window, and chunked-local attention.
//!
//! Local attention restricts which keys a query may see by their distance in
//! *token positions*. Positions are bottom-right aligned (query row `t` of `Q`
//! sits at key position `t + K - Q`, so cached prefixes are handled) and, with
//! a token-valid mask, count only real tokens: `ordinal = cumsum(valid) - 1`.
//! Interior padding therefore consumes no window capacity, matching caches
//! that compact each row before appending it. Causal order is unchanged for
//! valid pairs because ordinals increase strictly across real tokens.
//!
//! With query position `p` and key position `j`:
//!
//! | rule | excluded when |
//! | --- | --- |
//! | causal | `j > p` |
//! | sliding window `W` | `j + W <= p` (keeps the `W` newest positions `p - W < j <= p`) |
//! | chunked local `C` | `floor(j / C) != floor(p / C)` |
//!
//! Sliding windows bound only the past; combine them with `causal` for the
//! usual decoder window (Mistral, Gemma 2/3/4, gpt-oss, OLMo 3). Chunked local
//! attention is Llama 4's iRoPE `chunked_attention`. Positions are integer
//! vectors that broadcast in comparisons, so the only materialized plane is
//! one Boolean `[batch or 1, 1, Q, K]` tensor expanded (not copied) across
//! heads; no `[batch, head, Q, K]` integer plane is built.
//!
//! # Absolute positions and integer range
//!
//! Causality and sliding windows depend only on position *differences*, so
//! they are unaffected by where the key timeline starts. Chunk boundaries are
//! absolute: when a cache has dropped earlier keys, [`MaskGeometry::key_start`]
//! gives the absolute position of the first key, and [`chunk_frame`] turns
//! `(chunk, key_start)` into an equivalent local frame. Tensor arithmetic uses
//! only local positions below `K`: windows of at least `K` positions and
//! chunks covering the whole span exclude nothing and are dropped, and the
//! remaining parameters are below `K`. Every value therefore stays below
//! `2 * K` and is checked against the backend's integer dtype (often `i32`)
//! instead of wrapping silently.

use core::{fmt, ops::Range};

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use super::{
    mask::{
        aligned_query_positions, audit_plane, check_token_mask, ensure_positions_fit, expand_to,
        key_positions, signed_length, union_exclusions,
    },
    plan::chunk_frame,
};
use crate::{Error, Result, architecture::TokenMask};

/// Which keys a query may attend to, measured in token positions.
///
/// Families with per-layer local/global schedules choose one rule per layer
/// and build masks with [`Self::exclusion_mask`] instead of re-implementing
/// window algebra. [`super::sliding_window_key_range`] gives the matching key
/// slice for bounded chunked execution.
///
/// ```
/// use burn::backend::Flex;
/// use transformers::attention::{AttentionConnectivity, MaskGeometry};
///
/// // A causal two-token window over three tokens (`true` excludes a link).
/// let mask = AttentionConnectivity::SlidingWindow { window: 2 }
///     .exclusion_mask::<Flex>(true, MaskGeometry::new(1, 1, 3, 3), None, None, &Default::default())
///     .expect("valid geometry")
///     .expect("causal windows exclude links");
/// assert_eq!(
///     mask.into_data().to_vec::<bool>().unwrap(),
///     [false, true, true, false, false, true, true, false, false],
/// );
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionConnectivity {
    /// Every key is visible, subject only to causality and padding.
    #[default]
    Global,
    /// Only the `window` newest positions up to the query are visible.
    ///
    /// Query position `p` sees keys `p - window < j`, matching Hugging Face's
    /// `sliding_window` convention. A window of one sees only the query token.
    SlidingWindow {
        /// Number of positions visible to each query, including itself.
        window: usize,
    },
    /// Positions attend only within their aligned chunk of `chunk` tokens.
    ///
    /// Query position `p` sees keys `[C * floor(p / C), C * floor(p / C) + C)`.
    /// Positions are absolute: chunks align to position zero of the whole
    /// sequence, so a cache that dropped earlier keys must report their count
    /// through [`MaskGeometry::key_start`].
    ChunkedLocal {
        /// Width of each independent attention chunk.
        chunk: usize,
    },
}

impl AttentionConnectivity {
    /// Rejects zero-width windows and chunks, which would exclude every key.
    pub fn validate(self) -> Result<()> {
        match self {
            Self::SlidingWindow { window: 0 } => Err(Error::configuration(
                "attention.connectivity.window",
                "must be greater than zero",
            )),
            Self::ChunkedLocal { chunk: 0 } => Err(Error::configuration(
                "attention.connectivity.chunk",
                "must be greater than zero",
            )),
            Self::Global | Self::SlidingWindow { .. } | Self::ChunkedLocal { .. } => Ok(()),
        }
    }

    /// Scalar oracle: whether `key_position` is excluded for `query_position`.
    ///
    /// This is the exact rule the tensor builder vectorizes. Zero-sized
    /// windows and chunks exclude everything, mirroring [`Self::validate`].
    #[must_use]
    pub const fn excludes(self, causal: bool, query_position: usize, key_position: usize) -> bool {
        if causal && key_position > query_position {
            return true;
        }
        match self {
            Self::Global => false,
            // `key + window <= query`, compared as a gap to avoid overflow.
            Self::SlidingWindow { window } => {
                window == 0
                    || (query_position >= key_position && query_position - key_position >= window)
            }
            Self::ChunkedLocal { chunk } => {
                chunk == 0 || query_position / chunk != key_position / chunk
            }
        }
    }

    /// Builds padding, causal, and connectivity exclusions for one attention call.
    ///
    /// Returns `[batch, heads, query, key]` (an expanded view of one
    /// `[batch or 1, 1, query, key]` Boolean plane), or `None` when nothing is
    /// excluded. Queries are the bottom-right-aligned suffix of the key
    /// timeline. With a `key_mask`, positions are each row's valid-token
    /// ordinals; otherwise they are column indices. Both are offset by
    /// [`MaskGeometry::key_start`], which only chunked-local rules observe.
    /// `query_mask` and `key_mask` use `true == valid`, must live on `device`,
    /// and exclude invalid query rows and key columns.
    ///
    /// Local rules and ordinal positions require `key >= query`; only unpadded
    /// global causal attention may have more queries than keys (its leading
    /// rows then see nothing). Lengths whose position arithmetic exceeds the
    /// backend integer dtype are rejected rather than wrapped.
    pub fn exclusion_mask<B: Backend>(
        self,
        causal: bool,
        geometry: MaskGeometry,
        query_mask: Option<&TokenMask<B>>,
        key_mask: Option<&TokenMask<B>>,
        device: &B::Device,
    ) -> Result<Option<Tensor<B, 4, Bool>>> {
        ExclusionPlan::new(self, causal, geometry, query_mask, key_mask, device)?.chunk(
            0..geometry.query_sequence,
            0..geometry.key_sequence,
            geometry.heads,
        )
    }
}

/// Precomputed positions and validity from which any query/key chunk's mask is sliced.
///
/// Building positions once (`O(batch * key)` integers) lets bounded execution
/// materialize only one chunk's `[batch, 1, rows, keys]` Boolean plane at a time.
pub(crate) struct ExclusionPlan<B: Backend> {
    /// Visibility rule over local positions.
    rule: PositionRule,
    /// Whether keys after the query position are excluded.
    causal: bool,
    /// Batch rows.
    batch: usize,
    /// Optional `[batch, Q]` query validity (`true == valid`).
    query_valid: Option<Tensor<B, 2, Bool>>,
    /// Optional `[batch, K]` key validity (`true == valid`).
    key_valid: Option<Tensor<B, 2, Bool>>,
    /// Query `[batch or 1, Q]` and key `[batch or 1, K]` positions, when a rule needs them.
    positions: Option<(Tensor<B, 2, Int>, Tensor<B, 2, Int>)>,
}

impl<B: Backend> ExclusionPlan<B> {
    /// Validates the rule, extents, alignment, and token masks, and derives positions.
    ///
    /// Token masks are validated from metadata only; no `[query, key]` plane
    /// is built until [`Self::chunk`] requests one chunk.
    pub(crate) fn new(
        connectivity: AttentionConnectivity,
        causal: bool,
        geometry: MaskGeometry,
        query_mask: Option<&TokenMask<B>>,
        key_mask: Option<&TokenMask<B>>,
        device: &B::Device,
    ) -> Result<Self> {
        connectivity.validate()?;
        geometry.validate()?;
        let MaskGeometry {
            batch,
            query_sequence,
            key_sequence,
            ..
        } = geometry;
        check_token_masks(
            [
                (query_mask, "query_mask", query_sequence),
                (key_mask, "key_mask", key_sequence),
            ],
            batch,
            device,
        )?;
        let global = connectivity == AttentionConnectivity::Global;
        if key_sequence < query_sequence && (!global || (causal && key_mask.is_some())) {
            return Err(Error::shape(
                "attention.connectivity",
                "key sequence at least as long as query sequence",
                format!("query={query_sequence}, key={key_sequence}"),
            ));
        }
        let rule = PositionRule::new(connectivity, geometry)?;
        let positions = if causal || rule != PositionRule::Unrestricted {
            ensure_positions_fit::<B>(device, rule.peak(key_sequence)?)?;
            Some(match key_mask {
                Some(mask) => {
                    let ordinals = valid_token_ordinals(mask);
                    let queries =
                        ordinals
                            .clone()
                            .narrow(1, key_sequence - query_sequence, query_sequence);
                    (queries, ordinals)
                }
                None => (
                    aligned_query_positions::<B>(query_sequence, key_sequence, device)?,
                    key_positions::<B>(key_sequence, device)?,
                ),
            })
        } else {
            None
        };
        Ok(Self {
            rule,
            causal,
            batch,
            query_valid: query_mask.map(|mask| mask.as_tensor().clone()),
            key_valid: key_mask.map(|mask| mask.as_tensor().clone()),
            positions,
        })
    }

    /// Builds the `[batch, heads, rows, keys]` exclusions of one query/key chunk.
    ///
    /// Ranges index the full query and key axes and must lie inside them.
    pub(crate) fn chunk(
        &self,
        query_range: Range<usize>,
        key_range: Range<usize>,
        heads: usize,
    ) -> Result<Option<Tensor<B, 4, Bool>>> {
        let batch = self.batch;
        let rows = query_range.len();
        let keys = key_range.len();
        let plane = [batch, 1, rows, keys];
        let query_invalid = self.query_valid.as_ref().map(|valid| {
            valid
                .clone()
                .slice([0..batch, query_range.clone()])
                .bool_not()
                .reshape([batch, 1, rows, 1])
                .expand(plane)
        });
        let key_invalid = self.key_valid.as_ref().map(|valid| {
            valid
                .clone()
                .slice([0..batch, key_range.clone()])
                .bool_not()
                .reshape([batch, 1, 1, keys])
                .expand(plane)
        });
        let positional = self.positions.as_ref().and_then(|(queries, all_keys)| {
            positional_exclusion(
                self.rule,
                self.causal,
                queries.clone().narrow(1, query_range.start, rows),
                all_keys.clone().narrow(1, key_range.start, keys),
            )
        });
        let padding = union_exclusions(query_invalid, key_invalid);
        Ok(union_exclusions(padding, positional)
            .map(|exclusion| exclusion.expand([batch, heads, rows, keys])))
    }
}

impl fmt::Display for AttentionConnectivity {
    /// Formats the rule compactly for diagnostics, e.g. `sliding_window(4096)`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Global => formatter.write_str("global"),
            Self::SlidingWindow { window } => write!(formatter, "sliding_window({window})"),
            Self::ChunkedLocal { chunk } => write!(formatter, "chunked_local({chunk})"),
        }
    }
}

/// Extents of one `[batch, head, query, key]` exclusion mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskGeometry {
    /// Batch rows.
    pub batch: usize,
    /// Attention heads sharing the mask.
    pub heads: usize,
    /// Query tokens, the suffix of the key timeline.
    pub query_sequence: usize,
    /// Key tokens, including any cached prefix.
    pub key_sequence: usize,
    /// Absolute position of the first key (zero unless a cache dropped earlier keys).
    ///
    /// Only chunked-local rules depend on it. With a key mask it counts the
    /// valid tokens dropped before the first key and must be the same for
    /// every row.
    pub key_start: usize,
}

impl MaskGeometry {
    /// Describes one attention call's score-plane extents, with keys starting at position zero.
    #[must_use]
    pub const fn new(
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence: usize,
    ) -> Self {
        Self {
            batch,
            heads,
            query_sequence,
            key_sequence,
            key_start: 0,
        }
    }

    /// Sets the absolute position of the first key, e.g. the tokens a ring cache evicted.
    #[must_use]
    pub const fn with_key_start(mut self, key_start: usize) -> Self {
        self.key_start = key_start;
        self
    }

    /// Rejects empty axes and lengths that do not fit position arithmetic.
    fn validate(self) -> Result<()> {
        if self.batch == 0 || self.heads == 0 || self.query_sequence == 0 || self.key_sequence == 0
        {
            return Err(Error::configuration(
                "attention.mask_shape",
                "batch, heads, query sequence, and key sequence must be non-zero",
            ));
        }
        signed_length(self.query_sequence, "query length exceeds i64")?;
        signed_length(self.key_sequence, "key length exceeds i64")?;
        if self.key_start.checked_add(self.key_sequence).is_none() {
            return Err(Error::configuration(
                "attention.mask_shape.key_start",
                "absolute key positions must fit usize",
            ));
        }
        Ok(())
    }
}

/// Connectivity rewritten for local positions in `[0, K)` with small, exact parameters.
///
/// Windows of at least `K` positions, and chunks whose span holds no boundary,
/// exclude nothing and become [`Self::Unrestricted`]; every remaining parameter
/// is below `K` (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PositionRule {
    /// No connectivity exclusion.
    Unrestricted,
    /// Exclude `key + window <= query`.
    Window(
        /// Visible positions per query, `1 <= window < K`.
        i64,
    ),
    /// Exclude links whose `floor((position + phase) / width)` differ.
    Chunk {
        /// Frame width, `1 <= width <= K`.
        width: i64,
        /// Frame phase, `0 <= phase < width`.
        phase: i64,
    },
}

impl PositionRule {
    /// Reduces a validated rule over a validated geometry (so `K` fits `i64`).
    fn new(connectivity: AttentionConnectivity, geometry: MaskGeometry) -> Result<Self> {
        let span = geometry.key_sequence;
        Ok(match connectivity {
            AttentionConnectivity::Global => Self::Unrestricted,
            // No local gap reaches `K`, so a window that wide excludes nothing.
            AttentionConnectivity::SlidingWindow { window } if window >= span => Self::Unrestricted,
            AttentionConnectivity::SlidingWindow { window } => {
                Self::Window(signed_length(window, "sliding window exceeds i64")?)
            }
            AttentionConnectivity::ChunkedLocal { chunk } => {
                match chunk_frame(chunk, geometry.key_start, span) {
                    None => Self::Unrestricted,
                    Some((width, phase)) => Self::Chunk {
                        width: signed_length(width, "chunk frame exceeds i64")?,
                        phase: signed_length(phase, "chunk frame exceeds i64")?,
                    },
                }
            }
        })
    }

    /// Largest integer the rule's tensor arithmetic forms from positions below `span`.
    fn peak(self, span: usize) -> Result<i64> {
        let last = signed_length(span, "key length exceeds i64")? - 1;
        // Both addends are below `span`, so the sum is below `2 * span`;
        // `checked_add` still guards spans beyond `i64::MAX / 2`.
        let extent = match self {
            Self::Unrestricted => 0,
            Self::Window(window) => window,
            Self::Chunk { phase, .. } => phase,
        };
        last.checked_add(extent).ok_or_else(|| {
            Error::configuration("attention.positions", "position arithmetic exceeds i64")
        })
    }
}

/// Validates optional `[batch, length]` token masks from metadata, including their device.
fn check_token_masks<B: Backend, const N: usize>(
    masks: [(Option<&TokenMask<B>>, &str, usize); N],
    batch: usize,
    device: &B::Device,
) -> Result<()> {
    for (mask, name, length) in masks {
        let Some(mask) = mask else { continue };
        check_token_mask(mask, name, batch, length)?;
        if mask.as_tensor().device() != *device {
            return Err(Error::configuration(
                name,
                "token mask must share the attention device",
            ));
        }
    }
    Ok(())
}

/// Returns valid-token ordinals `cumsum(valid) - 1` per row, with padding set to zero.
///
/// The value at a padded position is irrelevant because padding exclusions
/// remove it; zero keeps every ordinal non-negative for chunk division.
pub fn valid_token_ordinals<B: Backend>(mask: &TokenMask<B>) -> Tensor<B, 2, Int> {
    let valid = mask.as_tensor().clone();
    (valid.clone().int().cumsum(1) - 1).mask_fill(valid.bool_not(), 0)
}

/// Compares local query and key positions under causality and one reduced rule.
///
/// `query_positions` is `[batch or 1, Q]` and `key_positions` is
/// `[batch or 1, K]`; the result is one `[batch or 1, 1, Q, K]` plane, or `None`
/// when neither causality nor the rule excludes anything.
fn positional_exclusion<B: Backend>(
    rule: PositionRule,
    causal: bool,
    query_positions: Tensor<B, 2, Int>,
    key_positions: Tensor<B, 2, Int>,
) -> Option<Tensor<B, 4, Bool>> {
    let [query_batch, query_sequence] = query_positions.dims();
    let [key_batch, key_sequence] = key_positions.dims();
    let plane = [query_batch.max(key_batch), 1, query_sequence, key_sequence];
    // A position column against a position row: each comparison broadcasts to
    // one Boolean plane without materializing integer planes.
    let query = query_positions.reshape([query_batch, 1, query_sequence, 1]);
    let key = key_positions.reshape([key_batch, 1, 1, key_sequence]);
    let causal_exclusion = causal.then(|| {
        audit_plane(plane);
        key.clone().greater(query.clone())
    });
    let rule_exclusion = match rule {
        PositionRule::Unrestricted => None,
        PositionRule::Window(window) => {
            audit_plane(plane);
            Some((key + window).lower_equal(query))
        }
        PositionRule::Chunk { width, phase } => {
            // Positions and phase are non-negative, so truncating division is floor division.
            let chunk_index = |positions: Tensor<B, 4, Int>| {
                if phase == 0 {
                    positions.div_scalar(width)
                } else {
                    (positions + phase).div_scalar(width)
                }
            };
            audit_plane(plane);
            Some(chunk_index(key).not_equal(chunk_index(query)))
        }
    };
    union_exclusions(causal_exclusion, rule_exclusion).map(|mask| expand_to(mask, plane))
}

#[cfg(test)]
#[path = "connectivity_tests.rs"]
mod tests;
