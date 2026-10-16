//! Score-plane execution policies and their verified chunk plans.
//!
//! A score plane holds one element per `[batch, head, query, key]` link.
//! [`AttentionExecutionPolicy`] bounds every such plane by evaluating complete
//! query rows in chunks; [`AttentionExecutionPlan`] is the resulting immutable
//! schedule. The integer arithmetic lives in the dependency-free `plan` module
//! and is verified with Verus; this module adds the public types and maps
//! planning failures to contextual errors.

use core::ops::Range;

use burn::tensor::backend::Backend;

use super::plan::{self, ChunkLimits, ChunkPlanError, plan_query_chunks, query_chunk_range};
use crate::{Error, Result};

/// Controls the peak score-plane size of an attention operation.
///
/// A score plane contains one element for every
/// `[batch, head, query, key]` pair. Query chunking preserves the mathematical
/// attention rows because every chunk still sees its complete allowed key
/// range, while changing peak temporary storage from `O(query * key)` to
/// `O(query_chunk * key)`. Backends are synchronized between chunks so dropped
/// score, mask, and probability buffers can be reclaimed before the next chunk.
///
/// This policy bounds score-like intermediates, not parameters, Q/K/V storage,
/// the linear-size output, or an autodiff tape. On an autodiff backend the
/// tape retains every chunk's workspace anyway, so chunking cannot lower the
/// training peak. Autodiff backends therefore evaluate one [`Self::Unbounded`]
/// call with the same values. [`Self::effective_on`] states that rule, and
/// [`crate::attention::GroupedQueryAttention::attend_bounded`] applies it
/// directly; the Muse Glimmer and MLA attention paths apply the same rule
/// through their own autodiff checks, so every consumer of this policy shares
/// one contract. Size training memory for the full `O(query * key)` plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionExecutionPolicy {
    /// Evaluate all query rows together without a score-plane bound.
    ///
    /// This is useful for small golden-oracle calls and explicitly provisioned
    /// training runs. Its peak attention workspace is quadratic for self
    /// attention.
    Unbounded,
    /// Split query rows while retaining the complete key support of every row.
    QueryChunked {
        /// Hard upper bound on query rows submitted in one attention call.
        max_query_tokens: usize,
        /// Hard upper bound on elements in any score-like chunk plane.
        max_score_elements: usize,
    },
}

impl Default for AttentionExecutionPolicy {
    /// Uses a conservative production bound suitable for 32K Muse inference.
    fn default() -> Self {
        Self::QueryChunked {
            max_query_tokens: 512,
            max_score_elements: 1 << 26,
        }
    }
}

impl AttentionExecutionPolicy {
    /// Constructs a synchronized query-chunked policy.
    #[must_use]
    pub const fn query_chunked(max_query_tokens: usize, max_score_elements: usize) -> Self {
        Self::QueryChunked {
            max_query_tokens,
            max_score_elements,
        }
    }

    /// Returns the policy that a call on `device` actually executes.
    ///
    /// Autodiff backends answer [`Self::Unbounded`]: their tape keeps every
    /// chunk's score, mask, and probability planes alive until backward, so
    /// query chunking would add synchronizations without lowering the peak.
    /// Chunked and unbounded evaluation compute the same rows, so the values
    /// are unchanged. Every other backend executes `self` as given. Call
    /// [`Self::validate`] first: this mapping does not check limits, so an
    /// invalid policy must not be silently replaced.
    #[must_use]
    pub fn effective_on<B: Backend>(self, device: &B::Device) -> Self {
        if B::ad_enabled(device) {
            Self::Unbounded
        } else {
            self
        }
    }

    /// Validates non-zero chunk and workspace limits.
    pub fn validate(self) -> Result<()> {
        if let Self::QueryChunked {
            max_query_tokens,
            max_score_elements,
        } = self
        {
            if max_query_tokens == 0 {
                return Err(Error::configuration(
                    "attention.execution.max_query_tokens",
                    "must be greater than zero",
                ));
            }
            if max_score_elements == 0 {
                return Err(Error::configuration(
                    "attention.execution.max_score_elements",
                    "must be greater than zero",
                ));
            }
        }
        Ok(())
    }

    /// Plans query chunks for a conservative upper bound on keys per chunk.
    ///
    /// `key_sequence_bound` may be smaller than the physical key tensor when a
    /// caller can prove that each chunk slices a local window first (see
    /// [`sliding_window_key_range`]). The caller remains responsible for
    /// checking every concrete key slice against the resulting score-element
    /// limit. The integer planning is verified with Verus (`verus/attention.rs`).
    pub fn plan(
        self,
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence_bound: usize,
    ) -> Result<AttentionExecutionPlan> {
        let limits = match self {
            Self::Unbounded => None,
            Self::QueryChunked {
                max_query_tokens,
                max_score_elements,
            } => Some(ChunkLimits {
                max_query_tokens,
                max_score_elements,
            }),
        };
        let geometry = plan_query_chunks(batch, heads, query_sequence, key_sequence_bound, limits)
            .map_err(chunk_plan_error)?;
        Ok(AttentionExecutionPlan {
            query_sequence,
            key_sequence_bound,
            query_chunk_size: geometry.query_chunk_size,
            chunk_count: geometry.chunk_count,
            peak_score_elements: geometry.peak_score_elements,
            score_element_limit: limits.map(|limits| limits.max_score_elements),
        })
    }

    /// Returns the configured maximum query rows before score-bound reduction.
    ///
    /// This is `min(max_query_tokens, query_sequence)` (verified with Verus);
    /// every planned chunk holds at most this many rows.
    #[must_use]
    pub const fn maximum_query_tokens(self, query_sequence: usize) -> usize {
        match self {
            Self::Unbounded => query_sequence,
            Self::QueryChunked {
                max_query_tokens, ..
            } => plan::capped_query_rows(max_query_tokens, query_sequence),
        }
    }

    /// Checks that one concrete chunk stays within this policy's score bound.
    pub fn validate_chunk(
        self,
        batch: usize,
        heads: usize,
        query_sequence: usize,
        key_sequence: usize,
    ) -> Result<()> {
        let plan = self.plan(batch, heads, query_sequence, key_sequence)?;
        if plan.chunk_count != 1 {
            return Err(Error::BackendCapability(format!(
                "attention chunk [{batch}, {heads}, {query_sequence}, {key_sequence}] exceeds the configured single-chunk score bound"
            )));
        }
        Ok(())
    }
}

/// Immutable allocation plan produced by [`AttentionExecutionPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionExecutionPlan {
    /// Complete query length covered by this plan.
    query_sequence: usize,
    /// Conservative maximum key length permitted for each planned chunk.
    key_sequence_bound: usize,
    /// Query rows evaluated in each chunk except possibly the final remainder.
    query_chunk_size: usize,
    /// Number of sequential attention calls.
    chunk_count: usize,
    /// Maximum score elements produced by the planned shape.
    peak_score_elements: usize,
    /// Configured bound, or `None` for unbounded execution.
    score_element_limit: Option<usize>,
}

impl AttentionExecutionPlan {
    /// Returns the complete query length covered by this plan.
    #[must_use]
    pub const fn query_sequence(self) -> usize {
        self.query_sequence
    }

    /// Returns the conservative maximum key length permitted for each chunk.
    #[must_use]
    pub const fn key_sequence_bound(self) -> usize {
        self.key_sequence_bound
    }

    /// Returns query rows evaluated per chunk.
    #[must_use]
    pub const fn query_chunk_size(self) -> usize {
        self.query_chunk_size
    }

    /// Returns the number of sequential chunks.
    #[must_use]
    pub const fn chunk_count(self) -> usize {
        self.chunk_count
    }

    /// Returns the maximum number of elements in one score-like plane.
    #[must_use]
    pub const fn peak_score_elements(self) -> usize {
        self.peak_score_elements
    }

    /// Returns the configured score-element limit when execution is bounded.
    #[must_use]
    pub const fn score_element_limit(self) -> Option<usize> {
        self.score_element_limit
    }

    /// Returns the query rows of chunk `index`, or `None` past the last chunk.
    ///
    /// Chunk `i` covers `[i * c, min((i + 1) * c, Q))`; consecutive chunks
    /// share their boundary and together partition `0..Q` (verified with Verus).
    #[must_use]
    pub fn query_range(self, index: usize) -> Option<Range<usize>> {
        query_chunk_range(
            self.query_chunk_size,
            self.chunk_count,
            self.query_sequence,
            index,
        )
        .map(|(start, end)| start..end)
    }

    /// Iterates every chunk's query rows in order.
    pub fn query_ranges(self) -> impl Iterator<Item = Range<usize>> {
        (0..self.chunk_count).filter_map(move |index| self.query_range(index))
    }
}

/// Returns the keys visible to aligned query rows under a causal sliding window.
///
/// Queries are the bottom-right suffix of the keys (`query_sequence <=
/// key_sequence`), so row `t` sits at key position `t + K - Q` and sees keys
/// `(t + K - Q - window, t + K - Q]`. The result is the exact union over
/// `query_rows`, `[max(0, start + K - Q + 1 - window), end + K - Q)`, whose
/// width is at most `query_rows.len() + window - 1`. Returns `None` for an
/// empty or out-of-bounds row range, `query_sequence > key_sequence`, or a
/// zero window. Exactness (the range is the union of the rows' visible keys)
/// is verified with Verus.
#[must_use]
pub fn sliding_window_key_range(
    query_sequence: usize,
    key_sequence: usize,
    query_rows: Range<usize>,
    window: usize,
) -> Option<Range<usize>> {
    plan::sliding_window_key_range(
        query_sequence,
        key_sequence,
        query_rows.start,
        query_rows.end,
        window,
    )
    .map(|(start, end)| start..end)
}

/// Maps a verified planning failure to the crate's contextual error.
fn chunk_plan_error(error: ChunkPlanError) -> Error {
    match error {
        ChunkPlanError::ZeroMaxQueryTokens => Error::configuration(
            "attention.execution.max_query_tokens",
            "must be greater than zero",
        ),
        ChunkPlanError::ZeroMaxScoreElements => Error::configuration(
            "attention.execution.max_score_elements",
            "must be greater than zero",
        ),
        ChunkPlanError::EmptyAxis => Error::configuration(
            "attention.execution_shape",
            "batch, heads, query sequence, and key bound must be non-zero",
        ),
        ChunkPlanError::RowOverflow => Error::configuration(
            "attention.execution_shape",
            "score-plane dimensions overflow usize",
        ),
        ChunkPlanError::RowExceedsLimit {
            row_elements,
            max_score_elements,
        } => Error::BackendCapability(format!(
            "one attention query row requires {row_elements} score elements, exceeding the configured bound of {max_score_elements}"
        )),
        ChunkPlanError::PeakOverflow => Error::configuration(
            "attention.execution_shape",
            "planned score-plane size overflows usize",
        ),
    }
}
