//! Pure integer constraints behind specification validation and grouped reorders.
//!
//! Serializable specifications ([`super::spec`]) combine floating-point
//! options, shared component configurations, and field-path error messages.
//! Their integer rules, nonzero dimensions, head grouping, routed-expert
//! counts, projection sizes, and group-preserving reorders, are decided here,
//! free of Burn types, floating point, and crate errors, so Verus checks these
//! exact bodies through `verus/experiment.rs` (see `verus/README.md`). Callers
//! map each outcome to the same contextual [`crate::Error`] as before.
//!
//! # Verified properties
//!
//! - [`attention_heads`] and [`delta_net_heads`] classify a head layout as
//!   having a zero dimension, indivisible groups, or valid groups, in that
//!   precedence. A valid layout has groups of exactly `heads / groups` heads:
//!   head `h` maps to the in-range group `h / group_size`, and each group is
//!   the contiguous block of `group_size` heads it starts.
//! - [`mamba2_dimensions`] accepts exactly nonzero dimensions whose group size
//!   divides the head count.
//! - [`expert_counts`] accepts exactly a nonzero expert count and width with
//!   `1 <= experts_per_token <= routed_experts`.
//! - [`mla_dimensions`] rejects zero dimensions, then a query-key width
//!   `content + rotary` that overflows `usize`; [`mla_qk_head_dim`] returns
//!   exactly `min(content + rotary, usize::MAX)`.
//! - [`cross_projection_fits`] accepts exactly nonzero heads and head widths
//!   whose inner width `heads * head_dim` and projection size
//!   `heads * head_dim * hidden` both fit in `usize`.
//! - [`stack_dimensions`] reports the first zero among vocabulary, width,
//!   layer count, and position limit, in that order.
//! - [`group_reorder`] rejects a group size that is zero or does not divide
//!   the batch, then reports the first row whose source leaves its group;
//!   an accepted reorder keeps every row in its group, and for a full batch
//!   every source index is in bounds.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Outcome of the integer checks on a grouped head layout.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadCheck {
    /// A head count or channel width is zero.
    ZeroDimension,
    /// The grouped head count is not a multiple of the group count.
    IndivisibleGroups,
    /// Every dimension is positive and the heads split into equal groups.
    Valid,
}

/// Classifies grouped-query attention heads: `query_heads` split over `key_value_heads`.
///
/// Zero dimensions take precedence over indivisible groups. A valid layout
/// shares each key/value head among exactly `query_heads / key_value_heads`
/// query heads, and query head `h` reads key/value head `h / group_size`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == HeadCheck::ZeroDimension)
            == (query_heads == 0 || key_value_heads == 0 || head_dim == 0),
        (check == HeadCheck::IndivisibleGroups) == (query_heads > 0 && key_value_heads > 0
            && head_dim > 0 && query_heads % key_value_heads != 0),
        check == HeadCheck::Valid ==> heads_grouped(query_heads as int, key_value_heads as int),
))]
pub(crate) fn attention_heads(
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> HeadCheck {
    if query_heads == 0 || key_value_heads == 0 || head_dim == 0 {
        return HeadCheck::ZeroDimension;
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return HeadCheck::IndivisibleGroups;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_heads_grouped(query_heads as int, key_value_heads as int);
    }
    HeadCheck::Valid
}

/// Classifies Gated DeltaNet heads: `value_heads` split over `key_heads`.
///
/// Any zero head count, head width, or convolution kernel takes precedence;
/// otherwise the value heads must form equal groups over the key heads, so
/// value head `h` reads key head `h / (value_heads / key_heads)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == HeadCheck::ZeroDimension) == (key_heads == 0 || value_heads == 0
            || key_head_dim == 0 || value_head_dim == 0 || convolution_kernel == 0),
        (check == HeadCheck::IndivisibleGroups) == (key_heads > 0 && value_heads > 0
            && key_head_dim > 0 && value_head_dim > 0 && convolution_kernel > 0
            && value_heads % key_heads != 0),
        check == HeadCheck::Valid ==> heads_grouped(value_heads as int, key_heads as int),
))]
pub(crate) fn delta_net_heads(
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    convolution_kernel: usize,
) -> HeadCheck {
    if key_heads == 0
        || value_heads == 0
        || key_head_dim == 0
        || value_head_dim == 0
        || convolution_kernel == 0
    {
        return HeadCheck::ZeroDimension;
    }
    if !value_heads.is_multiple_of(key_heads) {
        return HeadCheck::IndivisibleGroups;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_heads_grouped(value_heads as int, key_heads as int);
    }
    HeadCheck::Valid
}

/// Whether Mamba2 dimensions are nonzero and `group_size` divides `heads`.
///
/// When accepted, the heads split into `heads / group_size` state-space
/// groups of exactly `group_size` heads each.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(valid =>
    ensures
        valid == (heads > 0 && head_dim > 0 && state_dim > 0 && convolution_kernel > 0
            && group_size > 0 && heads % group_size == 0),
        valid ==> heads_grouped(heads as int, (heads / group_size) as int),
))]
pub(crate) fn mamba2_dimensions(
    heads: usize,
    head_dim: usize,
    state_dim: usize,
    convolution_kernel: usize,
    group_size: usize,
) -> bool {
    if heads == 0
        || head_dim == 0
        || state_dim == 0
        || convolution_kernel == 0
        || group_size == 0
        || !heads.is_multiple_of(group_size)
    {
        return false;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(heads as int, group_size as int);
        let groups = heads as int / group_size as int;
        assert(groups * group_size == heads);
        assert(groups > 0) by (nonlinear_arith)
            requires groups * group_size == heads, heads > 0, group_size > 0, groups >= 0;
        vstd::arithmetic::div_mod::lemma_div_multiples_vanish(group_size as int, groups);
        assert(heads as int % groups == 0) by {
            vstd::arithmetic::div_mod::lemma_mod_multiples_basic(group_size as int, groups);
            assert(group_size as int * groups == heads);
        }
        lemma_heads_grouped(heads as int, groups);
    }
    true
}

/// Outcome of the integer checks on a sparse mixture-of-experts layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertCheck {
    /// There are no routed experts, or experts have no intermediate channels.
    Empty,
    /// The number of experts selected per token is not in `1..=routed_experts`.
    SelectionOutOfRange,
    /// Expert counts and widths are usable.
    Valid,
}

/// Classifies routed-expert counts: nonzero experts and width, then `1 <= k <= experts`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == ExpertCheck::Empty) == (routed_experts == 0 || expert_intermediate_size == 0),
        (check == ExpertCheck::SelectionOutOfRange) == (routed_experts > 0
            && expert_intermediate_size > 0
            && (experts_per_token == 0 || experts_per_token > routed_experts)),
        (check == ExpertCheck::Valid) == (expert_intermediate_size > 0
            && 1 <= experts_per_token <= routed_experts),
))]
pub(crate) fn expert_counts(
    routed_experts: usize,
    experts_per_token: usize,
    expert_intermediate_size: usize,
) -> ExpertCheck {
    if routed_experts == 0 || expert_intermediate_size == 0 {
        return ExpertCheck::Empty;
    }
    if experts_per_token == 0 || experts_per_token > routed_experts {
        return ExpertCheck::SelectionOutOfRange;
    }
    ExpertCheck::Valid
}

/// Outcome of the integer checks on multi-head latent attention dimensions.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlaCheck {
    /// A head count, head width, or rank is zero.
    ZeroDimension,
    /// The query-key width `content + rotary` does not fit in `usize`.
    HeadDimOverflow,
    /// Every dimension is positive and the query-key width is representable.
    Valid,
}

/// Classifies MLA dimensions: zero dimensions first, then query-key width overflow.
///
/// `query_lora_rank` is optional, but a present rank must be positive.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == MlaCheck::ZeroDimension) == (query_heads == 0 || content_head_dim == 0
            || rotary_head_dim == 0 || value_head_dim == 0 || key_value_lora_rank == 0
            || query_lora_rank == Some(0usize)),
        (check == MlaCheck::HeadDimOverflow) == (check != MlaCheck::ZeroDimension
            && content_head_dim + rotary_head_dim > usize::MAX),
        (check == MlaCheck::Valid) == (check != MlaCheck::ZeroDimension
            && content_head_dim + rotary_head_dim <= usize::MAX),
))]
pub(crate) fn mla_dimensions(
    query_heads: usize,
    content_head_dim: usize,
    rotary_head_dim: usize,
    value_head_dim: usize,
    key_value_lora_rank: usize,
    query_lora_rank: Option<usize>,
) -> MlaCheck {
    let zero_query_rank = match query_lora_rank {
        Some(rank) => rank == 0,
        None => false,
    };
    if query_heads == 0
        || content_head_dim == 0
        || rotary_head_dim == 0
        || value_head_dim == 0
        || key_value_lora_rank == 0
        || zero_query_rank
    {
        return MlaCheck::ZeroDimension;
    }
    if content_head_dim.checked_add(rotary_head_dim).is_none() {
        return MlaCheck::HeadDimOverflow;
    }
    MlaCheck::Valid
}

/// The MLA query-key width `content + rotary`, saturated at `usize::MAX`.
///
/// Validation rejects an overflowing sum first; saturation keeps
/// configuration builders total for unvalidated specifications.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(width =>
    ensures
        content_head_dim + rotary_head_dim <= usize::MAX ==> width == content_head_dim + rotary_head_dim,
        content_head_dim + rotary_head_dim > usize::MAX ==> width == usize::MAX,
))]
pub(crate) fn mla_qk_head_dim(content_head_dim: usize, rotary_head_dim: usize) -> usize {
    content_head_dim.saturating_add(rotary_head_dim)
}

/// Whether cross-attention heads are nonzero and `heads * head_dim * hidden` fits in `usize`.
///
/// The inner width `heads * head_dim` must fit as well, even for a zero
/// residual width. The full product is the element count of each
/// `hidden -> heads * head_dim` projection matrix, so acceptance means the
/// inner width and every projection size are representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fits =>
    ensures
        fits == (heads > 0 && head_dim > 0 && heads * head_dim <= usize::MAX
            && heads * head_dim * hidden_size <= usize::MAX),
))]
pub(crate) fn cross_projection_fits(heads: usize, head_dim: usize, hidden_size: usize) -> bool {
    let projection_elements = match heads.checked_mul(head_dim) {
        Some(inner) => inner.checked_mul(hidden_size),
        None => None,
    };
    heads != 0 && head_dim != 0 && projection_elements.is_some()
}

/// Outcome of the global dimension checks of one modular stack.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StackCheck {
    /// The vocabulary is empty.
    ZeroVocabulary,
    /// The residual width is zero.
    ZeroHidden,
    /// The stack has no blocks.
    NoLayers,
    /// The maximum supported position count is zero.
    ZeroPositions,
    /// Every global dimension is positive.
    Valid,
}

/// Reports the first zero among vocabulary, width, layer count, and position limit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == StackCheck::ZeroVocabulary) == (vocabulary_size == 0),
        (check == StackCheck::ZeroHidden) == (vocabulary_size > 0 && hidden_size == 0),
        (check == StackCheck::NoLayers) == (vocabulary_size > 0 && hidden_size > 0 && layers == 0),
        (check == StackCheck::ZeroPositions) == (vocabulary_size > 0 && hidden_size > 0
            && layers > 0 && max_position_embeddings == 0),
        (check == StackCheck::Valid) == (vocabulary_size > 0 && hidden_size > 0 && layers > 0
            && max_position_embeddings > 0),
))]
pub(crate) fn stack_dimensions(
    vocabulary_size: usize,
    hidden_size: usize,
    layers: usize,
    max_position_embeddings: usize,
) -> StackCheck {
    if vocabulary_size == 0 {
        StackCheck::ZeroVocabulary
    } else if hidden_size == 0 {
        StackCheck::ZeroHidden
    } else if layers == 0 {
        StackCheck::NoLayers
    } else if max_position_embeddings == 0 {
        StackCheck::ZeroPositions
    } else {
        StackCheck::Valid
    }
}

/// Outcome of checking a reorder that must keep every row inside its group.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupReorder {
    /// The group size is zero or does not divide the batch.
    InvalidGroupSize,
    /// Row `row` reads a source from another group.
    LeavesGroup {
        /// First destination row whose source lies outside its group.
        row: usize,
    },
    /// Every row reads a source from its own group.
    Preserved,
}

/// Checks that `indices[row]` stays in group `row / group_size` for every row.
///
/// Rows are grouped consecutively, `group_size` at a time, and the group
/// size must divide `batch`. The first row whose source leaves its group is
/// reported. When every row stays in its group and `indices` covers the whole
/// batch, every source index is also below `batch`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(check =>
    ensures
        (check == GroupReorder::InvalidGroupSize) == (group_size == 0 || batch % group_size != 0),
        match check {
            GroupReorder::InvalidGroupSize => true,
            GroupReorder::LeavesGroup { row } => {
                &&& row < indices.len()
                &&& indices@[row as int] / group_size != row / group_size
                &&& forall|earlier: int| 0 <= earlier < row
                    ==> indices@[earlier] as int / group_size as int == earlier / group_size as int
            },
            GroupReorder::Preserved => {
                &&& forall|row: int| 0 <= row < indices.len()
                    ==> indices@[row] as int / group_size as int == row / group_size as int
                &&& indices.len() == batch ==> forall|row: int| 0 <= row < indices.len()
                    ==> indices@[row] < batch
            },
        },
))]
pub(crate) fn group_reorder(indices: &[usize], group_size: usize, batch: usize) -> GroupReorder {
    if group_size == 0 || !batch.is_multiple_of(group_size) {
        return GroupReorder::InvalidGroupSize;
    }
    let mut row = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= row <= indices.len(),
            group_size > 0,
            batch % group_size == 0,
            forall|earlier: int| 0 <= earlier < row
                ==> indices@[earlier] as int / group_size as int == earlier / group_size as int,
        decreases indices.len() - row,
    ))]
    while row < indices.len() {
        if indices[row] / group_size != row / group_size {
            return GroupReorder::LeavesGroup { row };
        }
        row += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        if indices.len() == batch {
            assert forall|row: int| 0 <= row < indices.len() implies indices@[row] < batch by {
                lemma_same_group_in_bounds(
                    indices@[row] as int,
                    row,
                    group_size as int,
                    batch as int,
                );
            }
        }
    }
    GroupReorder::Preserved
}

#[cfg(verus_keep_ghost)]
verus! {

/// `heads` split into `groups` equal groups: head `h` belongs to group `h / (heads / groups)`.
///
/// The group size is positive and exact, every head maps to a valid group,
/// and group `g` is exactly the contiguous block of heads
/// `g * size..(g + 1) * size`, so every group has exactly `size` members.
pub(crate) open spec fn heads_grouped(heads: int, groups: int) -> bool {
    let size = heads / groups;
    &&& heads > 0
    &&& groups > 0
    &&& size >= 1
    &&& size * groups == heads
    &&& forall|head: int| 0 <= head < heads ==> {
        &&& 0 <= #[trigger] (head / size) < groups
        &&& (head / size) * size <= head < (head / size) * size + size
    }
}

/// Divisible positive head counts form equal groups that every head maps into.
pub(crate) proof fn lemma_heads_grouped(heads: int, groups: int)
    requires
        heads > 0,
        groups > 0,
        heads % groups == 0,
    ensures
        heads_grouped(heads, groups),
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(heads, groups);
    let size = heads / groups;
    assert(size * groups == heads) by (nonlinear_arith)
        requires heads == groups * size + 0;
    assert(size >= 1) by (nonlinear_arith)
        requires size * groups == heads, heads > 0, groups > 0;
    assert forall|head: int| 0 <= head < heads implies {
        &&& 0 <= #[trigger] (head / size) < groups
        &&& (head / size) * size <= head < (head / size) * size + size
    } by {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(head, size);
        vstd::arithmetic::div_mod::lemma_mod_pos_bound(head, size);
        let quotient = head / size;
        assert(quotient * size == size * quotient) by (nonlinear_arith);
        assert(quotient >= 0) by (nonlinear_arith)
            requires head == size * quotient + head % size, 0 <= head % size < size, head >= 0, size >= 1;
        assert(quotient < groups) by (nonlinear_arith)
            requires
                head == size * quotient + head % size,
                head % size >= 0,
                head < heads,
                size * groups == heads,
                size >= 1;
    }
}

/// A source in the same group as an in-batch row is itself inside the batch.
///
/// With `batch` a multiple of `size`, group `row / size` of a row below
/// `batch` ends at or before `batch`, and so does every member of that group.
pub(crate) proof fn lemma_same_group_in_bounds(source: int, row: int, size: int, batch: int)
    requires
        0 <= source,
        0 <= row < batch,
        size > 0,
        batch % size == 0,
        source / size == row / size,
    ensures
        source < batch,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(source, size);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(source, size);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(row, size);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(row, size);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(batch, size);
    let group = row / size;
    let groups = batch / size;
    assert(group < groups) by (nonlinear_arith)
        requires
            row == size * group + row % size,
            row % size >= 0,
            row < batch,
            batch == size * groups,
            size > 0;
    assert(source < batch) by (nonlinear_arith)
        requires
            source == size * group + source % size,
            source % size < size,
            group < groups,
            batch == size * groups,
            size > 0;
}

} // verus!

#[cfg(test)]
#[path = "constraints_tests.rs"]
mod tests;
