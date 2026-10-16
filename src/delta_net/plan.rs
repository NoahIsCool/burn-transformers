//! Dependency-free integer planning for Gated DeltaNet prefill.
//!
//! The chunked prefill splits `L` tokens into chunks of `C = min(chunk_size,
//! L)` tokens, so there are `n = ceil(L / C)` chunks and only the last may be
//! short; the token axis is zero-padded once to `n * C`. Chunks are batched in
//! balanced blocks of at most `B` chunks: `m = ceil(n / B)` blocks whose sizes
//! differ by at most one, the larger first. Each chunk's unit lower-triangular
//! system `(I + L)^-1` is inverted by recursive doubling over a `P x P` tile,
//! where `P` is the least power of two not below `C`; level `k` holds `2^k`
//! diagonal blocks of side `P / 2^k`, down to side two. The fused Q/K/V
//! convolution carries `2 * K + V` channels, with `K = key_heads *
//! key_head_dim` query (and key) channels and `V = value_heads *
//! value_head_dim` value channels, laid out as queries, keys, then values.
//!
//! # Verified planning
//!
//! Every function here is checked in place by Verus through
//! `verus/mixers.rs`. The proofs establish, for every admissible input, that
//! no arithmetic overflows and that the results are exact: the chunk width and
//! count cover the tokens with less than one chunk of padding; block ranges
//! are consecutive, start at zero, end at `n`, hold between one and `B`
//! chunks, and follow the balanced size formula; the doubling schedule starts
//! at the least power of two not below the width and halves exactly down to
//! two, and its levels tile the `P x P` tile with twice as many blocks per
//! level; the projection widths are exact or rejected exactly when the
//! channel count overflows; the query, key, and value slices tile the fused
//! channels and key heads repeat exactly onto the value heads; and the
//! recurrence-versus-chunked choice equals its closed-form cost comparison
//! and is monotone in the segment length and the state size.

use core::ops::Range;

use crate::division::ceil_div_usize;
#[cfg(verus_keep_ghost)]
use crate::division::ceiling;
#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// First chunk of balanced block `i` among `blocks`: earlier blocks hold `chunks / blocks`
/// chunks each, plus one for each of the first `chunks % blocks` blocks.
pub open spec fn balanced_start(chunks: int, blocks: int, i: int) -> int {
    i * (chunks / blocks) + if i < chunks % blocks { i } else { chunks % blocks }
}

/// Chunks in balanced block `i`: `chunks / blocks`, plus one for the first `chunks % blocks` blocks.
pub open spec fn balanced_size(chunks: int, blocks: int, i: int) -> int {
    chunks / blocks + if i < chunks % blocks { 1int } else { 0int }
}

/// Whether `value` is a positive power of two.
pub open spec fn is_power_of_two(value: int) -> bool
    decreases value,
{
    if value <= 0 {
        false
    } else if value == 1 {
        true
    } else {
        value % 2 == 0 && is_power_of_two(value / 2)
    }
}

/// `x` saturated to the largest `usize`.
pub open spec fn saturated(x: int) -> int {
    if x > usize::MAX { usize::MAX as int } else { x }
}

/// Closed form of [`prefers_recurrence`]: one token, or a recurrence no costlier than chunking.
pub open spec fn recurrence_preferred(sequence: int, state_elements: int) -> bool {
    sequence == 1 || sequence * saturated(6_144 + state_elements) <= 73_728
}

} // verus!

/// Recurrence cost of one token beyond its state work, in state-element units.
///
/// Calibrated on the Flex CPU backend: a recurrent token issues about eight
/// small kernels (~4.5 µs), and a state element costs ~0.4 ns more in the
/// recurrence than in the batched chunk algebra, so one token's fixed cost is
/// worth about 6 000 state elements.
#[cfg_attr(verus_keep_ghost, verus_verify)]
const RECURRENCE_TOKEN_OVERHEAD: usize = 6_144;

/// Fixed cost of a chunked prefill call in the same units (~30 µs of kernels).
#[cfg_attr(verus_keep_ghost, verus_verify)]
const CHUNKED_FIXED_OVERHEAD: usize = 12 * RECURRENCE_TOKEN_OVERHEAD;

/// Returns whether the recurrence is expected to beat the chunked algorithm.
///
/// A single token always takes the recurrence, which is then one exact step.
/// Otherwise the recurrence is modelled as `L * (token overhead + state)` and
/// the chunked algorithm as a fixed launch overhead plus cheaper batched
/// per-token work, with `state_elements = heads * key_width * value_width`.
/// With Qwen3.5's `16 * 128 * 128` states the chunked algorithm wins from two
/// tokens on; tiny experimental states (for example `2 * 4 * 3`) use the
/// recurrence up to about a dozen tokens, where the chunked algorithm's ~100
/// kernel launches dominate.
///
/// The proof shows the result is exactly `L == 1 || L * min(6144 + state,
/// usize::MAX) <= 73728`, so an overflowing cost always chooses chunking, and
/// `lemma_recurrence_choice_monotone` shows that shorter segments and smaller
/// states never switch the recurrence off.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(prefers =>
    ensures
        prefers == recurrence_preferred(sequence as int, state_elements as int),
))]
#[must_use]
pub(super) fn prefers_recurrence(sequence: usize, state_elements: usize) -> bool {
    if sequence == 1 {
        return true;
    }
    match sequence.checked_mul(RECURRENCE_TOKEN_OVERHEAD.saturating_add(state_elements)) {
        Some(cost) => cost <= CHUNKED_FIXED_OVERHEAD,
        None => false,
    }
}

/// Chunk width and count of one chunked prefill.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChunkPlan {
    /// Chunk width `C = min(chunk_size, L)`.
    pub(super) chunk: usize,
    /// Number of chunks `ceil(L / C)`.
    pub(super) chunks: usize,
    /// Zero-padded token count `chunks * C`, less than one chunk above `L`.
    pub(super) padded: usize,
}

/// Plans the chunks of an `L = sequence` token prefill with chunks of at most `chunk_size`.
///
/// The proof shows `C = min(chunk_size, L)` and `n = ceil(L / C)` satisfy
/// `1 <= C <= L` and `(n - 1) * C < L <= n * C`, and that the padded length
/// `n * C` exceeds `L` by less than one chunk and never overflows; a prompt no
/// longer than `chunk_size` is one unpadded chunk. The bound on `L` holds for
/// every materialized token axis, since an F32 operand of `L` tokens alone
/// needs `4 * L` bytes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(plan =>
    requires
        0 < sequence <= usize::MAX / 2,
        chunk_size > 0,
    ensures
        plan.chunk == if chunk_size < sequence { chunk_size } else { sequence },
        1 <= plan.chunk <= sequence,
        plan.chunks == ceiling(sequence as int, plan.chunk as int),
        plan.chunks >= 1,
        (plan.chunks - 1) * plan.chunk < sequence,
        sequence <= plan.chunks * plan.chunk,
        plan.chunks * plan.chunk < sequence + plan.chunk,
        plan.padded == plan.chunks * plan.chunk,
        sequence <= chunk_size ==> plan.chunks == 1 && plan.chunk == sequence,
))]
pub(super) fn chunk_plan(sequence: usize, chunk_size: usize) -> ChunkPlan {
    let chunk = chunk_size.min(sequence);
    let chunks = ceil_div_usize(sequence, chunk);
    #[cfg(verus_keep_ghost)]
    proof! {
        assert((chunks - 1) * chunk == chunks * chunk - chunk) by (nonlinear_arith);
        if sequence <= chunk_size {
            vstd::arithmetic::div_mod::lemma_div_basics(sequence as int);
            vstd::arithmetic::div_mod::lemma_mod_self_0(sequence as int);
        }
    }
    let padded = chunks * chunk;
    ChunkPlan {
        chunk,
        chunks,
        padded,
    }
}

/// Splits `chunks` chunks into balanced consecutive blocks of at most `limit` chunks.
///
/// Uses `ceil(chunks / limit)` blocks whose sizes differ by at most one, the
/// larger blocks first. Compared with greedy blocks of exactly `limit`, this
/// never leaves a short final block: its inverse would cost as many
/// sequential levels as a full one while batching less work. A zero `limit`
/// acts as one; zero chunks produce no block.
///
/// The proof shows, with `L = max(limit, 1)` and `m = ceil(chunks / L)`
/// blocks, that block `i` has `chunks / m + [i < chunks % m]` chunks, between
/// one and `L`, starting at `i * (chunks / m) + min(i, chunks % m)`; that each
/// block ends where the next starts; and that the blocks cover exactly
/// `0..chunks`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ranges =>
    ensures
        ranges.len() == ceiling(chunks as int, if limit == 0 { 1int } else { limit as int }),
        (ranges.len() == 0) == (chunks == 0),
        ranges.len() > 0 ==> ranges[0].start == 0,
        ranges.len() > 0 ==> ranges[ranges.len() - 1].end == chunks,
        forall|i: int| 0 <= i < ranges.len() ==> {
            &&& #[trigger] ranges[i].start == balanced_start(chunks as int, ranges.len() as int, i)
            &&& ranges[i].end - ranges[i].start == balanced_size(chunks as int, ranges.len() as int, i)
            &&& 1 <= ranges[i].end - ranges[i].start
            &&& ranges[i].end - ranges[i].start <= if limit == 0 { 1int } else { limit as int }
        },
        forall|i: int| 0 <= i && i + 1 < ranges.len() ==>
            #[trigger] ranges[i].end == ranges[i + 1].start,
))]
pub(super) fn block_chunk_ranges(chunks: usize, limit: usize) -> Vec<Range<usize>> {
    let limit = limit.max(1);
    let blocks = ceil_div_usize(chunks, limit);
    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(blocks);
    if blocks == 0 {
        return ranges;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_balanced_sizes(chunks as int, limit as int, blocks as int);
    }
    // The first `chunks % blocks` blocks hold one extra chunk.
    let (base, extra) = (chunks / blocks, chunks % blocks);
    let mut start = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 < blocks,
            base == chunks / blocks,
            extra == chunks % blocks,
            1 <= limit,
            1 <= base + if extra > 0 { 1int } else { 0int },
            base + if extra > 0 { 1int } else { 0int } <= limit,
            blocks * base + extra == chunks,
            index <= blocks,
            ranges.len() == index,
            start == balanced_start(chunks as int, blocks as int, index as int),
            forall|i: int| 0 <= i < index ==> {
                &&& #[trigger] ranges[i].start == balanced_start(chunks as int, blocks as int, i)
                &&& ranges[i].end - ranges[i].start == balanced_size(chunks as int, blocks as int, i)
            },
            forall|i: int| 0 <= i && i + 1 < index ==> #[trigger] ranges[i].end == ranges[i + 1].start,
            index > 0 ==> ranges[index - 1].end == start,
        decreases blocks - index,
    ))]
    while index < blocks {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_block_start_bound(chunks as int, blocks as int, index as int);
        }
        let size = base + if index < extra { 1 } else { 0 };
        ranges.push(start..start + size);
        start += size;
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(start == chunks);
    }
    ranges
}

/// Returns the diagonal block sizes `P, P / 2, …, 2` of the doubling inverse of a `width x width` system.
///
/// `P` is the least power of two not below `width`. The inverse first splits
/// `P` down to `2 x 2` blocks (the pairs of consecutive sizes) and then
/// combines them back up. The proof shows that `sizes[0] = P` is a power of
/// two with `width <= P < 2 * width`, that every size is twice the next, that
/// the last is two, and that `sizes[k] * 2^k = P`, so level `k` holds exactly
/// `P / sizes[k] = 2^k` blocks.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sizes =>
    requires
        2 <= width <= usize::MAX / 2,
    ensures
        sizes.len() >= 1,
        is_power_of_two(sizes[0] as int),
        width <= sizes[0] < 2 * width,
        sizes[sizes.len() - 1] == 2,
        forall|k: int| 0 <= k && k + 1 < sizes.len() ==> #[trigger] sizes[k] == 2 * sizes[k + 1],
        forall|k: int| 0 <= k < sizes.len() ==> {
            &&& #[trigger] sizes[k] >= 2
            &&& is_power_of_two(sizes[k] as int)
            &&& sizes[k] * vstd::arithmetic::power2::pow2(k as nat) == sizes[0]
        },
))]
pub(super) fn inverse_block_sizes(width: usize) -> Vec<usize> {
    let mut size = 1usize;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            2 <= width <= usize::MAX / 2,
            1 <= size,
            is_power_of_two(size as int),
            size == 1 || size < 2 * width,
            size <= 2 * width,
        decreases 2 * width - size,
    ))]
    while size < width {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_double_power_of_two(size as int);
        }
        size *= 2;
    }
    let mut sizes: Vec<usize> = Vec::new();
    let mut block = size;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::power2::lemma2_to64();
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            2 <= width <= size < 2 * width,
            is_power_of_two(size as int),
            is_power_of_two(block as int),
            sizes.len() == 0 ==> block == size,
            sizes.len() > 0 ==> sizes[sizes.len() - 1] == 2 * block,
            block * vstd::arithmetic::power2::pow2(sizes.len() as nat) == size,
            forall|k: int| 0 <= k && k + 1 < sizes.len() ==> #[trigger] sizes[k] == 2 * sizes[k + 1],
            forall|k: int| 0 <= k < sizes.len() ==> {
                &&& #[trigger] sizes[k] >= 2
                &&& is_power_of_two(sizes[k] as int)
                &&& sizes[k] * vstd::arithmetic::power2::pow2(k as nat) == size
            },
        decreases block,
    ))]
    while block >= 2 {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_halve_power_of_two(block as int);
            vstd::arithmetic::power2::lemma_pow2_unfold((sizes.len() + 1) as nat);
            assert((block / 2) * vstd::arithmetic::power2::pow2((sizes.len() + 1) as nat) == size)
                by (nonlinear_arith)
                requires
                    block == 2 * (block / 2),
                    vstd::arithmetic::power2::pow2((sizes.len() + 1) as nat)
                        == 2 * vstd::arithmetic::power2::pow2(sizes.len() as nat),
                    block * vstd::arithmetic::power2::pow2(sizes.len() as nat) == size;
        }
        sizes.push(block);
        block /= 2;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        // A power of two below two is one, so the last pushed size is two.
        assert(block == 1);
        vstd::arithmetic::power2::lemma_pow2_pos(0);
        vstd::arithmetic::power2::lemma2_to64();
    }
    sizes
}

/// Doubling-inverse geometry of one level: `count` diagonal blocks of `block x block`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InverseLevel {
    /// Side of every diagonal block at this level.
    pub(super) block: usize,
    /// Side of each of the two child blocks, `block / 2`.
    pub(super) half: usize,
    /// Number of diagonal blocks, `P / block`.
    pub(super) count: usize,
}

/// Returns the per-level block geometry of the doubling inverse of a `width x width` system.
///
/// Level `k` holds `count` diagonal blocks of side `block`, each split into
/// two children of side `half`. The proof shows the levels start with one
/// `P x P` block, where `P` is the least power of two not below `width`;
/// that every level tiles the `P x P` tile exactly (`count * block == P`)
/// with `block == 2 * half`; that each next level has the children's side
/// and twice as many blocks; and that the last level has blocks of side two.
/// The inverse narrows and reshapes with exactly these extents.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(levels =>
    requires
        2 <= width <= usize::MAX / 2,
    ensures
        levels.len() >= 1,
        levels[0].count == 1,
        is_power_of_two(levels[0].block as int),
        width <= levels[0].block < 2 * width,
        levels[levels.len() - 1].block == 2,
        forall|k: int| 0 <= k < levels.len() ==> {
            &&& #[trigger] levels[k].block == 2 * levels[k].half
            &&& levels[k].half >= 1
            &&& levels[k].count * levels[k].block == levels[0].block
        },
        forall|k: int| 0 <= k && k + 1 < levels.len() ==> {
            &&& #[trigger] levels[k + 1].block == levels[k].half
            &&& levels[k + 1].count == 2 * levels[k].count
        },
))]
pub(super) fn inverse_levels(width: usize) -> Vec<InverseLevel> {
    let sizes = inverse_block_sizes(width);
    let mut levels: Vec<InverseLevel> = Vec::with_capacity(sizes.len());
    let mut count = 1usize;
    let mut level = 0usize;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::power2::lemma2_to64();
    }
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            level <= sizes.len(),
            sizes.len() >= 1,
            levels.len() == level,
            count == vstd::arithmetic::power2::pow2(level as nat),
            level < sizes.len() ==> count * sizes[level as int] == sizes[0],
            forall|k: int| 0 <= k && k + 1 < sizes.len() ==> #[trigger] sizes[k] == 2 * sizes[k + 1],
            forall|k: int| 0 <= k < sizes.len() ==> {
                &&& #[trigger] sizes[k] >= 2
                &&& is_power_of_two(sizes[k] as int)
                &&& sizes[k] * vstd::arithmetic::power2::pow2(k as nat) == sizes[0]
            },
            forall|k: int| 0 <= k < level ==> {
                &&& #[trigger] levels[k].block == sizes[k]
                &&& levels[k].block == 2 * levels[k].half
                &&& levels[k].count == vstd::arithmetic::power2::pow2(k as nat)
            },
        decreases sizes.len() - level,
    ))]
    while level < sizes.len() {
        let block = sizes[level];
        #[cfg(verus_keep_ghost)]
        proof! {
            vstd::arithmetic::power2::lemma_pow2_pos(level as nat);
            lemma_halve_power_of_two(block as int);
            // `count * block == P` with `block >= 2`, so doubling `count` stays within `P`.
            assert(2 * count <= count * block) by (nonlinear_arith)
                requires block >= 2, count >= 1;
            vstd::arithmetic::power2::lemma_pow2_unfold((level + 1) as nat);
            if level + 1 < sizes.len() {
                assert(2 * count * sizes[level + 1] == sizes[0]) by (nonlinear_arith)
                    requires
                        count * block == sizes[0],
                        block == 2 * sizes[level + 1];
            }
        }
        levels.push(InverseLevel {
            block,
            half: block / 2,
            count,
        });
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(levels[level as int].block == sizes[level as int]);
            assert(levels[level as int].block == 2 * levels[level as int].half);
            assert(levels[level as int].count == vstd::arithmetic::power2::pow2(level as nat));
        }
        count *= 2;
        level += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::power2::lemma_pow2_pos(0);
        assert(levels[0].block == sizes[0]);
        assert forall|k: int| 0 <= k < levels.len() implies {
            &&& #[trigger] levels[k].block == 2 * levels[k].half
            &&& levels[k].half >= 1
            &&& levels[k].count * levels[k].block == levels[0].block
        } by {
            assert(levels[k].block == sizes[k]);
            assert(sizes[k] * vstd::arithmetic::power2::pow2(k as nat) == sizes[0]);
            assert(levels[k].count * levels[k].block == sizes[0]) by (nonlinear_arith)
                requires
                    levels[k].count == vstd::arithmetic::power2::pow2(k as nat),
                    levels[k].block == sizes[k],
                    sizes[k] * vstd::arithmetic::power2::pow2(k as nat) == sizes[0];
        }
        assert forall|k: int| 0 <= k && k + 1 < levels.len() implies
            #[trigger] levels[k + 1].block == levels[k].half
                && levels[k + 1].count == 2 * levels[k].count
        by {
            assert(levels[k].block == sizes[k]);
            assert(levels[k + 1].block == sizes[k + 1]);
            assert(sizes[k] == 2 * sizes[k + 1]);
            vstd::arithmetic::power2::lemma_pow2_unfold((k + 1) as nat);
        }
    }
    levels
}

/// Widths of a Gated DeltaNet's fused query/key/value projection.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProjectionWidths {
    /// Query width, equal to the key width: `key_heads * key_head_dim`.
    pub(super) key_width: usize,
    /// Value width `value_heads * value_head_dim`.
    pub(super) value_width: usize,
    /// Fused convolution channels `2 * key_width + value_width`.
    pub(super) channels: usize,
}

/// Returns the fused Q/K/V widths, with `2 * key_heads * key_head_dim + value_heads * value_head_dim` channels.
///
/// The proof shows the key, value, and channel widths are exact whenever the
/// channel count fits in `usize`, and that `None` is returned exactly when it
/// does not; every partial product then fits as well, so no separate check of
/// the key or value width can fail afterwards.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(widths =>
    ensures
        match widths {
            Some(widths) => {
                &&& widths.key_width == key_heads * key_head_dim
                &&& widths.value_width == value_heads * value_head_dim
                &&& widths.channels == 2 * widths.key_width + widths.value_width
            },
            None => 2 * (key_heads * key_head_dim) + value_heads * value_head_dim > usize::MAX,
        },
))]
pub(super) fn projection_widths(
    key_heads: usize,
    key_head_dim: usize,
    value_heads: usize,
    value_head_dim: usize,
) -> Option<ProjectionWidths> {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(key_heads * key_head_dim >= 0) by (nonlinear_arith);
        assert(value_heads * value_head_dim >= 0) by (nonlinear_arith);
    }
    let key_width = key_heads.checked_mul(key_head_dim)?;
    let query_key_width = key_width.checked_mul(2)?;
    let value_width = value_heads.checked_mul(value_head_dim)?;
    let channels = query_key_width.checked_add(value_width)?;
    Some(ProjectionWidths {
        key_width,
        value_width,
        channels,
    })
}

/// Channel layout of a mixer's convolved Q/K/V activations.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MixerLayout {
    /// Query channels are `0..key_width` and key channels `key_width..2 * key_width`.
    pub(super) key_width: usize,
    /// First value channel, `2 * key_width`.
    pub(super) value_start: usize,
    /// Value channels are `value_start..value_start + value_width`.
    pub(super) value_width: usize,
}

/// Splits a mixer's convolved Q/K/V channels into query, key, and value slices.
///
/// The proof shows the slices `0..K`, `K..2K`, and `2K..2K + V` (with
/// `K = key_heads * key_head_dim` and `V = value_heads * value_head_dim`) are
/// consecutive and end exactly at the fused channel count, so they tile the
/// convolution output with no gap or overlap. The precondition holds for a
/// mixer built from a validated configuration, whose channel count
/// [`projection_widths`] accepted; the arithmetic is the mixer's own, so
/// behavior is unchanged.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(layout =>
    requires
        2 * (key_heads * key_head_dim) + value_heads * value_head_dim <= usize::MAX,
    ensures
        layout.key_width == key_heads * key_head_dim,
        layout.value_start == layout.key_width + layout.key_width,
        layout.value_width == value_heads * value_head_dim,
        layout.value_start + layout.value_width
            == 2 * (key_heads * key_head_dim) + value_heads * value_head_dim,
))]
pub(super) fn mixer_layout(
    key_heads: usize,
    key_head_dim: usize,
    value_heads: usize,
    value_head_dim: usize,
) -> MixerLayout {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(key_heads * key_head_dim >= 0) by (nonlinear_arith);
        assert(value_heads * value_head_dim >= 0) by (nonlinear_arith);
    }
    let key_width = key_heads * key_head_dim;
    let value_width = value_heads * value_head_dim;
    MixerLayout {
        key_width,
        value_start: key_width * 2,
        value_width,
    }
}

/// Returns the elements `value_heads * key_head_dim * value_head_dim` of one row's recurrent state.
///
/// The proof shows the product is exact and that its partial product cannot
/// overflow first. The precondition holds whenever the state tensor
/// `[batch, value_heads, key_head_dim, value_head_dim]` of a validated mixer
/// exists, because its element count is then representable.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(elements =>
    requires
        key_head_dim > 0,
        value_head_dim > 0,
        value_heads * key_head_dim * value_head_dim <= usize::MAX,
    ensures
        elements == value_heads * key_head_dim * value_head_dim,
))]
pub(super) fn state_elements(
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(0 <= value_heads * key_head_dim <= value_heads * key_head_dim * value_head_dim)
            by (nonlinear_arith)
            requires key_head_dim > 0, value_head_dim > 0, value_heads >= 0;
    }
    value_heads * key_head_dim * value_head_dim
}

/// Returns how many consecutive value heads share each query/key head.
///
/// The proof shows `repetitions * key_heads == value_heads`, so repeating
/// each of the `key_heads` heads `repetitions` times yields exactly
/// `value_heads` contiguous groups. The precondition is the divisibility that
/// [`check_delta_geometry`] accepts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(repetitions =>
    requires
        key_heads > 0,
        value_heads % key_heads == 0,
    ensures
        repetitions * key_heads == value_heads,
))]
pub(super) fn head_repetitions(value_heads: usize, key_heads: usize) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(value_heads as int, key_heads as int);
        assert((value_heads / key_heads) * key_heads == value_heads) by (nonlinear_arith)
            requires
                value_heads == key_heads * (value_heads / key_heads) + value_heads % key_heads,
                value_heads % key_heads == 0;
    }
    value_heads / key_heads
}

/// First violated integer constraint of a Gated DeltaNet configuration.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DeltaGeometryViolation {
    /// A model width, head count, or the convolution kernel is zero.
    ZeroWidth,
    /// Value heads are not a whole multiple of key heads.
    ValueHeads,
}

/// Checks that every width is non-zero and value heads group key heads evenly.
///
/// The proof shows acceptance exactly when all six quantities are non-zero
/// and `key_heads` divides `value_heads`, with zero widths reported first.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(checked =>
    ensures
        match checked {
            Ok(()) => {
                &&& hidden > 0 && key_heads > 0 && value_heads > 0
                &&& key_head_dim > 0 && value_head_dim > 0 && kernel > 0
                &&& value_heads % key_heads == 0
            },
            Err(DeltaGeometryViolation::ZeroWidth) => {
                hidden == 0 || key_heads == 0 || value_heads == 0 || key_head_dim == 0
                    || value_head_dim == 0 || kernel == 0
            },
            Err(DeltaGeometryViolation::ValueHeads) => {
                &&& hidden > 0 && key_heads > 0 && value_heads > 0
                &&& key_head_dim > 0 && value_head_dim > 0 && kernel > 0
                &&& value_heads % key_heads != 0
            },
        },
))]
pub(super) fn check_delta_geometry(
    hidden: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    kernel: usize,
) -> Result<(), DeltaGeometryViolation> {
    if hidden == 0
        || key_heads == 0
        || value_heads == 0
        || key_head_dim == 0
        || value_head_dim == 0
        || kernel == 0
    {
        return Err(DeltaGeometryViolation::ZeroWidth);
    }
    if !value_heads.is_multiple_of(key_heads) {
        return Err(DeltaGeometryViolation::ValueHeads);
    }
    Ok(())
}

#[cfg(verus_keep_ghost)]
verus! {

/// Shorter segments and smaller states never switch the recurrence off.
///
/// If `L` tokens of a state with `S` elements prefer the recurrence, so do
/// `L' <= L` tokens of a state with `S' <= S` elements: the modelled
/// recurrence cost `L * min(6144 + S, usize::MAX)` only shrinks.
pub proof fn lemma_recurrence_choice_monotone(
    sequence: int,
    shorter: int,
    state_elements: int,
    smaller: int,
)
    requires
        1 <= shorter <= sequence,
        0 <= smaller <= state_elements,
        recurrence_preferred(sequence, state_elements),
    ensures
        recurrence_preferred(shorter, smaller),
{
    if shorter != 1 {
        let cost = saturated(6_144 + state_elements);
        let reduced = saturated(6_144 + smaller);
        assert(0 <= reduced <= cost);
        assert(shorter * reduced <= sequence * cost) by (nonlinear_arith)
            requires 1 <= shorter <= sequence, 0 <= reduced <= cost;
    }
}

/// The integer ceiling covers the numerator by less than one denominator and never exceeds it.
///
/// The shared `crate::division::lemma_ceiling_bounds` states the same facts
/// about the same `ceiling`. This module keeps its own proof because
/// replacing it with a call to the shared lemma perturbs the solver context
/// of the unrelated `inverse_levels` proof here enough to make that proof
/// fail (Verus 0.2026.10.04, Z3 4.16.0).
proof fn lemma_ceiling(numerator: int, denominator: int)
    requires
        numerator >= 0,
        denominator > 0,
    ensures
        ceiling(numerator, denominator) * denominator >= numerator,
        ceiling(numerator, denominator) * denominator < numerator + denominator,
        ceiling(numerator, denominator) <= numerator,
        numerator > 0 ==> ceiling(numerator, denominator) >= 1,
        ceiling(numerator, denominator) >= 0,
{
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(numerator, denominator);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(numerator, denominator);
    vstd::arithmetic::div_mod::lemma_div_pos_is_pos(numerator, denominator);
    let ceil = ceiling(numerator, denominator);
    if remainder == 0 {
        assert(ceil * denominator == numerator) by (nonlinear_arith)
            requires numerator == denominator * quotient, ceil == quotient;
        assert(ceil <= numerator) by (nonlinear_arith)
            requires numerator == denominator * quotient, ceil == quotient, denominator >= 1,
                quotient >= 0;
        if numerator > 0 {
            assert(ceil >= 1) by (nonlinear_arith)
                requires
                    numerator == denominator * quotient,
                    ceil == quotient,
                    numerator > 0,
                    denominator > 0,
                    quotient >= 0;
        }
    } else {
        assert(ceil * denominator == numerator - remainder + denominator) by (nonlinear_arith)
            requires numerator == denominator * quotient + remainder, ceil == quotient + 1;
        assert(quotient * denominator <= numerator - remainder) by (nonlinear_arith)
            requires numerator == denominator * quotient + remainder;
        assert(quotient + 1 <= numerator) by (nonlinear_arith)
            requires
                numerator == denominator * quotient + remainder,
                remainder >= 1,
                denominator >= 1,
                quotient >= 0;
    }
}

/// Balanced blocks of `ceil(chunks / limit)` hold between one and `limit` chunks.
proof fn lemma_balanced_sizes(chunks: int, limit: int, blocks: int)
    requires
        limit >= 1,
        chunks >= 1,
        blocks == ceiling(chunks, limit),
    ensures
        blocks >= 1,
        blocks <= chunks,
        blocks * (chunks / blocks) + chunks % blocks == chunks,
        chunks / blocks >= 1,
        chunks / blocks + if chunks % blocks > 0 { 1int } else { 0int } <= limit,
{
    lemma_ceiling(chunks, limit);
    let base = chunks / blocks;
    let extra = chunks % blocks;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(chunks, blocks);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(chunks, blocks);
    assert(blocks * base + extra == chunks) by (nonlinear_arith)
        requires chunks == blocks * base + extra;
    // At least one chunk per block, since there are no more blocks than chunks.
    assert(base >= 1) by (nonlinear_arith)
        requires chunks == blocks * base + extra, 0 <= extra < blocks, blocks <= chunks;
    // `blocks * limit >= chunks`, so the ceiling of `chunks / blocks` is at most `limit`.
    assert(base + if extra > 0 { 1int } else { 0int } <= limit) by (nonlinear_arith)
        requires
            chunks == blocks * base + extra,
            0 <= extra < blocks,
            blocks * limit >= chunks,
            blocks >= 1;
}

/// Block `index < blocks` ends where block `index + 1` starts, within the covered chunks.
proof fn lemma_block_start_bound(chunks: int, blocks: int, index: int)
    requires
        chunks >= 0,
        0 <= index < blocks,
        blocks * (chunks / blocks) + chunks % blocks == chunks,
        chunks / blocks >= 0,
        0 <= chunks % blocks < blocks,
    ensures
        balanced_start(chunks, blocks, index + 1)
            == balanced_start(chunks, blocks, index) + balanced_size(chunks, blocks, index),
        balanced_start(chunks, blocks, index + 1) <= chunks,
        balanced_start(chunks, blocks, index) >= 0,
{
    let base = chunks / blocks;
    let extra = chunks % blocks;
    assert((index + 1) * base == index * base + base) by (nonlinear_arith);
    assert((index + 1) * base <= blocks * base) by (nonlinear_arith)
        requires index + 1 <= blocks, base >= 0;
    assert(index * base >= 0) by (nonlinear_arith)
        requires index >= 0, base >= 0;
}

/// Doubling a power of two gives a power of two.
proof fn lemma_double_power_of_two(value: int)
    requires
        is_power_of_two(value),
    ensures
        is_power_of_two(2 * value),
{
    assert((2 * value) % 2 == 0);
    assert((2 * value) / 2 == value);
}

/// Halving a power of two of at least two gives a power of two.
proof fn lemma_halve_power_of_two(value: int)
    requires
        is_power_of_two(value),
        value >= 2,
    ensures
        is_power_of_two(value / 2),
        value == 2 * (value / 2),
        value / 2 >= 1,
{
}

} // verus!

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
