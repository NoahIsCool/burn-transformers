//! Host plans that pack ragged stateful rows into one batched block and back.
//!
//! A stateful call supplies a `[batch, S]` rectangle whose row `b` has valid
//! columns `c_{b,0} < … < c_{b,n_b-1}` (`n_b = 0` marks an inactive decode
//! row). The batched decoder evaluates a left-aligned `[batch, Q]` block with
//! `Q = max_b n_b`: slot `(b, i)` holds row `b`'s `i`-th valid token for
//! `i < n_b` and padding for `n_b <= i < Q`, so every row's new tokens occupy
//! columns `0..n_b` with padding after them, exactly the layout the cache's
//! batched append expects (slot `i` is written to cache position `L_b + i`).
//!
//! Packing and unpacking are gathers over flattened row-major indices:
//!
//! - compaction source of slot `(b, i)`: `b * S + c_{b,i}` for `i < n_b`, and
//!   `b * S` (any in-row column; the slot is overwritten) otherwise;
//! - scatter source of column `(b, s)`: `b * Q + k` when `s = c_{b,k}`, and
//!   `b * Q` otherwise (the column is zeroed after the gather).
//!
//! When every row's valid columns are a prefix `0..n_b` (no padding, right
//! padding, or inactive rows), both maps are views — a narrow and a zero
//! extension — and no index tensor is uploaded. [`compaction_sources`] and
//! [`scatter_sources`] are verified in place by Verus (`verus/t5_gpt.rs`):
//! for all inputs they never overflow, and every index they return addresses
//! the flattened source tensor. [`row_positions`], which plans every slot's
//! causal visibility and relative-bias position, is verified to compute
//! exactly its documented formulas, never letting a slot see past its row's
//! appended history.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// How a call's valid tokens map onto the batched `[batch, Q]` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Packing {
    /// Every row's valid columns are `0..n_b`: the block is the first `Q` columns.
    Prefix,
    /// Some row has leading or interior padding: flat gather indices.
    Gather {
        /// Compaction source of every block slot, `batch * Q` entries.
        compact: Vec<usize>,
        /// Scatter source of every input column, `batch * S` entries.
        scatter: Vec<usize>,
    },
}

/// Packing plan for one validated stateful call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BatchedRows {
    /// Rows `B`.
    pub(super) batch: usize,
    /// Input columns `S`, including padding.
    pub(super) sequence: usize,
    /// Block width `Q = max_b n_b`; zero when every row is inactive.
    pub(super) block: usize,
    /// New tokens `n_b` of every row.
    pub(super) counts: Vec<usize>,
    /// Kernel strategy for packing and unpacking.
    pub(super) packing: Packing,
}

impl BatchedRows {
    /// Plans a call from each row's sorted valid columns below `sequence`.
    ///
    /// Returns `None` when the columns are malformed (unsorted, duplicated,
    /// out of range) or a flat index would overflow `usize`; validated
    /// stateful inputs never are.
    pub(super) fn new(valid_columns: &[Vec<usize>], sequence: usize) -> Option<Self> {
        let batch = valid_columns.len();
        let counts: Vec<usize> = valid_columns.iter().map(Vec::len).collect();
        let block = counts.iter().copied().max().unwrap_or(0);
        if block > sequence {
            return None;
        }
        let prefix = valid_columns.iter().all(|columns| {
            columns
                .iter()
                .enumerate()
                .all(|(slot, &column)| slot == column)
        });
        let packing = if prefix {
            Packing::Prefix
        } else {
            let flat: Vec<usize> = valid_columns.iter().flatten().copied().collect();
            Packing::Gather {
                compact: compaction_sources(&counts, &flat, sequence, block)?,
                scatter: scatter_sources(&counts, &flat, sequence, block)?,
            }
        };
        Some(Self {
            batch,
            sequence,
            block,
            counts,
            packing,
        })
    }

    /// Whether block slot `(b, i)` is padding (`i >= n_b`), row-major over `[batch, Q]`.
    pub(super) fn padded_slots(&self) -> Vec<bool> {
        let mut padded = Vec::with_capacity(self.batch * self.block);
        for &count in &self.counts {
            padded.extend((0..self.block).map(|slot| slot >= count));
        }
        padded
    }

    /// Whether any block slot is padding.
    pub(super) fn has_padded_slots(&self) -> bool {
        self.counts.iter().any(|&count| count < self.block)
    }

    /// Whether every row is inactive (no new token anywhere).
    pub(super) fn is_empty(&self) -> bool {
        self.block == 0
    }
}

/// Lists the flat compaction source of every block slot.
///
/// `counts[b]` valid columns of row `b` are stored consecutively in
/// `columns`. Slot `(b, i)` reads `b * sequence + columns[offset_b + i]` for
/// `i < counts[b]` and `b * sequence` otherwise. Returns `None` for a count
/// above `block`, a column at or beyond `sequence`, too few columns, or a
/// flat index outside `batch * sequence` (which only malformed inputs reach).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    ensures
        match sources {
            Some(sources) => {
                &&& sources.len() == counts.len() * block
                &&& forall|j: int| 0 <= j < sources.len()
                    ==> #[trigger] sources[j] < counts.len() * sequence
            },
            None => true,
        },
))]
pub(super) fn compaction_sources(
    counts: &[usize],
    columns: &[usize],
    sequence: usize,
    block: usize,
) -> Option<Vec<usize>> {
    let batch = counts.len();
    let (limit, total) = match (batch.checked_mul(sequence), batch.checked_mul(block)) {
        (Some(limit), Some(total)) => (limit, total),
        _ => return None,
    };
    let mut sources: Vec<usize> = Vec::with_capacity(total);
    let mut offset: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == counts.len(),
            limit == batch * sequence,
            sources.len() == row * block,
            forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
        decreases batch - row,
    ))]
    while row < batch {
        let count = counts[row];
        let base = match row.checked_mul(sequence) {
            Some(base) if count <= block => base,
            _ => return None,
        };
        let mut slot: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                slot <= block,
                row < batch,
                sources.len() == row * block + slot,
                forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
            decreases block - slot,
        ))]
        while slot < block {
            let column = if slot < count {
                let index = match offset.checked_add(slot) {
                    Some(index) if index < columns.len() => index,
                    _ => return None,
                };
                columns[index]
            } else {
                0
            };
            let source = match base.checked_add(column) {
                Some(source) if column < sequence && source < limit => source,
                _ => return None,
            };
            sources.push(source);
            slot += 1;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row * block + block == (row + 1) * block) by (nonlinear_arith);
        }
        offset = offset.checked_add(count)?;
        row += 1;
    }
    Some(sources)
}

/// Lists the flat scatter source of every input column.
///
/// Column `(b, s)` reads block slot `b * block + k` when `s` is row `b`'s
/// `k`-th valid column and `b * block` otherwise. Returns `None` under the
/// conditions of [`compaction_sources`] and for unsorted or duplicated
/// columns. With `block = 0` every row is inactive and every source is `0`
/// (callers zero the whole output).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(sources =>
    ensures
        match sources {
            Some(sources) => {
                &&& sources.len() == counts.len() * sequence
                &&& block > 0 ==> forall|j: int| 0 <= j < sources.len()
                    ==> #[trigger] sources[j] < counts.len() * block
            },
            None => true,
        },
))]
pub(super) fn scatter_sources(
    counts: &[usize],
    columns: &[usize],
    sequence: usize,
    block: usize,
) -> Option<Vec<usize>> {
    let batch = counts.len();
    let (limit, total) = match (batch.checked_mul(block), batch.checked_mul(sequence)) {
        (Some(limit), Some(total)) => (limit, total),
        _ => return None,
    };
    let mut sources: Vec<usize> = Vec::with_capacity(total);
    let mut offset: usize = 0;
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= batch,
            batch == counts.len(),
            limit == batch * block,
            sources.len() == row * sequence,
            block > 0 ==> forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
        decreases batch - row,
    ))]
    while row < batch {
        let count = counts[row];
        let base = match row.checked_mul(block) {
            Some(base) if count <= block && (block == 0 || base < limit) => base,
            _ => return None,
        };
        // `rank` walks the row's sorted columns while `column` scans 0..sequence.
        let mut rank: usize = 0;
        let mut column: usize = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                column <= sequence,
                rank <= count,
                row < batch,
                block == 0 || base < limit,
                sources.len() == row * sequence + column,
                block > 0 ==> forall|j: int| 0 <= j < sources.len() ==> #[trigger] sources[j] < limit,
            decreases sequence - column,
        ))]
        while column < sequence {
            let mut source = base;
            if rank < count {
                let index = match offset.checked_add(rank) {
                    Some(index) if index < columns.len() => index,
                    _ => return None,
                };
                if columns[index] == column {
                    source = match base.checked_add(rank) {
                        Some(source) if source < limit => source,
                        _ => return None,
                    };
                    rank += 1;
                }
            }
            sources.push(source);
            column += 1;
        }
        if rank != count {
            // Some valid column was unsorted, duplicated, or out of range.
            return None;
        }
        #[cfg(verus_keep_ghost)]
        proof! {
            assert(row * sequence + sequence == (row + 1) * sequence) by (nonlinear_arith);
        }
        offset = offset.checked_add(count)?;
        row += 1;
    }
    Some(sources)
}

/// Query positions of one row's block slots: `(visible, query)`.
///
/// Row `b` has committed length `L = length` and `n = count` new tokens in a
/// block of `Q = block` slots. Slot `i` is the row's `i`-th new token when
/// `i < n` and padding otherwise. Two positions are planned per slot:
///
/// - `query[i] = L + i`, the absolute position whose relative bias the slot
///   uses (padding slots continue the sequence; their outputs are discarded);
/// - `visible[i]`, the last key position the slot may attend to under the
///   causal rule `key <= visible[i]`: `L + i` for a new token, the row's last
///   new position `L + n - 1` for a padding slot (so no padded query row is
///   ever empty), and `L - 1` (only the committed history) when the row is
///   inactive (`n = 0`).
///
/// Consequently no slot of an active row sees a key beyond `L + n - 1`, the
/// row's own appended history. Returns `None` when `count > block` or a
/// position would not fit `i64` or `usize` (`length + block` overflows).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(positions =>
    ensures
        match positions {
            Some((visible, query)) => {
                &&& count <= block
                &&& visible.len() == block
                &&& query.len() == block
                &&& forall|i: int| 0 <= i < block ==> #[trigger] query[i] == length + i
                &&& forall|i: int| 0 <= i < block ==> #[trigger] visible[i] == (
                    if count == 0 {
                        length - 1
                    } else if i < count {
                        length + i
                    } else {
                        length + count - 1
                    })
                &&& count > 0 ==> forall|i: int| 0 <= i < block
                    ==> #[trigger] visible[i] <= length + count - 1
            },
            None => count > block || length + block > usize::MAX || length + block > i64::MAX,
        },
))]
pub(super) fn row_positions(
    length: usize,
    count: usize,
    block: usize,
) -> Option<(Vec<i64>, Vec<i64>)> {
    if count > block {
        return None;
    }
    let end = length.checked_add(block)?;
    // `usize` has at most 64 bits, so widening to `u64` is lossless.
    if end as u64 > i64::MAX as u64 {
        return None;
    }
    let start = length as i64;
    let mut visible: Vec<i64> = Vec::with_capacity(block);
    let mut query: Vec<i64> = Vec::with_capacity(block);
    let mut slot: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            slot <= block,
            count <= block,
            length + block <= i64::MAX,
            start == length,
            visible.len() == slot,
            query.len() == slot,
            forall|i: int| 0 <= i < slot ==> #[trigger] query[i] == length + i,
            forall|i: int| 0 <= i < slot ==> #[trigger] visible[i] == (
                if count == 0 {
                    length - 1
                } else if i < count {
                    length + i
                } else {
                    length + count - 1
                }),
        decreases block - slot,
    ))]
    while slot < block {
        let offset = slot as i64;
        query.push(start + offset);
        let shown = if count == 0 {
            start - 1
        } else if slot < count {
            start + offset
        } else {
            start + count as i64 - 1
        };
        visible.push(shown);
        slot += 1;
    }
    Some((visible, query))
}

/// Whether `indices` keeps every row inside its group of `group_size` consecutive rows.
///
/// True exactly when `group_size > 0`, `group_size` divides the batch
/// `indices.len()`, and `indices[d] / group_size == d / group_size` for every
/// row `d`. Beam search reorders only within a source's beams, which share
/// one encoder binding, so such a reorder can leave that binding in place.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(preserved =>
    ensures
        preserved == (group_size > 0
            && indices.len() % group_size == 0
            && forall|d: int| 0 <= d < indices.len()
                ==> #[trigger] indices[d] / group_size == d / group_size as int),
))]
pub(super) fn groups_preserved(indices: &[usize], group_size: usize) -> bool {
    if group_size == 0 || !indices.len().is_multiple_of(group_size) {
        return false;
    }
    let mut row: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            row <= indices.len(),
            group_size > 0,
            forall|d: int| 0 <= d < row ==> #[trigger] indices[d] / group_size == d / group_size as int,
        decreases indices.len() - row,
    ))]
    while row < indices.len() {
        if indices[row] / group_size != row / group_size {
            return false;
        }
        row += 1;
    }
    true
}

#[cfg(test)]
#[path = "packing_tests.rs"]
mod tests;
