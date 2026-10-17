//! Pure byte planning for GPT-NeoX packed query/key/value projections.
//!
//! # Packed layout
//!
//! A GPT-NeoX checkpoint stores the attention input projection as one tensor
//! whose `R = 3 q d` rows are ordered `[head, component, head_channel]`, with
//! `q` query heads of width `d` and components query, key, value (`c = 0, 1,
//! 2`). A weight row holds `h` input scalars and a bias row one scalar, so with
//! `e` bytes per scalar every row is `w = (h or 1) e` bytes and the payload is
//! `R w` bytes. Head `j`'s component `c` is the contiguous byte block
//!
//! `[(3 j + c) b, (3 j + c + 1) b)`, where `b = d w`.
//!
//! Component `c` is rebuilt by concatenating its blocks for `j = 0..q`, so its
//! output byte `o` (head `floor(o / b)`, offset `o mod b`) is source byte
//!
//! `s_c(o) = (3 floor(o / b) + c) b + (o mod b)`.
//!
//! The map `(c, o) -> s_c(o)` is a bijection from `{0, 1, 2} x [0, q b)` onto
//! `[0, 3 q b)`: every packed byte lands in exactly one component position, so
//! the three projections partition the payload with nothing dropped or read
//! twice.
//!
//! # Verified properties
//!
//! Verus checks every function here in place through `verus/loaders.rs` (see
//! `verus/README.md`):
//!
//! - [`packed_rows`] accepts exactly the geometries the sink can split (nonzero
//!   extents, configured head and residual widths, `q d = h`, and as many
//!   key/value heads as query heads) and then returns `R = 3 q d` and the
//!   per-component row count `q d`, or reports that `R` overflows `usize`;
//! - [`packed_payload`] returns `w = (h or 1) e` and `R w` exactly, or `None`
//!   exactly when either product overflows `usize`;
//! - [`packed_component`] never indexes outside the payload and returns `q b`
//!   bytes whose byte `o` is payload byte `s_c(o)`, and
//!   `lemma_packed_source_bijection` proves the partition above;
//! - [`packed_staging_peak`] is the exact peak `P + max_i floor(B_i / 3)` of
//!   host staging when one component buffer of the largest packed tensor is
//!   live next to the packed payload, or `None` exactly when it overflows
//!   `u64`.
//!
//! Building Burn `TensorData`, dtype sizes, and installing parameters stay in
//! the unverified sink in `load.rs`, covered by the GPT-NeoX load tests.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Payload byte copied to output byte `offset` of component `component`, for
/// per-head component blocks of `block` bytes: `(3 floor(o / b) + c) b + o mod b`.
pub open spec fn packed_source(component: int, offset: int, block: int) -> int {
    ((offset / block) * 3 + component) * block + offset % block
}

/// Component whose block holds payload byte `source`: `floor(source / b) mod 3`.
pub open spec fn packed_component_of(source: int, block: int) -> int {
    (source / block) % 3
}

/// Output position within its component of payload byte `source`:
/// `floor(floor(source / b) / 3) b + source mod b`.
pub open spec fn packed_offset_of(source: int, block: int) -> int {
    ((source / block) / 3) * block + source % block
}

/// Largest per-component share `floor(B_i / 3)` of the first `count` packed
/// storage sizes, zero when there are none.
pub open spec fn largest_component(storage: Seq<u64>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        0
    } else {
        let earlier = largest_component(storage, count - 1);
        let share = storage[count - 1] as int / 3;
        if share > earlier {
            share
        } else {
            earlier
        }
    }
}

/// Output byte `o = j b + r` of head `j` reads payload byte `(3 j + c) b + r`.
proof fn lemma_head_offset(head: int, rest: int, block: int, component: int)
    requires
        head >= 0,
        block > 0,
        0 <= rest < block,
    ensures
        (head * block + rest) / block == head,
        (head * block + rest) % block == rest,
        packed_source(component, head * block + rest, block) == (head * 3 + component) * block
            + rest,
{
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        head * block + rest,
        block,
        head,
        rest,
    );
}

/// Every output position maps into the payload and names its own head block.
proof fn lemma_packed_source_in_range(heads: int, block: int, component: int, offset: int)
    requires
        block > 0,
        0 <= component < 3,
        0 <= offset < heads * block,
    ensures
        0 <= offset / block < heads,
        0 <= offset % block < block,
        0 <= packed_source(component, offset, block) < 3 * heads * block,
        packed_source(component, offset, block) == (3 * (offset / block) + component) * block
            + offset % block,
{
    let head = offset / block;
    let rest = offset % block;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(offset, block);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(offset, block);
    assert(0 <= head < heads) by (nonlinear_arith)
        requires
            offset == block * head + rest,
            0 <= rest < block,
            0 <= offset < heads * block,
    ;
    assert(0 <= (head * 3 + component) * block + rest < 3 * heads * block) by (nonlinear_arith)
        requires
            0 <= head < heads,
            0 <= component < 3,
            0 <= rest < block,
    ;
}

/// The component split is a bijection from `{0, 1, 2} x [0, q b)` onto `[0, 3 q b)`.
///
/// Distinct component positions read distinct payload bytes, every one of
/// them inside the payload, and every payload byte is read by some position.
pub proof fn lemma_packed_source_bijection(heads: int, block: int)
    requires
        heads >= 0,
        block > 0,
    ensures
        forall|component: int, offset: int|
            0 <= component < 3 && 0 <= offset < heads * block ==> 0 <= #[trigger] packed_source(
                component,
                offset,
                block,
            ) < 3 * heads * block,
        forall|first: int, first_offset: int, second: int, second_offset: int|
            0 <= first < 3 && 0 <= first_offset < heads * block && 0 <= second < 3 && 0
                <= second_offset < heads * block && #[trigger] packed_source(
                first,
                first_offset,
                block,
            ) == #[trigger] packed_source(second, second_offset, block) ==> first == second
                && first_offset == second_offset,
        forall|source: int|
            0 <= source < 3 * heads * block ==> {
                let component = #[trigger] packed_component_of(source, block);
                let offset = packed_offset_of(source, block);
                &&& 0 <= component < 3
                &&& 0 <= offset < heads * block
                &&& packed_source(component, offset, block) == source
            },
{
    assert forall|component: int, offset: int|
        0 <= component < 3 && 0 <= offset < heads * block implies 0 <= #[trigger] packed_source(
        component,
        offset,
        block,
    ) < 3 * heads * block by {
        lemma_packed_source_in_range(heads, block, component, offset);
    }
    assert forall|first: int, first_offset: int, second: int, second_offset: int|
        0 <= first < 3 && 0 <= first_offset < heads * block && 0 <= second < 3 && 0
            <= second_offset < heads * block && #[trigger] packed_source(
            first,
            first_offset,
            block,
        ) == #[trigger] packed_source(second, second_offset, block) implies first == second
        && first_offset == second_offset by {
        lemma_packed_source_injective(heads, block, first, first_offset, second, second_offset);
    }
    assert forall|source: int| 0 <= source < 3 * heads * block implies {
        let component = #[trigger] packed_component_of(source, block);
        let offset = packed_offset_of(source, block);
        &&& 0 <= component < 3
        &&& 0 <= offset < heads * block
        &&& packed_source(component, offset, block) == source
    } by {
        lemma_packed_source_preimage(heads, block, source);
    }
}

/// Two component positions that read the same payload byte are the same position.
proof fn lemma_packed_source_injective(
    heads: int,
    block: int,
    first: int,
    first_offset: int,
    second: int,
    second_offset: int,
)
    requires
        block > 0,
        0 <= first < 3,
        0 <= second < 3,
        0 <= first_offset < heads * block,
        0 <= second_offset < heads * block,
        packed_source(first, first_offset, block) == packed_source(second, second_offset, block),
    ensures
        first == second,
        first_offset == second_offset,
{
    lemma_packed_source_in_range(heads, block, first, first_offset);
    lemma_packed_source_in_range(heads, block, second, second_offset);
    let first_head = first_offset / block;
    let second_head = second_offset / block;
    let first_rest = first_offset % block;
    let second_rest = second_offset % block;
    let source = packed_source(first, first_offset, block);
    // The block index and offset of the shared source byte are unique.
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        source,
        block,
        3 * first_head + first,
        first_rest,
    );
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        source,
        block,
        3 * second_head + second,
        second_rest,
    );
    // So are the head and component of that block index.
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        3 * first_head + first,
        3,
        first_head,
        first,
    );
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod_converse(
        3 * second_head + second,
        3,
        second_head,
        second,
    );
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(first_offset, block);
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(second_offset, block);
}

/// Every payload byte is read by the component position decoded from its block.
proof fn lemma_packed_source_preimage(heads: int, block: int, source: int)
    requires
        heads >= 0,
        block > 0,
        0 <= source < 3 * heads * block,
    ensures
        0 <= packed_component_of(source, block) < 3,
        0 <= packed_offset_of(source, block) < heads * block,
        packed_source(packed_component_of(source, block), packed_offset_of(source, block), block)
            == source,
{
    let index = source / block;
    let rest = source % block;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(source, block);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(source, block);
    assert(0 <= index < 3 * heads) by (nonlinear_arith)
        requires
            source == block * index + rest,
            0 <= rest < block,
            0 <= source < 3 * heads * block,
    ;
    let head = index / 3;
    let component = index % 3;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(index, 3);
    vstd::arithmetic::div_mod::lemma_mod_pos_bound(index, 3);
    assert(0 <= head < heads) by (nonlinear_arith)
        requires
            index == 3 * head + component,
            0 <= component < 3,
            0 <= index < 3 * heads,
    ;
    let offset = head * block + rest;
    assert(0 <= offset < heads * block) by (nonlinear_arith)
        requires
            0 <= head < heads,
            0 <= rest < block,
            offset == head * block + rest,
    ;
    lemma_head_offset(head, rest, block, component);
    assert((head * 3 + component) * block + rest == source) by (nonlinear_arith)
        requires
            index == 3 * head + component,
            source == block * index + rest,
    ;
}

/// The largest component share bounds every share and is attained (or zero).
pub proof fn lemma_largest_component_bounds(storage: Seq<u64>, count: int)
    requires
        0 <= count <= storage.len(),
    ensures
        largest_component(storage, count) >= 0,
        forall|index: int|
            0 <= index < count ==> #[trigger] storage[index] as int / 3 <= largest_component(
                storage,
                count,
            ),
        count == 0 ==> largest_component(storage, count) == 0,
        count > 0 ==> exists|index: int|
            0 <= index < count && #[trigger] storage[index] as int / 3 == largest_component(
                storage,
                count,
            ),
    decreases count,
{
    if count > 0 {
        lemma_largest_component_bounds(storage, count - 1);
        if count > 1 {
            let earlier = largest_component(storage, count - 1);
            let witness = choose|index: int|
                0 <= index < count - 1 && #[trigger] storage[index] as int / 3 == earlier;
            if storage[count - 1] as int / 3 <= earlier {
                assert(storage[witness] as int / 3 == largest_component(storage, count));
            }
        }
    }
}

} // verus!

/// Row counts of a packed projection whose geometry the sink can split.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackedRows {
    /// Rows of the packed tensor, `3 q d`.
    pub(super) rows: usize,
    /// Rows of each component projection, `q d` (the residual width).
    pub(super) component_rows: usize,
}

/// Why a packed projection cannot be split.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PackedQkvError {
    /// Head counts or widths are zero or disagree with the configuration.
    Geometry,
    /// The packed row count `3 q d` does not fit `usize`.
    Overflow,
}

/// Validates a layer's attention geometry and returns its packed row counts.
///
/// The layer is accepted exactly when `q`, `d`, and its residual width are
/// nonzero, `q` and the residual width equal the configured values, `q d`
/// equals the configured residual width, and there are as many key/value heads
/// as query heads (GPT-NeoX has no grouped heads). The result is `R = 3 q d`
/// rows with `q d` rows per component; `Overflow` means only that `R` does not
/// fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(rows =>
    ensures
        ({
            let valid = heads > 0 && head_dim > 0 && hidden > 0 && heads == expected_heads
                && hidden == expected_hidden && heads * head_dim == expected_hidden
                && key_value_heads == heads;
            match rows {
                Ok(rows) => {
                    &&& valid
                    &&& rows.component_rows == heads * head_dim
                    &&& rows.component_rows == hidden
                    &&& rows.rows == 3 * (heads * head_dim)
                },
                Err(PackedQkvError::Geometry) => !valid,
                Err(PackedQkvError::Overflow) => valid && 3 * (heads * head_dim) > usize::MAX,
            }
        }),
))]
pub(super) fn packed_rows(
    heads: usize,
    head_dim: usize,
    hidden: usize,
    expected_heads: usize,
    expected_hidden: usize,
    key_value_heads: usize,
) -> Result<PackedRows, PackedQkvError> {
    let width = heads.checked_mul(head_dim);
    if heads == 0
        || head_dim == 0
        || hidden == 0
        || heads != expected_heads
        || hidden != expected_hidden
        || width != Some(expected_hidden)
        || key_value_heads != heads
    {
        return Err(PackedQkvError::Geometry);
    }
    match expected_hidden.checked_mul(3) {
        Some(rows) => Ok(PackedRows {
            rows,
            component_rows: expected_hidden,
        }),
        None => Err(PackedQkvError::Overflow),
    }
}

/// Byte sizes of one packed row and of the whole packed payload.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PackedPayload {
    /// Bytes per packed row, `w = row_elements * element_bytes`.
    pub(super) row_bytes: usize,
    /// Bytes of the whole packed tensor, `R w`.
    pub(super) bytes: usize,
}

/// Computes the row and total byte sizes of a packed tensor.
///
/// Returns `w = row_elements * element_bytes` and `rows * w`, or `None`
/// exactly when either product does not fit `usize`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(payload =>
    ensures
        match payload {
            Some(payload) => {
                &&& payload.row_bytes == row_elements * element_bytes
                &&& payload.bytes == rows * (row_elements * element_bytes)
            },
            None => row_elements * element_bytes > usize::MAX
                || rows * (row_elements * element_bytes) > usize::MAX,
        },
))]
pub(super) fn packed_payload(
    rows: usize,
    row_elements: usize,
    element_bytes: usize,
) -> Option<PackedPayload> {
    let row_bytes = row_elements.checked_mul(element_bytes)?;
    let bytes = rows.checked_mul(row_bytes)?;
    Some(PackedPayload { row_bytes, bytes })
}

/// Copies one component's head blocks out of a packed payload, in head order.
///
/// With `b = head_dim * row_bytes`, the result has `q b` bytes and its byte `o`
/// is payload byte `(3 floor(o / b) + c) b + o mod b`; together with
/// `lemma_packed_source_bijection` this means the three components partition
/// the payload. The caller establishes the precondition from [`packed_rows`]
/// and [`packed_payload`] and the payload-length check.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
    requires
        heads > 0,
        head_dim > 0,
        component < 3,
        3 * (heads * head_dim) <= usize::MAX,
        payload@.len() == 3 * (heads * head_dim) * row_bytes,
    ensures
        bytes@.len() == heads * (head_dim * row_bytes),
        forall|offset: int|
            0 <= offset < bytes@.len() ==> #[trigger] bytes@[offset] == payload@[packed_source(
                component as int,
                offset,
                head_dim * row_bytes,
            )],
))]
pub(super) fn packed_component(
    payload: &[u8],
    heads: usize,
    head_dim: usize,
    row_bytes: usize,
    component: usize,
) -> Vec<u8> {
    let total = payload.len();
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(3 * heads <= 3 * (heads * head_dim)) by (nonlinear_arith)
            requires head_dim > 0, heads > 0;
        assert(payload@.len() == 3 * heads * (head_dim * row_bytes)) by (nonlinear_arith)
            requires payload@.len() == 3 * (heads * head_dim) * row_bytes;
        assert(head_dim * row_bytes <= payload@.len()) by (nonlinear_arith)
            requires
                payload@.len() == 3 * heads * (head_dim * row_bytes),
                heads > 0,
                head_dim * row_bytes >= 0;
    }
    let block = head_dim * row_bytes;
    // One third of the payload, exactly the `q b` bytes gathered below.
    let mut bytes = Vec::with_capacity(total / 3);
    let mut head = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            head <= heads,
            component < 3,
            3 * heads <= usize::MAX,
            total == payload@.len(),
            block == head_dim * row_bytes,
            payload@.len() == 3 * heads * block,
            bytes@.len() == head * block,
            forall|offset: int|
                0 <= offset < head * block ==> #[trigger] bytes@[offset]
                    == payload@[packed_source(component as int, offset, block as int)],
        decreases heads - head,
    ))]
    while head < heads {
        #[cfg(verus_keep_ghost)]
        proof! {
            assert((head * 3 + component) * block + block <= 3 * heads * block) by (nonlinear_arith)
                requires
                    head < heads,
                    component < 3,
                    block >= 0;
            assert((head * 3 + component) * block <= (head * 3 + component) * block + block);
            assert(head * block + block == (head + 1) * block) by (nonlinear_arith);
        }
        let start = (head * 3 + component) * block;
        let end = start + block;
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = bytes@;
        }
        bytes.extend_from_slice(&payload[start..end]);
        #[cfg(verus_keep_ghost)]
        proof! {
            assert forall|offset: int| 0 <= offset < (head + 1) * block implies #[trigger] bytes@[offset]
                == payload@[packed_source(component as int, offset, block as int)] by {
                if offset >= head * block {
                    let rest = offset - head * block;
                    lemma_head_offset(head as int, rest, block as int, component as int);
                    assert(bytes@[offset] == payload@[start + rest]);
                } else {
                    assert(bytes@[offset] == before[offset]);
                }
            }
        }
        head += 1;
    }
    bytes
}

/// Peak host staging when one packed component buffer is live.
///
/// Returns `P + max_i floor(B_i / 3)` for base peak `P` and packed storage
/// sizes `B_i` (adding zero when there are none), or `None` exactly when that
/// sum does not fit `u64`. `lemma_largest_component_bounds` shows the maximum
/// bounds every `floor(B_i / 3)` and is attained.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(staged =>
    ensures
        match staged {
            Some(staged) => staged == peak + largest_component(storage@, storage@.len() as int),
            None => peak + largest_component(storage@, storage@.len() as int) > u64::MAX,
        },
))]
pub(super) fn packed_staging_peak(peak: u64, storage: &[u64]) -> Option<u64> {
    let mut largest: u64 = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= storage.len(),
            largest == largest_component(storage@, index as int),
        decreases storage.len() - index,
    ))]
    while index < storage.len() {
        largest = largest.max(storage[index] / 3);
        index += 1;
    }
    peak.checked_add(largest)
}

#[cfg(test)]
#[path = "packed_tests.rs"]
mod tests;
