//! Exact decoding of MXFP4 (OCP microscaling FP4) checkpoint tensors.
//!
//! # Format
//!
//! The OCP Microscaling Formats (MX) v1.0 element type MXFP4 stores a *block*
//! of 32 values as 32 four-bit FP4 E2M1 codes that share one eight-bit E8M0
//! power-of-two scale. gpt-oss checkpoints store every routed-expert
//! projection as a pair of tensors:
//!
//! * `*_blocks`, `u8 [experts, rows, groups, 16]`: byte `b` of block `g` of row
//!   `r` packs decoded column `32 g + 2 b` in its low nibble and column
//!   `32 g + 2 b + 1` in its high nibble;
//! * `*_scales`, `u8 [experts, rows, groups]` (or the identical `F8_E8M0`
//!   encoding): the scale code `s` shared by block `g` of row `r`.
//!
//! An FP4 E2M1 code `c = 8σ + m` (sign `σ`, magnitude code `m = 0..=7`) is
//! `(-1)^σ · v(m)` with `v = [0, 0.5, 1, 1.5, 2, 3, 4, 6]`: `m = 1` is the
//! subnormal `0.5`, and `m >= 2` is `(1 + (m mod 2)/2) · 2^(⌊m/2⌋ - 1)`. The
//! block value is `fp4(c) · 2^(s - 127)`.
//!
//! # Reference semantics
//!
//! Transformers 5.2.0 (`integrations/mxfp4.py`, `convert_moe_packed_tensors`)
//! decodes rows `[experts, rows, 32 · groups]` and returns them transposed,
//! `[experts, 32 · groups, rows]`: the input-major `[in, out]` expert
//! parameter its MoE multiplies as `x @ W`. [`decode_transposed`] produces
//! exactly that tensor:
//!
//! * every value with `s <= 254` is `fp4(c) · 2^(s - 127)` rounded **once** to
//!   the target dtype (round to nearest, ties to even). Such values are exact
//!   in binary64, so [`exact_value_bits`] encodes them exactly and the caller's
//!   encoder rounds that encoding (F32 and BF16 are exact except overflow to
//!   infinity at `|value| >= 2^128`);
//! * scale code 255 (E8M0 "NaN") follows Transformers, which evaluates the
//!   factor `2^128` as `+inf`: zero codes decode to NaN and every other code
//!   to `±inf`. Every NaN is the positive quiet NaN, which is what PyTorch
//!   writes on aarch64 (and what its BF16/F16 converters canonicalize to).
//!
//! These rules reproduce `convert_moe_packed_tensors(..., dtype=bfloat16)` and
//! `dtype=float32` bit for bit, and `from_pretrained`'s F16 parameters (which
//! Transformers decodes to BF16 and then casts) bit for bit; see
//! `mxfp4_tests.rs`. Calling `convert_moe_packed_tensors(..., dtype=float16)`
//! directly instead rounds the factor `2^(s - 127)` to F16 before multiplying,
//! which differs only for exponents outside `[-24, 15]` or results below
//! `2^-14`; the pinned gpt-oss-20b scales use exponents `-12..=9` only, where
//! all three decodings coincide and every decoded weight is exact in F16.
//!
//! # Verification
//!
//! The integer core of this module (shape validation with overflow-free
//! extents and an exact classification of every rejected shape, the block,
//! scale, and value counts, block and scale offsets, the transposed output
//! offset, the decode table's index bijection, the thread partition of the
//! decode into whole units, each unit's coordinates and output slice, the
//! nibble split, and the exact binary64 encoding of a decoded value) has
//! no crate or Burn dependencies and is verified in place by Verus through
//! `verus/gpt_oss.rs`. The byte copies of the table-lookup kernel and the
//! dtype encoders are ordinary Rust, tested bit for bit against Transformers
//! over every `(scale, code)` pair and on a seeded multi-expert tensor.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Bytes of one MXFP4 block: 16 bytes pack 32 four-bit codes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const BLOCK_BYTES: usize = 16;

/// Values of one MXFP4 block, all sharing one scale.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const BLOCK_VALUES: usize = 32;

/// Number of E8M0 scale codes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const SCALE_CODES: usize = 256;

/// Number of FP4 E2M1 codes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const FP4_CODES: usize = 16;

/// Entries of a decode table, one per `(scale, code)` pair, indexed `16 · scale + code`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const TABLE_ENTRIES: usize = SCALE_CODES * FP4_CODES;

/// Binary64 encoding of the positive quiet NaN Transformers produces for scale code 255.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const CANONICAL_NAN_BITS: u64 = 0x7FF8_0000_0000_0000;

/// Binary64 encoding of `+inf`, the magnitude of nonzero codes under scale code 255.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const INFINITY_BITS: u64 = 0x7FF0_0000_0000_0000;

/// Value of the binary64 sign bit, `2^63`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

/// Value of one binary64 exponent step, `2^52`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const EXPONENT_UNIT: u64 = 0x0010_0000_0000_0000;

/// Value of the leading binary64 fraction bit, `2^51` (the mantissa of `1.5`).
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const HALF_FRACTION: u64 = 0x0008_0000_0000_0000;

/// Biased binary64 exponent of `2^(s - 128)`, minus `s`: `1023 - 128`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const EXPONENT_OFFSET: u64 = 895;

/// Fewest decoded values worth a second worker thread.
///
/// Below this a decode takes microseconds and thread start-up would dominate.
#[cfg_attr(verus_keep_ghost, verus_verify)]
pub(crate) const PARALLEL_MIN_VALUES: usize = 1 << 20;

/// Extents of one validated `[experts, rows, groups, 16]` MXFP4 tensor pair.
///
/// Every derived count ([`Self::columns`], [`Self::values`],
/// [`Self::block_bytes`], [`Self::scale_count`]) is representable in `usize`
/// and in `u64`, which [`mxfp4_geometry`] establishes once.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mxfp4Geometry {
    /// Leading (expert) axis `E`.
    pub(crate) experts: usize,
    /// Rows per expert `R`: output features of a projection.
    pub(crate) rows: usize,
    /// Blocks per row `G`; a row decodes to `32 G` columns.
    pub(crate) groups: usize,
    /// Decoded columns per row, `32 G`: input features of a projection.
    pub(crate) columns: usize,
}

/// Why a blocks/scales shape pair is not an MXFP4 tensor this decoder accepts.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mxfp4ShapeError {
    /// Blocks are not rank four with a trailing axis of 16 bytes.
    Blocks,
    /// Scales are not rank three or differ from the blocks' leading three axes.
    Scales,
    /// An axis is zero.
    Empty,
    /// The decoded element count or a byte count overflows `usize` or `u64`.
    Overflow,
}

impl core::fmt::Display for Mxfp4ShapeError {
    /// Completes "blocks `[..]` and scales `[..]` ..." with the violated rule.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::Blocks => "are not rank-4 blocks [experts, rows, groups, 16]",
            Self::Scales => "disagree: scales must be [experts, rows, groups] like the blocks",
            Self::Empty => "have an empty axis",
            Self::Overflow => "decode to more values than this host can address",
        })
    }
}

/// Validates a blocks/scales shape pair and returns the decoded shape `[E, 32 G, R]`.
///
/// # Errors
///
/// Returns the [`Mxfp4ShapeError`] of [`mxfp4_geometry`].
pub(crate) fn decoded_shape(blocks: &[u64], scales: &[u64]) -> Result<Vec<u64>, Mxfp4ShapeError> {
    let geometry = mxfp4_geometry(blocks, scales)?;
    // Every extent fits `u64`: the geometry bounds their product.
    Ok(vec![
        geometry.experts as u64,
        geometry.columns as u64,
        geometry.rows as u64,
    ])
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: blocks are rank four with a trailing axis of 16 bytes.
pub(crate) open spec fn blocks_shaped(blocks: Seq<u64>) -> bool {
    blocks.len() == 4 && blocks[3] == 16
}

/// Specification: scales are rank three and equal the blocks' leading axes.
pub(crate) open spec fn scales_match(blocks: Seq<u64>, scales: Seq<u64>) -> bool {
    &&& scales.len() == 3
    &&& scales[0] == blocks[0] && scales[1] == blocks[1] && scales[2] == blocks[2]
}

/// Specification: the expert, row, and group axes are all nonzero.
pub(crate) open spec fn axes_nonzero(blocks: Seq<u64>) -> bool {
    blocks[0] > 0 && blocks[1] > 0 && blocks[2] > 0
}

/// Specification: the decoded value count `E · R · 32 G` fits `usize` and `u64`.
pub(crate) open spec fn count_fits(blocks: Seq<u64>) -> bool {
    &&& blocks[0] * blocks[1] * (32 * blocks[2]) <= usize::MAX
    &&& blocks[0] * blocks[1] * (32 * blocks[2]) <= u64::MAX
}

/// Specification: the shape pairs [`mxfp4_geometry`] accepts, and no others.
///
/// Blocks `[E, R, G, 16]`, scales `[E, R, G]`, nonzero `E`, `R`, `G`, and a
/// decoded value count `E · R · 32 G` that fits both `usize` and `u64`.
pub(crate) open spec fn accepts(blocks: Seq<u64>, scales: Seq<u64>) -> bool {
    &&& blocks_shaped(blocks)
    &&& scales_match(blocks, scales)
    &&& axes_nonzero(blocks)
    &&& count_fits(blocks)
}

/// Specification: geometry `g` describes `[e, r, q, 16]` blocks and fits every count in `usize`.
pub(crate) open spec fn describes(g: Mxfp4Geometry) -> bool {
    &&& g.experts > 0 && g.rows > 0 && g.groups > 0
    &&& g.columns == 32 * g.groups
    &&& g.experts * g.rows * g.columns <= usize::MAX
    &&& g.experts * g.rows * g.columns <= u64::MAX
}

/// A row-major offset `(i·b + j)·c + k` and its partial products stay below `a·b·c`.
pub proof fn lemma_mul3_bounds(a: int, b: int, c: int, i: int, j: int, k: int)
    requires
        0 <= i < a,
        0 <= j < b,
        0 <= k < c,
    ensures
        (i * b + j) * c + k < a * b * c,
        0 <= (i * b + j) * c + k,
        0 <= i * b <= i * b + j,
        i * b + j < a * b,
        i * b + j <= (i * b + j) * c,
        (i * b + j) * c < a * b * c,
{
    assert(i * b + j <= (i * b + j) * c) by (nonlinear_arith)
        requires 0 <= i, 0 <= j, 0 <= b, c >= 1;
    assert(0 <= i * b) by (nonlinear_arith) requires 0 <= i, 0 <= b;
    assert(i * b + j < a * b) by (nonlinear_arith) requires 0 <= i < a, 0 <= j < b;
    assert((i * b + j) * c + k < (i * b + j + 1) * c) by (nonlinear_arith)
        requires 0 <= k < c, 0 <= i * b + j;
    assert((i * b + j + 1) * c <= a * b * c) by (nonlinear_arith)
        requires i * b + j + 1 <= a * b, 0 <= c;
    assert(0 <= (i * b + j) * c) by (nonlinear_arith) requires 0 <= i * b + j, 0 <= c;
}

} // verus!

/// Validates a blocks/scales shape pair and returns its extents.
///
/// Blocks must be `[E, R, G, 16]` and scales `[E, R, G]` with nonzero axes.
/// The decoded tensor holds `E · R · 32 G` values; the function rejects any
/// pair whose value count (and therefore its byte counts, at most eight bytes
/// per value) does not fit both `usize` and `u64`.
///
/// Verified: the result is `Ok` exactly for accepted pairs, and each error is
/// exactly the first violated rule in the order blocks, scales, empty axis,
/// overflow, so a diagnostic never misnames the defect.
///
/// # Errors
///
/// Returns the first [`Mxfp4ShapeError`] the pair violates.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(geometry =>
    ensures
        geometry matches Ok(_) <==> accepts(blocks@, scales@),
        geometry matches Err(Mxfp4ShapeError::Blocks) <==> !blocks_shaped(blocks@),
        geometry matches Err(Mxfp4ShapeError::Scales) <==>
            blocks_shaped(blocks@) && !scales_match(blocks@, scales@),
        geometry matches Err(Mxfp4ShapeError::Empty) <==>
            blocks_shaped(blocks@) && scales_match(blocks@, scales@) && !axes_nonzero(blocks@),
        geometry matches Err(Mxfp4ShapeError::Overflow) <==>
            blocks_shaped(blocks@) && scales_match(blocks@, scales@) && axes_nonzero(blocks@)
                && !count_fits(blocks@),
        geometry matches Ok(g) ==> {
            &&& describes(g)
            &&& blocks@.len() == 4 && scales@.len() == 3
            &&& g.experts == blocks@[0] && g.rows == blocks@[1] && g.groups == blocks@[2]
            &&& blocks@[3] == 16
            &&& scales@[0] == blocks@[0] && scales@[1] == blocks@[1] && scales@[2] == blocks@[2]
        },
))]
pub(crate) fn mxfp4_geometry(
    blocks: &[u64],
    scales: &[u64],
) -> Result<Mxfp4Geometry, Mxfp4ShapeError> {
    if blocks.len() != 4 || blocks[3] != 16 {
        return Err(Mxfp4ShapeError::Blocks);
    }
    if scales.len() != 3
        || scales[0] != blocks[0]
        || scales[1] != blocks[1]
        || scales[2] != blocks[2]
    {
        return Err(Mxfp4ShapeError::Scales);
    }
    if blocks[0] == 0 || blocks[1] == 0 || blocks[2] == 0 {
        return Err(Mxfp4ShapeError::Empty);
    }
    // Completeness: with nonzero axes every partial product divides the value
    // count, so a count within range overflows none of the checks below.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(32 * blocks@[2] <= blocks@[0] * blocks@[1] * (32 * blocks@[2])) by (nonlinear_arith)
            requires blocks@[0] >= 1, blocks@[1] >= 1, blocks@[2] >= 1;
        assert(blocks@[0] * blocks@[1] <= blocks@[0] * blocks@[1] * (32 * blocks@[2])) by (nonlinear_arith)
            requires blocks@[0] >= 1, blocks@[1] >= 1, blocks@[2] >= 1;
    }
    let limit = if (usize::MAX as u64) < u64::MAX {
        usize::MAX as u64
    } else {
        u64::MAX
    };
    let Some(columns) = blocks[2].checked_mul(32) else {
        return Err(Mxfp4ShapeError::Overflow);
    };
    let Some(matrix) = blocks[0].checked_mul(blocks[1]) else {
        return Err(Mxfp4ShapeError::Overflow);
    };
    let Some(values) = matrix.checked_mul(columns) else {
        return Err(Mxfp4ShapeError::Overflow);
    };
    if values > limit {
        return Err(Mxfp4ShapeError::Overflow);
    }
    // Every factor divides `values <= limit`, so each fits `usize`.
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(blocks@[0] * blocks@[1] * columns == values);
        assert(blocks@[0] <= values) by (nonlinear_arith)
            requires blocks@[1] >= 1, columns >= 1, values == blocks@[0] * blocks@[1] * columns, blocks@[0] >= 0;
        assert(blocks@[1] <= values) by (nonlinear_arith)
            requires blocks@[0] >= 1, columns >= 1, values == blocks@[0] * blocks@[1] * columns, blocks@[1] >= 0;
        assert(columns <= values) by (nonlinear_arith)
            requires blocks@[0] >= 1, blocks@[1] >= 1, values == blocks@[0] * blocks@[1] * columns, columns >= 0;
    }
    Ok(Mxfp4Geometry {
        experts: blocks[0] as usize,
        rows: blocks[1] as usize,
        groups: blocks[2] as usize,
        columns: columns as usize,
    })
}

impl Mxfp4Geometry {
    /// Returns the decoded value count `E · R · 32 G`.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(values =>
        requires
            describes(self),
        ensures
            values == self.experts * self.rows * self.columns,
    ))]
    #[must_use]
    pub(crate) fn values(self) -> usize {
        values_of(self)
    }

    /// Returns the packed block bytes `E · R · G · 16` (half the value count).
    ///
    /// Verified: exactly sixteen bytes for each of the `E · R · G` blocks.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(bytes =>
        requires
            describes(self),
        ensures
            bytes == self.experts * self.rows * self.groups * 16,
    ))]
    #[must_use]
    pub(crate) fn block_bytes(self) -> usize {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_factors(self);
        }
        values_of(self) / 2
    }

    /// Returns the scale count `E · R · G` (a 32nd of the value count).
    ///
    /// Verified: exactly one scale for each of the `E · R · G` blocks.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(count =>
        requires
            describes(self),
        ensures
            count == self.experts * self.rows * self.groups,
    ))]
    #[must_use]
    pub(crate) fn scale_count(self) -> usize {
        #[cfg(verus_keep_ghost)]
        proof! {
            lemma_count_factors(self);
        }
        values_of(self) / BLOCK_VALUES
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// The value count is 32 values, or 16 block bytes, per scale.
proof fn lemma_count_factors(g: Mxfp4Geometry)
    requires
        describes(g),
    ensures
        g.experts * g.rows * g.columns == 32 * (g.experts * g.rows * g.groups),
        (g.experts * g.rows * g.columns) / 2 == g.experts * g.rows * g.groups * 16,
        (g.experts * g.rows * g.columns) / 32 == g.experts * g.rows * g.groups,
{
    let blocks = g.experts * g.rows * g.groups;
    assert(g.experts * g.rows * g.columns == 32 * blocks) by (nonlinear_arith)
        requires g.columns == 32 * g.groups, blocks == g.experts * g.rows * g.groups;
    assert(blocks * 16 == g.experts * g.rows * g.groups * 16);
}

} // verus!

/// Returns the decoded value count `E · R · 32 G` of a validated geometry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(values =>
    requires
        describes(geometry),
    ensures
        values == geometry.experts * geometry.rows * geometry.columns,
))]
pub(crate) fn values_of(geometry: Mxfp4Geometry) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(geometry.experts * geometry.rows <= geometry.experts * geometry.rows * geometry.columns)
            by (nonlinear_arith)
            requires geometry.columns >= 1, geometry.experts * geometry.rows >= 0;
    }
    geometry.experts * geometry.rows * geometry.columns
}

/// Splits one block byte into its two FP4 codes `(low, high)`.
///
/// The low nibble is the even decoded column `32 g + 2 b`, the high nibble the
/// odd column `32 g + 2 b + 1`, as Transformers reads `blk & 0x0F` and
/// `blk >> 4`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(codes =>
    ensures
        codes.0 == byte % 16,
        codes.1 == byte / 16,
        codes.0 < 16 && codes.1 < 16,
        codes.0 + 16 * codes.1 == byte,
))]
pub(crate) const fn fp4_codes(byte: u8) -> (u8, u8) {
    (byte % 16, byte / 16)
}

/// Returns the binary64 encoding of the decoded value of `code` under `scale`.
///
/// With magnitude code `m = code mod 8` and exponent `e = scale - 127`:
///
/// * `m = 0`: a zero with the code's sign (`+0` or `-0`);
/// * `m = 1`: `±2^(e - 1)`, biased binary64 exponent `scale + 895`, no fraction;
/// * `m >= 2`: `±(1 + (m mod 2)/2) · 2^(⌊m/2⌋ - 1 + e)`, biased exponent
///   `⌊m/2⌋ + scale + 895` and fraction `2^51` exactly when `m` is odd.
///
/// Both exponents lie in `895..=1152`, inside binary64's normal range, so the
/// encoding is exact. Scale code 255 yields the canonical positive quiet NaN
/// for `m = 0` and `±inf` otherwise (see the module documentation).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(bits =>
    requires
        code < 16,
    ensures
        scale == 255 && code % 8 == 0 ==> bits == CANONICAL_NAN_BITS,
        scale == 255 && code % 8 != 0 ==> bits == (code / 8) as u64 * SIGN_BIT + INFINITY_BITS,
        scale < 255 && code % 8 == 0 ==> bits == (code / 8) as u64 * SIGN_BIT,
        scale < 255 && code % 8 == 1 ==>
            bits == (code / 8) as u64 * SIGN_BIT + (scale as u64 + EXPONENT_OFFSET) * EXPONENT_UNIT,
        scale < 255 && code % 8 >= 2 ==> bits == (code / 8) as u64 * SIGN_BIT
            + ((code % 8) / 2 + scale as u64 + EXPONENT_OFFSET) * EXPONENT_UNIT
            + (code % 2) as u64 * HALF_FRACTION,
))]
pub(crate) fn exact_value_bits(scale: u8, code: u8) -> u64 {
    let sign = (code / 8) as u64 * SIGN_BIT;
    let magnitude = code % 8;
    if scale == 255 {
        if magnitude == 0 {
            return CANONICAL_NAN_BITS;
        }
        return sign + INFINITY_BITS;
    }
    if magnitude == 0 {
        return sign;
    }
    if magnitude == 1 {
        return sign + (scale as u64 + EXPONENT_OFFSET) * EXPONENT_UNIT;
    }
    let exponent = (magnitude / 2) as u64 + scale as u64 + EXPONENT_OFFSET;
    let fraction = (code % 2) as u64 * HALF_FRACTION;
    sign + exponent * EXPONENT_UNIT + fraction
}

/// Returns the byte offset of block `group` of `row` of `expert` in the blocks tensor.
///
/// The block occupies bytes `[offset, offset + 16)`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(offset =>
    requires
        describes(geometry),
        expert < geometry.experts,
        row < geometry.rows,
        group < geometry.groups,
    ensures
        offset == ((expert * geometry.rows + row) * geometry.groups + group) * 16,
        offset + 16 <= geometry.experts * geometry.rows * geometry.columns / 2,
))]
pub(crate) fn block_offset(
    geometry: Mxfp4Geometry,
    expert: usize,
    row: usize,
    group: usize,
) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_mul3_bounds(
            geometry.experts as int, geometry.rows as int, geometry.groups as int,
            expert as int, row as int, group as int,
        );
        let index = (expert * geometry.rows + row) * geometry.groups + group;
        assert(geometry.experts * geometry.rows * geometry.columns
            == 32 * (geometry.experts * geometry.rows * geometry.groups)) by (nonlinear_arith)
            requires geometry.columns == 32 * geometry.groups;
    }
    scale_offset(geometry, expert, row, group) * BLOCK_BYTES
}

/// Returns the offset of the scale shared by block `group` of `row` of `expert`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(offset =>
    requires
        describes(geometry),
        expert < geometry.experts,
        row < geometry.rows,
        group < geometry.groups,
    ensures
        offset == (expert * geometry.rows + row) * geometry.groups + group,
        offset < geometry.experts * geometry.rows * geometry.groups,
        (offset + 1) * 32 <= geometry.experts * geometry.rows * geometry.columns,
))]
pub(crate) fn scale_offset(
    geometry: Mxfp4Geometry,
    expert: usize,
    row: usize,
    group: usize,
) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_mul3_bounds(
            geometry.experts as int, geometry.rows as int, geometry.groups as int,
            expert as int, row as int, group as int,
        );
        assert(geometry.experts * geometry.rows * geometry.columns
            == 32 * (geometry.experts * geometry.rows * geometry.groups)) by (nonlinear_arith)
            requires geometry.columns == 32 * geometry.groups;
        assert(geometry.experts * geometry.rows * geometry.groups <= usize::MAX);
    }
    (expert * geometry.rows + row) * geometry.groups + group
}

/// Returns the element offset of decoded `(expert, column, row)` in the transposed output.
///
/// The output is `[E, 32 G, R]` row-major, so the value of row `row` at
/// decoded column `column` lands at `(expert · 32 G + column) · R + row`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(offset =>
    requires
        describes(geometry),
        expert < geometry.experts,
        column < geometry.columns,
        row < geometry.rows,
    ensures
        offset == (expert * geometry.columns + column) * geometry.rows + row,
        offset < geometry.experts * geometry.rows * geometry.columns,
))]
pub(crate) fn transposed_offset(
    geometry: Mxfp4Geometry,
    expert: usize,
    column: usize,
    row: usize,
) -> usize {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_mul3_bounds(
            geometry.experts as int, geometry.columns as int, geometry.rows as int,
            expert as int, column as int, row as int,
        );
        assert(geometry.experts * geometry.columns * geometry.rows
            == geometry.experts * geometry.rows * geometry.columns) by (nonlinear_arith);
    }
    (expert * geometry.columns + column) * geometry.rows + row
}

/// Returns the decode-table entry of `code` under `scale`: `16 · scale + (code mod 16)`.
///
/// Verified: the index is in bounds of the [`TABLE_ENTRIES`]-entry table and is
/// a bijection with [`table_coordinates`], so every `(scale, code)` pair owns
/// exactly one entry.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(index =>
    ensures
        index == 16 * scale + code % 16,
        index < TABLE_ENTRIES,
))]
const fn table_index(scale: u8, code: u8) -> usize {
    scale as usize * FP4_CODES + (code % 16) as usize
}

/// Returns the `(scale, code)` pair stored at decode-table entry `index`.
///
/// Verified: the inverse of [`table_index`], so filling entries in index order
/// visits every pair once, with scales outer and codes inner.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(pair =>
    requires
        index < TABLE_ENTRIES,
    ensures
        pair.1 < 16,
        16 * pair.0 + pair.1 == index,
))]
const fn table_coordinates(index: usize) -> (u8, u8) {
    ((index / FP4_CODES) as u8, (index % FP4_CODES) as u8)
}

/// A decode table: the target encoding of every `(scale, code)` pair.
///
/// Entry `16 · scale + code` holds the native-endian bytes of the decoded
/// value of `code` under `scale` in a `W`-byte target dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mxfp4Table<const W: usize> {
    /// [`TABLE_ENTRIES`] encodings indexed `16 · scale + code`.
    entries: Vec<[u8; W]>,
}

impl<const W: usize> Mxfp4Table<W> {
    /// Builds the table by encoding every exact binary64 value with `encode`.
    ///
    /// `encode` receives [`exact_value_bits`] and returns the target dtype's
    /// native-endian bytes: the identity for F64, a cast for F32 (exact up to
    /// overflow), and one round-to-nearest-even step for BF16 and F16.
    pub(crate) fn new(mut encode: impl FnMut(u64) -> [u8; W]) -> Self {
        let mut entries = Vec::with_capacity(TABLE_ENTRIES);
        // Entry `i` holds pair `table_coordinates(i)`: scales outer, codes inner.
        for index in 0..TABLE_ENTRIES {
            let (scale, code) = table_coordinates(index);
            entries.push(encode(exact_value_bits(scale, code)));
        }
        Self { entries }
    }

    /// Returns the encoding of `code` under `scale`.
    #[must_use]
    pub(crate) fn get(&self, scale: u8, code: u8) -> [u8; W] {
        self.entries[table_index(scale, code)]
    }
}

/// How [`decode_transposed`] splits its `E · G` independent units over threads.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodePartition {
    /// Output bytes of one unit: its `32 · R` values of `W` bytes each.
    pub(crate) unit_bytes: usize,
    /// Units per thread chunk, `⌈(E · G) / threads⌉`.
    pub(crate) units_per_thread: usize,
    /// Output bytes of one full thread chunk, `units_per_thread · unit_bytes`.
    pub(crate) chunk_bytes: usize,
    /// Threads to run the chunks on; one means the caller's thread alone.
    pub(crate) threads: usize,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the exact rounded-up quotient `⌈dividend / divisor⌉` for `divisor > 0`.
pub open spec fn ceiling(dividend: int, divisor: int) -> int {
    if dividend % divisor == 0 {
        dividend / divisor
    } else {
        dividend / divisor + 1
    }
}

} // verus!

/// Splits a decode of `geometry` into per-thread chunks of whole units.
///
/// `available` is the parallelism the caller may use and `scalar_bytes` the
/// target dtype width `W`. Unit `u = e · G + g` owns the `32 · R` output values
/// of expert `e`, column group `g` (see [`unit_coordinates`]); the output is
/// cut into chunks of `units_per_thread` whole units.
///
/// Verified: the thread count is one below `PARALLEL_MIN_VALUES` values and
/// `min(available, E · G)` otherwise; `units_per_thread` is the exact ceiling
/// of `E · G` over the threads, so the chunks number at most `threads` and
/// cover every unit exactly once (`(n - 1) · k < E · G <= n · k` for `n`
/// chunks of `k` units); the `E · G` units of `unit_bytes` partition the
/// `values · W` output bytes exactly; and every product is overflow-free.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(partition =>
    requires
        describes(geometry),
        available >= 1,
        scalar_bytes >= 1,
        geometry.experts * geometry.rows * geometry.columns * scalar_bytes <= usize::MAX,
    ensures
        partition.threads == (if geometry.experts * geometry.rows * geometry.columns
            < PARALLEL_MIN_VALUES {
            1
        } else if available < geometry.experts * geometry.groups {
            available as int
        } else {
            geometry.experts * geometry.groups
        }),
        1 <= partition.threads <= geometry.experts * geometry.groups,
        partition.threads <= available,
        partition.unit_bytes == 32 * geometry.rows * scalar_bytes,
        partition.unit_bytes >= 1,
        geometry.experts * geometry.groups * partition.unit_bytes
            == geometry.experts * geometry.rows * geometry.columns * scalar_bytes,
        partition.units_per_thread
            == ceiling(geometry.experts * geometry.groups, partition.threads as int),
        1 <= partition.units_per_thread <= geometry.experts * geometry.groups,
        partition.chunk_bytes == partition.units_per_thread * partition.unit_bytes,
        partition.chunk_bytes <= geometry.experts * geometry.rows * geometry.columns
            * scalar_bytes,
        ceiling(geometry.experts * geometry.groups, partition.units_per_thread as int)
            <= partition.threads,
        (ceiling(geometry.experts * geometry.groups, partition.units_per_thread as int) - 1)
            * partition.units_per_thread < geometry.experts * geometry.groups,
        geometry.experts * geometry.groups <= ceiling(
            geometry.experts * geometry.groups,
            partition.units_per_thread as int,
        ) * partition.units_per_thread,
))]
pub(crate) fn decode_partition(
    geometry: Mxfp4Geometry,
    available: usize,
    scalar_bytes: usize,
) -> DecodePartition {
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_unit_sizes(geometry, scalar_bytes);
    }
    let values = values_of(geometry);
    let units = geometry.experts * geometry.groups;
    let unit_bytes = BLOCK_VALUES * geometry.rows * scalar_bytes;
    let threads = if values < PARALLEL_MIN_VALUES {
        1
    } else if available < units {
        available
    } else {
        units
    };
    let quotient = units / threads;
    #[cfg(verus_keep_ghost)]
    proof! {
        // A nonzero remainder needs at least two threads, so `quotient < units`.
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(units as int, threads as int);
        assert(quotient * threads <= units) by (nonlinear_arith)
            requires
                units == threads * quotient + units % threads,
                units % threads >= 0,
        ;
        if units % threads != 0 {
            assert(quotient < units) by (nonlinear_arith)
                requires quotient * threads <= units, threads >= 2, units >= 1;
        }
    }
    // A named remainder keeps this Verus-checkable (`div_ceil` and
    // `is_multiple_of` have no specification).
    let remainder = units % threads;
    let units_per_thread = if remainder == 0 {
        quotient
    } else {
        quotient + 1
    };
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_chunk_cover(units as int, threads as int, units_per_thread as int);
        assert(units_per_thread * unit_bytes <= units * unit_bytes) by (nonlinear_arith)
            requires units_per_thread <= units, unit_bytes >= 0;
    }
    DecodePartition {
        unit_bytes,
        units_per_thread,
        chunk_bytes: units_per_thread * unit_bytes,
        threads,
    }
}

/// Returns the expert and column group of decode unit `unit = e · G + g`.
///
/// Verified: `(e, g)` is in range and recovers `unit`, and unit `u`'s values
/// are exactly the transposed-output elements `[u · 32 R, (u + 1) · 32 R)`:
/// decoded column `32 g + j` of row `r` lies at element `u · 32 R + j · R + r`
/// of the `[E, 32 G, R]` output, so consecutive `unit_bytes` output chunks
/// belong to consecutive units.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(coordinates =>
    requires
        describes(geometry),
        unit < geometry.experts * geometry.groups,
    ensures
        coordinates.0 < geometry.experts,
        coordinates.1 < geometry.groups,
        unit == coordinates.0 * geometry.groups + coordinates.1,
        forall|j: int, r: int| 0 <= j < 32 && 0 <= r < geometry.rows ==>
            (coordinates.0 * geometry.columns + 32 * coordinates.1 + j) * geometry.rows + r
                == #[trigger] unit_element(geometry.rows as int, unit as int, j, r),
))]
pub(crate) fn unit_coordinates(geometry: Mxfp4Geometry, unit: usize) -> (usize, usize) {
    let expert = unit / geometry.groups;
    let group = unit % geometry.groups;
    #[cfg(verus_keep_ghost)]
    proof! {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(unit as int, geometry.groups as int);
        assert(unit == expert * geometry.groups + group) by (nonlinear_arith)
            requires unit == geometry.groups * expert + group;
        assert(expert < geometry.experts) by (nonlinear_arith)
            requires
                unit == expert * geometry.groups + group,
                unit < geometry.experts * geometry.groups,
                0 <= group,
                geometry.groups > 0,
        ;
        assert forall|j: int, r: int| 0 <= j < 32 && 0 <= r < geometry.rows implies
            (expert * geometry.columns + 32 * group + j) * geometry.rows + r
                == #[trigger] unit_element(geometry.rows as int, unit as int, j, r)
        by {
            assert((expert * geometry.columns + 32 * group + j) * geometry.rows + r
                == unit * (32 * geometry.rows) + j * geometry.rows + r) by (nonlinear_arith)
                requires
                    geometry.columns == 32 * geometry.groups,
                    unit == expert * geometry.groups + group,
            ;
        }
    }
    (expert, group)
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: element `u · 32 R + j · R + r` of the transposed output.
pub open spec fn unit_element(rows: int, unit: int, j: int, r: int) -> int {
    unit * (32 * rows) + j * rows + r
}

/// The per-unit byte count is positive and the units tile the output bytes.
proof fn lemma_unit_sizes(g: Mxfp4Geometry, scalar_bytes: usize)
    requires
        describes(g),
        scalar_bytes >= 1,
        g.experts * g.rows * g.columns * scalar_bytes <= usize::MAX,
    ensures
        1 <= g.experts * g.groups <= g.experts * g.rows * g.columns,
        32 * g.rows * scalar_bytes <= g.experts * g.rows * g.columns * scalar_bytes,
        32 * g.rows * scalar_bytes >= 1,
        32 * g.rows <= usize::MAX,
        g.experts * g.groups * (32 * g.rows * scalar_bytes)
            == g.experts * g.rows * g.columns * scalar_bytes,
{
    assert(g.experts * g.rows * g.columns == (g.experts * g.groups) * (32 * g.rows))
        by (nonlinear_arith)
        requires g.columns == 32 * g.groups;
    assert(g.experts * g.groups >= 1) by (nonlinear_arith)
        requires g.experts >= 1, g.groups >= 1;
    assert(g.experts * g.groups <= (g.experts * g.groups) * (32 * g.rows)) by (nonlinear_arith)
        requires g.rows >= 1, g.experts * g.groups >= 0;
    assert(32 * g.rows <= (g.experts * g.groups) * (32 * g.rows)) by (nonlinear_arith)
        requires g.experts >= 1, g.groups >= 1, g.rows >= 0;
    assert(32 * g.rows * scalar_bytes <= (g.experts * g.groups) * (32 * g.rows) * scalar_bytes)
        by (nonlinear_arith)
        requires g.experts >= 1, g.groups >= 1, g.rows >= 0, scalar_bytes >= 1;
    assert(32 * g.rows <= 32 * g.rows * scalar_bytes) by (nonlinear_arith)
        requires scalar_bytes >= 1, g.rows >= 0;
    assert(32 * g.rows * scalar_bytes >= 1) by (nonlinear_arith)
        requires scalar_bytes >= 1, g.rows >= 1;
    assert(g.experts * g.groups * (32 * g.rows * scalar_bytes)
        == (g.experts * g.groups) * (32 * g.rows) * scalar_bytes) by (nonlinear_arith);
}

/// Chunks of `k = ⌈units / threads⌉` units number at most `threads` and cover every unit once.
proof fn lemma_chunk_cover(units: int, threads: int, k: int)
    requires
        units >= 1,
        1 <= threads <= units,
        k == ceiling(units, threads),
    ensures
        1 <= k <= units,
        ceiling(units, k) <= threads,
        (ceiling(units, k) - 1) * k < units,
        units <= ceiling(units, k) * k,
{
    let q = units / threads;
    let r = units % threads;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(units, threads);
    assert(units == q * threads + r);
    assert(0 <= r < threads);
    // k * threads >= units, so ceil(units / k) <= threads.
    assert(k * threads >= units) by (nonlinear_arith)
        requires units == q * threads + r, 0 <= r < threads, k == (if r == 0 { q } else { q + 1 });
    assert(q >= 1) by (nonlinear_arith)
        requires units == q * threads + r, 0 <= r < threads, threads <= units, threads >= 1;
    assert(k <= units) by (nonlinear_arith)
        requires
            units == q * threads + r,
            0 <= r < threads,
            k == (if r == 0 { q } else { q + 1 }),
            threads >= 1,
            q >= 1,
    ;
    let n = ceiling(units, k);
    let qq = units / k;
    let rr = units % k;
    vstd::arithmetic::div_mod::lemma_fundamental_div_mod(units, k);
    assert(units == qq * k + rr);
    assert(n * k >= units) by (nonlinear_arith)
        requires units == qq * k + rr, 0 <= rr < k, n == (if rr == 0 { qq } else { qq + 1 });
    assert((n - 1) * k < units) by (nonlinear_arith)
        requires
            units == qq * k + rr,
            0 <= rr < k,
            n == (if rr == 0 { qq } else { qq + 1 }),
            units >= 1,
    ;
    assert(n <= threads) by (nonlinear_arith)
        requires (n - 1) * k < units, units <= k * threads, k >= 1;
}

} // verus!

/// Decodes a validated blocks/scales pair into Transformers' `[E, 32 G, R]` layout.
///
/// `blocks` holds `geometry.block_bytes()` bytes, `scales` `geometry.scale_count()`
/// codes, and `output` exactly `geometry.values() · W` bytes; every value is
/// written. The work splits into `E · G` independent units, one per expert and
/// column group: unit `(e, g)` owns the contiguous output rows
/// `[e · 32 G + 32 g, e · 32 G + 32 g + 32)` and reads block `g` of every row of
/// expert `e`. Units are distributed over up to
/// [`std::thread::available_parallelism`] scoped threads, so large tensors
/// decode in parallel without synchronization (each thread owns its outputs).
///
/// # Errors
///
/// Returns [`Mxfp4ShapeError::Overflow`] when a slice length disagrees with
/// the geometry, before writing anything.
pub(crate) fn decode_transposed<const W: usize>(
    blocks: &[u8],
    scales: &[u8],
    geometry: Mxfp4Geometry,
    table: &Mxfp4Table<W>,
    output: &mut [u8],
) -> Result<(), Mxfp4ShapeError> {
    let values = geometry.values();
    if blocks.len() != geometry.block_bytes()
        || scales.len() != geometry.scale_count()
        || values.checked_mul(W) != Some(output.len())
    {
        return Err(Mxfp4ShapeError::Overflow);
    }
    // The partition stays on one thread below the threshold, so skip the system query there.
    let available = if values < PARALLEL_MIN_VALUES {
        1
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    };
    let partition = decode_partition(geometry, available, W);
    let unit_bytes = partition.unit_bytes;
    let units_per_thread = partition.units_per_thread;
    std::thread::scope(|scope| {
        for (chunk, slice) in output.chunks_mut(partition.chunk_bytes).enumerate() {
            let first_unit = chunk * units_per_thread;
            let mut work = move || {
                for (offset, unit) in slice.chunks_exact_mut(unit_bytes).enumerate() {
                    decode_unit(blocks, scales, geometry, table, first_unit + offset, unit);
                }
            };
            if partition.threads == 1 {
                work();
            } else {
                scope.spawn(work);
            }
        }
    });
    Ok(())
}

/// Decodes unit `unit = e · G + g`: block `g` of every row of expert `e`.
///
/// `output` is the unit's `[32, R]` slice of the transposed tensor: decoded
/// column `32 g + j` of row `r` lands at element `j · R + r`, i.e. at
/// [`transposed_offset`] minus the unit's first element. [`unit_coordinates`]
/// proves that first element is `u · 32 R`, the start of the unit's output
/// chunk, and that `base` below is exactly `r`.
fn decode_unit<const W: usize>(
    blocks: &[u8],
    scales: &[u8],
    geometry: Mxfp4Geometry,
    table: &Mxfp4Table<W>,
    unit: usize,
    output: &mut [u8],
) {
    let (expert, group) = unit_coordinates(geometry, unit);
    let rows = geometry.rows;
    let first_column = BLOCK_VALUES * group;
    let unit_start = transposed_offset(geometry, expert, first_column, 0);
    for row in 0..rows {
        let scale = scales[scale_offset(geometry, expert, row, group)];
        let start = block_offset(geometry, expert, row, group);
        // Element of column `32 g` of this row, relative to the unit.
        let base = transposed_offset(geometry, expert, first_column, row) - unit_start;
        for (byte_index, &byte) in blocks[start..start + BLOCK_BYTES].iter().enumerate() {
            let (low, high) = fp4_codes(byte);
            let even = (base + 2 * byte_index * rows) * W;
            let odd = even + rows * W;
            output[even..even + W].copy_from_slice(&table.get(scale, low));
            output[odd..odd + W].copy_from_slice(&table.get(scale, high));
        }
    }
}

#[cfg(test)]
#[path = "mxfp4_tests.rs"]
mod tests;
