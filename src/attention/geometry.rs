//! Pure shape validation and causal routing of one attention call, checked by Verus.
//!
//! # Operand shapes
//!
//! One grouped call takes queries `[B, H, Q, d]`, native keys `[B, G, K, d]`,
//! values `[B, G, K, d_v]`, an optional exclusion mask `[B, H, Q, K]`, an
//! optional additive bias `[1 or B, H, Q, K]`, and optional sink logits `[H]`,
//! with `G` dividing `H`. [`operand_geometry`] accepts exactly these shapes
//! and reports the first failed check; [`projection_fault`] does the same for
//! a projected layer whose head counts and widths are configured.
//!
//! # Causal alignment
//!
//! Causality is bottom-right aligned: query row `i` of `Q` sits at key
//! position `i + K - Q` and excludes keys `j > i + K - Q`. Some fused kernels
//! align top-left (`j > i`). The two agree on every link exactly when
//! `Q = K`, and a single query row (`Q = 1`) sees every key under bottom-right
//! alignment, so its causal mask excludes nothing. [`causal_mask_needed`] and
//! [`native_route`] rely on exactly these facts, which Verus proves.
//!
//! # Verified properties
//!
//! Every function here is checked through `verus/attention.rs`: validators
//! accept exactly the documented shapes with each fault reported in order,
//! a skipped causal mask excludes no link, and the backend's own causal flag
//! is used only where its alignment equals bottom-right alignment.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Bottom-right causal alignment excludes key `key` for query row `row` of `Q` rows over `K` keys.
pub open spec fn bottom_right_excludes(row: int, key: int, queries: int, keys: int) -> bool {
    key > row + keys - queries
}

/// Top-left causal alignment (as some fused kernels use) excludes key `key` for row `row`.
pub open spec fn top_left_excludes(row: int, key: int) -> bool {
    key > row
}

/// Two rank-4 shapes are equal.
pub open spec fn same4(actual: [usize; 4], expected: [usize; 4]) -> bool {
    &&& actual[0] == expected[0]
    &&& actual[1] == expected[1]
    &&& actual[2] == expected[2]
    &&& actual[3] == expected[3]
}

/// An additive bias `[1 or B, H, Q, K]` broadcasts over the score plane `[B, H, Q, K]`.
pub open spec fn bias_fits(bias: [usize; 4], score: [usize; 4]) -> bool {
    &&& (bias[0] == 1 || bias[0] == score[0])
    &&& bias[1] == score[1]
    &&& bias[2] == score[2]
    &&& bias[3] == score[3]
}

} // verus!

/// Whether an optional rank-4 extent is present and differs from `expected`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(differs =>
    ensures
        differs == (actual is Some && !same4(actual->Some_0, expected)),
))]
fn mismatched(actual: Option<[usize; 4]>, expected: [usize; 4]) -> bool {
    match actual {
        Some(actual) => {
            actual[0] != expected[0]
                || actual[1] != expected[1]
                || actual[2] != expected[2]
                || actual[3] != expected[3]
        }
        None => false,
    }
}

/// Whether an optional additive bias is present and does not broadcast as `[1 or B, H, Q, K]`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(misfit =>
    ensures
        misfit == (bias is Some && !bias_fits(bias->Some_0, score)),
))]
fn misfit_bias(bias: Option<[usize; 4]>, score: [usize; 4]) -> bool {
    match bias {
        Some(bias) => {
            (bias[0] != 1 && bias[0] != score[0])
                || bias[1] != score[1]
                || bias[2] != score[2]
                || bias[3] != score[3]
        }
        None => false,
    }
}

/// Validated extents of one grouped attention call.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadGeometry {
    /// Batch rows.
    pub(crate) batch: usize,
    /// Query heads.
    pub(crate) query_heads: usize,
    /// Native key/value heads; divides `query_heads`.
    pub(crate) key_value_heads: usize,
    /// Query tokens.
    pub(crate) query_sequence: usize,
    /// Key/value tokens.
    pub(crate) key_sequence: usize,
    /// Query/key channels per head.
    pub(crate) head_dim: usize,
}

impl HeadGeometry {
    /// Query heads that share one key/value head.
    ///
    /// Verified exact for every geometry [`operand_geometry`] returns (it
    /// guarantees non-empty, grouped heads): `groups * key_value_heads ==
    /// query_heads` with at least one query head per group.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(groups =>
        requires
            self.key_value_heads >= 1,
            self.query_heads % self.key_value_heads == 0,
        ensures
            groups == self.query_heads / self.key_value_heads,
            groups * self.key_value_heads == self.query_heads,
            self.query_heads >= 1 ==> groups >= 1,
    ))]
    pub(crate) const fn groups(self) -> usize {
        #[cfg(verus_keep_ghost)]
        proof! {
            let heads = self.query_heads as int;
            let shared = self.key_value_heads as int;
            let groups = heads / shared;
            vstd::arithmetic::div_mod::lemma_fundamental_div_mod(heads, shared);
            assert(groups * shared == shared * groups) by (nonlinear_arith);
            if heads >= 1 {
                assert(groups >= 1) by (nonlinear_arith)
                    requires
                        heads == shared * groups,
                        heads >= 1,
                        shared >= 1;
            }
        }
        self.query_heads / self.key_value_heads
    }
}

/// Why the operands of one grouped attention call disagree.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandFault {
    /// A query, key, or value axis is empty.
    EmptyAxis,
    /// Key/value heads do not divide query heads.
    UngroupedHeads,
    /// Keys or values disagree with the queries' batch and width or with each other.
    KeyValueShape,
    /// The mask is not `[B, H, Q, K]`.
    MaskShape,
    /// The bias is not `[1 or B, H, Q, K]`.
    BiasShape,
    /// The sink logits are not `[H]`.
    SinkShape,
}

/// Validates grouped attention operand shapes, returning their geometry or the first fault.
///
/// `query` is `[B, H, Q, d]`, `key` `[B, G, K, d]`, `value` `[B, G, K, d_v]`;
/// optional `mask`, `bias`, and `sinks` are the extents of those operands.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    ensures
        ({
            let nonempty = query[0] >= 1 && query[1] >= 1 && query[2] >= 1 && query[3] >= 1
                && key[1] >= 1 && key[2] >= 1 && value[3] >= 1;
            let grouped = nonempty && query[1] % key[1] == 0;
            let paired = grouped && key[0] == query[0] && key[3] == query[3]
                && value[0] == query[0] && value[1] == key[1] && value[2] == key[2];
            let score = [query[0], query[1], query[2], key[2]];
            let masked = paired && (mask is None || same4(mask->Some_0, score));
            let biased = masked && (bias is None || bias_fits(bias->Some_0, score));
            match result {
                Ok(geometry) => {
                    &&& biased
                    &&& (sinks is None || sinks->Some_0 == query[1])
                    &&& geometry == HeadGeometry {
                        batch: query[0],
                        query_heads: query[1],
                        key_value_heads: key[1],
                        query_sequence: query[2],
                        key_sequence: key[2],
                        head_dim: query[3],
                    }
                },
                Err(OperandFault::EmptyAxis) => !nonempty,
                Err(OperandFault::UngroupedHeads) => nonempty && !grouped,
                Err(OperandFault::KeyValueShape) => grouped && !paired,
                Err(OperandFault::MaskShape) => paired && !masked,
                Err(OperandFault::BiasShape) => masked && !biased,
                Err(OperandFault::SinkShape) => biased && sinks is Some && sinks->Some_0
                    != query[1],
            }
        }),
))]
pub(crate) fn operand_geometry(
    query: [usize; 4],
    key: [usize; 4],
    value: [usize; 4],
    mask: Option<[usize; 4]>,
    bias: Option<[usize; 4]>,
    sinks: Option<usize>,
) -> Result<HeadGeometry, OperandFault> {
    let batch = query[0];
    let query_heads = query[1];
    let query_sequence = query[2];
    let head_dim = query[3];
    let key_value_heads = key[1];
    let key_sequence = key[2];
    if batch == 0
        || query_heads == 0
        || query_sequence == 0
        || head_dim == 0
        || key_value_heads == 0
        || key_sequence == 0
        || value[3] == 0
    {
        return Err(OperandFault::EmptyAxis);
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return Err(OperandFault::UngroupedHeads);
    }
    if key[0] != batch
        || key[3] != head_dim
        || value[0] != batch
        || value[1] != key_value_heads
        || value[2] != key_sequence
    {
        return Err(OperandFault::KeyValueShape);
    }
    let score = [batch, query_heads, query_sequence, key_sequence];
    if mismatched(mask, score) {
        return Err(OperandFault::MaskShape);
    }
    if misfit_bias(bias, score) {
        return Err(OperandFault::BiasShape);
    }
    let sink_mismatch = match sinks {
        Some(sinks) => sinks != query_heads,
        None => false,
    };
    if sink_mismatch {
        return Err(OperandFault::SinkShape);
    }
    Ok(HeadGeometry {
        batch,
        query_heads,
        key_value_heads,
        query_sequence,
        key_sequence,
        head_dim,
    })
}

/// Configured head layout of a projected attention layer.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeadLayout {
    /// Configured query heads.
    pub(crate) query_heads: usize,
    /// Configured native key/value heads.
    pub(crate) key_value_heads: usize,
    /// Configured channels per head.
    pub(crate) head_dim: usize,
    /// Whether the layer applies an output gate.
    pub(crate) gated: bool,
}

/// Why a projected layer's tensors disagree with its configuration.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionFault {
    /// Batch, query sequence, or key sequence is empty.
    EmptyAxis,
    /// Queries are not `[B, H, Q, d]`.
    QueryShape,
    /// Keys are not `[B, G, K, d]`.
    KeyShape,
    /// Values are not `[B, G, K, d]`.
    ValueShape,
    /// An ungated layer received a gate.
    UnexpectedGate,
    /// A gated layer's gate is not `[B, Q, H * d]`.
    GateShape,
    /// A gated layer received no gate.
    MissingGate,
    /// The mask is not `[B, H, Q, K]`.
    MaskShape,
    /// The bias is not `[1 or B, H, Q, K]`.
    BiasShape,
}

/// Validates a projected layer's tensor shapes against its configured layout.
///
/// Batch `B`, `Q`, and `K` are read from the query and key tensors; every
/// other extent must equal the configuration. Checks run in the order of the
/// [`ProjectionFault`] variants. The gate width `H * d` must be representable,
/// which every validated configuration guarantees.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(fault =>
    requires
        layout.query_heads * layout.head_dim <= usize::MAX,
    ensures
        ({
            let batch = query[0];
            let queries = query[2];
            let keys = key[2];
            let nonempty = batch >= 1 && queries >= 1 && keys >= 1;
            let query_ok = nonempty && same4(
                query,
                [batch, layout.query_heads, queries, layout.head_dim],
            );
            let kv = [batch, layout.key_value_heads, keys, layout.head_dim];
            let key_ok = query_ok && same4(key, kv);
            let value_ok = key_ok && same4(value, kv);
            let gate_width = layout.query_heads * layout.head_dim;
            let gate_ok = value_ok && if layout.gated {
                gate is Some && gate->Some_0[0] == batch && gate->Some_0[1] == queries
                    && gate->Some_0[2] == gate_width
            } else {
                gate is None
            };
            let score = [batch, layout.query_heads, queries, keys];
            let mask_ok = gate_ok && (mask is None || same4(mask->Some_0, score));
            let bias_ok = mask_ok && (bias is None || bias_fits(bias->Some_0, score));
            match fault {
                None => bias_ok,
                Some(ProjectionFault::EmptyAxis) => !nonempty,
                Some(ProjectionFault::QueryShape) => nonempty && !query_ok,
                Some(ProjectionFault::KeyShape) => query_ok && !key_ok,
                Some(ProjectionFault::ValueShape) => key_ok && !value_ok,
                Some(ProjectionFault::UnexpectedGate) => value_ok && !layout.gated && gate is Some,
                Some(ProjectionFault::GateShape) => value_ok && layout.gated && gate is Some
                    && !gate_ok,
                Some(ProjectionFault::MissingGate) => value_ok && layout.gated && gate is None,
                Some(ProjectionFault::MaskShape) => gate_ok && !mask_ok,
                Some(ProjectionFault::BiasShape) => mask_ok && !bias_ok,
            }
        }),
))]
pub(crate) fn projection_fault(
    layout: HeadLayout,
    query: [usize; 4],
    key: [usize; 4],
    value: [usize; 4],
    gate: Option<[usize; 3]>,
    mask: Option<[usize; 4]>,
    bias: Option<[usize; 4]>,
) -> Option<ProjectionFault> {
    let batch = query[0];
    let query_sequence = query[2];
    let key_sequence = key[2];
    if batch == 0 || query_sequence == 0 || key_sequence == 0 {
        return Some(ProjectionFault::EmptyAxis);
    }
    if query[1] != layout.query_heads || query[3] != layout.head_dim {
        return Some(ProjectionFault::QueryShape);
    }
    if key[0] != batch || key[1] != layout.key_value_heads || key[3] != layout.head_dim {
        return Some(ProjectionFault::KeyShape);
    }
    if value[0] != batch
        || value[1] != layout.key_value_heads
        || value[2] != key_sequence
        || value[3] != layout.head_dim
    {
        return Some(ProjectionFault::ValueShape);
    }
    let gate_width = layout.query_heads * layout.head_dim;
    match gate {
        None => {
            if layout.gated {
                return Some(ProjectionFault::MissingGate);
            }
        }
        Some(gate) => {
            if !layout.gated {
                return Some(ProjectionFault::UnexpectedGate);
            }
            if gate[0] != batch || gate[1] != query_sequence || gate[2] != gate_width {
                return Some(ProjectionFault::GateShape);
            }
        }
    }
    let score = [batch, layout.query_heads, query_sequence, key_sequence];
    if mismatched(mask, score) {
        return Some(ProjectionFault::MaskShape);
    }
    if misfit_bias(bias, score) {
        return Some(ProjectionFault::BiasShape);
    }
    None
}

/// Whether a causal call needs an explicit bottom-right mask: `causal && Q != 1`.
///
/// A single query row sits at the last key position, so bottom-right
/// causality excludes none of its keys and the mask can be skipped.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(needed =>
    ensures
        needed == (causal && query_sequence != 1),
        causal && !needed ==> forall|row: int, key: int, keys: int|
            0 <= row < query_sequence && 0 <= key < keys ==> !#[trigger] bottom_right_excludes(
                row,
                key,
                query_sequence as int,
                keys,
            ),
))]
pub(crate) const fn causal_mask_needed(causal: bool, query_sequence: usize) -> bool {
    causal && query_sequence != 1
}

/// Backend-native evaluation strategy of one attention call.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeRoute {
    /// Append one sink key per KV head; `causal` says whether an explicit mask is needed.
    Sinks {
        /// Whether to build a bottom-right causal mask.
        causal: bool,
    },
    /// Square causal plane: repeat K/V and pass the backend's own causal flag.
    SquareCausal,
    /// Fold query groups into rows; `causal` says whether an explicit mask is needed.
    Folded {
        /// Whether to build a bottom-right causal mask.
        causal: bool,
    },
}

/// Chooses the backend-native strategy for a call with `Q` queries and `K` keys.
///
/// Sinks always take their own path. Otherwise the backend's causal flag is
/// used only for square planes, where top-left and bottom-right alignment
/// exclude exactly the same links; every other call folds groups and builds an
/// explicit mask when [`causal_mask_needed`] says causality excludes anything.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(route =>
    ensures
        match route {
            NativeRoute::Sinks { causal } => {
                &&& has_sinks
                &&& causal == (is_causal && query_sequence != 1)
            },
            NativeRoute::SquareCausal => {
                &&& !has_sinks && is_causal && query_sequence != 1
                &&& query_sequence == key_sequence
                &&& forall|row: int, key: int|
                    0 <= row < query_sequence && 0 <= key < key_sequence ==> #[trigger]
                        top_left_excludes(row, key) == bottom_right_excludes(
                        row,
                        key,
                        query_sequence as int,
                        key_sequence as int,
                    )
            },
            NativeRoute::Folded { causal } => {
                &&& !has_sinks
                &&& causal == (is_causal && query_sequence != 1)
                &&& !(causal && query_sequence == key_sequence)
            },
        },
))]
pub(crate) fn native_route(
    is_causal: bool,
    has_sinks: bool,
    query_sequence: usize,
    key_sequence: usize,
) -> NativeRoute {
    let causal = causal_mask_needed(is_causal, query_sequence);
    if has_sinks {
        return NativeRoute::Sinks { causal };
    }
    if causal && query_sequence == key_sequence {
        return NativeRoute::SquareCausal;
    }
    NativeRoute::Folded { causal }
}

/// Elementwise broadcast of two rank-4 shapes: the larger extent on every axis.
///
/// Size-one axes stretch to the other operand's extent, so for broadcast-compatible
/// operands this is their common shape.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(shape =>
    ensures
        forall|axis: int| 0 <= axis < 4 ==> #[trigger] shape[axis]
            == if first[axis] >= second[axis] { first[axis] } else { second[axis] },
))]
pub(crate) fn broadcast_shape(first: [usize; 4], second: [usize; 4]) -> [usize; 4] {
    [
        first[0].max(second[0]),
        first[1].max(second[1]),
        first[2].max(second[2]),
        first[3].max(second[3]),
    ]
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
