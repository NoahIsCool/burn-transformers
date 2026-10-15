//! Model-independent runtime glue shared by every family implementation.
//!
//! Families differ in their mixers, normalization, and checkpoint wiring, but
//! all of them must enforce the same public input contract, number tokens,
//! choose which hidden states reach the vocabulary projection and, when they
//! decode against an external cache, compact padded rows and restore them.
//! This module owns that glue once so every family gets the same semantics,
//! errors, and performance characteristics:
//!
//! * [`input`] validates stateless [`CausalLmInput`](crate::CausalLmInput)s
//!   against an [`InputContract`] and resolves explicit or implicit positions;
//! * [`logits`] selects `All`, `Last`, or explicit positions before the
//!   vocabulary projection;
//! * [`rows`] validates stateful calls and plans per-row compaction and
//!   scatter with a constant number of kernels per tensor;
//! * [`checks`] batches every validation statistic into one device-to-host
//!   readback;
//! * [`plan`] holds the dependency-free index planning, verified in place by
//!   Verus through `verus/models.rs`.
//!
//! # Error convention
//!
//! Geometry and value-range violations (empty grids, mismatched extents,
//! out-of-range tokens or positions, fully masked rows, invalid logits
//! selections) are [`Error::Shape`](crate::Error::Shape) naming the offending
//! tensor; device mismatches and failed readbacks are
//! [`Error::BackendCapability`](crate::Error::BackendCapability); cache
//! cursor and capacity violations are
//! [`Error::TensorCache`](crate::Error::TensorCache).
//!
//! The module deliberately contains no model-name branches: a family passes
//! its tensor names and numbering policy as data.

pub(crate) mod checks;
pub(crate) mod input;
pub(crate) mod logits;
pub(crate) mod plan;
pub(crate) mod rows;
#[cfg(test)]
mod test_support;

pub(crate) use checks::{
    DeferredChecks, ensure_grid, ensure_nonempty, ensure_same_device, range_error,
};
pub(crate) use input::{
    DeferredInput, ImplicitPositions, InputContract, InputNames, PaddedPositions, ValidatedInput,
    check_nonempty_mask, defer_input, validate_input, validate_nonempty_mask,
    validate_position_values, validate_token_values,
};
pub(crate) use logits::{
    select_hidden, select_hidden_with_last_columns, validate_logits_selection,
};
pub(crate) use rows::{
    CompactRow, EmptyRows, HostInput, StatefulContract, StatefulRows, concat_rows,
    ensure_static_rotary_context, implicit_position_ids, index_tensor, prepare_stateful_rows,
};
