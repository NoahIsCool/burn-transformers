//! Verus harness for the architecture-owned checkpoint loaders.
//!
//! Verifies, in place, the specifications attached to the pure integer cores
//! that the family checkpoint contracts and tensor sinks call:
//!
//! - `src/models/gpt_neox/load/packed.rs`: the packed query/key/value split.
//!   `packed_rows` accepts exactly the geometries the sink can split and
//!   returns `R = 3 q d` and `q d` rows; `packed_payload` returns the row and
//!   payload byte sizes exactly or reports overflow; `packed_component` never
//!   indexes outside the payload and returns each component's bytes in head
//!   order, byte `o` being payload byte `(3 floor(o / b) + c) b + o mod b`,
//!   and `lemma_packed_source_bijection` proves the three components
//!   partition the payload; `packed_staging_peak` is the exact staging peak
//!   `P + max_i floor(B_i / 3)` or reports `u64` overflow.
//!
//! See `verus/README.md` for the method, toolchain, and command. String
//! parsing of tensor names, `format!`-built inventories, Burn tensor
//! construction, dtype sizes, and parameter installation stay in the
//! unverified wrappers and are covered by each family's checkpoint and load
//! tests.

#![allow(unused)]
// Loop invariants are attributes on `while` statements, which Verus parses as
// expression attributes; ordinary builds never see these attributes.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]

#[path = "../src/models/gpt_neox/load/packed.rs"]
pub mod gpt_neox_packed;
