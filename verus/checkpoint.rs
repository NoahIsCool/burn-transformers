//! Verus harness for the checkpoint loader's pure layout, naming, scheduling, and packing logic.
//!
//! Verifies, in place, the specifications attached to these sources.
//!
//! # Layout (`src/checkpoint/layout.rs`)
//!
//! - `element_count` returns the exact mathematical shape product when every
//!   left-to-right prefix product fits in `u64`, and `None` exactly when some
//!   prefix overflows, matching the pinned SafeTensors validator;
//!   `checked_range_product` is the same rule on a subslice;
//! - `storage_bytes` returns `n` with `8 * n == elements * bits`, and rejects
//!   overflow and trailing partial bytes precisely;
//! - `validate_contiguous_tiling` returns `Ok` exactly when the sorted
//!   SafeTensors byte intervals tile the data section (soundness and
//!   completeness): they chain from zero, each beginning where the previous
//!   ended, and the last ends at the data-section size, so every interval lies
//!   inside the section, nonempty intervals are disjoint, and their lengths sum
//!   exactly to the data-section size. Each error names the first violation,
//!   with every earlier interval proved valid;
//! - `tensor_file_offset` computes `8 + header + begin` or reports overflow, and
//!   `data_section` places the data section at `8 + header`, ending exactly at
//!   the end of the file, rejecting an empty or oversized header;
//! - `axis_view` has a complete specification: it succeeds exactly when every
//!   running product before and after the axis fits `u64` (a later zero cannot
//!   hide an overflowing prefix) and the host checks pass, returning exact
//!   `[outer, axis, inner]` products whose `outer * axis` and
//!   `outer * axis * inner` fit `usize` (exactly `slice_block`'s
//!   preconditions) and whose product is the full shape product; each error
//!   names the first failed check, with every earlier check proved to pass;
//!   `axis_elements` and `range_within` decide their products and ranges
//!   exactly;
//! - `slice_block` keeps every slice and concatenation block inside its tensor
//!   with overflow-free arithmetic, and consecutive target blocks partition the
//!   output in order;
//! - `section_starts` returns prefix-sum section starts only when the split
//!   lengths sum exactly to the axis, so every section lies inside the axis, and
//!   reports overflow or the exact mismatching total otherwise;
//! - `extend_concatenation` adds a source's axis length to the running target
//!   shape exactly when the source agrees on every other axis and the sum fits,
//!   and leaves the shape unchanged on either error;
//! - `checked_host_product` and `dense_byte_len` compute host element and byte
//!   counts or report the overflowing running product.
//!
//! # Untrusted names (`src/checkpoint/lexical.rs`)
//!
//! - `lexical_path_violation` names exactly the first broken rule (empty, NUL,
//!   backslash, Windows drive prefix) and returns `None` exactly when none is
//!   broken;
//! - `normal_components` holds exactly when the path is not rooted, does not
//!   begin with a `.` component, and has no `..` component anywhere (the Unix
//!   `Path::components` rule, cross-checked exhaustively by tests), so an
//!   accepted path cannot climb out of the directory it is joined to;
//!   `canonical_components` additionally rejects empty and `.` components, so a
//!   provenance shard has one key;
//! - `decimal_layer_between` holds exactly when a name is `prefix ++ layer ++
//!   suffix` with a canonical decimal layer (digits, no leading zero except
//!   `0`), with non-overlapping anchors;
//! - `integral_lexeme_digits` accepts exactly the lexemes whose part before the
//!   first `.` is nonempty digits and whose fraction, if any, is nonempty zeros,
//!   returning the integer digits' length; `is_lowercase_sha256` accepts exactly
//!   64 lowercase hexadecimal bytes.
//!
//! # Planning and execution
//!
//! - `src/checkpoint/mapping/aliases.rs`: `walk_alias_chain` always terminates
//!   (each alias is visited at most once) and returns exactly the first event of
//!   the deterministic chain from its start: the aliases before it are distinct
//!   and each links onward, and the reported alias is the first one that is
//!   revisited (a cycle), owned by a primary target, or dangling, so every
//!   resolved alias reaches its storage owner along an acyclic chain;
//! - `src/checkpoint/mapping/ledger.rs`: the target claim ledger starts with
//!   every expected target open; `claim` succeeds exactly on an open target and
//!   changes only that entry, so a primary operation and an alias (or two of
//!   either) can never both own one target, whichever is declared first;
//!   `resolve` turns exactly a declared alias into a resolved one; and
//!   `is_produced` holds exactly for primaries and resolved aliases, the
//!   complement being the targets reported missing;
//! - `src/checkpoint/runtime/schedule.rs`: `execution_group_indices` returns
//!   exactly the group of each leader (a classless target, or the first target
//!   of its shared-source class) in index order, and those groups partition the
//!   plan: every target lies in exactly one group, each group is nonempty and
//!   strictly increasing, and a group's targets share one class, so the
//!   executor installs every primary target exactly once;
//! - `src/checkpoint/runtime/permutation.rs`: `probe_dimension` lengths strictly
//!   increase with the axis (so probe axes are pairwise distinct), and for
//!   distinct input lengths `permutation_of` succeeds exactly when the mapped
//!   shape is a rearrangement of the probe, recovering a bijection `axes` with
//!   `output[k] == input[axes[k]]`;
//! - `src/checkpoint/runtime/rounding.rs`: `round_shifted` returns the integer
//!   nearest `significand / 2^shift` with ties to even (within half a unit, an
//!   exact tie gives an even result), bounded by `2^(53 - shift)`, for every
//!   shift the binary64-to-16-bit narrowing uses.
//!
//! # Quantized storage
//!
//! - `src/checkpoint/quantization/packing.rs`: Q8/Q4 packed lengths (`n` and
//!   `ceil(n / 2)`, never above `n`), the two's-complement nibble and byte
//!   encodings and their inverses, nibble joining and splitting, and code reads
//!   that return exactly the stored code (Q8 byte `i`; Q4 low nibble of byte
//!   `i / 2` for even `i`, high nibble for odd `i`) and `None` exactly past the
//!   end; `push_code` preserves the packing invariant (a buffer holding codes
//!   `0..i` becomes one holding `0..=i`, every earlier code unchanged, the new
//!   code readable as pushed, and an unused Q4 high nibble kept zero), so
//!   packing then reading is the identity; `first_code_violation` returns
//!   `None` exactly for canonical buffers and otherwise names the first short
//!   read, reserved code, or dirty Q4 padding;
//! - `src/checkpoint/division.rs` (shared with `verus/perf.rs`): exact
//!   ceilings in `u64` and `usize`, each the least `q` with `n <= q · d`,
//!   never above `n` and positive for positive `n`;
//! - `src/checkpoint/quantization/counts.rs`: persisted group (`⌈n / G⌉`), packed-byte, and scale-byte counts,
//!   with the scale length proved never to overflow for `G >= 32`; and the Burn
//!   block count `rows · (columns / G)`, proved equal to the flat group count
//!   `⌈rows · columns / G⌉` whenever `G` divides the width; `group_index` places
//!   each element in its group and below the group count, so dequantization
//!   never indexes past a validated scale vector;
//! - `src/checkpoint/quantization/staging.rs`: `bounded_add` admits exactly the
//!   sums within a limit (so a running total kept below its limit stays below
//!   it, with no wraparound), and `admit_tensor` admits a tensor exactly when
//!   the new count, its payload, and the new staged total are within their
//!   limits, naming the first breached limit otherwise.
//!
//! See `verus/README.md` for the method, toolchain, and command. Tensor copies
//! driven by these plans, file I/O, error mapping, the string-keyed planner
//! bookkeeping around the ledger (name lookups and the consumed-source set),
//! and the bit assembly of the floating-point rounding encoder (sign, exponent
//! field, NaN payload; checked exhaustively at every 16-bit midpoint and
//! against independent oracles) remain covered by tests rather than proofs.

// Loop invariants are attributes on `while` expressions, which rustc accepts
// only with this feature; it applies solely under Verus.
#![cfg_attr(verus_keep_ghost, feature(proc_macro_hygiene))]
#![allow(unused)]

#[path = "../src/checkpoint/division.rs"]
pub mod division;

#[path = "../src/checkpoint/layout.rs"]
pub mod layout;

#[path = "../src/checkpoint/lexical.rs"]
pub mod lexical;

#[path = "../src/checkpoint/runtime/permutation.rs"]
pub mod permutation;

#[path = "../src/checkpoint/runtime/schedule.rs"]
pub mod schedule;

#[path = "../src/checkpoint/runtime/rounding.rs"]
pub mod rounding;

#[path = "../src/checkpoint/mapping/aliases.rs"]
pub mod aliases;

#[path = "../src/checkpoint/mapping/ledger.rs"]
pub mod ledger;

#[path = "../src/checkpoint/quantization/packing.rs"]
pub mod packing;

#[path = "../src/checkpoint/quantization/counts.rs"]
pub mod counts;

#[path = "../src/checkpoint/quantization/staging.rs"]
pub mod staging;

/// Mirrors the crate path `crate::checkpoint::division` used by the counts.
pub mod checkpoint {
    pub use crate::division;
}
