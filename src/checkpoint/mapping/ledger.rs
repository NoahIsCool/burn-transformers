//! Claim ledger enforcing that every expected model target is produced at most once.
//!
//! Each expected target, numbered by its position in the architecture's
//! expectation list, moves through a small state machine:
//!
//! ```text
//! Open --claim(Primary)--> Primary
//! Open --claim(Alias)----> Alias --resolve--> ResolvedAlias
//! ```
//!
//! A claim succeeds only on an `Open` target, so a primary operation and an
//! alias, or two of either, can never both own one target whichever is
//! declared first (the planner's collision rule). After planning, the
//! produced targets are exactly the `Primary` and `ResolvedAlias` entries, and
//! every other expected target is reported missing.
//!
//! The module has no crate or Burn dependencies so that Verus can check it in
//! place through `verus/checkpoint.rs`.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Planning state of one expected target.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetState {
    /// No operation has claimed the target yet.
    Open,
    /// A primary operation produces the target's storage.
    Primary,
    /// An alias was declared for the target and awaits resolution.
    Alias,
    /// The target's alias resolved to a storage owner.
    ResolvedAlias,
}

/// Per-target planning states, indexed by expectation position.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug)]
pub(crate) struct TargetLedger {
    /// State of each expected target.
    pub(crate) states: Vec<TargetState>,
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: a target in `state` is produced by the plan.
pub(crate) open spec fn produced(state: TargetState) -> bool {
    state == TargetState::Primary || state == TargetState::ResolvedAlias
}

} // verus!

/// Creates a ledger of `count` open targets.
///
/// Verified: exactly `count` entries, all `Open`.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(ledger =>
    ensures
        ledger.states@.len() == count,
        forall|i: int| 0 <= i < count ==> #[trigger] ledger.states@[i] == TargetState::Open,
))]
pub(crate) fn open_ledger(count: usize) -> TargetLedger {
    let mut states = Vec::new();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            states@.len() <= count,
            forall|i: int| 0 <= i < states@.len() ==> #[trigger] states@[i] == TargetState::Open,
        decreases count - states@.len(),
    ))]
    while states.len() < count {
        states.push(TargetState::Open);
    }
    TargetLedger { states }
}

impl TargetLedger {
    /// Claims open target `index` as `claimed` (`Primary` or `Alias`).
    ///
    /// Verified: succeeds exactly when the target was `Open` and then records
    /// `claimed` there and nowhere else; a failed claim (a collision) changes
    /// nothing. Hence each target is claimed at most once.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(claimed_now =>
        requires
            index < old(self).states@.len(),
            claimed == TargetState::Primary || claimed == TargetState::Alias,
        ensures
            claimed_now == (old(self).states@[index as int] == TargetState::Open),
            final(self).states@ == if claimed_now {
                old(self).states@.update(index as int, claimed)
            } else {
                old(self).states@
            },
    ))]
    pub(crate) fn claim(&mut self, index: usize, claimed: TargetState) -> bool {
        // A `match` keeps the state test Verus-checkable (derived `==` has no specification).
        match self.states[index] {
            TargetState::Open => {
                self.states[index] = claimed;
                true
            }
            TargetState::Primary | TargetState::Alias | TargetState::ResolvedAlias => false,
        }
    }

    /// Marks declared alias `index` as resolved to its storage owner.
    ///
    /// Verified: succeeds exactly when the target was an unresolved `Alias`,
    /// then makes it `ResolvedAlias` and changes nothing else; otherwise the
    /// ledger is unchanged.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(resolved =>
        requires
            index < old(self).states@.len(),
        ensures
            resolved == (old(self).states@[index as int] == TargetState::Alias),
            final(self).states@ == if resolved {
                old(self).states@.update(index as int, TargetState::ResolvedAlias)
            } else {
                old(self).states@
            },
    ))]
    pub(crate) fn resolve(&mut self, index: usize) -> bool {
        match self.states[index] {
            TargetState::Alias => {
                self.states[index] = TargetState::ResolvedAlias;
                true
            }
            TargetState::Open | TargetState::Primary | TargetState::ResolvedAlias => false,
        }
    }

    /// Returns whether target `index` is produced: a primary or a resolved alias.
    ///
    /// Verified: true exactly for `Primary` and `ResolvedAlias`.
    #[cfg_attr(verus_keep_ghost, verus_verify)]
    #[cfg_attr(verus_keep_ghost, verus_spec(is_produced =>
        requires
            index < self.states@.len(),
        ensures
            is_produced == produced(self.states@[index as int]),
    ))]
    pub(crate) fn is_produced(&self, index: usize) -> bool {
        match self.states[index] {
            TargetState::Primary | TargetState::ResolvedAlias => true,
            TargetState::Open | TargetState::Alias => false,
        }
    }
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
