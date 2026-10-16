//! Resolution of tied-parameter alias chains to their storage owners.
//!
//! A mapping plan may declare alias `a` of target `b`, where `b` is itself an
//! alias or a storage-owning primary target. Numbering the alias targets
//! `0..A`, alias `i` links to its source: the storage owner ([`AliasLink::Owner`]
//! when the source is a primary target), another alias ([`AliasLink::Next`]),
//! or nothing the plan produces ([`AliasLink::Dangling`]). Resolving alias `s`
//! walks `s, next(s), next(next(s)), …` until it reaches an alias whose source
//! is the owner, a dangling source, or an alias it already visited (a cycle).
//!
//! The module has no crate or Burn dependencies so that Verus can check it in
//! place through `verus/checkpoint.rs`; the planner numbers the aliases and
//! maps the outcome back to target names.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Where one alias's source points.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AliasLink {
    /// The source is a storage-owning primary target.
    Owner,
    /// The source is the alias whose index this holds.
    Next(
        /// Index of the source alias.
        usize,
    ),
    /// The source is neither a primary target nor an alias.
    Dangling,
}

/// The first event on an alias chain's walk.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AliasWalk {
    /// The source of the alias whose index this holds is the storage owner.
    Owner(
        /// Index of the last alias on the chain.
        usize,
    ),
    /// The alias whose index this holds was reached a second time.
    Cycle(
        /// Index of the first alias visited twice.
        usize,
    ),
    /// The source of the alias whose index this holds names nothing the plan produces.
    Dangling(
        /// Index of the alias with the dangling source.
        usize,
    ),
}

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: the alias reached from `at` in one step (`at` itself at a chain end).
pub(crate) open spec fn step(links: Seq<AliasLink>, at: int) -> int {
    match links[at] {
        AliasLink::Next(next) => next as int,
        _ => at,
    }
}

/// Specification: the alias reached from `start` after `steps` steps.
pub(crate) open spec fn walk(links: Seq<AliasLink>, start: int, steps: nat) -> int
    decreases steps,
{
    if steps == 0 {
        start
    } else {
        step(links, walk(links, start, (steps - 1) as nat))
    }
}

/// Specification: the first `steps` walk positions are distinct aliases that each link onward.
pub(crate) open spec fn simple_prefix(links: Seq<AliasLink>, start: int, steps: nat) -> bool {
    &&& forall|m: nat| m < steps ==> {
        &&& 0 <= #[trigger] walk(links, start, m) < links.len()
        &&& links[walk(links, start, m)] is Next
    }
    &&& forall|a: nat, b: nat| a < b < steps
        ==> #[trigger] walk(links, start, a) != #[trigger] walk(links, start, b)
}

/// Specification: every `Next` link names an alias.
pub(crate) open spec fn links_closed(links: Seq<AliasLink>) -> bool {
    forall|i: int| 0 <= i < links.len() ==> match #[trigger] links[i] {
        AliasLink::Next(next) => next < links.len(),
        _ => true,
    }
}

/// Specification: `outcome` is the first event after the simple prefix of `steps` steps.
pub(crate) open spec fn first_event(
    links: Seq<AliasLink>,
    start: int,
    steps: nat,
    outcome: AliasWalk,
) -> bool {
    let at = walk(links, start, steps);
    &&& simple_prefix(links, start, steps)
    &&& steps <= links.len()
    &&& 0 <= at < links.len()
    &&& match outcome {
        AliasWalk::Cycle(alias) => alias == at
            && exists|m: nat| m < steps && #[trigger] walk(links, start, m) == at,
        AliasWalk::Owner(alias) => alias == at && links[at] is Owner
            && forall|m: nat| m < steps ==> #[trigger] walk(links, start, m) != at,
        AliasWalk::Dangling(alias) => alias == at && links[at] is Dangling
            && forall|m: nat| m < steps ==> #[trigger] walk(links, start, m) != at,
    }
}

} // verus!

/// Walks the alias chain from alias `start` to its first event.
///
/// Each step first checks whether the current alias was already visited
/// (a cycle), then marks it and follows its link: an owner or dangling source
/// ends the walk, another alias continues it.
///
/// Verified: the walk always terminates, after at most one visit per alias,
/// and the outcome is exactly the first event of the deterministic chain: the
/// aliases before it are distinct and each links onward, and the reported
/// alias is the first one that is revisited, owned, or dangling. In
/// particular `Owner(k)` means following the sources from `start` reaches
/// alias `k`, whose source is a storage owner, without a cycle.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(outcome =>
    requires
        start < links@.len(),
        links_closed(links@),
    ensures
        exists|steps: nat| first_event(links@, start as int, steps, outcome),
))]
pub(crate) fn walk_alias_chain(links: &[AliasLink], start: usize) -> AliasWalk {
    let count = links.len();
    let mut visited: Vec<bool> = Vec::new();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            visited@.len() <= count,
            forall|i: int| 0 <= i < visited@.len() ==> !#[trigger] visited@[i],
        decreases count - visited@.len(),
    ))]
    while visited.len() < count {
        visited.push(false);
    }
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost mut steps: nat = 0;
        let ghost mut seen: Set<int> = Set::empty();
    }
    let mut cursor = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            count == links@.len(),
            visited@.len() == count,
            links_closed(links@),
            cursor < count,
            cursor as int == walk(links@, start as int, steps),
            simple_prefix(links@, start as int, steps),
            seen.len() == steps,
            forall|x: int| #[trigger] seen.contains(x) ==> 0 <= x < count,
            forall|x: int| 0 <= x < count ==> (#[trigger] visited@[x] <==> seen.contains(x)),
            forall|x: int| #[trigger] seen.contains(x)
                <==> exists|m: nat| m < steps && #[trigger] walk(links@, start as int, m) == x,
            steps <= count,
        decreases count - steps,
    ))]
    while !visited[cursor] {
        visited[cursor] = true;
        #[cfg(verus_keep_ghost)]
        proof! {
            // A new alias joins the visited set, which stays within `0..count`.
            assert(!seen.contains(cursor as int));
            vstd::set_lib::lemma_int_range(0, count as int);
            vstd::set_lib::lemma_len_subset(
                seen.insert(cursor as int),
                vstd::set_lib::set_int_range(0, count as int),
            );
        }
        match links[cursor] {
            AliasLink::Owner => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(first_event(links@, start as int, steps, AliasWalk::Owner(cursor)));
                }
                return AliasWalk::Owner(cursor);
            }
            AliasLink::Dangling => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(first_event(links@, start as int, steps, AliasWalk::Dangling(cursor)));
                }
                return AliasWalk::Dangling(cursor);
            }
            AliasLink::Next(next) => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    let previous = steps;
                    let earlier = seen;
                    assert(walk(links@, start as int, (previous + 1) as nat) == next as int);
                    seen = seen.insert(cursor as int);
                    steps = steps + 1;
                    assert forall|x: int| #[trigger] seen.contains(x)
                        <==> exists|m: nat| m < steps && #[trigger] walk(links@, start as int, m) == x
                    by {
                        if x == cursor as int {
                            assert(walk(links@, start as int, previous) == x);
                        }
                    }
                    // The new position `cursor` was unvisited, so it differs from every earlier one.
                    assert forall|a: nat, b: nat| a < b < steps
                        implies #[trigger] walk(links@, start as int, a)
                            != #[trigger] walk(links@, start as int, b)
                    by {
                        if b == previous {
                            assert(earlier.contains(walk(links@, start as int, a)));
                        }
                    }
                }
                cursor = next;
            }
        }
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(seen.contains(cursor as int));
        assert(first_event(links@, start as int, steps, AliasWalk::Cycle(cursor)));
    }
    AliasWalk::Cycle(cursor)
}

#[cfg(test)]
#[path = "aliases_tests.rs"]
mod tests;
