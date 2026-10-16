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

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: `visited` marks exactly the first `steps` walk positions from `start`.
pub(crate) open spec fn marks_walk(
    links: Seq<AliasLink>,
    visited: Seq<bool>,
    start: int,
    from: nat,
    steps: nat,
) -> bool {
    forall|x: int| 0 <= x < visited.len() ==> (#[trigger] visited[x] <==> exists|m: nat|
        from <= m < steps && #[trigger] walk(links, start, m) == x)
}

/// Specification: every entry of `visited` is clear.
pub(crate) open spec fn all_clear(visited: Seq<bool>) -> bool {
    forall|x: int| 0 <= x < visited.len() ==> !#[trigger] visited[x]
}

} // verus!

/// Walks the alias chain from alias `start` to its first event.
///
/// `visited` is caller-owned scratch space with one entry per alias, all
/// clear on entry; the walk marks the aliases it passes and clears them again
/// before returning, so one buffer serves every walk of a resolution pass and
/// each walk costs time proportional to its chain, not to the alias count.
///
/// Each step first checks whether the current alias was already visited
/// (a cycle), then follows its link: an owner or dangling source ends the
/// walk, another alias is marked and continues it.
///
/// Verified: the walk always terminates, after at most one visit per alias,
/// and the outcome is exactly the first event of the deterministic chain: the
/// aliases before it are distinct and each links onward, and the reported
/// alias is the first one that is revisited, owned, or dangling. In
/// particular `Owner(k)` means following the sources from `start` reaches
/// alias `k`, whose source is a storage owner, without a cycle. On return
/// every `visited` entry is clear again.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(outcome =>
    requires
        start < links@.len(),
        links_closed(links@),
        old(visited)@.len() == links@.len(),
        all_clear(old(visited)@),
    ensures
        exists|steps: nat| first_event(links@, start as int, steps, outcome),
        final(visited)@.len() == links@.len(),
        all_clear(final(visited)@),
))]
pub(crate) fn walk_alias_chain(
    links: &[AliasLink],
    start: usize,
    visited: &mut [bool],
) -> AliasWalk {
    let (outcome, marked) = mark_alias_chain(links, start, visited);
    clear_alias_chain(links, start, marked, visited);
    outcome
}

/// Walks from `start` to the first event, marking each alias that links onward.
///
/// Verified: returns the first event after exactly `marked` distinct onward
/// links, and on return `visited` marks exactly those `marked` aliases.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(result =>
    requires
        start < links@.len(),
        links_closed(links@),
        old(visited)@.len() == links@.len(),
        all_clear(old(visited)@),
    ensures
        first_event(links@, start as int, result.1 as nat, result.0),
        final(visited)@.len() == links@.len(),
        marks_walk(links@, final(visited)@, start as int, 0, result.1 as nat),
))]
fn mark_alias_chain(links: &[AliasLink], start: usize, visited: &mut [bool]) -> (AliasWalk, usize) {
    #[cfg(verus_keep_ghost)]
    proof_decl! {
        let ghost mut seen: Set<int> = Set::empty();
    }
    let mut marked: usize = 0;
    let mut cursor = start;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            visited@.len() == links@.len(),
            links_closed(links@),
            cursor < links@.len(),
            cursor as int == walk(links@, start as int, marked as nat),
            simple_prefix(links@, start as int, marked as nat),
            seen.len() == marked,
            forall|x: int| #[trigger] seen.contains(x) ==> 0 <= x < links@.len(),
            forall|x: int| 0 <= x < links@.len() ==> (#[trigger] visited@[x] <==> seen.contains(x)),
            forall|x: int| #[trigger] seen.contains(x)
                <==> exists|m: nat| m < marked && #[trigger] walk(links@, start as int, m) == x,
            marked <= links@.len(),
        decreases links@.len() - marked,
    ))]
    while !visited[cursor] {
        match links[cursor] {
            AliasLink::Owner => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(!seen.contains(cursor as int));
                    assert(first_event(links@, start as int, marked as nat, AliasWalk::Owner(cursor)));
                }
                return (AliasWalk::Owner(cursor), marked);
            }
            AliasLink::Dangling => {
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(!seen.contains(cursor as int));
                    assert(first_event(links@, start as int, marked as nat, AliasWalk::Dangling(cursor)));
                }
                return (AliasWalk::Dangling(cursor), marked);
            }
            AliasLink::Next(next) => {
                visited[cursor] = true;
                #[cfg(verus_keep_ghost)]
                proof! {
                    // A new alias joins the visited set, which stays within `0..links@.len()`.
                    assert(!seen.contains(cursor as int));
                    vstd::set_lib::lemma_int_range(0, links@.len() as int);
                    vstd::set_lib::lemma_len_subset(
                        seen.insert(cursor as int),
                        vstd::set_lib::set_int_range(0, links@.len() as int),
                    );
                    let previous = marked as nat;
                    let earlier = seen;
                    assert(walk(links@, start as int, (previous + 1) as nat) == next as int);
                    seen = seen.insert(cursor as int);
                    assert forall|x: int| #[trigger] seen.contains(x)
                        <==> exists|m: nat| m < previous + 1
                            && #[trigger] walk(links@, start as int, m) == x
                    by {
                        if x == cursor as int {
                            assert(walk(links@, start as int, previous) == x);
                        }
                    }
                    // The new position `cursor` was unvisited, so it differs from every earlier one.
                    assert forall|a: nat, b: nat| a < b < previous + 1
                        implies #[trigger] walk(links@, start as int, a)
                            != #[trigger] walk(links@, start as int, b)
                    by {
                        if b == previous {
                            assert(earlier.contains(walk(links@, start as int, a)));
                        }
                    }
                }
                marked += 1;
                cursor = next;
            }
        }
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert(seen.contains(cursor as int));
        assert(first_event(links@, start as int, marked as nat, AliasWalk::Cycle(cursor)));
    }
    (AliasWalk::Cycle(cursor), marked)
}

/// Clears the marks of the first `marked` aliases on the walk from `start`.
///
/// Verified: given marks of exactly that distinct prefix, every entry is
/// clear on return.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(
    requires
        start < links@.len(),
        links_closed(links@),
        old(visited)@.len() == links@.len(),
        marked <= links@.len(),
        simple_prefix(links@, start as int, marked as nat),
        marks_walk(links@, old(visited)@, start as int, 0, marked as nat),
    ensures
        final(visited)@.len() == links@.len(),
        all_clear(final(visited)@),
))]
fn clear_alias_chain(links: &[AliasLink], start: usize, marked: usize, visited: &mut [bool]) {
    let mut cursor = start;
    let mut cleared: usize = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            visited@.len() == links@.len(),
            links_closed(links@),
            cleared <= marked,
            marked <= links@.len(),
            cursor < links@.len(),
            cursor as int == walk(links@, start as int, cleared as nat),
            simple_prefix(links@, start as int, marked as nat),
            marks_walk(links@, visited@, start as int, cleared as nat, marked as nat),
        decreases marked - cleared,
    ))]
    while cleared < marked {
        #[cfg(verus_keep_ghost)]
        proof_decl! {
            let ghost before = visited@;
        }
        visited[cursor] = false;
        #[cfg(verus_keep_ghost)]
        proof! {
            // The cleared alias appears once on the distinct prefix, so no later mark is lost.
            assert forall|x: int| 0 <= x < visited@.len() implies (#[trigger] visited@[x]
                <==> exists|m: nat| cleared + 1 <= m < marked
                    && #[trigger] walk(links@, start as int, m) == x)
            by {
                if x == cursor as int {
                    assert forall|m: nat| cleared + 1 <= m < marked
                        implies #[trigger] walk(links@, start as int, m) != x
                    by {
                        assert(walk(links@, start as int, cleared as nat)
                            != walk(links@, start as int, m));
                    }
                } else {
                    assert(visited@[x] == before[x]);
                    if exists|m: nat| cleared <= m < marked
                        && #[trigger] walk(links@, start as int, m) == x {
                        let m = choose|m: nat| cleared <= m < marked
                            && #[trigger] walk(links@, start as int, m) == x;
                        assert(m != cleared);
                    }
                }
            }
            assert(walk(links@, start as int, (cleared + 1) as nat)
                == step(links@, walk(links@, start as int, cleared as nat)));
        }
        cursor = match links[cursor] {
            AliasLink::Next(next) => next,
            _ => cursor,
        };
        cleared += 1;
    }
}

#[cfg(test)]
#[path = "aliases_tests.rs"]
mod tests;
