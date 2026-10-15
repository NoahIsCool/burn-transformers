//! Pure precedence and tie bookkeeping behind transfer plans.
//!
//! A transfer plan resolves each target parameter against ordered rules and
//! then reconciles existing parameter sharing. The string matching, tensor
//! staging, and error formatting stay in [`super`]; the decisions they feed
//! are integer and enumeration logic, free of Burn types and allocation, so
//! Verus checks these exact bodies through `verus/transfer.rs` (see
//! `verus/README.md`).
//!
//! # Verified properties
//!
//! - [`select_rule`] implements the documented precedence exactly: a kept
//!   target is never selected; otherwise the first exact rule wins; otherwise
//!   exactly one prefix/matching rule is selected, none leaves the target
//!   unselected, and two or more are reported as overlapping.
//! - [`target_tie`] accepts a pre-existing target tie exactly when no member
//!   was selected, or the plan unties it, or every member was selected with
//!   one shared source identity, transformation, and trainability.
//! - [`first_fanned_out_path`] returns exactly the smallest source path that
//!   reaches two different target tie classes, so `None` means the map from
//!   source path to target class is a function.
//! - [`class_leaders`] assigns every entry the first entry of its target tie
//!   class; two entries share a leader exactly when they share a class, so
//!   shared identities join each class and never merge two classes.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// How one transfer rule relates to one target parameter path.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuleMatch {
    /// The rule does not supply this target.
    Unrelated,
    /// An exact copy or transform names this target.
    Exact,
    /// A prefix or matching-path rule covers this target.
    Fallback,
}

/// Which rule, if any, supplies one target parameter.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuleSelection {
    /// The target keeps its initialization.
    Unselected,
    /// Rule `index` supplies the target.
    Rule(
        /// Index of the selected rule in plan order.
        usize,
    ),
    /// Several prefix or matching rules cover the target and none is exact.
    Overlapping,
}

/// Resolves one target: kept targets first, then the first exact rule, then a unique fallback.
///
/// `kept` says whether a keep-initialized prefix covers the target, and
/// `matches[i]` how rule `i` relates to it. Several fallbacks without an
/// exact override are ambiguous, so rule order never silently chooses
/// between overlapping prefixes.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(selection =>
    ensures
        match selection {
            RuleSelection::Unselected => kept || (!has_exact(matches@)
                && count_fallbacks(matches@, matches.len() as int) == 0),
            RuleSelection::Rule(index) => {
                &&& !kept
                &&& index < matches.len()
                &&& (is_first_exact(matches@, index as int) || {
                    &&& !has_exact(matches@)
                    &&& matches@[index as int] == RuleMatch::Fallback
                    &&& count_fallbacks(matches@, matches.len() as int) == 1
                    &&& forall|other: int| 0 <= other < matches.len() && other != index
                        ==> matches@[other] != RuleMatch::Fallback
                })
            },
            RuleSelection::Overlapping => !kept && !has_exact(matches@)
                && count_fallbacks(matches@, matches.len() as int) >= 2,
        },
))]
pub(crate) fn select_rule(kept: bool, matches: &[RuleMatch]) -> RuleSelection {
    if kept {
        return RuleSelection::Unselected;
    }
    let mut fallback: Option<usize> = None;
    let mut fallbacks: usize = 0;
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            !kept,
            0 <= index <= matches.len(),
            fallbacks == count_fallbacks(matches@, index as int),
            fallbacks <= index,
            forall|earlier: int| 0 <= earlier < index ==> matches@[earlier] != RuleMatch::Exact,
            match fallback {
                None => {
                    &&& fallbacks == 0
                    &&& forall|earlier: int| 0 <= earlier < index
                        ==> matches@[earlier] != RuleMatch::Fallback
                },
                Some(first) => {
                    &&& first < index
                    &&& matches@[first as int] == RuleMatch::Fallback
                    &&& fallbacks >= 1
                    &&& fallbacks == 1 ==> forall|other: int| 0 <= other < index && other != first
                        ==> matches@[other] != RuleMatch::Fallback
                },
            },
        decreases matches.len() - index,
    ))]
    while index < matches.len() {
        match matches[index] {
            RuleMatch::Exact => return RuleSelection::Rule(index),
            RuleMatch::Fallback => {
                if fallback.is_none() {
                    fallback = Some(index);
                }
                fallbacks += 1;
            }
            RuleMatch::Unrelated => {}
        }
        index += 1;
    }
    match fallback {
        None => RuleSelection::Unselected,
        Some(first) => {
            if fallbacks == 1 {
                RuleSelection::Rule(first)
            } else {
                RuleSelection::Overlapping
            }
        }
    }
}

/// Selection state of one target that already shares a parameter identity.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StagedKey {
    /// Identity of the source parameter supplying the target.
    pub(crate) origin: u64,
    /// Transform rule index, or `None` for a direct copy.
    pub(crate) transform: Option<usize>,
    /// Whether the target participates in differentiation.
    pub(crate) trainable: bool,
}

/// How a transfer treats one group of target paths that share a parameter identity.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TargetTie {
    /// No member is selected; the tie keeps its identity and is reported as preserved.
    Untouched,
    /// The plan explicitly unties the selected group.
    Untied,
    /// Every member receives the same source, transformation, and trainability.
    Consistent,
    /// Members are selected inconsistently; preserving the tie would be wrong.
    Inconsistent,
}

/// Classifies a target tie group from each member's staged selection.
///
/// `members[i]` is `None` when member `i` keeps its initialization. A group
/// with any selected member is untied under `untie`; otherwise every member
/// must be selected with the same key.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(tie =>
    ensures
        (tie == TargetTie::Untouched) == (forall|i: int| 0 <= i < members.len()
            ==> members@[i].is_none()),
        (tie == TargetTie::Untied) == (untie && exists|i: int| 0 <= i < members.len()
            && members@[i].is_some()),
        (tie == TargetTie::Consistent) == (!untie && members.len() > 0 && members@[0].is_some()
            && forall|i: int| #![trigger members@[i]] 0 <= i < members.len()
                ==> members@[i] == members@[0]),
))]
pub(crate) fn target_tie(members: &[Option<StagedKey>], untie: bool) -> TargetTie {
    let mut selected = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= selected <= members.len(),
            forall|i: int| 0 <= i < selected ==> members@[i].is_none(),
        decreases members.len() - selected,
    ))]
    while selected < members.len() && members[selected].is_none() {
        selected += 1;
    }
    if selected == members.len() {
        return TargetTie::Untouched;
    }
    if untie {
        return TargetTie::Untied;
    }
    let mut member = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            !untie,
            0 <= member <= members.len(),
            selected < members.len(),
            members@[selected as int].is_some(),
            forall|i: int| 0 <= i < selected ==> members@[i].is_none(),
            forall|i: int| #![trigger members@[i]] 0 <= i < member
                ==> members@[i] == members@[selected as int],
        decreases members.len() - member,
    ))]
    while member < members.len() {
        if !same_selection(members[member], members[selected]) {
            #[cfg(verus_keep_ghost)]
            proof! {
                if selected > 0 {
                    assert(members@[0].is_none());
                } else {
                    assert(members@[member as int] != members@[0]);
                }
            }
            return TargetTie::Inconsistent;
        }
        member += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        if selected > 0 {
            assert(members@[0] == members@[selected as int]);
        }
    }
    TargetTie::Consistent
}

/// Whether two members are both selected with the same origin, transformation, and trainability.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(same =>
    ensures
        same == (member.is_some() && member == reference),
))]
fn same_selection(member: Option<StagedKey>, reference: Option<StagedKey>) -> bool {
    match (member, reference) {
        (Some(member), Some(reference)) => {
            let same_transform = match (member.transform, reference.transform) {
                (Some(left), Some(right)) => left == right,
                (None, None) => true,
                _ => false,
            };
            member.origin == reference.origin
                && same_transform
                && member.trainable == reference.trainable
        }
        _ => false,
    }
}

/// Finds the smallest source path that reaches two different target tie classes.
///
/// Entry `i` of a preserved source tie copies source path `paths[i]` (dense,
/// sorted path indices below `path_count`) into target tie class
/// `classes[i]`. Preserving the source tie is only coherent when each path
/// reaches one class; `None` means the path-to-class map is a function.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(found =>
    requires
        paths.len() == classes.len(),
    ensures
        match found {
            None => forall|path: int| 0 <= path < path_count ==> !fanned_out(paths@, classes@, path),
            Some(path) => {
                &&& path < path_count
                &&& fanned_out(paths@, classes@, path as int)
                &&& forall|smaller: int| 0 <= smaller < path ==> !fanned_out(paths@, classes@, smaller)
            },
        },
))]
pub(crate) fn first_fanned_out_path(
    paths: &[usize],
    classes: &[u64],
    path_count: usize,
) -> Option<usize> {
    let mut path = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= path <= path_count,
            paths.len() == classes.len(),
            forall|smaller: int| 0 <= smaller < path ==> !fanned_out(paths@, classes@, smaller),
        decreases path_count - path,
    ))]
    while path < path_count {
        let mut class: Option<u64> = None;
        let mut entry = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                0 <= entry <= paths.len(),
                paths.len() == classes.len(),
                path < path_count,
                forall|smaller: int| 0 <= smaller < path ==> !fanned_out(paths@, classes@, smaller),
                match class {
                    None => forall|i: int| 0 <= i < entry ==> paths@[i] != path,
                    Some(seen) => {
                        &&& exists|i: int| 0 <= i < entry && paths@[i] == path && classes@[i] == seen
                        &&& forall|i: int| 0 <= i < entry && paths@[i] == path ==> classes@[i] == seen
                    },
                },
            decreases paths.len() - entry,
        ))]
        while entry < paths.len() {
            if paths[entry] == path {
                match class {
                    None => class = Some(classes[entry]),
                    Some(seen) => {
                        if seen != classes[entry] {
                            #[cfg(verus_keep_ghost)]
                            proof! {
                                let witness = choose|i: int| 0 <= i < entry && paths@[i] == path
                                    && classes@[i] == seen;
                                assert(fanned_out(paths@, classes@, path as int)) by {
                                    assert(paths@[witness] == path && paths@[entry as int] == path
                                        && classes@[witness] != classes@[entry as int]);
                                }
                            }
                            return Some(path);
                        }
                    }
                }
            }
            entry += 1;
        }
        path += 1;
    }
    None
}

/// Assigns every entry the first entry of its target tie class.
///
/// Entry `i` of a fan-out group copies into the target tie class
/// `classes[i]`; members of one class must share one parameter identity.
/// `leaders[i]` is the first entry with the same class, so two entries
/// share a leader exactly when they share a class: classes are joined, and
/// distinct classes are never merged. A singleton class leads itself.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(leaders =>
    ensures
        leaders.len() == classes.len(),
        forall|i: int| 0 <= i < classes.len() ==> is_class_leader(classes@, i, leaders@[i] as int),
        forall|i: int, j: int| #![trigger leaders@[i], leaders@[j]]
            0 <= i < classes.len() && 0 <= j < classes.len()
            ==> ((classes@[i] == classes@[j]) <==> (leaders@[i] == leaders@[j])),
))]
pub(crate) fn class_leaders(classes: &[u64]) -> Vec<usize> {
    let mut leaders: Vec<usize> = Vec::new();
    let mut entry = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            0 <= entry <= classes.len(),
            leaders.len() == entry,
            forall|i: int| 0 <= i < entry ==> is_class_leader(classes@, i, leaders@[i] as int),
        decreases classes.len() - entry,
    ))]
    while entry < classes.len() {
        let mut leader = 0;
        #[cfg_attr(verus_keep_ghost, verus_spec(
            invariant
                0 <= leader <= entry,
                entry < classes.len(),
                forall|j: int| 0 <= j < leader ==> classes@[j] != classes@[entry as int],
            decreases entry - leader,
        ))]
        while leader < entry && classes[leader] != classes[entry] {
            leader += 1;
        }
        leaders.push(leader);
        entry += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        assert forall|i: int, j: int| #![trigger leaders@[i], leaders@[j]]
            0 <= i < classes.len() && 0 <= j < classes.len()
            implies ((classes@[i] == classes@[j]) <==> (leaders@[i] == leaders@[j])) by {
            assert(is_class_leader(classes@, i, leaders@[i] as int));
            assert(is_class_leader(classes@, j, leaders@[j] as int));
            if classes@[i] == classes@[j] && leaders@[i] < leaders@[j] {
                assert(classes@[leaders@[i] as int] == classes@[j]);
            }
            if classes@[i] == classes@[j] && leaders@[j] < leaders@[i] {
                assert(classes@[leaders@[j] as int] == classes@[i]);
            }
        }
    }
    leaders
}

#[cfg(verus_keep_ghost)]
verus! {

/// Whether `leader` is the first entry whose class equals entry `entry`'s class.
pub(crate) open spec fn is_class_leader(classes: Seq<u64>, entry: int, leader: int) -> bool {
    &&& 0 <= leader <= entry
    &&& classes[leader] == classes[entry]
    &&& forall|earlier: int| 0 <= earlier < leader ==> classes[earlier] != classes[entry]
}

/// Whether some rule names the target exactly.
pub(crate) open spec fn has_exact(matches: Seq<RuleMatch>) -> bool {
    exists|i: int| 0 <= i < matches.len() && matches[i] == RuleMatch::Exact
}

/// Whether rule `index` is the first exact rule.
pub(crate) open spec fn is_first_exact(matches: Seq<RuleMatch>, index: int) -> bool {
    &&& 0 <= index < matches.len()
    &&& matches[index] == RuleMatch::Exact
    &&& forall|earlier: int| 0 <= earlier < index ==> matches[earlier] != RuleMatch::Exact
}

/// Number of fallback rules among the first `count` rules.
pub(crate) open spec fn count_fallbacks(matches: Seq<RuleMatch>, count: int) -> int
    decreases count,
{
    if count <= 0 {
        0
    } else {
        count_fallbacks(matches, count - 1) + if matches[count - 1] == RuleMatch::Fallback {
            1int
        } else {
            0int
        }
    }
}

/// Counting more rules never decreases the fallback count.
pub(crate) proof fn lemma_count_fallbacks_monotone(matches: Seq<RuleMatch>, shorter: int, longer: int)
    requires
        0 <= shorter <= longer,
    ensures
        count_fallbacks(matches, shorter) <= count_fallbacks(matches, longer),
    decreases longer - shorter,
{
    if shorter < longer {
        lemma_count_fallbacks_monotone(matches, shorter, longer - 1);
    }
}

/// A fallback at `index` adds one to every count that includes it.
pub(crate) proof fn lemma_count_fallbacks_includes(matches: Seq<RuleMatch>, index: int, count: int)
    requires
        0 <= index < count <= matches.len(),
        matches[index] == RuleMatch::Fallback,
    ensures
        count_fallbacks(matches, count) >= count_fallbacks(matches, index) + 1,
{
    lemma_count_fallbacks_monotone(matches, index + 1, count);
}

/// Whether source path `path` is copied into two different target tie classes.
pub(crate) open spec fn fanned_out(paths: Seq<usize>, classes: Seq<u64>, path: int) -> bool {
    exists|first: int, second: int| {
        &&& 0 <= first < paths.len()
        &&& 0 <= second < paths.len()
        &&& paths[first] == path
        &&& paths[second] == path
        &&& classes[first] != classes[second]
    }
}

} // verus!

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
