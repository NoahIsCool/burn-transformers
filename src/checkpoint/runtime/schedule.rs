//! Grouping of primary load-plan work into indivisible execution groups.
//!
//! Targets that slice one fused source share a *class*; every other target has
//! none. The executor reads a shared source once and materializes all of its
//! slices together, so the targets of one class form a single group, while
//! every other target is a group of its own. [`execution_group_indices`]
//! emits each group, in plan-index order, at its *leader*: the first target of
//! its class, or the target itself when it has no class.
//!
//! The groups partition the plan: every target lies in exactly one group, each
//! group lists its targets in increasing order, and the targets of one group
//! share one class. Consequently the executor installs every primary target
//! exactly once.
//!
//! The module has no crate or Burn dependencies so that Verus can check it in
//! place through `verus/checkpoint.rs`; the caller assigns the classes and
//! orders the groups: outsized groups first in plan order, then the rest by
//! physical source location.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

#[cfg(verus_keep_ghost)]
verus! {

/// Specification: target `i` leads its group, being classless or the first of its class.
pub open spec fn is_leader(classes: Seq<Option<usize>>, i: int) -> bool {
    match classes[i] {
        None => true,
        Some(class) => forall|j: int| 0 <= j < i ==> classes[j] != Some(class),
    }
}

/// Specification: the leaders among targets `0..upto`, in increasing order.
pub open spec fn leaders(classes: Seq<Option<usize>>, upto: int) -> Seq<usize>
    decreases upto,
{
    if upto <= 0 {
        Seq::empty()
    } else if is_leader(classes, upto - 1) {
        leaders(classes, upto - 1).push((upto - 1) as usize)
    } else {
        leaders(classes, upto - 1)
    }
}

/// Specification: the targets in `from..upto` whose class is `class`, in increasing order.
pub open spec fn members(classes: Seq<Option<usize>>, class: usize, from: int, upto: int) -> Seq<usize>
    decreases upto - from,
{
    if upto <= from {
        Seq::empty()
    } else if classes[upto - 1] == Some(class) {
        members(classes, class, from, upto - 1).push((upto - 1) as usize)
    } else {
        members(classes, class, from, upto - 1)
    }
}

/// Specification: the group led by `leader`: itself when classless, else its class from `leader` on.
pub open spec fn group_of(classes: Seq<Option<usize>>, leader: int) -> Seq<usize> {
    match classes[leader] {
        None => seq![leader as usize],
        Some(class) => members(classes, class, leader, classes.len() as int),
    }
}

/// Specification: `values` strictly increases.
pub open spec fn increasing(values: Seq<usize>) -> bool {
    forall|k: int, l: int| 0 <= k < l < values.len() ==> #[trigger] values[k] < #[trigger] values[l]
}

/// Specification: target `x` lies in some group.
pub open spec fn in_some_group(groups: Seq<Vec<usize>>, x: int) -> bool {
    exists|g: int| 0 <= g < groups.len() && #[trigger] groups[g]@.contains(x as usize)
}

/// Specification: the groups led by the first `count` leaders.
pub open spec fn groups_match(
    classes: Seq<Option<usize>>,
    groups: Seq<Vec<usize>>,
    lead: Seq<usize>,
) -> bool {
    &&& groups.len() == lead.len()
    &&& forall|g: int| 0 <= g < groups.len()
        ==> #[trigger] groups[g]@ == group_of(classes, lead[g] as int)
}

/// Membership in `members` is exactly the range and class test.
proof fn lemma_members_contains(
    classes: Seq<Option<usize>>,
    class: usize,
    from: int,
    upto: int,
    x: int,
)
    requires
        0 <= from,
        upto <= classes.len() <= usize::MAX,
        0 <= x <= usize::MAX,
    ensures
        members(classes, class, from, upto).contains(x as usize)
            <==> (from <= x < upto && classes[x] == Some(class)),
    decreases upto - from,
{
    if upto <= from {
        assert(!members(classes, class, from, upto).contains(x as usize));
    } else {
        lemma_members_contains(classes, class, from, upto - 1, x);
        let previous = members(classes, class, from, upto - 1);
        if classes[upto - 1] == Some(class) {
            let current = previous.push((upto - 1) as usize);
            assert(current.contains(x as usize) <==> (previous.contains(x as usize)
                || (upto - 1) as usize == x as usize)) by {
                if current.contains(x as usize) {
                    let k = choose|k: int| 0 <= k < current.len() && current[k] == x as usize;
                    if k < previous.len() {
                        assert(previous[k] == x as usize);
                    }
                }
                if previous.contains(x as usize) {
                    let k = choose|k: int| 0 <= k < previous.len() && previous[k] == x as usize;
                    assert(current[k] == x as usize);
                }
                if (upto - 1) as usize == x as usize {
                    assert(current[previous.len() as int] == x as usize);
                }
            }
        }
    }
}

/// Elements of `members` lie in its range, have its class, and strictly increase.
proof fn lemma_members_shape(
    classes: Seq<Option<usize>>,
    class: usize,
    from: int,
    upto: int,
)
    requires
        0 <= from,
        upto <= classes.len(),
        classes.len() <= usize::MAX,
    ensures
        forall|k: int| 0 <= k < members(classes, class, from, upto).len() ==> {
            &&& from <= #[trigger] members(classes, class, from, upto)[k] < upto
            &&& classes[members(classes, class, from, upto)[k] as int] == Some(class)
        },
        increasing(members(classes, class, from, upto)),
    decreases upto - from,
{
    if upto > from {
        lemma_members_shape(classes, class, from, upto - 1);
    }
}

/// Membership in `leaders` is exactly the range and leader test, and leaders strictly increase.
proof fn lemma_leaders_shape(classes: Seq<Option<usize>>, upto: int)
    requires
        0 <= upto <= classes.len(),
        classes.len() <= usize::MAX,
    ensures
        forall|k: int| 0 <= k < leaders(classes, upto).len() ==> {
            &&& 0 <= #[trigger] leaders(classes, upto)[k] < upto
            &&& is_leader(classes, leaders(classes, upto)[k] as int)
        },
        increasing(leaders(classes, upto)),
        forall|x: int| 0 <= x < upto && is_leader(classes, x)
            ==> #[trigger] leaders(classes, upto).contains(x as usize),
    decreases upto,
{
    if upto > 0 {
        lemma_leaders_shape(classes, upto - 1);
        let previous = leaders(classes, upto - 1);
        if is_leader(classes, upto - 1) {
            let current = previous.push((upto - 1) as usize);
            assert forall|x: int| 0 <= x < upto && is_leader(classes, x)
                implies #[trigger] current.contains(x as usize)
            by {
                if x < upto - 1 {
                    assert(leaders(classes, upto - 1).contains(x as usize));
                    let k = choose|k: int| 0 <= k < previous.len() && previous[k] == x as usize;
                    assert(current[k] == x as usize);
                } else {
                    assert(current[previous.len() as int] == x as usize);
                }
            }
        }
    }
}

/// Some target of class `class` at or before `x` is the first of its class.
proof fn lemma_first_member(classes: Seq<Option<usize>>, class: usize, x: int) -> (first: int)
    requires
        0 <= x < classes.len(),
        classes[x] == Some(class),
    ensures
        0 <= first <= x,
        classes[first] == Some(class),
        is_leader(classes, first),
    decreases x,
{
    if exists|j: int| 0 <= j < x && classes[j] == Some(class) {
        let j = choose|j: int| 0 <= j < x && classes[j] == Some(class);
        lemma_first_member(classes, class, j)
    } else {
        x
    }
}

/// Groups matching the leaders of every target partition the targets by class.
proof fn lemma_partition(classes: Seq<Option<usize>>, groups: Seq<Vec<usize>>)
    requires
        classes.len() <= usize::MAX,
        groups_match(classes, groups, leaders(classes, classes.len() as int)),
    ensures
        forall|x: int| 0 <= x < classes.len() ==> #[trigger] in_some_group(groups, x),
        forall|g: int, h: int, x: usize| 0 <= g < groups.len() && 0 <= h < groups.len()
            && #[trigger] groups[g]@.contains(x) && #[trigger] groups[h]@.contains(x)
            ==> g == h,
        forall|g: int| 0 <= g < groups.len() ==> #[trigger] groups[g]@.len() >= 1,
        forall|g: int| 0 <= g < groups.len() ==> increasing(#[trigger] groups[g]@),
        forall|g: int, k: int, l: int| 0 <= g < groups.len() && 0 <= k < groups[g]@.len()
            && 0 <= l < groups[g]@.len()
            ==> classes[#[trigger] groups[g]@[k] as int] == classes[#[trigger] groups[g]@[l] as int]
                && (k != l ==> classes[groups[g]@[k] as int] is Some),
{
    let n = classes.len() as int;
    let lead = leaders(classes, n);
    lemma_leaders_shape(classes, n);
    // Facts about one group, by its leader's class.
    assert forall|g: int| 0 <= g < groups.len() implies {
        &&& #[trigger] groups[g]@.len() >= 1
        &&& increasing(groups[g]@)
        &&& forall|k: int| 0 <= k < groups[g]@.len() ==> {
            &&& 0 <= #[trigger] groups[g]@[k] < n
            &&& classes[groups[g]@[k] as int] == classes[lead[g] as int]
            &&& (groups[g]@.len() > 1 ==> classes[lead[g] as int] is Some)
        }
        &&& (classes[lead[g] as int] is None ==> groups[g]@ == seq![lead[g]])
    } by {
        let leader = lead[g] as int;
        assert(0 <= leader < n);
        match classes[leader] {
            None => {
                assert(groups[g]@ == seq![lead[g]]);
            },
            Some(class) => {
                lemma_members_shape(classes, class, leader, n);
                lemma_members_contains(classes, class, leader, n, leader);
                assert(groups[g]@.contains(leader as usize));
            },
        }
    }
    // Every target lies in the group of the first target of its class.
    assert forall|x: int| 0 <= x < n implies #[trigger] in_some_group(groups, x) by {
        let leader = match classes[x] {
            None => x,
            Some(class) => lemma_first_member(classes, class, x),
        };
        assert(is_leader(classes, leader));
        assert(lead.contains(leader as usize));
        let g = choose|g: int| 0 <= g < lead.len() && lead[g] == leader as usize;
        match classes[x] {
            None => {
                assert(groups[g]@ == seq![leader as usize]);
                assert(groups[g]@[0] == x as usize);
            },
            Some(class) => {
                lemma_members_contains(classes, class, leader, n, x);
            },
        }
        assert(groups[g]@.contains(x as usize));
    }
    // Two groups sharing a target share its first-of-class leader.
    assert forall|g: int, h: int, x: usize| 0 <= g < groups.len() && 0 <= h < groups.len()
        && #[trigger] groups[g]@.contains(x) && #[trigger] groups[h]@.contains(x)
        implies g == h
    by {
        let kg = choose|k: int| 0 <= k < groups[g]@.len() && groups[g]@[k] == x;
        let kh = choose|k: int| 0 <= k < groups[h]@.len() && groups[h]@[k] == x;
        let lg = lead[g] as int;
        let lh = lead[h] as int;
        assert(classes[x as int] == classes[lg] && classes[x as int] == classes[lh]);
        match classes[x as int] {
            None => {
                assert(groups[g]@ == seq![lead[g]]);
                assert(groups[h]@ == seq![lead[h]]);
            },
            Some(class) => {
                lemma_members_contains(classes, class, lg, n, x as int);
                lemma_members_contains(classes, class, lh, n, x as int);
                // Both leaders are the first of `class`, so neither precedes the other.
                if lg < lh {
                    assert(!is_leader(classes, lh));
                } else if lh < lg {
                    assert(!is_leader(classes, lg));
                }
            },
        }
        assert(lead[g] == lead[h]);
        if g < h {
            assert(lead[g] < lead[h]);
        } else if h < g {
            assert(lead[h] < lead[g]);
        }
    }
}

} // verus!

/// Groups plan indices `0..classes.len()` by shared-source class.
///
/// `classes[i]` is the class of target `i`'s fused source, or `None` when the
/// target shares no source. Each group is emitted at its leader (the first
/// target of its class, or a classless target itself) and lists the class's
/// targets in increasing order.
///
/// Verified: the result is exactly the groups of the leaders in index order,
/// and they partition the targets: every target lies in exactly one group,
/// each group is nonempty and strictly increasing, and all targets of a group
/// share one class (a classless target is alone).
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(groups =>
    requires
        forall|i: int| 0 <= i < classes@.len() ==> match #[trigger] classes@[i] {
            Some(class) => class < class_count,
            None => true,
        },
    ensures
        groups_match(classes@, groups@, leaders(classes@, classes@.len() as int)),
        forall|x: int| 0 <= x < classes@.len() ==> #[trigger] in_some_group(groups@, x),
        forall|g: int, h: int, x: usize| 0 <= g < groups@.len() && 0 <= h < groups@.len()
            && #[trigger] groups@[g]@.contains(x) && #[trigger] groups@[h]@.contains(x)
            ==> g == h,
        forall|g: int| 0 <= g < groups@.len() ==> #[trigger] groups@[g]@.len() >= 1,
        forall|g: int| 0 <= g < groups@.len() ==> increasing(#[trigger] groups@[g]@),
        forall|g: int, k: int, l: int| 0 <= g < groups@.len() && 0 <= k < groups@[g]@.len()
            && 0 <= l < groups@[g]@.len()
            ==> classes@[#[trigger] groups@[g]@[k] as int] == classes@[#[trigger] groups@[g]@[l] as int]
                && (k != l ==> classes@[groups@[g]@[k] as int] is Some),
))]
pub(crate) fn execution_group_indices(
    classes: &[Option<usize>],
    class_count: usize,
) -> Vec<Vec<usize>> {
    let count = classes.len();
    let mut emitted: Vec<bool> = Vec::new();
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            emitted@.len() <= class_count,
            forall|c: int| 0 <= c < emitted@.len() ==> !#[trigger] emitted@[c],
        decreases class_count - emitted@.len(),
    ))]
    while emitted.len() < class_count {
        emitted.push(false);
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index = 0;
    #[cfg_attr(verus_keep_ghost, verus_spec(
        invariant
            index <= count,
            count == classes@.len(),
            emitted@.len() == class_count,
            forall|i: int| 0 <= i < classes@.len() ==> match #[trigger] classes@[i] {
                Some(class) => class < class_count,
                None => true,
            },
            forall|c: int| 0 <= c < class_count ==> (#[trigger] emitted@[c] <==> exists|j: int|
                0 <= j < index && classes@[j] == Some(c as usize)),
            groups_match(classes@, groups@, leaders(classes@, index as int)),
        decreases count - index,
    ))]
    while index < count {
        match classes[index] {
            None => {
                let group = vec![index];
                #[cfg(verus_keep_ghost)]
                proof! {
                    assert(group@ == seq![index]);
                }
                groups.push(group);
            }
            Some(class) => {
                if !emitted[class] {
                    emitted[class] = true;
                    let mut group = Vec::new();
                    let mut member = index;
                    #[cfg_attr(verus_keep_ghost, verus_spec(
                        invariant
                            index <= member <= count,
                            count == classes@.len(),
                            group@ == members(classes@, class, index as int, member as int),
                        decreases count - member,
                    ))]
                    while member < count {
                        if classes[member] == Some(class) {
                            group.push(member);
                        }
                        member += 1;
                    }
                    groups.push(group);
                }
            }
        }
        index += 1;
    }
    #[cfg(verus_keep_ghost)]
    proof! {
        lemma_partition(classes@, groups@);
    }
    groups
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
