//! Oracle and property checks of alias-chain resolution.

use std::collections::BTreeSet;

use proptest::prelude::*;

use super::*;

/// The original visited-set loop over the same links, as an independent oracle.
fn reference_walk(links: &[AliasLink], start: usize) -> AliasWalk {
    let mut cursor = start;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(cursor) {
            return AliasWalk::Cycle(cursor);
        }
        match links[cursor] {
            AliasLink::Owner => return AliasWalk::Owner(cursor),
            AliasLink::Dangling => return AliasWalk::Dangling(cursor),
            AliasLink::Next(next) => cursor = next,
        }
    }
}

/// Walks once with a fresh buffer and checks that the walk leaves it clear.
fn walk(links: &[AliasLink], start: usize) -> AliasWalk {
    let mut visited = vec![false; links.len()];
    let outcome = walk_alias_chain(links, start, &mut visited);
    assert!(
        visited.iter().all(|&marked| !marked),
        "walk left marks behind"
    );
    outcome
}

/// Chains reach owners, dangling sources, or the first repeated alias.
#[test]
fn chains_end_at_the_first_event() {
    use AliasLink::{Dangling, Next, Owner};
    let links = [Next(1), Next(2), Owner, Next(3), Next(5), Next(4), Dangling];
    assert_eq!(walk(&links, 0), AliasWalk::Owner(2));
    assert_eq!(walk(&links, 2), AliasWalk::Owner(2));
    assert_eq!(walk(&links, 3), AliasWalk::Cycle(3));
    assert_eq!(walk(&links, 4), AliasWalk::Cycle(4));
    assert_eq!(walk(&links, 6), AliasWalk::Dangling(6));
}

/// Builds closed links from arbitrary seeds: `Next` targets wrap into range.
fn closed_links(seeds: &[(u8, usize)]) -> Vec<AliasLink> {
    seeds
        .iter()
        .map(|&(kind, target)| match kind % 4 {
            0 => AliasLink::Owner,
            1 => AliasLink::Dangling,
            _ => AliasLink::Next(target % seeds.len()),
        })
        .collect()
}

proptest! {
    /// Every walk matches the original loop on random closed link tables.
    #[test]
    fn walks_match_the_visited_set_loop(
        seeds in proptest::collection::vec((any::<u8>(), any::<usize>()), 1..16),
    ) {
        let links = closed_links(&seeds);
        // One shared buffer across every start, as the planner uses it.
        let mut visited = vec![false; links.len()];
        for start in 0..links.len() {
            prop_assert_eq!(
                walk_alias_chain(&links, start, &mut visited),
                reference_walk(&links, start)
            );
            prop_assert!(visited.iter().all(|&marked| !marked));
        }
    }
}
