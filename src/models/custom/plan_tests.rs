//! Displacement-table geometry and history-append planning: examples, edges, and properties.

use super::*;

/// The table radius is `min(max_distance, C - 1)` and its length `2R + 1`.
#[test]
fn displacement_table_clamps_to_the_shorter_of_distance_and_context() {
    assert_eq!(displacement_table(128, 0), None);
    assert_eq!(
        displacement_table(128, 1),
        Some(DisplacementTable {
            radius: 0,
            length: 1
        })
    );
    assert_eq!(
        displacement_table(128, 10),
        Some(DisplacementTable {
            radius: 9,
            length: 19
        })
    );
    assert_eq!(
        displacement_table(8, 1000),
        Some(DisplacementTable {
            radius: 8,
            length: 17
        })
    );
    // `2R + 1` must fit `usize`: the largest representable radius is `usize::MAX / 2`.
    assert_eq!(displacement_table(usize::MAX, usize::MAX), None);
    assert_eq!(displacement_table(usize::MAX / 2 + 1, usize::MAX), None);
    assert_eq!(
        displacement_table(usize::MAX / 2, usize::MAX),
        Some(DisplacementTable {
            radius: usize::MAX / 2,
            length: usize::MAX
        })
    );
}

/// Slots clamp out-of-range displacements and invert the stored displacement exactly.
#[test]
fn displacement_slots_clamp_and_invert() {
    for radius in [0usize, 1, 3, 17] {
        for slot in 0..=2 * radius {
            let displacement = slot_displacement(slot, radius);
            assert_eq!(displacement, slot as i64 - radius as i64);
            assert_eq!(displacement_slot(displacement, radius), slot);
        }
        assert_eq!(displacement_slot(i64::MIN, radius), 0);
        assert_eq!(displacement_slot(-(radius as i64) - 5, radius), 0);
        assert_eq!(displacement_slot(radius as i64 + 5, radius), 2 * radius);
        assert_eq!(displacement_slot(i64::MAX, radius), 2 * radius);
    }
}
