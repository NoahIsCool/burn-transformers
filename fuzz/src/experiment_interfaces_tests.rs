//! Deterministic seed replays for the public-interface fuzz oracles.

use super::exercise;

/// Replays every committed oracle branch before launching a sanitizer campaign.
#[test]
fn covers_interface_boundaries_without_a_fuzzer() {
    for seed in [
        include_bytes!("../seeds/experiment_interfaces/valid-selection").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/permuted-selection").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/overflow-selection").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/duplicate-axis").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/fully-masked").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/invalid-broadcast").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/preserved-ties").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/untied-parameters").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/failed-transform-shape").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/failed-transform-callback").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/invalid-transfer-path").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/finite-loss").as_slice(),
        include_bytes!("../seeds/experiment_interfaces/invalid-loss-axis-label").as_slice(),
    ] {
        exercise(seed);
    }
    exercise(&[]);
}
