//! Range-hazard declarations, their alternatives, the refusal policy, and the plan check.

use std::{fs::File, io::Write, path::Path};

use proptest::prelude::*;

use super::*;
use crate::checkpoint::{
    DTypePolicy, ExpectedTensor, LoadOptions, MappingPlan, PlannedTransform, build_load_plan,
    discover_checkpoint,
};

/// A declaration of the shape a family publishes, used throughout.
const F16_OVERFLOW: RangeHazard = RangeHazard::new(
    "Example (model_type example)",
    TensorDType::F16,
    "measured: block 3's residual reaches 9.1e4 in the reference's F16 graph",
);

/// Every castable float, the dtypes a plan can execute.
const CASTABLE: [TensorDType; 4] = [
    TensorDType::F16,
    TensorDType::BF16,
    TensorDType::F32,
    TensorDType::F64,
];

/// Every SafeTensors dtype, castable or not.
const ALL: [TensorDType; 20] = [
    TensorDType::BOOL,
    TensorDType::F4,
    TensorDType::F6_E2M3,
    TensorDType::F6_E3M2,
    TensorDType::U8,
    TensorDType::I8,
    TensorDType::F8_E5M2,
    TensorDType::F8_E4M3,
    TensorDType::F8_E8M0,
    TensorDType::I16,
    TensorDType::U16,
    TensorDType::F16,
    TensorDType::BF16,
    TensorDType::I32,
    TensorDType::U32,
    TensorDType::F32,
    TensorDType::C64,
    TensorDType::F64,
    TensorDType::I64,
    TensorDType::U64,
];

/// Builds a plan whose storage-owning targets have the given dtypes, plus one alias.
///
/// Targets are named `t0`, `t1`, ... in order, so the plan is in canonical
/// target order. The alias takes the first target's dtype, as a planner
/// records it, and must not change any count.
fn plan_with(dtypes: &[TensorDType]) -> LoadPlan {
    let tensor = |target: String, dtype: TensorDType, transform: PlannedTransform| {
        crate::checkpoint::PlannedTensor {
            target,
            sources: Vec::new(),
            transform,
            shape: vec![1],
            dtype,
        }
    };
    LoadPlan {
        equivalences: Vec::new(),
        tensors: dtypes
            .iter()
            .enumerate()
            .map(|(index, dtype)| tensor(format!("t{index}"), *dtype, PlannedTransform::Identity))
            .collect(),
        aliases: dtypes
            .first()
            .map(|dtype| {
                tensor(
                    "head".into(),
                    *dtype,
                    PlannedTransform::Alias {
                        source_target: "t0".into(),
                    },
                )
            })
            .into_iter()
            .collect(),
        report: Default::default(),
        memory: Default::default(),
    }
}

/// Writes a one-file checkpoint of one-element tensors `(name, dtype)` and discovers it.
fn manifest_with(
    root: &Path,
    tensors: &[(&str, TensorDType)],
) -> crate::checkpoint::CheckpointManifest {
    let mut offset = 0;
    let mut entries = Vec::new();
    for (name, dtype) in tensors {
        let bytes = dtype.storage_bytes(1).unwrap();
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[1],"data_offsets":[{offset},{}]}}"#,
            offset + bytes
        ));
        offset += bytes;
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&vec![0; usize::try_from(offset).unwrap()])
        .unwrap();
    discover_checkpoint(root, &LoadOptions::default()).unwrap()
}

/// The declaration keeps its fields, is usable in constants, and matches exactly its dtype.
#[test]
fn a_declaration_exposes_its_fields_and_matches_only_its_dtype() {
    assert_eq!(F16_OVERFLOW.checkpoints(), "Example (model_type example)");
    assert_eq!(F16_OVERFLOW.dtype(), TensorDType::F16);
    assert!(F16_OVERFLOW.evidence().starts_with("measured: block 3"));
    // A new declaration certifies no packed path; the builder sets only that flag.
    assert!(!F16_OVERFLOW.packed_path_certified());
    /// The same declaration with a certified packed path, built in a constant because the builder is a `const fn`.
    const CERTIFIED: RangeHazard = F16_OVERFLOW.with_certified_packed_path();
    assert!(CERTIFIED.packed_path_certified());
    assert_ne!(CERTIFIED, F16_OVERFLOW);
    assert_eq!(
        (
            CERTIFIED.checkpoints(),
            CERTIFIED.dtype(),
            CERTIFIED.evidence()
        ),
        (
            F16_OVERFLOW.checkpoints(),
            F16_OVERFLOW.dtype(),
            F16_OVERFLOW.evidence()
        )
    );
    for dtype in ALL {
        assert_eq!(
            F16_OVERFLOW.applies_to(dtype),
            dtype == TensorDType::F16,
            "{dtype}"
        );
    }
    // `applies_to` is a `const fn`, so a family can check its declarations at compile time.
    const {
        assert!(F16_OVERFLOW.applies_to(TensorDType::F16));
        assert!(!F16_OVERFLOW.applies_to(TensorDType::F32));
    };
    let copy = F16_OVERFLOW;
    assert_eq!(copy, F16_OVERFLOW);
    assert_ne!(
        RangeHazard::new(
            F16_OVERFLOW.checkpoints(),
            TensorDType::BF16,
            F16_OVERFLOW.evidence()
        ),
        F16_OVERFLOW
    );
}

/// Each castable float's largest finite value is `(2 - 2^-m) · 2^(2^(e-1) - 1)`; other dtypes have none.
#[test]
fn largest_finite_values_follow_the_format_formula() {
    let formula = |exponent_bits: i32, mantissa_bits: i32| {
        (2.0 - 2f64.powi(-mantissa_bits)) * 2f64.powi(2i32.pow(exponent_bits as u32 - 1) - 1)
    };
    assert_eq!(largest_finite(TensorDType::F16), Some(formula(5, 10)));
    assert_eq!(largest_finite(TensorDType::F16), Some(65_504.0));
    assert_eq!(largest_finite(TensorDType::BF16), Some(formula(8, 7)));
    assert_eq!(largest_finite(TensorDType::F32), Some(formula(8, 23)));
    assert_eq!(largest_finite(TensorDType::F64), Some(f64::MAX));
    for dtype in ALL {
        assert_eq!(
            largest_finite(dtype).is_some(),
            dtype.is_castable_float(),
            "{dtype}"
        );
    }
}

/// Alternatives are the wider castable dtypes, F32 first, with F64 only for an F32 hazard.
#[test]
fn alternatives_are_the_wider_castable_dtypes() {
    let alternatives = |dtype| RangeHazard::new("checkpoints", dtype, "evidence").alternatives();
    assert_eq!(
        alternatives(TensorDType::F16),
        [TensorDType::F32, TensorDType::BF16]
    );
    assert_eq!(alternatives(TensorDType::BF16), [TensorDType::F32]);
    assert_eq!(alternatives(TensorDType::F32), [TensorDType::F64]);
    assert!(alternatives(TensorDType::F64).is_empty());
    for dtype in ALL.into_iter().filter(|dtype| !dtype.is_castable_float()) {
        assert!(alternatives(dtype).is_empty(), "{dtype}");
    }
    for dtype in CASTABLE {
        let limit = largest_finite(dtype).unwrap();
        for alternative in alternatives(dtype) {
            assert!(
                largest_finite(alternative).unwrap() > limit,
                "{dtype} -> {alternative}"
            );
        }
    }
}

/// The policy defaults to refusing and serializes by variant name; unknown names are rejected.
#[test]
fn the_policy_defaults_to_refuse_and_round_trips() {
    assert_eq!(RangeHazardPolicy::default(), RangeHazardPolicy::Refuse);
    for (policy, json) in [
        (RangeHazardPolicy::Refuse, r#""Refuse""#),
        (RangeHazardPolicy::Accept, r#""Accept""#),
    ] {
        assert_eq!(serde_json::to_string(&policy).unwrap(), json);
        assert_eq!(
            serde_json::from_str::<RangeHazardPolicy>(json).unwrap(),
            policy
        );
    }
    for invalid in [r#""accept""#, r#""Clamp""#, "true", "null"] {
        assert!(
            serde_json::from_str::<RangeHazardPolicy>(invalid).is_err(),
            "{invalid}"
        );
    }
}

/// Serialized options written before the field existed take the refusing default.
#[test]
fn load_options_without_the_policy_deserialize_to_refuse() {
    let mut json = serde_json::to_value(LoadOptions::default()).unwrap();
    assert_eq!(json["range_hazards"], "Refuse");
    json.as_object_mut().unwrap().remove("range_hazards");
    let options: LoadOptions = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(options.range_hazards, RangeHazardPolicy::Refuse);
    assert_eq!(options, LoadOptions::default());
    json["range_hazards"] = "Accept".into();
    let options: LoadOptions = serde_json::from_value(json).unwrap();
    assert_eq!(options.range_hazards, RangeHazardPolicy::Accept);
}

/// A matching plan is refused with its first parameter and count; aliases add nothing.
#[test]
fn a_matching_plan_is_refused_with_its_first_parameter_and_count() {
    let plan = plan_with(&[TensorDType::F32, TensorDType::F16, TensorDType::F16]);
    let error = check_range_hazards(&plan, &[F16_OVERFLOW], RangeHazardPolicy::Refuse).unwrap_err();
    let CheckpointError::RangeHazard {
        hazard,
        target,
        parameters,
    } = &error
    else {
        panic!("expected a range-hazard refusal, got {error:?}");
    };
    assert_eq!(*hazard, F16_OVERFLOW);
    assert_eq!(target, "t1");
    assert_eq!(*parameters, 2);
    // The alias of `t0` would also count if aliases were inspected.
    let plan = plan_with(&[TensorDType::F16, TensorDType::F32]);
    let Err(CheckpointError::RangeHazard {
        target, parameters, ..
    }) = check_range_hazards(&plan, &[F16_OVERFLOW], RangeHazardPolicy::Refuse)
    else {
        panic!("an F16 parameter must be refused");
    };
    assert_eq!((target.as_str(), parameters), ("t0", 1));
}

/// Accepting, an empty declaration list, an empty plan, and other dtypes all pass.
#[test]
fn accepted_undeclared_and_unmatched_plans_pass() {
    let f16 = plan_with(&[TensorDType::F16]);
    check_range_hazards(&f16, &[F16_OVERFLOW], RangeHazardPolicy::Accept).unwrap();
    check_range_hazards(&f16, &[], RangeHazardPolicy::Refuse).unwrap();
    check_range_hazards(&plan_with(&[]), &[F16_OVERFLOW], RangeHazardPolicy::Refuse).unwrap();
    for dtype in ALL.into_iter().filter(|dtype| *dtype != TensorDType::F16) {
        check_range_hazards(
            &plan_with(&[dtype, dtype]),
            &[F16_OVERFLOW],
            RangeHazardPolicy::Refuse,
        )
        .unwrap_or_else(|error| panic!("{dtype}: {error}"));
    }
}

/// With several declarations, the first one in declaration order that matches is reported.
#[test]
fn the_first_matching_declaration_is_reported() {
    let bf16 = RangeHazard::new("Other", TensorDType::BF16, "other evidence");
    let plan = plan_with(&[TensorDType::BF16, TensorDType::F16]);
    for (hazards, expected) in [
        ([F16_OVERFLOW, bf16], F16_OVERFLOW),
        ([bf16, F16_OVERFLOW], bf16),
    ] {
        let Err(CheckpointError::RangeHazard { hazard, .. }) =
            check_range_hazards(&plan, &hazards, RangeHazardPolicy::Refuse)
        else {
            panic!("both declarations match");
        };
        assert_eq!(hazard, expected);
    }
}

/// The message names checkpoints, dtype, count, first parameter, evidence, alternatives, and opt-in.
#[test]
fn the_refusal_message_names_everything_a_caller_needs() {
    let message = check_range_hazards(
        &plan_with(&[TensorDType::F16, TensorDType::F16, TensorDType::F16]),
        &[F16_OVERFLOW],
        RangeHazardPolicy::Refuse,
    )
    .unwrap_err()
    .to_string();
    assert_eq!(
        message,
        "Example (model_type example) checkpoints are known to overflow F16, so loading 3 \
         planned parameters in F16 (first `t0`) is refused: activations take their parameters' \
         dtype, and values beyond F16's largest finite magnitude become infinite and then NaN. \
         Evidence: measured: block 3's residual reaches 9.1e4 in the reference's F16 graph. \
         Load the checkpoint in F32 (`DTypePolicy::Cast(TensorDType::F32)`) or BF16 \
         (`DTypePolicy::Cast(TensorDType::BF16)`) on a backend that can store it (Burn 0.21's \
         Metal backend cannot) instead. Packed Q4/Q8 weights with F32 activations are not \
         offered: the family has not certified a packed path for these checkpoints, and the \
         automatic loader refuses a quantized Burnpack sidecar for them. To evaluate the \
         faithful F16 graph anyway, for example to study the overflow, set \
         `LoadOptions::range_hazards` to `RangeHazardPolicy::Accept`; its logits are expected \
         to be non-finite."
    );
    let single = refusal_message(&F16_OVERFLOW, "only", 1);
    assert!(
        single.contains("loading 1 planned parameter in F16 (first `only`)"),
        "{single}"
    );
}

/// The alternatives sentence covers one, several, and no wider dtypes, with and without a certified packed path.
///
/// Packed weights keep F32 activations, so they are mentioned exactly when
/// F32 is offered: as an alternative when the family certifies its packed
/// path, and otherwise as not offered, naming the automatic loader's refusal.
#[test]
fn the_alternatives_sentence_covers_every_count() {
    let text = |dtype| alternatives_text(&RangeHazard::new("c", dtype, "e"));
    let certified =
        |dtype| alternatives_text(&RangeHazard::new("c", dtype, "e").with_certified_packed_path());
    let not_offered = "Packed Q4/Q8 weights with F32 activations are not offered: the family has \
                       not certified a packed path for these checkpoints, and the automatic \
                       loader refuses a quantized Burnpack sidecar for them.";
    let offered = "or with packed Q4/Q8 weights with F32 activations (a quantized Burnpack \
                   sidecar over an F32 plan, whose packed path the family certifies).";
    let f64_only = "Load the checkpoint in F64 (`DTypePolicy::Cast(TensorDType::F64)`) instead.";
    assert_eq!(text(TensorDType::F32), f64_only);
    assert_eq!(certified(TensorDType::F32), f64_only);
    assert_eq!(
        text(TensorDType::BF16),
        format!(
            "Load the checkpoint in F32 (`DTypePolicy::Cast(TensorDType::F32)`) instead. \
             {not_offered}"
        )
    );
    assert_eq!(
        certified(TensorDType::BF16),
        format!(
            "Load the checkpoint in F32 (`DTypePolicy::Cast(TensorDType::F32)`) instead, {offered}"
        )
    );
    let none = "No dtype the loader can cast to is wider than F64.";
    assert_eq!(text(TensorDType::F64), none);
    assert_eq!(certified(TensorDType::F64), none);
    let both = "Load the checkpoint in F32 (`DTypePolicy::Cast(TensorDType::F32)`) or BF16 \
                (`DTypePolicy::Cast(TensorDType::BF16)`) on a backend that can store it (Burn \
                0.21's Metal backend cannot) instead";
    assert_eq!(text(TensorDType::F16), format!("{both}. {not_offered}"));
    assert_eq!(certified(TensorDType::F16), format!("{both}, {offered}"));
    for dtype in ALL.into_iter().filter(|dtype| !dtype.is_castable_float()) {
        let none = format!("No dtype the loader can cast to is wider than {dtype}.");
        assert_eq!(text(dtype), none);
        assert_eq!(certified(dtype), none);
    }
}

/// Real plans: `Cast(F16)` and F16 storage under `Auto` are refused; F32, BF16, and accepting pass.
///
/// The check reads the dtype the planner chose, so a cast and a stored
/// dtype kept by `Auto` are refused alike, and a cast away from F16 storage
/// is not refused.
#[test]
fn planned_dtypes_decide_whatever_policy_chose_them() {
    let expected = [
        ExpectedTensor::new("a", [1_u64]),
        ExpectedTensor::new("b", [1_u64]),
    ];
    let mappings = MappingPlan::identity(&expected);
    let plan = |stored: TensorDType, dtype_policy: DTypePolicy| {
        let root = tempfile::TempDir::new().unwrap();
        let manifest = manifest_with(root.path(), &[("a", stored), ("b", stored)]);
        let options = LoadOptions {
            dtype_policy,
            ..LoadOptions::default()
        };
        build_load_plan(&manifest, &expected, &mappings, &options).unwrap()
    };
    let refused = |plan: &LoadPlan| {
        check_range_hazards(plan, &[F16_OVERFLOW], RangeHazardPolicy::Refuse).is_err()
    };
    let cast_f16 = plan(TensorDType::F32, DTypePolicy::Cast(TensorDType::F16));
    assert!(refused(&cast_f16));
    check_range_hazards(&cast_f16, &[F16_OVERFLOW], RangeHazardPolicy::Accept).unwrap();
    assert!(refused(&plan(TensorDType::F16, DTypePolicy::Auto)));
    assert!(refused(&plan(TensorDType::F16, DTypePolicy::Preserve)));
    assert!(!refused(&plan(
        TensorDType::F16,
        DTypePolicy::Cast(TensorDType::F32)
    )));
    assert!(!refused(&plan(
        TensorDType::F16,
        DTypePolicy::Cast(TensorDType::BF16)
    )));
    assert!(!refused(&plan(TensorDType::BF16, DTypePolicy::Auto)));
    assert!(!refused(&plan(TensorDType::F32, DTypePolicy::Auto)));
}

/// The shared planner refuses a plan in a dtype its mapping plan declares, under the caller's policy.
///
/// This is the load path every family contract builds on, so a declaration
/// on the mapping plan is enforced however the planner is reached. The
/// refusal equals the direct check's, accepting returns the plan the
/// undeclared mapping plan yields, and strict-validation failures still take
/// precedence over the hazard.
#[test]
fn the_planner_enforces_the_hazards_its_mapping_plan_declares() {
    let expected = [
        ExpectedTensor::new("a", [1_u64]),
        ExpectedTensor::new("b", [1_u64]),
    ];
    let undeclared = MappingPlan::identity(&expected);
    let mut declared = undeclared.clone();
    declared.declare_range_hazards([F16_OVERFLOW]);
    assert_eq!(declared.range_hazards(), [F16_OVERFLOW]);
    assert!(undeclared.range_hazards().is_empty());
    assert_eq!(declared.operations(), undeclared.operations());
    let root = tempfile::TempDir::new().unwrap();
    let manifest = manifest_with(
        root.path(),
        &[("a", TensorDType::F32), ("b", TensorDType::F16)],
    );
    let options = |dtype_policy: DTypePolicy, range_hazards: RangeHazardPolicy| LoadOptions {
        dtype_policy,
        range_hazards,
        ..LoadOptions::default()
    };
    let cast = DTypePolicy::Cast(TensorDType::F16);
    let reference = build_load_plan(
        &manifest,
        &expected,
        &undeclared,
        &options(cast, RangeHazardPolicy::Refuse),
    )
    .unwrap();
    for (dtype_policy, parameters, first) in [
        (cast, 2, "a"),
        (DTypePolicy::Auto, 1, "b"),
        (DTypePolicy::Preserve, 1, "b"),
    ] {
        let refused = build_load_plan(
            &manifest,
            &expected,
            &declared,
            &options(dtype_policy, RangeHazardPolicy::default()),
        )
        .unwrap_err();
        let CheckpointError::RangeHazard {
            hazard,
            target,
            parameters: count,
        } = &refused
        else {
            panic!("{dtype_policy:?}: expected a range-hazard refusal, got {refused:?}");
        };
        assert_eq!(
            (*hazard, target.as_str(), *count),
            (F16_OVERFLOW, first, parameters),
            "{dtype_policy:?}"
        );
    }
    let direct = check_range_hazards(&reference, &[F16_OVERFLOW], RangeHazardPolicy::Refuse)
        .unwrap_err()
        .to_string();
    let planned = build_load_plan(
        &manifest,
        &expected,
        &declared,
        &options(cast, RangeHazardPolicy::Refuse),
    )
    .unwrap_err()
    .to_string();
    assert_eq!(planned, direct);
    let accepted = build_load_plan(
        &manifest,
        &expected,
        &declared,
        &options(cast, RangeHazardPolicy::Accept),
    )
    .unwrap();
    assert_eq!(accepted, reference);
    // Planning away from the declared dtype is not refused.
    build_load_plan(
        &manifest,
        &expected,
        &declared,
        &options(
            DTypePolicy::Cast(TensorDType::F32),
            RangeHazardPolicy::Refuse,
        ),
    )
    .unwrap();
    // A strict-validation failure is reported before the hazard is checked.
    let extra = tempfile::TempDir::new().unwrap();
    let unexpected = manifest_with(
        extra.path(),
        &[
            ("a", TensorDType::F16),
            ("b", TensorDType::F16),
            ("extra", TensorDType::F16),
        ],
    );
    assert!(matches!(
        build_load_plan(
            &unexpected,
            &expected,
            &declared,
            &options(cast, RangeHazardPolicy::Refuse),
        ),
        Err(CheckpointError::StrictValidation {
            missing: 0,
            unexpected: 1,
            ..
        })
    ));
}

/// Declarations accumulate in order, and the first matching one is the one the planner reports.
#[test]
fn declarations_accumulate_in_order() {
    let bf16 = RangeHazard::new("Other", TensorDType::BF16, "other evidence");
    let expected = [ExpectedTensor::new("a", [1_u64])];
    let mut mappings = MappingPlan::identity(&expected);
    mappings
        .declare_range_hazards([bf16])
        .declare_range_hazards([F16_OVERFLOW, bf16]);
    assert_eq!(mappings.range_hazards(), [bf16, F16_OVERFLOW, bf16]);
    let root = tempfile::TempDir::new().unwrap();
    let manifest = manifest_with(root.path(), &[("a", TensorDType::F16)]);
    let Err(CheckpointError::RangeHazard { hazard, .. }) =
        build_load_plan(&manifest, &expected, &mappings, &LoadOptions::default())
    else {
        panic!("F16 storage under Auto must be refused");
    };
    assert_eq!(hazard, F16_OVERFLOW);
}

/// Serialization writes only the operations, so a deserialized plan declares no hazards.
///
/// The serialized format is the one written before plans carried hazards,
/// and a round trip that drops a family's declarations is visible as an
/// unequal plan rather than a silent copy.
#[test]
fn serialized_mapping_plans_carry_no_hazards() {
    let expected = [ExpectedTensor::new("a", [1_u64])];
    let undeclared = MappingPlan::identity(&expected);
    let mut declared = undeclared.clone();
    declared.declare_range_hazards([F16_OVERFLOW]);
    let json = serde_json::to_string(&declared).unwrap();
    assert_eq!(json, serde_json::to_string(&undeclared).unwrap());
    assert!(!json.contains("range_hazards"), "{json}");
    let restored: MappingPlan = serde_json::from_str(&json).unwrap();
    assert!(restored.range_hazards().is_empty());
    assert_eq!(restored, undeclared);
    assert_ne!(restored, declared);
}

proptest! {
    /// A refusal happens exactly when some parameter has the hazard's dtype, with the right count.
    #[test]
    fn refusal_is_exactly_a_matching_parameter(
        dtypes in prop::collection::vec(prop::sample::select(CASTABLE.to_vec()), 0..12),
        hazard in prop::sample::select(CASTABLE.to_vec()),
    ) {
        let declaration = RangeHazard::new("checkpoints", hazard, "evidence");
        let plan = plan_with(&dtypes);
        let matching: Vec<usize> = dtypes
            .iter()
            .enumerate()
            .filter_map(|(index, dtype)| (*dtype == hazard).then_some(index))
            .collect();
        prop_assert!(check_range_hazards(&plan, &[declaration], RangeHazardPolicy::Accept).is_ok());
        match check_range_hazards(&plan, &[declaration], RangeHazardPolicy::Refuse) {
            Ok(()) => prop_assert!(matching.is_empty()),
            Err(CheckpointError::RangeHazard { target, parameters, .. }) => {
                prop_assert_eq!(target, format!("t{}", matching[0]));
                prop_assert_eq!(parameters, matching.len());
            }
            Err(other) => prop_assert!(false, "unexpected error {other:?}"),
        }
    }
}
