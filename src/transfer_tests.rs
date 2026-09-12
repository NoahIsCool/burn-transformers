//! Deterministic transfer tests for selection, aliasing, tensor layouts, and rollback.

use burn::{
    backend::Flex,
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{DType, Tensor},
};

use super::*;

/// CPU test backend with F32 values and I32 indices.
type B = Flex<f32, i32>;

/// Tiny affine parameter container with independently selectable weight and bias.
#[derive(Module, Debug)]
struct Block<B: Backend> {
    /// Matrix with two input rows and a configurable output width.
    weight: Param<Tensor<B, 2>>,
    /// Bias vector with one entry per output column.
    bias: Param<Tensor<B, 1>>,
}

/// Nested module fixture with repeated encoder paths and a separate prediction head.
#[derive(Module, Debug)]
struct Experiment<B: Backend> {
    /// Encoder blocks whose vector indices become parameter path components.
    encoder: Vec<Block<B>>,
    /// Independently selectable one-column output block.
    head: Block<B>,
}

/// Construct a block whose parameters all contain the requested constant value.
fn block(width: usize, value: f32) -> Block<B> {
    let device = Default::default();
    Block {
        weight: Param::from_tensor(Tensor::full([2, width], value, &device)),
        bias: Param::from_tensor(Tensor::full([width], value, &device)),
    }
}

/// Construct nested constant-valued blocks with the requested encoder widths.
fn experiment(widths: &[usize], value: f32) -> Experiment<B> {
    Experiment {
        encoder: widths.iter().map(|&width| block(width, value)).collect(),
        head: block(1, value),
    }
}

/// Read a dense F32 fixture tensor, failing the test if decoding is unsupported.
fn values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .to_vec()
        .expect("fixture tensor must contain dense F32 values")
}

/// Prefix selection copies matching blocks and reports unchanged additions and omissions.
#[test]
fn prefix_transfer_reports_reused_omitted_and_new_parameters() {
    let source = experiment(&[3], 7.);
    let target = experiment(&[3, 4], 2.);
    let (actual, report) = TransferPlan::new()
        .copy_prefix("encoder", "encoder")
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(actual.encoder[0].weight.val()), vec![7.; 6]);
    assert_eq!(values(actual.encoder[1].weight.val()), vec![2.; 8]);
    assert_eq!(values(actual.head.weight.val()), vec![2.; 2]);
    assert_eq!(values(target.encoder[0].weight.val()), vec![2.; 6]);
    assert_eq!(report.reused.len(), 2);
    assert_eq!(report.omitted.len(), 2);
    assert_eq!(report.newly_initialized.len(), 4);
    assert_eq!(report.reused[1].source, "encoder.0.weight");
    assert_eq!(report.reused[1].target, "encoder.0.weight");
    assert_eq!(report.reused[1].shape, [2, 3]);
    assert!(report.newly_initialized[0].reason.contains("no source"));
    assert_eq!(parameters(&actual).unwrap()[1].name, "encoder.0.weight");
}

/// Lazy linear layers that expose whether selection initializes omitted parameters.
#[derive(Module, Debug)]
struct LazySource<B: Backend> {
    /// Layer selected by the transfer plan.
    selected: Linear<B>,
    /// Layer that must remain lazy when it is not selected.
    omitted: Linear<B>,
}

/// Selecting one lazy layer leaves all parameter tensors of the omitted layer unread.
#[test]
fn unselected_source_parameters_are_not_initialized_or_read() {
    let device = Default::default();
    let source = LazySource {
        selected: LinearConfig::new(2, 3).init::<B>(&device),
        omitted: LinearConfig::new(2, 8).init::<B>(&device),
    };
    assert!(!source.selected.weight.is_initialized());
    assert!(!source.omitted.weight.is_initialized());
    let target = experiment(&[3], 2.);
    let (_, report) = TransferPlan::new()
        .copy_prefix("selected", "encoder.0")
        .apply(&source, &target)
        .unwrap();
    assert!(source.selected.weight.is_initialized());
    assert!(!source.omitted.weight.is_initialized());
    assert!(!source.omitted.bias.as_ref().unwrap().is_initialized());
    assert_eq!(
        report
            .omitted
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["omitted.bias", "omitted.weight"]
    );
}

/// Shape mismatches reject direct copying unless initialization or a transform is explicit.
#[test]
fn changed_shapes_require_reinitialization_or_an_explicit_transform() {
    let source = block(3, 7.);
    let target = block(2, 2.);
    assert!(
        TransferPlan::new()
            .copy_matching()
            .apply(&source, &target)
            .is_err()
    );
    let (unchanged, report) = TransferPlan::new()
        .copy_matching()
        .on_shape_change(ShapeChangePolicy::Reinitialize)
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(unchanged.weight.val()), vec![2.; 4]);
    assert_eq!(report.newly_initialized.len(), 2);
    assert!(report.newly_initialized[0].reason.contains("differs"));
    assert_eq!(report.omitted.len(), 2);

    let (cropped, report) = TransferPlan::new()
        .transform(
            "weight",
            "weight",
            "select first two output channels",
            |x: Tensor<B, 2>| Ok(x.slice([0..2, 0..2])),
        )
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(cropped.weight.val()), vec![7.; 4]);
    assert_eq!(values(cropped.bias.val()), vec![2.; 2]);
    assert_eq!(report.transformed[0].sources, ["weight"]);
    assert_eq!(report.transformed[0].shape, [2, 2]);
}

/// Transform failures preserve target values, and invalid paths fail before callbacks run.
#[test]
fn failed_transforms_and_invalid_rules_leave_the_target_unchanged() {
    let source = experiment(&[2, 2], 7.);
    let target = experiment(&[2, 2], 2.);
    let failed = TransferPlan::new()
        .copy_prefix("encoder", "encoder")
        .transform::<2, 2>(
            "encoder.1.weight",
            "encoder.1.weight",
            "failing experiment",
            |_| Err(Error::configuration("probe", "intentional failure")),
        )
        .apply(&source, &target);
    let error = failed.unwrap_err().to_string();
    assert!(error.contains("encoder.1.weight"));
    assert!(error.contains("intentional failure"));
    assert_eq!(values(target.encoder[0].weight.val()), vec![2.; 4]);
    assert_eq!(values(target.encoder[1].weight.val()), vec![2.; 4]);

    let called = std::cell::Cell::new(false);
    let invalid = TransferPlan::new()
        .transform(
            "encoder.0.weight",
            "encoder.0.weight",
            "must not run",
            |x: Tensor<B, 2>| {
                called.set(true);
                Ok(x)
            },
        )
        .copy("nonexistent.weight", "head.weight")
        .apply(&source, &target);
    assert!(invalid.is_err());
    assert!(!called.get());
}

/// Exact transforms and keep rules override matching while ambiguous rules are rejected.
#[test]
fn exact_overrides_and_explicit_initialization_are_unambiguous() {
    let source = experiment(&[2], 7.);
    let target = experiment(&[2], 2.);
    let (actual, report) = TransferPlan::new()
        .copy_matching()
        .transform(
            "encoder.0.weight",
            "encoder.0.weight",
            "halve weights",
            |x: Tensor<B, 2>| Ok(x * 0.5),
        )
        .keep_initialized("head")
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(actual.encoder[0].weight.val()), vec![3.5; 4]);
    assert_eq!(values(actual.head.weight.val()), vec![2.; 2]);
    assert_eq!(report.reused.len(), 1);
    assert_eq!(report.transformed.len(), 1);
    assert_eq!(report.newly_initialized.len(), 2);
    assert!(
        TransferPlan::new()
            .copy("head.weight", "head.weight")
            .copy("head.weight", "head.weight")
            .apply(&source, &target)
            .is_err()
    );
    assert!(
        TransferPlan::new()
            .copy_prefix("encoder", "encoder")
            .copy_matching()
            .apply(&source, &target)
            .is_err()
    );
    assert!(
        TransferPlan::new()
            .copy_prefix("encode", "encoder")
            .apply(&source, &target)
            .is_err()
    );
}

/// Matching disjoint parameter layouts returns an actionable configuration error.
#[test]
fn matching_nothing_returns_an_actionable_error() {
    let error = TransferPlan::new()
        .copy_matching()
        .apply(&pair(false, 7.), &block(2, 2.))
        .unwrap_err();
    assert!(error.to_string().contains("no matching parameter paths"));
}

/// Pair of equal-shaped matrices with controllable parameter identity sharing.
#[derive(Module, Debug)]
struct Pair<B: Backend> {
    /// First matrix, which owns the shared identity in tied fixtures.
    left: Param<Tensor<B, 2>>,
    /// Second matrix, either an alias of `left` or an independent parameter.
    right: Param<Tensor<B, 2>>,
}

/// Construct two constant matrices with tied or independent parameter identities.
fn pair(tied: bool, value: f32) -> Pair<B> {
    let left = Param::from_tensor(Tensor::full([2, 2], value, &Default::default()));
    let right = if tied {
        left.clone()
    } else {
        Param::from_tensor(Tensor::full([2, 2], value, &Default::default()))
    };
    Pair { left, right }
}

/// Source aliases receive a fresh shared identity unless explicit untying is requested.
#[test]
fn source_ties_are_preserved_and_can_be_explicitly_removed() {
    let source = pair(true, 7.);
    let target = pair(false, 2.);
    let (tied, report) = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .unwrap();
    assert_eq!(tied.left.id, tied.right.id);
    assert_ne!(tied.left.id, source.left.id);
    assert_eq!(report.preserved_ties, [vec!["left", "right"]]);
    assert_eq!(
        parameters(&tied).unwrap()[1].tied_to.as_deref(),
        Some("left")
    );

    let (untied, report) = TransferPlan::new()
        .copy_matching()
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .unwrap();
    assert_ne!(untied.left.id, untied.right.id);
    assert_eq!(values(untied.left.val()), values(untied.right.val()));
    assert_eq!(report.untied, [vec!["left", "right"]]);
}

/// Updating one target alias requires explicit untying and leaves its peer initialized.
#[test]
fn partial_target_alias_updates_require_intentional_untying() {
    let source = pair(false, 7.);
    let target = pair(true, 2.);
    let error = TransferPlan::new()
        .copy("left", "left")
        .apply(&source, &target)
        .unwrap_err();
    assert!(error.to_string().contains("TiePolicy::Untie"));
    let (actual, report) = TransferPlan::new()
        .copy("left", "left")
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .unwrap();
    assert_ne!(actual.left.id, actual.right.id);
    assert_eq!(values(actual.left.val()), vec![7.; 4]);
    assert_eq!(values(actual.right.val()), vec![2.; 4]);
    assert_eq!(report.untied, [vec!["left", "right"]]);
    assert_eq!(target.left.id, target.right.id);
}

/// Different transform rules on a shared source cannot silently create independent owners.
#[test]
fn transformed_aliases_reject_accidental_untying() {
    let source = pair(true, 7.);
    let target = pair(false, 2.);
    let error = TransferPlan::new()
        .copy_matching()
        .transform("left", "left", "halve", |x: Tensor<B, 2>| Ok(x * 0.5))
        .apply(&source, &target)
        .unwrap_err();
    assert!(error.to_string().contains("source aliases"));
}

/// Equal source identities with divergent tensor values are rejected unless untied.
#[test]
fn diverged_source_alias_values_are_never_silently_collapsed() {
    let mut source = pair(true, 7.);
    source.right = source.right.map(|x| x * 2.);
    let target = pair(false, 2.);
    let error = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .unwrap_err();
    assert!(error.to_string().contains("unequal values"));
    let (actual, _) = TransferPlan::new()
        .copy_matching()
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(actual.left.val()), vec![7.; 4]);
    assert_eq!(values(actual.right.val()), vec![14.; 4]);
}

/// Several alias groups plus an independent parameter for cross-group invariants.
#[derive(Module, Debug)]
struct AliasGroups<B: Backend> {
    /// Separately owned pairs whose indices produce distinct nested paths.
    groups: Vec<Pair<B>>,
    /// Equal-valued singleton that must never become tied merely by value equality.
    singleton: Param<Tensor<B, 2>>,
}

/// Construct alias groups and a singleton with independently specified initial values.
fn alias_groups(groups: &[(bool, f32)], singleton: f32) -> AliasGroups<B> {
    AliasGroups {
        groups: groups
            .iter()
            .map(|&(tied, value)| pair(tied, value))
            .collect(),
        singleton: Param::from_tensor(Tensor::full([2, 2], singleton, &Default::default())),
    }
}

/// Separate source ties and untouched target ties survive together without merging equal values.
#[test]
fn several_alias_groups_preserve_independent_owners_and_untouched_groups() {
    let source = alias_groups(&[(true, 7.), (true, 7.), (false, 11.)], 7.);
    let target = alias_groups(&[(false, 2.), (true, 3.), (true, 5.)], 2.);
    let untouched_id = target.groups[2].left.id;
    let (actual, report) = TransferPlan::new()
        .copy_prefix("groups.0", "groups.0")
        .copy_prefix("groups.1", "groups.1")
        .copy("singleton", "singleton")
        .apply(&source, &target)
        .expect("compatible independent alias groups should transfer");

    for index in 0..2 {
        assert_eq!(actual.groups[index].left.id, actual.groups[index].right.id);
        assert_ne!(actual.groups[index].left.id, source.groups[index].left.id);
        assert_ne!(actual.groups[index].left.id, target.groups[index].left.id);
        assert_eq!(values(actual.groups[index].left.val()), vec![7.; 4]);
    }
    assert_ne!(actual.groups[0].left.id, actual.groups[1].left.id);
    assert_ne!(actual.singleton.id, actual.groups[0].left.id);
    assert_ne!(actual.singleton.id, actual.groups[1].left.id);
    assert_eq!(values(actual.singleton.val()), vec![7.; 4]);
    assert_eq!(actual.groups[2].left.id, untouched_id);
    assert_eq!(actual.groups[2].right.id, untouched_id);
    assert_eq!(values(actual.groups[2].right.val()), vec![5.; 4]);
    assert_eq!(
        report.preserved_ties,
        [
            vec!["groups.0.left", "groups.0.right"],
            vec!["groups.1.left", "groups.1.right"],
            vec!["groups.2.left", "groups.2.right"],
        ]
    );
    assert!(report.untied.is_empty());
    assert_eq!(report.reused.len(), 5);
    assert_eq!(report.newly_initialized.len(), 2);
    assert_eq!(report.omitted.len(), 2);
}

/// Untying affects selected source and target groups while an unselected group retains sharing.
#[test]
fn mixed_alias_groups_untie_only_affected_parameters() {
    let source = alias_groups(&[(true, 7.), (false, 9.), (true, 11.)], 7.);
    let target = alias_groups(&[(false, 2.), (true, 3.), (true, 5.)], 2.);
    let untouched_id = target.groups[2].left.id;
    let (actual, report) = TransferPlan::new()
        .copy_prefix("groups.0", "groups.0")
        .copy("groups.1.right", "groups.1.right")
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .expect("explicit untying should permit a partially selected target group");

    assert_ne!(actual.groups[0].left.id, actual.groups[0].right.id);
    assert_eq!(values(actual.groups[0].left.val()), vec![7.; 4]);
    assert_eq!(values(actual.groups[0].right.val()), vec![7.; 4]);
    assert_ne!(actual.groups[1].left.id, actual.groups[1].right.id);
    assert_eq!(actual.groups[1].left.id, target.groups[1].left.id);
    assert_eq!(values(actual.groups[1].left.val()), vec![3.; 4]);
    assert_eq!(values(actual.groups[1].right.val()), vec![9.; 4]);
    assert_eq!(actual.groups[2].left.id, untouched_id);
    assert_eq!(actual.groups[2].right.id, untouched_id);
    assert_eq!(
        report.untied,
        [
            vec!["groups.0.left", "groups.0.right"],
            vec!["groups.1.left", "groups.1.right"],
        ]
    );
    assert_eq!(
        report.preserved_ties,
        [vec!["groups.2.left", "groups.2.right"]]
    );
}

/// A divergent value in one selected alias group leaves every target group and singleton unchanged.
#[test]
fn failed_alias_group_validation_preserves_the_complete_target() {
    let mut source = alias_groups(&[(true, 7.), (true, 9.)], 11.);
    source.groups[1].right = source.groups[1].right.clone().map(|value| value * 2.);
    let target = alias_groups(&[(false, 2.), (true, 3.)], 5.);
    let original = parameters(&target).expect("dense target metadata should be inspectable");
    let error = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .expect_err("inconsistent source aliases must reject the whole transfer");

    assert!(error.to_string().contains("unequal values"));
    assert_eq!(parameters(&target).expect("target remains valid"), original);
    for (group, expected) in target.groups.iter().zip([2., 3.]) {
        assert_eq!(values(group.left.val()), vec![expected; 4]);
        assert_eq!(values(group.right.val()), vec![expected; 4]);
    }
    assert_eq!(values(target.singleton.val()), vec![5.; 4]);
    assert_eq!(values(source.groups[1].right.val()), vec![18.; 4]);
}

/// A later selected target alias still fails preservation when its first sorted peer is untouched.
#[test]
fn selecting_only_a_later_target_alias_rejects_implicit_untying() {
    let source = pair(false, 7.);
    let target = pair(true, 2.);
    let original_id = target.left.id;
    let error = TransferPlan::new()
        .copy("right", "right")
        .apply(&source, &target)
        .expect_err("an unselected first alias must not conceal a partial update");
    assert!(error.to_string().contains("target aliases"));
    assert_eq!(target.left.id, original_id);
    assert_eq!(target.right.id, original_id);
    assert_eq!(values(target.right.val()), vec![2.; 4]);
}

/// Shared source identities with different shapes or dtypes fail before equality reduction.
#[test]
fn incompatible_alias_layouts_are_rejected_before_comparing_tensor_values() {
    for change_shape in [true, false] {
        let mut source = pair(true, 7.);
        let mut target = pair(false, 2.);
        if change_shape {
            source.right = source.right.map(|value| value.reshape([1, 4]));
            target.right = target.right.map(|value| value.reshape([1, 4]));
        } else {
            source.right = source.right.map(|value| value.cast(DType::F64));
            target.right = target.right.map(|value| value.cast(DType::F64));
        }
        let original_right = target.right.val().into_data();
        let error = TransferPlan::new()
            .copy_matching()
            .apply(&source, &target)
            .expect_err("preserving aliases requires matching runtime layouts");
        assert!(error.to_string().contains("source aliases"));
        assert_eq!(values(target.left.val()), vec![2.; 4]);
        assert_eq!(target.right.val().into_data(), original_right);
    }
}

/// Dtype conversions need an explicit transform whose output shape and rank are validated.
#[test]
fn dtype_conversion_must_be_explicit_and_results_are_checked() {
    let source = block(2, 7.);
    let mut target = block(2, 2.);
    target.weight = target.weight.map(|x| x.cast(DType::F64));
    assert!(
        TransferPlan::new()
            .copy("weight", "weight")
            .apply(&source, &target)
            .is_err()
    );
    let (actual, report) = TransferPlan::new()
        .transform("weight", "weight", "convert to f64", |x: Tensor<B, 2>| {
            Ok(x.cast(DType::F64))
        })
        .apply(&source, &target)
        .unwrap();
    assert_eq!(actual.weight.val().dtype(), DType::F64);
    assert_eq!(report.transformed.len(), 1);
    assert!(
        TransferPlan::new()
            .transform("weight", "weight", "invalid shape", |x: Tensor<B, 2>| Ok(
                x.narrow(1, 0, 1)
            ))
            .apply(&source, &target)
            .is_err()
    );
    assert!(
        TransferPlan::new()
            .transform::<1, 1>("weight", "bias", "invalid callback rank", Ok)
            .apply(&source, &target)
            .is_err()
    );
}

/// Transferred column-layout parameters retain Burn record mappers across save and reload.
#[test]
fn transfer_preserves_burn_parameter_layout_mappers_on_record_roundtrip() {
    let device = Default::default();
    let config = LinearConfig::new(2, 3).with_layout(LinearLayout::Col);
    let source = config.init::<B>(&device);
    let target = config.init::<B>(&device);
    let (actual, _) = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .unwrap();
    let record = actual.clone().into_record();
    let expected_record = source.clone().into_record();
    assert_eq!(
        values(record.weight.val()),
        values(expected_record.weight.val())
    );
    let restored = config.init::<B>(&device).load_record(record);
    let input = Tensor::<B, 2>::ones([1, 2], &device);
    assert_eq!(
        values(actual.forward(input.clone())),
        values(restored.forward(input))
    );
}

/// Preserved aliases share a fresh gradient leaf and respect the target freezing policy.
#[cfg(feature = "autodiff")]
#[test]
fn transferred_ties_share_gradients_without_linking_to_source_parameters() {
    use burn::backend::Autodiff;
    /// Differentiable wrapper for the deterministic fixture backend.
    type Ad = Autodiff<B>;
    let left = Param::from_tensor(Tensor::<Ad, 2>::ones([2, 2], &Default::default()));
    let source = Pair {
        left: left.clone(),
        right: left,
    };
    let target = Pair {
        left: Param::from_tensor(Tensor::<Ad, 2>::zeros([2, 2], &Default::default())),
        right: Param::from_tensor(Tensor::<Ad, 2>::zeros([2, 2], &Default::default())),
    };
    let (actual, _) = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .unwrap();
    let gradients = (actual.left.val() + actual.right.val() * 3.)
        .sum()
        .backward();
    assert_eq!(
        values(actual.left.val().grad(&gradients).unwrap()),
        vec![4.; 4]
    );
    assert_eq!(
        values(actual.right.val().grad(&gradients).unwrap()),
        vec![4.; 4]
    );
    assert!(source.left.val().grad(&gradients).is_none());

    let frozen = target.no_grad();
    let (frozen, _) = TransferPlan::new()
        .copy_matching()
        .apply(&source, &frozen)
        .unwrap();
    assert!(!frozen.left.val().is_require_grad());
    assert!(!frozen.right.val().is_require_grad());
}

/// Empty operation descriptions and unknown keep paths fail before any callback side effect.
#[test]
fn rule_preflight_rejects_empty_descriptions_and_unknown_keep_paths_before_callbacks() {
    let source = experiment(&[2], 7.0);
    let target = experiment(&[2], 2.0);
    let original_id = target.encoder[0].weight.id;
    for description in ["", " \n\t"] {
        let called = std::cell::Cell::new(false);
        let error = TransferPlan::new()
            .transform(
                "encoder.0.weight",
                "encoder.0.weight",
                description,
                |x: Tensor<B, 2>| {
                    called.set(true);
                    Ok(x * 3.0)
                },
            )
            .apply(&source, &target)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("describe the explicit transformation")
        );
        assert!(!called.get());
    }
    for path in ["", "missing", "encoder.10"] {
        let called = std::cell::Cell::new(false);
        let error = TransferPlan::new()
            .transform(
                "encoder.0.weight",
                "encoder.0.weight",
                "triple",
                |x: Tensor<B, 2>| {
                    called.set(true);
                    Ok(x * 3.0)
                },
            )
            .keep_initialized(path)
            .apply(&source, &target)
            .unwrap_err();
        assert!(error.to_string().contains("transfer.keep_initialized"));
        assert!(!called.get());
    }
    assert_eq!(target.encoder[0].weight.id, original_id);
    assert_eq!(values(target.encoder[0].weight.val()), vec![2.0; 4]);
}

/// Nested block fixture whose child has weight and bias parameter suffixes.
#[derive(Module, Debug)]
struct WeightBranch<B: Backend> {
    /// Child module used to validate relative prefix matching.
    branch: Block<B>,
}

/// Nested pair fixture whose child has left and right parameter suffixes.
#[derive(Module, Debug)]
struct PairBranch<B: Backend> {
    /// Child with a deliberately incompatible parameter layout for prefix tests.
    branch: Pair<B>,
}

/// Existing module prefixes are insufficient when no relative parameter names match.
#[test]
fn existing_module_prefixes_without_matching_parameter_suffixes_are_rejected() {
    // Both modules exist, but weight/bias cannot match left/right by relative path.
    let source = WeightBranch {
        branch: block(2, 7.0),
    };
    let target = PairBranch {
        branch: pair(true, 2.0),
    };
    let original_id = target.branch.left.id;
    let error = TransferPlan::new()
        .copy_prefix("branch", "branch")
        .apply(&source, &target)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no matching relative parameter paths")
    );
    assert_eq!(target.branch.left.id, original_id);
    assert_eq!(target.branch.right.id, original_id);
    assert_eq!(values(target.branch.left.val()), vec![2.0; 4]);
}

/// An empty plan preserves initialized identities and reports every unused source and target.
#[test]
fn an_empty_plan_preserves_initialized_aliases_and_reports_all_dispositions() {
    let source = pair(false, 7.0);
    let target = pair(true, 2.0);
    let original_id = target.left.id;
    let (actual, report) = TransferPlan::<B>::default()
        .apply(&source, &target)
        .unwrap();
    assert_eq!(actual.left.id, original_id);
    assert_eq!(actual.right.id, original_id);
    assert_eq!(values(actual.left.val()), vec![2.0; 4]);
    assert!(report.reused.is_empty());
    assert!(report.transformed.is_empty());
    assert_eq!(report.omitted.len(), 2);
    assert_eq!(report.newly_initialized.len(), 2);
    assert_eq!(report.preserved_ties, [vec!["left", "right"]]);
}

/// Selected packed tensors are rejected while omitted packed source payloads remain untouched.
#[test]
fn packed_parameters_require_explicit_conversion_but_unselected_sources_stay_packed() {
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    };
    let device = Default::default();
    let packed_data = quantize_tensor_data(
        &burn::tensor::TensorData::new(
            (0..64).map(|i| i as f32 / 8.0 - 4.0).collect::<Vec<_>>(),
            [2, 32],
        ),
        WeightQuantizationConfig {
            scheme: SymmetricQuantization::Q4,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
    .into_burn_qfloat_data()
    .unwrap();
    let mut source = block(32, 7.0);
    source.weight = Param::from_tensor(Tensor::<B, 2>::from_data(packed_data, &device));
    let original_packed = source.weight.val().into_data();
    let target = block(32, 2.0);
    let original_target_id = target.weight.id;
    assert!(
        parameters(&source)
            .unwrap_err()
            .to_string()
            .contains("dense floating parameters")
    );
    let error = TransferPlan::new()
        .copy("weight", "weight")
        .apply(&source, &target)
        .unwrap_err();
    assert!(error.to_string().contains("dense floating parameters"));
    assert_eq!(target.weight.id, original_target_id);
    assert_eq!(values(target.weight.val()), vec![2.0; 64]);

    // Choosing only the dense bias never implicitly dequantizes the unused weight.
    let (actual, report) = TransferPlan::new()
        .copy("bias", "bias")
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(actual.bias.val()), vec![7.0; 32]);
    assert_eq!(values(actual.weight.val()), vec![2.0; 64]);
    assert_eq!(report.reused[0].source, "bias");
    assert_eq!(report.omitted[0].name, "weight");
    assert!(matches!(source.weight.val().dtype(), DType::QFloat(_)));
    assert_eq!(source.weight.val().into_data(), original_packed);
}
