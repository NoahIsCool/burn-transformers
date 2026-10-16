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

/// Packed sources are listed, rejected when selected, and untouched when omitted.
#[test]
fn packed_parameters_require_explicit_conversion_but_unselected_sources_stay_packed() {
    let mut source = block(32, 7.0);
    source.weight = packed_weight();
    let original_packed = source.weight.val().into_data();
    let target = block(32, 2.0);
    let original_target_id = target.weight.id;
    // Packed weights are listed for inspection but marked non-transferable.
    let listed = parameters(&source).expect("packed checkpoints remain inspectable");
    let weight = listed.iter().find(|info| info.name == "weight").unwrap();
    assert!(matches!(weight.dtype, DType::QFloat(_)));
    assert_eq!(weight.shape, [2, 32]);
    assert!(!weight.is_transferable());
    assert!(
        listed
            .iter()
            .find(|info| info.name == "bias")
            .unwrap()
            .is_transferable()
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

// Fan-out copies versus existing parameter sharing.

/// One source path copied to two targets yields independent identities under either policy.
#[test]
fn fan_out_copies_receive_independent_identities_and_report_no_ties() {
    let source = pair(false, 7.);
    let target = pair(false, 2.);
    for policy in [TiePolicy::Preserve, TiePolicy::Untie] {
        let (actual, report) = TransferPlan::new()
            .copy("left", "left")
            .copy("left", "right")
            .with_ties(policy)
            .apply(&source, &target)
            .expect("copying one untied source twice is ordinary fan-out");
        assert_ne!(actual.left.id, actual.right.id, "{policy:?}");
        assert_ne!(actual.left.id, source.left.id);
        assert_eq!(values(actual.left.val()), vec![7.; 4]);
        assert_eq!(values(actual.right.val()), vec![7.; 4]);
        assert!(report.preserved_ties.is_empty(), "{policy:?}");
        assert!(report.untied.is_empty(), "{policy:?}");
        assert_eq!(report.reused.len(), 2);
        assert_eq!(report.omitted.len(), 1);
        assert_eq!(report.omitted[0].name, "right");
        assert!(
            parameters(&actual)
                .unwrap()
                .iter()
                .all(|p| p.tied_to.is_none())
        );
    }
}

/// Depth up-scaling by duplicating a prefix creates separately owned layers.
#[test]
fn duplicated_prefixes_build_independent_deeper_stacks() {
    let source = experiment(&[2], 7.);
    let target = experiment(&[2, 2], 2.);
    let (actual, report) = TransferPlan::new()
        .copy_prefix("encoder.0", "encoder.0")
        .copy_prefix("encoder.0", "encoder.1")
        .apply(&source, &target)
        .expect("layer duplication must not require an explicit tie policy");
    assert_ne!(actual.encoder[0].weight.id, actual.encoder[1].weight.id);
    assert_ne!(actual.encoder[0].bias.id, actual.encoder[1].bias.id);
    for block in &actual.encoder {
        assert_eq!(values(block.weight.val()), vec![7.; 4]);
        assert_eq!(values(block.bias.val()), vec![7.; 2]);
    }
    assert!(report.preserved_ties.is_empty());
    assert_eq!(report.reused.len(), 4);
    assert_eq!(
        report
            .newly_initialized
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["head.bias", "head.weight"]
    );
}

/// Different transforms of one untied source, as in perturbed expert upcycling, both apply.
#[test]
fn several_transforms_of_one_untied_source_are_independent() {
    let source = pair(false, 7.);
    let target = pair(false, 2.);
    let (actual, report) = TransferPlan::new()
        .transform("left", "left", "scale by two", |x: Tensor<B, 2>| Ok(x * 2.))
        .transform("left", "right", "negate", |x: Tensor<B, 2>| Ok(-x))
        .apply(&source, &target)
        .expect("independent transforms of one source must not be treated as aliases");
    assert_ne!(actual.left.id, actual.right.id);
    assert_eq!(values(actual.left.val()), vec![14.; 4]);
    assert_eq!(values(actual.right.val()), vec![-7.; 4]);
    assert_eq!(report.transformed.len(), 2);
    assert!(report.preserved_ties.is_empty());
}

/// Fan-out into a target that is already tied keeps that target tie unless explicitly untied.
#[test]
fn fan_out_into_a_tied_target_preserves_the_target_tie() {
    let source = pair(false, 7.);
    let target = pair(true, 2.);
    let plan = || {
        TransferPlan::new()
            .copy("left", "left")
            .copy("left", "right")
    };
    let (tied, report) = plan().apply(&source, &target).unwrap();
    assert_eq!(tied.left.id, tied.right.id);
    assert_ne!(tied.left.id, target.left.id);
    assert_eq!(values(tied.right.val()), vec![7.; 4]);
    assert_eq!(report.preserved_ties, [vec!["left", "right"]]);

    let (untied, report) = plan()
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .unwrap();
    assert_ne!(untied.left.id, untied.right.id);
    assert_eq!(report.untied, [vec!["left", "right"]]);
    assert!(report.preserved_ties.is_empty());

    let error = TransferPlan::new()
        .copy("left", "left")
        .transform("left", "right", "negate", |x: Tensor<B, 2>| Ok(-x))
        .apply(&source, &target)
        .expect_err("a tied target cannot receive two different values");
    assert!(error.to_string().contains("target aliases"));
}

/// Copying one member of a source tie to extra targets is ambiguous unless explicitly untied.
#[test]
fn fanning_out_a_tied_source_requires_an_explicit_policy() {
    let source = alias_groups(&[(true, 7.)], 3.);
    let target = alias_groups(&[(false, 2.)], 2.);
    let original = parameters(&target).unwrap();
    let plan = || {
        TransferPlan::new()
            .copy_prefix("groups.0", "groups.0")
            .copy("groups.0.left", "singleton")
    };
    let error = plan()
        .apply(&source, &target)
        .expect_err("joining the copied singleton to the tied pair would be implicit");
    let message = error.to_string();
    assert!(message.contains("groups.0.left"), "{message}");
    assert!(message.contains("TiePolicy::Untie"), "{message}");
    assert_eq!(parameters(&target).unwrap(), original);

    let (actual, report) = plan()
        .with_ties(TiePolicy::Untie)
        .apply(&source, &target)
        .unwrap();
    let ids = [
        actual.groups[0].left.id,
        actual.groups[0].right.id,
        actual.singleton.id,
    ];
    assert!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
    assert_eq!(values(actual.singleton.val()), vec![7.; 4]);
    assert_eq!(
        report.untied,
        [vec!["groups.0.left", "groups.0.right", "singleton"]]
    );
}

/// A flat bank of vectors whose entries may share parameter identities.
#[derive(Module, Debug)]
struct Bank<B: Backend> {
    /// Rank-one parameters addressed as `weights.<index>`.
    weights: Vec<Param<Tensor<B, 1>>>,
}

/// Builds a bank whose entries share one parameter per equal group label.
///
/// Entry `i` holds the constant `offset + group[i]`, so tied entries also hold
/// equal values, as genuine aliases must.
fn bank(groups: &[usize], offset: f32) -> Bank<B> {
    let device = Default::default();
    let mut owners = BTreeMap::<usize, Param<Tensor<B, 1>>>::new();
    Bank {
        weights: groups
            .iter()
            .map(|&group| {
                owners
                    .entry(group)
                    .or_insert_with(|| {
                        Param::from_tensor(Tensor::full([2], offset + group as f32, &device))
                    })
                    .clone()
            })
            .collect(),
    }
}

proptest::proptest! {
    /// Arbitrary copy maps reproduce the fan-out/tie oracle exactly.
    ///
    /// For a source alias group `g`, let `P(g)` be the distinct source paths of `g`
    /// that some target copies. The group is a preserved tie when `|P(g)| >= 2`;
    /// it is ambiguous (rejected) when it is a tie and one of its paths feeds
    /// two targets. Otherwise targets share an identity exactly when they copy
    /// from one tied group, and every untied copy is independent.
    #[test]
    fn copy_maps_share_identities_exactly_for_used_source_ties(
        source_groups in proptest::collection::vec(0usize..3, 1..5),
        sources_for_targets in proptest::collection::vec(0usize..5, 1..6),
        untie in proptest::bool::ANY,
    ) {
        let source = bank(&source_groups, 10.0);
        let target = bank(&(0..sources_for_targets.len()).collect::<Vec<_>>(), -10.0);
        let chosen = sources_for_targets
            .iter()
            .map(|index| index % source_groups.len())
            .collect::<Vec<_>>();
        let mut plan = TransferPlan::new();
        for (target_index, source_index) in chosen.iter().enumerate() {
            plan = plan.copy(format!("weights.{source_index}"), format!("weights.{target_index}"));
        }
        if untie {
            plan = plan.with_ties(TiePolicy::Untie);
        }

        let mut used_paths = BTreeMap::<usize, BTreeSet<usize>>::new();
        let mut uses = BTreeMap::<usize, usize>::new();
        for &source_index in &chosen {
            used_paths.entry(source_groups[source_index]).or_default().insert(source_index);
            *uses.entry(source_index).or_default() += 1;
        }
        let tied = |group: usize| used_paths[&group].len() >= 2;
        let ambiguous = chosen.iter().any(|&source_index| {
            tied(source_groups[source_index]) && uses[&source_index] >= 2
        });

        let result = plan.apply(&source, &target);
        if ambiguous && !untie {
            proptest::prop_assert!(result.is_err());
            return Ok(());
        }
        let (actual, _) = result.unwrap();
        for (left, &left_source) in chosen.iter().enumerate() {
            proptest::prop_assert_eq!(
                values(actual.weights[left].val()),
                vec![10.0 + source_groups[left_source] as f32; 2]
            );
            for (right, &right_source) in chosen.iter().enumerate() {
                let group = source_groups[left_source];
                let shared = left == right
                    || (!untie && group == source_groups[right_source] && tied(group));
                proptest::prop_assert_eq!(
                    actual.weights[left].id == actual.weights[right].id,
                    shared
                );
            }
        }
    }
}

#[cfg(feature = "autodiff")]
mod fan_out_training {
    //! Fan-out copies used as separate layers receive separate optimizer updates.
    //!
    //! Copying one source parameter into two target layers must give each copy
    //! its own parameter identity: one SGD step on the Autodiff backend then
    //! moves each copy by its own gradient, so the copies diverge.

    use burn::{
        backend::Autodiff,
        optim::{GradientsParams, Optimizer, SgdConfig},
    };

    use super::*;

    /// Differentiable CPU backend for optimizer-ownership checks.
    type Ad = Autodiff<B>;

    /// A two-layer stack of vectors, each scaled differently by the loss.
    #[derive(Module, Debug)]
    struct Stack<B: Backend> {
        /// Layer parameters addressed as `layers.<index>`.
        layers: Vec<Param<Tensor<B, 1>>>,
    }

    /// Builds independent constant layers on the differentiable backend.
    fn stack(count: usize, value: f32) -> Stack<Ad> {
        let device = Default::default();
        Stack {
            layers: (0..count)
                .map(|_| Param::from_tensor(Tensor::full([2], value, &device)))
                .collect(),
        }
    }

    /// Duplicated layers pass the optimizer-ownership preflight and both train.
    ///
    /// With loss `sum(l0) + 2 sum(l1)` and SGD at rate 0.1, the copies must move
    /// from 1.0 to 0.9 and 0.8. A shared identity would leave one copy stale.
    #[test]
    fn duplicated_layers_each_receive_their_own_gradient_step() {
        let source = stack(1, 1.0);
        let target = stack(2, 0.0);
        let (model, _) = TransferPlan::new()
            .copy("layers.0", "layers.0")
            .copy("layers.0", "layers.1")
            .apply(&source, &target)
            .unwrap();
        crate::training::validate_trainable_module(&model)
            .expect("fan-out copies must be separately owned trainable parameters");
        let loss = model.layers[0].val().sum() + model.layers[1].val().sum() * 2.0;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let model = SgdConfig::new().init().step(0.1, model, grads);
        let read = |param: &Param<Tensor<Ad, 1>>| {
            param
                .val()
                .into_data()
                .to_vec::<f32>()
                .expect("dense F32 layer")
        };
        let first = read(&model.layers[0]);
        let second = read(&model.layers[1]);
        assert!(
            first.iter().all(|value| (value - 0.9).abs() < 1e-6),
            "{first:?}"
        );
        assert!(
            second.iter().all(|value| (value - 0.8).abs() < 1e-6),
            "{second:?}"
        );
    }
}

/// Builds a Q4-packed `[2, 32]` weight whose dense values are `i / 8 - 4`.
fn packed_weight() -> Param<Tensor<B, 2>> {
    use crate::checkpoint::quantization::{
        QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
        quantize_tensor_data,
    };
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
    Param::from_tensor(Tensor::<B, 2>::from_data(packed_data, &Default::default()))
}

/// A target with packed weights still receives float-only transfers into its dense parameters.
#[test]
fn packed_targets_accept_transfers_into_their_dense_parameters_only() {
    let source = block(32, 7.0);
    let mut target = block(32, 2.0);
    target.weight = packed_weight();
    let original_packed = target.weight.val().into_data();
    let (actual, report) = TransferPlan::new()
        .copy("bias", "bias")
        .apply(&source, &target)
        .expect("an unselected packed target must not block a dense transfer");
    assert_eq!(values(actual.bias.val()), vec![7.0; 32]);
    assert_eq!(actual.weight.val().into_data(), original_packed);
    assert_eq!(actual.weight.id, target.weight.id);
    assert_eq!(report.newly_initialized[0].name, "weight");

    let error = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .expect_err("replacing a packed target needs an explicit conversion");
    let message = error.to_string();
    assert!(message.contains("`weight` -> `weight`"), "{message}");
    assert!(message.contains("dequantize explicitly"), "{message}");
    assert_eq!(target.weight.val().into_data(), original_packed);
}

/// Hand-written module whose `visit` and `map` traversals can deliberately disagree.
///
/// Burn's derive keeps both traversals consistent; a hand-written module can
/// break that contract. Transfer must then fail closed rather than install a
/// replacement at the wrong rank or silently skip a planned parameter.
#[derive(Clone, Debug)]
struct Inconsistent<B: Backend> {
    /// The `[2, 2]` parameter that `visit` reports at path `w`.
    matrix: Param<Tensor<B, 2>>,
    /// The `[4]` parameter that `map` offers at path `w` under [`Defect::RankShift`].
    flat: Param<Tensor<B, 1>>,
    /// Which traversal contract, if any, this fixture breaks.
    defect: Defect,
}

/// Traversal contracts a hand-written module can honor or break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Defect {
    /// `visit` and `map` both traverse `w` once at rank 2.
    Consistent,
    /// `visit` reports `w` three times, so parameter paths collide.
    DuplicatePath,
    /// `map` offers a rank-1 parameter where `visit` reported a rank-2 one.
    RankShift,
    /// `map` never offers `w` although `visit` reported it.
    SkippedParameter,
}

impl<B: Backend> Module<B> for Inconsistent<B> {
    /// Stateless record: the fixture is never serialized.
    type Record = ();

    /// Reports no devices; the fixture is never moved.
    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    /// Returns the fixture unchanged; it is never forked.
    fn fork(self, _: &B::Device) -> Self {
        self
    }

    /// Returns the fixture unchanged; it is never moved.
    fn to_device(self, _: &B::Device) -> Self {
        self
    }

    /// Reports `w` at rank 2, three times under [`Defect::DuplicatePath`].
    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        let repeats = if self.defect == Defect::DuplicatePath {
            3
        } else {
            1
        };
        for _ in 0..repeats {
            visitor.enter_module("w", "Inconsistent");
            visitor.visit_float(&self.matrix);
            visitor.exit_module("w", "Inconsistent");
        }
    }

    /// Offers `w` at rank 2, at rank 1, or not at all, according to the defect.
    fn map<M: burn::module::ModuleMapper<B>>(mut self, mapper: &mut M) -> Self {
        match self.defect {
            Defect::Consistent | Defect::DuplicatePath => {
                mapper.enter_module("w", "Inconsistent");
                self.matrix = mapper.map_float(self.matrix);
                mapper.exit_module("w", "Inconsistent");
            }
            Defect::RankShift => {
                mapper.enter_module("w", "Inconsistent");
                self.flat = mapper.map_float(self.flat);
                mapper.exit_module("w", "Inconsistent");
            }
            Defect::SkippedParameter => {}
        }
        self
    }

    /// Ignores the empty record.
    fn load_record(self, _: Self::Record) -> Self {
        self
    }

    /// Produces the empty record.
    fn into_record(self) -> Self::Record {}
}

/// Builds an [`Inconsistent`] fixture whose parameters hold `value`.
fn inconsistent(defect: Defect, value: f32) -> Inconsistent<B> {
    let device = Default::default();
    Inconsistent {
        matrix: Param::from_tensor(Tensor::full([2, 2], value, &device)),
        flat: Param::from_tensor(Tensor::full([4], value, &device)),
        defect,
    }
}

/// Colliding parameter paths are rejected on either side before any value is staged.
#[test]
fn duplicate_module_parameter_paths_are_rejected_in_sources_and_targets() {
    let honest = inconsistent(Defect::Consistent, 2.0);
    let duplicated = inconsistent(Defect::DuplicatePath, 7.0);
    for (source, target) in [(&duplicated, &honest), (&honest, &duplicated)] {
        let error = TransferPlan::new()
            .copy("w", "w")
            .apply(source, target)
            .unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, reason }
                if field == "w" && reason == "duplicate module parameter path"),
            "{error}"
        );
    }
}

/// A target whose `map` disagrees with its `visit` fails instead of returning a partial transfer.
///
/// Validation plans against `visit`; installation runs through `map`. When
/// `map` offers the planned parameter at another rank, the error names that
/// parameter; when it never offers it, the error names the target module.
/// Either way `apply` returns only the error, never the staged clone.
#[test]
fn inconsistent_target_traversals_fail_closed() {
    let source = inconsistent(Defect::Consistent, 7.0);
    for (defect, expected_field, expected_reason) in [
        (
            Defect::RankShift,
            "w",
            "module mapper rank changed after transfer validation",
        ),
        (
            Defect::SkippedParameter,
            "transfer.target",
            "module mapper did not visit all planned parameters",
        ),
    ] {
        let target = inconsistent(defect, 2.0);
        let error = TransferPlan::new()
            .copy("w", "w")
            .apply(&source, &target)
            .unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, reason }
                if field == expected_field && reason == expected_reason),
            "{defect:?}: {error}"
        );
    }
    let (copied, report) = TransferPlan::new()
        .copy("w", "w")
        .apply(&source, &inconsistent(Defect::Consistent, 2.0))
        .unwrap();
    assert_eq!(values(copied.matrix.val()), vec![7.0; 4]);
    assert_eq!(report.reused.len(), 1);
}

/// A bare parameter is its own module root and is addressed by the path `$`.
#[test]
fn a_root_parameter_is_addressed_as_dollar() {
    let device = Default::default();
    let source = Param::from_tensor(Tensor::<B, 1>::full([3], 7.0, &device));
    let target = Param::from_tensor(Tensor::<B, 1>::full([3], 2.0, &device));
    let (copied, report) = TransferPlan::new()
        .copy("$", "$")
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(copied.val()), vec![7.0; 3]);
    assert_eq!(report.reused.len(), 1);
    let error = TransferPlan::new()
        .copy("weight", "$")
        .apply(&source, &target)
        .unwrap_err();
    assert!(error.to_string().contains("`weight` -> `$`"), "{error}");
}

/// Exact and prefix rules name missing or empty paths before any transfer work.
#[test]
fn mapping_rules_require_existing_nonempty_paths_on_both_sides() {
    let source = experiment(&[2], 7.0);
    let target = experiment(&[2], 2.0);
    let error = TransferPlan::new()
        .copy("head.weight", "head.missing")
        .apply(&source, &target)
        .unwrap_err();
    assert!(
        error.to_string().contains(
            "exact mapping `head.weight` -> `head.missing` requires both parameter paths to exist"
        ),
        "{error}"
    );
    for (from, to) in [("", "encoder"), ("encoder", ""), ("encoder", "missing")] {
        let error = TransferPlan::new()
            .copy_prefix(from, to)
            .apply(&source, &target)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires nonempty existing module paths"),
            "{from:?} -> {to:?}: {error}"
        );
    }
}

/// A prefix equal to a complete parameter path selects exactly that parameter.
#[test]
fn a_prefix_naming_a_whole_parameter_copies_only_that_parameter() {
    let source = experiment(&[2], 7.0);
    let target = experiment(&[2], 2.0);
    let (copied, report) = TransferPlan::new()
        .copy_prefix("encoder.0.weight", "encoder.0.weight")
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(copied.encoder[0].weight.val()), vec![7.0; 4]);
    assert_eq!(values(copied.encoder[0].bias.val()), vec![2.0; 2]);
    assert_eq!(report.reused.len(), 1);
    assert_eq!(report.reused[0].target, "encoder.0.weight");
}

/// `copy_matching` leaves target-only paths initialized and reports them as unselected.
#[test]
fn copy_matching_retains_target_only_parameters() {
    let source = experiment(&[2], 7.0);
    let target = experiment(&[2, 3], 2.0);
    let (copied, report) = TransferPlan::new()
        .copy_matching()
        .apply(&source, &target)
        .unwrap();
    assert_eq!(values(copied.encoder[0].weight.val()), vec![7.0; 4]);
    assert_eq!(values(copied.encoder[1].weight.val()), vec![2.0; 6]);
    let retained = report
        .newly_initialized
        .iter()
        .map(|disposition| disposition.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(retained, ["encoder.1.bias", "encoder.1.weight"]);
}

/// A transform to a rank-zero output is rejected before its callback runs.
#[test]
fn rank_zero_transform_outputs_are_rejected_before_the_callback() {
    let source = experiment(&[2], 7.0);
    let target = experiment(&[2], 2.0);
    let called = std::cell::Cell::new(false);
    let error = TransferPlan::new()
        .transform(
            "head.bias",
            "head.bias",
            "collapse",
            |_: Tensor<B, 1>| -> Result<Tensor<B, 0>> {
                called.set(true);
                Err(Error::configuration("test", "the callback must not run"))
            },
        )
        .apply(&source, &target)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains(
                "collapse: invalid configuration at transfer.transform: parameter ranks must be positive"
            ),
        "{error}"
    );
    assert!(!called.get());
}
