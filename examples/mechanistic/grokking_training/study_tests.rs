//! Grokking training and checkpoint regressions.

use super::*;
use burn::{
    backend::{Autodiff, Flex},
    nn::loss::CrossEntropyLossConfig,
    optim::GradientsParams,
    tensor::ElementConversion,
};
use transformers::intervention::Hooks;
type TestBackend = Autodiff<Flex>;

#[test]
fn native_optimizer_discovers_every_parameter_and_updates_the_transformer() {
    let device = Default::default();
    TestBackend::seed(&device, 42);
    let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
    assert_eq!(model.num_params(), 226816);
    let (input, targets) = batch::<TestBackend>(&[0, 2, 113, 791, 4001, 12211], &device);
    let mut hooks = Hooks::new();
    let captured = hooks.capture_with_grad(crate::model::mlp_activation());
    let logits = model.forward_with_hooks(input.clone(), &mut hooks).unwrap();
    let baseline = logits.clone().into_data().to_vec::<f32>().unwrap();
    let ce = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, targets);
    let loss = ce
        + hooks
            .get(&captured)
            .unwrap()
            .square()
            .mean()
            .mul_scalar(0.001);
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    assert_eq!(
        grads.len(),
        11,
        "embedding, positions, four attention matrices, four MLP tensors, readout"
    );
    let mut optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(1.0)
        .init();
    let model = optimizer.step(1e-3, model, grads);
    let after = model
        .forward(input)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(after.iter().all(|value| value.is_finite()));
    assert!(
        baseline
            .iter()
            .zip(after)
            .any(|(before, after)| (before - after).abs() > 1e-5)
    );
}

#[test]
fn connected_activation_loss_reaches_the_prefix_without_the_readout() {
    let device = Default::default();
    TestBackend::seed(&device, 9);
    let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
    let (input, _) = batch::<TestBackend>(&[1, 7, 113, 902], &device);
    let mut hooks = Hooks::new();
    let captured = hooks.capture_with_grad(crate::model::mlp_activation());
    let _ = model.forward_with_hooks(input, &mut hooks).unwrap();
    let auxiliary = hooks.get(&captured).unwrap().square().mean();
    let gradients = GradientsParams::from_grads(auxiliary.backward(), &model);
    assert_eq!(
        gradients.len(),
        8,
        "an intermediate loss reaches embedding, positions, attention and MLP input; the downstream MLP output/readout are outside its graph"
    );
}

#[test]
fn burn_model_and_optimizer_records_resume_the_next_update_exactly() {
    let device = Default::default();
    TestBackend::seed(&device, 7);
    let model = GrokkingTransformer::<TestBackend>::init_random(&device).unwrap();
    let (input, targets) = batch::<TestBackend>(&[1, 40, 700, 5555], &device);
    let loss_function = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamWConfig::new()
        .with_beta_2(0.98)
        .with_epsilon(1e-8)
        .with_weight_decay(1.0)
        .init();
    let loss = loss_function.forward(model.forward(input.clone()).unwrap(), targets.clone());
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let model = optimizer.step(1e-3, model, grads);
    let dir = tempfile::tempdir().unwrap();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(model.clone().into_record(), dir.path().join("model"))
        .unwrap();
    recorder
        .record(optimizer.to_record(), dir.path().join("optimizer"))
        .unwrap();
    let restored = GrokkingTransformer::<TestBackend>::init_random(&device)
        .unwrap()
        .load_record(recorder.load(dir.path().join("model"), &device).unwrap());
    let mut restored_optimizer = optimizer.clone().load_record(
        recorder
            .load(dir.path().join("optimizer"), &device)
            .unwrap(),
    );
    let loss_a = loss_function.forward(model.forward(input.clone()).unwrap(), targets.clone());
    let loss_b = loss_function.forward(restored.forward(input.clone()).unwrap(), targets);
    assert_eq!(
        loss_a.clone().into_scalar().elem::<f32>(),
        loss_b.clone().into_scalar().elem::<f32>()
    );
    let grads_a = GradientsParams::from_grads(loss_a.backward(), &model);
    let grads_b = GradientsParams::from_grads(loss_b.backward(), &restored);
    let a = optimizer.step(1e-3, model, grads_a);
    let b = restored_optimizer.step(1e-3, restored, grads_b);
    assert_eq!(
        a.forward(input.clone())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        b.forward(input)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
fn resume_rejects_earlier_targets_benchmarks_and_changed_training_sources() {
    let config = Settings {
        backend: "flex".into(),
        steps: 20,
        seed: 0,
        output: "unused".into(),
        resume: true,
        benchmark: false,
        analyze_only: false,
    };
    let mut checkpoint = Checkpoint {
        step: 10,
        directory: "unused".into(),
        settings: config.clone(),
        observations: vec![],
        train_split_sha256: String::new(),
    };
    let mut provenance = serde_json::json!({
        "protocol_sha256": format!("{:x}", Sha256::digest(include_bytes!("manifest.json"))),
        "source_sha256": source_fingerprints()
    });
    assert!(validate_resume(&config, &checkpoint, &provenance).is_ok());
    let mut earlier = config.clone();
    earlier.steps = 9;
    assert!(
        validate_resume(&earlier, &checkpoint, &provenance)
            .unwrap_err()
            .to_string()
            .contains("below saved")
    );
    checkpoint.settings.benchmark = true;
    assert!(
        validate_resume(&config, &checkpoint, &provenance)
            .unwrap_err()
            .to_string()
            .contains("timing probes")
    );
    checkpoint.settings.benchmark = false;
    provenance["source_sha256"]["model"] = serde_json::json!("changed");
    assert!(
        validate_resume(&config, &checkpoint, &provenance)
            .unwrap_err()
            .to_string()
            .contains("training sources")
    );
    provenance["source_sha256"] = source_fingerprints();
    provenance["protocol_sha256"] = serde_json::json!("changed");
    assert!(
        validate_resume(&config, &checkpoint, &provenance)
            .unwrap_err()
            .to_string()
            .contains("protocol changed")
    );
}

#[test]
fn fourier_power_identifies_a_known_frequency_and_ignores_dc() {
    let weights: Vec<f32> = (0..P)
        .map(|token| (3.0 + (std::f64::consts::TAU * 7.0 * token as f64 / P as f64).cos()) as f32)
        .collect();
    let powers = fourier_power(&weights, 1, false);
    assert!(powers[6] > 1.0 - 1e-12);
    assert!((powers.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
