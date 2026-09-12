//! Clock/Pizza split and differentiable-training regressions.

use super::*;
use burn::{
    backend::{Autodiff, Flex},
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
};
/// Checks a seeded 2784/697 split repeats exactly and partitions every modular-addition pair once.
#[test]
fn split_is_deterministic_disjoint_and_complete() {
    let data = Dataset::new(20260910);
    assert_eq!(data.train.len(), 2784);
    assert_eq!(data.test.len(), 697);
    assert_eq!(data.train, Dataset::new(20260910).train);
    let mut all = data.train.clone();
    all.extend(data.test);
    all.sort();
    assert_eq!(all, (0..P * P).collect::<Vec<_>>());
}
/// Checks both attention-mixture endpoints expose trainable parameters and connected captures while one update lowers loss.
#[test]
fn optimizer_discovers_recipe_parameters_and_hooks_preserve_learning() {
    /// Autodifferentiable CPU backend for optimizer and activation-gradient checks.
    type Ad = Autodiff<Flex>;
    let device = Default::default();
    Ad::seed(&device, 42);
    for alpha in [0., 1.] {
        let model = Transformer::<Ad>::init_random(alpha, &device).unwrap();
        assert_eq!(model.num_params(), 212608);
        let diagnostic = gradient_symmetry(&model).unwrap();
        assert_eq!(
            diagnostic["individual_cosines"].as_array().unwrap().len(),
            100
        );
        let (inputs, labels) = Dataset::new(42).training_batch::<Ad>(&device, true);
        let mut hooks = Hooks::new();
        let captured = hooks.capture_with_grad(Layer::decoder(0).mlp_activation());
        hooks.edit(Layer::decoder(0).mlp_output(), |hidden| hidden);
        let logits = model
            .forward_ids_with_hooks(inputs.clone(), &mut hooks)
            .unwrap();
        let auxiliary = hooks.get(&captured).unwrap().square().mean();
        let auxiliary_gradients = auxiliary.backward();
        let embedding_gradient = model
            .embedding
            .val()
            .grad(&auxiliary_gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(embedding_gradient.iter().all(|v| v.is_finite()));
        assert!(embedding_gradient.iter().any(|v| v.abs() > 1e-8));
        // A new graph is needed after backward has consumed the first one.
        drop(logits);
        let logits = model
            .forward_ids_with_hooks(inputs.clone(), &mut hooks)
            .unwrap();
        let loss = transformers::training::cross_entropy(logits, labels.clone()).unwrap();
        let before = loss.clone().into_scalar();
        let gradients = loss.backward();
        let embedding_before = model.embedding.val().into_data().to_vec::<f32>().unwrap();
        let gradients = GradientsParams::from_grads(gradients, &model);
        let mut optimizer = AdamWConfig::new()
            .with_beta_2(0.98)
            .with_epsilon(1e-8)
            .with_weight_decay(2.)
            .init();
        let trained = optimizer.step(0.001, model, gradients);
        assert_ne!(
            embedding_before,
            trained.embedding.val().into_data().to_vec::<f32>().unwrap()
        );
        let (inputs, labels) = Dataset::new(42).training_batch::<Flex>(&device, true);
        let logits = trained
            .valid()
            .forward_ids_with_hooks(inputs, &mut Hooks::new())
            .unwrap();
        let after = transformers::training::cross_entropy(logits, labels)
            .unwrap()
            .into_scalar();
        assert!(
            after < before,
            "alpha={alpha}: optimizer should lower tiny-batch CE {before}->{after}"
        );
    }
}
