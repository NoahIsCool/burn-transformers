//! Isolate learned embedding circles in the released Clock and Pizza models.
//!
//! Prepare: python3 examples/mechanistic/clock_and_pizza/prepare.py
//! Run: cargo run --release --locked --offline --example clock_and_pizza
//! See examples/mechanistic/clock_and_pizza/README.md for the exact paper protocol.
//!
//! This checkpoint analysis projects centered F32 token embeddings into PCA
//! subspaces before adding positions. Full-grid logits `[59,59,59]` are checked
//! against independently prepared reference tensors. Energy-matched random
//! subspaces, ordered removal/restoration, and attention normalization provide
//! controls; training new weights is a separate `clock_and_pizza_train` example.

#[cfg(not(feature = "flex"))]
/// Explain the CPU backend feature required to replay these reference fixtures.
fn main() {
    eprintln!("enable the `flex` feature to run this example");
    std::process::exit(1);
}

#[cfg(feature = "flex")]
#[path = "mechanistic/clock_and_pizza/measure.rs"]
mod measure;
#[cfg(feature = "flex")]
#[path = "mechanistic/clock_and_pizza/model.rs"]
mod model;

#[cfg(feature = "flex")]
/// Replay both pinned models and compare PCA interventions with paper/reference metrics.
/// Save current replay measurements beside the prepared fixtures; keep published
/// results and the documented checkpoint/paper accuracy discrepancy distinct.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::{Distribution, Tensor, backend::Backend};
    use model::{B, Fixture, HIDDEN, P, token_embedding};
    use transformers::intervention::{Hooks, Layer};

    let directory = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "target/mechanistic/clock_and_pizza".into()),
    );
    let mut results = serde_json::Map::new();
    for name in ["pizza", "clock"] {
        let fixture = Fixture::load(&directory, name)?;
        let model = &fixture.model;
        let mean = fixture.mean.clone().unsqueeze::<3>();
        println!(
            "\n{name}: attention rate alpha={}, all {} input pairs",
            model.alpha,
            P * P
        );
        println!(
            "{:28} {:>11} {:>13}",
            "embedding intervention", "accuracy", "distance q"
        );

        // Observe raw token embeddings. Positional embeddings are added later;
        // projecting this point changes exactly the embedding table in the paper.
        let baseline = run(model, &mut Hooks::new())?;
        let mut reference_error =
            measure::assert_reference(&baseline, &fixture.references["baseline"]);
        let mut observe = Hooks::new();
        let tokens = observe.capture(token_embedding());
        let weights = observe.capture(Layer::decoder(0).self_attention().pattern().batch(0));
        let observed = run(model, &mut observe)?;
        let no_op_error = measure::max_error(&baseline, &observed);
        assert!(no_op_error < 1e-5, "observing must preserve the graph");
        let original_tokens = observe.get(&tokens)?;
        let pattern = observe.get(&weights)?;
        let expected_row_sum = 2.0 - model.alpha as f32;
        for sum in pattern.sum_dim(3).into_data().to_vec::<f32>()? {
            assert!(
                (sum - expected_row_sum).abs() < 1e-5,
                "J contains ones, not probabilities"
            );
        }

        let mut rows = vec![row("baseline", &baseline)];
        let mut circle_fve = Vec::new();
        // These published ranges have fixed reference logits. For another
        // subspace, reuse this edit in a separate Hooks collection without that
        // fixed-reference assertion. PCA reconstruction includes its mean.
        for (label, first, width, frequency) in [
            ("first2", 0, 2, Some(17)),
            ("circle2", 2, 2, Some(3)),
            ("circle3", 4, 2, Some(44)),
            ("first6", 0, 6, None),
        ] {
            let basis = fixture.components.clone().narrow(1, first, width);
            let mut isolate = Hooks::new();
            isolate.edit(token_embedding(), |tokens| {
                (tokens - mean.clone())
                    .matmul(basis.clone().unsqueeze())
                    .matmul(basis.clone().transpose().unsqueeze())
                    + mean.clone()
            });
            let logits = run(model, &mut isolate)?;
            reference_error = reference_error.max(measure::assert_reference(
                &logits,
                &fixture.references[label],
            ));
            rows.push(row(label, &logits));
            if name == "pizza"
                && let Some(frequency) = frequency
            {
                circle_fve.push((
                    measure::formula_fve(&logits, frequency, false),
                    measure::formula_fve(&logits, frequency, true),
                ));
            }
        }

        // A matched control keeps an unrelated rank-two subspace. Match the
        // retained embedding energy so loss of magnitude is not the only change.
        B::seed(&model.embedding.device(), 1729);
        let random = Tensor::<B, 2>::random(
            [HIDDEN, 2],
            Distribution::Normal(0.0, 1.0),
            &model.embedding.device(),
        );
        let first = random.clone().narrow(1, 0, 1);
        let first = first.clone() / first.square().sum().sqrt().into_scalar();
        let second = random.narrow(1, 1, 1);
        let second = second.clone()
            - first
                .clone()
                .matmul(first.clone().transpose().matmul(second));
        let second = second.clone() / second.square().sum().sqrt().into_scalar();
        let random_basis = Tensor::cat(vec![first, second], 1);
        let centered = model.embedding.val() - fixture.mean.clone();
        let signal_energy = centered
            .clone()
            .matmul(fixture.components.clone().narrow(1, 0, 2))
            .square()
            .sum()
            .into_scalar();
        let random_energy = centered
            .matmul(random_basis.clone())
            .square()
            .sum()
            .into_scalar();
        let scale = (signal_energy / random_energy).sqrt();
        let mut random_control = Hooks::new();
        random_control.edit(token_embedding(), |tokens| {
            (tokens - mean.clone())
                .matmul(random_basis.clone().unsqueeze())
                .matmul(random_basis.clone().transpose().unsqueeze())
                * scale
                + mean.clone()
        });
        let random_logits = run(model, &mut random_control)?;
        rows.push(row("random2 (matched energy)", &random_logits));

        // Remove the first circle, then restore the captured clean activations.
        // Ordered public actions make the source and destination explicit.
        let basis = fixture.components.clone().narrow(1, 0, 2);
        let mut remove = Hooks::new();
        remove.edit(token_embedding(), |tokens| {
            tokens.clone()
                - (tokens - mean.clone())
                    .matmul(basis.clone().unsqueeze())
                    .matmul(basis.clone().transpose().unsqueeze())
        });
        let removed = run(model, &mut remove)?;
        rows.push(row("remove first circle", &removed));
        remove.replace(token_embedding(), original_tokens);
        let restored = run(model, &mut remove)?;
        let restoration_error = measure::max_error(&baseline, &restored);
        assert!(
            restoration_error < 1e-5,
            "restoration must recover full logits"
        );
        rows.push(row("restore clean embeddings", &restored));

        // The distinction between J and uniform attention is consequential.
        // This intentionally different rule is a control, not a paper reproduction.
        let mut uniform_control = Hooks::new();
        uniform_control.edit(Layer::decoder(0).self_attention().pattern(), |weights| {
            weights / expected_row_sum
        });
        let uniform_logits = run(model, &mut uniform_control)?;
        let uniform_logit_change = measure::max_error(&baseline, &uniform_logits);
        if name == "pizza" {
            assert!(uniform_logit_change > 0.1);
            rows.push(row("normalize J (control)", &uniform_logits));
        }

        let q = measure::distance_irrelevance(&baseline);
        let expected_q = if name == "pizza" { 0.17 } else { 0.85 };
        assert!(
            (q - expected_q).abs() < 0.01,
            "published rounded distance irrelevance"
        );
        for (circle, &(clock, pizza)) in circle_fve.iter().enumerate() {
            let published = [(0.7541, 0.9918), (0.7562, 0.9918), (0.7538, 0.9928)][circle];
            assert!(
                (clock - published.0).abs() < 0.0002 && (pizza - published.1).abs() < 0.0002,
                "paper Table 2 must reproduce within 0.02 percentage points"
            );
            println!(
                "Table 2 circle {}: Clock FVE {:.4}%, Pizza FVE {:.4}%",
                circle + 1,
                clock * 100.,
                pizza * 100.
            );
        }
        println!(
            "reference max error {reference_error:.2e}; observation {no_op_error:.2e}; restoration {restoration_error:.2e}"
        );
        results.insert(
            name.into(),
            serde_json::json!({
                "rows":rows, "circle_fve_clock_pizza":circle_fve,
                "reference_max_error":reference_error, "observation_max_error":no_op_error,
                "restoration_max_error":restoration_error, "attention_row_sum":expected_row_sum,
                "normalized_pattern_logit_change":uniform_logit_change,
                "random_control_seed":1729, "random_control_scale":scale,
            }),
        );
    }
    println!(
        "\nAccuracy caveat: the public Pizza checkpoint's first six PCs reproduce the paper's later 99.7%, not section 3.3's 91.4%; see the protocol note."
    );
    std::fs::write(
        directory.join("rust_results.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    Ok(())
}

#[cfg(feature = "flex")]
/// Evaluate all 59² input pairs using the supplied hooks and return flat F32 logits.
/// A single complete-grid call keeps capture and restoration row alignment explicit.
fn run(
    model: &model::Transformer,
    hooks: &mut transformers::intervention::Hooks<'_, model::B>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(model
        .forward_with_hooks(0, model::P * model::P, hooks)?
        .into_data()
        .to_vec::<f32>()?)
}

#[cfg(feature = "flex")]
/// Print and serialize complete-grid accuracy and distance irrelevance for one condition.
fn row(label: &str, logits: &[f32]) -> serde_json::Value {
    let accuracy = measure::accuracy(logits);
    let q = measure::distance_irrelevance(logits);
    println!("{label:28} {:10.4}% {:13.6}", accuracy * 100., q);
    serde_json::json!({"intervention":label, "accuracy":accuracy, "distance_irrelevance_q":q})
}
