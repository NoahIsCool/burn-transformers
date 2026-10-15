//! Decode intermediate Pythia states with the authors' released affine translators.
//!
//! Prepare once: python3 examples/mechanistic/tuned_lens/prepare.py
//! Reference: python3 examples/mechanistic/tuned_lens/reference.py
//! Run: cargo run --release --locked --features tokenizers --example tuned_lens
//! The short held-out fixture is an adaptation; no model or lens is trained.
//!
//! The raw lens projects a residual h through the frozen final LayerNorm and
//! vocabulary readout. The tuned lens first applies h + Wh + b; translator zero
//! sees embeddings and translator i sees the residual after block i - 1.
//! Activations are `[batch, token, hidden]` and logits `[batch, token, vocabulary]`.
//! Causal, unpadded F32 forwards use detached observations. Full-vocabulary
//! `KL(final || lens)` and shifted next-token CE measure predictive quality;
//! changing this decoder alone does not establish a causal effect on the model.

#[cfg(not(all(feature = "flex", feature = "tokenizers")))]
/// Explain the required Cargo features and exit before accessing local artifacts.
fn main() {
    eprintln!("enable the `flex` and `tokenizers` features to run this example");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "tokenizers"))]
#[path = "mechanistic/tuned_lens/study.rs"]
mod study;

#[cfg(all(feature = "flex", feature = "tokenizers"))]
/// Run the fixed local experiment, retaining measurements and numerical controls.
fn main() -> study::Result<()> {
    use burn::{module::Initializer, nn::LinearConfig};
    use std::{path::PathBuf, time::Instant};
    use study::{B, Experiment, LayerMetrics, log_probabilities, losses, max_error};
    use transformers::{
        ForwardOptions, LogitsSelection,
        intervention::{Hooks, Layer, Stack},
    };

    // Change the inspected translator here. Translator 0 sees embeddings;
    // translator i>0 sees the residual after block i-1, before its next block.
    /// Index of the released translator replaced by a zero correction for the identity control.
    const REPLACE_TRANSLATOR: usize = 6;
    /// Number of positions from the first document shown in the prediction visualization.
    const DISPLAY_TOKENS: usize = 8;
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/mechanistic/tuned_lens"));
    let started = Instant::now();
    if !root.join("preparation.json").is_file() {
        return Err(format!(
            "{}: no prepared artifacts; run python3 examples/mechanistic/tuned_lens/prepare.py, \
             or pass the prepared directory as the first argument",
            root.display()
        )
        .into());
    }
    let (study, fixture, preparation) = Experiment::load(&root)?;
    let (reference, oracle) = study::load_reference(&root, &preparation)?;
    if REPLACE_TRANSLATOR >= study.translators.len()
        || DISPLAY_TOKENS > fixture.records[0].input_ids.len()
    {
        return Err(
            "choose an existing translator and a display window within the first record".into(),
        );
    }
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("model/tokenizer.json"))
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let stages = std::iter::once(Stack::Decoder.embedding())
        .chain((0..11).map(|i| Layer::decoder(i).residual_post()))
        .collect::<Vec<_>>();
    let mut metrics = (0..12)
        .map(|index| LayerMetrics {
            index,
            point: stages[index].info().name,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let (mut input_tokens, mut ce_tokens, mut final_ce) = (0_usize, 0_usize, 0.0);
    let (mut parity_error, mut observation_error, mut replacement_error) =
        (0.0_f32, 0.0_f32, 0.0_f32);
    let mut table_rows = Vec::new();
    let mut table_labels = Vec::new();

    for (record_index, record) in fixture.records.iter().enumerate() {
        // Request only the residual states the lens needs. Captures stay on
        // device, and observing them leaves the backbone computation unchanged.
        let mut hooks = Hooks::new();
        let saved = stages
            .iter()
            .map(|point| hooks.capture(point))
            .collect::<Vec<_>>();
        let output = study
            .backbone
            .forward_with_hooks(record.input(), &options, &mut hooks)?;
        let final_log_probs = log_probabilities(output.logits.clone())?;
        let tokens = record.input_ids.len();
        final_ce += losses(
            final_log_probs.clone(),
            final_log_probs.clone(),
            &record.input_ids,
        )?
        .1;
        input_tokens += tokens;
        ce_tokens += tokens - 1;
        if record_index == 0 {
            let ordinary = study.backbone.forward(record.input(), &options)?;
            observation_error = max_error(output.logits.clone(), ordinary.logits)?;
            parity_error = max_error(
                output.logits.narrow(1, 0, 8),
                oracle.tensor("reference.final", [1, 8, 50304])?,
            )?;
            table_labels = record.input_ids[tokens - DISPLAY_TOKENS..]
                .iter()
                .map(|&id| {
                    tokenizer
                        .decode(&[id as u32], false)
                        .map_err(|error| std::io::Error::other(error.to_string()))
                })
                .collect::<std::io::Result<_>>()?;
            table_rows.push((
                "Final model".into(),
                study::prediction_cells(
                    final_log_probs.clone(),
                    tokens - DISPLAY_TOKENS,
                    DISPLAY_TOKENS,
                    &tokenizer,
                )?,
            ));
        }

        for (index, handle) in saved.iter().enumerate() {
            let hidden = hooks.get(handle)?; // [batch=1, token, residual channel=768]
            let raw_logits = study.backbone.project_hidden(hidden.clone())?;

            // The saved Linear is a correction to identity. Apply the small
            // auxiliary model, then the pretrained final LayerNorm and readout.
            let translated = hidden.clone() + study.translators[index].forward(hidden.clone());
            let tuned_logits = study.backbone.project_hidden(translated)?;
            let raw = log_probabilities(raw_logits.clone())?;
            let tuned = log_probabilities(tuned_logits.clone())?;
            let (raw_kl, raw_ce) = losses(raw.clone(), final_log_probs.clone(), &record.input_ids)?;
            let (tuned_kl, tuned_ce) =
                losses(tuned.clone(), final_log_probs.clone(), &record.input_ids)?;
            metrics[index].raw_kl += raw_kl;
            metrics[index].raw_ce += raw_ce;
            metrics[index].tuned_kl += tuned_kl;
            metrics[index].tuned_ce += tuned_ce;

            if record_index == 0 {
                for (kind, logits) in [("raw", raw_logits.clone()), ("tuned", tuned_logits)] {
                    parity_error = parity_error.max(max_error(
                        logits.narrow(1, 0, 8),
                        oracle.tensor(&format!("reference.{kind}.{index}"), [1, 8, 50304])?,
                    )?);
                }
                let label = if index == 0 {
                    "Embedding".into()
                } else {
                    format!("After block {}", index - 1)
                };
                table_rows.push((
                    format!("{label}: raw"),
                    study::prediction_cells(
                        raw,
                        tokens - DISPLAY_TOKENS,
                        DISPLAY_TOKENS,
                        &tokenizer,
                    )?,
                ));
                table_rows.push((
                    format!("{label}: tuned"),
                    study::prediction_cells(
                        tuned,
                        tokens - DISPLAY_TOKENS,
                        DISPLAY_TOKENS,
                        &tokenizer,
                    )?,
                ));
                if index == REPLACE_TRANSLATOR {
                    // Swap in another ordinary module. A zero correction gives
                    // the raw logit lens exactly; it is a useful identity control.
                    let replacement = LinearConfig::new(768, 768)
                        .with_initializer(Initializer::Zeros)
                        .init::<B>(&Default::default());
                    let replaced = hidden.clone() + replacement.forward(hidden);
                    replacement_error =
                        max_error(study.backbone.project_hidden(replaced)?, raw_logits)?;
                }
            }
        }
    }

    study::write_prediction_table(&root, &table_labels, &table_rows)?;
    study::write_report(
        &root,
        study::Measurements {
            layers: metrics,
            input_tokens,
            ce_tokens,
            final_ce,
            parity_error,
            observation_error,
            replacement_error,
            replacement_translator: REPLACE_TRANSLATOR,
            wall_seconds: started.elapsed().as_secs_f64(),
        },
        &reference,
        &preparation,
    )
}
