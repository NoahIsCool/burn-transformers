//! Change actual GPT-2 MLP activations and measure its preference for later years.
//!
//! Prepare/reference separately; this executable reads only local pinned artifacts.
//! See examples/mechanistic/greater_than/README.md for the bounded paper adaptation.
//!
//! A node patch substitutes clean MLP output `[batch, 1, hidden=768]` into a
//! corrupted causal forward. A direct-path test instead changes the contribution
//! to the final residual while keeping later components at their clean values.
//! F32 model logits are normalized in host F64 over all 50,257 vocabulary items.
//! For clean suffix y, preference is P(suffix > y) - P(suffix <= y); numeric-token
//! probabilities retain their full-vocabulary mass. The fixed 194 prompt pairs
//! provide a bounded adaptation, with clean-disjoint splits and reused corruptions.

#[cfg(not(feature = "flex"))]
/// Explain the required Cargo features and exit before accessing local artifacts.
fn main() {
    eprintln!("enable the flex feature to run this CPU example");
    std::process::exit(1);
}

#[cfg(feature = "flex")]
#[path = "mechanistic/greater_than/study.rs"]
mod study;

#[cfg(feature = "flex")]
/// Execute the pinned clean/corrupt node interventions and separate direct-path controls.
///
/// All metrics keep the original clean year cutoff; each mini-batch is released
/// after its full condition inventory has been measured.
fn main() -> study::Result<()> {
    use std::{path::PathBuf, time::Instant};
    use study::Study;
    use transformers::{
        ForwardOptions,
        intervention::{Hooks, Layer},
    };

    // All indices are zero-based. MLP outputs have axes [batch, token, channel]
    // and are selected after the output projection, before residual addition.
    /// Zero-based final prompt position where logits and MLP interventions are measured.
    const READOUT: usize = 12;
    /// Zero-based decoder layers included in individual node interventions.
    const NODE_LAYERS: [usize; 5] = [2, 8, 9, 10, 11];
    /// Late decoder layers patched together in the joint-node condition.
    const LATE_LAYERS: [usize; 4] = [8, 9, 10, 11];
    /// Maximum prompt pairs evaluated together in each tensor batch.
    const BATCH_SIZE: usize = 8;
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/mechanistic/greater_than"));
    let started = Instant::now();
    if !root.join("preparation.json").is_file() {
        return Err(format!(
            "{}: no prepared artifacts; run python3 examples/mechanistic/greater_than/prepare.py \
             --checkpoint /local/gpt2, or pass the prepared directory as the first argument",
            root.display()
        )
        .into());
    }
    let (model, mut study) = Study::load(&root)?;
    let options = ForwardOptions::default(); // Only the final input token's logits.

    for start in (0..study.len()).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(study.len());
        let clean_input = study.input(start..end, false);
        let corrupt_input = study.input(start..end, true);

        // Matched prompts differ only at position 8: the start suffix becomes 01.
        // Metrics always keep the ORIGINAL clean suffix as their target.
        let ordinary = model.forward(clean_input.clone(), &options)?;
        study.record("clean", start, ordinary.logits)?;
        let mut empty = Hooks::new();
        let output = model.forward_with_hooks(clean_input.clone(), &options, &mut empty)?;
        study.record("empty_clean", start, output.logits)?;

        let mut clean_hooks = Hooks::new();
        let mut corrupt_hooks = Hooks::new();
        let clean_handles = (0..12)
            .map(|layer| clean_hooks.capture(Layer::decoder(layer).mlp_output().token(READOUT)))
            .collect::<Vec<_>>();
        let corrupt_handles = (0..12)
            .map(|layer| corrupt_hooks.capture(Layer::decoder(layer).mlp_output().token(READOUT)))
            .collect::<Vec<_>>();
        let final_residual = clean_hooks.capture(Layer::decoder(11).residual_post().token(READOUT));
        let clean = model.forward_with_hooks(clean_input.clone(), &options, &mut clean_hooks)?;
        let corrupt =
            model.forward_with_hooks(corrupt_input.clone(), &options, &mut corrupt_hooks)?;
        study.record("observe_clean", start, clean.logits)?;
        study.record("corrupt", start, corrupt.logits)?;
        let clean_mlp = clean_handles
            .iter()
            .map(|h| clean_hooks.get(h))
            .collect::<transformers::Result<Vec<_>>>()?;
        let corrupt_mlp = corrupt_handles
            .iter()
            .map(|h| corrupt_hooks.get(h))
            .collect::<transformers::Result<Vec<_>>>()?;
        for layer in 0..12 {
            study.check_capture("clean", layer, start, clean_mlp[layer].clone())?;
            study.check_capture("corrupt", layer, start, corrupt_mlp[layer].clone())?;
        }

        for layer in NODE_LAYERS {
            let point = Layer::decoder(layer).mlp_output().token(READOUT);

            // Patch the paired CLEAN source into a CORRUPTED ordinary forward.
            // Reversed and same-run replacements expose direction and identity.
            for (name, destination, replacement) in [
                ("node_patch", &corrupt_input, &clean_mlp[layer]),
                ("node_reverse", &clean_input, &corrupt_mlp[layer]),
                ("identity", &clean_input, &clean_mlp[layer]),
            ] {
                let mut hooks = Hooks::new();
                hooks.edit(&point, |_| replacement.clone());
                let output = model.forward_with_hooks(destination.clone(), &options, &mut hooks)?;
                study.record(&format!("{name}_{layer}"), start, output.logits)?;
            }

            let mut ablation = Hooks::new();
            ablation.edit(&point, |activation| activation * 0.0);
            let output = model.forward_with_hooks(clean_input.clone(), &options, &mut ablation)?;
            study.record(&format!("zero_{layer}"), start, output.logits)?;

            // Callbacks run in registration order: erase, then restore this node.
            // Restoration must reproduce clean logits, not just the same answer.
            let mut restoration = Hooks::new();
            restoration.edit(&point, |activation| activation * 0.0);
            restoration.edit(&point, |_| clean_mlp[layer].clone());
            let output =
                model.forward_with_hooks(clean_input.clone(), &options, &mut restoration)?;
            study.record(&format!("restore_{layer}"), start, output.logits)?;
        }

        // Useful variation: replace all four late nodes in the SAME forward.
        // Their joint effect need not equal the sum of individual effects.
        let mut combined = Hooks::new();
        for layer in LATE_LAYERS {
            let replacement = clean_mlp[layer].clone();
            combined.edit(
                Layer::decoder(layer).mlp_output().token(READOUT),
                move |_| replacement.clone(),
            );
        }
        let output = model.forward_with_hooks(corrupt_input, &options, &mut combined)?;
        study.record("node_patch_all_late", start, output.logits)?;

        // Separate experiment: change only a component's DIRECT readout path.
        // Later components retain clean-run values; this is not a node patch.
        let clean_final = clean_hooks.get(&final_residual)?;
        study.check_readout(start, model.readout(clean_final.clone())?)?;
        for layer in 0..12 {
            let residual =
                clean_final.clone() - clean_mlp[layer].clone() + corrupt_mlp[layer].clone();
            study.record(&format!("direct_{layer}"), start, model.readout(residual)?)?;
        }
        study.finish_batch(start..end);
        eprintln!(
            "Greater-Than: measured {end}/{} paired prompts",
            study.len()
        );
    }

    study.finish(&root, started.elapsed().as_secs_f64())
}
