//! Activation patching on a real checkpoint, written once for every causal family.
//!
//! ```text
//! cargo run --release --locked --features tokenizers --example activation_patching -- MODEL_DIR
//! ```
//!
//! `MODEL_DIR` is any local checkpoint that `AutoCausalLm` loads: GPT-2,
//! GPT-NeoX, Llama, Mistral, Qwen, SmolLM, OLMo, Gemma, and the other
//! registered families. Nothing is downloaded.
//!
//! The task is indirect-object identification (Wang et al., 2022,
//! <https://arxiv.org/abs/2211.00593>). After "When Mary and John went to the
//! store, John gave a drink to", a language model should prefer " Mary" to
//! " John". The corrupted prompt replaces the second "John" with "Mary",
//! which flips that preference. The metric is the final-token logit
//! difference `d = logit(" Mary") - logit(" John")`.
//!
//! Patching copies one vector of the clean run's residual stream, after block
//! `L` at token `t`, into the corrupted run and reports the recovered fraction
//! `(d_patched - d_corrupt) / (d_clean - d_corrupt)`. A value near 1 means that
//! single vector carries what the prediction needs to know about the changed
//! name; a value near 0 means it carries none of it. Tokens before the change
//! are identical in both prompts, so only the changed token and later ones
//! are patched. Two entries are exact by construction: after the last block,
//! patching the final token recovers exactly 1 and patching any other token
//! exactly 0, because the readout sees only the final token's vector.
//!
//! On SmolLM-135M and Qwen3.5-0.8B the information first sits on the changed
//! name's own token and, a few blocks before the end, moves to the final
//! token, the pattern Wang et al. report for GPT-2 small.
//!
//! The experiment uses only two traits that every loaded bundle implements:
//! `HookedModel` runs the whole stateless forward pass with hooks, and
//! `StagedModel::layer` names block `L` in the model's own stack. Weights are
//! cast to F32 for the Flex CPU backend. Residual activations have axes
//! `[batch, token, channel]`, and a token selection keeps a length-one token
//! axis, so a captured `[1, 1, channel]` slice replaces exactly one vector. No
//! decode cache or dropout participates.

use std::{error::Error as StdError, ops::Range, time::Instant};

use burn::{backend::Flex, tensor::Tensor};
use transformers::{
    AutoCausalLm, AutoLoadOptions, CausalLmInput, CausalLmOutput, EncodeOptions, ForwardOptions,
    LocalTokenizer,
    architecture::{HookedModel, StagedModel},
    checkpoint::{DTypePolicy, LoadOptions, TensorDType},
    intervention::Hooks,
};

/// CPU backend with F32 activations and I32 token IDs.
type Cpu = Flex<f32, i32>;

/// Prompt whose repeated subject is "John", so the indirect object is " Mary".
const CLEAN: &str = "When Mary and John went to the store, John gave a drink to";
/// The same prompt with the repeated subject changed to "Mary", flipping the answer.
const CORRUPT: &str = "When Mary and John went to the store, Mary gave a drink to";
/// Correct continuation of the clean prompt.
const ANSWER: &str = " Mary";
/// Correct continuation of the corrupted prompt, and the clean prompt's distractor.
const DISTRACTOR: &str = " John";

/// Loads the checkpoint, tokenizes both prompts, and prints the patching table.
fn main() -> Result<(), Box<dyn StdError>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: activation_patching MODEL_DIR")?;
    let device = Default::default();
    let options = AutoLoadOptions {
        checkpoint: LoadOptions {
            dtype_policy: DTypePolicy::Cast(TensorDType::F32),
            ..LoadOptions::default()
        },
        ..AutoLoadOptions::default()
    };
    let model = AutoCausalLm::<Cpu>::load_local(&root, &options, &device)?;
    let tokenizer = model
        .tokenizer()
        .ok_or("the model directory has no tokenizer.json")?;

    // Both prompts must align token by token; patching starts where they differ.
    let clean = tokenizer.encode(CLEAN, &EncodeOptions::default())?;
    let corrupt = tokenizer.encode(CORRUPT, &EncodeOptions::default())?;
    let tokens = &clean.token_ids()[0];
    if tokens.len() != corrupt.token_ids()[0].len() {
        return Err("this tokenizer splits the two names differently; edit the prompts".into());
    }
    let first_change = tokens
        .iter()
        .zip(&corrupt.token_ids()[0])
        .position(|(clean, corrupt)| clean != corrupt)
        .ok_or("the prompts tokenize identically")?;
    let positions = first_change..tokens.len();
    let answer = single_token(tokenizer, ANSWER)?;
    let distractor = single_token(tokenizer, DISTRACTOR)?;

    let patches = model.num_layers() * positions.len();
    eprintln!("Running {patches} patched forward passes...");
    let started = Instant::now();
    let sweep = patching_sweep(
        &model,
        clean.to_causal_input(&device)?,
        corrupt.to_causal_input(&device)?,
        positions.clone(),
        answer,
        distractor,
    )?;
    println!(
        "logit({ANSWER:?}) - logit({DISTRACTOR:?}): clean {:.3}, corrupted {:.3}",
        sweep.clean, sweep.corrupt
    );
    println!("\nRecovered fraction after patching the residual stream leaving block L at token t:");
    print!("{:>5}", "L");
    for position in positions.clone() {
        let token = tokenizer.decode(&tokens[position..=position], false)?;
        print!("{:>10}", format!("{token:?}"));
    }
    println!();
    for (layer, row) in sweep.recovered.iter().enumerate() {
        print!("{layer:>5}");
        for recovered in row {
            print!("{recovered:>10.2}");
        }
        println!();
    }
    println!(
        "\n{patches} patched forward passes in {:.1}s.",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Logit differences of both runs and the recovered fraction of every patch.
struct Sweep {
    /// Final-token `logit(answer) - logit(distractor)` of the clean run.
    clean: f32,
    /// The same difference for the unpatched corrupted run.
    corrupt: f32,
    /// `recovered[layer][i]` patches the residual leaving `layer` at `positions.start + i`.
    recovered: Vec<Vec<f32>>,
}

/// Patches every `(block, token)` residual vector of `clean` into `corrupt`, one at a time.
///
/// Both inputs are single prompts of equal length. `positions` lists the
/// tokens to patch; earlier tokens must be identical in both prompts. The
/// bounds name no family, so any loaded checkpoint or freshly built model
/// with experiment support runs through this function unchanged.
fn patching_sweep<M>(
    model: &M,
    clean: CausalLmInput<Cpu>,
    corrupt: CausalLmInput<Cpu>,
    positions: Range<usize>,
    answer: usize,
    distractor: usize,
) -> Result<Sweep, Box<dyn StdError>>
where
    M: HookedModel<Cpu, Input = CausalLmInput<Cpu>, Output = CausalLmOutput<Cpu>>
        + StagedModel<Cpu>,
{
    // The default options project only the final token to vocabulary logits.
    let options = ForwardOptions::default();

    // 1. Run the clean prompt once, caching the residual stream after every block.
    let mut cache = Hooks::new();
    let handles = (0..model.num_layers())
        .map(|layer| cache.capture(model.layer(layer).residual_post()))
        .collect::<Vec<_>>();
    let clean_logits = model
        .forward_with_hooks(clean, &options, &mut cache)?
        .logits;
    let clean_difference = logit_difference(clean_logits, answer, distractor);
    let corrupt_logits = model
        .forward_with_hooks(corrupt.clone(), &options, &mut Hooks::new())?
        .logits;
    let corrupt_difference = logit_difference(corrupt_logits, answer, distractor);
    let gap = clean_difference - corrupt_difference;
    if gap.abs() < 1e-3 {
        return Err("the model scores both prompts alike, so there is nothing to recover".into());
    }

    // 2. Patch one clean residual vector at a time into the corrupted run.
    let mut recovered = Vec::with_capacity(handles.len());
    for (layer, handle) in handles.iter().enumerate() {
        let clean_residual = cache.get(handle)?;
        let mut row = Vec::with_capacity(positions.len());
        for position in positions.clone() {
            let mut patch = Hooks::new();
            patch.replace(
                model.layer(layer).residual_post().token(position),
                clean_residual.clone().narrow(1, position, 1),
            );
            let patched = model.forward_with_hooks(corrupt.clone(), &options, &mut patch)?;
            let patched = logit_difference(patched.logits, answer, distractor);
            row.push((patched - corrupt_difference) / gap);
        }
        recovered.push(row);
    }
    Ok(Sweep {
        clean: clean_difference,
        corrupt: corrupt_difference,
        recovered,
    })
}

/// Returns the ID of `text`, which must encode to exactly one token without special tokens.
fn single_token(tokenizer: &LocalTokenizer, text: &str) -> Result<usize, Box<dyn StdError>> {
    let options = EncodeOptions {
        add_special_tokens: false,
        ..EncodeOptions::default()
    };
    match tokenizer.encode(text, &options)?.token_ids()[0].as_slice() {
        [id] => Ok(*id as usize),
        ids => Err(format!("{text:?} encodes to {} tokens, not one", ids.len()).into()),
    }
}

/// Final-token `logit(answer) - logit(distractor)` from `[batch = 1, 1, vocabulary]` logits.
///
/// The subtraction happens on the device, so only one F32 scalar is read back.
fn logit_difference(logits: Tensor<Cpu, 3>, answer: usize, distractor: usize) -> f32 {
    let [_, _, vocabulary] = logits.dims();
    let logits = logits.reshape([vocabulary]);
    let answer = logits.clone().slice(answer..answer + 1);
    let distractor = logits.slice(distractor..distractor + 1);
    (answer - distractor).into_scalar()
}

#[cfg(test)]
#[path = "patching/activation_patching_tests.rs"]
mod tests;
