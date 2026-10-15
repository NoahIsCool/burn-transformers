//! Change the sequence mixer, residual width, normalization, and task head in ordinary Rust.
//!
//! Run: cargo run --release --locked --example custom_composition
//!
//! By default this is an architecture demonstration with random weights.
//! Train a warm-color majority classifier with native Burn Adam:
//! `cargo run --release --locked --features autodiff --example custom_composition -- --train`
//! Read training/custom_composition.rs for the differentiable capture and learning loop.
//!
//! The first residual block uses bidirectional, position-free (NoPE) grouped-query
//! attention, matching the permutation-invariant task. The next
//! mixer broadcasts `W(mean_tokens(h))`, with no softmax or causal mask. Learned
//! projections explicitly change residual width `8 -> 12 -> 6`; mean pooling
//! produces `[batch, 6]` features and a `[batch, 2]` task readout. Every token is
//! valid, so neither mixer nor pooling applies a padding mask. There is no dropout
//! or decode cache. Training uses F32 Flex autodiff and registered Burn parameters;
//! the training module owns loss construction, optimization, and record checks.
//!
//! The demonstration also assembles a Llama-style block from the same parts:
//! RMS pre-normalization, rotary grouped-query attention (`RotaryAttention`), and
//! a SwiGLU feed-forward layer. Rotary scores depend only on relative positions,
//! so shifting every position by a constant leaves the block's output unchanged.

#[cfg(feature = "autodiff")]
#[path = "training/custom_composition.rs"]
mod training;

use burn::{
    backend::Flex,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, backend::Backend},
};
use transformers::{
    Result,
    architecture::{NormPlacement, NormalizationSpec},
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig},
    experiment::{
        AttentionPolicy, ExecutionContext, HiddenTransform, ResidualBlock, RotaryAttention,
        WidthAdapter,
    },
    feed_forward::{
        DenseFeedForward, DenseFeedForwardConfig, GatedFeedForward, GatedFeedForwardConfig,
    },
    intervention::{Hooks, Layer, PointInfo, Stack},
    operations,
    position::{RopeConfig, RopeScaling},
};

// This custom mixer has no attention scores or softmax. It projects the sequence
// mean and shares that context with every token. This example has no padding.
/// Projects an unmasked sequence mean and broadcasts it to every token.
#[derive(Module, Debug)]
struct MeanMixer<B: Backend> {
    /// Learned channel projection applied to the broadcast sequence mean.
    projection: Linear<B>,
}

impl<B: Backend> HiddenTransform<B> for MeanMixer<B> {
    /// Project the mean of `[batch, token, channel]` states and broadcast it to every token.
    /// All positions participate; this mixer assumes the example's unpadded inputs.
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _context: &ExecutionContext<B>,
        _layer: Layer,
        _hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let shape = hidden.dims();
        operations::linear(hidden.mean_dim(1).expand(shape), &self.projection)
    }
}

/// Residual block pairing position-free grouped-query attention with a dense feed-forward layer.
type AttentionBlock<B> = ResidualBlock<B, GroupedQueryAttention<B>, DenseFeedForward<B>>;
/// Llama-style block: rotary grouped-query attention and a SwiGLU feed-forward layer.
type LlamaBlock<B> = ResidualBlock<B, RotaryAttention<B>, GatedFeedForward<B>>;
/// Mean-mixing residual block wrapped by explicit input/output width adapters.
type ChangedBlock<B> = WidthAdapter<B, ResidualBlock<B, MeanMixer<B>, DenseFeedForward<B>>>;

// All new modules, width adapters, and the task head are owned here. Burn records
// and optimizers can therefore see their parameters without inspecting callbacks.
/// Owns the heterogeneous encoder blocks, width adapters, and task readout.
#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    /// Token lookup producing eight-channel residual states.
    embeddings: Embedding<B>,
    /// Eight-channel attention block at encoder layer zero.
    first: AttentionBlock<B>,
    /// Twelve-channel mean-mixing block with eight-channel input and six-channel output.
    changed: ChangedBlock<B>,
    /// Binary classifier applied to the mean-pooled six-channel residual.
    head: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    /// Register a width-eight attention block, a width-twelve mean mixer, and a six-channel readout boundary.
    fn init(device: &B::Device) -> Result<Self> {
        let attention = GroupedQueryAttentionConfig::new(8, 2, 1, 4).init(device)?;
        let first = ResidualBlock::new(
            8,
            attention,
            DenseFeedForwardConfig::new(8, 16).init(device)?,
            NormalizationSpec::Rms {
                epsilon: 1e-6,
                zero_centered: false,
            },
            NormPlacement::Pre,
            device,
        )?;
        let mixer = MeanMixer {
            projection: LinearConfig::new(12, 12).init(device),
        };
        let second = ResidualBlock::without_normalization(
            12,
            mixer,
            DenseFeedForwardConfig::new(12, 24).init(device)?,
        )?;
        Ok(Self {
            embeddings: EmbeddingConfig::new(32, 8).init(device),
            first,
            // Explicit learned boundaries: 8 -> 12 -> custom block -> 6 channels.
            changed: WidthAdapter::new(second, 8, 12, 6, device)?,
            head: LinearConfig::new(6, 2).init(device),
        })
    }

    #[cfg(feature = "autodiff")]
    /// Borrow representative registered matrices used to check gradient flow and optimizer updates.
    fn training_weights(&self) -> Vec<(&'static str, &burn::module::Param<Tensor<B, 2>>)> {
        vec![
            ("embedding", &self.embeddings.weight),
            ("attention", &self.first.mixer.q_proj.weight),
            (
                "width input",
                &self.changed.input_projection.as_ref().unwrap().weight,
            ),
            (
                "custom mean mixer",
                &self.changed.transform.mixer.projection.weight,
            ),
            (
                "width output",
                &self.changed.output_projection.as_ref().unwrap().weight,
            ),
            ("task head", &self.head.weight),
        ]
    }

    /// List the embedding and both residual blocks' valid semantic hook addresses.
    fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Encoder.embedding().info()];
        points.extend(self.first.activation_points(Layer::encoder(0)));
        points.extend(self.changed.activation_points(Layer::encoder(1)));
        points
    }

    /// Run bidirectional attention and the custom mixer, then mean-pool into `[batch, 2]` logits.
    /// Positions `0..sequence` accompany the context; all input positions are treated as valid.
    fn forward(&self, tokens: Tensor<B, 2, Int>, hooks: &mut Hooks<'_, B>) -> Result<Tensor<B, 2>> {
        let [batch, sequence] = tokens.dims();
        // Contiguous positions are known on the host, so building them reads nothing back.
        let context = ExecutionContext::contiguous(batch, sequence, &tokens.device())?
            .with_attention(AttentionPolicy::bidirectional());
        hooks.run(self.activation_points(), |hooks| {
            let hidden =
                hooks.apply(Stack::Encoder.embedding(), self.embeddings.forward(tokens))?;
            let hidden = self
                .first
                .forward(hidden, &context, Layer::encoder(0), hooks)?;
            let hidden = self
                .changed
                .forward(hidden, &context, Layer::encoder(1), hooks)?;
            let pooled = hidden.mean_dim(1).squeeze_dim::<2>(1);
            operations::linear(pooled, &self.head)
        })
    }
}

/// Build a width-eight Llama-style block: RMS pre-norm, rotary GQA, and SwiGLU.
fn llama_block<B: Backend>(device: &B::Device) -> Result<LlamaBlock<B>> {
    // Two query heads share one key/value head; RoPE rotates all four channels of each head.
    let rope = RopeConfig::new(4, 4, 10_000.0, RopeScaling::None)
        .map_err(|error| transformers::Error::configuration("rope", error.to_string()))?;
    ResidualBlock::new(
        8,
        RotaryAttention::init(&GroupedQueryAttentionConfig::new(8, 2, 1, 4), rope, device)?,
        GatedFeedForwardConfig::new(8, 16).init(device)?,
        NormalizationSpec::Rms {
            epsilon: 1e-6,
            zero_centered: false,
        },
        NormPlacement::Pre,
        device,
    )
}

/// Show that a causal Llama-style block depends only on relative positions.
///
/// Rotations satisfy `<R(m) q, R(n) k> = <q, R(n - m) k>`, so shifting every
/// position by the same offset leaves attention scores, and therefore the block
/// output, unchanged up to F32 rounding of the larger rotation angles.
fn demonstrate_rotary_block() -> Result<f32> {
    /// CPU backend for the rotary demonstration.
    type B = Flex<f32, i32>;
    let device = Default::default();
    let block = llama_block::<B>(&device)?;
    let hidden = Tensor::<B, 3>::random(
        [1, 4, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let layer = Layer::decoder(0);
    let at_zero = ExecutionContext::contiguous(1, 4, &device)?;
    let shifted = ExecutionContext::new(
        Tensor::<B, 2, Int>::from_ints([[7, 8, 9, 10]], &device),
        None,
    )?;
    let reference = block.run(hidden.clone(), &at_zero, layer)?;
    let moved = block.run(hidden, &shifted, layer)?;
    Ok((moved - reference).abs().max().into_scalar())
}

/// Run the structural demonstration, or the training loop when `--train` is supplied.
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--train") {
        #[cfg(feature = "autodiff")]
        return training::run();
        #[cfg(not(feature = "autodiff"))]
        return Err("training requires --features autodiff".into());
    }
    Ok(demonstrate()?)
}

/// Verify that a selected custom-block activation can be ablated and restored without editing weights.
fn demonstrate() -> Result<()> {
    /// CPU backend for the deterministic activation intervention demonstration.
    type B = Flex<f32, i32>;
    let device = Default::default();
    B::seed(&device, 42);
    let model = Classifier::<B>::init(&device)?;
    let tokens = Tensor::from_ints([[1, 2, 3], [4, 5, 6]], &device);
    let point = Layer::encoder(1).mlp_activation().token(0);
    let mut hooks = Hooks::new();
    let neurons = hooks.capture(point.clone());
    let clean = model.forward(tokens.clone(), &mut hooks)?;
    let saved = hooks.get(&neurons)?;
    assert_eq!(saved.dims(), [2, 1, 24]);

    hooks.edit(point.clone(), |activation| activation * 0.0);
    let ablated = model.forward(tokens.clone(), &mut hooks)?;
    let effect = (ablated - clean.clone()).abs().max().into_scalar();
    assert!(effect > 1e-6);
    hooks.replace(point, saved);
    let restored = model.forward(tokens, &mut hooks)?;
    let error = (restored - clean).abs().max().into_scalar();
    assert!(error < 1e-6);
    println!("Two blocks, two mixer algorithms, widths 8 -> 12 -> 6, two task outputs.");
    println!(
        "Selected neurons: [2, 1, 24]; ablation effect: {effect:.6}; restoration error: {error:.6}."
    );
    println!(
        "Registered parameters: {}. Add --features autodiff and -- --train to learn the synthetic task.",
        model.num_params()
    );
    let shift_error = demonstrate_rotary_block()?;
    assert!(shift_error < 1e-4);
    println!(
        "Llama-style rotary block: shifting every position by 7 changes outputs by {shift_error:.2e}."
    );
    Ok(())
}
