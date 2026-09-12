//! Change the sequence mixer, residual width, normalization, and task head in ordinary Rust.
//!
//! Run: cargo run --release --locked --example custom_composition
//!
//! By default this is an architecture demonstration with random weights.
//! Train a warm-color majority classifier with native Burn Adam:
//! `cargo run --release --features autodiff --example custom_composition -- --train`
//! Read training/custom_composition.rs for the differentiable capture and learning loop.
//!
//! The first residual block uses bidirectional grouped-query attention. The next
//! mixer broadcasts `W(mean_tokens(h))`, with no softmax or causal mask. Learned
//! projections explicitly change residual width `8 -> 12 -> 6`; mean pooling
//! produces `[batch, 6]` features and a `[batch, 2]` task readout. Every token is
//! valid, so neither mixer nor pooling applies a padding mask. There is no dropout
//! or decode cache. Training uses F32 Flex autodiff and registered Burn parameters;
//! the training module owns loss construction, optimization, and record checks.

#[cfg(not(feature = "flex"))]
/// Report the missing Flex backend required by the local composition example.
fn main() {
    eprintln!("enable the `flex` feature to run this example");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "autodiff", feature = "std"))]
#[path = "training/custom_composition.rs"]
mod training;

#[cfg(feature = "flex")]
/// Dispatch to the structural demonstration or the explicitly requested training loop.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--train") {
        #[cfg(all(feature = "autodiff", feature = "std"))]
        return training::run();
        #[cfg(not(all(feature = "autodiff", feature = "std")))]
        return Err("training requires --features std,autodiff".into());
    }
    Ok(experiment::run()?)
}

#[cfg(feature = "flex")]
mod experiment {
    //! An open two-mixer graph with explicit residual-width boundaries and shared hook points.

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
            AttentionPolicy, ExecutionContext, HiddenTransform, ResidualBlock, WidthAdapter,
        },
        feed_forward::{DenseFeedForward, DenseFeedForwardConfig},
        intervention::{Hooks, Layer, PointInfo, Stack},
        operations,
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

    /// Residual block pairing grouped-query attention with a dense feed-forward layer.
    type AttentionBlock<B> = ResidualBlock<B, GroupedQueryAttention<B>, DenseFeedForward<B>>;
    /// Mean-mixing residual block wrapped by explicit input/output width adapters.
    type ChangedBlock<B> = WidthAdapter<B, ResidualBlock<B, MeanMixer<B>, DenseFeedForward<B>>>;

    // All new modules, width adapters, and the task head are owned here. Burn records
    // and optimizers can therefore see their parameters without inspecting callbacks.
    /// Owns the heterogeneous encoder blocks, width adapters, and task readout.
    #[derive(Module, Debug)]
    pub(super) struct Classifier<B: Backend> {
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
        pub(super) fn init(device: &B::Device) -> Result<Self> {
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

        #[cfg(all(feature = "autodiff", feature = "std"))]
        /// Borrow representative registered matrices used to check gradient flow and optimizer updates.
        pub(super) fn training_weights(
            &self,
        ) -> Vec<(&'static str, &burn::module::Param<Tensor<B, 2>>)> {
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
        /// Token positions accompany the context; all input positions are treated as valid.
        pub(super) fn forward(
            &self,
            tokens: Tensor<B, 2, Int>,
            hooks: &mut Hooks<'_, B>,
        ) -> Result<Tensor<B, 2>> {
            let [batch, sequence] = tokens.dims();
            let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &tokens.device())
                .reshape([1, sequence])
                .expand([batch, sequence]);
            let context = ExecutionContext::new(positions, None)?
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

    /// Verify that a selected custom-block activation can be ablated and restored without editing weights.
    pub fn run() -> Result<()> {
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
        Ok(())
    }
}
