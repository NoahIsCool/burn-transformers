//! Build a sentiment-classifier graph from T5 methods, with random weights.
//!
//! Run: `cargo run --release --example t5_classifier`
//!
//! The default demonstrates an untrained architecture; its scores have no sentiment meaning.
//! Train a small synthetic adjective task with native Burn Adam:
//! `cargo run --release --features autodiff --example t5_classifier -- --train`
//! Read training/t5_classifier.rs for the loop, held-out split, and record round-trip.
//!
//! T5 encoder states are `[batch, token, 16]`. Layers keep that residual width
//! while changing head counts to `2, 4, 1` and MLP widths to `32, 48, 24`.
//! Attention is bidirectional; the Boolean token mask uses `true` for real tokens.
//! Pooling computes `sum(mask * h) / sum(mask)` and requires at least one real
//! token per row. A registered `h + W(h)` adapter is inserted between encoder
//! stages before the two-class readout. This recipe sets dropout to zero, uses no
//! decoder cache, and trains on F32 Flex with ordinary Burn parameter records.

#[cfg(not(feature = "flex"))]
/// Report the missing Flex backend required by the classifier example.
fn main() {
    eprintln!("enable the `flex` feature to run this example");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "autodiff", feature = "std"))]
#[path = "training/t5_classifier.rs"]
mod training;

#[cfg(feature = "flex")]
/// Run the untrained T5 composition, or train its synthetic task when `--train` is supplied.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--train") {
        #[cfg(all(feature = "autodiff", feature = "std"))]
        return training::run();
        #[cfg(not(all(feature = "autodiff", feature = "std")))]
        return Err("training requires --features std,autodiff".into());
    }
    experiment::run()
}

#[cfg(feature = "flex")]
mod experiment {
    //! A heterogeneous T5 encoder with an inserted registered adapter and masked readout.

    use burn::{
        backend::Flex,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Bool, Int, Tensor, backend::Backend},
    };
    use transformers::{
        CausalLmInput, TokenMask,
        experiment::AttentionPolicy,
        intervention::{Hooks, Layer},
        models::t5::{T5Config, T5Encoder, T5EncoderRecipe, T5FeedForwardProjection},
    };

    type Cpu = Flex<f32, i32>;

    // Own every trainable component in one Burn module so parameter traversal,
    // checkpoint records, and optimizers include the inserted small model.
    #[derive(Module, Debug)]
    pub(super) struct SentimentClassifier<B: Backend> {
        encoder: T5Encoder<B>,
        adapter: Linear<B>,
        classifier: Linear<B>,
    }

    impl<B: Backend> SentimentClassifier<B> {
        /// Initialize a three-layer T5 encoder with heterogeneous head/MLP widths and registered adapter/head modules.
        /// The residual width remains 16 and dropout is explicitly disabled.
        pub(super) fn init(device: &B::Device) -> transformers::Result<Self> {
            // 1. Begin with T5's actual encoder graph, then edit individual layers.
            //    We keep the residual width at 16 while varying attention and MLP widths.
            let mut recipe = T5EncoderRecipe::from_config(T5Config {
                vocab_size: 12,
                d_model: 16,
                d_kv: 4,
                d_ff: 32,
                num_heads: 2,
                num_layers: 3,
                num_decoder_layers: Some(1),
                relative_attention_num_buckets: 8,
                relative_attention_max_distance: 32,
                feed_forward_proj: T5FeedForwardProjection::GatedGelu,
                dropout_rate: 0.0,
                ..T5Config::default()
            })?;
            recipe.layers[1].attention_heads = 4;
            recipe.layers[1].feed_forward_width = 48;
            recipe.layers[2].attention_heads = 1;
            recipe.layers[2].feed_forward_width = 24;
            Ok(Self {
                encoder: recipe.init::<B>(device)?,
                adapter: LinearConfig::new(16, 16).init(device),
                classifier: LinearConfig::new(16, 2).init(device),
            })
        }

        #[cfg(all(feature = "autodiff", feature = "std"))]
        /// Insert the adapter between encoder stages, normalize, and pool real tokens into two-class logits.
        /// The supplied mask must contain at least one valid token per batch row.
        pub(super) fn logits(&self, input: CausalLmInput<B>) -> transformers::Result<Tensor<B, 2>> {
            let mut state = self.encoder.embed(input)?;
            state = self.encoder.run_layers(state, 0..1)?;
            // The inserted module belongs to Self, so its parameters join Adam's updates.
            state.hidden = state.hidden.clone() + self.adapter.forward(state.hidden);
            state = self.encoder.run_layers(state, 1..3)?;
            state = self.encoder.normalize(state)?;
            let valid = state
                .context
                .token_mask
                .as_ref()
                .expect("this task supplies padding")
                .as_tensor()
                .clone()
                .float()
                .unsqueeze_dim::<3>(2);
            let counts = valid.clone().sum_dim(1);
            let pooled = (state.hidden * valid).sum_dim(1) / counts;
            Ok(self.classifier.forward(pooled.squeeze_dim::<2>(1)))
        }

        #[cfg(all(feature = "autodiff", feature = "std"))]
        /// Borrow representative embeddings, attention, MLP, adapter, and head matrices for training checks.
        pub(super) fn training_weights(
            &self,
        ) -> Vec<(&'static str, &burn::module::Param<Tensor<B, 2>>)> {
            vec![
                ("embedding", &self.encoder.shared.weight),
                (
                    "layer 0 attention",
                    &self.encoder.encoder.block[0]
                        .self_attention
                        .self_attention
                        .attention
                        .q_proj
                        .weight,
                ),
                (
                    "layer 1 attention",
                    &self.encoder.encoder.block[1]
                        .self_attention
                        .self_attention
                        .attention
                        .q_proj
                        .weight,
                ),
                (
                    "layer 2 attention",
                    &self.encoder.encoder.block[2]
                        .self_attention
                        .self_attention
                        .attention
                        .q_proj
                        .weight,
                ),
                (
                    "layer 1 MLP",
                    &self.encoder.encoder.block[1]
                        .feed_forward
                        .gated
                        .as_ref()
                        .unwrap()
                        .gate_proj
                        .weight,
                ),
                ("inserted adapter", &self.adapter.weight),
                ("task head", &self.classifier.weight),
            ]
        }
    }

    /// Demonstrate padded T5 stage execution, an inserted adapter, and selected MLP capture with random weights.
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let device = Default::default();
        Cpu::seed(&device, 7);

        let model = SentimentClassifier::<Cpu>::init(&device)?;

        // Tiny explicit vocabulary: 0=padding, 1=this, 2=film, 3=delights,
        // 4=story, 5=is, 6=awful. The first sentence has one padding position.
        let source = CausalLmInput {
            token_ids: Tensor::<Cpu, 2, Int>::from_ints([[1, 2, 3, 0], [1, 4, 5, 6]], &device),
            token_mask: Some(TokenMask::new(Tensor::<Cpu, 2, Bool>::from_bool(
                [[true, true, true, false], [true, true, true, true]].into(),
                &device,
            ))),
            position_ids: None,
        };

        // 2. Run the first block, insert an ordinary small Burn model, then resume.
        //    Positions and padding travel with the hidden tensor in its context.
        let mut state = model.encoder.embed(source)?;
        state.context = state
            .context
            .with_attention(AttentionPolicy::bidirectional());
        state = model.encoder.run_layers(state, 0..1)?;
        state.hidden = state.hidden.clone() + model.adapter.forward(state.hidden);

        // 3. Inspect only one token's intermediate MLP features in the next block.
        let mut hooks = Hooks::new();
        let features = hooks.capture(Layer::encoder(1).mlp_activation().token(0));
        state = model
            .encoder
            .run_layers_with_hooks(state, 1..3, &mut hooks)?;
        state = model.encoder.normalize(state)?;
        println!("Captured MLP features: {:?}", hooks.get(&features)?.dims());

        // 4. Pool real tokens and attach a two-class task head. No decoder or
        //    vocabulary projection was constructed anywhere in this example.
        let valid = state
            .context
            .token_mask
            .as_ref()
            .unwrap()
            .as_tensor()
            .clone()
            .float()
            .unsqueeze_dim::<3>(2);
        let token_count = valid.clone().sum_dim(1);
        let pooled = (state.hidden * valid).sum_dim(1) / token_count;
        let scores = model.classifier.forward(pooled.squeeze_dim::<2>(1));
        let scores = scores.into_data().to_vec::<f32>()?;
        for (sentence, pair) in ["this film delights", "this story is awful"]
            .iter()
            .zip(scores.chunks_exact(2))
        {
            println!(
                "{sentence:?}: untrained negative={:.3}, positive={:.3}",
                pair[0], pair[1]
            );
        }
        println!("Graph ready: heterogeneous T5 encoder → inserted module → pooling → task head.");
        Ok(())
    }
}
