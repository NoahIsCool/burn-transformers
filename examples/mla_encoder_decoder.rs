//! Compose an encoder-decoder with shared MLA and intervene in its compressed states.
//!
//! Run: cargo run --release --locked --example mla_encoder_decoder
//!
//! Random weights, three source tokens, two target tokens, and no downloads.
//! By default this demonstrates composition and intervention with random weights.
//! Train a source-counting task with bidirectional decoder queries and native Burn AdamW:
//! `cargo run --release --features autodiff --example mla_encoder_decoder -- --train`
//! Read training/mla_encoder_decoder.rs for the task split, loop, and record round-trip.

#[cfg(not(feature = "flex"))]
fn main() {
    eprintln!("enable the `flex` feature to run this example");
    std::process::exit(1);
}

#[cfg(all(feature = "flex", feature = "autodiff", feature = "std"))]
#[path = "training/mla_encoder_decoder.rs"]
mod training;

#[cfg(feature = "flex")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--train") {
        #[cfg(all(feature = "autodiff", feature = "std"))]
        return training::run();
        #[cfg(not(all(feature = "autodiff", feature = "std")))]
        return Err("training requires --features std,autodiff".into());
    }
    experiment::run()?;
    Ok(())
}

#[cfg(feature = "flex")]
mod experiment {
    use burn::{
        backend::Flex,
        module::Module,
        nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
        tensor::{Int, Tensor, backend::Backend},
    };
    use transformers::{
        Result,
        feed_forward::{GatedFeedForward, GatedFeedForwardConfig},
        intervention::{AttentionPoints, Hooks, Layer},
        mla::{MultiLatentAttention, MultiLatentAttentionConfig},
        normalization::{RmsNorm, RmsNormConfig},
    };

    type B = Flex<f32, i32>;

    // Every attention operation is the same shared module. The caller chooses
    // where queries/keys originate and whether future tokens are visible.
    #[derive(Module, Debug)]
    struct AttentionBlock<B: Backend> {
        norm: RmsNorm<B>,
        attention: MultiLatentAttention<B>,
    }

    impl<B: Backend> AttentionBlock<B> {
        fn init(device: &B::Device) -> Result<Self> {
            Ok(Self {
                norm: RmsNormConfig::new(8).init(device)?,
                attention: MultiLatentAttentionConfig::new(8, 2, 2, 2, 4, 3, None).init(device)?,
            })
        }

        fn forward(
            &self,
            hidden: Tensor<B, 3>,
            source: Option<Tensor<B, 3>>,
            causal: bool,
            points: AttentionPoints,
            hooks: &mut Hooks<'_, B>,
        ) -> Result<Tensor<B, 3>> {
            let query = self.norm.forward(hidden.clone());
            let source = source.unwrap_or_else(|| query.clone());
            let projection = self
                .attention
                .project_with_hooks(query, source, points, hooks)?;
            // No RoPE is needed for this small demonstration. To add it, call
            // projection.with_rotary(query_positions, source_positions, ...).
            let attended = self
                .attention
                .attend_with_hooks(projection, None, None, causal, points, hooks)?;
            Ok(hidden + attended)
        }
    }

    #[derive(Module, Debug)]
    struct FeedForwardBlock<B: Backend> {
        norm: RmsNorm<B>,
        feed_forward: GatedFeedForward<B>,
    }

    impl<B: Backend> FeedForwardBlock<B> {
        fn init(device: &B::Device) -> Result<Self> {
            Ok(Self {
                norm: RmsNormConfig::new(8).init(device)?,
                feed_forward: GatedFeedForwardConfig::new(8, 16).init(device)?,
            })
        }

        fn forward(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
            Ok(hidden.clone() + self.feed_forward.forward(self.norm.forward(hidden))?)
        }
    }

    // The wiring follows encoder self-attention → decoder causal attention →
    // decoder cross-attention. It is an experimental architecture, not a T5
    // checkpoint recipe. The adapter is registered alongside the other modules.
    #[derive(Module, Debug)]
    pub(super) struct EncoderDecoder<B: Backend> {
        embedding: Embedding<B>,
        encoder_attention: AttentionBlock<B>,
        encoder_feed_forward: FeedForwardBlock<B>,
        encoder_norm: RmsNorm<B>,
        decoder_attention: AttentionBlock<B>,
        cross_attention: AttentionBlock<B>,
        decoder_feed_forward: FeedForwardBlock<B>,
        decoder_norm: RmsNorm<B>,
        task_head: Linear<B>,
        latent_adapter: Linear<B>,
    }

    impl<B: Backend> EncoderDecoder<B> {
        pub(super) fn init(device: &B::Device) -> Result<Self> {
            Ok(Self {
                embedding: EmbeddingConfig::new(16, 8).init(device),
                encoder_attention: AttentionBlock::init(device)?,
                encoder_feed_forward: FeedForwardBlock::init(device)?,
                encoder_norm: RmsNormConfig::new(8).init(device)?,
                decoder_attention: AttentionBlock::init(device)?,
                cross_attention: AttentionBlock::init(device)?,
                decoder_feed_forward: FeedForwardBlock::init(device)?,
                decoder_norm: RmsNormConfig::new(8).init(device)?,
                task_head: LinearConfig::new(8, 3).init(device),
                latent_adapter: LinearConfig::new(3, 3).init(device),
            })
        }

        #[cfg(all(feature = "autodiff", feature = "std"))]
        pub(super) fn training_logits(
            &self,
            source: Tensor<B, 2, Int>,
            queries: Tensor<B, 2, Int>,
        ) -> Result<Tensor<B, 3>> {
            let mut hooks = Hooks::new();
            hooks.edit(Layer::decoder(0).cross_attention().kv_latent(), |latent| {
                latent.clone() + self.latent_adapter.forward(latent) * 0.1
            });
            // Both target slots are known queries, so each may see the other's query.
            self.forward_with_visibility(source, queries, false, &mut hooks)
        }

        #[cfg(all(feature = "autodiff", feature = "std"))]
        pub(super) fn training_weights(
            &self,
        ) -> Vec<(&'static str, &burn::module::Param<Tensor<B, 2>>)> {
            vec![
                ("embedding", &self.embedding.weight),
                (
                    "decoder MLA",
                    &self
                        .decoder_attention
                        .attention
                        .q_proj
                        .as_ref()
                        .unwrap()
                        .weight,
                ),
                (
                    "encoder MLA",
                    &self.encoder_attention.attention.kv_a_proj_with_mqa.weight,
                ),
                (
                    "cross MLA",
                    &self.cross_attention.attention.kv_b_proj.weight,
                ),
                ("latent adapter", &self.latent_adapter.weight),
                ("task head", &self.task_head.weight),
            ]
        }

        fn forward(
            &self,
            source_ids: Tensor<B, 2, Int>,
            target_ids: Tensor<B, 2, Int>,
            hooks: &mut Hooks<'_, B>,
        ) -> Result<Tensor<B, 3>> {
            self.forward_with_visibility(source_ids, target_ids, true, hooks)
        }

        fn forward_with_visibility(
            &self,
            source_ids: Tensor<B, 2, Int>,
            target_ids: Tensor<B, 2, Int>,
            decoder_causal: bool,
            hooks: &mut Hooks<'_, B>,
        ) -> Result<Tensor<B, 3>> {
            let encoder_point = Layer::encoder(0).self_attention();
            let decoder_point = Layer::decoder(0).self_attention();
            let cross_point = Layer::decoder(0).cross_attention();
            let mut catalog = self
                .encoder_attention
                .attention
                .activation_points(encoder_point);
            catalog.extend(
                self.decoder_attention
                    .attention
                    .activation_points(decoder_point),
            );
            catalog.extend(
                self.cross_attention
                    .attention
                    .activation_points(cross_point),
            );

            hooks.run(catalog, |hooks| {
                let encoder = self.encoder_attention.forward(
                    self.embedding.forward(source_ids),
                    None,
                    false,
                    encoder_point,
                    hooks,
                )?;
                let encoder = self
                    .encoder_norm
                    .forward(self.encoder_feed_forward.forward(encoder)?);
                let decoder = self.decoder_attention.forward(
                    self.embedding.forward(target_ids),
                    None,
                    decoder_causal,
                    decoder_point,
                    hooks,
                )?;
                let decoder = self.cross_attention.forward(
                    decoder,
                    Some(encoder),
                    false,
                    cross_point,
                    hooks,
                )?;
                let decoder = self
                    .decoder_norm
                    .forward(self.decoder_feed_forward.forward(decoder)?);
                Ok(self.task_head.forward(decoder))
            })
        }
    }

    pub fn run() -> Result<()> {
        let device = Default::default();
        B::seed(&device, 23);
        let model = EncoderDecoder::<B>::init(&device)?;
        let source = Tensor::from_ints([[1, 2, 3]], &device);
        let target = Tensor::from_ints([[4, 5]], &device);
        let latent = Layer::decoder(0).cross_attention().kv_latent();

        let mut hooks = Hooks::new();
        let saved = hooks.capture(&latent);
        let clean = model.forward(source.clone(), target.clone(), &mut hooks)?;
        let compressed = hooks.get(&saved)?;
        assert_eq!(compressed.dims(), [1, 3, 3]);
        assert_eq!(clean.dims(), [1, 2, 3]);
        println!(
            "Three source tokens compress to {:?}; two target outputs have shape {:?}.",
            compressed.dims(),
            clean.dims()
        );

        // Transform the shared compressed source representation. The attention
        // operation expands these modified states into both keys and values.
        let mut intervene = Hooks::new();
        intervene.edit(&latent, |states| {
            states.clone() + model.latent_adapter.forward(states) * 0.1
        });
        let changed = model.forward(source.clone(), target.clone(), &mut intervene)?;
        let change = (changed - clean.clone()).abs().max().into_scalar();
        assert!(change > 1e-6);
        println!("Compressed-state adapter changed task logits by {change:.6}.");

        // Restore after the adapter, so downstream computations recover the
        // original output without editing any parameters or family source code.
        intervene.replace(&latent, compressed);
        let restored = model.forward(source, target, &mut intervene)?;
        let error = (restored - clean).abs().max().into_scalar();
        assert!(error < 1e-6);
        println!("Restoration error: {error:.6}.");
        Ok(())
    }
}
