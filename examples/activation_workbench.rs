//! Inspect, ablate, restore, and transform transformer activations with public APIs.
//!
//! Run: cargo run --release --locked --example activation_workbench
//!
//! The tiny randomly initialized models need no checkpoint or tokenizer. The
//! checks demonstrate intervention behavior, not a learned circuit or task score.
//!
//! Part one works on one configurable model: it captures a token's MLP
//! neurons, ablates and restores them, inserts a small residual adapter, and
//! enables future-token attention. Part two runs one family-generic experiment
//! function, written once against the `HookedModel` and `StagedModel` traits,
//! on three different families: the modular decoder, GPT-2, and a
//! Llama-family model selected from its configuration by `AutoCausalLm`. A
//! checkpoint loaded with `AutoCausalLm::load_local` runs through the same
//! function unchanged, because a loaded bundle implements both traits.
//!
//! Hidden tensors use `[batch, token, channel]`; selecting a token retains a
//! length-one token axis. Captures detach by default, while edits replace values
//! consumed by later operations. Flex uses F32 activations and I32 token IDs.
//! No dropout or decode cache is used.

#[cfg(not(feature = "flex"))]
/// Report the missing Flex backend before any model or tensor is constructed.
fn main() {
    eprintln!("enable the `flex` feature to run this example");
    std::process::exit(1);
}

#[cfg(feature = "flex")]
/// Run the seeded activation intervention checks on the Flex backend.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    experiment::run()
}

#[cfg(feature = "flex")]
mod experiment {
    //! Registered random-weight graphs and scoped F32 activation interventions.

    use burn::{
        backend::Flex,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Tensor, backend::Backend},
    };
    use transformers::{
        CausalLmInput, CausalLmOutput, ForwardOptions, LogitsSelection,
        architecture::{
            Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
            FeedForwardSpec, HookedModel, MixerSpec, NormPlacement, NormalizationSpec,
            PositionEncodingSpec, ResidualTopology, StagedModel,
        },
        intervention::{Hooks, Layer},
        models::{
            AutoCausalLm,
            custom::ModularCausalLm,
            gpt2::{Gpt2Config, Gpt2ForCausalLm},
        },
    };

    /// CPU tensor backend with F32 activations and I32 token IDs.
    type B = Flex<f32, i32>;

    // Owning both modules here registers all parameters for Burn optimizers and
    // serialization. Hooks borrow this owner only for the duration of a run.
    /// Owns the causal model and hook-inserted adapter for native parameter traversal.
    #[derive(Module, Debug)]
    struct ExperimentModel<B: Backend> {
        /// Configurable decoder whose named activations are inspected and edited.
        model: ModularCausalLm<B>,
        /// Width-preserving projection inserted into a residual activation by a hook.
        adapter: Linear<B>,
    }

    /// Run both parts: one model in depth, then one generic function on three families.
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let device = Default::default();
        B::seed(&device, 42);
        let input = CausalLmInput {
            token_ids: Tensor::from_data([[1, 2, 3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let corrupted = CausalLmInput {
            token_ids: Tensor::from_data([[3, 2, 1]], &device),
            ..input.clone()
        };

        // The spec-derived policy draws N(0, 0.02^2) projections, depth-scales the
        // residual writers, and leaves normalization at one. `apply_lazy` defers
        // every draw to the parameter's first use; only the policy's value is
        // kept, and a parameter overwritten by a loaded record is never drawn.
        let spec = tiny_architecture();
        let policy = ModularCausalLm::<B>::scaled_init_policy(&spec, 0.02);
        let (model, _report) = policy.apply_lazy(ModularCausalLm::init(spec, &device)?)?;
        let experiment = ExperimentModel::<B> {
            model,
            adapter: LinearConfig::new(8, 8).init(&device),
        };
        one_model_in_depth(&experiment, input.clone())?;

        println!("\nOne experiment function, three families:");
        residual_experiment(
            "modular decoder",
            &experiment.model,
            input.clone(),
            corrupted.clone(),
        )?;
        let gpt2 = Gpt2ForCausalLm::<B>::init(
            Gpt2Config {
                vocab_size: 16,
                n_embd: 8,
                n_head: 2,
                n_layer: 2,
                n_positions: 16,
                n_inner: Some(16),
                ..Default::default()
            },
            &device,
        )?;
        residual_experiment("GPT-2", &gpt2, input.clone(), corrupted.clone())?;
        // `AutoCausalLm` selects the family from `model_type`, exactly as it
        // does for a checkpoint's config.json.
        let llama = AutoCausalLm::<B>::from_config_json(
            r#"{"model_type": "llama", "vocab_size": 16, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "max_position_embeddings": 16}"#,
            &device,
        )?;
        residual_experiment("Llama via AutoCausalLm", &llama, input, corrupted)?;
        Ok(())
    }

    /// One family-generic experiment: catalog, activation cache, ablation, and staged patching.
    ///
    /// `corrupted` is a second prompt of the same shape used as the patching
    /// target. The bounds name no family. The modular models, GPT-2, GPT-NeoX, every
    /// `AutoCausalLm` family with experiment support, and loaded checkpoint
    /// bundles all satisfy them.
    fn residual_experiment<M>(
        name: &str,
        model: &M,
        input: CausalLmInput<B>,
        corrupted: CausalLmInput<B>,
    ) -> transformers::Result<()>
    where
        M: HookedModel<B, Input = CausalLmInput<B>, Output = CausalLmOutput<B>> + StagedModel<B>,
    {
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };

        // 1. Discover the catalog and cache every activation, TransformerLens style.
        let points = model.activation_points()?;
        let mut cache = Hooks::new();
        let handles = points
            .iter()
            .map(|point| cache.capture_dynamic(point.clone()))
            .collect::<Vec<_>>();
        let clean = model.forward_with_hooks(input.clone(), &options, &mut cache)?;
        let mut cached_values = 0;
        for handle in &handles {
            cached_values += cache.get_dynamic(handle)?.shape.iter().product::<usize>();
        }

        // 2. Ablate the residual stream leaving block 0 and measure the effect.
        let mut ablate = Hooks::new();
        ablate.edit(Layer::decoder(0).residual_post(), |hidden| {
            hidden.zeros_like()
        });
        let ablated = model.forward_with_hooks(input.clone(), &options, &mut ablate)?;
        let change = max_difference(ablated.logits, clean.logits);
        assert!(change > 1e-6, "{name}: ablating a residual changed nothing");

        // 3. Patch across stages: run a corrupted prompt through block 0, transplant
        //    the clean prompt's residual stream there, and resume. Every later block
        //    then reads the clean residual, so the result is exactly the clean stack.
        let clean_state = model.run_layers(model.embed(input.clone())?, 0..1)?;
        let mut patched = model.run_layers(model.embed(corrupted)?, 0..1)?;
        patched.hidden = clean_state.hidden;
        let resumed = model.normalize(model.run_layers(patched, 1..model.num_layers())?)?;
        let whole = model.run_stack(input)?;
        let staged_error = max_difference(resumed.hidden, whole.hidden);
        assert!(staged_error == 0.0, "{name}: stages must compose exactly");

        println!(
            "  {name}: {} points ({cached_values} cached values); ablating the block-0 residual changes logits by up to {change:.4}; patching it into a corrupted run over {} staged blocks restores the clean stack exactly",
            points.len(),
            model.num_layers(),
        );
        Ok(())
    }

    /// Demonstrate selected capture, ablation, ordered restoration, an inserted adapter,
    /// and a change of causal visibility on one model.
    fn one_model_in_depth(
        experiment: &ExperimentModel<B>,
        input: CausalLmInput<B>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model = &experiment.model;
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };

        // Select the second token in the first block. Indices start at zero;
        // slices keep their axes: [batch=1, token=1, channel=MLP width].
        // This dense MLP point exposes post-ReLU features before the down projection.
        let neurons = Layer::decoder(0).mlp_activation().token(1);
        let mut observe = Hooks::new();
        let captured = observe.capture(&neurons);
        let clean = model.forward_with_hooks(input.clone(), &options, &mut observe)?;
        let original = observe.get(&captured)?;
        let mlp_width = match &model.spec().layers[0].feed_forward {
            FeedForwardSpec::Dense(mlp) => mlp.intermediate_size,
            _ => unreachable!("this example constructs a dense MLP"),
        };
        assert_eq!(original.dims(), [1, 1, mlp_width]);
        println!("Captured activation shape: {:?}", original.dims());

        // Captures remain on device and detach by default. Editing a selected
        // tensor changes subsequent computation. Other tokens at that activation
        // point stay intact; later attention can propagate the edit to them.
        let mut intervene = Hooks::new();
        intervene.edit(&neurons, |x| x.zeros_like());
        let ablated = model.forward_with_hooks(input.clone(), &options, &mut intervene)?;
        let ablation_change = max_difference(ablated.logits, clean.logits.clone());
        assert!(ablation_change > 1e-6, "ablation should change the output");
        println!("Ablation: maximum logit change = {ablation_change:.6}");

        // Actions execute in registration order. The same hooks now zero the
        // selected activation, then restore it before the model continues.
        intervene.replace(&neurons, original);
        let restored = model.forward_with_hooks(input.clone(), &options, &mut intervene)?;
        let restoration_error = max_difference(restored.logits, clean.logits.clone());
        assert!(
            restoration_error < 1e-6,
            "restoring should recover every logit"
        );
        println!("Restoration: maximum logit error = {restoration_error:.6}");

        // A callback can borrow an ordinary Burn module. This small residual
        // adapter transforms one token's hidden state between the two blocks.
        // It must preserve the selected shape, dtype, and device. ExperimentModel
        // above owns the adapter's parameters; this callback only executes it.
        let mut insert = Hooks::new();
        insert.edit(Layer::decoder(0).residual_post().token(1), |hidden| {
            hidden.clone() + experiment.adapter.forward(hidden) * 0.1
        });
        let adapted = model.forward_with_hooks(input.clone(), &options, &mut insert)?;
        let adapter_change = max_difference(adapted.logits, clean.logits.clone());
        assert!(
            adapter_change > 1e-6,
            "the inserted module should affect logits"
        );
        println!("Inserted module: maximum logit change = {adapter_change:.6}");

        // Fundamental attention changes are explicit, too: allow future tokens
        // in this stateless pass. The first token can now use the whole sequence.
        let bidirectional = model.forward_bidirectional(input, &options)?;
        let first_token_change = max_difference(
            bidirectional.logits.narrow(1, 0, 1),
            clean.logits.narrow(1, 0, 1),
        );
        assert!(first_token_change > 1e-6);
        println!("Bidirectional attention: first-token logit change = {first_token_change:.6}");
        Ok(())
    }

    /// Largest absolute elementwise difference between two equally shaped tensors.
    fn max_difference(left: Tensor<B, 3>, right: Tensor<B, 3>) -> f32 {
        (left - right).abs().max().into_scalar()
    }

    // The experiments above are independent of this recipe. Change either layer's
    // heads or MLP width here; retain matching dimensions in inserted modules.
    /// Build two width-eight pre-norm blocks with grouped-query attention and ReLU MLPs.
    /// The fixed vocabulary and absence of positional encoding keep this a small structural fixture.
    fn tiny_architecture() -> DecoderSpec {
        let norm = NormalizationSpec::Rms {
            epsilon: 1e-6,
            zero_centered: false,
        };
        let layer = DecoderLayerSpec {
            mixer: MixerSpec::Attention(AttentionSpec {
                query_heads: 2,
                key_value_heads: 1,
                head_dim: 4,
                query_key_norm: false,
                output_gate: false,
                position: PositionEncodingSpec::None,
            }),
            feed_forward: FeedForwardSpec::Dense(DenseFeedForwardSpec {
                intermediate_size: 12,
                activation: Activation::Relu,
                bias: false,
            }),
            mixer_norm: norm,
            feed_forward_norm: norm,
            norm_placement: NormPlacement::Pre,
            residual: ResidualTopology::Add,
        };
        DecoderSpec {
            vocabulary_size: 16,
            hidden_size: 8,
            layers: vec![layer.clone(), layer],
            final_norm: norm,
            tie_word_embeddings: false,
            max_position_embeddings: 16,
        }
    }
}
