//! Inspect, ablate, restore, and transform a transformer activation with public APIs.
//!
//! Run: cargo run --release --locked --example activation_workbench
//!
//! This tiny randomly initialized model needs no checkpoint or tokenizer. The
//! checks demonstrate intervention behavior, not a learned circuit or task score.
//!
//! Hidden tensors use `[batch, token, channel]`; selecting a token retains a
//! length-one token axis. Captures detach by default, while edits replace values
//! consumed by later operations. The residual adapter computes `h + 0.1 W(h)`
//! without changing shape, dtype, or device. Flex uses F32 activations and I32 IDs.
//! The ordinary pass is causal; a separate stateless pass enables future-token
//! attention. No dropout or decode cache is used. The owning Burn module registers
//! both the backbone and adapter for parameter traversal and records.

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
    //! Registered random-weight graph and scoped F32 activation interventions.

    use burn::{
        backend::Flex,
        module::Module,
        nn::{Linear, LinearConfig},
        tensor::{Tensor, backend::Backend},
    };
    use transformers::{
        CausalLmInput, ForwardOptions, LogitsSelection,
        architecture::{
            Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
            FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
            ResidualTopology,
        },
        intervention::{Hooks, Layer},
        models::custom::ModularCausalLm,
    };

    type B = Flex<f32, i32>;

    // Owning both modules here registers all parameters for Burn optimizers and
    // serialization. Hooks borrow this owner only for the duration of a run.
    #[derive(Module, Debug)]
    struct ExperimentModel<B: Backend> {
        model: ModularCausalLm<B>,
        adapter: Linear<B>,
    }

    /// Demonstrate selected capture, ablation, ordered restoration, and an inserted adapter.
    /// The final comparison changes causal visibility while retaining the same weights.
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let device = Default::default();
        B::seed(&device, 42);
        let experiment = ExperimentModel::<B> {
            model: ModularCausalLm::init(tiny_architecture(), &device)?,
            adapter: LinearConfig::new(8, 8).init(&device),
        };
        let model = &experiment.model;
        let input = CausalLmInput {
            token_ids: Tensor::from_data([[1, 2, 3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let options = ForwardOptions {
            logits: LogitsSelection::All,
            ..Default::default()
        };

        // Select the second token in the first block. Indices start at zero;
        // slices keep their axes: [batch=1, token=1, channel=MLP width].
        // The initial recipe below uses an MLP width of 12.
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
        let ablation_change = (ablated.logits - clean.logits.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(ablation_change > 1e-6, "ablation should change the output");
        println!("Ablation: maximum logit change = {ablation_change:.6}");

        // Actions execute in registration order. The same hooks now zero the
        // selected activation, then restore it before the model continues.
        intervene.replace(&neurons, original);
        let restored = model.forward_with_hooks(input.clone(), &options, &mut intervene)?;
        let restoration_error = (restored.logits - clean.logits.clone())
            .abs()
            .max()
            .into_scalar();
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
        let adapter_change = (adapted.logits - clean.logits.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            adapter_change > 1e-6,
            "the inserted module should affect logits"
        );
        println!("Inserted module: maximum logit change = {adapter_change:.6}");

        // Fundamental attention changes are explicit, too: allow future tokens
        // in this stateless pass. The first token can now use the whole sequence.
        let bidirectional = model.forward_bidirectional(input, &options)?;
        let first_token_change = (bidirectional.logits.narrow(1, 0, 1)
            - clean.logits.narrow(1, 0, 1))
        .abs()
        .max()
        .into_scalar();
        assert!(first_token_change > 1e-6);
        println!("Bidirectional attention: first-token logit change = {first_token_change:.6}");

        // Discover valid point names and their axis meanings for any model with
        // this interface using model.activation_points(). No weights are edited.
        Ok(())
    }

    // The experiment above is independent of this recipe. Change either layer's
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
