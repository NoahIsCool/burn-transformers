//! Train a transformer experiment using Burn's own supervised Learner.
//!
//! cargo run --release --locked --features train --example burn_training
//! This small symbolic classification task needs no downloads.
//!
//! Inputs are unpadded I32 `[batch, 3]` token IDs. A bidirectional F32 backbone
//! produces `[batch, token, 8]` states; an adapter edits token zero and a two-class
//! head reads that token. Its residual is `h + 0.1 dropout(W(h))`. Burn applies
//! inverted dropout with probability 0.1 during autodiff execution and disables
//! it on the `.valid()` backend. Mean categorical cross entropy drives native
//! `TrainStep`; `InferenceStep` exposes the same outputs without backward.
//! Burn's Learner owns batching, Adam updates, and checkpointing. The reload check
//! preserves training-module layout metadata before conversion to evaluation.
//! This fixed-length experiment needs no padding mask or decode cache.
//!
//! The backbone starts from the crate's recommended from-scratch policy,
//! `ModularCausalLm::scaled_init_policy`: `N(0, 0.02^2)` embeddings and
//! projections, with residual-writing projections scaled by `1/sqrt(2 * layers)`.
//! The Learner logs every per-epoch metric under the output directory; the
//! terminal shows only the held-out result.

use burn::{
    backend::{Autodiff, Flex},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::{AutodiffModule, Module},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig},
    optim::AdamConfig,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric, MetricDefinition},
        renderer::{
            EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
            MetricsRendererEvaluation, MetricsRendererTraining, ProgressType, TrainingProgress,
        },
    },
};
use transformers::{
    CausalLmInput,
    architecture::{
        Activation, AttentionSpec, DecoderLayerSpec, DecoderSpec, DenseFeedForwardSpec,
        FeedForwardSpec, MixerSpec, NormPlacement, NormalizationSpec, PositionEncodingSpec,
        ResidualTopology,
    },
    intervention::{Hooks, Layer},
    models::custom::ModularCausalLm,
};

/// CPU tensor backend used for evaluation and data loading.
type Cpu = Flex<f32, i32>;
/// Autodifferentiable CPU backend used by the native Burn learner.
type Training = Autodiff<Cpu>;

// Burn owns and discovers every parameter, including the inserted adapter.
/// Registers the symbolic classifier and inserted adapter as one trainable module.
#[derive(Module, Debug)]
struct Experiment<B: Backend> {
    /// Bidirectionally executed decoder that encodes the three symbolic tokens.
    backbone: ModularCausalLm<B>,
    /// Width-preserving projection inserted after the first decoder block.
    adapter: Linear<B>,
    /// Training-time regularizer for the adapter contribution; disabled by `.valid()`.
    dropout: Dropout,
    /// Binary readout from the first token's final hidden state.
    classifier: Linear<B>,
}

impl<B: Backend> Experiment<B> {
    /// Register the fixed backbone, adapter, 0.1 dropout, and binary readout in one Burn module.
    fn init(device: &B::Device) -> Self {
        let spec = recipe();
        let policy = ModularCausalLm::<B>::scaled_init_policy(&spec, 0.02);
        Self {
            backbone: ModularCausalLm::init_with(spec, &policy, device)
                .expect("valid fixed recipe"),
            adapter: LinearConfig::new(8, 8).init(device),
            // Regularization is explicit in the experiment. Burn disables
            // this Dropout when `.valid()` converts to the inner backend.
            dropout: DropoutConfig::new(0.1).init(),
            classifier: LinearConfig::new(8, 2).init(device),
        }
    }

    /// Produce binary logits and mean cross entropy after editing token zero in a bidirectional pass.
    /// The backend determines dropout behavior; all three input tokens are valid.
    fn classify(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let mut hooks = Hooks::new();
        hooks.edit(Layer::decoder(0).residual_post().token(0), |hidden| {
            hidden.clone() + self.dropout.forward(self.adapter.forward(hidden)) * 0.1
        });
        let hidden = self
            .backbone
            .forward_hidden_bidirectional_with_hooks(
                CausalLmInput {
                    token_ids: batch.tokens,
                    token_mask: None,
                    position_ids: None,
                },
                &mut hooks,
            )
            .expect("batch follows the fixed input contract");
        let logits = self
            .classifier
            .forward(hidden.narrow(1, 0, 1).squeeze_dim::<2>(1));
        // Ordinary Burn loss is sufficient for two classes. Large-vocabulary
        // examples also demonstrate the crate's stable differentiable losses.
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.labels.clone());
        ClassificationOutput::new(loss, logits, batch.labels)
    }
}

// These are native Burn interfaces, with no transformers-specific trainer.
impl<B: AutodiffBackend> TrainStep for Experiment<B> {
    /// Batched symbolic token IDs and binary targets for one training step.
    type Input = Batch<B>;
    /// Classification loss, logits, and targets passed to Burn training metrics.
    type Output = ClassificationOutput<B>;
    /// Build native Burn training output by differentiating the classification loss through the owning module.
    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.classify(batch);
        TrainOutput::new(self, output.loss.clone().backward(), output)
    }
}

impl<B: Backend> InferenceStep for Experiment<B> {
    /// Batched symbolic token IDs and binary targets for one evaluation step.
    type Input = Batch<B>;
    /// Classification loss, logits, and targets passed to Burn evaluation metrics.
    type Output = ClassificationOutput<B>;
    /// Compute classification metrics without invoking backward; evaluation callers use the `.valid()` model.
    fn step(&self, batch: Self::Input) -> Self::Output {
        self.classify(batch)
    }
}

/// Train with Burn data loaders and Adam, then verify deterministic evaluation and native checkpoint reload.
/// The held-out split changes nuisance-token combinations without changing the first-token labeling rule.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/training/burn_learner".into());
    std::fs::create_dir_all(&directory)?;
    let device = Default::default();
    Training::seed(&device, 42);
    let model = Experiment::<Training>::init(&device);
    transformers::training::validate_trainable_module(&model)?;
    let initial = evaluate(&model.valid());

    let train = DataLoaderBuilder::<Training, _, _>::new(SymbolBatcher)
        .batch_size(8)
        .shuffle(42)
        .build(InMemDataset::new(items(false)));
    let valid = DataLoaderBuilder::<Cpu, _, _>::new(SymbolBatcher)
        .batch_size(8)
        .build(InMemDataset::new(items(true)));
    let learner = Learner::new(model, AdamConfig::new().init(), 0.02);

    // Burn supplies batching, optimization, evaluation, and checkpointing.
    let learned = SupervisedTraining::new(&directory, train, valid)
        .num_epochs(30)
        .metric_train_numeric(LossMetric::<Cpu>::new())
        .metric_valid_numeric(LossMetric::<Cpu>::new())
        .metric_train_numeric(AccuracyMetric::<Cpu>::new())
        .metric_valid_numeric(AccuracyMetric::<Cpu>::new())
        .with_file_checkpointer(NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .renderer(QuietRenderer)
        .launch(learner)
        .model;
    let final_metrics = evaluate(&learned);
    // The native checkpointer saved the training owner, preserving layout
    // mappers. Reload into a fresh training owner before converting to eval.
    let reloaded = Experiment::<Training>::init(&device)
        .load_file(
            std::path::Path::new(&directory).join("checkpoint/model-30"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )?
        .valid();
    let expected = learned
        .classify(SymbolBatcher.batch(items(true), &device))
        .output;
    let repeated = learned
        .classify(SymbolBatcher.batch(items(true), &device))
        .output;
    let restored = reloaded
        .classify(SymbolBatcher.batch(items(true), &device))
        .output;
    let evaluation_repeat_error = (expected.clone() - repeated).abs().max().into_scalar();
    let checkpoint_reload_error = (expected - restored).abs().max().into_scalar();
    assert_eq!(evaluation_repeat_error, 0.0, "evaluation disables dropout");
    assert_eq!(
        checkpoint_reload_error, 0.0,
        "native checkpoint preserves logits"
    );
    println!(
        "Held-out symbolic task: loss {:.6} -> {:.6}; accuracy {:.1}% -> {:.1}%",
        initial.0,
        final_metrics.0,
        initial.1 * 100.0,
        final_metrics.1 * 100.0
    );
    assert!(
        final_metrics.0 < initial.0 * 0.5,
        "held-out loss should improve"
    );
    assert!(
        final_metrics.1 >= 0.9,
        "the tiny symbolic task should be learned"
    );
    let report = serde_json::json!({
        "task": "classify first token; held-out nuisance-token combinations",
        "seed": 42, "epochs": 30, "batch_size": 8, "learning_rate": 0.02,
        "training_examples": 16, "held_out_examples": 16,
        "initial_loss": initial.0, "final_loss": final_metrics.0,
        "initial_accuracy": initial.1, "final_accuracy": final_metrics.1,
        "native_burn_learner": true, "dropout": 0.1,
        "initialization": "ModularCausalLm::scaled_init_policy(std = 0.02)",
        "evaluation_repeat_error": evaluation_repeat_error,
        "checkpoint_reload_error": checkpoint_reload_error,
        "claim": "training integration on a symbolic fixture, not natural-language sentiment quality"
    });
    std::fs::write(
        std::path::Path::new(&directory).join("result.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

// Without burn-train's optional `tui` dashboard, Burn's default renderer prints
// a debug line for every iteration. Metrics still reach the experiment logs.
/// Learner renderer that keeps progress off the terminal.
struct QuietRenderer;

impl MetricsRendererTraining for QuietRenderer {
    /// Ignores a training metric update; the Learner's file logger records it.
    fn update_train(&mut self, _state: MetricState) {}
    /// Ignores a validation metric update; the Learner's file logger records it.
    fn update_valid(&mut self, _state: MetricState) {}
    /// Draws no per-iteration training progress.
    fn render_train(&mut self, _item: TrainingProgress, _indicators: Vec<ProgressType>) {}
    /// Draws no per-iteration validation progress.
    fn render_valid(&mut self, _item: TrainingProgress, _indicators: Vec<ProgressType>) {}
}

impl MetricsRendererEvaluation for QuietRenderer {
    /// Ignores a test metric update; this example runs no separate test phase.
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
    /// Draws no test progress.
    fn render_test(&mut self, _item: EvaluationProgress, _indicators: Vec<ProgressType>) {}
}

impl MetricsRenderer for QuietRenderer {
    /// Nothing is drawn, so there is nothing to keep open.
    fn manual_close(&mut self) {}
    /// Needs no metric layout because nothing is drawn.
    fn register_metric(&mut self, _definition: MetricDefinition) {}
}

/// One symbolic sequence and its first-token classification target.
#[derive(Clone, Debug)]
struct Item {
    /// Three input token IDs, including two nuisance tokens.
    tokens: [i32; 3],
    /// Binary class determined by the first token.
    label: i32,
}
/// Device-resident symbolic examples prepared for classification.
#[derive(Clone, Debug)]
struct Batch<B: Backend> {
    /// Input token IDs with axes `[batch, 3]`.
    tokens: Tensor<B, 2, Int>,
    /// Binary target IDs with shape `[batch]`.
    labels: Tensor<B, 1, Int>,
}
/// Packs symbolic examples into device-resident token and target tensors.
#[derive(Clone)]
struct SymbolBatcher;
impl<B: Backend> Batcher<B, Item, Batch<B>> for SymbolBatcher {
    /// Pack symbolic rows into I32 `[batch, 3]` token IDs and `[batch]` labels on the requested device.
    fn batch(&self, items: Vec<Item>, device: &B::Device) -> Batch<B> {
        let count = items.len();
        Batch {
            tokens: Tensor::from_data(
                TensorData::new(
                    items
                        .iter()
                        .flat_map(|item| item.tokens)
                        .collect::<Vec<_>>(),
                    [count, 3],
                ),
                device,
            ),
            labels: Tensor::from_data(
                TensorData::new(
                    items.iter().map(|item| item.label).collect::<Vec<_>>(),
                    [count],
                ),
                device,
            ),
        }
    }
}
/// Enumerate sixteen labeled sequences, choosing seen or held-out nuisance-token combinations.
fn items(held_out: bool) -> Vec<Item> {
    (0..2)
        .flat_map(|label| {
            (3..11).map(move |nuisance| Item {
                tokens: [
                    label + 1,
                    nuisance,
                    if held_out {
                        3 + (nuisance - 2) % 8
                    } else {
                        nuisance
                    },
                ],
                label,
            })
        })
        .collect()
}
/// Return held-out mean cross entropy in nats and argmax accuracy on the non-autodiff Flex model.
fn evaluate(model: &Experiment<Cpu>) -> (f32, f32) {
    let output = model.classify(SymbolBatcher.batch(items(true), &Default::default()));
    let accuracy = output
        .output
        .argmax(1)
        .squeeze_dim::<1>(1)
        .equal(output.targets)
        .float()
        .mean()
        .into_scalar();
    (output.loss.into_scalar(), accuracy)
}
/// Define two width-eight pre-norm attention/ReLU blocks for the symbolic Learner fixture.
fn recipe() -> DecoderSpec {
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
            intermediate_size: 16,
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
        embedding_scale: None,
        final_logit_softcap: None,
    }
}
