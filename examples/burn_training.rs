//! Train a transformer experiment using Burn's own supervised Learner.
//!
//! cargo run --release --locked --features train --example burn_training
//! This small symbolic classification task needs no downloads.

#[cfg(not(all(feature = "train", feature = "flex")))]
fn main() {
    eprintln!("enable `train,flex` to run the native Burn Learner example");
    std::process::exit(1);
}

#[cfg(all(feature = "train", feature = "flex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    experiment::run()
}

#[cfg(all(feature = "train", feature = "flex"))]
mod experiment {
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
            ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput,
            TrainStep,
            metric::{AccuracyMetric, LossMetric},
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

    type Cpu = Flex<f32, i32>;
    type Training = Autodiff<Cpu>;

    // Burn owns and discovers every parameter, including the inserted adapter.
    #[derive(Module, Debug)]
    struct Experiment<B: Backend> {
        backbone: ModularCausalLm<B>,
        adapter: Linear<B>,
        dropout: Dropout,
        classifier: Linear<B>,
    }

    impl<B: Backend> Experiment<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                backbone: ModularCausalLm::init(recipe(), device).expect("valid fixed recipe"),
                adapter: LinearConfig::new(8, 8).init(device),
                // Regularization is explicit in the experiment. Burn disables
                // this Dropout when `.valid()` converts to the inner backend.
                dropout: DropoutConfig::new(0.1).init(),
                classifier: LinearConfig::new(8, 2).init(device),
            }
        }

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
        type Input = Batch<B>;
        type Output = ClassificationOutput<B>;
        fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
            let output = self.classify(batch);
            TrainOutput::new(self, output.loss.clone().backward(), output)
        }
    }

    impl<B: Backend> InferenceStep for Experiment<B> {
        type Input = Batch<B>;
        type Output = ClassificationOutput<B>;
        fn step(&self, batch: Self::Input) -> Self::Output {
            self.classify(batch)
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
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

    #[derive(Clone, Debug)]
    struct Item {
        tokens: [i32; 3],
        label: i32,
    }
    #[derive(Clone, Debug)]
    struct Batch<B: Backend> {
        tokens: Tensor<B, 2, Int>,
        labels: Tensor<B, 1, Int>,
    }
    #[derive(Clone)]
    struct SymbolBatcher;
    impl<B: Backend> Batcher<B, Item, Batch<B>> for SymbolBatcher {
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
        }
    }
}
