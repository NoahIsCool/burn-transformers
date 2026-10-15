//! Closed, backend-generic dispatch over executable model families.
//!
//! Checkpoint family selection belongs at the runtime boundary, not inside
//! shared attention or feed-forward implementations. [`AutoCausalLm`] and
//! [`AutoSeq2SeqLm`] therefore contain concrete modules and dispatch each public
//! operation by enum variant. Their matching state enums prevent a cache built
//! for one architecture from being passed to another architecture by accident.
//!
//! The registry advertises only executable graphs. Configuration-only schemas
//! for families still under construction remain available from their family
//! modules, but are intentionally rejected here until their forward and
//! checkpoint paths meet the same contract.
//!
//! The dispatcher does not alter tensor axes, floating precision, masks, or attention
//! equations. Causal inputs remain `[batch, token]`, residuals `[batch, token, hidden]`,
//! and logits `[batch, selected_token, vocabulary]`. Family implementations own
//! validation and numerical contracts. Parameter enums stay separate from external
//! state enums; mismatched causal model/state variants fail before execution.
//!
//! # One list of causal families
//!
//! Every causal family is declared once, in the `causal_registry!` invocation
//! below. The macro generates the four parallel enums ([`AutoCausalConfig`],
//! [`AutoCausalConfigRef`], [`AutoCausalDecodeState`], [`AutoCausalLm`]),
//! `MODEL_TYPES`, and every dispatch `match`, so they cannot drift apart.
//! Adding a dense Llama-style decoder, for example, is one registry entry plus
//! one `CausalFamily` implementation in `auto/family.rs` describing how its
//! checkpoint is planned.
//!
//! # Experiments through the automatic enums
//!
//! Activation hooks and staged execution dispatch through the family-generic
//! [`HookedModel`] and [`StagedModel`] traits, so a loaded checkpoint is
//! inspected exactly as its concrete family would be:
//!
//! - [`AutoCausalLm`] implements both traits. Each registry entry declares
//!   `experiments: supported` (the family implements both traits and every
//!   call delegates to them) or `experiments: pending` (the family has no hook
//!   implementation yet; every experiment call fails with
//!   [`Error::BackendCapability`] naming its `model_type`, before any
//!   callback runs).
//! - [`AutoSeq2SeqLm`] implements [`HookedModel`] over the complete
//!   encoder-decoder pass.
//!
//! The dispatch adds no computation: a hooked pass through the enum is the
//! family's own hooked pass, and its catalog is the family's own catalog.

// Dispatch enums hold whole parameter graphs inline (Burn cannot traverse a
// boxed `Module`), so their variants differ in size by design. The allowance is
// module-wide because `#[derive(Module)]` also emits a record enum from inside
// the registry macro, where an item-level attribute cannot reach it.
#![allow(clippy::large_enum_variant)]

use std::{ops::Range, path::Path};

use burn::{module::Module, tensor::backend::Backend};

use crate::{
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput,
    DecodeState, EncoderOutput, Error, ForwardOptions, LoadedModel, Result, Seq2SeqInput,
    Seq2SeqLanguageModel, Seq2SeqOutput,
    architecture::{HookedModel, StagedModel},
    cache::CacheConfig,
    experiment::HiddenState,
    intervention::{Hooks, PointInfo, Stack},
};

use super::{
    deepseek_v2::{DeepseekV2Config, DeepseekV2DecodeState, DeepseekV2ForCausalLm},
    llama::{LlamaDecodeState, LlamaFamilyConfig, LlamaForCausalLm},
    muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerForCausalLm, MuseGlimmerTextConfig},
    nemotron_h::{NemotronHConfig, NemotronHDecodeState, NemotronHForCausalLm},
    qwen3_5::{Qwen3_5DecodeState, Qwen3_5ForCausalLm, Qwen3_5TextConfig},
    qwen3_moe::{Qwen3MoeConfig, Qwen3MoeDecodeState, Qwen3MoeForCausalLm},
    t5::{T5Config, T5DecodeState, T5ForConditionalGeneration},
};
use crate::models::config::{
    ConfigError, ConfigLoadError, ModelConfig, Validate, parse_text_config,
};

// Submodules document themselves with `//!`; an outer `///` here would make
// rustdoc resolve their intra-doc links from this module's scope.
mod defaults; // Verified Hugging Face generation-default arithmetic.
mod family; // Family-specific steps of strict automatic causal loading.
mod load; // Local artifact loading and generation-default normalization.

pub use load::AutoLoadOptions;

// The causal registry.

/// Selects one of two expansions by a registry entry's `experiments` capability.
///
/// `supported` keeps the first block and `pending` the second, so a generated
/// `match` arm either delegates to the family's [`HookedModel`] or
/// [`StagedModel`] implementation or reports the capability gap. The kept
/// tokens expand as one block expression. Any other word is a compile-time
/// error, which keeps the registry's vocabulary closed.
macro_rules! by_experiments {
    (supported { $($supported:tt)* } else { $($pending:tt)* }) => {{ $($supported)* }};
    (pending { $($supported:tt)* } else { $($pending:tt)* }) => {{ $($pending)* }};
}

/// Defines the causal registry from one list of executable families.
///
/// Each entry gives the enum variant, its documentation, the checkpoint
/// `model_type`, optional `aliases` (further `model_type` values executed by
/// the same family, such as `mistral` for the Llama family), the family's
/// model, configuration, and decode-state types, and its `experiments`
/// capability (`supported` or `pending`, see the module documentation). The
/// expansion defines [`AutoCausalConfig`], [`AutoCausalConfigRef`],
/// [`AutoCausalDecodeState`], and [`AutoCausalLm`] with one variant per entry,
/// together with every method whose body is a per-family `match`, including
/// the [`HookedModel`] and [`StagedModel`] dispatch.
macro_rules! causal_registry {
    ($(
        $(#[$family_doc:meta])*
        $variant:ident {
            model_type: $model_type:literal,
            $(aliases: [$($alias:literal),+ $(,)?],)?
            model: $model:ident,
            config: $config:ident,
            state: $state:ident,
            experiments: $experiments:ident $(,)?
        }
    ),+ $(,)?) => {
        /// Registry-level configuration associated with [`CausalLanguageModel`].
        ///
        /// The detailed checkpoint configuration remains available through
        /// [`AutoCausalLm::config`]. This lightweight discriminator lets generic
        /// code use the architecture trait without duplicating a model's complete
        /// config inside the dispatch enum. Each variant names one executable
        /// causal family.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum AutoCausalConfig {
            $( $(#[$family_doc])* $variant, )+
        }

        impl AutoCausalConfig {
            /// Every registered causal family, in registry order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Returns the family's primary checkpoint `model_type` discriminator.
            ///
            /// A family registered under aliases reports its primary type; the
            /// exact checkpoint value is in its concrete configuration.
            pub const fn model_type(self) -> &'static str {
                match self {
                    $(Self::$variant => $model_type,)+
                }
            }

            /// Returns the family registered for a checkpoint `model_type` or alias, if any.
            pub fn from_model_type(model_type: &str) -> Option<Self> {
                match model_type {
                    $($model_type $($(| $alias)+)? => Some(Self::$variant),)+
                    _ => None,
                }
            }

            /// Whether the family's activation hooks and staged execution are available
            /// through [`AutoCausalLm`] ([`HookedModel`] and [`StagedModel`]).
            ///
            /// Families without them fail every experiment call with
            /// [`Error::BackendCapability`]; ordinary inference is unaffected.
            pub const fn supports_experiments(self) -> bool {
                match self {
                    $(Self::$variant => by_experiments!($experiments { true } else { false }),)+
                }
            }
        }

        impl ArchitectureConfig for AutoCausalConfig {
            /// Checkpoint discriminators (primary types and aliases) accepted by the registry.
            const MODEL_TYPES: &'static [&'static str] = &[$($model_type, $($($alias,)+)?)+];

            /// Registry discriminators are constructed only for executable families.
            fn validate(&self) -> Result<()> {
                Ok(())
            }
        }

        /// Immutable architecture configuration borrowed from an automatic causal model.
        #[derive(Debug, Clone, Copy)]
        #[non_exhaustive]
        pub enum AutoCausalConfigRef<'a> {
            $(
                $(#[$family_doc])*
                $variant(
                    /// Borrowed configuration owned by the selected model.
                    &'a $config,
                ),
            )+
        }

        impl AutoCausalConfigRef<'_> {
            /// Returns the registered family of the borrowed configuration.
            pub fn family(self) -> AutoCausalConfig {
                match self {
                    $(Self::$variant(_) => AutoCausalConfig::$variant,)+
                }
            }

            /// Returns the family's primary checkpoint `model_type` discriminator.
            ///
            /// See [`AutoCausalConfig::model_type`] for families with aliases.
            pub fn model_type(self) -> &'static str {
                self.family().model_type()
            }

            /// Returns the model vocabulary size.
            pub fn vocabulary_size(self) -> usize {
                match self {
                    $(Self::$variant(config) => config.vocab_size,)+
                }
            }
        }

        // External state dispatch; no caches participate in Burn parameter traversal.

        /// External decode state paired with an [`AutoCausalLm`] variant.
        ///
        /// States are boxed so the enum stays small however large one family's
        /// cache bookkeeping is.
        #[derive(Debug)]
        #[non_exhaustive]
        pub enum AutoCausalDecodeState<B: Backend> {
            $(
                $(#[$family_doc])*
                $variant(
                    /// The family's own cache and sequence-length state.
                    Box<$state<B>>,
                ),
            )+
        }

        impl<B: Backend> DecodeState<B> for AutoCausalDecodeState<B> {
            /// Returns the selected state's batch width, without inspecting parameter tensors.
            fn batch_size(&self) -> usize {
                match self {
                    $(Self::$variant(state) => state.batch_size(),)+
                }
            }

            /// Returns the selected architecture's declared decode capacity in tokens.
            fn capacity(&self) -> usize {
                match self {
                    $(Self::$variant(state) => state.capacity(),)+
                }
            }

            /// Borrows committed per-row logical lengths from the concrete decode state.
            fn lengths(&self) -> &[usize] {
                match self {
                    $(Self::$variant(state) => state.lengths(),)+
                }
            }

            /// Delegates reset to the concrete state, preserving its architecture-specific invariants.
            fn reset(&mut self) -> Result<()> {
                match self {
                    $(Self::$variant(state) => state.reset(),)+
                }
            }

            /// Reorders batch rows through the concrete state's validation and update path.
            fn reorder(&mut self, indices: &[usize]) -> Result<()> {
                match self {
                    $(Self::$variant(state) => state.reorder(indices),)+
                }
            }

            /// Delegates truncation to the architecture; unsupported recurrent rewinds remain errors.
            fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
                match self {
                    $(Self::$variant(state) => state.truncate(lengths),)+
                }
            }

            /// Borrows the device on which the concrete decode state is allocated.
            fn device(&self) -> &B::Device {
                match self {
                    $(Self::$variant(state) => state.device(),)+
                }
            }
        }

        // Executable module dispatch and architecture-trait adapters.

        /// Automatic dispatch over all causal architectures with executable graphs.
        ///
        /// Variants hold their parameter graphs inline: Burn 0.21 does not
        /// implement `Module` for `Box<M>`, so boxing them would prevent
        /// derive-generated device, dtype, and parameter traversal.
        #[derive(Module, Debug)]
        #[non_exhaustive]
        pub enum AutoCausalLm<B: Backend> {
            $(
                $(#[$family_doc])*
                $variant(
                    /// Parameter graph of the selected family.
                    $model<B>,
                ),
            )+
        }

        impl<B: Backend> AutoCausalLm<B> {
            /// Checkpoint model types (primary types and aliases) accepted by this executable registry.
            pub const MODEL_TYPES: &'static [&'static str] =
                <AutoCausalConfig as ArchitectureConfig>::MODEL_TYPES;

            /// Parses a direct or wrapped configuration and initializes its lazy graph.
            pub fn from_config_json(json: &str, device: &B::Device) -> Result<Self> {
                match detect_model_type(json)?.as_str() {
                    $(
                        $model_type $($(| $alias)+)? => {
                            let config = <$config as ModelConfig>::from_json_str(json)?;
                            $model::init(config, device).map(Self::$variant)
                        }
                    )+
                    model_type => Err(unsupported_causal_model_type(model_type)),
                }
            }

            /// Returns the registered family of the selected model.
            pub fn family(&self) -> AutoCausalConfig {
                match self {
                    $(Self::$variant(_) => AutoCausalConfig::$variant,)+
                }
            }

            /// Returns the concrete architecture configuration without cloning it.
            pub fn config(&self) -> AutoCausalConfigRef<'_> {
                match self {
                    $(Self::$variant(model) => AutoCausalConfigRef::$variant(model.config()),)+
                }
            }

            /// Evaluates a stateless causal forward pass through the selected variant.
            pub fn forward(
                &self,
                input: CausalLmInput<B>,
                options: &ForwardOptions,
            ) -> Result<CausalLmOutput<B>> {
                match self {
                    $(Self::$variant(model) => CausalLanguageModel::forward(model, input, options),)+
                }
            }

            /// Allocates architecture-matched external decode state.
            pub fn new_decode_state(
                &self,
                config: CacheConfig,
                batch_size: usize,
                device: &B::Device,
            ) -> Result<AutoCausalDecodeState<B>> {
                match self {
                    $(
                        Self::$variant(model) => {
                            CausalLanguageModel::new_decode_state(model, config, batch_size, device)
                                .map(|state| AutoCausalDecodeState::$variant(Box::new(state)))
                        }
                    )+
                }
            }

            /// Prefills architecture-matched state from a prompt.
            // With a single registered family the mismatch arm is unreachable.
            #[allow(unreachable_patterns)]
            pub fn prefill(
                &self,
                input: CausalLmInput<B>,
                state: &mut AutoCausalDecodeState<B>,
                options: &ForwardOptions,
            ) -> Result<CausalLmOutput<B>> {
                match (self, state) {
                    $(
                        (Self::$variant(model), AutoCausalDecodeState::$variant(state)) => {
                            CausalLanguageModel::prefill(model, input, state.as_mut(), options)
                        }
                    )+
                    _ => Err(foreign_causal_state_error()),
                }
            }

            /// Decodes one or more continuation tokens against matched state.
            // With a single registered family the mismatch arm is unreachable.
            #[allow(unreachable_patterns)]
            pub fn decode(
                &self,
                input: DecodeInput<B>,
                state: &mut AutoCausalDecodeState<B>,
                options: &ForwardOptions,
            ) -> Result<CausalLmOutput<B>> {
                match (self, state) {
                    $(
                        (Self::$variant(model), AutoCausalDecodeState::$variant(state)) => {
                            CausalLanguageModel::decode(model, input, state.as_mut(), options)
                        }
                    )+
                    _ => Err(foreign_causal_state_error()),
                }
            }

            /// Runs the shared strict loading pipeline for the family named by `model_type`.
            fn load_registered_family(
                model_type: &str,
                root: &Path,
                json: &str,
                options: &AutoLoadOptions,
                device: &B::Device,
            ) -> Result<LoadedModel<Self>> {
                match model_type {
                    $(
                        $model_type $($(| $alias)+)? => load::load_causal_family::<B, $model<B>>(
                            root, json, model_type, options, device,
                        ),
                    )+
                    model_type => Err(unsupported_causal_model_type(model_type)),
                }
            }

            /// Lists the selected model's stateless activation points, in emission order.
            ///
            /// This is the family's own [`HookedModel`] catalog. A family whose
            /// registry entry is `experiments: pending` returns
            /// [`Error::BackendCapability`] naming its `model_type` rather than an
            /// empty, misleading catalog (see
            /// [`AutoCausalConfig::supports_experiments`]). Loaded checkpoints
            /// expose this method through `loaded.model()`.
            pub fn activation_points(&self) -> Result<Vec<PointInfo>> {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        HookedModel::<B>::activation_points(model)
                    } else {
                        let _ = model;
                        Err(self.experiments_unavailable())
                    }),)+
                }
            }

            /// Observes or changes activations during the selected model's stateless forward.
            ///
            /// Delegates to the family's [`HookedModel::forward_with_hooks`], so
            /// its catalog, edit constraints, and numerics apply unchanged. A
            /// family without hook support fails before any callback or model
            /// computation, even for an empty collection. Use [`Self::forward`]
            /// for ordinary execution. Failed calls invalidate captures left from
            /// earlier runs of these hooks.
            pub fn forward_with_hooks(
                &self,
                input: CausalLmInput<B>,
                options: &ForwardOptions,
                hooks: &mut Hooks<'_, B>,
            ) -> Result<CausalLmOutput<B>> {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        HookedModel::<B>::forward_with_hooks(model, input, options, hooks)
                    } else {
                        let _ = model;
                        reject_hooks(hooks, self.experiments_unavailable())
                    }),)+
                }
            }
        }

        impl<B: Backend> StagedModel<B> for AutoCausalLm<B> {
            /// Number of blocks the selected family can stage; zero without staged support.
            fn num_layers(&self) -> usize {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        StagedModel::<B>::num_layers(model)
                    } else {
                        let _ = model;
                        0
                    }),)+
                }
            }

            /// The family's staged stack; every registered causal family is a decoder.
            fn stack(&self) -> Stack {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        StagedModel::<B>::stack(model)
                    } else {
                        let _ = model;
                        Stack::Decoder
                    }),)+
                }
            }

            /// Delegates to the family's [`StagedModel::embed_with_hooks`].
            fn embed_with_hooks(
                &self,
                input: CausalLmInput<B>,
                hooks: &mut Hooks<'_, B>,
            ) -> Result<HiddenState<B>> {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        StagedModel::<B>::embed_with_hooks(model, input, hooks)
                    } else {
                        let _ = model;
                        reject_hooks(hooks, self.experiments_unavailable())
                    }),)+
                }
            }

            /// Delegates to the family's [`StagedModel::run_layers_with_hooks`].
            fn run_layers_with_hooks(
                &self,
                state: HiddenState<B>,
                range: Range<usize>,
                hooks: &mut Hooks<'_, B>,
            ) -> Result<HiddenState<B>> {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        StagedModel::<B>::run_layers_with_hooks(model, state, range, hooks)
                    } else {
                        let _ = model;
                        reject_hooks(hooks, self.experiments_unavailable())
                    }),)+
                }
            }

            /// Delegates to the family's [`StagedModel::normalize_with_hooks`].
            fn normalize_with_hooks(
                &self,
                state: HiddenState<B>,
                hooks: &mut Hooks<'_, B>,
            ) -> Result<HiddenState<B>> {
                match self {
                    $(Self::$variant(model) => by_experiments!($experiments {
                        StagedModel::<B>::normalize_with_hooks(model, state, hooks)
                    } else {
                        let _ = model;
                        reject_hooks(hooks, self.experiments_unavailable())
                    }),)+
                }
            }
        }

        $(
            impl<B: Backend> From<$model<B>> for AutoCausalLm<B> {
                /// Wraps the concrete module without copying, loading, or transforming its parameters.
                fn from(model: $model<B>) -> Self {
                    Self::$variant(model)
                }
            }
        )+
    };
}

// `experiments: pending` marks a family whose `HookedModel`/`StagedModel`
// implementation has not landed on this branch; switching it to `supported`
// is the whole integration step once it has.
causal_registry! {
    /// DeepSeek-V2 decoder with multi-head latent attention and sparse experts.
    DeepseekV2 {
        model_type: "deepseek_v2",
        model: DeepseekV2ForCausalLm,
        config: DeepseekV2Config,
        state: DeepseekV2DecodeState,
        experiments: pending,
    },
    /// Muse Glimmer dense decoder with sliding/full hybrid attention.
    MuseGlimmer {
        model_type: "muse_glimmer_text",
        model: MuseGlimmerForCausalLm,
        config: MuseGlimmerTextConfig,
        state: MuseGlimmerDecodeState,
        experiments: pending,
    },
    /// NVIDIA Nemotron-H decoder with interleaved Mamba2, attention, dense, and MoE blocks.
    NemotronH {
        model_type: "nemotron_h",
        model: NemotronHForCausalLm,
        config: NemotronHConfig,
        state: NemotronHDecodeState,
        experiments: supported,
    },
    /// Qwen3.5/Qwen3.8 gated-DeltaNet and full-attention hybrid decoder.
    Qwen3_5 {
        model_type: "qwen3_5_text",
        model: Qwen3_5ForCausalLm,
        config: Qwen3_5TextConfig,
        state: Qwen3_5DecodeState,
        experiments: supported,
    },
    /// Qwen3 decoder with token-routed sparse SwiGLU experts.
    Qwen3Moe {
        model_type: "qwen3_moe",
        model: Qwen3MoeForCausalLm,
        config: Qwen3MoeConfig,
        state: Qwen3MoeDecodeState,
        experiments: supported,
    },
    /// Llama-family dense decoder: Llama 2/3.x, Mistral, Qwen2/2.5, Qwen3, SmolLM3, and OLMo 2.
    Llama {
        model_type: "llama",
        aliases: ["mistral", "qwen2", "qwen3", "smollm3", "olmo2"],
        model: LlamaForCausalLm,
        config: LlamaFamilyConfig,
        state: LlamaDecodeState,
        experiments: supported,
    },
}

/// Reports a causal `model_type` with no executable registry entry.
fn unsupported_causal_model_type(model_type: &str) -> Error {
    Error::UnsupportedArchitecture(format!(
        "causal model_type `{model_type}` is not executable; supported values: {}",
        <AutoCausalConfig as ArchitectureConfig>::MODEL_TYPES.join(", ")
    ))
}

/// Reports a decode state allocated by a different causal family.
fn foreign_causal_state_error() -> Error {
    Error::Cache("automatic causal decode state belongs to a different architecture".into())
}

impl<B: Backend> AutoCausalLm<B> {
    /// Names the selected family when its hooks and staged execution are unavailable.
    #[allow(
        dead_code,
        reason = "called only from `experiments: pending` registry entries, which may be none"
    )]
    fn experiments_unavailable(&self) -> Error {
        let supported = AutoCausalConfig::ALL
            .iter()
            .filter(|family| family.supports_experiments())
            .map(|family| family.model_type())
            .collect::<Vec<_>>();
        Error::BackendCapability(format!(
            "activation hooks and staged execution are not implemented for model_type `{}`; automatic causal families with experiment support: {}",
            self.config().model_type(),
            supported.join(", ")
        ))
    }
}

/// Fails a hook-scoped call for a family without experiment support.
///
/// An empty scope runs first so captures left in `hooks` by earlier runs are
/// cleared, exactly as any failed hooked pass clears them; its own result (a
/// preflight rejection when actions are registered) is superseded by `error`,
/// which names the family. No callback or model computation runs.
#[allow(
    dead_code,
    reason = "called only from `experiments: pending` registry entries, which may be none"
)]
fn reject_hooks<B: Backend, T>(hooks: &mut Hooks<'_, B>, error: Error) -> Result<T> {
    let _cleared = hooks.run(Vec::new(), |_| Ok(()));
    Err(error)
}

impl<B: Backend> HookedModel<B> for AutoCausalLm<B> {
    /// Token IDs, optional padding mask, and optional positions.
    type Input = CausalLmInput<B>;
    /// Selected logits and requested diagnostics.
    type Output = CausalLmOutput<B>;

    /// Runs [`AutoCausalLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        AutoCausalLm::activation_points(self)
    }

    /// Runs [`AutoCausalLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> CausalLanguageModel<B> for AutoCausalLm<B> {
    /// Family discriminator used by generic causal-model clients.
    type Config = AutoCausalConfig;
    /// External state tagged with its concrete causal family.
    type State = AutoCausalDecodeState<B>;

    /// Delegates the causal trait entry point to the checked enum dispatch.
    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::forward(self, input, options)
    }

    /// Allocates external state through the checked inherent method without registering caches as parameters.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        AutoCausalLm::new_decode_state(self, config, batch_size, device)
    }

    /// Dispatches prompt prefill and rejects a state belonging to another architecture.
    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::prefill(self, input, state, options)
    }

    /// Dispatches continuation tokens with architecture-matched causal decode state.
    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        AutoCausalLm::decode(self, input, state, options)
    }
}

// Encoder-decoder registry.

/// Registry-level configuration associated with [`Seq2SeqLanguageModel`].
///
/// The complete checkpoint configuration remains available through
/// [`AutoSeq2SeqLm::config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutoSeq2SeqConfig {
    /// T5, FLAN-T5, T5 v1.1, and UL2 runtime.
    T5,
}

impl ArchitectureConfig for AutoSeq2SeqConfig {
    /// Checkpoint discriminator accepted by the encoder-decoder configuration registry.
    const MODEL_TYPES: &'static [&'static str] = &["t5"];

    /// Registry discriminators are constructed only for executable families.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Immutable architecture configuration borrowed from an automatic seq2seq model.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AutoSeq2SeqConfigRef<'a> {
    /// T5, FLAN-T5, T5 v1.1, or UL2 configuration.
    T5(
        /// Borrowed T5 configuration owned by the selected model.
        &'a T5Config,
    ),
}

impl AutoSeq2SeqConfigRef<'_> {
    /// Returns the exact checkpoint `model_type` discriminator.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::T5(_) => "t5",
        }
    }

    /// Returns the model vocabulary size.
    pub fn vocabulary_size(self) -> usize {
        match self {
            Self::T5(config) => config.vocab_size,
        }
    }
}

// External encoder-decoder state dispatch.

/// External decoder/cross-attention state paired with [`AutoSeq2SeqLm`].
#[derive(Debug)]
#[non_exhaustive]
pub enum AutoSeq2SeqDecodeState<B: Backend> {
    /// T5 decoder self-attention and immutable encoder cross-attention state.
    T5(
        /// T5 decoder cache and fixed encoder cross-attention state.
        T5DecodeState<B>,
    ),
}

impl<B: Backend> DecodeState<B> for AutoSeq2SeqDecodeState<B> {
    /// Returns the selected state's batch width, without inspecting parameter tensors.
    fn batch_size(&self) -> usize {
        match self {
            Self::T5(state) => state.batch_size(),
        }
    }

    /// Returns the selected architecture's declared decode capacity in tokens.
    fn capacity(&self) -> usize {
        match self {
            Self::T5(state) => state.capacity(),
        }
    }

    /// Borrows committed per-row logical lengths from the concrete decode state.
    fn lengths(&self) -> &[usize] {
        match self {
            Self::T5(state) => state.lengths(),
        }
    }

    /// Delegates reset to the concrete state, preserving its architecture-specific invariants.
    fn reset(&mut self) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::reset(state),
        }
    }

    /// Reorders batch rows through the concrete state's validation and update path.
    fn reorder(&mut self, indices: &[usize]) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::reorder(state, indices),
        }
    }

    /// Delegates truncation to the architecture; unsupported recurrent rewinds remain errors.
    fn truncate(&mut self, lengths: &[usize]) -> Result<()> {
        match self {
            Self::T5(state) => DecodeState::truncate(state, lengths),
        }
    }

    /// Borrows the device on which the concrete decode state is allocated.
    fn device(&self) -> &B::Device {
        match self {
            Self::T5(state) => state.device(),
        }
    }
}

/// Automatic dispatch over all encoder-decoder architectures with executable graphs.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum AutoSeq2SeqLm<B: Backend> {
    /// T5, FLAN-T5, T5 v1.1, and UL2 encoder-decoder graph.
    T5(
        /// T5 encoder-decoder parameter graph selected for execution.
        T5ForConditionalGeneration<B>,
    ),
}

impl<B: Backend> AutoSeq2SeqLm<B> {
    /// Checkpoint model types currently accepted by this executable registry.
    pub const MODEL_TYPES: &'static [&'static str] = &["t5"];

    /// Parses a direct configuration and initializes its lazy graph.
    pub fn from_config_json(json: &str, device: &B::Device) -> Result<Self> {
        match detect_model_type(json)?.as_str() {
            "t5" => {
                let config = T5Config::from_json_str(json)?;
                T5ForConditionalGeneration::init(config, device).map(Self::T5)
            }
            model_type => Err(Error::UnsupportedArchitecture(format!(
                "seq2seq model_type `{model_type}` is not executable; supported values: {}",
                Self::MODEL_TYPES.join(", ")
            ))),
        }
    }

    /// Returns the concrete architecture configuration without cloning it.
    pub fn config(&self) -> AutoSeq2SeqConfigRef<'_> {
        match self {
            Self::T5(model) => AutoSeq2SeqConfigRef::T5(model.config()),
        }
    }

    /// Encodes source tokens bidirectionally through the selected variant.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        match self {
            Self::T5(model) => Seq2SeqLanguageModel::encode(model, source, options),
        }
    }

    /// Evaluates the complete stateless encoder-decoder graph.
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match self {
            Self::T5(model) => Seq2SeqLanguageModel::forward(model, input, options),
        }
    }

    /// Lists the selected model's encoder and decoder activation points, in emission order.
    ///
    /// This is the family's own [`HookedModel`] catalog: every encoder point,
    /// then every decoder point. Loaded checkpoints expose this method through
    /// `loaded.model()`.
    pub fn activation_points(&self) -> Result<Vec<PointInfo>> {
        match self {
            Self::T5(model) => HookedModel::<B>::activation_points(model),
        }
    }

    /// Observes or changes tensors during the complete stateless encoder-decoder graph.
    ///
    /// Delegates to the family's [`HookedModel::forward_with_hooks`], including
    /// its documented mask and cache restrictions; see
    /// [`T5ForConditionalGeneration::forward_with_hooks`].
    pub fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        match self {
            Self::T5(model) => HookedModel::<B>::forward_with_hooks(model, input, options, hooks),
        }
    }

    /// Allocates architecture-matched decoder state for the selected model.
    pub fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<AutoSeq2SeqDecodeState<B>> {
        match self {
            Self::T5(model) => {
                Seq2SeqLanguageModel::new_decode_state(model, config, batch_size, device)
                    .map(AutoSeq2SeqDecodeState::T5)
            }
        }
    }

    /// Prefills an architecture-matched decoder from reusable encoder output.
    pub fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut AutoSeq2SeqDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match (self, state) {
            (Self::T5(model), AutoSeq2SeqDecodeState::T5(state)) => {
                Seq2SeqLanguageModel::prefill_decoder(model, encoder, decoder, state, options)
            }
        }
    }

    /// Decodes continuation tokens using matched encoder and decoder state.
    pub fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut AutoSeq2SeqDecodeState<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        match (self, state) {
            (Self::T5(model), AutoSeq2SeqDecodeState::T5(state)) => {
                Seq2SeqLanguageModel::decode(model, encoder, input, state, options)
            }
        }
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for AutoSeq2SeqLm<B> {
    /// Family discriminator used by generic encoder-decoder clients.
    type Config = AutoSeq2SeqConfig;
    /// External decoder state tagged with its encoder-decoder family.
    type State = AutoSeq2SeqDecodeState<B>;

    /// Dispatches bidirectional source encoding through the sequence-to-sequence trait.
    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        AutoSeq2SeqLm::encode(self, source, options)
    }

    /// Delegates the sequence-to-sequence trait entry point to the concrete graph.
    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::forward(self, input, options)
    }

    /// Allocates external state through the checked inherent method without registering caches as parameters.
    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        AutoSeq2SeqLm::new_decode_state(self, config, batch_size, device)
    }

    /// Dispatches decoder prompt prefill using reusable encoder output and external state.
    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::prefill_decoder(self, encoder, decoder, state, options)
    }

    /// Dispatches continuation tokens with the reusable encoder output and decoder state.
    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::decode(self, encoder, input, state, options)
    }
}

impl<B: Backend> HookedModel<B> for AutoSeq2SeqLm<B> {
    /// Source and target tokens of one complete pass.
    type Input = Seq2SeqInput<B>;
    /// Encoder states and decoder logits.
    type Output = Seq2SeqOutput<B>;

    /// Runs [`AutoSeq2SeqLm::activation_points`].
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        AutoSeq2SeqLm::activation_points(self)
    }

    /// Runs [`AutoSeq2SeqLm::forward_with_hooks`].
    fn forward_with_hooks(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Seq2SeqOutput<B>> {
        AutoSeq2SeqLm::forward_with_hooks(self, input, options, hooks)
    }
}

impl<B: Backend> From<T5ForConditionalGeneration<B>> for AutoSeq2SeqLm<B> {
    /// Wraps the concrete module without copying, loading, or transforming its parameters.
    fn from(model: T5ForConditionalGeneration<B>) -> Self {
        Self::T5(model)
    }
}

/// Text configuration selected exactly as a family parser would select it.
///
/// Parsing goes through [`parse_text_config`], the path every family
/// `from_json_str` takes, so wrapper selection (`text_config`,
/// `language_config`, `llm_config`) and normalization of NVIDIA's legacy bare
/// `Infinity` literal cannot drift between family detection and family
/// parsing.
#[derive(serde::Deserialize)]
#[serde(transparent)]
struct SelectedTextConfig(
    /// The selected nested or direct configuration object.
    serde_json::Value,
);

impl Validate for SelectedTextConfig {
    /// Detection checks only the discriminator, after selection.
    fn validate(&self) -> core::result::Result<(), ConfigError> {
        Ok(())
    }
}

/// Extracts the architecture discriminator from direct or text-wrapped JSON.
pub(super) fn detect_model_type(json: &str) -> Result<String> {
    let SelectedTextConfig(candidate) =
        parse_text_config::<SelectedTextConfig>(json).map_err(|error| match error {
            ConfigLoadError::Json(error) => {
                Error::configuration("auto.config_json", format!("invalid JSON: {error}"))
            }
            ConfigLoadError::Validation(error) => {
                Error::configuration("auto.config_json", error.to_string())
            }
        })?;
    candidate
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::configuration(
                "auto.model_type",
                "configuration must contain a string model_type",
            )
        })
}

#[cfg(test)]
#[path = "auto_tests.rs"]
/// Registry parsing, family dispatch, and lifecycle regression tests.
mod tests;
