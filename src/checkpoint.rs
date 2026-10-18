//! Safe, bounded discovery and planning for local SafeTensors checkpoints.
//!
//! Model repositories are input data, not trusted code. In particular, a Hugging
//! Face shard index controls filenames and tensor names. This module therefore
//! validates paths before opening them, bounds all metadata allocations, verifies
//! every byte range in each SafeTensors header, and compares the complete observed
//! tensor set with an explicit architecture-owned mapping plan.
//!
//! Discovery is deliberately separate from parameter application. Pass one reads
//! only the index and shard headers and produces a [`CheckpointManifest`]. Pass two
//! is described by a [`LoadPlan`] and executed through [`execute_load_plan`]. The
//! executor stages one operation at a time: one source payload and its
//! transformation output (both sources for an MXFP4 blocks/scales pair, or the
//! two payloads of an equivalence comparison), bounded by
//! [`LoadPlan::required_host_staging_bytes`]. It then transfers the resulting
//! [`burn::tensor::TensorData`] into an architecture-owned [`TensorSink`]. That
//! separation makes malformed artifacts fail before expensive model allocations
//! occur and avoids staging the complete checkpoint in host memory.
//!
//! # Loading a concrete family
//!
//! [`crate::models::AutoCausalLm`] chooses the family from `config.json`. When
//! the family is known, initialize its model from the configuration and call
//! one of the two methods every family model offers:
//!
//! | Method | Use |
//! | --- | --- |
//! | `load_checkpoint(root, &LoadOptions)` | Strictly load a local checkpoint, detecting any parameter layout the family publishes. |
//! | `load_checkpoint_with(root, &LoadOptions, family_options)` | The same with the family's explicit `…CheckpointOptions`, for families that have one. |
//!
//! Both perform no network access and return an [`ExecutionOutcome`] (the
//! audited [`LoadReport`] and [`ExecutionStats`]) or the family's
//! `…CheckpointLoadError`. Every such error has variants for:
//!
//! - an invalid configuration;
//! - a [`CheckpointError`] from discovery or strict planning. Qwen3.5 and Muse
//!   also report a failed layout detection this way, as
//!   [`CheckpointError::LanguageNamespaceNotFound`] or
//!   [`CheckpointError::AmbiguousLanguageNamespace`], and Gemma as
//!   [`CheckpointError::LanguageEmbeddingNotFound`] or
//!   [`CheckpointError::AmbiguousLanguageEmbedding`]. A plan that executes a
//!   dtype the family declares its checkpoints overflow (Gemma 3 in F16) is
//!   [`CheckpointError::RangeHazard`] unless [`LoadOptions::range_hazards`]
//!   accepts it ([`RangeHazard`], [`check_range_hazards`]);
//! - family-specific preconditions: GPT-2 reports a failed layout detection
//!   as `Namespace` and rejects non-floating storage as `NonFloatParameter`,
//!   and Qwen3.5 rejects MTP options that disagree with the model's MTP graph
//!   as `UnloadedMtpGraph` or `MissingMtpGraph`;
//! - an execution failure, whose partial report `execution_failure()` returns.
//!
//! Every variant except the execution failure is raised before any parameter
//! is written; after an execution failure the model must be discarded. The
//! family's `…LoadError` is the per-tensor error its [`TensorSink`] raises
//! inside that execution failure.
//!
//! | Family model | `load_checkpoint` detects | `load_checkpoint_with` options |
//! | --- | --- | --- |
//! | [`Qwen3_5ForCausalLm`](crate::models::qwen3_5::Qwen3_5ForCausalLm) | multimodal or text-only layout; MTP tensors exactly when the model owns an MTP graph | [`Qwen3_5CheckpointOptions`](crate::models::qwen3_5::Qwen3_5CheckpointOptions): layout and MTP |
//! | [`MuseGlimmerForCausalLm`](crate::models::muse_glimmer::MuseGlimmerForCausalLm) | multimodal or text-only layout | [`MuseGlimmerCheckpointOptions`](crate::models::muse_glimmer::MuseGlimmerCheckpointOptions): layout |
//! | [`GemmaForCausalLm`](crate::models::gemma::GemmaForCausalLm) | text-only, multimodal, or original multimodal release | [`GemmaCheckpointOptions`](crate::models::gemma::GemmaCheckpointOptions): layout |
//! | [`NemotronHForCausalLm`](crate::models::nemotron_h::NemotronHForCausalLm) | its one layout | [`NemotronHCheckpointOptions`](crate::models::nemotron_h::NemotronHCheckpointOptions): provenance for reviewed NVIDIA artifacts |
//! | [`LlamaForCausalLm`](crate::models::llama::LlamaForCausalLm), [`Qwen3MoeForCausalLm`](crate::models::qwen3_moe::Qwen3MoeForCausalLm), [`DeepseekV2ForCausalLm`](crate::models::deepseek_v2::DeepseekV2ForCausalLm), [`GptOssForCausalLm`](crate::models::gpt_oss::GptOssForCausalLm) | their one layout (gpt-oss MXFP4 experts decode exactly) | an options type without fields yet |
//! | [`Gpt2ForCausalLm`](crate::models::gpt2::Gpt2ForCausalLm) | backbone or language-model namespace | none |
//! | [`GptNeoXForCausalLm`](crate::models::gpt_neox::GptNeoXForCausalLm), [`T5ForConditionalGeneration`](crate::models::t5::T5ForConditionalGeneration) | their one layout | none |
//!
//! Packed (quantized Burnpack) weights are loaded through
//! [`crate::models::AutoCausalLm`] for the families whose packed path is
//! certified. Every family's sink, certified or not, installs each packed
//! weight exactly or as its exact dequantized values (see
//! [`install_packed_parameter`]), so driving a sink directly through
//! [`quantization::execute_local_quantized_load_plan`] never stores a
//! requantized approximation.

pub(crate) mod accounting;
mod artifact;
mod budget;
mod dtype;
mod error;
mod hazard;
mod layout;
mod lexical;
mod mapping;
mod mxfp4;
mod namespace;
mod options;
pub mod quantization;
mod report;
mod runtime;

pub(crate) use artifact::discover_checkpoint_with_index_total_size_policy;
pub use artifact::{
    ArtifactSource, CheckpointManifest, ResolvedIndexTensorAlias, ShardInventory, TensorInventory,
    discover_checkpoint,
};
pub(crate) use budget::require_source_tensors;
pub use dtype::TensorDType;
pub use error::{CheckpointError, Result};
pub use hazard::{RangeHazard, RangeHazardPolicy, check_range_hazards};
pub(crate) use mapping::build_load_plan_with_index_total_size_policy;
pub use mapping::{
    ExpectedTensor, KeyMatcher, LoadPlan, MappingPlan, PlannedSource, PlannedTensor,
    PlannedTensorEquivalence, PlannedTransform, TensorOperation, TensorTransform, build_load_plan,
};
pub(crate) use namespace::LanguageNamespace;
pub use options::{
    ArtifactLimits, ArtifactSelection, DTypePolicy, IndexAliasSizeAccounting, IndexTensorAlias,
    LoadOptions,
};
pub use report::{
    AliasRecord, CheckpointEquivalenceRecord, DTypeConversion, LoadReport, MemoryEstimate,
    QuantizationRecord, TransformRecord,
};
#[cfg(test)]
pub(crate) use runtime::packed_sink_support;
pub use runtime::{
    ExecutionFailure, ExecutionFailureCause, ExecutionOutcome, ExecutionResult, ExecutionStats,
    LocalShardSource, PackedInstall, PackedLayoutPolicy, ParameterInstallError, SourceStats,
    SourceTensor, TensorSink, TensorSource, execute_load_plan, execute_local_load_plan,
    install_float_parameter, install_packed_parameter, packed_layout_is_exact,
};

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
