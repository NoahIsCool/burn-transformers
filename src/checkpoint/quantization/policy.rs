//! Conservative name-and-shape policy for weight-only quantization.
//!
//! The policy intentionally operates before numerical conversion. Large matrix
//! weights in projections and feed-forward experts are eligible. Parameters whose
//! small size or numerical sensitivity makes weight-only quantization undesirable
//! are classified explicitly so manifests can explain every skipped tensor.

use serde::{Deserialize, Serialize};

/// Reason that a checkpoint tensor is excluded from weight quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationExclusion {
    /// Multi-token-prediction parameters remain in their source precision.
    Mtp,
    /// Token or positional embedding tables remain in source precision.
    Embedding,
    /// Normalization weights and scales remain in source precision.
    Normalization,
    /// Additive bias parameters remain in source precision.
    Bias,
    /// Numerically sensitive recurrent decay/time-step scalars remain unpacked.
    RecurrentScalar,
    /// Sequence-mixer convolution kernels remain in source precision initially.
    RecurrentConvolution,
    /// Learned relative-position and other additive bias tables stay unpacked.
    BiasTable,
    /// The key does not denote a conventional weight parameter.
    NotWeight,
    /// Rank-zero and rank-one tensors are never considered matrix weights.
    ScalarOrVector,
    /// Rank-three and higher tensors are not projection matrices.
    NonMatrix,
}

/// Eligibility decision for one architecture-owned parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "reason")]
pub enum QuantizationEligibility {
    /// Tensor may be converted by the symmetric reference quantizer.
    Eligible,
    /// Tensor must remain in its original representation.
    Excluded(QuantizationExclusion),
}

/// Classifies a model parameter for conservative weight-only quantization.
///
/// `shape` uses logical row-major dimensions. The classifier is deliberately
/// architecture-agnostic and conservative; a model family may impose additional
/// exclusions, but must not bypass the MTP, normalization, bias, embedding, or
/// recurrent-scalar exclusions represented here.
#[must_use]
pub fn classify_weight_for_quantization(name: &str, shape: &[usize]) -> QuantizationEligibility {
    let components: Vec<_> = name.split('.').collect();
    let last = components.last().copied().unwrap_or_default();

    if components.iter().any(|component| {
        let component = component.to_ascii_lowercase();
        component == "mtp" || component.starts_with("mtp_") || component == "multi_token_prediction"
    }) {
        return QuantizationEligibility::Excluded(QuantizationExclusion::Mtp);
    }

    if is_recurrent_scalar(&components) {
        return QuantizationEligibility::Excluded(QuantizationExclusion::RecurrentScalar);
    }

    if components.iter().any(|component| is_convolution(component)) {
        return QuantizationEligibility::Excluded(QuantizationExclusion::RecurrentConvolution);
    }

    if last.eq_ignore_ascii_case("bias") {
        return QuantizationEligibility::Excluded(QuantizationExclusion::Bias);
    }

    if components.iter().any(|component| is_bias_table(component)) {
        return QuantizationEligibility::Excluded(QuantizationExclusion::BiasTable);
    }

    if components
        .iter()
        .any(|component| is_normalization(component))
    {
        return QuantizationEligibility::Excluded(QuantizationExclusion::Normalization);
    }

    if components.iter().any(|component| is_embedding(component)) {
        return QuantizationEligibility::Excluded(QuantizationExclusion::Embedding);
    }

    if !last.eq_ignore_ascii_case("weight") {
        return QuantizationEligibility::Excluded(QuantizationExclusion::NotWeight);
    }

    if shape.len() < 2 {
        return QuantizationEligibility::Excluded(QuantizationExclusion::ScalarOrVector);
    }
    if shape.len() != 2 {
        return QuantizationEligibility::Excluded(QuantizationExclusion::NonMatrix);
    }

    QuantizationEligibility::Eligible
}

fn is_embedding(component: &str) -> bool {
    let component = component.to_ascii_lowercase();
    matches!(
        component.as_str(),
        "embed_tokens"
            | "embeddings"
            | "embedding"
            | "word_embeddings"
            | "position_embeddings"
            | "token_embeddings"
            | "tok_embeddings"
            | "wte"
            | "wpe"
            | "shared"
    ) || component.starts_with("embed_")
        || component.ends_with("_embedding")
        || component.ends_with("_embeddings")
}

fn is_normalization(component: &str) -> bool {
    let component = component.to_ascii_lowercase();
    component.contains("norm")
        || component == "ln_f"
        || component == "ln"
        || component.starts_with("ln_")
}

fn is_convolution(component: &str) -> bool {
    let component = component.to_ascii_lowercase();
    component == "conv" || component.starts_with("conv1d") || component.starts_with("conv_")
}

fn is_bias_table(component: &str) -> bool {
    let component = component.to_ascii_lowercase();
    component.ends_with("_bias") || component.ends_with("_biases")
}

fn is_recurrent_scalar(components: &[&str]) -> bool {
    components.iter().any(|component| {
        matches!(
            component.to_ascii_lowercase().as_str(),
            "a_log"
                | "d"
                | "dt_bias"
                | "time_step"
                | "time_step_bias"
                | "decay"
                | "decay_rate"
                | "recurrent_scale"
        )
    })
}
