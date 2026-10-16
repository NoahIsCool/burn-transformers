//! Checked Nemotron-H checkpoint tensor widths.
//!
//! The hybrid contract derives several fused widths from configuration
//! extents. With `I = mamba_num_heads * mamba_head_dim` Mamba value channels,
//! `G` groups of `N` SSM states, and `H` Mamba heads:
//!
//! - the depthwise convolution runs over `xBC`, `C_conv = I + 2 G N`
//!   channels (values, then the grouped B and C state projections);
//! - the fused Mamba input projection emits `[z, xBC, dt]`, so its width is
//!   `I + C_conv + H = 2 I + 2 G N + H`;
//! - attention uses the query width `q d` and native key/value width `kv d`;
//! - routed experts read the latent width when one is configured, otherwise
//!   the residual width `h`;
//! - the MTP stage fuses the embedding and hidden state, a `2 h` input.
//!
//! `I`, `G N`, `q d`, and `kv d` are multiplied in `usize` as the mixers do;
//! the fused sums are formed in the portable `u64` checkpoint representation.
//!
//! # Verified properties
//!
//! Verus checks [`checkpoint_dimensions`] in place through `verus/loaders.rs`
//! (see `verus/README.md`): every width is exact, every extent is copied
//! unchanged, an overflowing width is reported by the first term that does
//! not fit (in the order the contract has always checked them), and the MTP
//! fusion width is `2 h` or `None` exactly when that overflows `u64`. A
//! `usize` extent always fits `u64`, so the conversions cannot fail.

#[cfg(verus_keep_ghost)]
use vstd::prelude::*;

/// Configuration extents that determine every checkpoint tensor width.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct DimensionInputs {
    /// `vocab_size`.
    pub(super) vocab: usize,
    /// `hidden_size`.
    pub(super) hidden: usize,
    /// `intermediate_size`.
    pub(super) intermediate: usize,
    /// `num_attention_heads`.
    pub(super) attention_heads: usize,
    /// `num_key_value_heads`.
    pub(super) key_value_heads: usize,
    /// `head_dim`.
    pub(super) head_dim: usize,
    /// `mamba_num_heads`.
    pub(super) mamba_heads: usize,
    /// `mamba_head_dim`.
    pub(super) mamba_head_dim: usize,
    /// `n_groups`.
    pub(super) groups: usize,
    /// `ssm_state_size`.
    pub(super) state_size: usize,
    /// `conv_kernel`.
    pub(super) conv_kernel: usize,
    /// `n_routed_experts`.
    pub(super) routed_experts: usize,
    /// `moe_intermediate_size`.
    pub(super) moe_intermediate: usize,
    /// `moe_shared_expert_intermediate_size`.
    pub(super) shared_intermediate: usize,
    /// `moe_latent_size`.
    pub(super) latent: Option<usize>,
}

/// Checked dimensions reused by contract sections.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy)]
pub(super) struct Dimensions {
    /// Vocabulary rows in the embedding and output projection.
    pub(super) vocab: u64,
    /// Residual-stream width.
    pub(super) hidden: u64,
    /// Expanded width of each dense feed-forward block.
    pub(super) intermediate: u64,
    /// Query projection width, `num_attention_heads * head_dim`.
    pub(super) attention_query: u64,
    /// Key/value projection width, `num_key_value_heads * head_dim`.
    pub(super) attention_kv: u64,
    /// Mamba value channels, `mamba_num_heads * mamba_head_dim`.
    pub(super) mamba_intermediate: u64,
    /// Convolved channels: Mamba values plus both grouped B/C state projections.
    pub(super) mamba_convolution: u64,
    /// Fused Mamba gate, convolution-input, and per-head timestep width.
    pub(super) mamba_projection: u64,
    /// Independent Mamba heads with learned timestep, decay, and skip scalars.
    pub(super) mamba_heads: u64,
    /// Temporal width of the depthwise convolution kernel.
    pub(super) mamba_kernel: u64,
    /// Routed experts and router output channels.
    pub(super) routed_experts: u64,
    /// Expanded width inside each routed expert.
    pub(super) moe_intermediate: u64,
    /// Expanded width inside the shared expert.
    pub(super) shared_intermediate: u64,
    /// Routed expert input/output width, using latent width when configured.
    pub(super) expert_input: u64,
    /// Concatenated embedding and hidden-state width at the MTP input, `2 h`,
    /// or `None` when it does not fit `u64` (only an MTP stage needs it).
    pub(super) double_hidden: Option<u64>,
}

/// The first fused width that does not fit, in checking order.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DimensionError {
    /// `mamba_num_heads * mamba_head_dim` exceeds `usize`.
    MambaIntermediate,
    /// `n_groups * ssm_state_size` exceeds `usize`.
    StateGroups,
    /// `2 G N` exceeds `u64`.
    ConvolutionStates,
    /// `I + 2 G N` exceeds `u64`.
    Convolution,
    /// `I + C_conv + H` exceeds `u64`.
    Projection,
    /// `num_attention_heads * head_dim` exceeds `usize`.
    AttentionQuery,
    /// `num_key_value_heads * head_dim` exceeds `usize`.
    AttentionKeyValue,
}

impl DimensionError {
    /// Configuration expression named in the contract's overflow diagnostic.
    pub(super) const fn field(self) -> &'static str {
        match self {
            Self::MambaIntermediate => "mamba_num_heads * mamba_head_dim",
            Self::StateGroups => "n_groups * ssm_state_size",
            Self::ConvolutionStates | Self::Convolution => "mamba convolution",
            Self::Projection => "mamba projection",
            Self::AttentionQuery => "num_attention_heads * head_dim",
            Self::AttentionKeyValue => "num_key_value_heads * head_dim",
        }
    }

    /// Diagnostic naming the overflowing operation and representation.
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::MambaIntermediate
            | Self::StateGroups
            | Self::AttentionQuery
            | Self::AttentionKeyValue => "dimension product overflows usize",
            Self::ConvolutionStates => "dimension product overflows u64",
            Self::Convolution | Self::Projection => "dimension sum overflows u64",
        }
    }
}

#[cfg(verus_keep_ghost)]
verus! {

/// `value` is representable as a `usize` extent.
pub open spec fn fits_usize(value: int) -> bool {
    value <= usize::MAX
}

/// `value` is representable in the portable checkpoint dimension type.
pub open spec fn fits_u64(value: int) -> bool {
    value <= u64::MAX
}

} // verus!

/// Computes every Nemotron-H checkpoint width from configuration extents.
///
/// Returns the exact widths listed in the module documentation, or the first
/// fused width (in the order of [`DimensionError`]) that does not fit.
#[cfg_attr(verus_keep_ghost, verus_verify)]
#[cfg_attr(verus_keep_ghost, verus_spec(dimensions =>
    ensures
        ({
            let values = inputs.mamba_heads * inputs.mamba_head_dim;
            let states = inputs.groups * inputs.state_size;
            let convolution = values + 2 * states;
            let projection = values + convolution + inputs.mamba_heads;
            let query = inputs.attention_heads * inputs.head_dim;
            let key_value = inputs.key_value_heads * inputs.head_dim;
            match dimensions {
                Ok(dimensions) => {
                    &&& dimensions.vocab == inputs.vocab
                    &&& dimensions.hidden == inputs.hidden
                    &&& dimensions.intermediate == inputs.intermediate
                    &&& dimensions.mamba_heads == inputs.mamba_heads
                    &&& dimensions.mamba_kernel == inputs.conv_kernel
                    &&& dimensions.routed_experts == inputs.routed_experts
                    &&& dimensions.moe_intermediate == inputs.moe_intermediate
                    &&& dimensions.shared_intermediate == inputs.shared_intermediate
                    &&& dimensions.expert_input == match inputs.latent {
                        Some(latent) => latent,
                        None => inputs.hidden,
                    }
                    &&& dimensions.mamba_intermediate == values
                    &&& dimensions.mamba_convolution == convolution
                    &&& dimensions.mamba_projection == projection
                    &&& dimensions.mamba_projection == 2 * values + 2 * states + inputs.mamba_heads
                    &&& dimensions.attention_query == query
                    &&& dimensions.attention_kv == key_value
                    &&& match dimensions.double_hidden {
                        Some(width) => width == 2 * inputs.hidden,
                        None => !fits_u64(2 * inputs.hidden),
                    }
                },
                Err(DimensionError::MambaIntermediate) => !fits_usize(values),
                Err(DimensionError::StateGroups) => fits_usize(values) && !fits_usize(states),
                Err(DimensionError::ConvolutionStates) => fits_usize(values) && fits_usize(states)
                    && !fits_u64(2 * states),
                Err(DimensionError::Convolution) => fits_usize(values) && fits_u64(2 * states)
                    && !fits_u64(convolution),
                Err(DimensionError::Projection) => fits_usize(values) && fits_u64(convolution)
                    && !fits_u64(projection),
                Err(DimensionError::AttentionQuery) => fits_usize(values) && fits_u64(projection)
                    && !fits_usize(query),
                Err(DimensionError::AttentionKeyValue) => fits_usize(values)
                    && fits_u64(projection) && fits_usize(query) && !fits_usize(key_value),
            }
        }),
))]
pub(super) fn checkpoint_dimensions(inputs: DimensionInputs) -> Result<Dimensions, DimensionError> {
    let mamba_heads = inputs.mamba_heads as u64;
    let Some(values) = inputs.mamba_heads.checked_mul(inputs.mamba_head_dim) else {
        return Err(DimensionError::MambaIntermediate);
    };
    let mamba_intermediate = values as u64;
    let Some(states) = inputs.groups.checked_mul(inputs.state_size) else {
        return Err(DimensionError::StateGroups);
    };
    let Some(both_states) = (states as u64).checked_mul(2) else {
        return Err(DimensionError::ConvolutionStates);
    };
    let Some(convolution) = mamba_intermediate.checked_add(both_states) else {
        return Err(DimensionError::Convolution);
    };
    let projection = match mamba_intermediate.checked_add(convolution) {
        Some(gate_and_input) => match gate_and_input.checked_add(mamba_heads) {
            Some(projection) => projection,
            None => return Err(DimensionError::Projection),
        },
        None => return Err(DimensionError::Projection),
    };
    let Some(attention_query) = inputs.attention_heads.checked_mul(inputs.head_dim) else {
        return Err(DimensionError::AttentionQuery);
    };
    let Some(attention_kv) = inputs.key_value_heads.checked_mul(inputs.head_dim) else {
        return Err(DimensionError::AttentionKeyValue);
    };
    let hidden = inputs.hidden as u64;
    let expert_input = match inputs.latent {
        Some(latent) => latent as u64,
        None => hidden,
    };
    Ok(Dimensions {
        vocab: inputs.vocab as u64,
        hidden,
        intermediate: inputs.intermediate as u64,
        attention_query: attention_query as u64,
        attention_kv: attention_kv as u64,
        mamba_intermediate,
        mamba_convolution: convolution,
        mamba_projection: projection,
        mamba_heads,
        mamba_kernel: inputs.conv_kernel as u64,
        routed_experts: inputs.routed_experts as u64,
        moe_intermediate: inputs.moe_intermediate as u64,
        shared_intermediate: inputs.shared_intermediate as u64,
        expert_input,
        double_hidden: hidden.checked_mul(2),
    })
}

#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;
