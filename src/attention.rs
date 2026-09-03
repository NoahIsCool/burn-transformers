//! Projected full, grouped-query, and multi-query attention.
//!
//! Attention maps a query to a convex combination of values using
//! `softmax(Q K^T / sqrt(d)) V`. Grouped-query attention stores fewer key/value
//! heads than query heads and repeats each KV head across a contiguous query
//! group only for the attention operation. The cache therefore retains native
//! KV heads instead of paying the repeated representation's memory cost.
//!
//! Rotary and relative-position transformations are deliberately separate from
//! projection and attention dispatch. This lets one architecture rotate only a
//! prefix, another use NoPE layers, and T5 provide an additive relative bias
//! without duplicating projection code.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{
        Bool, Tensor,
        activation::{sigmoid, silu},
        backend::Backend,
        module::attention,
        ops::AttentionModuleOptions,
    },
};

use crate::normalization::RmsNorm;
use crate::{Error, Result, architecture::TokenMask, normalization::RmsNormConfig, operations};

/// Multiplicative gate emitted alongside Qwen/Muse query projections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionOutputGate {
    /// Do not emit or apply an output gate.
    #[default]
    None,
    /// Multiply head outputs by the logistic sigmoid of the gate projection.
    Sigmoid,
    /// Multiply head outputs by `gate * sigmoid(gate)`.
    Silu,
}

/// Configuration for full MHA, GQA, or MQA projections and attention dispatch.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GroupedQueryAttentionConfig {
    /// Input and output residual-stream width.
    pub hidden_size: usize,
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of cached key/value heads.
    pub key_value_heads: usize,
    /// Query/key/value width per head.
    pub head_dim: usize,
    /// Whether all projection layers include additive biases.
    pub bias: bool,
    /// Optional Q/K per-head RMS normalization epsilon.
    pub query_key_norm_epsilon: Option<f64>,
    /// Whether Q/K norm checkpoints store `scale - 1` instead of `scale`.
    pub query_key_norm_zero_centered: bool,
    /// Optional multiplicative output-gate convention.
    pub output_gate: AttentionOutputGate,
    /// Optional non-default score scale. `None` preserves fused-kernel eligibility.
    pub scale: Option<f64>,
    /// Optional tanh softcap applied to raw attention logits.
    pub softcap: Option<f64>,
}

impl GroupedQueryAttentionConfig {
    /// Creates bias-free attention with conventional `1/sqrt(head_dim)` scaling.
    pub fn new(
        hidden_size: usize,
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            hidden_size,
            query_heads,
            key_value_heads,
            head_dim,
            bias: false,
            query_key_norm_epsilon: None,
            query_key_norm_zero_centered: false,
            output_gate: AttentionOutputGate::None,
            scale: None,
            softcap: None,
        }
    }

    /// Validates projection sizes, GQA grouping, and numerical options.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(Error::configuration(
                "attention",
                "hidden_size, query_heads, key_value_heads, and head_dim must be non-zero",
            ));
        }
        if !self.query_heads.is_multiple_of(self.key_value_heads) {
            return Err(Error::configuration(
                "attention.key_value_heads",
                "must divide query_heads exactly",
            ));
        }
        let query_width = self.query_heads.checked_mul(self.head_dim);
        let key_value_width = self.key_value_heads.checked_mul(self.head_dim);
        let query_output_width = query_width.and_then(|width| {
            if self.output_gate == AttentionOutputGate::None {
                Some(width)
            } else {
                width.checked_mul(2)
            }
        });
        if query_output_width.is_none() || key_value_width.is_none() {
            return Err(Error::configuration(
                "attention",
                "projected head dimensions overflow usize",
            ));
        }
        if self
            .query_key_norm_epsilon
            .is_some_and(|epsilon| !epsilon.is_finite() || epsilon <= 0.0)
        {
            return Err(Error::configuration(
                "attention.query_key_norm_epsilon",
                "must be finite and greater than zero when present",
            ));
        }
        if self
            .scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(Error::configuration(
                "attention.scale",
                "must be finite and greater than zero when present",
            ));
        }
        if self
            .softcap
            .is_some_and(|softcap| !softcap.is_finite() || softcap <= 0.0)
        {
            return Err(Error::configuration(
                "attention.softcap",
                "must be finite and greater than zero when present",
            ));
        }
        Ok(())
    }

    /// Initializes checkpoint-native `[out, in]` projections and optional Q/K norms.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<GroupedQueryAttention<B>> {
        self.validate()?;
        let query_width = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::configuration("attention", "query width overflows usize"))?;
        let key_value_width = self
            .key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::configuration("attention", "key/value width overflows usize"))?;
        let query_output_width = if self.output_gate == AttentionOutputGate::None {
            query_width
        } else {
            query_width.checked_mul(2).ok_or_else(|| {
                Error::configuration("attention", "gated query width overflows usize")
            })?
        };
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(self.bias)
                .with_layout(LinearLayout::Col)
                .init(device)
        };
        let query_key_norm = |epsilon| {
            RmsNormConfig::new(self.head_dim)
                .with_epsilon(epsilon)
                .with_zero_centered(self.query_key_norm_zero_centered)
                .init(device)
        };
        let (query_norm, key_norm) = match self.query_key_norm_epsilon {
            Some(epsilon) => (
                Some(query_key_norm(epsilon)?),
                Some(query_key_norm(epsilon)?),
            ),
            None => (None, None),
        };
        Ok(GroupedQueryAttention {
            q_proj: linear(self.hidden_size, query_output_width),
            k_proj: linear(self.hidden_size, key_value_width),
            v_proj: linear(self.hidden_size, key_value_width),
            o_proj: linear(query_width, self.hidden_size),
            query_norm,
            key_norm,
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            output_gate: self.output_gate,
            scale: self.scale,
            softcap: self.softcap,
        })
    }
}

/// Query, key, value, and optional gate tensors before position transformation.
#[derive(Clone, Debug)]
pub struct AttentionProjection<B: Backend> {
    /// Queries shaped `[batch, query_heads, query_sequence, head_dim]`.
    pub query: Tensor<B, 4>,
    /// Native keys shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    pub key: Tensor<B, 4>,
    /// Native values shaped `[batch, key_value_heads, key_sequence, head_dim]`.
    pub value: Tensor<B, 4>,
    /// Optional gate shaped like the flattened query-head output.
    pub gate: Option<Tensor<B, 3>>,
}

/// Checkpoint-compatible grouped-query attention projections.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    /// Query projection, optionally concatenated with an output gate.
    pub q_proj: Linear<B>,
    /// Native key-head projection.
    pub k_proj: Linear<B>,
    /// Native value-head projection.
    pub v_proj: Linear<B>,
    /// Projection from concatenated query-head outputs to the residual stream.
    pub o_proj: Linear<B>,
    /// Optional per-query-head normalization.
    pub query_norm: Option<RmsNorm<B>>,
    /// Optional per-key-head normalization.
    pub key_norm: Option<RmsNorm<B>>,
    /// Input/output width of the residual stream.
    pub hidden_size: usize,
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of native key/value heads.
    pub key_value_heads: usize,
    /// Per-head channel width.
    pub head_dim: usize,
    /// Output-gating convention.
    pub output_gate: AttentionOutputGate,
    /// Optional explicit attention score scale.
    pub scale: Option<f64>,
    /// Optional attention-logit softcap.
    pub softcap: Option<f64>,
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Projects self-attention Q/K/V tensors without applying a position transform.
    pub fn project_self(&self, hidden: Tensor<B, 3>) -> Result<AttentionProjection<B>> {
        self.project(hidden.clone(), hidden)
    }

    /// Projects query states and independent key/value source states.
    pub fn project(
        &self,
        query_hidden: Tensor<B, 3>,
        key_value_hidden: Tensor<B, 3>,
    ) -> Result<AttentionProjection<B>> {
        let [batch, query_sequence, query_hidden_size] = query_hidden.dims();
        let [key_batch, key_sequence, key_value_hidden_size] = key_value_hidden.dims();
        if batch == 0 || query_sequence == 0 || query_hidden_size != self.hidden_size {
            return Err(Error::shape(
                "attention.query_input",
                format!("[batch>0, sequence>0, {}]", self.hidden_size),
                format!("[{batch}, {query_sequence}, {query_hidden_size}]"),
            ));
        }
        if key_batch != batch || key_sequence == 0 || key_value_hidden_size != self.hidden_size {
            return Err(Error::shape(
                "attention.key_value_input",
                format!("[{batch}, sequence>0, {}]", self.hidden_size),
                format!("[{key_batch}, {key_sequence}, {key_value_hidden_size}]"),
            ));
        }
        let query_width = self.query_heads * self.head_dim;
        let projected_query = operations::linear(query_hidden, &self.q_proj)?;
        let (query, gate) = if self.output_gate == AttentionOutputGate::None {
            (
                projected_query
                    .reshape([batch, query_sequence, self.query_heads, self.head_dim])
                    .swap_dims(1, 2),
                None,
            )
        } else {
            // Qwen-family gated projections are interleaved by head:
            // `[head_0.query, head_0.gate, head_1.query, ...]`.
            let projected_query = projected_query.reshape([
                batch,
                query_sequence,
                self.query_heads,
                self.head_dim * 2,
            ]);
            (
                projected_query
                    .clone()
                    .narrow(3, 0, self.head_dim)
                    .swap_dims(1, 2),
                Some(
                    projected_query
                        .narrow(3, self.head_dim, self.head_dim)
                        .reshape([batch, query_sequence, query_width]),
                ),
            )
        };
        let key = operations::linear(key_value_hidden.clone(), &self.k_proj)?
            .reshape([batch, key_sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let value = operations::linear(key_value_hidden, &self.v_proj)?
            .reshape([batch, key_sequence, self.key_value_heads, self.head_dim])
            .swap_dims(1, 2);
        let query = self
            .query_norm
            .as_ref()
            .map_or(query.clone(), |norm| norm.forward(query));
        let key = self
            .key_norm
            .as_ref()
            .map_or(key.clone(), |norm| norm.forward(key));
        Ok(AttentionProjection {
            query,
            key,
            value,
            gate,
        })
    }

    /// Dispatches projected attention after the caller applies positional transforms.
    ///
    /// `mask` follows Burn's low-level convention: `true` means excluded. Public
    /// token-valid masks should first pass through [`padding_attention_mask`].
    pub fn attend(
        &self,
        projection: AttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        bias: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        validate_projection(self, &projection, mask.as_ref(), bias.as_ref())?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key = repeat_key_value(projection.key, self.query_heads)?;
        let value = repeat_key_value(projection.value, self.query_heads)?;
        let context = attention(
            projection.query,
            key,
            value,
            mask,
            bias,
            AttentionModuleOptions {
                scale: self.scale,
                softcap: self.softcap,
                is_causal: causal,
            },
        )
        .swap_dims(1, 2)
        .reshape([batch, query_sequence, self.query_heads * self.head_dim]);
        let context = match (self.output_gate, projection.gate) {
            (AttentionOutputGate::None, None) => context,
            (AttentionOutputGate::Sigmoid, Some(gate)) => context * sigmoid(gate),
            (AttentionOutputGate::Silu, Some(gate)) => context * silu(gate),
            _ => {
                return Err(Error::shape(
                    "attention.gate",
                    format!("a gate matching {:?}", self.output_gate),
                    "a mismatched gate payload",
                ));
            }
        };
        operations::linear(context, &self.o_proj)
    }

    /// Evaluates attention without RoPE or relative-position bias.
    pub fn forward_without_position_encoding(
        &self,
        hidden: Tensor<B, 3>,
        token_mask: Option<&TokenMask<B>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let projection = self.project_self(hidden)?;
        let [batch, _, query_sequence, _] = projection.query.dims();
        let key_sequence = projection.key.dims()[2];
        let mask = padding_attention_mask(
            token_mask,
            token_mask,
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
        )?;
        self.attend(projection, mask, None, causal)
    }
}

/// Repeats native KV heads into contiguous query-head groups for one attention call.
pub fn repeat_key_value<B: Backend>(
    tensor: Tensor<B, 4>,
    query_heads: usize,
) -> Result<Tensor<B, 4>> {
    let [batch, key_value_heads, sequence, head_dim] = tensor.dims();
    if key_value_heads == 0 || query_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(Error::configuration(
            "attention.head_grouping",
            format!(
                "query_heads ({query_heads}) must be non-zero and divisible by key_value_heads ({key_value_heads})"
            ),
        ));
    }
    let groups = query_heads / key_value_heads;
    if groups == 1 {
        return Ok(tensor);
    }
    Ok(tensor
        .unsqueeze_dim::<5>(2)
        .expand([batch, key_value_heads, groups, sequence, head_dim])
        .reshape([batch, query_heads, sequence, head_dim]))
}

/// Validates projected tensor topology before dispatching a backend attention kernel.
fn validate_projection<B: Backend>(
    attention: &GroupedQueryAttention<B>,
    projection: &AttentionProjection<B>,
    mask: Option<&Tensor<B, 4, Bool>>,
    bias: Option<&Tensor<B, 4>>,
) -> Result<()> {
    let [batch, query_heads, query_sequence, query_dim] = projection.query.dims();
    let [key_batch, key_value_heads, key_sequence, key_dim] = projection.key.dims();
    let [value_batch, value_heads, value_sequence, value_dim] = projection.value.dims();
    let expected_query = [
        batch,
        attention.query_heads,
        query_sequence,
        attention.head_dim,
    ];
    if [batch, query_heads, query_sequence, query_dim] != expected_query {
        return Err(Error::shape(
            "attention.query",
            format!("{:?}", expected_query),
            format!("{:?}", projection.query.dims()),
        ));
    }
    let expected_key = [
        batch,
        attention.key_value_heads,
        key_sequence,
        attention.head_dim,
    ];
    if [key_batch, key_value_heads, key_sequence, key_dim] != expected_key {
        return Err(Error::shape(
            "attention.key",
            format!("{:?}", expected_key),
            format!("{:?}", projection.key.dims()),
        ));
    }
    let expected_value = [
        batch,
        attention.key_value_heads,
        key_sequence,
        attention.head_dim,
    ];
    if [value_batch, value_heads, value_sequence, value_dim] != expected_value {
        return Err(Error::shape(
            "attention.value",
            format!("{:?}", expected_value),
            format!("{:?}", projection.value.dims()),
        ));
    }
    let expected_gate = [
        batch,
        query_sequence,
        attention.query_heads * attention.head_dim,
    ];
    match (attention.output_gate, projection.gate.as_ref()) {
        (AttentionOutputGate::None, None) => {}
        (AttentionOutputGate::None, Some(_)) => {
            return Err(Error::shape(
                "attention.gate",
                "no gate for ungated attention",
                "a gate tensor",
            ));
        }
        (_, Some(gate)) if gate.dims() == expected_gate => {}
        (_, Some(gate)) => {
            return Err(Error::shape(
                "attention.gate",
                format!("{:?}", expected_gate),
                format!("{:?}", gate.dims()),
            ));
        }
        (_, None) => {
            return Err(Error::shape(
                "attention.gate",
                format!("{:?}", expected_gate),
                "no gate tensor",
            ));
        }
    }
    let expected_additive = [batch, attention.query_heads, query_sequence, key_sequence];
    if let Some(mask) = mask
        && mask.dims() != expected_additive
    {
        return Err(Error::shape(
            "attention.mask",
            format!("{:?}", expected_additive),
            format!("{:?}", mask.dims()),
        ));
    }
    if let Some(bias) = bias {
        let [bias_batch, bias_heads, bias_query, bias_key] = bias.dims();
        if ![1, batch].contains(&bias_batch)
            || bias_heads != attention.query_heads
            || bias_query != query_sequence
            || bias_key != key_sequence
        {
            return Err(Error::shape(
                "attention.bias",
                format!(
                    "[batch in {{1, {batch}}}, {}, {query_sequence}, {key_sequence}]",
                    attention.query_heads
                ),
                format!("{:?}", bias.dims()),
            ));
        }
    }
    Ok(())
}

/// Converts semantic valid-token masks into Burn's `true == excluded` mask.
pub fn padding_attention_mask<B: Backend>(
    query_mask: Option<&TokenMask<B>>,
    key_mask: Option<&TokenMask<B>>,
    batch: usize,
    heads: usize,
    query_sequence: usize,
    key_sequence: usize,
) -> Result<Option<Tensor<B, 4, Bool>>> {
    if batch == 0 || heads == 0 || query_sequence == 0 || key_sequence == 0 {
        return Err(Error::configuration(
            "attention.mask_shape",
            "batch, heads, query sequence, and key sequence must be non-zero",
        ));
    }
    let query_invalid = if let Some(mask) = query_mask {
        let [mask_batch, mask_sequence] = mask.as_tensor().dims();
        if [mask_batch, mask_sequence] != [batch, query_sequence] {
            return Err(Error::shape(
                "query_mask",
                format!("[{batch}, {query_sequence}]"),
                format!("[{mask_batch}, {mask_sequence}]"),
            ));
        }
        Some(
            mask.as_tensor()
                .clone()
                .bool_not()
                .reshape([batch, 1, query_sequence, 1])
                .expand([batch, heads, query_sequence, key_sequence]),
        )
    } else {
        None
    };
    let key_invalid = if let Some(mask) = key_mask {
        let [mask_batch, mask_sequence] = mask.as_tensor().dims();
        if [mask_batch, mask_sequence] != [batch, key_sequence] {
            return Err(Error::shape(
                "key_mask",
                format!("[{batch}, {key_sequence}]"),
                format!("[{mask_batch}, {mask_sequence}]"),
            ));
        }
        Some(
            mask.as_tensor()
                .clone()
                .bool_not()
                .reshape([batch, 1, 1, key_sequence])
                .expand([batch, heads, query_sequence, key_sequence]),
        )
    } else {
        None
    };
    Ok(match (query_invalid, key_invalid) {
        (Some(query), Some(key)) => Some(query.bool_or(key)),
        (Some(query), None) => Some(query),
        (None, Some(key)) => Some(key),
        (None, None) => None,
    })
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Flex,
        tensor::{Bool, Tensor},
    };

    use crate::architecture::TokenMask;

    use super::{
        AttentionOutputGate, AttentionProjection, GroupedQueryAttentionConfig,
        padding_attention_mask, repeat_key_value,
    };

    /// Qwen's projected query width may exceed its residual width.
    #[test]
    fn attention_accepts_non_square_query_projection() {
        GroupedQueryAttentionConfig::new(10, 3, 1, 4)
            .validate()
            .expect("query width need not equal residual width");
    }

    /// Projected-width arithmetic is checked before an initializer sees a shape.
    #[test]
    fn attention_rejects_projection_dimension_overflow() {
        let error = GroupedQueryAttentionConfig::new(8, usize::MAX, 1, 2)
            .validate()
            .expect_err("query head product must not wrap");
        assert!(error.to_string().contains("overflow"));
    }

    /// KV repetition duplicates each native head into one contiguous GQA group.
    #[test]
    fn key_value_repetition_preserves_group_order() {
        let device = Default::default();
        let tensor = Tensor::<Flex, 4>::from_floats([[[[1.0]], [[2.0]]]], &device);
        let repeated = repeat_key_value(tensor, 4)
            .expect("four query heads group over two key/value heads")
            .to_data();
        assert_eq!(
            repeated
                .as_slice::<f32>()
                .expect("Flex f32 data should expose an f32 slice"),
            &[1.0, 1.0, 2.0, 2.0]
        );
    }

    /// Invalid head ratios are reported instead of reaching a division or reshape panic.
    #[test]
    fn key_value_repetition_rejects_invalid_grouping() {
        let device = Default::default();
        let tensor = Tensor::<Flex, 4>::zeros([1, 2, 1, 3], &device);
        let error = repeat_key_value(tensor, 3).expect_err("three heads cannot group over two");
        assert!(error.to_string().contains("head_grouping"));
    }

    /// External projected-attention callers receive shape diagnostics before dispatch.
    #[test]
    fn attention_rejects_malformed_projection() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let projection = AttentionProjection {
            query: Tensor::zeros([1, 3, 2, 3], &device),
            key: Tensor::zeros([1, 2, 2, 3], &device),
            value: Tensor::zeros([1, 2, 2, 3], &device),
            gate: None,
        };
        let error = module
            .attend(projection, None, None, true)
            .expect_err("wrong query-head count must fail");
        assert!(error.to_string().contains("attention.query"));
    }

    /// Projection validates residual width before dispatching a linear backend kernel.
    #[test]
    fn attention_rejects_malformed_projection_input() {
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(8, 4, 2, 3)
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let error = module
            .project_self(Tensor::zeros([1, 2, 7], &device))
            .expect_err("wrong residual width must fail");
        assert!(error.to_string().contains("attention.query_input"));
    }

    /// Valid-token masks invert exactly once at the low-level attention boundary.
    #[test]
    fn padding_mask_marks_invalid_queries_and_keys() {
        let device = Default::default();
        let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, false]].into(),
            &device,
        ));
        let mask = padding_attention_mask(Some(&valid), Some(&valid), 1, 1, 2, 2)
            .expect("valid mask shape should pass")
            .expect("provided masks should produce a tensor")
            .to_data();
        assert_eq!(
            mask.as_slice::<bool>()
                .expect("boolean data should expose a bool slice"),
            &[false, true, true, true]
        );
    }

    /// Invalid mask extents return a contextual shape error rather than panicking.
    #[test]
    fn padding_mask_rejects_wrong_shape() {
        let device = Default::default();
        let valid = TokenMask::new(Tensor::<Flex, 2, Bool>::from_bool(
            [[true, true, true]].into(),
            &device,
        ));
        let error = padding_attention_mask(Some(&valid), None, 1, 1, 2, 2)
            .expect_err("mismatched query mask must fail");
        assert!(error.to_string().contains("query_mask"));
    }

    /// Projection, GQA dispatch, output gating, and output projection preserve shape.
    #[test]
    fn gated_attention_preserves_residual_shape() {
        let device = Default::default();
        let mut config = GroupedQueryAttentionConfig::new(8, 4, 2, 3);
        config.output_gate = AttentionOutputGate::Silu;
        config.query_key_norm_epsilon = Some(1e-6);
        let module = config
            .init::<Flex>(&device)
            .expect("valid attention should initialize");
        let output = module
            .forward_without_position_encoding(
                Tensor::<Flex, 3>::zeros([2, 5, 8], &device),
                None,
                true,
            )
            .expect("attention should execute");
        assert_eq!(output.dims(), [2, 5, 8]);
    }
}
