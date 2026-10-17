//! Keys for stateless passes: every query attends over the input's own tokens.
//!
//! A layer's visibility is causality, its connectivity (global, or a sliding
//! window of `W` positions), and key padding. With `p` and `j` the
//! valid-token ordinals of a query and a key, key `j` is visible when
//!
//! ```text
//! key_valid(j)  ∧  j <= p  ∧  (global  ∨  p - W < j)
//! ```
//!
//! which is the reference's mask (`kv_idx > q_idx - W` for sliding layers)
//! for unpadded and left-padded batches. Padded queries are *not* excluded;
//! the learned sinks give every softmax row mass in any case, so no row is
//! ever undefined.
//!
//! One `[batch, H, token, token]` exclusion plane (an expanded
//! `[batch, 1, token, token]` Boolean) is built per distinct rule and call and
//! shared by every layer using that rule; an unpadded global layer builds
//! none and lets the attention call apply causality in its fused kernel.
//! Staged execution can replace a layer's rule with an explicit
//! [`AttentionPolicy`]; padding stays excluded and the policy's bias is
//! added.

use burn::tensor::{Bool, Tensor, backend::Backend};

use crate::{
    Result,
    architecture::TokenMask,
    attention::{
        AttentionConnectivity, AttentionProjection, GroupedQueryAttention, MaskGeometry,
        padding_attention_mask,
    },
    experiment::AttentionPolicy,
    intervention::{AttentionPoints, Hooks},
    position::RotaryAngles,
};

use super::{GptOssModel, LayerKeys};

/// The exclusions, bias, and causality of one layer's attention call.
#[derive(Debug, Clone)]
struct LayerVisibility<B: Backend> {
    /// `[batch, H, token, token]` exclusions (`true` = excluded), or none.
    mask: Option<Tensor<B, 4, Bool>>,
    /// Optional additive score bias, already expanded.
    bias: Option<Tensor<B, 4>>,
    /// Whether the attention call applies causality itself.
    causal: bool,
}

/// Stateless keys: one visibility per layer and the call's rotary tables.
pub(super) struct SequenceKeys<B: Backend> {
    /// Rotary tables in the activation dtype.
    angles: RotaryAngles<B>,
    /// Visibility of each decoder layer, indexed by layer.
    layers: Vec<LayerVisibility<B>>,
}

impl<B: Backend> SequenceKeys<B> {
    /// Builds every layer's visibility from the model's schedule.
    ///
    /// # Errors
    ///
    /// Returns mask construction errors (extents or device mismatches).
    pub(super) fn new(
        model: &GptOssModel<B>,
        angles: RotaryAngles<B>,
        token_mask: Option<&TokenMask<B>>,
        shape: [usize; 2],
    ) -> Result<Self> {
        Self::with_policies(model, angles, token_mask, &[], shape)
    }

    /// Builds visibility where `policies[l]`, when present, replaces layer `l`'s rule.
    ///
    /// `shape` is the call's `[batch, token]` grid. Rules shared by several
    /// layers build one plane.
    ///
    /// # Errors
    ///
    /// Returns mask or policy validation errors.
    pub(super) fn with_policies(
        model: &GptOssModel<B>,
        angles: RotaryAngles<B>,
        token_mask: Option<&TokenMask<B>>,
        policies: &[Option<AttentionPolicy<B>>],
        shape: [usize; 2],
    ) -> Result<Self> {
        let heads = model.architecture.query_heads;
        let [batch, tokens] = shape;
        let device = model.embed_tokens.weight.lazy_device();
        let geometry = MaskGeometry::new(batch, heads, tokens, tokens);
        let mut shared: Vec<(AttentionConnectivity, LayerVisibility<B>)> = Vec::new();
        let mut layers = Vec::with_capacity(model.layers.len());
        for (index, layer) in model.layers.iter().enumerate() {
            if let Some(Some(policy)) = policies.get(index) {
                layers.push(policy_visibility(policy, token_mask, geometry, &device)?);
                continue;
            }
            let rule = layer.connectivity;
            let visibility = match shared.iter().find(|(known, _)| *known == rule) {
                Some((_, visibility)) => visibility.clone(),
                None => {
                    let visibility = rule_visibility(rule, token_mask, geometry, &device)?;
                    shared.push((rule, visibility.clone()));
                    visibility
                }
            };
            layers.push(visibility);
        }
        Ok(Self { angles, layers })
    }
}

impl<B: Backend> LayerKeys<B> for SequenceKeys<B> {
    /// Returns the call's rotary tables.
    fn angles(&self) -> &RotaryAngles<B> {
        &self.angles
    }

    /// Attends over the call's own keys with layer `layer`'s visibility.
    fn attend(
        &mut self,
        layer: usize,
        attention: &GroupedQueryAttention<B>,
        projection: AttentionProjection<B>,
        points: AttentionPoints,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let visibility = self.layers.get(layer).cloned().ok_or_else(|| {
            crate::Error::configuration("gpt_oss.layer", format!("no visibility for layer {layer}"))
        })?;
        attention.attend_with_hooks(
            projection,
            visibility.mask,
            visibility.bias,
            visibility.causal,
            points,
            hooks,
        )
    }
}

/// Visibility of a schedule rule: causal, connectivity, and key padding.
///
/// An unpadded global layer needs no plane: the attention call applies
/// causality itself (a fused kernel on most backends).
fn rule_visibility<B: Backend>(
    rule: AttentionConnectivity,
    token_mask: Option<&TokenMask<B>>,
    geometry: MaskGeometry,
    device: &B::Device,
) -> Result<LayerVisibility<B>> {
    if rule == AttentionConnectivity::Global && token_mask.is_none() {
        return Ok(LayerVisibility {
            mask: None,
            bias: None,
            causal: true,
        });
    }
    let mask = rule.exclusion_mask(true, geometry, None, token_mask, device)?;
    Ok(LayerVisibility {
        causal: mask.is_none(),
        mask,
        bias: None,
    })
}

/// Visibility of an explicit staged policy: key padding plus the policy's links and bias.
fn policy_visibility<B: Backend>(
    policy: &AttentionPolicy<B>,
    token_mask: Option<&TokenMask<B>>,
    geometry: MaskGeometry,
    device: &B::Device,
) -> Result<LayerVisibility<B>> {
    let shape = [
        geometry.batch,
        geometry.heads,
        geometry.query_sequence,
        geometry.key_sequence,
    ];
    let padding = padding_attention_mask(
        None,
        token_mask,
        geometry.batch,
        geometry.heads,
        geometry.query_sequence,
        geometry.key_sequence,
    )?;
    Ok(LayerVisibility {
        mask: policy.exclusion_mask(padding, shape, device)?,
        bias: policy.expanded_bias(shape, device)?,
        causal: policy.causal,
    })
}
