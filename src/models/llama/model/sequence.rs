//! Stateless passes: every query attends over the input's own tokens.
//!
//! A layer's visibility is causality, its connectivity (global or a sliding
//! window of `W` positions), and key padding. With positions `p` and `j` the
//! valid-token ordinals of a query and a key, key `j` is visible when
//!
//! ```text
//! key_valid(j)  ∧  j <= p  ∧  (global  ∨  p - W < j)
//! ```
//!
//! which is exactly the reference's mask for left-padded and unpadded batches
//! (Transformers compares physical columns, which agree with ordinals for
//! every valid pair unless padding is interior). Padded queries are *not*
//! excluded, so every score row keeps at least one key (see the parent
//! module's padding section).
//!
//! One `[batch, 1, token, token]` Boolean plane is built per distinct rule
//! and call and shared by every layer using that rule; an unpadded global
//! layer builds none and lets the backend apply causality in its fused
//! kernel. The stateless forward passes no mask when validation proved
//! every token valid, so an all-true mask is unpadded too. Staged execution
//! can replace a layer's rule with an explicit [`AttentionPolicy`]; padding
//! stays excluded and the policy's bias is added.
//!
//! [`backbone`] and [`finish`] are the stateless pass itself, generic over
//! any [`DecoderGraph`]: validate the input with one readback, build the
//! call's rotary tables and masks, embed, run every block, normalize, and
//! project the requested positions.

use burn::tensor::{Bool, Tensor, backend::Backend};

use crate::{
    CausalLmInput, CausalLmOutput, Error, ForwardOptions, Result,
    architecture::TokenMask,
    attention::{
        AttentionConnectivity, AttentionProjection, GroupedQueryAttention, MaskGeometry,
        padding_attention_mask,
    },
    experiment::AttentionPolicy,
    intervention::{AttentionPoints, Hooks, Stack},
    models::shared::{select_hidden, validate_input},
    position::RotaryAngles,
};

use super::graph::{DecoderBlock, DecoderGraph, LayerKeys, RotaryTables};

// The stateless pass.

/// Final states and requested diagnostics of one backbone pass.
pub(crate) struct Backbone<B: Backend> {
    /// Final-normalized states `[batch, token, d]`.
    pub(crate) hidden: Tensor<B, 3>,
    /// Embedding, every block output, and the final states, when requested.
    pub(crate) hidden_states: Vec<Tensor<B, 3>>,
}

/// Runs embedding, every block, and the final norm inside the caller's hook scope.
///
/// Implicit positions are valid-token ordinals. With `retain_hidden_states`,
/// the result also holds the embedding, every block output, and the
/// final-normalized states (`L + 2` tensors).
///
/// # Errors
///
/// Returns input validation errors, rotary or mask construction errors, and
/// hook errors.
pub(crate) fn backbone<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: &CausalLmInput<B>,
    retain_hidden_states: bool,
    hooks: &mut Hooks<'_, B>,
) -> Result<Backbone<B>> {
    let device = graph.parameter_device();
    let validated = validate_input(input, &device, &graph.input_contract())?;
    let positions = validated.require_positions(G::LABELS.inputs.position_ids)?;
    let angles = graph.rotary_tables(positions, validated.evaluated_context)?;
    // A mask proven all-valid is dropped: ordinals then equal columns, so
    // global layers keep the fused causal path and build no plane.
    let padding = input.token_mask.as_ref().filter(|_| !validated.fully_valid);
    let mut keys = SequenceKeys::new(graph, angles, padding, input.token_ids.dims())?;
    let mut hidden = hooks.apply(
        Stack::Decoder.embedding(),
        graph.embed(input.token_ids.clone()),
    )?;
    let mut hidden_states = Vec::new();
    for (index, block) in graph.blocks().iter().enumerate() {
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        hidden = block.forward(hidden, index, &mut keys, hooks)?;
    }
    if retain_hidden_states {
        hidden_states.push(hidden.clone());
    }
    let hidden = hooks.apply(Stack::Decoder.final_norm(), graph.final_norm(hidden))?;
    if retain_hidden_states {
        hidden_states.push(hidden.clone());
    }
    Ok(Backbone {
        hidden,
        hidden_states,
    })
}

/// Projects the requested positions of a backbone pass and assembles the output.
///
/// # Errors
///
/// Returns logits-selection and projection errors.
pub(crate) fn finish<B: Backend, G: DecoderGraph<B>>(
    graph: &G,
    input: CausalLmInput<B>,
    options: &ForwardOptions,
    backbone: Backbone<B>,
) -> Result<CausalLmOutput<B>> {
    let selected = select_hidden(
        backbone.hidden.clone(),
        input.token_mask.as_ref(),
        &options.logits,
    )?;
    let mut output = CausalLmOutput::from_logits(graph.project_logits(selected)?);
    output.final_hidden_state = options.hidden_states.then_some(backbone.hidden);
    output.hidden_states = backbone.hidden_states;
    Ok(output)
}

// Stateless keys.

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

/// Stateless keys: masks per rule and the call's rotary tables.
pub(crate) struct SequenceKeys<B: Backend> {
    /// Rotary tables of every rotary layer.
    angles: RotaryTables<B>,
    /// Visibility of each decoder layer, indexed by layer.
    layers: Vec<LayerVisibility<B>>,
    /// Error field prefix of the owning family.
    prefix: &'static str,
}

impl<B: Backend> SequenceKeys<B> {
    /// Builds every layer's visibility from the graph's schedule.
    ///
    /// # Errors
    ///
    /// Returns mask construction errors (extents or device mismatches).
    pub(crate) fn new<G: DecoderGraph<B>>(
        graph: &G,
        angles: RotaryTables<B>,
        token_mask: Option<&TokenMask<B>>,
        shape: [usize; 2],
    ) -> Result<Self> {
        Self::with_policies(graph, angles, token_mask, &[], shape)
    }

    /// Builds visibility where `policies[l]`, when present, replaces layer `l`'s rule.
    ///
    /// `shape` is the call's `[batch, token]` grid. Rules shared by several
    /// layers build one plane.
    ///
    /// # Errors
    ///
    /// Returns mask or policy validation errors.
    pub(crate) fn with_policies<G: DecoderGraph<B>>(
        graph: &G,
        angles: RotaryTables<B>,
        token_mask: Option<&TokenMask<B>>,
        policies: &[Option<AttentionPolicy<B>>],
        shape: [usize; 2],
    ) -> Result<Self> {
        let heads = graph.geometry().query_heads;
        let [batch, tokens] = shape;
        let device = graph.parameter_device();
        let geometry = MaskGeometry::new(batch, heads, tokens, tokens);
        let mut shared: Vec<(AttentionConnectivity, LayerVisibility<B>)> = Vec::new();
        let mut layers = Vec::with_capacity(graph.blocks().len());
        for (index, block) in graph.blocks().iter().enumerate() {
            if let Some(Some(policy)) = policies.get(index) {
                layers.push(policy_visibility(policy, token_mask, geometry, &device)?);
                continue;
            }
            let rule = block.connectivity();
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
        Ok(Self {
            angles,
            layers,
            prefix: G::LABELS.prefix,
        })
    }
}

impl<B: Backend> LayerKeys<B> for SequenceKeys<B> {
    /// Returns layer `layer`'s rotary tables.
    fn angles(&self, layer: usize) -> Option<&RotaryAngles<B>> {
        self.angles.for_layer(layer)
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
            Error::configuration(
                format!("{}.layer", self.prefix),
                format!("no visibility for layer {layer}"),
            )
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
