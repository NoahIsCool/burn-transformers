//! Decoder-only modular language model assembly.
//!
//! The model owns only trainable parameters and immutable topology. Autoregressive session
//! history remains in [`ModularDecodeState`](super::ModularDecodeState), preserving a clean
//! stateless graph for autodiff and model surgery.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLanguageModel, CausalLmInput, CausalLmOutput, DecodeInput, Error,
    ForwardOptions, Result, architecture::DecoderSpec, cache::CacheConfig,
};

use super::{
    common::{
        AttentionDirection, BackboneOutput, ModularLayer, ModularNorm, causal_output,
        evaluated_context, router_diagnostics, select_hidden, tied_projection,
        validate_and_resolve_input, validate_decoder_execution, validate_forward_options,
    },
    state::{ModularDecodeState, StatePurpose},
};

/// A heterogeneous decoder-only transformer assembled from [`DecoderSpec`].
#[derive(Module, Debug)]
pub struct ModularCausalLm<B: Backend> {
    /// Token lookup table.
    pub embed_tokens: Embedding<B>,
    /// Ordered heterogeneous residual blocks.
    layers: Vec<ModularLayer<B>>,
    /// Final normalization before vocabulary projection.
    final_norm: ModularNorm<B>,
    /// Independent vocabulary projection, absent when embeddings are tied.
    pub lm_head: Option<Linear<B>>,
    /// Validated topology excluded from parameter records.
    #[module(skip)]
    spec: DecoderSpec,
}

impl<B: Backend> ModularCausalLm<B> {
    /// Validates a modular topology and allocates its Burn parameter graph.
    ///
    /// Unsupported but well-formed features return a capability error before the embedding or
    /// any layer parameter is initialized. This makes failed experiment construction cheap and
    /// prevents an unavailable mixer from being approximated silently.
    pub fn init(spec: DecoderSpec, device: &B::Device) -> Result<Self> {
        validate_decoder_execution(&spec, AttentionDirection::Causal, "custom.causal")?;
        let layers = spec
            .layers
            .iter()
            .map(|layer| ModularLayer::init(layer, spec.hidden_size, None, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: EmbeddingConfig::new(spec.vocabulary_size, spec.hidden_size).init(device),
            layers,
            final_norm: ModularNorm::init(spec.final_norm, spec.hidden_size, device)?,
            lm_head: (!spec.tie_word_embeddings).then(|| {
                LinearConfig::new(spec.hidden_size, spec.vocabulary_size)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col)
                    .init(device)
            }),
            spec,
        })
    }

    /// Returns the exact serializable topology used to construct the graph.
    #[must_use]
    pub const fn spec(&self) -> &DecoderSpec {
        &self.spec
    }

    /// Evaluates embeddings, heterogeneous blocks, and final normalization.
    pub(super) fn forward_backbone(
        &self,
        input: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<BackboneOutput<B>> {
        let positions = validate_and_resolve_input(
            input,
            self.spec.vocabulary_size,
            self.spec.max_position_embeddings,
            "custom.causal",
        )?;
        let context = evaluated_context(positions.clone())?;
        let mut hidden = self.embed_tokens.forward(input.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.layers {
            let (next, router) = layer.forward(
                hidden,
                positions.clone(),
                input.token_mask.as_ref(),
                AttentionDirection::Causal,
                context,
                None,
            )?;
            hidden = next;
            if let Some(router) = router {
                routers.push(router);
            }
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.final_norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(BackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }

    /// Applies an independent or exactly tied vocabulary projection.
    pub(super) fn project(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match &self.lm_head {
            Some(head) => crate::operations::linear(hidden, head),
            None => Ok(tied_projection(hidden, self.embed_tokens.weight.val())),
        }
    }

    /// Evaluates the stateless, differentiable decoder graph.
    pub fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        validate_forward_options(options)?;
        let backbone = self.forward_backbone(&input, options.hidden_states)?;
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.token_mask.as_ref(),
            &options.logits,
        )?;
        let routers = router_diagnostics(&backbone.routers, input.token_mask.as_ref(), options)?;
        Ok(causal_output(
            self.project(selected)?,
            backbone.hidden,
            backbone.hidden_states,
            routers,
            options,
        ))
    }
}

impl ArchitectureConfig for DecoderSpec {
    const MODEL_TYPES: &'static [&'static str] = &["modular_causal"];

    fn validate(&self) -> Result<()> {
        validate_decoder_execution(self, AttentionDirection::Causal, "custom.causal")
    }
}

impl<B: Backend> CausalLanguageModel<B> for ModularCausalLm<B> {
    type Config = DecoderSpec;
    type State = ModularDecodeState<B>;

    fn forward(
        &self,
        input: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        ModularCausalLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        if config.capacity() > self.spec.max_position_embeddings {
            return Err(Error::configuration(
                "custom.causal.decode_state.capacity",
                format!(
                    "{} exceeds max_position_embeddings {}",
                    config.capacity(),
                    self.spec.max_position_embeddings
                ),
            ));
        }
        ModularDecodeState::new(
            config,
            batch_size,
            self.spec.clone(),
            StatePurpose::Causal,
            device,
        )
    }

    fn prefill(
        &self,
        input: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        state.run_causal(self, input, options, true)
    }

    fn decode(
        &self,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<CausalLmOutput<B>> {
        state.run_causal(
            self,
            CausalLmInput {
                token_ids: input.token_ids,
                token_mask: input.token_mask,
                position_ids: input.position_ids,
            },
            options,
            false,
        )
    }
}
